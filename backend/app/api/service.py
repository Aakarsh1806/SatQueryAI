"""Persistent job state and a single in-process inference worker."""
import asyncio
from datetime import timedelta
import json
import logging
from pathlib import Path
import shutil
from threading import Event, RLock
from time import monotonic

from PIL import Image, ImageDraw
from sqlalchemy import select

from ..agent import Agent
from ..agent.demo import DemoCatalog
from ..db import RequestRecord, ResultRecord, TraceRecord, UploadRecord
from ..db.models import utcnow


class JobService:
    def __init__(self, settings, database, registry):
        self.settings, self.db, self.registry = settings, database, registry
        self.agent = Agent(registry, trace_dir=settings.trace_dir, artifact_dir=settings.storage_dir / "prepared",
                           demo_catalog=DemoCatalog(settings.demo_manifest) if settings.demo_mode else None)
        self.queue = asyncio.Queue()
        self._state_lock = RLock()
        self._active = {}
        self._cancel = {}

    def cancel(self, request_id):
        with self.db.session.begin() as session:
            job = session.get(RequestRecord, request_id)
            if job is None or job.status not in {"queued", "running"}:
                return
            job.options = dict(job.options or {}, cancellation_requested=True)
        with self._state_lock:
            self._cancel.setdefault(request_id, Event()).set()
            self._active.setdefault(request_id, {})["message"] = "Cancellation requested; waiting for the active operation and model cleanup."

    def cancel_event(self, request_id):
        with self._state_lock:
            return self._cancel.setdefault(request_id, Event())

    def progress(self, request_id, message):
        with self._state_lock:
            event = self._cancel.get(request_id)
            if not (event and event.is_set()):
                self._active.setdefault(request_id, {"started": monotonic()})["message"] = message

    def recover(self):
        with self.db.session.begin() as session:
            jobs = session.scalars(select(RequestRecord).where(RequestRecord.status.in_(["queued", "running"]))).all()
            for job in jobs:
                job.status = "queued"
                if (job.options or {}).get("cancellation_requested"):
                    self.cancel_event(job.id).set()
                # An interrupted job may have left a partially prepared directory.
                for kind in ("prepared", "public"):
                    shutil.rmtree(self.settings.storage_dir / kind / job.id, ignore_errors=True)
                self.queue.put_nowait(job.id)

    async def worker(self):
        while True:
            request_id = await self.queue.get()
            try:
                if request_id is None:
                    return
                await asyncio.to_thread(self.execute, request_id)
            except Exception:
                logging.getLogger(__name__).exception("Unable to persist job failure for %s", request_id)
            finally:
                self.queue.task_done()

    def execute(self, request_id):
        try:
            with self.db.session.begin() as session:
                job = session.get(RequestRecord, request_id)
                if job is None or job.status != "queued":
                    return
                job.status, job.updated_at = "running", utcnow()
                query, options = job.query, dict(job.options or {})
                if options.pop("cancellation_requested", False):
                    self.cancel_event(request_id).set()
                uploads = session.scalars(select(UploadRecord).where(UploadRecord.request_id == request_id).order_by(UploadRecord.id)).all()
                files = [{"path": row.stored_path, "modality_override": row.modality_override} for row in uploads]
            with self._state_lock:
                event = self._cancel.setdefault(request_id, Event())
                self._active.setdefault(request_id, {})["started"] = monotonic()
            response = self.agent.run(files, query, options=options, request_id=request_id,
                progress=lambda message: self.progress(request_id, message), cancelled=event.is_set)
            payload = response.model_dump(mode="json")
            public_paths = []
            public = self.settings.storage_dir / "public" / request_id
            public.mkdir(parents=True, exist_ok=True)
            for index, preview in enumerate(response.previews):
                target = public / f"preview_{index}.png"
                shutil.copyfile(preview, target)
                public_paths.append(str(target))
            overlays = []
            if response.success and response.result and response.routing.task == "grounding":
                with Image.open(response.previews[0]) as original:
                    image = original.convert("RGB")
                draw = ImageDraw.Draw(image)
                evidence = response.result.evidence
                for box, label in zip(evidence.boxes, evidence.labels):
                    draw.rectangle(box, outline="red", width=2)
                    draw.text((box[0], box[1]), label[:60], fill="red")
                if response.result.metadata.mode == "stub":
                    draw.text((2, 2), "STUB: synthetic boxes", fill="red")
                overlay = public / "grounding_overlay.png"
                if evidence.overlay_path and Path(evidence.overlay_path).is_file():
                    shutil.copyfile(evidence.overlay_path, overlay)
                else:
                    image.save(overlay)
                overlays.append(str(overlay))
            payload["previews"] = public_paths
            payload["evidence_overlays"] = overlays
            trace = json.loads(Path(response.trace_path).read_text()) if response.trace_path else None
            if trace is None:
                raise RuntimeError(response.error or "Agent did not write an execution trace.")
            trace["api_artifacts"] = {"previews": public_paths, "evidence_overlays": overlays}
            self.write_trace(request_id, trace)
            with self.db.session.begin() as session:
                job = session.get(RequestRecord, request_id)
                job.status = "cancelled" if trace.get("cancelled") else ("done" if response.success else "failed")
                job.error, job.updated_at = response.error, utcnow()
                job.task = response.routing.task if response.routing else None
                job.model_id = response.routing.model_id if response.routing else None
                for row in session.scalars(select(UploadRecord).where(UploadRecord.request_id == request_id)):
                    info = next((info for info in trace["validation"]["images"] if info["path"] == row.stored_path), None)
                    if info:
                        row.modality, row.bands = info["modality"], info["band_count"]
                        row.width, row.height = info["width"], info["height"]
                chosen = trace.get("chosen_model") or {}
                session.merge(TraceRecord(request_id=request_id, path=str(self.settings.trace_dir / f"{request_id}.json"),
                    model_id=job.model_id, version=chosen.get("version"), mode=chosen.get("mode"),
                    device=trace.get("device"), runtime_ms=trace.get("runtime_ms"), success=response.success))
                session.merge(ResultRecord(request_id=request_id, answer_text=response.result.answer if response.result else None,
                    evidence_paths=public_paths + overlays, report_path=None, payload=payload))
        except Exception as exc:
            # Keep the worker alive, and persist a terminal state even for unexpected failures.
            with self.db.session.begin() as session:
                job = session.get(RequestRecord, request_id)
                if job is None:
                    return
                job.status, job.error, job.updated_at = "failed", str(exc), utcnow()
                path = self.settings.trace_dir / f"{request_id}.json"
                trace = json.loads(path.read_text()) if path.exists() else {
                    "request_id": request_id, "timestamp": utcnow().isoformat(), "query": job.query,
                    "validation": {"valid": False, "images": []}, "detected_modalities": [],
                    "routing": None, "chosen_model": None, "preprocessing_steps": [],
                    "parameters": job.options, "runtime_ms": None, "device": None,
                    "output": None, "warnings": [], "errors": [],
                }
                trace["errors"].append({"stage": "api_worker", "type": type(exc).__name__, "message": str(exc)})
                trace["api_status"] = "failed"
                try:
                    self.write_trace(request_id, trace)
                    session.merge(TraceRecord(request_id=request_id, path=str(path), success=False))
                except OSError:
                    job.error += "; execution trace could not be saved"

        finally:
            with self._state_lock:
                self._active.pop(request_id, None)
                self._cancel.pop(request_id, None)

    def write_trace(self, request_id, trace):
        self.settings.trace_dir.mkdir(parents=True, exist_ok=True)
        path = self.settings.trace_dir / f"{request_id}.json"
        temporary = path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(trace, indent=2, allow_nan=False))
        temporary.replace(path)

    def cleanup(self):
        cutoff = utcnow() - timedelta(hours=self.settings.retention_hours)
        with self.db.session.begin() as session:
            expired = session.scalars(select(RequestRecord).where(
                RequestRecord.status.in_(["done", "failed", "cancelled"]), RequestRecord.updated_at < cutoff,
                RequestRecord.artifacts_expired.is_(False))).all()
            for job in expired:
                for directory in ("raw", "prepared", "public"):
                    target = self.settings.storage_dir / directory / job.id
                    if target.is_symlink():
                        target.unlink()
                    elif target.exists():
                        shutil.rmtree(target)
                job.artifacts_expired = True
        # Traces and relational history survive image retention expiry.

    async def periodic_cleanup(self):
        while True:
            await asyncio.sleep(self.settings.cleanup_interval_seconds)
            await asyncio.to_thread(self.cleanup)

    def describe(self, session, job):
        result = session.get(ResultRecord, job.id)
        payload = result.payload if result else {}
        def urls(paths):
            return [f"/api/assets/{job.id}/{Path(path).name}" for path in paths
                    if not job.artifacts_expired and Path(path).is_file()]
        trace = session.get(TraceRecord, job.id)
        uploads = session.scalars(select(UploadRecord).where(UploadRecord.request_id == job.id).order_by(UploadRecord.id)).all()
        with self._state_lock:
            active = dict(self._active.get(job.id, {}))
            cancellation_requested = bool(self._cancel.get(job.id) and self._cancel[job.id].is_set())
        return dict(progress_message=active.get("message", "Queued; waiting for the specialist" if job.status == "queued" else job.status),
            elapsed_seconds=max(0, monotonic() - active["started"]) if "started" in active else None,
            cancellation_requested=cancellation_requested, request_id=job.id, time=job.time.isoformat(), query=job.query,
            task=job.task, model_id=job.model_id, status=job.status, error=job.error, input_explanation=payload.get("input_explanation"),
            execution_type=(payload.get("result") or {}).get("metadata", {}).get("execution_type"),
            result=payload.get("result"), routing=payload.get("routing"),
            preview_urls=urls(payload.get("previews", [])), evidence_overlay_urls=urls(payload.get("evidence_overlays", [])),
            warnings=payload.get("warnings", []), summary=payload.get("summary", ""),
            trace_url=f"/api/traces/{job.id}" if trace else None, artifacts_expired=job.artifacts_expired,
            uploads=[dict(filename=row.filename, modality=row.modality, bands=row.bands,
                          size=row.size, width=row.width, height=row.height) for row in uploads])
