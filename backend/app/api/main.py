"""FastAPI application factory and development server entry point."""
import asyncio
import base64
from io import BytesIO
from contextlib import asynccontextmanager, suppress
import json
from pathlib import Path
import shutil
from uuid import UUID, uuid4

from fastapi import FastAPI, File, Form, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from sqlalchemy import func, select, text
from starlette.concurrency import run_in_threadpool
from starlette.staticfiles import StaticFiles

from ..db import Database, RequestRecord, TraceRecord, UploadRecord
from ..models.registry import ModelRegistry
from ..preprocessing import PreprocessingError, inspect, optical_to_rgb, sar_prepare
from PIL import Image
from .config import Settings
from .service import JobService
from .reports import build_report
from .uploads import BodyLimitMiddleware, save_upload


def canonical_id(value):
    try:
        return str(UUID(value))
    except ValueError:
        raise HTTPException(404, "Request not found.") from None


def parse_overrides(value, count):
    try:
        overrides = json.loads(value) if value else [None] * count
    except ValueError:
        raise HTTPException(422, 'modality_overrides must be a JSON list, e.g. ["optical", "sar"].') from None
    if not isinstance(overrides, list) or len(overrides) != count or any(item not in (None, "optical", "sar") for item in overrides):
        raise HTTPException(422, "Provide one modality override per image: optical, sar, or null.")
    return overrides


def create_app(settings=None, registry=None):
    settings = settings or Settings()
    registry = registry if registry is not None else ModelRegistry()
    database = Database(settings.database_url)
    service = JobService(settings, database, registry)
    public = settings.storage_dir / "public"
    public.mkdir(parents=True, exist_ok=True)

    @asynccontextmanager
    async def lifespan(app):
        database.initialize(registry)
        service.recover()
        await asyncio.to_thread(service.cleanup)
        worker = asyncio.create_task(service.worker())
        cleanup = asyncio.create_task(service.periodic_cleanup())
        app.state.worker = worker
        try:
            yield
        finally:
            cleanup.cancel()
            with suppress(asyncio.CancelledError):
                await cleanup
            await service.queue.put(None)
            await worker
            database.engine.dispose()

    app = FastAPI(title="SatQuery AI", lifespan=lifespan)
    app.state.settings, app.state.database, app.state.jobs = settings, database, service
    app.add_middleware(BodyLimitMiddleware, limit=settings.max_request_bytes)
    app.add_middleware(CORSMiddleware, allow_origins=settings.cors_origins,
                       allow_methods=["GET", "POST"], allow_headers=["*"])
    # Only explicitly copied PNGs/overlays are exposed; raw TIFFs and traces are private.
    app.mount("/api/assets", StaticFiles(directory=public, follow_symlink=False), name="assets")

    @app.get("/api/health")
    def health():
        with database.session() as session:
            session.execute(text("SELECT 1"))
        return {"status": "ok", "worker": "running" if not app.state.worker.done() else "stopped",
                "demo_mode": settings.demo_mode, "demo_entries": len(service.agent.demo_catalog.entries) if service.agent.demo_catalog else 0}

    @app.get("/api/models")
    def models():
        entries = []
        for entry in registry.entries.values():
            item = entry.model_dump()
            resolution = registry.resolve(entry.id) if entry.enabled else {"mode": entry.mode, "fallback_reason": "disabled"}
            item.update(configured_mode=entry.mode, **resolution)
            if entry.id == "grounding":
                item["availability_note"] = "Unvalidated: no trainer benchmark metrics; local sanity checks do not establish general accuracy."
            if entry.id == "fusion":
                item["availability_note"] = "Real adapter available, but no genuine aligned pair has been verified. Sample-pair testing is pending."
            entries.append(item)
        return {"models": entries}

    @app.post("/api/analyze", status_code=202)
    async def analyze(files: list[UploadFile] = File(...), query: str = Form(...), modality_overrides: str | None = Form(None)):
        if not 1 <= len(files) <= 2:
            raise HTTPException(422, "Upload one or two images; this task supports at most two images.")
        if not query.strip() or len(query) > 10000:
            raise HTTPException(422, "Enter a question or caption request (maximum 10000 characters).")
        overrides = parse_overrides(modality_overrides, len(files))
        request_id = str(uuid4())
        directory = settings.storage_dir / "raw" / request_id
        saved = []
        try:
            for index, file in enumerate(files):
                saved.append(await save_upload(file, directory, index, settings.max_upload_bytes))
            # Decode validation, modality detection and all inference happen in the worker.
            with database.session.begin() as session:
                session.add(RequestRecord(id=request_id, query=query, status="queued", options={}))
                session.flush()
                for item, override in zip(saved, overrides):
                    session.add(UploadRecord(request_id=request_id, **item, modality_override=override))
        except BaseException:
            shutil.rmtree(directory, ignore_errors=True)
            raise
        finally:
            for file in files:
                await file.close()
        service.queue.put_nowait(request_id)
        return {"request_id": request_id, "status": "queued", "status_url": f"/api/requests/{request_id}",
                "result": None, "routing": None, "preview_urls": [], "warnings": []}

    @app.get("/api/requests")
    def history(limit: int = Query(20, ge=1, le=100), offset: int = Query(0, ge=0)):
        with database.session() as session:
            total = session.scalar(select(func.count()).select_from(RequestRecord))
            rows = session.scalars(select(RequestRecord).order_by(RequestRecord.time.desc(), RequestRecord.id).limit(limit).offset(offset)).all()
            return {"total": total, "limit": limit, "offset": offset, "requests": [service.describe(session, row) for row in rows]}

    @app.get("/api/requests/{request_id}")
    def request_status(request_id: str):
        with database.session() as session:
            job = session.get(RequestRecord, canonical_id(request_id))
            if job is None:
                raise HTTPException(404, "Request not found.")
            return service.describe(session, job)

    @app.post("/api/requests/{request_id}/cancel", status_code=202)
    def cancel_request(request_id: str):
        request_id = canonical_id(request_id)
        with database.session() as session:
            job = session.get(RequestRecord, request_id)
            if job is None:
                raise HTTPException(404, "Request not found.")
            if job.status in {"queued", "running"}:
                service.cancel(request_id)
            return service.describe(session, job)

    @app.get("/api/traces/{request_id}")
    def trace(request_id: str):
        with database.session() as session:
            row = session.get(TraceRecord, canonical_id(request_id))
            if row is None or not Path(row.path).is_file():
                raise HTTPException(404, "Trace is not available yet or the request does not exist.")
            return FileResponse(row.path, media_type="application/json")

    @app.post("/api/inspect")
    async def inspect_upload(file: UploadFile = File(...), modality_override: str | None = Form(None)):
        if modality_override not in {None, "optical", "sar"}:
            raise HTTPException(422, "Modality override must be optical or sar.")
        directory = settings.storage_dir / "raw" / f"inspect-{uuid4()}"
        try:
            item = await save_upload(file, directory, 0, settings.max_upload_bytes)
            info = await run_in_threadpool(inspect, item["stored_path"], modality_override)
            problems = []
            expected_bands = 13 if info["modality"] == "optical" else 2
            if Path(item["stored_path"]).suffix.lower() not in {".tif", ".tiff"}:
                problems.append("use a GeoTIFF; PNG/JPG are not supported for real fusion")
            if info["band_count"] != expected_bands:
                problems.append(f"{info['modality']} needs {expected_bands} bands, got {info['band_count']}")
            if (info["width"], info["height"]) != (256, 256):
                problems.append("patch must be exactly 256 × 256")
            if not info["crs"]:
                problems.append("a non-null CRS is required")
            info["fusion_eligibility"] = {"eligible_individually": not problems,
                "message": "Real fusion: " + ("; ".join(problems) if problems else "individual format qualifies; matching CRS, transform, finite values and band order are checked with its pair.")}
            def make_thumbnail():
                if info["modality"] == "sar":
                    _, preview = sar_prepare(item["stored_path"], directory / "preview.png")
                else:
                    _, preview = optical_to_rgb(item["stored_path"], directory / "preview.png", sentinel_full=info["band_count"] == 13)
                with Image.open(preview) as image:
                    image.thumbnail((320, 240))
                    buffer = BytesIO()
                    image.save(buffer, format="PNG")
                return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")
            try:
                info["preview_data_url"] = await run_in_threadpool(make_thumbnail)
            except PreprocessingError as exc:
                info["preview_data_url"] = None
                info["warnings"].append(f"Preview unavailable: {exc}")
            info.pop("path", None)  # this temporary file is deleted immediately
            return info
        except PreprocessingError as exc:
            raise HTTPException(422, str(exc)) from exc
        finally:
            await file.close()
            shutil.rmtree(directory, ignore_errors=True)

    @app.get("/api/report/{request_id}")
    def report(request_id: str):
        request_id = canonical_id(request_id)
        with database.session.begin() as session:
            job = session.get(RequestRecord, request_id)
            if job is None:
                raise HTTPException(404, "Request not found.")
            if job.status not in {"done", "failed"}:
                raise HTTPException(409, "The analysis is still queued or running; download its report when it finishes.")
            path = build_report(session, request_id, settings)
        return FileResponse(path, media_type="text/html", filename=f"satquery-{request_id}.html")

    if settings.static_dir is not None:
        if not (settings.static_dir / "index.html").is_file():
            raise ValueError("SATQUERY_STATIC_DIR must contain a built frontend index.html")
        # Mounted last: API routes always take precedence, and raw files stay private.
        app.mount("/", StaticFiles(directory=settings.static_dir, html=True, follow_symlink=False), name="frontend")
    return app


app = create_app()
