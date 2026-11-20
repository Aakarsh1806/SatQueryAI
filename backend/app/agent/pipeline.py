"""One-request orchestration, prepared artifacts, and durable execution traces."""
from datetime import datetime, timezone
import json
import os
from pathlib import Path
from threading import RLock
from time import perf_counter
from typing import Literal
from uuid import UUID, uuid4

import numpy as np
import rasterio
from pydantic import Field, ValidationError

from ..models.registry import ModelRegistry, PROJECT_ROOT
from ..models.stubs import StubModel
from ..preprocessing import check_pair, domain_warnings, inspect, optical_to_rgb, sar_prepare, prepare_for, FusionInputError
from ..schemas import ChangeInput, FusionInput, GroundingInput, StandardResult, VLMInput
from ..schemas.contracts import Contract
from .routing import AgentInputError, RoutingDecision, detect_task

# Global across Agent instances: only one specialist may hold resources at once.
_MODEL_LOCK = RLock()
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")


class RequestCancelled(RuntimeError):
    """Cooperative cancellation; resources are released before returning."""


class ImageFile(Contract):
    path: str
    modality_override: Literal["optical", "sar"] | None = None


class AgentResponse(Contract):
    request_id: str
    success: bool
    result: StandardResult | None = None
    routing: RoutingDecision | None = None
    previews: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    summary: str = ""
    error: str | None = None
    trace_path: str | None = None
    input_explanation: str | None = None


def _save_bands(path, data, info, pair):
    """Persist aligned CHW bands, retaining names needed for RGB selection."""
    kwargs = {}
    if pair["crs"]:
        kwargs.update(crs=pair["crs"], transform=pair["transform"])
    else:
        # A synthetic transform is not a claim of geographic registration.
        kwargs["transform"] = rasterio.transform.from_origin(0, data.shape[1], 1, 1)
    with rasterio.open(path, "w", driver="GTiff", width=data.shape[2], height=data.shape[1],
                       count=data.shape[0], dtype="float32", nodata=np.nan, **kwargs) as dst:
        dst.write(data.astype("float32"))
        dst.descriptions = tuple(info["band_descriptions"])


class Agent:
    def __init__(self, registry=None, *, trace_dir=None, artifact_dir=None,
                 llm_fallback=None, enable_llm_fallback=False, demo_catalog=None):
        self.registry = registry if registry is not None else ModelRegistry()
        self.trace_dir = Path(trace_dir) if trace_dir is not None else PROJECT_ROOT / "traces"
        self.artifact_dir = Path(artifact_dir) if artifact_dir is not None else PROJECT_ROOT / "uploads/prepared"
        self.demo_catalog = demo_catalog
        self.llm_fallback = llm_fallback
        self.enable_llm_fallback = enable_llm_fallback

    def run(self, files, query, *, options=None, request_id=None, progress=None, cancelled=None):
        """Accept paths or ImageFile dictionaries; return UI output even on failure.

        Optical+SAR inputs may arrive in either order. Two optical inputs retain
        upload order (before, after). Options must be JSON-serializable.
        """
        def checkpoint(message):
            if cancelled and cancelled():
                raise RequestCancelled("Request cancelled; model resources released.")
            if progress:
                progress(message)

        started = perf_counter()
        request_id = str(UUID(str(request_id))) if request_id is not None else str(uuid4())
        response = AgentResponse(request_id=request_id, success=False)
        trace = dict(request_id=request_id, timestamp=datetime.now(timezone.utc).isoformat(),
                     query=query if isinstance(query, str) else str(query),
                     validation={"valid": False, "images": []}, detected_modalities=[],
                     routing=None, chosen_model=None, preprocessing_steps=[], parameters={},
                     runtime_ms=0, device=None, output=None, warnings=[], errors=[])
        stage = "validation"
        try:
            checkpoint("Validating images and detecting modalities")
            if not isinstance(files, (list, tuple)) or not 1 <= len(files) <= 2:
                raise AgentInputError("Upload one or two images; this request supports at most two images.")
            if not isinstance(query, str) or not query.strip():
                raise AgentInputError("Enter a question or a caption request.")
            params = dict(options or {})
            # Reject opaque objects/NaN before they can make a trace unwritable.
            json.dumps(params, allow_nan=False)
            trace["parameters"] = params.copy()
            images = []
            for file in files:
                item = ImageFile(path=str(file)) if isinstance(file, (str, Path)) else ImageFile.model_validate(file)
                info = inspect(item.path, modality_override=item.modality_override)
                if all(value["min"] is None for value in info["value_ranges"]):
                    raise AgentInputError(f"Image {Path(item.path).name} contains no valid pixels.")
                images.append(info)
                trace["validation"]["images"].append(info)
                trace["detected_modalities"].append({"path": info["path"], "guess": info["guessed_modality"],
                    "modality": info["modality"], "reason": info["modality_reason"]})
                response.warnings.extend(domain_warnings(info))
            decision = detect_task([info["modality"] for info in images], query,
                llm_fallback=self.llm_fallback, enable_llm_fallback=self.enable_llm_fallback)
            trace["validation"]["valid"] = True
            trace["routing"] = decision.model_dump()
            response.routing = decision
            if decision.task in {"change_caption", "classification"}:
                response.warnings.extend(decision.reasons)
            response.warnings.append("Routing confidence is a heuristic score, not calibrated model confidence.")
            entry = self.registry.entries[decision.model_id]
            trace["chosen_model"] = dict(id=entry.id, version=entry.version, configured_mode=entry.mode, mode=None)
            checkpoint("Preparing model-native inputs and separate display previews")
            stage = "preprocessing"
            workdir = self.artifact_dir / request_id
            workdir.mkdir(parents=True, exist_ok=False)
            inputs, model_query, prepared_native = self._prepare(images, query, decision, params, workdir, trace, response)
            stage = "model"
            trace["parameters"] = params.copy()
            result = self.demo_catalog.lookup(images, query, decision) if self.demo_catalog else None
            if result is not None:
                result.metadata.runtime_ms = (perf_counter() - started) * 1000
                trace["chosen_model"].update(mode=result.metadata.mode, version=result.metadata.version,
                    execution_type="pre-computed")
                trace["device"] = result.metadata.device
                trace["parameters"] = result.metadata.parameters
                trace["lifecycle"] = dict(exclusive_model_lock=False, loaded=False, unloaded=True,
                    reason="pre-computed playback; no model loaded")
            else:
                checkpoint("Waiting for exclusive model access")
                with _MODEL_LOCK:
                    checkpoint("Checking the selected specialist and execution mode")
                    model = self.registry.get(decision.model_id)
                    trace["chosen_model"].update(mode="stub" if isinstance(model, StubModel) else "real",
                        fallback_reason=getattr(model, "fallback_reason", None))
                    checkpoint("Generating synthetic stub output; no model weights are loaded." if isinstance(model, StubModel) else "Loading the selected real specialist. Cancellation waits for the active operation and cleanup.")
                    model.progress = progress
                    trace["lifecycle"] = {"exclusive_model_lock": True, "loaded": False, "unloaded": False}
                    try:
                        stage = "model_load"
                        model.load()
                        stage = "model"
                        trace["lifecycle"]["loaded"] = model.is_loaded
                        checkpoint("Running selected specialist")
                        run_inputs = prepared_native if getattr(model, "accepts_prepared_inputs", False) else None
                        result = StandardResult.model_validate(model.run(run_inputs if run_inputs is not None else inputs, model_query, params))
                        trace["device"] = result.metadata.device
                        trace["parameters"] = result.metadata.parameters
                        trace["chosen_model"].update(version=result.metadata.version, mode=result.metadata.mode)
                    finally:
                        model.unload()
                        trace["lifecycle"]["unloaded"] = not model.is_loaded
            checkpoint("Integrating supported output and execution trace")
            if result.metadata.execution_type is None:
                result.metadata.execution_type = "stub" if result.metadata.mode == "stub" else "live"
            trace["execution_type"] = result.metadata.execution_type
            if result.metadata.execution_type == "pre-computed":
                trace["demo_info"] = result.metadata.demo_info
            result.metadata.validation_status = entry.validation_status
            stage = "integration"
            response.result = result
            response.warnings.extend(result.metadata.warnings)
            if decision.model_id == "fusion" and result.metadata.mode == "stub":
                response.warnings.append("Model 4 adapter has not been verified on a real pair; stub output is a placeholder, not a land-cover prediction.")
            if result.metadata.mode == "stub":
                response.warnings.append("Stub demonstration: answers, boxes, and scores are synthetic; no model inference was performed.")
            if result.metadata.fallback_reason:
                response.warnings.append(f"Using stub fallback: {result.metadata.fallback_reason}.")
            if result.confidence and not result.confidence.calibrated:
                response.warnings.append(result.confidence.note)
            if decision.task == "grounding":
                response.summary = "Returned candidate boxes in original-image pixel coordinates; grounding is unvalidated and no masks are produced."
            elif decision.task == "change_caption":
                response.summary = "Returned a description comparing before A with after B; not a direct answer or change map."
            elif decision.task == "classification":
                response.summary = f"Land-cover class: {result.answer}."
            else:
                response.summary = result.answer
            if result.metadata.mode == "stub":
                response.summary = "Stub demonstration. " + response.summary
            if result.metadata.execution_type == "pre-computed":
                response.summary = "Pre-computed playback (not live inference). " + response.summary
            response.success = True
        except Exception as exc:
            if isinstance(exc, FusionInputError):
                response.input_explanation = str(exc)
                trace["validation"].update(valid=False, code="unsupported_fusion_input", explanation=str(exc))
                if response.routing:
                    response.routing.reasons.append(str(exc))
                    trace["routing"] = response.routing.model_dump()
            message = f"Invalid request: {exc}" if isinstance(exc, (ValidationError, TypeError)) else str(exc)
            if stage == "model_load" and not isinstance(exc, RequestCancelled):
                message = "Model could not be loaded. No live result was produced. " + message
            response.error = message or "The request could not be completed."
            response.summary = response.error
            trace["errors"].append({"stage": stage, "type": type(exc).__name__, "message": response.error})
        response.warnings = list(dict.fromkeys(response.warnings))
        trace["runtime_ms"] = (perf_counter() - started) * 1000
        trace["cancelled"] = any(error["type"] == "RequestCancelled" for error in trace["errors"])
        trace["warnings"] = response.warnings
        trace["output"] = response.model_dump(mode="json", exclude={"trace_path"})
        try:
            self.trace_dir.mkdir(parents=True, exist_ok=True)
            target = self.trace_dir / f"{request_id}.json"
            temporary = target.with_suffix(".json.tmp")
            temporary.write_text(json.dumps(trace, indent=2, allow_nan=False), encoding="utf-8")
            temporary.replace(target)
            response.trace_path = str(target)
        except OSError as exc:
            response.success = False
            response.error = f"Could not save the execution trace: {exc}"
            response.summary = response.error
        return response

    def _prepare(self, images, query, decision, params, workdir, trace, response):
        steps = trace["preprocessing_steps"]
        if decision.model_id == "fusion" and self.registry.resolve("fusion")["mode"] == "real":
            images = sorted(images, key=lambda info: info["modality"] != "optical")
            prepared = prepare_for("fusion", [image["path"] for image in images], output_dir=workdir,
                                   artifact_dir=self.registry.project_root / self.registry.entries["fusion"].weights_path)
            response.previews.extend(prepared.previews)
            response.warnings.extend(prepared.warnings)
            steps.extend(prepared.notes)
            return FusionInput(optical_path=prepared.native_paths[0], sar_path=prepared.native_paths[1]), query, prepared
        if decision.model_id in {"rs_vlm", "grounding", "change"}:
            entry = self.registry.entries[decision.model_id]
            prepared = prepare_for(decision.model_id, [image["path"] for image in images], output_dir=workdir, config=entry.preprocessing)
            response.previews.extend(prepared.previews)
            response.warnings.extend(prepared.warnings)
            steps.extend(prepared.notes)
            if decision.model_id == "grounding":
                return GroundingInput(image_path=prepared.native_paths[0], text_prompt=query, box_threshold=params.get("box_threshold",.3)), query, prepared
            if decision.model_id == "change":
                return ChangeInput(before_path=prepared.native_paths[0], after_path=prepared.native_paths[1]), query, prepared
            prompt = "Describe this satellite image in detail." if decision.task == "caption" else query
            suffix = " Answer with a short phrase."
            if decision.task == "qa" and not prompt.endswith(suffix): prompt += suffix
            params.update(prompt=prompt, task=decision.task)
            return VLMInput(image_path=prepared.native_paths[0], task=decision.task), prompt, prepared
        if decision.task == "classification":
            images = sorted(images, key=lambda info: info["modality"] != "optical")
        a, b = images
        mode = "optical+sar" if decision.task == "classification" else "before/after"
        pair = check_pair(a["path"], b["path"], mode, modality_a=a["modality"], modality_b=b["modality"])
        response.warnings.extend(pair["warnings"])
        steps.append(dict(operation="check_pair", mode=mode, source_a=a["path"], source_b=b["path"],
                          crs=pair["crs"], bounds=pair["bounds"],
                          transform=list(pair["transform"]) if pair["transform"] is not None else None,
                          shape_a=list(pair["array_a"].shape), shape_b=list(pair["array_b"].shape),
                          geospatial_checks_performed=pair["geospatial_checks_performed"], warnings=pair["warnings"]))
        raw_a, raw_b = workdir / "aligned_a.tif", workdir / "aligned_b.tif"
        _save_bands(raw_a, pair["array_a"], a, pair)
        _save_bands(raw_b, pair["array_b"], b, pair)
        _, preview_a = optical_to_rgb(raw_a, workdir / "a_rgb.png")
        response.previews.append(str(preview_a))
        steps.append(dict(operation="optical_to_rgb", source=str(raw_a), preview=str(preview_a), stretch_percentiles=[2, 98]))
        if decision.task == "change_caption":
            _, preview_b = optical_to_rgb(raw_b, workdir / "b_rgb.png")
            response.previews.append(str(preview_b))
            steps.append(dict(operation="optical_to_rgb", source=str(raw_b), preview=str(preview_b), stretch_percentiles=[2, 98]))
            return ChangeInput(before_path=str(preview_a), after_path=str(preview_b)), query, None
        sar, preview_b = sar_prepare(raw_b, workdir / "b_sar.png", input_scale=params.get("sar_input_scale", "auto"))
        normalized = workdir / "sar_normalized.tif"
        _save_bands(normalized, np.moveaxis(sar, -1, 0), b, pair)
        response.previews.append(str(preview_b))
        steps.append(dict(operation="sar_prepare", source=str(raw_b), preview=str(preview_b),
                          normalized_path=str(normalized), input_scale=params.get("sar_input_scale", "auto"),
                          db_range=[-30, 5], normalization=[0, 1]))
        return FusionInput(optical_path=str(preview_a), sar_path=str(normalized)), query, None
