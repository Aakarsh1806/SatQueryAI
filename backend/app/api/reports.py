"""Self-contained, print-friendly HTML analysis reports. No PDF dependency."""
import base64
from datetime import timezone
from html import escape
from io import BytesIO
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from uuid import uuid4

from PIL import Image
from sqlalchemy import select

from ..db import RequestRecord, ResultRecord, TraceRecord, UploadRecord
from ..preprocessing import optical_to_rgb, sar_prepare


def _image(path):
    if not path or not Path(path).is_file():
        return None
    with Image.open(path) as image:
        image = image.convert("RGB")
        image.thumbnail((520, 260))
        buffer = BytesIO()
        image.save(buffer, "PNG")
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")


def _figure(path, caption):
    source = _image(path)
    if source is None:
        return f'<figure><p>Image not available (retention may have expired).</p><figcaption>{escape(caption)}</figcaption></figure>'
    return f'<figure><img src="{source}" alt="{escape(caption, quote=True)}"><figcaption>{escape(caption)}</figcaption></figure>'


def build_report(session, request_id, settings):
    """Return a cached report or atomically create one and persist its path.

    Caller checks existence/status. Reports retain their embedded thumbnails
    after image retention expiry. Failed requests can also be reported honestly.
    """
    job = session.get(RequestRecord, request_id)
    row = session.get(ResultRecord, request_id)
    if row and row.report_path and Path(row.report_path).is_file():
        return Path(row.report_path)
    payload = row.payload if row else {}
    result = payload.get("result") or {}
    routing = payload.get("routing") or {}
    metadata = result.get("metadata") or {}
    trace_row = session.get(TraceRecord, request_id)
    trace = json.loads(Path(trace_row.path).read_text()) if trace_row and Path(trace_row.path).is_file() else {}
    mode = metadata.get("mode") or (trace.get("chosen_model") or {}).get("mode")
    mode_label = "pre-computed output (not live inference)" if metadata.get("execution_type") == "pre-computed" else ("stub output" if mode == "stub" else "real model output" if mode == "real" else "no model output")
    if metadata.get("demo_info", {}).get("placeholder"):
        mode_label += " · PLACEHOLDER, not a verified real-pair prediction"
    uploads = session.scalars(select(UploadRecord).where(UploadRecord.request_id == request_id).order_by(UploadRecord.id)).all()
    previews = payload.get("previews", [])
    # Fusion preprocessing orders optical first even if upload order was SAR first.
    preview_order = sorted(uploads, key=lambda upload: upload.modality != "optical") if job.task == "classification" else uploads
    preview_map = {upload.id: preview for upload, preview in zip(preview_order, previews)}
    figures = []
    with TemporaryDirectory(prefix="satquery-report-") as temp:
        for index, upload in enumerate(uploads):
            modality = upload.modality or upload.modality_override or "unknown"
            caption = f"Image {index + 1}: {upload.filename} · {modality}"
            path = preview_map.get(upload.id)
            if (not path or not Path(path).is_file()) and Path(upload.stored_path).is_file():
                try:
                    prepare = sar_prepare if modality == "sar" else optical_to_rgb
                    _, path = prepare(upload.stored_path, Path(temp) / f"{index}.png")
                except ValueError:
                    path = None
            figures.append(_figure(path, caption))
    evidence = result.get("evidence") or {}
    evidence_html = '<p>Evidence: not available.</p>'
    if job.task == "grounding" and evidence.get("boxes") is not None:
        overlays = payload.get("evidence_overlays", [])
        if not evidence["boxes"]:
            evidence_html = '<p>No boxes passed the threshold. Detection evidence: not available. Grounding is unvalidated.</p>'
        else:
            evidence_html = _figure(overlays[0] if overlays else None, "Grounding boxes · synthetic demonstration" if mode == "stub" else "Grounding candidate boxes (unvalidated, no masks)")
    elif job.task == "change_caption":
        evidence_html = '<p>Prepared before/after inputs for comparison; no change map or model-generated visual evidence.</p><div class="images">'
        for index, preview in enumerate(previews[:2]):
            evidence_html += _figure(preview, "A · Before" if index == 0 else "B · After")
        evidence_html += '</div>'
    elif evidence.get("top3"):
        evidence_html = '<p>Top-3 classes ' + ('(synthetic, uncalibrated)' if mode == "stub" else '(see confidence calibration label)') + ':</p><ul>'
        evidence_html += ''.join(f'<li>{escape(label)}: {probability:.1%}</li>' for label, probability in evidence["top3"])
        evidence_html += '</ul>'
    confidence = result.get("confidence")
    confidence_text = "not available"
    if not confidence and evidence.get("scores"):
        confidence_text = "Per-box scores (uncalibrated): " + "; ".join(f"{label}: {score:.1%}" for label,score in zip(evidence["labels"],evidence["scores"])) + ". Aggregate answer confidence: not available."
    if confidence:
        confidence_text = f'{confidence["value"]:.1%} · {"calibrated" if confidence["calibrated"] else "uncalibrated"} · {confidence["note"]}'
    routing_score = f"Routing score: {routing['confidence_of_routing']:.0%} · heuristic, uncalibrated (not model confidence)." if routing.get("confidence_of_routing") is not None else "Routing score: not available."
    compact = {key: trace.get(key) for key in ("request_id", "timestamp", "runtime_ms", "device", "chosen_model")}
    compact["preprocessing"] = [step.get("operation") for step in trace.get("preprocessing_steps", [])]
    compact["errors"] = trace.get("errors", [])
    warnings = list(payload.get("warnings", []))
    if mode == "stub" and not any("synthetic" in warning.lower() for warning in warnings):
        warnings.append("Stub output: synthetic demonstration, not real model inference.")
    if job.artifacts_expired:
        warnings.append("Original imagery expired under the configured retention period; some thumbnails may be unavailable.")
    timestamp = job.time.replace(tzinfo=timezone.utc) if job.time.tzinfo is None else job.time
    title = "SatQuery AI · Analysis report"
    document = f'''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<meta http-equiv="Content-Security-Policy" content="default-src 'none'; img-src data:; style-src 'unsafe-inline'">
<title>{title}</title><style>
@page{{size:A4;margin:12mm}}*{{box-sizing:border-box}}body{{font:12px/1.4 system-ui,sans-serif;color:#233b34;background:#eef2ed;margin:0;padding:20px}}article{{max-width:800px;margin:auto;background:white;padding:26px;border:1px solid #dce4dc}}h1{{font-size:23px;margin:0}}h2{{font-size:13px;margin:13px 0 5px}}p{{margin:4px 0;overflow-wrap:anywhere}}.header{{display:flex;justify-content:space-between;gap:15px;align-items:center}}.tag{{background:#fff0cc;color:#77571f;border:1px solid #ead9ad;padding:4px 8px;white-space:nowrap;font-weight:700}}.muted,figcaption{{color:#62746a;font-size:10px}}.images{{display:flex;gap:14px}}figure{{margin:0;flex:1;min-width:0}}figure img{{max-width:100%;max-height:128px;object-fit:contain;background:#f4f6f1}}figcaption{{overflow-wrap:anywhere}}.columns{{display:grid;grid-template-columns:1fr 1fr;gap:18px}}ul{{padding-left:18px;margin:4px 0}}li{{margin:2px 0}}pre{{font:9px/1.3 monospace;white-space:pre-wrap;overflow-wrap:anywhere;background:#f3f6f1;padding:8px;margin:4px 0}}.answer{{font-size:15px}}footer{{border-top:1px solid #dce4dc;margin-top:12px;padding-top:6px;font-size:9px;color:#62746a}}@media print{{body{{padding:0;background:white;font-size:10px}}article{{border:0;padding:0;max-width:none}}h1{{font-size:20px}}h2{{margin-top:9px}}figure img{{max-height:100px}}pre{{font-size:8px}}}}@media(max-width:550px){{.columns{{grid-template-columns:1fr}}body{{padding:8px}}article{{padding:16px}}.header{{flex-wrap:wrap}}}}
</style></head><body><article><header class="header"><h1>{title}</h1><span class="tag">{mode_label}</span></header>
<p class="muted">{escape(timestamp.isoformat())} · {escape(request_id)} · {escape(job.status)}</p>
<h2>Query</h2><p>{escape(job.query)}</p><h2>Input thumbnails & modalities</h2><div class="images">{''.join(figures)}</div>
<h2>Routing · {escape(metadata.get('model_name') or job.model_id or 'not available')}</h2><p>{escape(' '.join(routing.get('reasons', [])) or 'Routing not available.')}</p><p class="muted">{escape(routing_score)}</p>
<h2>Validation status</h2><p>{escape(metadata.get("validation_status") or "not available")}</p>
<h2>Answer</h2><p class="answer">{escape(result.get('answer') or job.error or 'not available')}</p>
<div class="columns"><section><h2>Confidence</h2><p>{escape(confidence_text)}</p><h2>Evidence / comparison</h2>{evidence_html}</section><section><h2>Warnings</h2><ul>{''.join('<li>' + escape(warning) + '</li>' for warning in warnings) or '<li>No additional warnings recorded.</li>'}</ul></section></div>
<h2>Compact execution trace</h2><pre>{escape(json.dumps(compact, indent=2))}</pre><footer>SIH26167 · Academic demonstration. Training domains: VRSBench / LEVIR-CC / Sentinel. Different imagery may perform poorly. Full execution trace: /api/traces/{escape(request_id)}. Open this file in a browser to print or save as PDF.</footer></article></body></html>'''
    directory = settings.storage_dir / "reports"
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / f"{request_id}.html"
    temporary = directory / f"{request_id}-{uuid4()}.tmp"
    temporary.write_text(document, encoding="utf-8")
    temporary.replace(target)
    if row is None:
        row = ResultRecord(request_id=request_id, answer_text=None, evidence_paths=[], payload={})
        session.add(row)
    row.report_path = str(target)
    return target
