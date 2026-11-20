PROJECT: SatQuery AI is a college project (SIH26167): a vision-language
assistant for satellite imagery. The user uploads imagery and asks a
natural-language question. An AI Agent validates inputs, picks ONE
specialist model, runs it, and returns answer + evidence + confidence +
execution trace. Input formats: GeoTIFF/TIFF (also allow PNG/JPG for demos).

FOUR SPECIALISTS (one model per query, never all four):
1. RS-VLM (Qwen2.5-VL-3B + LoRA adapter, trained on VRSBench): 1 optical
   image + question or caption request -> text. QA prompts must end with
   " Answer with a short phrase."; caption prompt is exactly
   "Describe this satellite image in detail."
2. Grounding (Grounding DINO, VRSBench): 1 optical image + text phrase ->
   boxes, labels, scores. Boxes only, NO masks. Contract:
   ground(image_path, text_prompt, box_threshold=0.3) ->
   {"boxes": [[x1,y1,x2,y2]] absolute pixels, "labels": [...], "scores": [...]}
3. Change (Chg2Cap, LEVIR-CC): 2 optical images (before=A, after=B) ->
   a change caption. It does NOT answer questions about changes and does
   NOT produce change maps. Images are 256x256.
4. Fusion (dual encoder, SEN12MS): optical + SAR pair of the same area ->
   land-cover class. Contract: fuse_predict(optical_path, sar_path) ->
   {"label": str, "confidence": float, "top3": [(label, prob), ...]}.
   Classifier only, no free-form answers. A single SAR image is not valid.

STANDARD RESULT OBJECT (every model returns this shape):
{answer, evidence, confidence, metadata}. Use confidence = null with
reason "not available" when a model cannot supply it. Never show
evidence or confidence a model cannot support. Confidence from softmax or
token log-probs must be labelled "uncalibrated".

ROUTING: 1 optical -> Model 1 or 2 (keywords like where/locate/find/
highlight/show/detect -> grounding, otherwise QA/caption); 2 optical ->
Model 3; optical+SAR -> Model 4; SAR only -> reject with a clear message.
Hybrid design: rules first; optional LLM fallback hook (disabled by default).

DESIGN RULES: modular, each model behind the same interface and a model
registry (supported inputs, tasks, outputs, version, weights path, enabled).
Lazy-load models, one at a time, unload after use. Device-agnostic:
cuda -> mps -> cpu; set PYTORCH_ENABLE_MPS_FALLBACK=1. Demo machine is a
MacBook Air M5 (no CUDA, no bitsandbytes). Stubs must exist for all four
models so the whole system runs without weights.

STORAGE: SQLite via SQLAlchemy (must be swappable to PostgreSQL by changing
the connection string). Images/weights/reports are files on disk; the DB
stores paths and metadata only. Every request writes a JSON execution trace.

STACK: Python + FastAPI backend, React (Vite) frontend, rasterio/numpy for
geospatial, PyTorch for models.

HONESTY RULES: no overclaiming in UI text or docs. Warn on domain gap
(models were trained on VRSBench / LEVIR-CC / Sentinel data; very
different imagery may give poor results). Datasets are academic-use only.

UI DESIGN SYSTEM: Follow docs/design_system.md and frontend/src/styles/tokens.css
for frontend changes. Use self-hosted IBM Plex Sans/Mono, Lucide line icons,
8px spacing, navy/blue colours and bordered surfaces; no gradients, glow,
glassmorphism, shadows or decorative animation. Use the shared components
in frontend/src/components/ui with keyboard support and visible focus.
Never remove or weaken Real / Stub / Pre-computed, unvalidated,
uncalibrated, not-verified, unavailable-evidence/confidence or domain-gap
labels. Global notices may collapse; result-specific honesty stays visible.
Read model modes from the API rather than hardcoding demo assumptions.
