# Full system verification — 11 November 2026

> Historical run record: preserves the configuration and measurements from that stage. For current modes, demo outcomes and caveats, see [final_review.md](final_review.md). Model 4 remains a stub, not yet verified on a real pair.

## Activation and evidence

All four adapters are implemented and execute in the pinned ML environment. All four **saved registry modes remain `stub`**: no genuine image sample is present in `data_samples/`, so no genuine-sample check script has passed. Generated fixtures are execution smoke tests only. `/api/models` exposes each effective/configured mode, validation status, and a human-readable `fallback_reason`; the upload panel's Model readiness disclosure displays these fields. Every successful result receives the registry's validation status, including stub results.

Model 1 is recorded as measured on VRSBench val **as reported by the user**, with scores still pending. Model 2 is unvalidated with no metrics. Model 3 and Model 4 carry the exact test figures supplied by the user; these figures were not independently reproduced by this task. Displayed validation figures describe the trained model, not synthetic stub answers or confidence calibration.

`docs/local_model_checks.json` records four explicit **skipped** outcomes, not passes. `scripts/check_system.py` runs sample-backed real-mode API checks in an isolated temporary database without changing YAML. When samples arrive, run:

```sh
.venv/bin/python scripts/check_model1.py
.venv/bin/python scripts/check_model3.py
.venv/bin/python scripts/check_system.py
```

Existing Model 1/3 scripts perform their fuller question/reference checks. The system script covers all four real adapters through POST, polling, result, and trace endpoints. Review predictions and the genuine-image/parity checks before changing only passing entries to `mode: real`. It intentionally does not auto-enable models or treat a no-samples exit as a pass. Sample provenance must be known by the operator; the script cannot authenticate dataset provenance.

Grounding samples belong in `data_samples/grounding/`. Fusion pairs use `data_samples/fusion/<pair>/optical.tif` and `sar.tif`, plus the teammate's expected prediction text for the optional parity test. Change pairs and references belong in `data_samples/change/` with `captions.txt`; layouts are described in the existing sample helpers/README.

## Input and result handling

The UI explains strict fusion requirements before upload: original 13-band Sentinel-2 DN GeoTIFF, original 2-band VV/VH dB GeoTIFF, exactly 256×256, identical non-null CRS and pixel grid. `/api/inspect` adds an individual format eligibility explanation alongside band count, dimensions, dtype and georeferencing. Pair georeferencing, finite values and band-order validation remain authoritative in preprocessing. An individual badge does not promise that the pair qualifies. PNG/RGB/wrong-size inputs get friendly requirements text.

Full 13-band Sentinel-2 inspection previews explicitly select B4/B3/B2 (indices 3/2/1 for B1-first stacks). These stretched RGB previews and SAR display previews never enter fusion inference. Raw normalization, strict no-resampling behavior, weights and class order are unchanged.

Change guidance and returned routing reasons explain that the model describes two same-area optical inputs, with no change map or direct change-question answer. Grounding shows the boxes/overlay and each box's uncalibrated score; aggregate confidence remains unavailable. All results display their stub/real mode and validation status.

## Slow requests and cancellation

GET request status now includes `progress_message`, `elapsed_seconds` and `cancellation_requested`. The UI updates its elapsed timer each second while polling and exposes Cancel request. Stages distinguish validation, native/display preparation, exclusive model access, loading, inference and integration. RS-VLM checks the local HF shard cache to distinguish cached loading from missing base files that will download; applying and strictly checking LoRA is a separate progress message. The local base cache was already complete during this verification; an actual fresh network download was not repeated.

`POST /api/requests/{id}/cancel` is idempotent, returns 202, rejects unknown IDs with 404, and leaves terminal jobs unchanged. Cancellation is cooperative: queued jobs are cancelled when dequeued, and an active load or inference operation finishes before cleanup and cancellation acknowledgement. It does not kill Torch inside a kernel or immediately abort an HTTP download. The UI explicitly states that it is waiting for the active operation and cleanup. A result produced after cancellation is discarded. Cancellation intent is persisted in request options and respected on restart; cancelled jobs retain traces/history and participate in upload retention cleanup. No report button is offered for cancelled/failed jobs.

## Memory and environment

The API uses one inference queue worker. A process-wide lock across Agent instances covers specialist construction, load, run and unload, so even concurrent Agent calls load at most one specialist. All four real adapters clear model references and call the shared cache helper (GC, CUDA cache if available, MPS cache if available). Agent `finally` cleanup also covers exceptions and cancellation. Traces record exclusive access, loaded state and unloaded state. Tests exercise two concurrent Agent instances, cancellation after load, and cache helper calls.

This guarantee is **per Python process**. Run one Uvicorn worker; do not launch multiple inference servers or `--workers > 1` on the demo machine. Cache release does not promise that process RSS returns to its original value; Python/Torch allocators and OS caches can retain memory. No forced process termination or OS-wide model lock was added.

Models 1 and 2 both strictly load and infer with Transformers **5.22.2**, Torch **2.15.0**, torchvision **0.29.2**, PEFT **0.21.2**, and the existing pinned ML requirements. A separate conflicting-model worker is unnecessary in this tested environment. Fusion remains CPU-only. No weights, teammate endpoints, eager server loading, or wildcard CORS were introduced.

## Measured checks

All following inference samples are **generated fixtures**, not satellite validation imagery. Tests use temporary all-real registry overrides; the normal YAML stays stub.

| Model | Real API smoke, POST to terminal | UI request trace total | UI model-reported inference/runtime | Device |
|---|---:|---:|---:|---|
| RS-VLM | 9.475 s | 18.933 s | 8.305 s | MPS |
| Grounding | 0.957 s | 1.228 s | 0.345 s | MPS |
| Chg2Cap | 1.827 s | 2.358 s | 0.226 s | MPS |
| Fusion | 0.249 s | 0.580 s | 0.054 s | CPU |

These are individual warm/cold mixed local runs, not benchmarks. API wall time includes queue, loading, preprocessing, tracing and persistence; UI trace totals exclude browser upload and final polling latency. First RS-VLM UI use also included runtime imports. No accuracy conclusion follows from these outputs.

UI checks used the real adapters in an isolated temporary server/database: upload→inspect→submit→poll→render was verified for all four. Grounding showed an overlay plus per-box scores; change showed the before/after slider, unavailable confidence and limitation note; fusion showed raw-input badges, top-3 bars and reported metrics; RS-VLM showed generated text with uncalibrated confidence and no invented evidence. RGB PNG + SAR rejection was verified. Cancelling RS-VLM during loading reached `cancelled` after 7.368 s total, with `loaded: true, unloaded: true` in its trace. A screenshot is saved at `docs/screenshots/full-fusion-result.png`.

All-real intent routing: **30/30 (100%)**, including invalid combinations. This is routing accuracy only.

Backend full suite: **95 passed, 2 skipped** (20.59 s); the two skips are the missing genuine grounding sample and aligned fusion parity pair. After the final progress/cancellation persistence changes, the affected API/Agent/real-model suites passed **45 tests** (16.98 s). An additional restart-cancellation test passed after those runs. Frontend: **3 tests passed**, production build passed. Existing dependency/deprecation warnings do not indicate failed inference.

No genuine-sample UI/API run or local dataset evaluation could be completed. Open items: genuine optical/grounding imagery, LEVIR-CC pairs and references to judge unconfirmed normalization, aligned SEN12MS optical/SAR pairs and expected labels, and Model 1 VRSBench scores.

## Local commands

From the repository root, in separate terminals:

```sh
.venv/bin/python -m uvicorn backend.app.api.main:app --host 127.0.0.1 --port 8000
cd frontend
npm run dev
```

Open http://127.0.0.1:5173. Temporary verification servers were stopped after testing.
