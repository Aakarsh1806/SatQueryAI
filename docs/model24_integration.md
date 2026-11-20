# Model 2 / Model 4 integration verification

> Historical run record: preserves the configuration and measurements from that stage. For current modes, demo outcomes and caveats, see [final_review.md](final_review.md). Model 4 remains a stub, not yet verified on a real pair.

11 November 2026 (Asia/Kolkata). In-process adapters reuse the teammates'
model/processor and normalization/inference logic. Neither teammate FastAPI
backend is imported or started. Their endpoints, eager module loading,
wildcard CORS and temporary-file upload flow are not copied.

The four checked-in YAML modes remain **stub**, preserving the previous request
that the user switches them. Real-mode checks used isolated runtime registries
and temporary storage/SQLite databases; no normal project history was populated.

## Model 2: Grounding DINO

- Loads the local full HF directory using `GroundingDinoForObjectDetection` and
  `AutoProcessor`, under **Transformers 5.22.2**, the same pinned runtime as
  Model 1. No packages were downgraded and no worker process was necessary.
- Uses Transformers' checkpoint-format migration (including changed Swin key
  names/tied decoder heads); its loading report has zero missing, unexpected,
  mismatched keys or load errors. This is a strict HF load check, not a claim
  that raw checkpoint names are identical to the v5 in-memory state dict.
- Image processor remains at the saved **512x512** size. It receives model-native
  RGB, with the teammate's image mean/std/rescale behavior retained. Native Torch
  attention kernels are selected by `disable_custom_kernels=True` for portability.
- The public method is `ground(image_path, text_prompt, box_threshold=0.3)` and
  returns only `{boxes, labels, scores}`. Text is stripped, lowercased and made
  period-terminated. Fixed `text_threshold=0.25` is recorded in result parameters.
- Transformers 5.22.2 calls the postprocessing cutoff `threshold`, replacing the
  teammate's older `box_threshold` keyword. Phrase strings come from `text_labels`,
  avoiding the deprecated labels alias. Detection rows map to absolute original
  pixel XYXY coordinates. Boxes are clipped to the image bounds; degenerate
  boxes are discarded; there are no masks. Scores retain full float precision.
- CPU/MPS/CUDA device selection uses CUDA → MPS → CPU. Any MPS placement or
  inference failure triggers one CPU retry and records the source/target/reason
  in `metadata.parameters.device_fallback` and result warnings. Real smoke
  inference succeeded on MPS without a fallback; fallback behavior is separately
  tested with a simulated unsupported-MPS exception.
- Grounding logits intentionally contain negative infinity for masked/padded
  text tokens. Validation permits that mask value while rejecting NaN,
  positive-infinite logits and non-finite box coordinates.
- Standard result evidence contains boxes/labels/per-box scores with calibration
  `uncalibrated` plus the saved PNG overlay path. The central API serves that PNG
  through `evidence_overlay_urls`; it reuses the adapter's overlay. Standalone
  `run` also writes an overlay using the standard prepared-artifact directory
  (or the caller's `artifact_dir`). Models unload and caches are cleared after use.
- There is no supported aggregate answer confidence, so `result.confidence` is
  null with reason `not available`. Per-box scores remain in evidence and are
  displayed explicitly as **Per-box scores · Uncalibrated** in the UI.
- Registry `validation_status: unvalidated` and real result metadata have an
  explicit **Unvalidated · no accuracy metrics** UI tag. No accuracy metrics
  are invented. An image that produces boxes is not evidence of accuracy.

## Model 4: strict raw optical/SAR fusion

`FusionModel.fuse_predict(optical_path, sar_path)` returns
`{label, confidence, top3: [(label, probability), ...]}`. TorchScript is loaded
lazily with `torch.jit.load(map_location="cpu")`, evaluated on CPU, then unloaded.
No MPS attempt is made. The existing Torch 2.15.0 runtime loads the supplied
artifact exported under the teammate's reported CPU runtime.

The dedicated `preprocessing/fusion.py` path validates **original** files before
loading the model. It never calls the generic pair aligner/resampler:

- Both extensions and GDAL driver must be TIFF/GeoTIFF; PNG/JPG or renamed PNGs
  are rejected with a specific explanation.
- Exactly 13 optical bands in B1,B2,B3,B4,B5,B6,B7,B8,B8A,B9,B10,B11,B12 order;
  exactly two SAR bands VV,VH. Wrong named band order is rejected, not reordered.
  If descriptions are absent, file order is assumed as documented and the
  unverifiable assumption is recorded. Band counts alone cannot prove semantics.
- Both images must be exactly 256x256 with non-null CRS and identical CRS,
  transform, width and height. No overlap cropping, reprojection or resizing.
- Raw arrays are cast to float32. Non-finite and nodata/masked pixels are rejected;
  there is no filling. Raw optical digital numbers are not divided by 10000 and
  are never RGB-stretched for inference. SAR is treated as already-dB without
  log conversion or clipping; actual physical units cannot be inferred reliably
  from numeric ranges alone, so users must supply the documented units.
- Exact per-band normalization comes from model4_normalization_stats.json:
  `(raw - training_mean) / training_std`, producing contiguous float32
  `[1,13,256,256]` and `[1,2,256,256]` arrays. Input/raw arrays and display PNGs
  remain separate; tests compare every normalized value with the saved statistics.
- Optical **display only** RGB is B4,B3,B2 (one-based indices 4,3,2 in this
  B1-first 13-band file), with percentile stretching. SAR display uses dB input
  explicitly; display clipping never changes the inference arrays.
- Finite `[1,8]` logits are softmaxed and ranked against metadata's eight classes:
  Forest, Shrubland, Savanna, Grassland, Wetlands, Croplands, Urban_Builtup, Water.
  Evidence is top-3 classes only. Global class confidence is **uncalibrated**.

Result metadata/warnings explicitly state: SEN12MS **summer only**, **8 classes**,
**256x256** patches; teammate-reported test accuracy about **76%** and macro-F1
about **63%**, not independently reproduced; weak classes **Grassland, Forest,
Wetlands**; other regions/seasons may perform worse. Registry status is
`reported_only`, distinguishing reported evaluation from our runtime wiring.

For unsupported real-fusion inputs, the job reaches a terminal `failed` state
under the existing queued/running/done/failed API contract, but provides
`input_explanation` and trace validation code `unsupported_fusion_input`.
The UI renders a calm **Input requirements** explanation instead of a generic
server-error message. It retains the routing reasons and writes a trace. No
model is instantiated for those invalid inputs, and there is no bad prediction.
In stub mode, the existing permissive demonstration flow is retained.

## Verification and timings

`.venv/bin/python -m pytest -q -s`: **91 passed, 2 skipped** in **19.18 s**.
Frontend `npm test`: **3 passed**; `npm run build` succeeded. `git diff --check`
passed. Existing Model 1/3 real API tests also passed under the same runtime.

| Real API wiring check | Final warm elapsed | Device | Outcome |
| --- | ---: | --- | --- |
| Model 2 | 0.998 s | MPS | done, no error |
| Model 4 | 0.232 s | CPU | done, no error |

Earlier isolated checks measured 1.560 s and 0.223 s respectively. Timings
include upload/job/load/poll overhead and are environment/input-dependent;
they are not performance guarantees. These API tests used **generated inputs**:
a constant-color optical PNG and a synthetic aligned 13+2-band GeoTIFF pair.
They prove format/native-tensor/endpoint wiring, not satellite model accuracy.

A separate UI check used a generated diagram with rectangles and circles. Model 2
produced eight candidate boxes and showed unvalidated/per-box-score/overlay UI
correctly. That diagram is also **not a genuine satellite image** and is not an
accuracy or domain-validation sample.

The 30-case routing evaluation was repeated with all four entries configured
**real**, and effective registry resolutions were real for all four. Result:
**30/30 = 100% intent routing agreement**. This is keyword/image-combination
routing evaluation, not end-to-end answer accuracy. Strict fusion-format
rejection is tested separately.

The two skipped tests are explicit:

1. Genuine-image grounding geometry check: no image exists in
   `data_samples/grounding/`. It will check absolute coordinate bounds when an
   image is provided, while retaining the unvalidated label (no ground truth).
2. Genuine fusion expected-prediction parity: no aligned pair plus teammate
   expectation exists in `data_samples/fusion/`. It will compare the class to
   the supplied expected.txt; it cannot reproduce accuracy/F1 from one pair.

See the README files in those directories for exact layouts. No genuine-image
box/accuracy result or CPU parity against the original teammate implementation
is claimed. Model 4 remains CPU-only until appropriate parity work is supplied.

No additional ML packages, downgraded Model 1 runtime, standalone teammate
backend, or separate worker process was introduced. Known upstream deprecation
warnings remain nonfatal. The earlier inventory is a historical audit; this file
records the subsequent Model 2/4 integration.
