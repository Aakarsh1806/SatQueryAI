# Real Model 1 / Model 3 integration verification

> Historical run record: preserves the configuration and measurements from that stage. For current modes, demo outcomes and caveats, see [final_review.md](final_review.md). Model 4 remains a stub, not yet verified on a real pair.

11 November 2026 (Asia/Kolkata). Python 3.11, Apple Silicon, 16 GiB unified
memory. All four checked-in registry entries remain **stub**. Real mode was
selected only in isolated test registries. The frontend and Model 2/4 stub
implementations are unchanged.

## Installed, tested runtime

Optional dependencies are separate in `backend/requirements-ml.txt`:

| Package | Version |
| --- | --- |
| torch | 2.15.0 |
| torchvision | 0.29.2 |
| transformers | 5.22.2 |
| peft | 0.21.2 |
| accelerate | 1.15.0 |
| qwen-vl-utils | 0.0.14 |
| einops | 0.8.2 |
| imageio | 2.38.1 |

`pip check` reports no broken requirements. No bitsandbytes is installed or used.
Resolved dependencies also include huggingface-hub 2.4.0, tokenizers 0.23.3 and
safetensors 0.8.0. Model 2 remains a stub; its handover's older Transformers pin
was not installed over this runtime.

For the requested real API check, the base Qwen/Qwen2.5-VL-3B-Instruct weights
were downloaded into the normal HF cache at snapshot
`66285546d2b821cf421d4f5eb2576359d3770cd3`. This records the base **used for this
verification**, not the unknown original LoRA training revision. The real loader
logs the initial-download requirement. Tests never download base weights
automatically; they verify cached shard completeness and skip when missing.

References: [Qwen2.5-VL](https://huggingface.co/docs/transformers/model_doc/qwen2_5_vl),
[PEFT loading](https://huggingface.co/docs/peft/package_reference/peft_model),
[pinned Chg2Cap](https://github.com/ShizhenChang/Chg2Cap/tree/7b8cda937002e614d51a6dab3d949aa7de77c176).

## Strict loading

- **Model 1:** local processor and chat template loaded successfully; all **696**
  LoRA tensor keys/shapes match the pinned architecture. A full materialized
  base+adapter API run also verified all loaded values after receiving-dtype
  casting. MPS used bfloat16. Exact QA/caption prompts and 100352/313600 pixel
  budgets are retained. Decoding returns new tokens only; confidence is the mean
  generated-token probability, explicitly uncalibrated and including EOS when
  generated. Evidence is null.
- **Model 3:** strict loads accepted **624** encoder keys, **80** attentive-encoder
  keys and **52** decoder keys, with no missing/unexpected/shape mismatches. The
  embedded **436** explicit token ids and max length **41** are used. Initial
  standalone loading took 3.34 s; final warm loading took 1.399 s. Upstream
  architecture and attention behavior are preserved, including its use of to_k
  for value projection; only pretrained initialization/device allocations and
  provenance/line endings were changed. No ImageNet download occurs. Vendor
  licence, academic-use notice and original source hashes are included.

## Native inputs versus display

`prepare_for(model_id, paths, output_dir=..., config=...)` returns a
`PreparedInputs` object with NumPy `tensors`, display `previews`, independent
`native_paths`, `notes` and `warnings`. Torch conversion is inside real adapters.
Models 1/2 receive RGB HWC in 0–255 units. Model 3 receives normalized float32
`[1,3,256,256]` arrays, ordered before A, after B. No display percentile stretch
is applied to native inference data; high-DN optical inputs require an explicit
prior RGB conversion rather than an implicit display transformation.

Chg2Cap preprocessing remains **UNCONFIRMED**. Default YAML `normalization:
upstream` uses 0–255 float32 RGB, mean [100.6790,99.5023,84.9932], std
[50.9820,48.4838,44.7057], no /255, aligned/resized 256×256. The explicit
`imagenet` switch uses RGB/255 and ImageNet mean/std. Notes and warnings record
this uncertainty. No generated change map or model confidence is claimed.
Model 4's all-band raw array path is reserved for the next task; its existing
stub flow remains available, without implementing trained normalization here.

## Real POST /api/analyze checks

These are **generated constant-color RGB input wiring checks**, not satellite
quality evaluation. They used temporary uploads/SQLite storage and real model
weights. POST returned 202, polling reached done, output and JSON traces reported
real mode, and native inputs, previews and strict-load metadata were validated.
The QA query was “What is visible?”; the change query was “What changed?” with
two identical generated images.

| Model | Final full-suite API elapsed | Status | Error | Device |
| --- | ---: | --- | --- | --- |
| 1 RS-VLM | 7.873 s | done | none | mps |
| 3 Chg2Cap | 1.755 s | done | none | mps |

An earlier isolated Model 1 run took 13.313 s; the final warm run took 7.873 s.
Model 3 initially took 5.348 s, then 1.755 s in the final warm run. API times
include load/job/poll overhead, excluding the preceding network download.
These are small-input, environment-specific engineering measurements, not
latency guarantees or model-quality results.

Artificial-input outputs: Model 1 returned `text`, with probability
0.3976835757493973 (**uncalibrated**); Model 3 returned
`the two scenes seem identical`, with null confidence. Neither establishes
usefulness on genuine imagery or validates the default normalization.

| Model | Process lifetime peak RSS | Current MPS allocation at inference end |
| --- | ---: | ---: |
| 1 | 3,562,831,872 bytes | 7,701,790,208 bytes |
| 3 | 3,562,831,872 bytes | 1,319,407,872 bytes |

RSS is the whole test process's lifetime high-water mark, not isolated per-model
peak memory. Current MPS allocation is **not a measured MPS peak**; do not sum
these counters and call the result total peak unified memory. The adapters
unload after use and clear device caches. Larger visual budgets and 128-token
captions have not been memory-benchmarked here.

## Tests and pending sample checks

Final `.venv/bin/python -m pytest -q -s`: **70 passed**, zero failures/skips in
15.51 s in this populated environment. Optional ML/weight tests skip on fresh
setups without the dependencies/artifacts/base cache. Stub-only startup was
separately checked not to import Torch, Transformers or PEFT. All four YAML
entries remain stub. Missing packages/artifacts retain fallback; strict-load
mismatches fail explicitly instead of being hidden by fallback.

Warnings include rasterio/Affine deprecations, Chg2Cap's mixed attention-mask
types and Transformers 5.22.2's future min_pixels/max_pixels API change. They did
not fail tests. Upgrading these pinned packages needs new compatibility checks.

Per user instruction, the manual sample scripts were **not run**: there are no
genuine optical or LEVIR-CC images/reference captions yet. Both scripts print
`no samples found`, exit 0 and avoid model loading when discovery is empty.
See `data_samples/README.md` for the supported layout and tab-separated manifest.
Model 1 has 20 short questions and prints answers/timing/memory. Model 3 prints
predicted/reference captions, possible repetition/weak-overlap warnings and the
explicit `--normalization imagenet` comparison suggestion. These diagnostics
are not accuracy metrics. No choice is silently marked confirmed.

Still needed: genuine-image answer/caption review, normalization confirmation,
and larger-image/caption performance testing. No UI or Model 2/4 integration was
performed. This document records the integration after the earlier audit in
`model_inventory.md`; that audit's historical readiness statements are not a
claim about the current implemented adapters.
