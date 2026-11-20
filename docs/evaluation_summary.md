# Local evaluation results — 11 November 2026

> Historical run record: preserves the configuration and measurements from that stage. For current modes, demo outcomes and caveats, see [final_review.md](final_review.md). Model 4 remains a stub, not yet verified on a real pair.

These are real predictions on the prepared local datasets, with no training or
weight edits. They are deliberately small sanity checks, not full benchmarks.
The samples were selected deterministically from source annotation order; the
QA/grounding sets emphasize vehicles and harbors, and four selected change pairs
involve residential construction. They are neither random nor representative
estimates of performance on all imagery.

| Evaluation | Inputs | Result | Total request time |
|---|---:|---|---:|
| Routing + input contract | 53 explicit cases | 53/53, 100% | No model inference |
| Model 1 real VRSBench validation QA | 20 images/questions | Exact 8/20 (40%); lenient 8/20 (40%) | 167.727 s |
| Model 2 real VRSBench validation grounding | 10 images/phrases | Mean IoU 0.367260; Acc@0.5 4/10 (40%) | 13.345 s |
| Model 3 real LEVIR-CC test change captions | 5 pairs / 25 references | BLEU-4 0.270183; ROUGE-L 0.537653; CIDEr 0.684510 | 8.971 s |
| Model 4 | No samples | Not run; prints `no samples found`, exit 0 | — |
| Domain-gap test | No samples | Not run; prints `no samples found`, exit 0 | — |

All 35 real model-evaluation requests completed without inference errors on MPS.
Each request strictly loads the model and unloads it afterwards; times include
loading, preprocessing, inference and cleanup (plus overlay rendering for Model 2).
The three models were evaluated sequentially. These are single local runs, not
latency benchmarks. The optional caption metrics use pycocoevalcap 1.2 and only
its Python BLEU/ROUGE/CIDEr scorers; no Java or METEOR/SPICE processes were invoked.
The package's wheel includes unused Java resources; it is optional, not part of
the application's lightweight dependency set.

## Reports and definitions

- [Routing: all expected/actual outcomes and reasons](routing_eval.md).
  Initial accuracy was 51/53: `show me a summary` and its uppercase equivalent
  incorrectly selected grounding. The caption rule now recognizes `summary`
  before the grounding `show` keyword; final accuracy is 53/53. The `where does
  the road go` case deliberately routes to grounding under the specified rules;
  it supports boxes, not a guaranteed road trajectory. Fusion format fixtures
  are temporary generated inputs for validation only, not genuine Model 4 samples.
- [Model 1: every question, reference, prediction, match and timing](model1_eval.md).
  Exact match normalizes Unicode, case and whitespace. Lenient match also removes
  punctuation/articles, maps zero–ten to digits and accepts a contiguous whole-token
  reference phrase. It can over-credit verbose answers; it is not the full VRSBench
  scoring protocol. All 20 captured prompts contain exactly one required QA suffix,
  added by the adapter.
- [Model 2: per-prompt IoU, detections, timings and all overlay links](model2_eval.md).
  The trainer supplied no benchmark metrics. Best IoU chooses the best-overlapping
  predicted box, not the highest-score box. No detections yield zero. Extra false
  positives are not penalized, so these numbers are not mAP or precision/recall.
  Scores are uncalibrated. Overlays retain the original pixel coordinates and add
  a legend below the image: green is reference, red is prediction. Their source
  pixels are academic-use data; generated PNGs are Git-ignored.
- [Model 3: predictions versus all 25 references and subset metrics](model3_eval.md).
  The 25 manifest rows are grouped into five pairs so inference occurs once per pair.
  References and predictions use lowercase regex tokenization, not Java PTB. CIDEr
  document frequencies come from this five-pair reference subset only. The user-reported
  full-test scores are BLEU-4 **0.542**, ROUGE-L **0.727**, CIDEr **1.29**; METEOR
  was not computed. Those full-test scores were not reproduced, and the local
  subset scores are not directly comparable. Predictions are not identical across
  all pairs; they nevertheless omit details or disagree with references in some
  cases. The configured upstream normalization remains **unconfirmed**; this tiny
  subset does not establish the original training preprocessing. No normalization
  selection or tuning was performed.
- [Model 4: not run](model4_eval.md) and [domain gap: not run](domain_gap.md).

Each model report has an adjacent JSON file containing all outputs, errors (none
for Models 1–3), parameters, devices, timings and package versions. Samples are
local test inputs only; datasets and derived image overlays must not be committed.

## API and test results

The full run `python -m pytest -s -q` completed with **109 passed, 1 skipped,
0 failed** in **32.63 s**. The single skip is the missing genuine aligned Model 4
pair/expected label. After refreshing readiness messages and the Model 4 parity
manifest format, the affected API/Agent/evaluation/grounding/fusion suites passed
again: **71 passed, 1 skipped, 0 failed**, **20.12 s**.

Real-sample POST → poll → done API timings from the full test run:

| Model | API wall time | Status |
|---|---:|---|
| RS-VLM | 12.959 s | done |
| Grounding | 1.318 s | done |
| Chg2Cap | 1.836 s | done |

API tests use httpx ASGITransport and the actual application lifespan/queue with
isolated SQLite databases and upload directories. They cover all application
endpoints: health, models, inspect, analyze, request status/history, trace, static
preview/evidence PNGs, reports, cancellation. Existing and new cases cover queued/
running/done/failed/cancelled states, restart recovery, persisted cancellation,
missing IDs, malformed images, extension/signature mismatches, oversized files,
chunked total-body limits, invalid modalities, invalid image counts, empty queries,
CORS and retention. Real-sample tests automatically skip if their sample files,
weights, required ML modules or cached Qwen base shards are missing; tests never
download base weights. Optional caption-metric tests skip without the metric package.
Synthetic unit/smoke checks remain clearly separate from genuine-sample validation.

## Registry status

No model modes were enabled by the evaluation scripts. Model 4 remains **stub**.
Models 1–3 retain their existing configured stub modes, while scripts/tests explicitly
instantiate the real adapters. Their stale "no samples supplied" readiness reasons
were replaced with links to completed local sanity checks. Model 1's displayed
local sample score is distinguished from missing full-benchmark figures; Model 2
still has no trainer-provided benchmark metrics. The old grounding warning claiming
that no genuine satellite-image check existed was corrected. Earlier verification
reports describe historical no-samples runs; this report supersedes that availability
status without rewriting historical measurements.

## Run commands

From the repository root:

```sh
# Optional; only needed for BLEU/ROUGE/CIDEr, not inference:
.venv/bin/python -m pip install -r backend/requirements-eval.txt

.venv/bin/python scripts/eval_routing.py
.venv/bin/python scripts/eval_model1.py
.venv/bin/python scripts/eval_model2.py
.venv/bin/python scripts/eval_model3.py
.venv/bin/python scripts/eval_model4.py
.venv/bin/python scripts/domain_gap.py
.venv/bin/python -m pytest -q
```

Run real model evaluations one at a time on the MacBook Air. Each script accepts
`--samples`, `--output` and (where applicable) `--device`; fusion always uses CPU.
Missing sample folders/manifests produce a `not run` report and `no samples found`.
Model 4 expects `<id>_s2.tif`, `<id>_s1.tif` and an `expected_labels.json` object
mapping each pair ID to its expected class, e.g. `{"pair01": "Forest"}`. Incomplete
pairs or absent expected-label manifests do not load Model 4. Existing report files
are replaced by the latest run; keep a copy or use `--output` for comparisons.
