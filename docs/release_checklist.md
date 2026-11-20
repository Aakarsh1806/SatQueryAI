# Local release / demo checklist

> Current configuration update: the user enabled Model 4 real mode before sample pairs arrived. It is still not verified on a real pair. The earlier stub-only walkthrough/check results below are historical; do not describe them as a real-pair validation.

This prepares a local run on another machine. It does not deploy a public service.
Keep Model 4 in stub mode until genuine aligned pairs are available and checked.

## Machine and configuration

- [ ] Python 3.11/3.12 and supported Node/npm installed; run `scripts/start_dev.sh --check`.
- [ ] Copy `.env.example` to ignored `.env`; set local paths, database URL,
  allowed origins, upload limits, retention days and device. No real secrets in Git.
- [ ] Weights in the named `weights/` folders, or equivalent environment-overridden
  paths; preserve processor/tokenizer/config/statistics files, not only weight binaries.
- [ ] `backend/models.yaml` (or `SATQUERY_MODELS_CONFIG`) correct: enabled flags,
  model paths, modes, preprocessing and honest validation status. Inspect `/api/models`.
- [ ] Model 4 remains `stub`; no claim of verification on a real pair.
- [ ] Optional ML requirements installed for live real modes. Default container
  build supports stubs/playback; `INSTALL_ML=true` adds ML dependencies for live use.
- [ ] Sufficient disk for packages, weights, HF cache, uploads and reports. Capacity
  and RAM budgets not established in the allowed inventory remain **to be added**.
- [ ] Model 1 base-model download completed in advance on the actual target
  machine/cache; verify offline operation if the presentation has no internet.
- [ ] Run a single backend/inference worker. Laptop on power; close memory-heavy
  applications and avoid concurrent real evaluation processes.

## Checks and samples

- [ ] Models 1–3 check/evaluation scripts passed on genuine local samples; inspect
  predictions, grounding geometry and change captions, not only exit codes.
- [ ] Samples and provenance present in `data_samples/`, with academic-use notices.
  No datasets, weights, demo payloads or derived image overlays committed.
- [ ] Run `pytest`, frontend tests and production build. Optional weights/dependency/
  sample skips reviewed; an empty sample folder is not a model-validation pass.
- [ ] If PostgreSQL is intended, run `scripts/check_postgres.py` with Docker and
  review `docs/postgres_check.md`. Confirm the selected DB URL is for this machine.
- [ ] Local launcher starts both servers, prints URLs and stops both on Ctrl+C.
- [ ] Optional Compose images build; static frontend, `/api/health`, upload,
  background status flow, trace/report downloads and history work locally.
- [ ] Docker weights/config mounts read-only; database/uploads/cache persist in
  volumes. Image contains no weights or credentials. CPU is the default; a GPU
  requires a compatible ML build/runtime and an explicit appropriate device setting.

## Optional presentation playback

- [ ] Generate local fixed entries with `scripts/prepare_demo.py` before the demo.
  Copy the ignored `demo/` catalogue plus its overlay files privately if permitted.
- [ ] Use `DEMO_MODE=true` only when intending playback for matching samples.
  Input bytes, ordered modalities, task and exact stripped query must match.
- [ ] Confirm **PRE-COMPUTED · not live inference** in UI, API, trace and report.
  Source model/device/runtime are provenance, not the current request's execution.
- [ ] Confirm unmatched inputs do not receive another sample's cached answer.
  They follow the normal live/stub path. There is no unsafe thread-killing timeout.
- [ ] Any Model 4 playback entry is visibly **PLACEHOLDER OUTPUT**, never represented
  as a real-pair prediction. No default fusion entry is supplied while samples are absent.
- [ ] Toggle playback off and recheck normal stub/live execution before claiming a
  live demonstration. Missing cached results never manufacture a pre-computed answer.

## Model 4 TODO

- [ ] **When pairs arrive: place them in `data_samples/fusion/`, run
  `scripts/eval_model4.py`, switch to real mode, re-run routing and demo checks.**
- [ ] Supply `<id>_s2.tif` (13 Sentinel-2 raw-DN bands), `<id>_s1.tif` (2 VV/VH dB
  bands), both 256×256 with matching CRS/transform, plus `expected_labels.json`.
- [ ] Inspect predictions/parity and errors before enabling. Replace "not yet
  verified on a real pair" only with evidence from the genuine-pair check.

## Final walkthrough gate

- [ ] Start `scripts/start_demo.sh` and verify empty history on its fresh profile.
- [ ] Follow `docs/demo_script.md`; confirm live/stub/pre-computed labels, grounding
  unvalidated status and persistent Model 4 stub warning on workspace/history.
- [ ] Review the known caption hallucination and grounding miss in `docs/final_review.md`;
  do not rewrite outputs or frame the selected demonstration as a blind benchmark.

After genuine Model 4 verification, the general `start_dev.sh` follows the YAML.
The presentation launcher defaults fusion to stub; an explicit
`SATQUERY_FUSION_MODE=real ./scripts/start_demo.sh` can select real only after
those pair checks and validation-status updates. Do not use that override now.
