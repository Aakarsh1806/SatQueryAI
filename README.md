# SatQuery AI

SatQuery AI is a college project for **SIH26167**: a vision-language assistant
for satellite imagery. Upload imagery and enter a question or description
request. A rules-first Agent validates the inputs, selects **one specialist per
query**, prepares its inputs, and returns its supported output with an execution
trace. Evidence and confidence depend on the specialist; either can be **not
available**. This is an academic demonstration, not a claim of expert-level analysis.

The backend uses Python/FastAPI, SQLAlchemy and file storage; the frontend uses
React/Vite. Rasterio, NumPy and Pillow handle imagery. PyTorch and the model
libraries are optional for real inference.

## Current status

All four adapters exist. **Models 1–3 in `backend/models.yaml` now use
`mode: real`; Model 4 is now also `mode: real`, enabled at user request before real-pair verification.**
Stub answers, boxes and scores are synthetic demonstrations and
are visibly labelled. Real-adapter evaluation scripts can run independently of
these saved modes; their execution does not enable real inference in the UI.

**Model 4 is implemented and now enabled in real mode at user request. No real
aligned sample pairs are available yet, so it is not verified on a real pair.**
Its reported training/test metrics are not proof of local real-pair inference.

| Input | Specialist | Supported output |
|---|---|---|
| One optical image + question or caption request | RS-VLM | Short answer or caption |
| One optical image + location/detection request | Grounding DINO | Boxes, labels and per-box scores; no masks |
| Before/after optical images of the same area | Chg2Cap | Change description only; no direct question answering or change map |
| Strictly aligned Sentinel-2 optical + Sentinel-1 SAR pair | Fusion | Land-cover class and top-3 scores; no free-form answer |
| SAR only | None | Clear rejection |

PNG/JPG inputs support optical demonstrations. Real fusion requires the original
GeoTIFF pair described in [model documentation](docs/models.md); an RGB preview
cannot replace a multispectral input.

## Five-line quick start: local application

Run from the repository root with Python **3.11 or 3.12** and Node.js/npm installed.
Replace `python3.11` with `python3.12` if appropriate. Run the final two commands
in **separate terminals**, both starting at the repository root. Open the URL
printed by Vite. These commands do not install ML packages or download weights. Without real
weights/ML prerequisites, the configured real adapters fall back to labelled stubs.

```sh
python3.11 -m venv .venv
.venv/bin/python -m pip install -r backend/requirements.txt
npm --prefix frontend install
.venv/bin/python -m uvicorn backend.app.api.main:app --host localhost
npm --prefix frontend run dev
```

Node requirement: **20.19+ or 22.12+** (matching Vite). The launcher checks it.

### Backend and frontend commands

Keep the backend in its own terminal:

```sh
.venv/bin/python -m uvicorn backend.app.api.main:app --host localhost
```

In another terminal, start the UI:

```sh
cd frontend
npm run dev
```

Vite proxies `/api` to the local backend. For a different backend, set
`SATQUERY_API_TARGET` before starting Vite, or use `VITE_API_BASE_URL` for an
explicit frontend API origin. The backend CORS allowlist is configured with
`SATQUERY_CORS_ORIGINS`. Use the backend's `/docs` page for the generated API
reference. Run a single backend worker on the demo machine.

## Weights and real mode

Keep model artifacts under the repository's ignored `weights/` directory. The
registry paths are relative to the repository root:

```text
weights/
├── satquery_rsvlm_lora/                # RS-VLM adapter + processor/tokenizer
├── SatQuery_Model2_Handover/           # Grounding HF model + processor files
├── chg2cap_3h/
│   └── run_01/
│       └── best.pth                   # Chg2Cap checkpoint, including vocabulary
└── SatQuery_Model4_Handover/
    └── model/
        ├── model4_fusion_torchscript.pt
        ├── model4_normalization_stats.json
        └── metadata.json
```

Do not use or launch the teammates' standalone FastAPI backends. The project
adapters reuse inference logic in-process through the project's API/Agent.

Install the optional pinned ML environment when preparing for real inference:

```sh
.venv/bin/python -m pip install -r backend/requirements-ml.txt
```

Edit only the intended entry in [backend/models.yaml](backend/models.yaml):

```yaml
# Fields within the existing rs_vlm entry; keep its other fields.
id: rs_vlm
mode: real         # Change back to stub for synthetic demonstration output.
enabled: true
weights_path: weights/satquery_rsvlm_lora
```

Restart the backend after changing the YAML. `enabled: false` disables the
specialist rather than selecting a stub. `/api/models` reports the configured
and effective mode, validation status and fallback reason. Real mode resolves
to a stub when required weights/files, a registered implementation or required
ML modules are absent. A load, strict-key mismatch or inference error is reported
as a failed request; it is **not silently converted into a successful stub answer**.
File existence alone does not establish compatibility, quality or validation.

Fusion real mode is enabled before sample verification at the user’s request.
Supply genuine aligned pairs and expected labels before claiming it is verified. The current small local checks for the other adapters
are described separately from trainer benchmark results in
[docs/models.md](docs/models.md).

### First real RS-VLM request: downloads, disk and RAM

The LoRA folder does not contain the base model. The first real request downloads
`Qwen/Qwen2.5-VL-3B-Instruct` from Hugging Face unless the necessary files are
already cached. Internet access is needed for missing files; subsequent offline
use requires a complete cache. The UI reports loading/download-related progress;
loading and strict adapter verification can take time. Evaluation scripts use
local cached base weights rather than downloading them implicitly.

The artifact inventory records these exact handover-folder sizes; they are
**not total runtime disk requirements**:

| Audited folder | Bytes recorded in the inventory |
|---|---:|
| `satquery_rsvlm_lora` | 308,755,700 |
| `SatQuery_Model2_Handover` | 690,318,189 |
| `chg2cap_3h` | 1,316,060,457 |
| `SatQuery_Model4_Handover` | 7,192,019 |

Model 1 base-model download/cache size: **to be added**. Total environment and
cache disk budget: **to be added**. Per-model peak RAM/unified-memory requirements
and recommended free RAM: **to be added**. Weight-file sizes do not measure RAM:
loading, model copies, tensors, image size and generation need additional memory.
Do not infer a guaranteed fit from the model name or adapter size.

### Mac / Apple Silicon troubleshooting

- The usual device order is **CUDA → MPS → CPU**. A MacBook Air has no CUDA;
  do not install bitsandbytes for this workflow. Use the project's ML requirements,
  not the legacy CUDA environment from an upstream training repository.
- The device helper sets `PYTORCH_ENABLE_MPS_FALLBACK=1` before importing Torch.
  If launching manually, it can also be set in the shell:
  `export PYTORCH_ENABLE_MPS_FALLBACK=1`. This supports eligible unsupported MPS
  operations; it is not a general out-of-memory recovery mechanism.
- Close other memory-heavy applications, avoid simultaneous evaluation processes,
  and use a single API worker. The queue and process-wide lock keep one specialist
  active at a time **within that process**. Multiple backend processes do not share
  that lock.
- Models unload after a request and release available device caches. This does
  not promise an immediate reduction in process RSS or recovery from every failure.
  If memory remains exhausted, stop/restart the backend after reducing the workload.
- To request CPU for RS-VLM, grounding or change, set `settings.device: cpu` on
  that registry entry and restart. Evaluation scripts also accept `--device cpu`.
  CPU performance and RAM requirements: **to be added**. Grounding retries on CPU
  when its MPS placement/inference fails and records the reason; the other adapters
  do not promise automatic retry for arbitrary MPS errors. Fusion is CPU-only.
- Cancellation is cooperative. An active download/load/inference operation can
  finish before cleanup and terminal `cancelled` status; the button does not
  force-kill a Torch kernel.

## Storage, testing and documentation

SQLite is the default database (`satquery.db`); `DATABASE_URL` configures the
SQLAlchemy connection. PostgreSQL requires its optional driver; the local switch check and setup are
documented below. Production deployment guidance remains **to be added**. Images, model weights, traces and reports
are files; the database stores paths, structured outputs and metadata, not image
bytes. Upload retention and storage paths are configurable. The report download
is self-contained **HTML**, which can be printed to PDF; it is not a native PDF export.

Datasets live outside Git (locally under `~/Datasets`) and prepared test inputs
are under ignored `data_samples/`. No training is performed here. Respect the
academic-use restrictions for VRSBench, LEVIR-CC and SEN12MS, and retain the
vendored Chg2Cap notices. Source images, annotations, weights and derived image
overlays must not be committed. Provenance is in [data_samples/README.md](data_samples/README.md).

```sh
.venv/bin/python -m pytest -q
.venv/bin/python scripts/eval_routing.py
.venv/bin/python scripts/eval_model1.py
.venv/bin/python scripts/eval_model2.py
.venv/bin/python scripts/eval_model3.py
.venv/bin/python scripts/eval_model4.py
.venv/bin/python scripts/domain_gap.py
```

Run real model evaluations sequentially. Optional caption metrics use
`backend/requirements-eval.txt`. Missing fusion samples print `no samples found`
and exit cleanly; the empty domain-gap test writes a not-run report. Tests needing
missing model weights, ML packages or sample inputs skip as appropriate.

- [Architecture and trace format](docs/architecture.md)
- [Models, training provenance and supplied metrics](docs/models.md)
- [Limitations and academic-use scope](docs/limitations.md)
- [Slides/spec correction checklist](docs/changes_needed_in_ppt.md)
- [Artifact inventory](docs/model_inventory.md) — historical static audit, not current runtime status
- [Local evaluation reports and metric definitions](docs/evaluation_summary.md) — small checks, not benchmarks

The current registry is the source of configured modes. Historical audit/check
reports describe the state when they were written; they do not automatically
change or validate today's configuration.

## Run on another machine: one command

From a local checkout:

```sh
./scripts/start_dev.sh
```

The launcher selects Python 3.11/3.12 (or `SATQUERY_PYTHON`), checks Node against
Vite's requirement (20.19+ or 22.12+), creates/checks `.venv`, installs lightweight
backend requirements, installs frontend packages with `npm ci` if absent, checks
ports, and supervises both servers. It prints the UI, API and API-docs URLs.
Ctrl+C stops both. It does not install ML weights or enable real modes.
`./scripts/start_dev.sh --check` checks Python/Node without starting servers.
If PostgreSQL is selected, the launcher also installs the optional driver.

**Windows:** use WSL2 with a Linux checkout/venv and run `bash scripts/start_dev.sh`.
For native PowerShell, create the venv with `py -3.11 -m venv .venv`, install using
`.venv\Scripts\python.exe -m pip install -r backend/requirements.txt`, then run
`.venv\Scripts\python.exe -m uvicorn backend.app.api.main:app` and
`npm --prefix frontend run dev` in separate terminals. The Bash supervisor is
not a native PowerShell script; do not reuse a Windows venv inside WSL.

### Environment configuration

Copy `.env.example` to `.env` locally; never commit the latter. Existing shell
variables take precedence. `DATABASE_URL` is the local backend URL; SQLite is
its default. PostgreSQL uses `postgresql+psycopg://...` with the optional
`backend/requirements-postgres.txt` driver. Replace example placeholders with
local, URL-encoded credentials in the ignored file, not in committed config.

`SATQUERY_RETENTION_DAYS` configures image retention (it takes precedence over the
legacy hours setting). Limits use bytes. `SATQUERY_DEVICE=auto|cpu|mps|cuda`
overrides Models 1–3; fusion stays CPU-only. `SATQUERY_MODELS_CONFIG` selects the
YAML file. Per-entry `SATQUERY_RS_VLM_WEIGHTS_PATH`,
`SATQUERY_GROUNDING_WEIGHTS_PATH`, `SATQUERY_CHANGE_WEIGHTS_PATH` and
`SATQUERY_FUSION_WEIGHTS_PATH` override paths. The corresponding `_MODE` variables
can override `stub`/`real` without editing YAML; prerequisites and fallback rules
still apply. Changing configuration requires restarting the backend. Model 4's
current default is real at user request, with genuine-pair verification still pending.

### Local Docker Compose / static production build

With Docker running:

```sh
docker compose up --build
```

Open **http://localhost:8080**. The frontend image builds Vite assets and serves
static files from nginx, proxying `/api` to the backend; it does not run a Vite dev
server. The backend is available at **http://localhost:8000**. Ports are bound to
loopback only. Stop the local stack with `docker compose down`; this retains data
volumes. Nothing is deployed to a remote machine by these commands.

**The image runs on CPU unless a compatible GPU runtime is provided.** The default
`INSTALL_ML=false` build supports stub and pre-computed playback without Torch.
For live adapters, set `INSTALL_ML=true` and rebuild. The ML-enabled image/CUDA
configuration is not verified by the lightweight image check; provide compatible
Linux ML wheels/GPU runtime and select `SATQUERY_DEVICE=cuda` or `auto` as
appropriate. MPS is not exposed through the Linux Docker container on a Mac.
The default container device is CPU; fusion always remains CPU. ML image size,
GPU build/runtime verification and real-model container RAM budgets: **to be added**.

Weights are **mounted read-only**, not copied into image layers. `WEIGHTS_DIR`
selects the host weights folder; `MODELS_CONFIG_FILE` selects the mounted YAML.
A Hugging Face cache volume supports first-run Model 1 downloads without baking
weights into the image. Persistent named volumes store SQLite/uploads/traces and
the HF cache. `.env`, secrets, datasets and local model artifacts are excluded from
the build context. Compose also mounts the ignored demo catalogue read-only.

Compose uses `DOCKER_DATABASE_URL`, separate from the local `DATABASE_URL`, to
avoid putting a relative host SQLite path inside the container. Its default is
SQLite in the persistent application-data volume. For PostgreSQL inside Compose,
use a PostgreSQL URL with hostname `postgres`; localhost would refer to the
backend container itself. PostgreSQL is an optional Compose profile, not required
for the default SQLite stack.

For a non-Docker static frontend, run `npm --prefix frontend ci` then
`npm --prefix frontend run build` and set `SATQUERY_STATIC_DIR=frontend/dist` before
starting FastAPI. Static mounting happens after all API routes and exposes only
the build directory. Do not use the repository root as the static directory.

### PostgreSQL service and switch check

The PostgreSQL service is defined in `docker-compose.yml` under the `postgres`
profile. Its password is read from an ignored secret file, not committed Compose
text. To start that service manually, first create a private local password file:

```sh
mkdir -p .secrets
(umask 077; openssl rand -hex 24 > .secrets/postgres_password)
docker compose --profile postgres up -d --wait postgres
```

Do not regenerate the password file for an existing initialized database volume:
changing the file alone does not change the stored database password. For local
backend access, set `DATABASE_URL` using that password, username/database
`satquery`, host `localhost`, and the configured PostgreSQL port. For a container
backend, use `DOCKER_DATABASE_URL` with host `postgres`. Put the URL in ignored
`.env` and restart, or pass it through the environment; do not print credentials
in screenshots or shared logs. Stop the PostgreSQL service when finished.

To run an isolated, repeatable check without manually putting passwords in a URL:

```sh
.venv/bin/python -m pip install -r backend/requirements-postgres.txt
.venv/bin/python scripts/check_postgres.py
```

This starts a separately named local Compose PostgreSQL service, checks a running
backend, redirects every DB-using pytest fixture into temporary PostgreSQL schemas,
records the real outcome in [docs/postgres_check.md](docs/postgres_check.md), and
stops its processes. It retains its ignored secret file and local database volume
for repeat checks. If Docker is unavailable, it records **not run**, never a fake
pass. `SATQUERY_POSTGRES_CHECK_PORT` changes the isolated check port if occupied.

### Optional, explicitly labelled pre-computed demo playback

Prepare a small catalogue in advance on a machine with real weights and samples:

```sh
.venv/bin/python scripts/prepare_demo.py
```

The script records genuine real-adapter outputs for a fixed sample/query list
from Models 1–3. It never silently substitutes stubs while recording and never
generates fake Model 4 samples. Missing fusion samples print `no samples found`.
The ignored `demo/manifest.json` lists the fixed queries, sample file provenance,
content hashes, source results and recording times. Copy that folder and required
input files privately to another machine if dataset permissions allow; keep all
payloads and overlays out of Git. Lookup depends on bytes, not old machine paths.

Set `DEMO_MODE=true` and optionally `DEMO_MANIFEST`, then restart the backend.
For Compose, use `DEMO_MODE=true docker compose up -d --wait`. A matching ordered
input/query/task returns the saved result with **pre-computed** in the API,
**PRE-COMPUTED · not live inference** in the UI/history, and explicit playback
provenance in the trace/report. No model is loaded for that playback request.
Saved confidence remains uncalibrated; saved runtime/device are source provenance,
not current execution. Unmatched requests follow normal stub/live routing.

This setting is chosen in advance when live models are unavailable or too slow;
it is not an unsafe runtime timeout that kills inference threads. Disable it for
an intended live demonstration. There is no default Model 4 demo entry. Any fusion
playback entry must be explicitly marked **placeholder**, and the UI/API/trace/report
must identify it as **PLACEHOLDER OUTPUT**, not a verified prediction. Switching
fusion to real later leaves normal strict live inference available; placeholder
playback never validates a real pair.

See the [local release checklist](docs/release_checklist.md), including the Model 4
TODO when aligned real pairs arrive. Local verification results and remaining
container ML/GPU limits are recorded separately from model benchmark claims.

[Local portability verification](docs/portability_check.md) records launcher,
container/static serving, demo labelling and SQLite/PostgreSQL test outcomes,
including what was not verified.

## Final demo profile

Run `./scripts/start_demo.sh` for a new local database/uploads/traces profile.
All four adapters request real inference; Model 4 remains unverified on a real pair. Verify the effective mode
and result tag before presenting. The [five-minute walkthrough](docs/demo_script.md)
names each sample/query and explains what to say when a result is slow or wrong.
[Final review](docs/final_review.md) contains the model summary table, actual
rehearsal outcomes, edge-state checks, limitations and open TODOs. The caption and
initial grounding miss are documented as observed, not silently corrected.

For explicitly labelled fallback playback of those same scenarios, package the
measured results with `.venv/bin/python scripts/prepare_demo.py --from-rehearsal docs/demo_rehearsal.json`
and launch with `DEMO_MODE=true ./scripts/start_demo.sh`. A pre-computed result is
not live inference or a correctness guarantee. No default Model 4 replay exists. Model 4 real mode requires its strict raw GeoTIFF pair.
