# Local portability verification

> Historical run record: preserves the configuration and measurements from that stage. For current modes, demo outcomes and caveats, see [final_review.md](final_review.md). Model 4 remains a stub, not yet verified on a real pair.

Nothing was deployed. These checks ran locally with the saved registry still
using stub modes, including Model 4. Docker was initially unavailable because the
daemon was stopped; after the user started Docker, actual container and PostgreSQL
checks completed. This report does not claim a real Model 4 pair exists.

| Check | Observed result |
|---|---|
| `scripts/start_dev.sh --check` | Python 3.11.15 and Node 26.7.0 accepted |
| Actual one-command start | Backend health and Vite frontend reachable; URLs printed |
| Launcher Ctrl+C/process-group interrupt | Both owned servers stopped; neither port remained occupied |
| Backend and frontend Docker builds | Passed for the lightweight, non-ML build |
| Compose startup | Backend healthy; nginx served the production build and proxied `/api` |
| Container SQLite request | Upload → queued/status polling → done; explicitly stub; trace accessible |
| Container demo request | Saved genuine Model 1 result replayed with `pre-computed` in API, trace and report; no model loaded |
| Browser UI | Fixed sample upload/inspect/query/poll rendered **PRE-COMPUTED · not live inference** and playback device; screenshot saved locally |
| Image artifact check | Unmounted image weights directory empty; no `.env` in image |
| Full SQLite pytest run | **115 passed, 1 skipped, 0 failed**, 34.85 s |
| Full PostgreSQL pytest run | **115 passed, 1 skipped, 0 failed**, 36.70 s; see [postgres_check.md](postgres_check.md) |
| Later targeted portability tests | **7 passed**, including independent registry `.env` loading with shell precedence |
| Frontend tests | **3 passed** |
| Production frontend build | Passed |
| `pip check` | No broken requirements |

The full-suite skip is the missing genuine aligned Model 4 parity pair/expected
label. The later standalone-registry test was added after the full runs; its
passing targeted result is reported separately rather than inflating those totals.
All tests requiring real model weights/dependencies/samples keep their existing
conditional skip behavior.

The first container startup failed because Rasterio required `libexpat.so.1`.
The backend Dockerfile now installs the required runtime library and `libgomp1`;
rebuilt containers passed startup and API/static checks. Build context exclusions
keep weights, datasets, private environment files, secrets and demo payloads out
of image layers. Read-only volumes supply weights/config/demo artifacts instead.

Three actual local real-adapter outputs were recorded for Models 1–3 by
`scripts/prepare_demo.py`; their inputs, query, hashes, timestamp, result and source
runtime/device are preserved in ignored `demo/manifest.json`. Matching playback
works without Torch in the lightweight container. Grounding overlays accompany
the catalogue. No fake Model 4 files or predictions were generated. Unit tests
verify a marked fusion placeholder is labelled throughout, and reject an
unmarked fusion playback entry. Unknown hashes/queries/modalities do not match.

The UI screenshot `docs/screenshots/precomputed-playback.png` contains academic-use
sample pixels and is Git-ignored. Demo results, dataset payloads and generated
secrets are also ignored. Do not package them for redistribution without the
appropriate permissions. No credentials are included in this report or `.env.example`.

Temporary application containers, the isolated PostgreSQL service and launcher
servers were stopped after verification. Named volumes/private secret files were
retained for repeat local checks; the user's Docker daemon was left running.

## Not verified

- The optional ML-enabled Linux container image and GPU runtime. Their build size,
  real-model latency and RAM requirements are **to be added**. The verified
  lightweight image serves stubs and labelled playback; it does not contain Torch.
- Native Windows PowerShell supervision; the automated launcher targets Bash on
  macOS/Linux/WSL. Native commands are documented separately.
- A real Model 4 optical/SAR pair. It remains stub until pairs and expected labels
  are supplied and the release checklist's explicit TODO is completed.
- Remote/public production deployment, which was not requested and was not performed.
