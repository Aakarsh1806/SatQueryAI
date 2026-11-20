# Optional pre-computed playback catalogue

Run `scripts/prepare_demo.py` with real weights, the ML environment and prepared
samples to record a fixed, hash-matched set. Generated manifest/results/overlay
files are local, ignored academic-use artifacts, never baked into the image.
Set DEMO_MODE=true and DEMO_MANIFEST=demo/manifest.json to play matching requests.
UI, API, reports and traces say **pre-computed**, never live inference. Unmatched
inputs/queries use the normal model/stub path. No runtime timeouts or killed Torch
threads are used: opt into playback in advance when a model is unavailable/slow.

Model 4 has no real pairs and no default demo entry. The loader permits fusion
playback only with an explicit placeholder flag, visibly labelled PLACEHOLDER;
it never presents such output as a real-pair prediction. Real fusion mode later
works normally with demo mode off, or with no matching placeholder entry.
