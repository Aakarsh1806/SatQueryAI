# Vendored Chg2Cap

Source: https://github.com/ShizhenChang/Chg2Cap
Commit: `7b8cda937002e614d51a6dab3d949aa7de77c176`.
See `LICENSE.txt` (MIT) and the upstream academic-use-only notice; use this
project for academic purposes only. Original byte hashes are in UPSTREAM_SHA256.json.

Local changes to model_encoder.py/model_decoder.py:
- All torchvision constructors use weights=None, preventing ImageNet downloads.
- CUDA-only allocations use the input tensor device instead.
- Added provenance comments; normalized line endings.

No network architecture, dimensions, parameter names, or decoding algorithm changed.
In particular upstream MultiHeadAtt.forward uses to_k for its value projection;
this is intentionally preserved for checkpoint behavior, not silently corrected.
reference/ contains the unchanged upstream LEVIR dataset and preprocessing code
for provenance only; the application does not import this dataset pipeline.
