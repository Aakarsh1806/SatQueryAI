# SatQuery AI — model inventory

Audit date: **11 November 2026 (Asia/Kolkata)**. Repository: `/Users/sriaakarsh/Documents/CODES/SatQuery-AI`.

## Scope, method, and limits

Read `AGENTS.md` and all supplied README, documentation/report, Python inference, requirements, JSON, and template files under the four requested folders. Large tokenizer JSON files were parsed fully and summarized structurally; Model 2's entire text vocabulary was counted. No model code was executed. No Torch, Transformers, PEFT, bitsandbytes, or other heavy ML package was installed or imported. No inference, architecture instantiation, strict state-dict load, or performance benchmark was performed. Only this inventory is added to the repository; application/configuration code and model artifacts remain unchanged.

Evidence labels used below:

- **Local fact:** visible file bytes, parsed configuration, code, safetensors header, or archive metadata.
- **Handover claim:** stated by a teammate's documentation; not independently reproduced here.
- **User-provided training fact:** supplied in this task or `AGENTS.md`, distinguished from packaged evidence.
- **Upstream reference:** publicly available Chg2Cap source at a pinned commit, not proof that the local training used an identical revision.
- **Proposed:** future work only; no integration has started.

Binary inspection used Python's standard-library `struct`, `json`, `zipfile`, and `pickletools`. Safetensors headers/offsets were checked against physical payload lengths. For Chg2Cap, pickle opcodes were symbolically interpreted into plain dictionaries and inert tensor/storage descriptors: no `pickle.load`, `torch.load`, pickle `GLOBAL` imports, executable `REDUCE`, or tensor data allocation. TorchScript source/debug metadata was read as archive text. CRC checks passed for every member of both PyTorch ZIP-format archives. These checks detect structural truncation, not correctness of learned values or runtime compatibility.

No `.zip` distribution file exists anywhere in the four folders. `best.pth` and the TorchScript `.pt` are ZIP-based **model serialization formats**, not handovers awaiting unpacking; leave them intact. No symlinks were found. `weights/.DS_Store` is an extra 10,244-byte Finder file outside the requested folders; it has no inference role.

| Model | Files | Exact total file bytes | Initial readiness |
| --- | ---: | ---: | --- |
| 1 RS-VLM | 7 | 308,755,700 | Substantial adapter + processor/tokenizer; base model/runtime not included in this folder |
| 2 Grounding | 14 | 690,318,189 | HF-style full model + processor/wrapper; real-image validation pending |
| 3 Change | 1 | 1,316,060,457 | State dictionaries + embedded vocab; standalone source/config/vocab file missing |
| 4 Fusion | 13 | 7,192,019 | Compact CPU TorchScript + stats/docs; API paths require attention |

All four actual weight files exceed 1,000,000 bytes. None is suspiciously tiny by the requested heuristic. Small JSON, requirements, README, source, and Finder files are expected; size alone does not make them defective. No trusted external checksum manifest was supplied. SHA-256 fingerprints recorded below identify the artifacts audited, rather than authenticate their origin.

## Model 1 — RS-VLM / Qwen2.5-VL LoRA

### Full file tree and completeness
```text
satquery_rsvlm_lora/
├── adapter_config.json  [1,183 bytes]
├── adapter_model.safetensors  [297,324,320 bytes]
├── chat_template.jinja  [1,017 bytes]
├── processor_config.json  [1,294 bytes]
├── README.md  [5,208 bytes]
├── tokenizer.json  [11,421,892 bytes]
└── tokenizer_config.json  [786 bytes]
```
- Adapter SHA-256: `78ac75a528e8284b4ab02065fa9309fc1c038fba853fe7a89b6ac0fcb2fb3740`.
- `adapter_model.safetensors` is 297,324,320 bytes, not a tiny placeholder. Header: 102,168 bytes; tensor payload: 297,222,144 bytes. Offsets form a contiguous payload ending exactly at EOF.
- Header contains 696 float32 tensors, 74,305,536 stored scalar elements. There are 504 language-side tensors and 192 visual-side tensors: the adapter affects **both** language and vision MLP modules, not language only. Language layer indices 0–35 and visual block indices 0–31 occur. The 348 A/B module pairs are consistent with language targets across 36 layers and visual MLP targets across 32 blocks; strict loading has not been tested.
- No base model weights, base `config.json`, training/inference Python, environment lockfile, evaluation report, or genuine sample is present in this folder. A base `config.json` is not required to make an adapter-only export legitimate, but this is **not** a standalone offline model package.
- No separate `preprocessor_config.json`, `special_tokens_map.json`, `vocab.json`, or `merges.txt` is present. Do not automatically declare the processor/tokenizer broken: the nested image processor settings and the tokenizer vocabulary/merges are included in `processor_config.json` and `tokenizer.json`. Whether the eventual pinned Transformers release loads this exact serialization must be tested.

### Documents/configuration read

| File | Observed contents |
| --- | --- |
| `README.md` | Generated PEFT model-card template, identifying Qwen/Qwen2.5-VL-3B-Instruct and PEFT 0.21.2. Most training, usage, license, authorship, hardware, and evaluation fields say “More Information Needed”; no usable inference instructions or scores. |
| `adapter_config.json` | Base `Qwen/Qwen2.5-VL-3B-Instruct`; `peft_type=LORA`, `task_type=CAUSAL_LM`, `r=32`, `lora_alpha=64`, dropout 0.05, bias `none`, inference mode true. Targets: `k_proj`, `q_proj`, `o_proj`, `gate_proj`, `v_proj`, `up_proj`, `down_proj`. `revision=null`, `modules_to_save=null`, empty rank/alpha patterns, DoRA/RS-LoRA/QALoRA disabled. Contains additional version-specific optional fields, mostly null. |
| `processor_config.json` | `Qwen2_5_VLProcessor`; nested `Qwen2VLImageProcessor` plus a video processor. RGB conversion, resize, rescale, normalize enabled. Image `size.shortest_edge=100352`, `longest_edge=313600`; patch 14, merge 2, temporal patch 2, resample 3. Video options exist but are outside this one-image project contract. |
| `tokenizer_config.json` | `Qwen2Tokenizer`, tokenizer backend, max sequence length 131072; `min_pixels=100352`, `max_pixels=313600`; pad `<|endoftext|>`, EOS `<|im_end|>`, no BOS/UNK. Lists multimodal/chat special tokens; `is_local=false`, `local_files_only=false`. These flags do not establish that the complete base weights exist. |
| `tokenizer.json` | Fully parsed BPE model: base vocabulary 151,643 entries, 151,387 merges, 22 added-token entries, NFC normalizer, split + byte-level pretokenizer, byte-level postprocessor/decoder. No saved tokenizer padding/truncation configuration. Counts distinguish base vocabulary from added tokens. |
| `chat_template.jinja` | Role-delimited Qwen chat template, default helpful-assistant system message, vision image/video marker expansion, optional numbered vision IDs, and an assistant generation prefix. It does not inject the SatQuery QA suffix or caption instruction automatically. |

### Entry points and exact input/output specification

**No callable inference entry point or model-definition Python is shipped.** The names above identify processor/base classes to load later; no tested `run`, generation parameters, decoding wrapper, or output JSON contract can be recovered from local code.

The project/user contract is **one optical RGB image + text → generated text**. A multispectral GeoTIFF must first become RGB through an explicitly selected preview/inference conversion; this adapter does not document direct arbitrary-band tensor ingestion. Class names: not applicable (generative QA/captioning, no fixed class list). No evidence image, grounding, or calibrated confidence is established by the package.

User-provided training prompts, to preserve verbatim:

- QA: `question + " Answer with a short phrase."` (including the initial space).
- Caption: `"Describe this satellite image in detail."`.

The saved pixel settings numerically match the user-provided training bounds:
`128*28*28 = 100352`, `400*28*28 = 313600`. These are pixel-count budgets, **not** a 100352-by-313600 image shape or a fixed 256/512 resize. The nested saved `size` field names require a compatible processor implementation; the audit does not execute resizing to verify its interpretation.

Saved image preprocessing: convert RGB; rescale by `1/255`; channelwise normalization with:

```text
mean = [0.48145466, 0.4578275, 0.40821073]
std  = [0.26862954, 0.26130258, 0.27577711]
```

Do not apply the same normalization twice. No generation `max_new_tokens`, sampling/beam settings, precision, attention implementation, or demonstrated tensor layout is supplied. The dynamic processor, not a guessed fixed tensor size, must define the eventual visual tensor inputs.

### Versions, evaluation, and MacBook Air risks

- **Local version evidence:** PEFT **0.21.2** in adapter config and model card. Python, PyTorch, Transformers, tokenizers library version, base model commit/revision, training precision, and accelerator are unspecified. The safetensors values themselves are float32.
- **Evaluation:** no numerical QA/caption metrics, test split, or latency measurements are supplied. VRSBench training is a project/user fact, not independently substantiated by this placeholder model card.
- The multi-billion-parameter base must be provided separately (other folders/Hugging Face caches were not audited); memory/activation cost will greatly exceed the 297 MB adapter. Available unified RAM and latency are unmeasured. Do not promise this fits the Air in a chosen precision.
- The adapter has vision targets and nested language-model key paths: use a matching base/PEFT/Transformers combination rather than assuming a different-version Qwen hierarchy is interchangeable.
- No code requires CUDA or bitsandbytes locally, but there is also no proven CPU/MPS inference path. Future device selection must obey `cuda → mps → cpu` with fallback, without CUDA-only quantization or attention assumptions.

### Open questions

1. Which exact base revision and Transformers/PyTorch/Python versions produced this export? Can the supplied nested processor config and all adapter keys be loaded strictly with them?
2. What precision, generation settings, and chat-template invocation were used during validation? Was there additional RGB stretching or resizing before the processor?
3. Where are the base weights, genuine input/output examples, measured evaluation scores, and model/dataset licensing/provenance details?
4. What CPU/MPS latency and peak unified-memory usage are acceptable on this specific Air configuration?

### Proposed integration plan — not implemented

Create one lazy `SpecialistModel` adapter around the explicit Qwen base plus PEFT adapter. Reuse the supplied adapter/config, tokenizer, processor settings and chat template. Match keys/versions first; preserve both vision and language adapter targets. Centralize exact QA/caption prompts and pixel budgets, control generation options, decode generated answer tokens only, and return standard metadata. Keep evidence and confidence null unless the implementation can actually support them; token-log-prob confidence, if introduced later, is uncalibrated. Validate processor loading and one genuine QA/caption example on CPU before trying MPS. There is no teammate inference server to reuse for Model 1.

## Model 2 — Grounding DINO teammate handover

### Full file tree and completeness
```text
SatQuery_Model2_Handover/
├── backend/
│   ├── app.py  [3,996 bytes]
│   └── requirements.txt  [125 bytes]
├── config.json  [2,652 bytes]
├── FRONTEND_INTEGRATION.md  [2,909 bytes]
├── inference.py  [2,569 bytes]
├── model.safetensors  [689,359,096 bytes]
├── preprocessor_config.json  [496 bytes]
├── README.md  [1,011 bytes]
├── requirements.txt  [82 bytes]
├── special_tokens_map.json  [695 bytes]
├── tokenizer.json  [711,396 bytes]
├── tokenizer_config.json  [1,267 bytes]
├── training_metadata.json  [387 bytes]
└── vocab.txt  [231,508 bytes]
```
- Weight SHA-256: `eaf76f68823f8a56ad22e7a3e4fd65bd1b00bb5862215a69e1b59986117fc71b`.
- This is a **Hugging Face `save_pretrained`-style full model directory**, not an adapter or original GroundingDINO `.pth` + standalone Python config. Architecture `GroundingDinoForObjectDetection`, model type `grounding-dino`, Swin backbone and BERT text encoder are specified in `config.json`.
- Safetensors header: 132,232 bytes, 990 tensors (978 F32, 12 I64), 172,277,902 stored scalar elements **including buffers**, not a trainable-parameter claim. Data payload 689,226,856 bytes; contiguous offsets exactly match physical file size. It is not suspiciously small.
- Model weights/config, image processor config, fast tokenizer JSON, text vocabulary, tokenizer config, and special-token map are present. No model-definition Python is provided: architecture implementation comes from Transformers. No separate `processor_config.json` is present, but `processor_class` is named in supplied processor/tokenizer configuration; actual `AutoProcessor` loading is still untested here.
- Missing: genuine test images, successful real-image outputs, numerical grounding evaluation report, training notebook/log/optimizer state, exact Torch/Python lockfile. No unpacking required.

### Documents/configuration read

| File | Observed contents |
| --- | --- |
| `README.md` | Recovered fine-tuned Grounding DINO Tiny, base `IDEA-Research/grounding-dino-tiny`, VRSBench, step 1300, Transformers 4.51.3. **Claims** strict zero-missing/unexpected-key load and independent reload; explicitly says satellite-image inference remains untested. |
| `FRONTEND_INTEGRATION.md` | Backend setup, `/health` and `/predict` form fields/response, absolute pixel XYXY drawing guidance, CPU caveat. Response confidence 0.82 is illustrative, **not** evaluation. Warns not to describe checkpoint recovery as a completed 120-minute training run. |
| `training_metadata.json` | Step 1300; training seconds `1918.3250926409992` (about 31.97 minutes); training image size 512; recovered/exported status; base/dataset/version consistent with README. No loss/mAP/accuracy or evaluation split. |
| `config.json` | Swin depths `[2,2,6,2]`, embed dim 96, heads `[3,6,12,24]`, channels 3; BERT hidden 768, 12 layers/heads, vocabulary 30522. Detection hidden 256, encoder/decoder 6 layers each, 900 queries, maximum text length 256, four feature levels; float32. `disable_custom_kernels=false`, `use_pretrained_backbone=false`, `use_timm_backbone=false`. Backbone `image_size=224` is architecture configuration; supplied image processor uses 512×512. |
| `preprocessor_config.json` | `GroundingDinoImageProcessor`, `GroundingDinoProcessor`, resize/scale/normalize/pad enabled, target `{height:512,width:512}`, resample 2, scale `1/255`, ImageNet mean/std listed below; COCO annotation format with conversion enabled, automatic padding (`pad_size=null`). |
| `tokenizer_config.json` | `BertTokenizer`, lowercase enabled, max tokenizer length 512, processor class GroundingDinoProcessor, PAD/UNK/CLS/SEP/MASK token definitions. Model text limit 256 is distinct from tokenizer's 512 setting. |
| `special_tokens_map.json` | Full definitions for `[PAD]`, `[UNK]`, `[CLS]`, `[SEP]`, `[MASK]`. |
| `tokenizer.json` | Fully parsed WordPiece tokenizer, 30,522 vocabulary entries, five added tokens, lowercase BERT normalizer/pretokenizer, CLS/SEP template processing, WordPiece decoder. No saved padding/truncation config. |
| `vocab.txt` | All 30,522 lines counted; PAD=0, UNK=100, CLS=101, SEP=102, MASK=103. This is a **text-token vocabulary**, not detection classes. |
| `requirements.txt` | `torch>=2.0`, `transformers==4.51.3`, `huggingface-hub==0.36.2`, `tokenizers==0.21.1`, `pillow`. |
| `backend/requirements.txt` | Same five entries plus `fastapi`, `uvicorn[standard]`, `python-multipart`. Torch/Pillow/backend package versions remain unpinned. |

### Exact entry points, preprocessing, and output

`inference.py`:

```python
SatQueryGrounding.__init__(self, model_dir=None, device=None)
SatQueryGrounding.ground(self, image_path, text_prompt,
                        box_threshold=0.3, text_threshold=0.25)
```

The constructor defaults to this folder, loads `AutoProcessor` and `AutoModelForZeroShotObjectDetection` from it, moves the model to the chosen device, and calls `eval()`. Device defaults to CUDA if available, otherwise CPU; **MPS is not auto-selected**. `ground` has `torch.inference_mode()` decoration.

Input: one Pillow-readable image path, converted to RGB, and a text phrase. The wrapper itself does not interpret geospatial bands or GeoTIFF reflectance; project preprocessing must provide correct RGB. It strips surrounding text whitespace and appends `.` if absent. It does not lowercase the prompt itself (tokenizer lowercasing is configured). Calls `processor(images=image, text=prompt, return_tensors="pt")`; moves tensors to the same device. Configured input image size 512×512; rescale `1/255`; normalize RGB using:

```text
mean = [0.485, 0.456, 0.406]
std  = [0.229, 0.224, 0.225]
```

Postprocessor receives outputs, `inputs.input_ids`, both thresholds, and `target_sizes=[image.size[::-1]]` (original height,width). Therefore returned coordinates are **absolute original-image pixel XYXY**, not normalized boxes. Detection coordinates are rounded to two decimals; confidence scores to four decimals. No masks. No fixed label/class list: decoded labels are prompted text phrases.

Actual wrapper output, **different from the requested project contract**:

```text
{
  image: str(path), prompt: formatted_prompt,
  image_width: int, image_height: int, detection_count: int,
  detections: [{box: [x1,y1,x2,y2], confidence: float, label: str}, ...]
}
```

It does **not** directly return `{boxes, labels, scores}`. The eventual adapter must map detection rows. There is no explicit score calibration, global answer confidence, per-run unload, or lazy registry lifecycle in this wrapper. The CLI accepts `--image`, `--prompt`, and `--box-threshold`; CLI does not expose its text threshold.

`backend/app.py` functions and endpoints:

```text
load_model()                         startup hook, eager model load
root()                               GET /
health()                             GET /health
predict(image: UploadFile=File(...), prompt: str=Form(...),
        box_threshold: float=Form(0.30), text_threshold: float=Form(0.25))
                                     POST /predict
```

Backend defaults `MODEL_DEVICE` to `cpu` even on CUDA-capable machines. It checks nonempty prompt and thresholds in [0,1], reads the **whole upload before** enforcing 20 MiB, converts to RGB, saves a temporary file, calls `detector.ground`, and removes the temporary file. It accepts any Pillow-readable decoded image; temporary suffix whitelist is PNG/JPG/JPEG/WebP, otherwise PNG. Response adds `success`, image filename/dimensions, `coordinate_format="pixel_xyxy"`, and a detection list containing `label`, `confidence`, `bbox`, and `box:{x1,y1,x2,y2}`. Default CORS is wildcard.

**Static import-order defect:** `from inference import SatQueryGrounding` occurs before adding the parent folder to `sys.path`. The documented `cd backend; uvicorn app:app` startup may fail to import `inference` unless the parent is already on `PYTHONPATH`. This is a source-level risk, not an executed failure in this audit.

### Evaluation and MacBook Air risks

No measured mAP/AP/IoU/precision/recall/accuracy is present. Step 1300 and elapsed training time are provenance, not evaluation. The zero-missing/unexpected-key and reload assertions are **handover claims**, not independently verified here; genuine satellite-image testing is explicitly outstanding.

Torch requirements do not mandate CUDA; CPU is explicitly supported and used by the packaged backend. MPS compatibility is untested, and `disable_custom_kernels=false` is a reason to verify a CPU-compatible/native attention path rather than assume custom CUDA ops work on Apple Silicon. Keep detection-score calibration honest: scores are uncalibrated, and do not invent a global confidence when only per-box scores are supported. Model/activations will require more memory than its 689 MB file. No bitsandbytes dependency is listed. Its pinned older Transformers environment may differ from the version needed for Model 1's export.

### Open questions

1. What are measured grounding metrics and their evaluation protocol/split? Can the teammate supply genuine image/phrase/box outputs?
2. Does the exact saved directory load with its pinned dependencies on Apple Silicon, and does postprocessing return phrase strings as expected in that version?
3. Which Python/Torch/torchvision versions and custom-kernel settings were actually used? Is CPU or MPS latency acceptable?
4. Should text threshold stay 0.25, and should scores/boxes retain full precision instead of the wrapper's rounding?
5. Can the startup/import-path issue be resolved in a future adapter without adopting the standalone server?

### Proposed integration plan — not implemented

Reuse local HF model/config/processor/tokenizer and the core prompt/postprocessing logic in `SatQueryGrounding`. Wrap it behind the project `SpecialistModel`, passing the selected device explicitly, exposing project `ground(image_path,text_prompt,box_threshold=0.3)`, and mapping rows to absolute-pixel `{boxes,labels,scores}`. Treat optional text threshold as a recorded parameter. Keep scores uncalibrated and report no unsupported aggregate confidence. Replace eager standalone FastAPI startup, wildcard CORS, temporary-file/upload duplication, path hacks and whole-file limit handling with the existing central API/agent infrastructure. Add strict-load and one real-image box-coordinate comparison before enabling real mode.

## Model 3 — Chg2Cap change captioning

### Full file tree and completeness
```text
chg2cap_3h/
└── run_01/
    └── best.pth  [1,316,060,457 bytes]
```
- Weight SHA-256: `667a4f72ed352a6c89c12a822f167b1fd931d96a490dbc346a4c82cc328b1233`.
- **`run_01/best.pth` exists; standalone `vocab.json` is missing; optional `last.pth` is missing.** Do not treat missing optional last checkpoint as a blocker.
- No README, requirements, config, training log, inference script, model definitions, or evaluation report is supplied locally. There are no such local documents to summarize, and no extra archive needs unpacking.
- `best.pth` is 1,316,060,457 bytes, not a placeholder. It contains 762 ZIP members, including 756 tensor storage entries and a 118,010-byte `best/data.pkl`. All archive CRCs pass. Archive serialization version `3`, little-endian, format version `1`, storage alignment `64`. **Serialization version is not the PyTorch package version.**

### Checkpoint metadata recovered without Torch

The exact top-level keys are:

```text
encoder_dict
encoder_trans_dict
decoder_dict
vocab
max_length
```

| Key | Verified contents |
| --- | --- |
| `encoder_dict` | 624 tensor descriptors; first convolution `[64,3,7,7]`, ResNet stage block counts `[3,4,23,3]`, final channels 2048. These are consistent with the user-provided ResNet-101 backbone. |
| `encoder_trans_dict` | 80 tensor descriptors; height and width embedding matrices `[16,1024]`; `selftrans` layer indices 0,1,2. Query projection `[2048,2048]`, first MLP `[512,2048]` and back projection `[2048,512]`; second stream uses 4096-channel representations. |
| `decoder_dict` | 52 tensor descriptors; `Conv1.weight=[2048,4096,1,1]`, vocabulary embedding `[436,2048]`, output projection `[436,2048]` and bias `[436]`; sinusoidal buffer `[41,1,2048]`; additional position embedding `[52,2048]`. One transformer decoder layer appears in key names. |
| `vocab` | **Embedded dict of 436 token→id entries**, contiguous unique ids 0–435. `<NULL>=0`, `<UNK>=1`, `<START>=2`, `<END>=3`. Therefore missing external `vocab.json` need not require retraining; the authoritative mapping is already serialized here. Do not substitute another LEVIR-CC vocabulary. |
| `max_length` | Integer **41**. |

All 756 tensor storage descriptors are tagged **`cuda:0`**. Tensor storage types include float and long, including batch-normalization counters. There is no top-level epoch, optimizer state, BLEU score, training version, or model-source commit. The file name “best” by itself does not establish an evaluation number or selection criterion. Vocabulary items are caption tokens, not a land-cover class list.

### Entry points, input/output, and preprocessing provenance

**No local callable inference function exists.** For comparison only, the original repository was read at commit `7b8cda937002e614d51a6dab3d949aa7de77c176`:

- [Official Chg2Cap repository](https://github.com/ShizhenChang/Chg2Cap).
- [Pinned test entry point](https://github.com/ShizhenChang/Chg2Cap/blob/7b8cda937002e614d51a6dab3d949aa7de77c176/test.py).
- [Pinned encoder definitions](https://github.com/ShizhenChang/Chg2Cap/blob/7b8cda937002e614d51a6dab3d949aa7de77c176/model/model_encoder.py).
- [Pinned decoder definitions](https://github.com/ShizhenChang/Chg2Cap/blob/7b8cda937002e614d51a6dab3d949aa7de77c176/model/model_decoder.py).
- [Pinned LEVIR-CC loader](https://github.com/ShizhenChang/Chg2Cap/blob/7b8cda937002e614d51a6dab3d949aa7de77c176/data/LEVIR_CC/LEVIRCC.py).

These files are **external reference**, not files added to `weights/`, and not evidence of the exact local training revision.

Exact relevant upstream interfaces:

```text
test.main(args)
Encoder.__init__(self, network)
Encoder.forward(self, imageA, imageB)
AttentiveEncoder.__init__(self, n_layers, feature_size, heads, hidden_dim,
                         attention_dim=512, dropout=0.0)
AttentiveEncoder.forward(self, img1, img2)
DecoderTransformer.__init__(self, encoder_dim, feature_dim, vocab_size,
                            max_lengths, word_vocab, n_head, n_layers, dropout)
DecoderTransformer.sample(self, x1, x2, k=1)
DecoderTransformer.sample1(self, x1, x2, k=1)
```

Upstream test defaults: `resnet101`, encoder/feature dim 2048, `feat_size=16`, heads 8, attentive layers 3, decoder layers 1, hidden dim 512, attention dim **2048** (explicitly overrides constructor default 512), dropout 0.1, max length 41, test batch size 1. Tensor shapes support many dimensions, but **heads, dropout, stride changes, training transforms and exact code revision cannot all be inferred from state dictionaries**.

The user confirms training used 256×256 before/after optical images, ResNet-101, feature-size setting 16, max length 41. Intended API: two RGB images (A before, B after) → one change-description string, no user-question answering, masks, or change map. No fixed object-class list applies.

Upstream LEVIR loader uses imageio, float32 arrays, HWC→CHW, and normalization directly in nominal 8-bit RGB units, **without dividing by 255**:

```text
mean = [100.6790, 99.5023, 84.9932]
std  = [50.9820, 48.4838, 44.7057]
normalized[channel] = (raw_pixel - mean[channel]) / std[channel]
```

These values are **upstream reference**, not saved training normalization evidence for this local run. Confirm with the trainer. The current generic per-image percentile stretching is not shown in that reference and may change inference distribution. The upstream ResNet forward downsamples by 32: 256×256 would normally yield 8×8 features. A learned positional capacity of 16 does not prove an actual 16×16 feature grid was used; investigate any local stride/interpolation modifications rather than treating `feat_size=16` as direct pixel resizing.

Upstream path: shared CNN on both images → attentive encoder pair → `decoder.sample`. Greedy `sample` returns a list of token ids, stopping at END; text decoding should invert the embedded token→id map and omit START/END/NULL. The upstream test indexes `list(word_vocab.keys())[id]`, which assumes insertion order; the local ids are explicit, so a proper inverse map is safer. No supported answer confidence or visual evidence is emitted by that sampling API.

### Versions, evaluation, and MacBook Air risks

No local runtime versions or evaluation numbers are provided. Do not assign the original paper's scores to this local `best.pth`.

[Upstream requirements at the pinned commit](https://github.com/ShizhenChang/Chg2Cap/blob/7b8cda937002e614d51a6dab3d949aa7de77c176/requirements.txt) list Torch **1.11.0+cu113**, torchvision **0.12.0+cu113**, NumPy 1.23.0, Pillow 8.2.0, einops 0.4.1, plus lmdb/h5py/matplotlib/opencv/tensorboard/protobuf/scikit-image/scikit-learn and pip pins. These are **upstream legacy CUDA requirements**, not a verified environment for the delivered checkpoint; do not install that list on the Air. The loader additionally imports imageio. Evaluation utilities have extra tool needs, which are not required just to generate a caption.

CUDA assumptions are concrete: checkpoint storages target `cuda:0`; upstream test moves all models/data with `.cuda()`, attentive encoder builds position indices on CUDA, decoder sampling creates target/mask/sequence tensors on CUDA. CPU loading needs explicit remapping and a device-portable rewrite of these allocations. The upstream encoder uses `pretrained=True`, which can cause an unnecessary network download even though the checkpoint already contains CNN weights. The checkpoint payload alone is around 1.3 GB, and simultaneous encoder/attention/decoder allocations and intermediate activations add substantial unified-memory demand. MPS support and latency are untested; no bitsandbytes dependency is present.

### Open questions

1. What exact training source commit/notebook and architecture changes produced `run_01/best.pth`? Confirm heads, dropout, stride/dilation/interpolation and all constructor arguments.
2. Are the upstream RGB mean/std and raw pixel units exactly those used for this run? Was percentile stretching ever used?
3. Can the trainer supply a genuine before/after pair, expected caption, epoch/selection criterion, evaluation protocol and measured scores?
4. Should a later integration export the embedded authoritative 436-token mapping to a separate JSON for portability? Its absence is documented, not silently replaced here.
5. Which modern Apple Silicon Python/Torch/torchvision/einops combination can load all three dictionaries strictly and generate identical text without CUDA?

### Proposed integration plan — not implemented

Obtain/pin the actual training definitions first; use upstream classes only after state-key/shape agreement is established. Reuse the checkpoint and embedded vocabulary/max length. Build a lazy change specialist that accepts two correctly normalized 256×256 RGB inputs, initializes the CNN without an external pretrained download, remaps storage to CPU, and makes all position/mask/token allocations device-aware. Reuse the CNN→attentive encoder→decoder sampling design; rewrite the dataset/evaluation CLI into a small per-pair inference method. Preserve A/B order and return only a change caption, null confidence with “not available,” and input previews labelled as inputs rather than a generated map. Confirm greedy output against a reference pair before considering MPS acceleration.

## Model 4 — optical/SAR fusion teammate handover

### Full file tree and completeness
```text
SatQuery_Model4_Handover/
├── backend/
│   ├── app.py  [6,637 bytes]
│   └── requirements.txt  [64 bytes]
├── documentation/
│   ├── DEVELOPER_HANDOVER.md  [3,885 bytes]
│   ├── FRONTEND_INTEGRATION.md  [4,768 bytes]
│   ├── MODEL_TECHNICAL_REPORT.md  [6,675 bytes]
│   ├── README.md  [2,129 bytes]
│   └── TESTED_ENVIRONMENT.txt  [519 bytes]
├── examples/
│   └── README.md  [337 bytes]
├── model/
│   ├── metadata.json  [620 bytes]
│   ├── model4_fusion_torchscript.pt  [7,157,655 bytes]
│   └── model4_normalization_stats.json  [1,760 bytes]
├── .DS_Store  [6,148 bytes]
└── README.md  [822 bytes]
```
- Weight SHA-256: `0ad943ff036969f544204a5d58c98d5b69c6450f08ea842edc93f6770813422c`.
- The 7,157,655-byte TorchScript file is consistent with a small ~1.8M-parameter float32 CNN, not evidence of a truncated transformer. All archive CRCs pass. The `.DS_Store` is Finder metadata.
- All three documented deployment artifacts exist under **`model/`**: TorchScript, normalization JSON, metadata JSON. However `backend/app.py` looks for all three **beside `app.py` in `backend/`**, where they are absent. README explicitly asks users to copy them. The package as currently laid out will fail on opening its first JSON path if imported unchanged. No copies were made in this audit.
- Missing: original model-class Python/notebook, training script, export script/cell, raw PyTorch checkpoint, optical-only and SAR-only baseline weights, original prediction logs/split manifests/confusion matrices, and genuine example TIFF pair. `examples/README.md` explicitly confirms no sample is included. These do not prevent a potentially usable TorchScript artifact, but limit reproduction and parity checking.

### All handover documents/configuration read

| File | Observed contents |
| --- | --- |
| Root `README.md` | Directory layout; copy artifacts into backend; local Uvicorn startup; explicitly says genuine GeoTIFF API inference and frontend integration are pending. |
| `documentation/README.md` | Eight-class dual encoder overview, Python 3.10+, CPU-compatible environment, rasterio/GDAL setup, endpoints and aligned 13+2-band 256-pixel requirements; real paired prediction still unverified. |
| `documentation/DEVELOPER_HANDOVER.md` | Artifact list, CPU specifications, evaluation summary, limits and integration responsibilities. **Claims verified** export/reload, finite synthetic logits/output shape, FastAPI import/startup, root/health, invalid input rejection and local HTTP startup. **Explicitly not verified:** genuine pair success, revised API parity with original model, frontend, partner machine. |
| `documentation/FRONTEND_INTEGRATION.md` | API v1.1.0, multipart inputs, exact bands/order/units/alignment, normalization rules, class mapping, top3 response, HTTP 400/500 errors, illustrative 94.32% example, CORS via FRONTEND_ORIGINS. Examples are not metrics. |
| `documentation/MODEL_TECHNICAL_REPORT.md` | SEN12MS summer, dataset split/caps, architecture/training summary, baselines and classwise results, CPU deployment and genuine-image verification limitations. Details below. |
| `documentation/TESTED_ENVIRONMENT.txt` | Claimed working Colab CPU environment versions, listed below; not a lockfile or proof of Mac compatibility. |
| `examples/README.md` | Requires genuine aligned S2/S1 13+2-band 256×256 pair; none supplied. |
| `model/metadata.json` | `LargerFusionModel`, TorchScript, 8 classes, 1,787,672 reported original parameters; input shapes, class list, best epoch 2; validation macro F1 0.6030353382505296 and test metrics. Does not identify export call/commit or input dtype. |
| `model/model4_normalization_stats.json` | SEN12MS summer, 7,247 training samples, 256×256, population pixel-level train-only statistics, 474,939,392 pixels per band, exact S1/S2 order/units/means/std. Full numerical values below. |
| `backend/requirements.txt` | Unpinned `fastapi`, `uvicorn[standard]`, `python-multipart`, `numpy`, `torch`, `rasterio`. No Transformers/torchvision/bitsandbytes requirement. |

### TorchScript export facts and model definition

The archive contains generated `code/__torch__/___torch_mangle_122.py`, `data.pkl`, `constants.pkl`, 32 float32 tensor constants, source debug metadata, serialization version `3`, and byteorder `little`.

Exact serialized callable:

```text
LargerFusionModel.forward(self, optical: Tensor, sar: Tensor) -> Tensor
```

`data.pkl` identifies `__torch__.___torch_mangle_122.LargerFusionModel`. Generated code has empty registered parameter/buffer lists and references weight constants; the graph is inlined. Its operations are paired convolution/ReLU stacks, adaptive average pooling to 1×1, flatten, concatenation, and two linear layers with intervening ReLU. First convolution shapes are `[48,13,3,3]` and `[48,2,3,3]`; stage channels 48→64→128→192; classifier weights `[256,384]` and `[8,256]`. This corroborates the documented architecture. There is no active dropout or batch normalization operation in the serialized forward; this is consistent with evaluation/export optimization, **not proof of a particular freeze/optimization call**.

**Tracing evidence:** embedded debug strings include `torch/jit/_trace.py` frames `trace_module`, `_trace_impl`, and `trace`, plus original forward frames in `/tmp/ipykernel_824/2622415992.py` and a Colab execution stack. This supports a **traced export**, rather than merely guessing from the `.pt` extension. The original export script/cell and flags are missing, so the exact `trace(...)` arguments and whether freeze/optimization followed it remain unknown.

All serialized weight constants use `FloatStorage` at **`cpu`**. There are 1,786,040 constant scalar elements in the optimized graph, distinct from the report's 1,787,672 original trainable parameters. The difference and removed BN operations are consistent with folding, but cannot establish the original export sequence by themselves. The artifact is CPU-resident float32; **the device used to create the trace's example inputs is not recorded conclusively**. No model inference was run here.

Metadata's deployment inputs are `[1,13,256,256]` optical and `[1,2,256,256]` SAR. Backend constructs contiguous **float32** tensors in exactly these shapes. Generated forward schema itself is generic `Tensor` and does not prove the exact original trace-example shape/dtype, dynamic-size support, or dynamic batch guarantees. Treat batch 1 and 256×256 as the supported documented contract until verified.

### Exact inputs, preprocessing, labels, and outputs

Required API inputs:

- `optical_file`: one genuine Sentinel-2 GeoTIFF, exactly 13 bands in file order **B1,B2,B3,B4,B5,B6,B7,B8,B8A,B9,B10,B11,B12**; 256×256; raw digital numbers, documented as uint16. API reads/casts values to float32; it **does not enforce the original dtype or validate band descriptions**.
- `sar_file`: one genuine Sentinel-1 GeoTIFF, exactly 2 bands in file order **VV,VH**, 256×256, already **dB**; cast to float32.
- Both must have non-null CRS and exactly matching CRS, affine transform, width, height. No reprojection, resampling, arbitrary band selection, or conversion from RGB/PNG is implemented. Band count is checked; semantic order is trusted. The code does not explicitly enforce the GDAL driver or filename extension as TIFF; its actual acceptance is based on raster readability and these metadata checks.
- All read samples must be finite. `src.read(out_dtype="float32")` is **not a masked nodata read**: finite nodata sentinels can pass this check, and nodata policy is undocumented.

`normalized = (raw - mean) / std` separately for each CHW band. Do **not** divide optical DNs by 10000, RGB-stretch, or feed a three-band PNG. The scale factor 10000 is recorded as physical provenance only. Do **not** log-convert SAR already in dB, clip it to the project's demo range, or scale it to 0–1 before this normalization. The teammate backend performs **no SAR log conversion or clipping**. If accepting linear SAR later, unit conversion must be explicit and validated; current packaged API trusts dB units.

Exact saved normalization values (train split only):

| Sensor / band | Mean | Population standard deviation |
| --- | ---: | ---: |
| S2 B1 | 1344.2035781483462 | 484.0961389781171 |
| S2 B2 | 1119.4291560427146 | 358.4442474586978 |
| S2 B3 | 1039.7258260881422 | 423.5337662161527 |
| S2 B4 | 986.5747855107373 | 635.1867693815674 |
| S2 B5 | 1187.9583881662948 | 612.820536893261 |
| S2 B6 | 1843.9942629458708 | 783.939961063333 |
| S2 B7 | 2164.2105789026655 | 968.2629924150517 |
| S2 B8 | 2092.1693198318662 | 961.5153002876779 |
| S2 B8A | 2369.6968720610985 | 1087.4761760979331 |
| S2 B9 | 611.5940672531117 | 539.8988090969144 |
| S2 B10 | 15.531858945067247 | 12.680100219583606 |
| S2 B11 | 1862.1692066068927 | 1069.15227210979 |
| S2 B12 | 1213.028093932036 | 901.4081685341993 |
| S1 VV | -12.02864839417873 | 4.870063490884392 |
| S1 VH | -19.1828373740414 | 5.706944295180828 |

The stats were computed over **474,939,392 pixels per band**, consistent with 7,247×256×256. Band selection is **all 13 optical + both SAR bands**, not a selection of RGB bands from the optical stack.

Class index mapping, exactly as metadata and docs:

| Index | Class |
| ---: | --- |
| 0 | Forest |
| 1 | Shrubland |
| 2 | Savanna |
| 3 | Grassland |
| 4 | Wetlands |
| 5 | Croplands |
| 6 | Urban_Builtup |
| 7 | Water |

Original dataset labels 8 (Snow/Ice) and 9 (Barren) were excluded per the technical report. The original-label-to-new-index mapping file and dataset label derivation are not included.

The raw TorchScript forward returns class logits (documented/tested shape claim `[1,8]`; graph final linear output width 8). Backend applies `torch.softmax(logits,dim=1)[0]`, then descending `torch.topk(...,k=3)`. Its output is:

```text
{
 success: true, model: metadata.model_name,
 predicted_class: str, class_index: int,
 confidence: float, confidence_percent: rounded float,
 top3: [{class: str, class_index: int, probability: float,
         probability_percent: rounded float}, ... exactly 3],
 device: "cpu",
 input: {optical_bands:13, sar_bands:2, width:256, height:256}
}
```

Confidence is an **uncalibrated softmax score**, not correctness probability. There is no free-form answer, visual explanation, or spatial evidence. The project contract `fuse_predict(optical_path,sar_path) -> {label,confidence,top3:[(label,prob),...]}` **does not exist as a callable** in this handover and needs an adapter mapping.

### FastAPI code and exact entry points

`backend/app.py`:

```text
root()                                      GET /
health()                                    GET /health
read_and_validate(optical_bytes, sar_bytes)  -> (optical_tensor, sar_tensor)
async predict(optical_file: UploadFile=File(...), sar_file: UploadFile=File(...))
                                            POST /predict
```

At module import it opens normalization and metadata JSON, constructs a CPU device, calls `torch.jit.load(...,map_location=DEVICE)` and `eval()`, builds broadcastable mean/std arrays, and checks standard deviations are positive. This is **eager global loading**, not the project lazy/unload lifecycle.

`read_and_validate` opens byte streams with rasterio, validates the input constraints above, normalizes, makes contiguous float32 arrays, then calls `torch.from_numpy(...).unsqueeze(0)` for both. `predict` reads at most `50*1024*1024 + 1` bytes per upload; empty/oversize inputs fail. Inference uses `torch.inference_mode`; output is mapped through metadata class names. Validation failures return HTTP 400 (including oversize, rather than 413); other prediction errors become generic HTTP 500; both files close in `finally`. CORS uses comma-separated `FRONTEND_ORIGINS`, default `*`, no credentials, GET/POST. FastAPI also supplies automatic docs/OpenAPI routes. No asynchronous queue, DB, trace writer, retention job, result report, or model unload exists in this teammate service.

### Framework versions, training, evaluation, and MacBook risks

`TESTED_ENVIRONMENT.txt` **claims observed in working Colab**, not independently checked:

| Component | Reported version |
| --- | --- |
| Python | 3.13.18 |
| PyTorch | 2.14.1+cpu |
| FastAPI | 0.141.1 |
| Uvicorn | 0.54.0 |
| NumPy | 2.4.6 |
| Rasterio | 1.5.2 |
| Starlette | 1.10.0 |
| python-multipart | 0.0.33 |

The general requirements are unpinned. These version strings are transcribed as supplied; this audit has not verified their availability, package provenance, artifact exporter version, or compatibility with project Python 3.11/3.12 or Apple Silicon. TorchScript serialization version `3` is **not** evidence of Torch 3.x. No CUDA is required by packaged inference (CPU is hardcoded), and no bitsandbytes/Transformers is used.

Technical report's training claims: SEN12MS **summer**; scene-separated selected train/validation subsets; 7,247 train, 1,129 validation, 4,529 independent test samples. Training capped at 1,000/class except 247 Wetlands; validation capped at 150/class except 79 Wetlands; test retains natural distribution. AdamW, learning rate 0.001, weight decay 0.0001, batch 16, seed 42, weighted cross-entropy, float16 autocast; early stopping/validation checkpoint selection. Export metadata says best epoch 2, validation macro F1 **0.6030353382505296**. Training device is not explicitly identified; float16 autocast does not itself prove a particular GPU.

**Reported independent-test numbers (not measured by this audit or API smoke tests):**

| Experiment | Accuracy | Macro F1 | Weighted F1 |
| --- | ---: | ---: | ---: |
| Optical only | 57.34% | 55.43% | 57.26% |
| SAR only | 24.22% | 27.52% | 26.61% |
| Optical + SAR fusion | **76.15%** | **62.89%** | **74.04%** |

Fusion classwise F1 claims: Shrubland ≈0.923; Savanna ≈0.865; Water ≈0.959; Croplands ≈0.673; Urban_Builtup ≈0.735; Grassland ≈0.105; Forest ≈0.308 (**24** test examples); Wetlands ≈0.466 (**109** examples). Grassland/Croplands confusion and tiny Forest support matter. Baseline checkpoints, full prediction files, class-support table and confusion matrices are absent, so these claims cannot be independently recomputed from the handover.

The compact CPU graph is a promising candidate for an Air demo, but actual TorchScript operator/runtime compatibility, latency and numerical parity remain untested. There is no need to port the original architecture simply to try the artifact later. Moving to MPS is optional future work, not something to assume is supported by the frozen/inlined graph. Default image-preview/RGB and generic SAR normalization in the current application are incompatible with this trained model's input distribution.

### Open questions

1. Can the teammate supply the original export cell/script, original architecture/training revision, exact exporter version, trace example shape/dtype/device, and freeze/optimization settings? Debug metadata supports tracing and CPU float32 constants, but does not answer every export detail.
2. Where is a genuine aligned 13+2-band pair with expected logits/predictions, and the original-vs-TorchScript-vs-packaged-API comparison?
3. Can the reported runtime versions and evaluation results be reproduced? Provide split manifests, label remapping/patch-label derivation, baseline checkpoints, class supports, prediction logs and confusion matrices.
4. What is the approved handling of nodata, linear SAR, band-order metadata, different patch extents/resolutions and non-summer/domain-shifted imagery? These are not validated by band count alone.
5. Does the CPU artifact load under a supported Apple Silicon Torch build and project Python version? Is MPS worth testing after CPU parity is proven?

### Proposed integration plan — not implemented

Reuse the existing TorchScript file, **exact** normalization JSON, class mapping and metadata, plus the useful finite-value/band/shape/alignment checks. Implement project `fuse_predict`/`SpecialistModel` around CPU `jit.load` lazily; map softmax/top3 to project labels/tuples and mark confidence uncalibrated. Accept artifact paths explicitly from the registry instead of copying files beside a standalone server. Reuse central uploads/queue/trace/report infrastructure; do not run the teammate FastAPI application or keep eager global model loads.

**Required future preprocessing separation:** feed aligned, semantically ordered **all-band raw** 13-channel optical DNs and two-channel dB SAR through the supplied bandwise mean/std. Keep RGB/SAR display previews separate from inference tensors. The current agent's three-channel optical PNG and clipped 0–1 SAR TIFF must **not** be sent into this real model. Decide whether arbitrary overlap cropping/resizing preserves the model's supported geospatial/patch semantics before enabling it; do not silently assume all aligned pairs meet SEN12MS training assumptions. Add genuine-pair parity checks before marking real mode verified.

## Cross-model integration gates — proposals only

1. `backend/models.yaml` currently has all entries in **stub** mode and paths `weights/rs_vlm`, `weights/grounding`, `weights/change`, `weights/fusion`. Those paths do not point to the four inspected folders. Leave them unchanged now; a future integration must set correct paths and actual version/provenance per adapter, not merely toggle real mode.
2. The registry has no real classes installed/registered by this audit; file existence is not runtime readiness. Preserve stub fallback until each adapter passes explicit load and genuine-input checks.
3. Treat display previews separately from model-native arrays. Model 4 definitely needs a distinct raw multispectral+dB path. Model 3 likely needs its exact training RGB distribution rather than automatic independent percentile stretching; confirm before changing it. Models 1/2 apply their saved processor normalization after RGB selection.
4. Version needs differ: Model 1 PEFT 0.21.2 and its nested processor export, Model 2 Transformers 4.51.3 pins, Model 3 legacy upstream CUDA requirements, and Model 4's reported CPU export environment must not be assumed compatible in one lockfile. Plan isolated compatibility checks before choosing a shared environment or process boundary.
5. Implement the existing lazy-load/unload interface, one active model at a time, device selection/fallback, JSON traces and provenance. Do not report model-derived evidence/confidence that its outputs cannot support. Dataset use remains academic and domain-gap warnings remain mandatory.
6. No integration code, registry edits, dependency changes, unpacking, model imports, inference runs or installation was performed in this task. The only repository deliverable is this Markdown inventory.
