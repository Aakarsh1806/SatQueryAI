# Specialist models and provenance

> Current mode update: all four adapters are configured real at the user’s request. Model 4 is not yet verified on a real pair; any older stub-only status below records the previous preparation phase. No benchmark or real-pair measurement is added by this mode switch.

SatQuery AI runs **one specialist per query** behind the same lazy-load/result
interface. All four adapters are implemented; all four are currently configured as
`real`, with prerequisite-based stub fallback. Model 4 was enabled before genuine-pair verification.
Evaluation scripts may explicitly run real adapters without
changing that configuration. **Model 4 is enabled but not yet verified on a
real pair**, because genuine aligned pairs have not been supplied.

Numerical provenance in this document is restricted to the
[artifact inventory](model_inventory.md), the current
[registry](../backend/models.yaml), and training facts supplied by the user.
The inventory is a historical static audit, not today's integration status.
Trainer/handover test scores and small local checks are distinguished below.
An adapter's successful execution does not validate a benchmark score.

## Numerical source map

| Claim | Allowed source | Interpretation |
|---|---|---|
| LoRA configuration and audited artifact sizes | `model_inventory.md` | Saved-artifact facts; sizes exclude the downloaded base model and runtime memory |
| RS-VLM and grounding local sanity-check figures | `backend/models.yaml`, `validation_status` | Small local checks, not trainer benchmark results |
| Grounding recovery step and approximate training duration | Inventory and user-supplied training details | Checkpoint recovery history, not an accuracy measurement |
| Chg2Cap full-test metrics, best checkpoint step and precision history | User-supplied training details; metrics also in registry | Supplied results/history, not independently reproduced here |
| Fusion test and baseline metrics | Inventory and user-supplied figures | Handover-reported; no local real-pair verification |
| Fusion rounded display figures | Registry | Rounded display of the supplied handover results |
| Base-model download size and peak RAM | Not established in the allowed sources | **to be added** |

## Model 1 — RS-VLM

**Qwen2.5-VL-3B fine-tuned with LoRA on VRSBench.** The base model is
`Qwen/Qwen2.5-VL-3B-Instruct`; the local folder contains the adapter and
processor/tokenizer, not a complete standalone base model.

### Training and loading

The saved LoRA configuration specifies rank **32**, alpha **64**, dropout **0.05**,
and target modules `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`
and `down_proj`. The inventory finds adapter tensors in both language and vision
MLP modules. The export identifies PEFT **0.21.2**. Training schedule, optimizer,
learning rate, training precision and exact training base revision: **to be added**.

The adapter loads the Qwen base through Transformers, applies PEFT, and strictly
checks adapter keys/shapes/loaded values. Missing base files download from Hugging
Face during ordinary first real use. The local processor/chat template is reused.
Mac/CPU inference uses the configured floating-point precision without
bitsandbytes. The base cache size and peak RAM requirement are **to be added**.

### Input, preprocessing and output

- Input: one optical RGB image and a question or caption request. Model-native
  RGB is in **0–255** units; the saved processor performs its own rescaling and
  normalization. Display percentile stretching is not substituted for native RGB.
- QA prompt: original question followed by exactly
  ` Answer with a short phrase.` (including the leading space).
- Caption prompt: exactly `Describe this satellite image in detail.`
- Processor pixel budgets: minimum **100352**, maximum **313600**. The registry
  currently uses greedy decoding with up to **20** new QA tokens or **128** new
  caption tokens. Only newly generated tokens are decoded.
- Output: text. No boxes, segmentation masks or model-generated visual evidence.
  The adapter can expose mean generated-token probability as **uncalibrated**
  confidence; it is not a calibrated probability that the answer is correct.

### Evaluation and limitations

The registry records a **20-sample VRSBench validation sanity check** with
**40% exact match** and **40% lenient match**. These are local small-sample
measurements, not full VRSBench benchmark scores. Full-benchmark QA/caption metrics
and their complete protocol: **to be added**. See the local evaluation report
for its match definitions; lenient matching is a heuristic rather than a semantic
correctness guarantee.

Generated text can hallucinate, miscount or omit details. It is not a grounding
or change-detection model. Results on imagery unlike VRSBench may be poor.

Registry ID: `rs_vlm`. Weights: `weights/satquery_rsvlm_lora/`.

## Model 2 — Grounding

**Grounding DINO Tiny fine-tuned on VRSBench.** The base is
`IDEA-Research/grounding-dino-tiny`. The handover is a Hugging Face-style full
model/processor folder using `GroundingDinoForObjectDetection` and `AutoProcessor`.

### Training and loading

The checkpoint was **recovered at step 1300**, after **about 32 minutes of
training**. It has **no reported trainer benchmark metrics**. Recovery/export and
strict loading do not establish a completed training schedule or detection quality.
The handover should not be described as a completed longer training run.
Training optimizer, learning rate, full schedule and selection protocol:
**to be added**. Training/processor image size was **512**.

The project loads the local model/processor in-process and checks loading keys.
It does not run the teammate FastAPI service. Device selection follows the shared
CUDA/MPS/CPU helper; MPS placement or inference failure retries on CPU and records
that fallback. The old package pin in the historical inventory is not an
instruction to downgrade the current shared ML environment; use the project's
ML requirements.

### Input, preprocessing and output

Input is one optical RGB image plus a text phrase. The prompt is stripped,
lowercased and terminated with a period. The saved processor resizes/rescales/
normalizes the native RGB image. Postprocessing uses original image dimensions.

```python
ground(image_path, text_prompt, box_threshold=0.3)
# -> {"boxes": [[x1, y1, x2, y2], ...], "labels": [...], "scores": [...]}
```

Coordinates are **absolute original-image pixel xyxy**, clipped to image bounds.
The text threshold is **0.25**, recorded in metadata. Outputs are boxes only,
with labels, **uncalibrated per-box scores** and a rendered PNG overlay. No masks
are produced. Aggregate answer confidence is **not available**; it must not be
invented from a box score. Labels are prompted phrases, not a fixed land-cover
class list.

### Evaluation and limitations

Trainer-reported benchmark metrics: **to be added**; none were supplied. The
current registry separately records a **10-sample local sanity check**:
**mean IoU 0.3673**, **Acc@0.5 40%**. That best-overlap check is not mAP and does
not penalize all extra detections; it must not be presented as trainer benchmark
accuracy. Grounding can miss objects, select the wrong region or return extra
boxes. A location query does not imply a segmentation map or a road trajectory.

Registry ID: `grounding`. Weights: `weights/SatQuery_Model2_Handover/`.

## Model 3 — Change captioning

**Chg2Cap trained on LEVIR-CC with a pretrained ResNet-101 backbone. It was not
fine-tuned from a pretrained Chg2Cap model.** The distinction concerns training
initialization: the CNN backbone was pretrained, while the change-captioning
system was trained for this task.

### Training and loading

User-supplied training history identifies the **best checkpoint at step 3207**.
Part of training ran in **FP32** after an **fp16 non-finite loss**. These details
were supplied by the user; the historical checkpoint audit did not recover
optimizer/step history from the delivered state dictionaries. Training optimizer,
learning-rate schedule, full run log and original training source revision:
**to be added**.

The adapter uses vendored upstream Chg2Cap code pinned at
`7b8cda937002e614d51a6dab3d949aa7de77c176`, with the upstream licence/notices retained.
That integration pin does not prove the original training used an identical
source revision. The checkpoint's CNN, attentive encoder and decoder dictionaries
load strictly with device-aware allocations and explicit ID-to-word decoding.
The checkpoint contains **436** vocabulary entries and maximum length **41**.
The adapter does not download ImageNet weights when reconstructing the network;
the checkpoint supplies its saved CNN parameters.

### Input, preprocessing and output

Input is an ordered pair of optical images of the same area: **A=before,
B=after**, prepared at **256×256**. The supported task is a **change description**.
A question about changes still receives a description plus a limitation note,
not direct change-question answering.

Training preprocessing remains **unconfirmed**. The configured default reproduces
the upstream loader: float32 RGB in **0–255** units, not divided by **255**, with
mean **[100.6790, 99.5023, 84.9932]** and standard deviation
**[50.9820, 48.4838, 44.7057]**. Images resize to **256×256** without per-image
percentile stretching for inference. The configurable ImageNet-normalization
alternative is an experiment option, not an established correction. The training
normalization confirmation is **to be added**.

Output is a caption string in the standard result object. Confidence is `null`
with reason `"not available"`. No change map, pixel localization or mask is
produced. Before/after previews are inputs, not generated evidence maps.

### Supplied evaluation and limitations

User-supplied **LEVIR-CC test-split** results, also recorded in the registry:

| Metric | Reported result |
|---|---:|
| BLEU-4 | 0.542 |
| ROUGE-L | 0.727 |
| CIDEr | 1.29 |
| METEOR | Not computed |

These full-split results are supplied training/evaluation facts, not a benchmark
reproduction by this repository. Full evaluation command, tokenizer/scoring
configuration and prediction logs: **to be added**. Local subset reports must
remain labelled as subset checks, not substitutes for this table.

The model was trained on imagery/captions emphasizing **buildings and roads**.
Descriptions may be generic, incomplete or wrong, especially outside that domain.
It is not a general change-question-answering system.

Registry ID: `change`. Weights: `weights/chg2cap_3h/run_01/best.pth`.

## Model 4 — Optical/SAR fusion

**A custom dual-encoder fusion network, about 1.8M parameters, trained from
scratch on a SEN12MS summer subset for 8 land-cover classes.** It is not a
vision-language model and does not generate free-form answers.

**Implementation status: adapter implemented, current mode `real` by user request, no genuine
aligned sample pairs available, not yet verified on a real pair.** TorchScript
loading/synthetic execution or a handover score does not establish real-pair
parity or correctness on this machine.

### Training and loading

The handover describes separate optical/SAR convolutional encoders, pooled
features, concatenation and a classifier. The original parameter count is
reported for the training architecture; optimized TorchScript constants are
not a new trainable-parameter count.

The inventory records scene-separated selected training/validation subsets,
weighted cross-entropy, AdamW, float16 autocast and validation-based checkpoint
selection. The summer subset and class imbalance constrain generalization.
Original training/export scripts and complete split manifests: **to be added**.
The artifact contains tracing evidence and CPU float32 constants; the exact
export call and original example-input device are **to be added**.

The project loads `model4_fusion_torchscript.pt` with CPU mapping and keeps it on
CPU. It reuses saved statistics and metadata in-process rather than launching
the handover backend. MPS use/parity is not established.

### Strict input contract and preprocessing

- Optical: a **13-band Sentinel-2 GeoTIFF**, in order
  **B1,B2,B3,B4,B5,B6,B7,B8,B8A,B9,B10,B11,B12**, containing raw digital numbers.
- SAR: a **2-band Sentinel-1 GeoTIFF**, in order **VV,VH**, already in **dB**.
- Both must be **256×256**, have non-null CRS, identical CRS/affine transform/
  dimensions and finite valid pixels. No automatic reprojection or resampling.
- Cast raw bands to float32 and apply each band's saved `(raw - mean) / std`
  from `model4_normalization_stats.json`. Do not divide optical values by
  **10000**, RGB-stretch native inputs, log-convert already-dB SAR or clip it to
  the display range. Native tensors are `[1,13,256,256]` and `[1,2,256,256]`.
- Display only: optical RGB uses **B4,B3,B2**, with a separate SAR preview.
  PNG/JPG, RGB-only TIFFs and previews are not valid real fusion inputs.

```python
fuse_predict(optical_path, sar_path)
# -> {"label": str, "confidence": float, "top3": [(label, probability), ...]}
```

The adapter applies softmax to class logits. Confidence and top-3 probabilities
are **uncalibrated**. Top-3 classes are ranking output, not a spatial explanation.

Classes, in saved order: **Forest, Shrubland, Savanna, Grassland, Wetlands,
Croplands, Urban_Builtup, Water**. It does not support arbitrary land-cover labels
or describe objects, changes, masks or general scene content.

### Supplied evaluation and limitations

These are **handover-reported test results**, not locally reproduced real-pair
results:

| Experiment / metric | Reported result |
|---|---:|
| Optical+SAR test accuracy | 76.15% |
| Optical+SAR test macro-F1 | 62.89% |
| Optical-only test accuracy | 57.34% |
| SAR-only test accuracy | 24.22% |

The registry displays rounded fusion figures of **76.2%** accuracy and **62.9%**
macro-F1. The table above retains the supplied handover precision. Baseline
scores are comparisons from training/evaluation, not additional inference modes
in this application; SAR-only uploads are rejected.

**Grassland, Forest and Wetlands** are weak classes in the supplied evaluation.
Other regions or seasons may perform worse. Summer-only training does not
establish year-round or global reliability. Genuine-pair parity, local accuracy,
real-pair latency and memory measurements: **to be added**.

Registry ID: `fusion`. Weights/statistics/metadata:
`weights/SatQuery_Model4_Handover/model/`.

## Academic-use scope

VRSBench, LEVIR-CC and SEN12MS are the project's training/test dataset sources;
local dataset copies are test inputs only. There is no training pipeline run here.
Datasets and the Chg2Cap code are used for **academic purposes only** under the
project's usage policy and supplied notices. The vendored Chg2Cap source retains
its MIT licence plus the upstream academic-use notice; do not remove or conflate
these notices with a newly invented project licence. Complete dataset licence
texts, attribution bibliography and approved redistribution guidance:
**to be added**. Do not commit datasets, weights or derived dataset images.
