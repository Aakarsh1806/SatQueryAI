# Slide deck and specification correction checklist

This checklist records the user's reported slide/spec issues and the current
implementation contracts. The actual deck/spec files were not supplied for this
review; slide titles, corrected page mapping and an edited-deck artifact are
**to be added**. Checking an item here requires verifying the actual slide/spec,
not merely creating this checklist.

## Project scope and claims

- [ ] State the datasets as **VRSBench, LEVIR-CC and SEN12MS only**. Remove
  **BigEarthNet, RSVQA and CDVQA** from data, training, evaluation and architecture
  claims wherever they appear. Local downloaded datasets are test inputs only;
  the application does not train models.
- [ ] Replace “every answer is backed by visual evidence and a confidence score”
  with: **“The selected specialist returns an answer and any evidence or scores
  it supports. Evidence and confidence may be not available; supplied scores
  are uncalibrated.”**
- [ ] Remove **“expert-level analysis”** and equivalent guarantees. Suggested
  wording: **“An academic assistant that routes supported satellite-imagery
  tasks to a specialist; outputs require human review.”**
- [ ] Show **one model per query**. Remove diagrams or text implying that all
  specialists run, vote, verify one another or supply an ensemble answer.
- [ ] Separate **implemented adapter**, **configured stub/real mode**, **local
  sanity check** and **trainer-reported benchmark result**. Do not use one as
  evidence for another. Show the actual mode tag on result screenshots.

## Model and output descriptions

- [ ] Model 1: **Qwen2.5-VL-3B fine-tuned with LoRA on VRSBench**. Do not describe
  the adapter as a standalone base model or imply that text answers carry boxes.
- [ ] Model 2: **Grounding DINO Tiny fine-tuned on VRSBench**; checkpoint recovered
  at **step 1300**, after **about 32 minutes**. No trainer benchmark metrics were
  reported. Do not present checkpoint recovery or local sanity checks as a full
  successful benchmark/training run.
- [ ] Replace grounding masks/segmentation examples and claims with **bounding
  boxes, labels and uncalibrated per-box scores**. Aggregate answer confidence
  is not available. “Highlight” does not add segmentation support.
- [ ] Model 3: **Chg2Cap trained on LEVIR-CC using a pretrained ResNet-101
  backbone**, not fine-tuned from a pretrained Chg2Cap. The best checkpoint was
  at **step 3207**; part of training used **FP32** after an **fp16 non-finite loss**,
  as supplied by the user.
- [ ] State that change inference produces **descriptions only**. Remove direct
  change-question-answering, change-map and change-mask claims. Label any
  before/after slider as input imagery rather than generated evidence.
- [ ] Model 4: **custom dual-encoder fusion network**, **about 1.8M parameters**,
  trained from scratch on a **SEN12MS summer subset**, **8 classes**.
- [ ] Describe fusion as **land-cover classification only**. Remove free-form
  QA, detection, segmentation or unrestricted scene-analysis claims.
- [ ] State fusion's strict input requirement: raw **256×256 GeoTIFF** optical/SAR
  pair, **13 Sentinel-2 bands** and **2 Sentinel-1 VV/VH bands in dB**, matching
  CRS and pixel grid. RGB/SAR previews are display only; no automatic resampling.
- [ ] Add a visible status: **“Model 4 adapter implemented; currently in stub
  mode; no genuine aligned samples available; not yet verified on a real pair.”**
  Do not show handover scores or synthetic predictions as a successful local
  real-pair demonstration.

## Evaluation and limitations

- [ ] Attribute Model 3's supplied LEVIR-CC test scores correctly:
  **BLEU-4 0.542, ROUGE-L 0.727, CIDEr 1.29; METEOR not computed**. Do not relabel
  subset measurements as full-test results. Full scoring protocol: **to be added**.
- [ ] Attribute Model 4's handover results: **76.15% test accuracy**, **62.89%
  macro-F1**, **57.34% optical-only accuracy**, **24.22% SAR-only accuracy**.
  These are reported evaluation comparisons, not locally verified real-pair
  results or extra application inference modes. Note weak **Grassland, Forest
  and Wetlands** classes and summer-only training.
- [ ] Mark missing full-benchmark figures for Model 1 and trainer benchmark
  figures for Model 2 as **to be added**. If local registry sanity-check numbers
  are included, retain their small-sample label and metric definition.
- [ ] Mark all provided confidence/routing scores **uncalibrated**, and explicitly
  show **not available** where unsupported. Add domain-gap limitations for
  resolution, region, season, sensor and preprocessing changes.
- [ ] Add academic-use restrictions and attribution for **VRSBench, LEVIR-CC,
  SEN12MS and Chg2Cap code**. Preserve the vendored MIT/upstream notices; do not
  claim the project grants unrestricted dataset redistribution rights.

## Numbering and references

- [ ] Correct the **two slides numbered 4**. Determine the intended ordering
  from the actual deck; corrected numbering is **to be added**.
- [ ] Fix the **wrong footer number on the references slide**. Its correct
  final number is **to be added** after the slide order is checked.
- [ ] **Add the LEVIR-CC reference**; complete approved bibliographic entry and
  slide placement are **to be added**.
- [ ] **Remove the BigEarthNet reference** and any references retained solely
  to support removed dataset claims.
- [ ] Check remaining references against the actual VRSBench, SEN12MS and model
  sources. Complete citations are **to be added** where not supplied; do not
  invent authors, publication years, titles or identifiers.
- [ ] Recheck numbering, figure captions, model names, dataset names and footers
  consistently across the slides and specification after the edits.

Use [models.md](models.md) for attributed model facts,
[limitations.md](limitations.md) for output boundaries and
[architecture.md](architecture.md) for the implemented flow. The
historical inventory should not be copied as if its earlier “not implemented”
status were still the application's current state.
