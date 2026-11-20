# Limitations and honest interpretation

> Current mode update: all four adapters are configured real at the user’s request. Model 4 is not yet verified on a real pair; any older stub-only status below records the previous preparation phase. No benchmark or real-pair measurement is added by this mode switch.

SatQuery AI is an academic demonstration. The supported output of each specialist
sets the limit of what the UI, reports, specification and slides may claim.
Neither a successful API response nor a numerical score establishes expert-level
analysis or a correct answer on an unfamiliar image.

## Output boundaries

| Specialist | Supported output | Unsupported claims |
|---|---|---|
| RS-VLM | Text QA or caption for one optical image | Guaranteed counts/correctness; model-generated visual evidence for every answer |
| Grounding | Absolute-pixel boxes, labels and per-box scores | Masks, pixel-perfect segmentation or a guaranteed road trajectory |
| Change | Text description comparing before and after | Change maps, change masks, direct change-question answering |
| Fusion | An 8-class land-cover prediction and top-3 scores | Free-form QA, object detection, change analysis or a spatial explanation |

Grounding returns **boxes only**, even for requests using words such as “highlight”
or “where.” The change model returns **text only** and was trained on captions
emphasizing **buildings and roads**. If the user asks a question about changes,
the application notes that it returns a description rather than a direct answer.
A before/after input comparison is not a generated change map.

## Fusion availability and input restrictions

**Model 4 has an implemented adapter enabled in real mode by user request, with no real
sample pairs available. It is not yet verified on a real pair.** Reported trainer
metrics and synthetic execution checks do not close this gap. Real-pair parity,
accuracy, latency and memory requirements are **to be added**.

Real fusion is an **8-class land-cover classifier** needing raw **256×256**
GeoTIFFs: **13 Sentinel-2 bands** in the saved order and **2 Sentinel-1 VV/VH bands
already in dB**. Both inputs must have matching georeferencing and non-null CRS.
The raw optical digital numbers and SAR dB values are normalized with saved
per-band statistics. RGB previews, PNG/JPG, clipped SAR display arrays and
arbitrarily resized/misaligned pairs are not substitutes. No automatic fusion
resampling is performed. Missing band descriptions mean semantic order cannot
be independently verified; correct units/order remain an input responsibility.

The fusion model was trained on **SEN12MS summer data only**. Forest, Grassland
and Wetlands are weak classes in the reported evaluation. Seasonal, regional,
sensor and processing differences can degrade performance. Optical-only and
SAR-only baseline scores do not make those valid standalone application inputs.

## Evidence, confidence and evaluation

All confidence scores currently supplied by the project are **uncalibrated**:
RS-VLM generated-token probabilities, grounding per-box scores, fusion softmax
scores, and heuristic routing scores. They are not interchangeable and do not
mean the answer has that probability of being correct. A calibrated confidence
procedure and its validation are **to be added**.

Chg2Cap confidence is **not available**. Grounding has no aggregate answer
confidence. RS-VLM and change captioning do not produce model-generated visual
evidence. The UI must show **not available** rather than fabricating evidence or
confidence. Fusion top-3 classes are classification outputs, not a visual proof.
Stub outputs and scores are synthetic and must remain visibly labelled.

Model 2 has **no trainer-reported benchmark metrics**. Its local sanity check
must not be presented as mAP or a general detection guarantee. Local QA matching
is a heuristic; a lenient match can over-credit a verbose or contradictory answer.
Small selected subsets do not estimate full-benchmark or deployment accuracy.
Supplied full-test figures are attributed in [models.md](models.md), separately
from local checks. **METEOR was not computed for Model 3.** A full independently
reproduced benchmark protocol/results is **to be added**.

## Domain gap and preprocessing uncertainty

Training data comes from **VRSBench, LEVIR-CC and SEN12MS**. Uploaded imagery may
differ in resolution, region, season, sensor, view, band composition or processing.
Coarse imagery such as **10 m Sentinel-2** can be unsuitable for specialists trained
mostly on higher-resolution optical imagery. Upscaling cannot recover missing
objects or detail. RGB-value checks and modality heuristics do not establish
whether an image is in-distribution. Grayscale optical imagery may be guessed SAR;
users can override the guess, but an override does not validate the image content.

PNG/JPG demonstrations have no geographic registration. When georeferencing is
missing, the application reports that overlap/CRS checks were skipped; it cannot
prove that two optical images show the same area. Display contrast enhancement
is separate from model-native preparation. In particular, Chg2Cap's training
normalization is still **unconfirmed**; the configured upstream default is not
proof of how the supplied checkpoint was trained. Trainer confirmation is
**to be added**.

The empty domain-gap sample folder means that test has **not been run yet**.
Domain-gap samples and conclusions are **to be added**; an empty test is not a pass.

## Routing, resources and deployment

**One model per query is deliberate.** There is no automatic ensemble, cross-model
agreement check or attempt to run all four specialists. The rules use input
combination and keywords, with an optional LLM fallback hook disabled by default.
Ambiguous optical queries can fall back to QA; heuristic routing is not a
validated understanding of every user intent. Contradictory or unsupported
requests can still receive a limited task output plus an explanation.

The background queue and exclusive model lock serialize specialists within one
backend process. They do not coordinate separate servers/processes. Loading,
inference and downloads may be slow; cancelling waits for a safe boundary and
cleanup rather than interrupting a running kernel. Device fallback is limited:
grounding can retry MPS failures on CPU, while other arbitrary errors may fail
rather than silently retry. Fusion is CPU-only. Cache release does not promise
that operating-system memory immediately returns to its starting level.

Per-model deployment RAM/disk budgets, sustained-load measurements, authentication
and production deployment guidance are **to be added**. The local API and HTML
report export should not be represented as a production-certified service.

## Academic use and provenance

Datasets **VRSBench, LEVIR-CC and SEN12MS**, and the vendored **Chg2Cap code**, are
used for **academic purposes only** in this project. Preserve source attribution,
the Chg2Cap MIT licence and the upstream academic-use notice. Dataset files,
annotations, weights and derived image overlays must not be committed. Local
dataset copies are used only as test inputs; nothing is trained here. Complete
dataset licence texts and attribution bibliography are **to be added**. This
document does not replace the original licences/notices.
