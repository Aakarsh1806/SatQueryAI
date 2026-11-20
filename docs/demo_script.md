# Five-minute local demo

> Current configuration update: the user enabled Model 4 real mode before sample pairs arrived. It is still not verified on a real pair. The earlier stub-only walkthrough/check results below are historical; do not describe them as a real-pair validation.

**Current configuration:** Models 1–3 request real inference; Model 2 remains
**unvalidated**. Model 4 is **stub until real aligned pairs arrive**, not yet
verified on a real pair. One specialist runs per query. No training occurs here.

## Before the audience arrives

- Connect power, close memory-heavy apps, check free disk/RAM, install the optional
  ML environment and cache Model 1's base download in advance. Do not spend the
  presentation downloading weights. Missing requirements cause labelled stub
  fallback or a clear failed-load message; neither is a real-model success.
- From the repository root run `./scripts/start_demo.sh`. It creates a fresh
  database/uploads/traces profile without deleting existing history. Open the
  printed frontend URL and inspect Model readiness: real, real, real, stub.
- Keep `DEMO_MODE=false` for the live walkthrough. Check History is empty. Keep
  the named local files below ready in the picker; they are academic-use test inputs.
- Fallback catalogue: `scripts/prepare_demo.py --from-rehearsal docs/demo_rehearsal.json`
  packages the measured outputs already recorded, without loading models. Or use
  `scripts/prepare_demo.py` to record them again on a suitable machine. To use
  playback, stop the launcher and run `DEMO_MODE=true ./scripts/start_demo.sh`.
  The exact files/queries below match the prepared catalogue. Say **pre-computed**
  and check that tag before presenting it; unmatched requests use normal execution.
  Playback retains wrong outputs too—it is not a correction or a live run.

The time slots below are a presentation plan, not guaranteed inference latency.
Recorded timings are in [final_review.md](final_review.md). If a request consumes
its slot, use the slow-response line and move on; do not imply it completed.

## 0:00–0:30 — Set expectations

Say: “This academic assistant routes one supported task to one specialist. Some
outputs have boxes or scores, some do not. Scores are uncalibrated. Grounding is
unvalidated. Fusion is currently a stub, with no real-pair verification.”

Show the persistent caveat banner and Model readiness. Point out the REAL,
STUB or PRE-COMPUTED result label; these are different execution paths.

## 0:30–1:15 — (a) Model 1 question

- **Image 1:** `data_samples/model1/P0003_0002.png`; remove Image 2.
- **Query:** `What color are the large vehicles seen in the image?`
- **Expected routing:** RS-VLM / short-phrase QA. The adapter adds the required
  suffix; do not type it into the query yourself.
- **Observed answer:** `Yellow`. This matches this sample's reference, not a
  guarantee for other questions. Show **Evidence: not available** and the
  uncalibrated token-probability label; do not call it answer accuracy.
- **Say:** “This is live generated text; no visual evidence is produced by this model.”
- **If slow:** “The model is loading or generating. Elapsed time is shown; I can
  cancel, wait, or explicitly switch to the pre-computed rehearsal output.”
- **If wrong:** “That answer is wrong for this image. The score is uncalibrated;
  it does not validate correctness. We retain the answer and trace as observed.”

## 1:15–2:00 — (b) Model 1 caption

- **Image 1:** `data_samples/model1/P0019_0021.png`; no Image 2.
- **Query:** `Describe this image`
- **Expected routing/prompt:** RS-VLM caption;
  `Describe this satellite image in detail.`
- **Expected type:** a description, with no boxes/masks or generated spatial evidence.
  The dataset reference describes residential properties, greenery and a small vehicle.
- **Observed limitation:** the rehearsal caption described buildings/vegetation
  but also suggested a body of water. That detail is not supported by the reference
  and is not apparent in the image. Do not assert water is present. The output is
  intentionally left unchanged so the limitation is visible.
- **Say:** “The model can describe a scene, but can hallucinate details. Please
  compare the caption with the image; its confidence is not calibrated.”
- **If slow:** “Caption generation has a longer output budget. We can cancel and
  show the labelled pre-computed version instead of pretending this finished.”
- **If wrong:** “This detail is unsupported. A fluent description is not evidence.”

## 2:00–2:45 — (c) Model 2 grounding

- **Image 1:** `data_samples/grounding/P0003_0002.png`; no Image 2.
- **Query:** `Detect vehicles`
- **Expected routing:** Grounding DINO, boxes only. The final API rehearsal returned
  **two candidate boxes**; future runs can differ or return none. Inspect positions
  rather than presenting the count as an annotation-verified result.
- **Say:** “These are candidate boxes with uncalibrated per-box scores, not masks
  or proof. The model is unvalidated; the trainer supplied no benchmark metrics.”
- **Show:** overlay, labels/scores, no aggregate answer confidence, unvalidated label.
- **If slow:** “Grounding is still loading/running. I can wait or use the explicitly
  labelled pre-computed result; neither path creates a benchmark claim.”
- **If wrong/empty:** “This is a missed or incorrect detection. No boxes means no
  detection evidence is available at this threshold.” The initial longer query
  `Locate the large yellow vehicle situated closest to the green area.` returned
  **zero boxes**; that miss is retained in the review. The shorter prompt was chosen
  for presentation after that exploratory check, not as a blind accuracy test.

## 2:45–3:30 — (d) Model 3 change description

- **Image 1:** `data_samples/change/test_000004_A.png` (before).
- **Image 2:** `data_samples/change/test_000004_B.png` (after).
- **Query:** `What changed between these images?`
- **Expected routing:** Chg2Cap. The limitation note says it describes changes,
  rather than directly answering a change question or generating a map.
- **Observed caption:** `the woods are removed and a t-shaped road with villas built along the roads`
- **Say:** “It describes vegetation removal and construction. The road-shape detail
  may be wrong: references describe an arc/winding road. The slider shows the
  input pair, not a change map. Confidence and model-generated visual evidence
  are not available. Training normalization still needs confirmation.”
- **If slow:** “The caption is pending; we can cancel or use labelled playback.”
- **If wrong:** “This is an imperfect text description, not a change map or a
  factual answer to every question. We keep the original output and references.”

## 3:30–4:00 — (e) Routing failure

- **Image 1 only:** `data_samples/model1/P0003_0002.png`; remove Image 2.
- **Query:** `What changed between these images?`
- **Expected:** failed request, no model answer; message:
  `This task needs two images: upload before and after optical images for change analysis.`
- **Say:** “The Agent refuses unsupported input instead of inventing a result.”
- **If slow:** “Validation has not returned yet; we will not claim the request passed.”
- **If behavior differs:** “This does not meet the expected validation contract;
  inspect the trace before continuing.” Never substitute a successful result.

## 4:00–4:30 — Model 4: STUB STATUS ONLY

**No sample files exist in `data_samples/fusion/`; do not invent a pair or show
an optical PNG as a real fusion test.** Expand Model readiness and show `stub`.
Say: “The fusion adapter is implemented, but real aligned optical/SAR pairs are
not available. It is a stub until pairs arrive and is not yet verified on a real
pair. Any stub class/scores are placeholders, not predictions. Reported training
metrics are not this machine's real-pair verification.”

No real inference is expected in this step. If a stub/placeholder is slow or
returns a class, explain that it is a synthetic placeholder—not a verified result.
If readiness unexpectedly says real, stop this step and inspect configuration;
do not claim verification. There is no default Model 4 pre-computed result.

## 4:30–5:00 — Trace, report and close

Open the execution trace and show chosen model/version/mode, parameters,
warnings and unload state. Playback traces explicitly say `pre-computed` and
record source provenance rather than claiming a current model load. Download
an HTML report if time allows. History stores failed/cancelled requests too.

Say: “These are limited academic models with domain gaps. Local small-sample
checks and supplied benchmark scores are different. Model 2 remains unvalidated,
Model 4 remains a stub, and all supported confidence scores are uncalibrated.”

## Edge-state rehearsal outside the five minutes

- Empty history: a new `start_demo.sh` profile shows the empty-history message.
- Rejected upload: select a non-image file; inspect the reason and replace it.
- Failed load: show the explicit failed-load/no-result message and saved trace.
  Do not corrupt real weights to demonstrate this; use a disposable test fixture.
- Download/slow loading: the UI shows progress/elapsed time without an answer.
  Existing cached weights mean a real fresh network download was not repeated;
  the download-state UI was exercised with a **clearly simulated** test fixture.
- Cancel: confirm cancellation waits for the active operation/cleanup, then shows
  `cancelled` with no result. It is not an immediate kernel/download abort.
- Playback: use exact matching files/query and verify the PRE-COMPUTED tag. Do not
  describe it as live, even if the original source result was produced by a real model.
