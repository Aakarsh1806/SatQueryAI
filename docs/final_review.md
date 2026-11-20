# Final review and demo readiness

> Current configuration update: the user enabled Model 4 real mode before sample pairs arrived. It is still not verified on a real pair. The earlier stub-only walkthrough/check results below are historical; do not describe them as a real-pair validation.

Review date: **11 November 2026 (Asia/Kolkata)**. No deployment or weight edits.
Current saved modes: **Models 1–3 real, Model 4 stub**. Prerequisite checks can
still produce a labelled stub fallback on another machine; inspect `/api/models`
and the actual result tag. Real mode is not a claim of general accuracy.

**Model 4 is implemented but remains a stub until genuine aligned pairs arrive;
it is not yet verified on a real pair.** No real fusion demo or sample was invented.

## Final model summary

| Model | Saved mode | Validation status | Measured numbers versus supplied figures | Remaining risks |
|---|---|---|---|---|
| RS-VLM | real | Small local VRSBench validation sanity check; full benchmark not supplied | Local 20-question exact match **40%**, lenient match **40%** | Hallucinated objects/source details, miscounts, domain gap; generated-token score uncalibrated; no spatial evidence; no full QA/caption benchmark |
| Grounding DINO Tiny | real | **Unvalidated**; no trainer benchmark metrics | Local 10-prompt sanity check **mean IoU 0.3673**, **Acc@0.5 40%**; not mAP/general accuracy | Prompt-sensitive; missed/wrong boxes and false positives; best-box metric over-credits extra boxes; no masks or aggregate confidence |
| Chg2Cap | real | Local test-pair execution checked; training normalization still unconfirmed | Local five-pair subset: **BLEU-4 0.270183, ROUGE-L 0.537653, CIDEr 0.684510**. User-reported full test: **0.542 / 0.727 / 1.29**, respectively; **METEOR not computed** | Generic/incomplete captions, road-shape mistakes, buildings/roads domain; description only; no change map/direct QA/confidence; subset tokenization/IDF differ from full benchmark |
| Optical/SAR fusion | **stub** | **Not verified on a real pair; no real pairs available** | No local real-pair measurements. Handover-reported **accuracy 76.15%, macro-F1 62.89%**; optical-only **57.34%** accuracy, SAR-only **24.22%** | Summer-only, 8 classes; weak Grassland/Forest/Wetlands; strict raw aligned inputs needed; softmax uncalibrated; placeholder scores are not predictions |

Local metric sources: registry and `model1_eval.json`, `model2_eval.json`,
`model3_eval.json`; definitions/selection caveats remain in those reports.
Supplied full-test/baseline figures come from the user/inventory, not this demo.
The table does not convert checkpoint recovery, load success or a demo answer
into benchmark validation.

## Clean-start checks

`scripts/rehearse_demo.py --with-tests` created a new private profile under
`.demo-rehearsal/`, with a fresh SQLite database and fresh uploads/traces. It
confirmed empty history before requests. The full suite used its own fresh
`tmp_path` DB/upload fixtures; unit/stub tests explicitly select stubs rather
than accidentally invoking production-configured real models. Optional real
adapter/sample tests still exercise real execution when dependencies exist.

**Full suite: 117 passed, 1 skipped, 0 failed (32.32 s).** The skip is the absent
genuine aligned Model 4 parity pair/expected labels. Frontend: **3 tests passed**,
production build passed. A later targeted API/Agent/model/portability run after
final warning/catalog changes also passed: **56 passed**; see the private verification log.
PostgreSQL's earlier local switch verification is preserved separately in
[postgres_check.md](postgres_check.md); this pass does not relabel its older
counts as a new PostgreSQL run.

All named live API scenarios were run from that clean state with playback off.
The raw request/result/trace data is in [demo_rehearsal.json](demo_rehearsal.json).

| Scenario | Files under data_samples/ | Observed outcome | POST-to-terminal seconds |
|---|---|---|---:|
| Model 1 QA | `model1/P0003_0002.png` | Live real answer **Yellow**; no evidence; uncalibrated token score | 10.029 |
| Model 1 caption | `model1/P0019_0021.png` | Live real caption about buildings/vegetation, including an unsupported water suggestion; no evidence | 11.759 |
| Model 2 grounding | `grounding/P0003_0002.png` | `Detect vehicles` produced **2 candidate boxes**; still unvalidated | 1.497 |
| Model 3 change | `change/test_000004_A.png`, `change/test_000004_B.png` | Live caption: “the woods are removed and a t-shaped road with villas built along the roads”; description-only note, confidence unavailable | 1.814 |
| One-image change failure | `model1/P0003_0002.png` | Correct rejection: needs two before/after optical images; no result | 0.061 |
| Model 4 status | No pairs supplied | Registry **stub**; no real-pair attempt or real prediction | Not run |

These are individual local wall times, including preparation/loading/cleanup and
polling; they are not latency guarantees. Current configured device was MPS for
Models 1–3. Traces confirm unload after each live request.

## UI and error-state review

A second fresh profile from `scripts/start_demo.sh` was exercised in the actual
browser. It started with empty history. All five live scenarios above were run
through upload → inspect → submit → poll → render, and Model readiness showed
real/real/real/stub. The UI rendered grounded per-box scores with “uncalibrated”
and “unvalidated,” the change input slider and description-only note, unavailable
confidence/evidence where appropriate, the expected rejected route, and the
persistent Model 4 caveat on workspace and history.

Observed edge states:

| State | How checked | Result |
|---|---|---|
| Empty history | Fresh real UI profile | Clear empty-history message and disabled pagination |
| Slow/loading request | Live loading plus controlled fixture | Progress/elapsed time and no premature answer |
| Model download state | Explicitly **simulated** disposable backend fixture | Visible “SIMULATED…no actual network transfer” message; no live-success claim |
| Cancellation | Cancelled that fixture while pending | “Waiting for active operation/cleanup,” then cancelled with no result; no immediate-abort promise |
| Model load failure | Explicitly simulated incompatible checkpoint, not damaged real weights | Failed/no-result message, load reason and trace; no synthetic successful replacement |
| Rejected upload | Selected `data_samples/README.md` | Clear “Use GeoTIFF/TIFF, PNG, or JPG image files” reason |
| Pre-computed fallback | Four actual saved live results replayed through an isolated fresh API with model construction forbidden | All labelled `pre-computed` in API/trace/report, no model loaded; no Model 4 entry |

Actual first-time network downloading was **not repeated** because the base cache
was already complete. The download-state UI check is not a network/download-speed
measurement. Fixture states are not real model failures or real predictions.
Browser and fixture records are in [demo_ui_rehearsal.json](demo_ui_rehearsal.json);
playback verification is in [demo_playback_check.json](demo_playback_check.json).
Screenshots under `docs/screenshots/final-demo-*.png` contain academic-use pixels
and are ignored. Review servers are stopped after verification; use the launcher
for the presentation.

## Problems found and actions

- **False readiness implication:** the UI used “Local check passed” whenever no
  fallback reason existed. It now says prerequisites are found and explicitly
  denies a quality-validation implication. Model-list failures are visible.
- **Missing persistent caveats:** workspace/history now always show grounding's
  unvalidated status and Model 4's stub/no-real-pair status. Reports include the
  actual result's validation status. Supplied Model 3/4 scores are attributed in
  registry text instead of implying locally reproduced benchmarks.
- **Empty detection state:** zero boxes now means detection evidence and aggregate
  confidence are unavailable; it is not displayed as a successful proof.
- **Ambiguous progress/failure wording:** stubs explicitly load no model weights;
  pre-computed integration no longer implies model unload/cache work happened.
  Load errors have a distinct `model_load` trace stage and a clear no-result message.
  The old unit assertion expecting a generic stage was updated to that contract.
- **Synthetic change/fusion claims:** the change stub calls its description a
  synthetic example, not an observation. Fusion stub warnings call it a placeholder
  with no real-pair verification; placeholder class names match the saved class list.
- **Model 1 caption quality:** the generated water detail is unsupported by the
  reference and is not apparent in the image. The output is retained, not edited;
  generated-text hallucination warnings and an explicit presenter caveat were added.
- **Grounding sensitivity:** the initial long “Locate the large yellow vehicle
  situated closest to the green area.” query yielded zero boxes. The miss is saved
  in [demo_grounding_initial_miss.json](demo_grounding_initial_miss.json). A later
  short `Detect vehicles` prompt returned two candidate boxes. This exploratory
  presentation-prompt selection is **not a blind metric improvement or benchmark**.
  Thresholds and weights were not changed.
- **Change-caption detail:** the caption's “t-shaped” road differs from the winding/
  arc descriptions in references. This is called out in the walkthrough rather
  than hidden. Normalization remains unconfirmed; no tuning was performed here.

The prepared catalogue now includes QA, caption, grounding and the actual changed
pair used in the walkthrough. It was packaged from measured live rehearsal outputs,
not fabricated. Its caption keeps the same unsupported detail. Pre-computed does
not mean correct, and playback never certifies a model's quality.

## Open TODOs

- **Model 4 pairs:** put genuine `<id>_s2.tif` / `<id>_s1.tif` pairs and
  `expected_labels.json` in `data_samples/fusion/`; verify raw units, band order,
  256×256 size and georeferencing; run `scripts/eval_model4.py`; inspect parity,
  then switch to real and rerun routing/demo checks. Keep stub until this is done.
- Obtain trainer benchmark metrics/protocols for Model 2. It remains unvalidated;
  improve robustness on missed/wrong boxes rather than claiming the demo proves accuracy.
- Obtain Model 1 full validation/caption metrics and a larger representative local
  evaluation; address hallucinations/counting failures. Calibrated scoring is absent.
- Confirm Chg2Cap's original training normalization and scoring/tokenization
  protocol. Full-test reproduction and METEOR, if required, are **to be added**.
- Add genuine domain-gap inputs; the current empty domain-gap test is not a pass.
- Record target-machine free disk and peak RAM budgets; resource estimates remain
  **to be added**. Pre-download the base model before the event; actual fresh
  download/network-failure checks remain separate from simulated UI states.
- Validate optional ML-enabled Linux/GPU containers if they will be used; only the
  lightweight container was checked in the portability pass. Native Windows
  supervisor verification and production hardening remain **to be added**.
- Complete licence/attribution bibliography and apply the slide/spec checklist to
  the actual deck. Do not commit datasets, weights, demo payloads or derived images.

Use [demo_script.md](demo_script.md) for the timed walkthrough and slow/wrong-output
wording, and [release_checklist.md](release_checklist.md) for preparation gates.
