# Model 2: 10-sample grounding sanity check

The trainer supplied no benchmark accuracy metrics. This local 10-sample sanity check is not a benchmark. IoU is the best overlap across all retained predicted boxes and ground-truth boxes for each prompt (oracle best-box selection, not highest-score selection). No predictions means IoU=0; errors count as misses. Acc@0.5 is the fraction of prompts whose best IoU is at least 0.5. This does not penalize extra false-positive boxes and is not mAP, precision or recall. Both prediction and reference coordinates use original absolute pixels. Box threshold=0.3, text threshold=0.25; scores are uncalibrated. Green boxes are reference; red boxes are predictions. Timings include lazy load, inference, cleanup and example rendering.

Run timestamp: 2026-11-11T09:20:15.502370+00:00

## Results

- **samples**: 10
- **errors**: 0
- **mean_IoU**: 0.36726
- **Acc@0.5**: 4/10 (40.0%)
- **total_seconds**: 13.345

Environment: torch 2.15.0, torchvision 0.29.2, transformers 5.22.2, peft 0.21.2, pycocoevalcap 1.2

| # | Image | Phrase | Best IoU | IoU ≥ 0.5 | Predicted boxes | Seconds | Overlay | Error |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | P0003_0002.png | The large yellow vehicle situated closest to the green area. | 0.8498768248413697 | True | 1 | 4.248 | [view](model2_examples/01_P0003_0002.png) |  |
| 2 | P0003_0004.png | The large vehicle is situated towards the middle-left side of the image. | 0.0 | False | 2 | 1.121 | [view](model2_examples/02_P0003_0004.png) |  |
| 3 | P0007_0001.png | The small vehicle located towards the top of the image. | 0.8963862530275882 | True | 1 | 0.827 | [view](model2_examples/03_P0007_0001.png) |  |
| 4 | P0007_0004.png | The large vehicle is at the left edge of the road. | 0.25611714516854217 | False | 1 | 1.203 | [view](model2_examples/04_P0007_0004.png) |  |
| 5 | P0007_0005.png | The large vehicle located on the right side. | 0.0 | False | 0 | 1.225 | [view](model2_examples/05_P0007_0005.png) |  |
| 6 | P0007_0006.png | The large vehicle is located at the right edge extending beyond the image frame. | 0.0 | False | 0 | 0.838 | [view](model2_examples/06_P0007_0006.png) |  |
| 7 | P0007_0009.png | The large vehicle located at the far right top of the image. | 0.0 | False | 0 | 1.215 | [view](model2_examples/07_P0007_0009.png) |  |
| 8 | P0019_0000.png | The vehicle is parked on a driveway, located near the bottom-middle of the image. | 0.8227381930778588 | True | 1 | 0.967 | [view](model2_examples/08_P0019_0000.png) |  |
| 9 | P0019_0001.png | The large vehicle is parked on a driveway to the left of the house. | 0.8474775849054927 | True | 1 | 0.846 | [view](model2_examples/09_P0019_0001.png) |  |
| 10 | P0019_0002.png | The harbor located near the top right corner of the image extending into the water. | 0.0 | False | 0 | 0.855 | [view](model2_examples/10_P0019_0002.png) |  |
