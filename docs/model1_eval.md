# Model 1: small-sample VRSBench QA

This is a selected validation sanity check, not the full VRSBench benchmark. No training or weight changes. Exact match folds case, Unicode and whitespace only. Lenient match removes punctuation/articles, maps English zero–ten to digits and accepts a contiguous whole-token reference phrase within the answer. This heuristic can over-credit verbose/contradictory answers; inspect each row. Errors count as misses. The original question is passed unchanged; the adapter adds exactly one " Answer with a short phrase." suffix. Each request lazy-loads and unloads; timings include loading, inference and cleanup.

Run timestamp: 2026-11-11T09:18:58.525398+00:00

## Results

- **samples**: 20
- **errors**: 0
- **exact_match**: 8/20 (40.0%)
- **lenient_match**: 8/20 (40.0%)
- **total_seconds**: 167.727

Environment: torch 2.15.0, torchvision 0.29.2, transformers 5.22.2, peft 0.21.2, pycocoevalcap 1.2

| # | Image | Question | Reference | Prediction | Exact | Lenient | Seconds | Error |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | P0003_0002.png | What color are the large vehicles seen in the image? | Yellow | Yellow | True | True | 13.405 |  |
| 2 | P0003_0004.png | How many vehicles are visible in the image? | 3 | 2 | False | False | 9.748 |  |
| 3 | P0007_0001.png | How many small vehicles are visible? | 1 | 1 | True | True | 7.406 |  |
| 4 | P0007_0004.png | What is the predominant color of the large vehicles? | white | White | True | True | 10.504 |  |
| 5 | P0007_0005.png | How many large vehicles are visible? | 4 | 2 | False | False | 7.421 |  |
| 6 | P0007_0006.png | How many large vehicles are in the image? | 5 | 10 | False | False | 7.652 |  |
| 7 | P0007_0009.png | What type of vehicles are mainly visible in the image? | large-vehicle | Small-vehicle | False | False | 7.364 |  |
| 8 | P0019_0000.png | What type of area is visible in the image? | residential | rural | False | False | 6.967 |  |
| 9 | P0019_0001.png | Is there a vehicle present in the image? | Yes | Yes | True | True | 7.788 |  |
| 10 | P0019_0002.png | How many harbors are visible in the image? | 3 | 2 | False | False | 7.006 |  |
| 11 | P0019_0003.png | What is the primary natural feature shown in the image? | Body of water | Waterbody | False | False | 8.161 |  |
| 12 | P0019_0004.png | How many harbors are visible in the image? | 2 | 2 | True | True | 9.666 |  |
| 13 | P0019_0005.png | How many small vehicles can be seen in the image? | 2 | 1 | False | False | 7.503 |  |
| 14 | P0019_0006.png | How many small vehicles are clearly visible in the image? | 4 | 1 | False | False | 8.291 |  |
| 15 | P0019_0007.png | What color is the small vehicle at the top-right position? | White | White | True | True | 9.118 |  |
| 16 | P0019_0009.png | How many harbors are visible in the image? | 4 | 2 | False | False | 7.485 |  |
| 17 | P0019_0012.png | How many harbors are visible in the image? | 3 | 2 | False | False | 8.076 |  |
| 18 | P0019_0016.png | What object is visible on the driveway? | small-vehicle | Vehicle | False | False | 8.36 |  |
| 19 | P0019_0017.png | How many harbors are visible in the image? | 2 | 2 | True | True | 8.15 |  |
| 20 | P0019_0020.png | How many harbors are visible in the image? | 2 | 2 | True | True | 7.656 |  |
