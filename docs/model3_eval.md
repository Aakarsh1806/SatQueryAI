# Model 3: small LEVIR-CC test subset

This selected five-pair subset is not the benchmark. Full-test-split scores supplied by the user: **BLEU-4 0.542, ROUGE-L 0.727, CIDEr 1.29; METEOR not computed**. Those full-split scores were not reproduced here. A=before, B=after. All raw reference captions are grouped per pair, so each pair is inferred once. Default upstream normalization remains UNCONFIRMED: RGB 0–255, resized 256×256, mean [100.679,99.5023,84.9932], std [50.982,48.4838,44.7057]; no percentile stretch or /255. Optional metrics use pycocoevalcap, lowercase alphanumeric/apostrophe regex tokens (not Java PTB tokenization). CIDEr document frequencies are fitted to these five reference groups only; all scores are sensitive to this tiny selection and are not directly comparable to the full-split numbers. Error outputs are empty captions in the denominator. No confidence or change map is inferred. Timings include loading and cleanup.

Run timestamp: 2026-11-11T09:20:33.688660+00:00

## Results

- **pairs**: 5
- **references**: 25
- **errors**: 0
- **BLEU-4**: 0.2701829278676448
- **ROUGE-L**: 0.537653285581111
- **CIDEr**: 0.6845100418628902
- **METEOR**: not computed
- **all_captions_identical**: False
- **total_seconds**: 8.971

Environment: torch 2.15.0, torchvision 0.29.2, transformers 5.22.2, peft 0.21.2, pycocoevalcap 1.2

| # | Before | After | Prediction | All references | Concerns | Seconds | Error |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | test_000001_A.png | test_000001_B.png | the two scenes seem identical |  there is no difference . \|  the two scenes seem identical . \|  the scene is the same as before . \|  no change has occurred . \|  almost nothing has changed . |  | 2.63 |  |
| 2 | test_000004_A.png | test_000004_B.png | the woods are removed and a t-shaped road with villas built along the roads |  the vegetation has been replaced by a road and many villas around . \|  trees are removed and an arc road is built with some buildings around . \|  a winding road is built and many houses are constructed beside it to replace the former vegetation . \|  forest is taken by the houses along the road . \|  the former forest has been replaced by a residential area with roads and houses . |  | 1.585 |  |
| 3 | test_000017_A.png | test_000017_B.png | the plants are removed and a residential area with many neatly arranged houses are built along |  a road appears at the bottom and many houses are scattered replacing the trees . \|  trees are removed and neatly arranged residential buildings with a circular road at the top of the area and a turning road through the buildings . \|  a road is built and many villas are constructed neatly on both sides of it . \|  a forest is built into residential areas and roads . \|  the desert has turned into a residential area with road which houses are built alongside . |  | 1.618 |  |
| 4 | test_000018_A.png | test_000018_B.png | the plants are removed and a residential area with many villas built along the roads |  the road has been rebuilt and the plants have been replaced by many villas . \|  trees and the road are replaced by a new road with neatly arranged residential buildings and another circular road at the top . \|  a road is built and many villas are constructed on both sides of it and the vegetation is also replaced by houses . \|  a large number of trees and space are replaced by a residential area and road . \|  the woods with a road going through have turned into the residential area with roads and houses . |  | 1.582 |  |
| 5 | test_000019_A.png | test_000019_B.png | the vegetation is replaced by the road |  the trees have been replaced by many buildings . \|  trees are replaced by residential buildings and the old arc dirt road is rebuilt . \|  a winding road is built and many houses are constructed around it to replace the former vegetation . \|  houses and roads take place of forest and spare ground . \|  the road is reconstructed with houses built alongside . |  | 1.556 |  |
