# Evaluation samples (local only)

Dataset root: `/Users/sriaakarsh/Datasets`. These copies are **test inputs only**. Nothing is trained here.

Academic-use licences apply. Do not redistribute or commit dataset images/annotations or the derived box overlay. This note does not grant any additional rights; retain and follow the source dataset licences. Source data is untouched. `.gitignore` keeps sample payloads and `docs/box_check.png` out of Git while permitting these README files.

## Inspected formats and selection

VRSBench EVAL QA: 37,409/37,409 entries match Images_val (9,349 distinct images). Referring: 16,159/16,159 entries match (9,318 distinct images). Selection follows EVAL order, one row per distinct image, without using missing images.

Original box format: obj_corner = [x1,y1,x2,y2,x3,y3,x4,y4], normalized polygon vertices. Convert to absolute axis-aligned xyxy = [clip(min(x),0,1)*width, clip(min(y),0,1)*height, clip(max(x),0,1)*width, clip(max(y),0,1)*height]. No resized coordinates, axis swaps, masks, or cxcywh assumption. Retain pixel precision instead of using quantized ground_truth strings. Every EVAL ground_truth exactly equals floor(100 × the clipped normalized envelope).

5,853 EVAL polygons extend beyond image boundaries; clipping matches the EVAL encoding. Do not join objects using obj_ids: repeated IDs in EVAL rows are not a reliable per-image object identity. Use each referring row’s own phrase and polygon.

LEVIR-CC: top-level images; filename, split, filepath, sentences[].raw. 1,929/1,929 annotated test rows have both A and B locally. Only test is sampled. All references are retained as separate manifest lines. First pair test_000001.png has changeflag=0 and explicit no-change captions; the other four have changeflag=1.

Box check outside this folder: `/Users/sriaakarsh/Documents/CODES/SatQuery-AI/docs/box_check.png` derives from VRSBench validation `Images_val/P0003_0002.png` and its EVAL referring phrase/polygon. The original image occupies the top part; the caption strip adds no coordinate offset.

## Every prepared file

| Local file | Source dataset | Split | Original filename / source |
|---|---|---|---|
| `change/README.md` | Generated documentation | not applicable | scripts/prepare_samples.py; no dataset payload |
| `change/captions.txt` | LEVIR-CC | test | LevirCCcaptions.json → images[].sentences[].raw for test_000001.png, test_000004.png, test_000017.png, test_000018.png, test_000019.png |
| `change/test_000001_A.png` | LEVIR-CC | test | images/test/A/test_000001.png |
| `change/test_000001_B.png` | LEVIR-CC | test | images/test/B/test_000001.png |
| `change/test_000004_A.png` | LEVIR-CC | test | images/test/A/test_000004.png |
| `change/test_000004_B.png` | LEVIR-CC | test | images/test/B/test_000004.png |
| `change/test_000017_A.png` | LEVIR-CC | test | images/test/A/test_000017.png |
| `change/test_000017_B.png` | LEVIR-CC | test | images/test/B/test_000017.png |
| `change/test_000018_A.png` | LEVIR-CC | test | images/test/A/test_000018.png |
| `change/test_000018_B.png` | LEVIR-CC | test | images/test/B/test_000018.png |
| `change/test_000019_A.png` | LEVIR-CC | test | images/test/A/test_000019.png |
| `change/test_000019_B.png` | LEVIR-CC | test | images/test/B/test_000019.png |
| `domain_gap/README.md` | Generated documentation | not applicable | scripts/prepare_samples.py; no dataset payload |
| `fusion/README.md` | Generated documentation | not applicable | scripts/prepare_samples.py; no dataset payload |
| `grounding/P0003_0002.png` | VRSBench | validation | Images_val/P0003_0002.png |
| `grounding/P0003_0004.png` | VRSBench | validation | Images_val/P0003_0004.png |
| `grounding/P0007_0001.png` | VRSBench | validation | Images_val/P0007_0001.png |
| `grounding/P0007_0004.png` | VRSBench | validation | Images_val/P0007_0004.png |
| `grounding/P0007_0005.png` | VRSBench | validation | Images_val/P0007_0005.png |
| `grounding/P0007_0006.png` | VRSBench | validation | Images_val/P0007_0006.png |
| `grounding/P0007_0009.png` | VRSBench | validation | Images_val/P0007_0009.png |
| `grounding/P0019_0000.png` | VRSBench | validation | Images_val/P0019_0000.png |
| `grounding/P0019_0001.png` | VRSBench | validation | Images_val/P0019_0001.png |
| `grounding/P0019_0002.png` | VRSBench | validation | Images_val/P0019_0002.png |
| `grounding/README.md` | Generated documentation | not applicable | scripts/prepare_samples.py; no dataset payload |
| `grounding/boxes.json` | VRSBench | validation | VRSBench_EVAL_referring.json; question + obj_corner for the 10 listed images |
| `model1/P0003_0002.png` | VRSBench | validation | Images_val/P0003_0002.png |
| `model1/P0003_0004.png` | VRSBench | validation | Images_val/P0003_0004.png |
| `model1/P0007_0001.png` | VRSBench | validation | Images_val/P0007_0001.png |
| `model1/P0007_0004.png` | VRSBench | validation | Images_val/P0007_0004.png |
| `model1/P0007_0005.png` | VRSBench | validation | Images_val/P0007_0005.png |
| `model1/P0007_0006.png` | VRSBench | validation | Images_val/P0007_0006.png |
| `model1/P0007_0009.png` | VRSBench | validation | Images_val/P0007_0009.png |
| `model1/P0019_0000.png` | VRSBench | validation | Images_val/P0019_0000.png |
| `model1/P0019_0001.png` | VRSBench | validation | Images_val/P0019_0001.png |
| `model1/P0019_0002.png` | VRSBench | validation | Images_val/P0019_0002.png |
| `model1/P0019_0003.png` | VRSBench | validation | Images_val/P0019_0003.png |
| `model1/P0019_0004.png` | VRSBench | validation | Images_val/P0019_0004.png |
| `model1/P0019_0005.png` | VRSBench | validation | Images_val/P0019_0005.png |
| `model1/P0019_0006.png` | VRSBench | validation | Images_val/P0019_0006.png |
| `model1/P0019_0007.png` | VRSBench | validation | Images_val/P0019_0007.png |
| `model1/P0019_0009.png` | VRSBench | validation | Images_val/P0019_0009.png |
| `model1/P0019_0012.png` | VRSBench | validation | Images_val/P0019_0012.png |
| `model1/P0019_0016.png` | VRSBench | validation | Images_val/P0019_0016.png |
| `model1/P0019_0017.png` | VRSBench | validation | Images_val/P0019_0017.png |
| `model1/P0019_0020.png` | VRSBench | validation | Images_val/P0019_0020.png |
| `model1/P0019_0021.png` | VRSBench | validation | Images_val/P0019_0021.png |
| `model1/P0019_0022.png` | VRSBench | validation | Images_val/P0019_0022.png |
| `model1/P0019_0023.png` | VRSBench | validation | Images_val/P0019_0023.png |
| `model1/README.md` | Generated documentation | not applicable | scripts/prepare_samples.py; no dataset payload |
| `model1/captions.json` | VRSBench | validation | Annotations_val/P0019_0021.json; Annotations_val/P0019_0022.json; Annotations_val/P0019_0023.json |
| `model1/questions.json` | VRSBench | validation | VRSBench_EVAL_vqa.json; image/question/ground_truth fields, 20 matched rows; no prompt suffix added |
| `preparation_summary.json` | Generated audit | VRSBench validation / LEVIR-CC test | Counts from EVAL JSONs and LevirCCcaptions.json; scripts/prepare_samples.py |
| `source_tree.txt` | Local dataset inventory | all installed splits listed; only validation/test copied | /Users/sriaakarsh/Datasets (inventory only) |
| `README.md` | Generated provenance documentation | not applicable | scripts/prepare_samples.py |
