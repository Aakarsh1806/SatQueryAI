# Routing and input-validation evaluation

Accuracy: **53/53 = 100.0%**. This is deterministic rule/input-contract coverage, not model accuracy or evidence of general routing quality.

No specialist is loaded. Fusion format fixtures are generated in a temporary directory solely to exercise the production raw-input validator; they are not Model 4 test samples. Model 4 remains stub. This directly scores the real fusion input contract, not the permissive synthetic stub demonstration path. The intent router chooses fusion from optical+SAR, then strict input validation rejects PNG/RGB/wrong-size/unregistered pairs. Ambiguous single-optical queries use the documented default QA route; no LLM fallback is enabled. A road-location query yields boxes, not a road trajectory.

Fixed during evaluation: "show me a summary" initially selected grounding because "show" matched but the caption rule omitted the noun "summary". Caption precedence now includes "summary" (including uppercase). The initial run had 51/53 matches; these two cases failed. No expected labels were changed.

## Wrong routes in the final run

None.

## Expected versus actual

| # | Inputs | Format | Query | Expected | Actual | Correct | Reason |
|---|---|---|---|---|---|---|---|
| 1 | optical | not applicable | What is visible? | qa | qa | True | One optical image with no caption/location keyword defaults to short-phrase QA. |
| 2 | optical | not applicable | How many buildings are visible? | qa | qa | True | One optical image with no caption/location keyword defaults to short-phrase QA. |
| 3 | optical | not applicable | Is there water? | qa | qa | True | One optical image with no caption/location keyword defaults to short-phrase QA. |
| 4 | optical | not applicable | What colour is the roof? | qa | qa | True | One optical image with no caption/location keyword defaults to short-phrase QA. |
| 5 | optical | not applicable | Describe this image | caption | caption | True | An explicit describe/caption/summarize request selects the fixed caption prompt. |
| 6 | optical | not applicable | Caption this scene | caption | caption | True | An explicit describe/caption/summarize request selects the fixed caption prompt. |
| 7 | optical | not applicable | Summarize this area | caption | caption | True | An explicit describe/caption/summarize request selects the fixed caption prompt. |
| 8 | optical | not applicable | Summarise the image | caption | caption | True | An explicit describe/caption/summarize request selects the fixed caption prompt. |
| 9 | optical | not applicable | show me a summary | caption | caption | True | An explicit describe/caption/summarize request selects the fixed caption prompt. |
| 10 | optical | not applicable | Describe where the road is | caption | caption | True | An explicit describe/caption/summarize request selects the fixed caption prompt. |
| 11 | optical | not applicable | SHOW ME A SUMMARY | caption | caption | True | An explicit describe/caption/summarize request selects the fixed caption prompt. |
| 12 | optical | not applicable | where does the road go | grounding | grounding | True | A location/detection keyword selects grounding (boxes only, no masks). |
| 13 | optical | not applicable | Where are storage tanks? | grounding | grounding | True | A location/detection keyword selects grounding (boxes only, no masks). |
| 14 | optical | not applicable | Locate the bridge | grounding | grounding | True | A location/detection keyword selects grounding (boxes only, no masks). |
| 15 | optical | not applicable | Find ships | grounding | grounding | True | A location/detection keyword selects grounding (boxes only, no masks). |
| 16 | optical | not applicable | Highlight buildings | grounding | grounding | True | A location/detection keyword selects grounding (boxes only, no masks). |
| 17 | optical | not applicable | Show roads | grounding | grounding | True | A location/detection keyword selects grounding (boxes only, no masks). |
| 18 | optical | not applicable | Detect vehicles | grounding | grounding | True | A location/detection keyword selects grounding (boxes only, no masks). |
| 19 | optical | not applicable | DETECT water | grounding | grounding | True | A location/detection keyword selects grounding (boxes only, no masks). |
| 20 | optical | not applicable | Wherever possible, count buildings | qa | qa | True | One optical image with no caption/location keyword defaults to short-phrase QA. |
| 21 | optical | not applicable | Is this a showcase building? | qa | qa | True | One optical image with no caption/location keyword defaults to short-phrase QA. |
| 22 | optical | not applicable | Is smoke detectable? | qa | qa | True | One optical image with no caption/location keyword defaults to short-phrase QA. |
| 23 | optical | not applicable | What about this? | qa | qa | True | One optical image with no caption/location keyword defaults to short-phrase QA. |
| 24 | optical | not applicable | Help | qa | qa | True | One optical image with no caption/location keyword defaults to short-phrase QA. |
| 25 | optical | not applicable | Classify the land cover | qa | qa | True | One optical image with no caption/location keyword defaults to short-phrase QA. |
| 26 | optical+optical | not applicable | What changed? | change_caption | change_caption | True | Two optical images select Chg2Cap (A=before, B=after). This model produces a description of changes, not a direct answer to change questions or a change map. |
| 27 | optical+optical | not applicable | How many buildings were added? | change_caption | change_caption | True | Two optical images select Chg2Cap (A=before, B=after). This model produces a description of changes, not a direct answer to change questions or a change map. |
| 28 | optical+optical | not applicable | Describe differences | change_caption | change_caption | True | Two optical images select Chg2Cap (A=before, B=after). This model produces a description of changes, not a direct answer to change questions or a change map. |
| 29 | optical+optical | not applicable | Find new roads | change_caption | change_caption | True | Two optical images select Chg2Cap (A=before, B=after). This model produces a description of changes, not a direct answer to change questions or a change map. |
| 30 | optical+optical | not applicable | Generate a change map | change_caption | change_caption | True | Two optical images select Chg2Cap (A=before, B=after). This model produces a description of changes, not a direct answer to change questions or a change map. |
| 31 | optical+optical | not applicable | Describe this image | change_caption | change_caption | True | Two optical images select Chg2Cap (A=before, B=after). This model produces a description of changes, not a direct answer to change questions or a change map. |
| 32 | optical+sar | valid | Classify land cover | classification | classification | True | An optical+SAR pair selects the land-cover classifier; it cannot provide free-form answers. |
| 33 | sar+optical | valid | What is this? | classification | classification | True | An optical+SAR pair selects the land-cover classifier; it cannot provide free-form answers. |
| 34 | optical+sar | valid | Where is the forest? | classification | classification | True | An optical+SAR pair selects the land-cover classifier; it cannot provide free-form answers. |
| 35 | optical+sar | valid | Describe the area | classification | classification | True | An optical+SAR pair selects the land-cover classifier; it cannot provide free-form answers. |
| 36 | optical+sar | png | Classify land cover | error | error | True | Fusion needs original GeoTIFF inputs: 13-band Sentinel-2 optical and 2-band VV/VH SAR. PNG/JPG and RGB previews are not supported. |
| 37 | optical+sar | rgb | Classify land cover | error | error | True | Optical GeoTIFF needs exactly 13 bands in Sentinel-2 order; received 3. |
| 38 | optical+sar | wrong_size | Classify land cover | error | error | True | Fusion needs two 256x256 patches. No automatic resizing or resampling is performed. |
| 39 | optical+sar | mismatched_grid | Classify land cover | error | error | True | Fusion inputs must have identical CRS, affine transform, width and height. Provide an already aligned pair; no reprojection or resampling is performed. |
| 40 | optical+sar | no_crs | Classify land cover | error | error | True | Both fusion GeoTIFFs need non-null CRS metadata and identical georeferencing. |
| 41 | sar | not applicable | Describe this image | error | error | True | SAR-only input is not supported; provide an optical image and a SAR image for land-cover classification. |
| 42 | sar | not applicable | Classify land cover | error | error | True | SAR-only input is not supported; provide an optical image and a SAR image for land-cover classification. |
| 43 | sar+sar | not applicable | Compare these images | error | error | True | SAR-only input is not supported; provide an optical image and a SAR image for land-cover classification. |
| 44 | optical+optical+optical | not applicable | Describe these images | error | error | True | Upload one or two images; this request supports at most two images. |
| 45 | optical+sar+optical | not applicable | Classify land cover | error | error | True | Upload one or two images; this request supports at most two images. |
| 46 | none | not applicable | Describe this image | error | error | True | Upload one or two images; this request supports at most two images. |
| 47 | optical | not applicable | What changed? | error | error | True | This task needs two images: upload before and after optical images for change analysis. |
| 48 | optical | not applicable | Compare before and after | error | error | True | This task needs two images: upload before and after optical images for change analysis. |
| 49 | optical | not applicable | What is the difference? | error | error | True | This task needs two images: upload before and after optical images for change analysis. |
| 50 | optical | not applicable |  | error | error | True | Enter a question or a caption request. |
| 51 | optical+optical | not applicable |     | error | error | True | Enter a question or a caption request. |
| 52 | optical+sar | not applicable |  | error | error | True | Enter a question or a caption request. |
| 53 | unknown | not applicable | Describe this image | error | error | True | Each image modality must be optical or sar. |
