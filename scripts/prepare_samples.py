"""Prepare small, provenance-tracked evaluation inputs; never train or modify datasets.

Only the inspected local VRSBench and LEVIR-CC schemas are supported. Unknown
schemas or inconsistent box encodings stop preparation before files are copied.
"""
import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import re
import shutil
import textwrap

from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
FUSION_NOTE = ('Model 4 samples will be added later: pairs <id>_s2.tif (13 bands) and '
               '<id>_s1.tif (2 bands, dB), 256x256, matching georeferencing, plus '
               'expected_labels.json')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read_json(path):
    return json.loads(path.read_text(encoding='utf-8'))


def image_path(folder, filename):
    require(isinstance(filename, str) and Path(filename).name == filename,
            f'Unsupported image filename: {filename!r}')
    return folder / filename


def envelope(row):
    """Exact polygon envelope in normalized xyxy; independently check EVAL encoding."""
    corners = row.get('obj_corner')
    require(isinstance(corners, list) and len(corners) == 8 and
            all(isinstance(n, (int, float)) and math.isfinite(n) for n in corners),
            f'Unknown obj_corner format in {row.get("image_id")}: {corners!r}')
    x, y = corners[::2], corners[1::2]
    clipped = [max(0., min(1., n)) for n in (min(x), min(y), max(x), max(y))]
    require(clipped[0] < clipped[2] and clipped[1] < clipped[3],
            f'Empty or invalid box in {row.get("image_id")}')
    encoded = row.get('ground_truth')
    require(isinstance(encoded, str) and re.fullmatch(r'\{<\d+><\d+><\d+><\d+>\}', encoded) is not None,
            f'Unknown referring ground_truth encoding: {encoded!r}')
    actual = [int(n) for n in re.findall(r'<(\d+)>', encoded)]
    expected = [int(n * 100) for n in clipped]
    require(actual == expected,
            f'Box format inconsistent for {row["image_id"]}: polygon envelope {expected}, EVAL {actual}; stop rather than guess.')
    return clipped


def distinct(rows, count, excluded=()):
    selected, seen = [], set(excluded)
    for row in rows:
        if row['image_id'] not in seen:
            seen.add(row['image_id'])
            selected.append(row)
            if len(selected) == count:
                return selected
    raise ValueError(f'Need {count} distinct images, found {len(selected)}')


def digest(path):
    with path.open('rb') as file:
        return hashlib.file_digest(file, 'sha256').hexdigest()


def copy_image(source, target):
    if target.exists():
        require(digest(source) == digest(target), f'Refusing to replace different existing file: {target}')
    else:
        shutil.copyfile(source, target)


def prepare(datasets, output, box_check):
    vrs, levir = datasets/'VRSBench', datasets/'LEVIR-CC'
    images = vrs/'Images_val'
    vqa = read_json(vrs/'VRSBench_EVAL_vqa.json')
    referring = read_json(vrs/'VRSBench_EVAL_referring.json')
    require(isinstance(vqa, list) and isinstance(referring, list), 'VRSBench EVAL files must be JSON lists.')
    matched_qa = [row for row in vqa if image_path(images, row['image_id']).is_file()]
    matched_ref = [row for row in referring if image_path(images, row['image_id']).is_file()]
    require(matched_qa and matched_ref, 'No matching QA or referring image entries; no boxes can be prepared.')
    for row in matched_qa:
        require(all(isinstance(row.get(k), str) and row[k].strip() for k in ('question', 'ground_truth')),
                f'Unknown VQA format: {row!r}')
    for row in matched_ref:
        require(isinstance(row.get('question'), str) and row['question'].strip(), 'Missing referring phrase')
        envelope(row)
    qa_rows = distinct(matched_qa, 20)
    ref_rows = distinct(matched_ref, 10)
    # Three additional optical images with genuine annotation captions.
    caption_rows = distinct(matched_qa, 3, [row['image_id'] for row in qa_rows])
    captions = []
    for row in caption_rows:
        annotation_path = vrs/'Annotations_val'/Path(row['image_id']).with_suffix('.json')
        annotation = read_json(annotation_path)
        require(annotation.get('image') == row['image_id'] and isinstance(annotation.get('caption'), str)
                and annotation['caption'].strip(), f'Caption schema unclear: {annotation_path}')
        captions.append({'image': row['image_id'], 'caption': annotation['caption']})
    levir_data = read_json(levir/'LevirCCcaptions.json')
    require(isinstance(levir_data, dict) and isinstance(levir_data.get('images'), list),
            'LEVIR-CC JSON must contain rows under images.')
    test_rows = [row for row in levir_data['images'] if row.get('split') == 'test']
    paired = []
    for row in test_rows:
        require(row.get('filepath') == 'test', f'Conflicting LEVIR test filepath: {row!r}')
        require(isinstance(row.get('sentences'), list) and row['sentences'] and
                all(isinstance(sentence.get('raw'), str) and sentence['raw'].strip() for sentence in row['sentences']),
                f'Unknown LEVIR sentences[].raw format: {row!r}')
        if all(image_path(levir/'images/test'/side, row['filename']).is_file() for side in ('A', 'B')):
            paired.append(row)
    unchanged = [row for row in paired if row.get('changeflag') == 0 and
                 any(re.search(r'no (?:change|difference)|identical|same as before', sentence['raw'], re.I)
                     for sentence in row['sentences'])]
    changed = [row for row in paired if row.get('changeflag') == 1]
    require(unchanged and len(changed) >= 4, 'Cannot select a caption-confirmed no-change pair plus four changed test pairs.')
    change_rows = [unchanged[0], *changed[:4]]
    for row in [*qa_rows, *caption_rows, *ref_rows]:
        with Image.open(images/row['image_id']) as image:
            image.verify()
    for row in change_rows:
        for sentence in row['sentences']:
            require('\t' not in sentence['raw'] and '\n' not in sentence['raw'], 'Unsupported caption tabs/newlines')
        for side in ('A', 'B'):
            with Image.open(levir/'images/test'/side/row['filename']) as image:
                require(image.size == (256, 256), f'Unexpected LEVIR dimensions: {row["filename"]} {image.size}')
                image.verify()
    # Everything above is read-only preflight. Do not guess or copy partial samples.
    first_annotation = vrs/'Annotations_val'/Path(matched_qa[0]['image_id']).with_suffix('.json')
    for name, sample in [('VRSBench_EVAL_vqa.json', matched_qa[0]),
                         ('VRSBench_EVAL_referring.json', matched_ref[0]),
                         (str(first_annotation), read_json(first_annotation)),
                         ('LEVIR-CC test row (top-level keys: '+str(list(levir_data))+')', change_rows[0])]:
        print('\nREAL ANNOTATION:', name, '\n'+json.dumps(sample, indent=2))
    output.mkdir(parents=True, exist_ok=True)
    for folder in ('model1', 'grounding', 'change', 'fusion', 'domain_gap'):
        (output/folder).mkdir(exist_ok=True)
    provenance = []
    def record(file, dataset, split, original):
        provenance.append((file, dataset, split, original))
    for folder, rows in [('model1', [*qa_rows, *caption_rows]), ('grounding', ref_rows)]:
        for row in rows:
            name = row['image_id']
            copy_image(images/name, output/folder/name)
            record(f'{folder}/{name}', 'VRSBench', 'validation', f'Images_val/{name}')
    questions = [{'image': row['image_id'], 'question': row['question'], 'answer': row['ground_truth']} for row in qa_rows]
    (output/'model1/questions.json').write_text(json.dumps(questions, indent=2)+'\n')
    (output/'model1/captions.json').write_text(json.dumps(captions, indent=2)+'\n')
    record('model1/questions.json', 'VRSBench', 'validation', 'VRSBench_EVAL_vqa.json; image/question/ground_truth fields, 20 matched rows; no prompt suffix added')
    record('model1/captions.json', 'VRSBench', 'validation', '; '.join('Annotations_val/'+Path(row['image']).with_suffix('.json').name for row in captions))
    boxes = []
    for row in ref_rows:
        with Image.open(images/row['image_id']) as image:
            x1, y1, x2, y2 = envelope(row)
            xyxy = [x1*image.width, y1*image.height, x2*image.width, y2*image.height]
        boxes.append({'image': row['image_id'], 'phrase': row['question'], 'boxes': [xyxy]})
    (output/'grounding/boxes.json').write_text(json.dumps(boxes, indent=2)+'\n')
    record('grounding/boxes.json', 'VRSBench', 'validation', 'VRSBench_EVAL_referring.json; question + obj_corner for the 10 listed images')
    manifest = []
    for row in change_rows:
        stem = Path(row['filename']).stem
        for side in ('A', 'B'):
            target = f'{stem}_{side}.png'
            copy_image(levir/'images/test'/side/row['filename'], output/'change'/target)
            record('change/'+target, 'LEVIR-CC', 'test', f'images/test/{side}/{row["filename"]}')
        # One tab-separated line per reference preserves every raw caption.
        for sentence in row['sentences']:
            require('\t' not in sentence['raw'] and '\n' not in sentence['raw'], 'Unsupported caption tabs/newlines')
            manifest.append(f'{stem}_A.png\t{stem}_B.png\t{sentence["raw"]}')
    (output/'change/captions.txt').write_text('\n'.join(manifest)+'\n')
    record('change/captions.txt', 'LEVIR-CC', 'test', 'LevirCCcaptions.json → images[].sentences[].raw for '+', '.join(row['filename'] for row in change_rows))
    readmes = {
        'model1': '20 distinct VRSBench validation QA images and 3 additional caption images. questions.json uses image/question/answer; captions.json uses image/caption. Filenames are relative to this folder. QA has no adapter suffix. Test inputs only; see ../README.md for provenance and licensing.',
        'grounding': '10 VRSBench validation images with real phrases and absolute-pixel xyxy boxes in boxes.json. Derived from normalized four-vertex obj_corner envelopes, clipped to image bounds. See ../README.md for conversion and provenance. Test inputs only.',
        'change': '5 LEVIR-CC test pairs; A=before, B=after. captions.txt contains 3 tab-separated fields per line: A filename, B filename, one original raw reference. Repeated pair filenames retain all references. One no-change pair is included. Test inputs only.',
        'fusion': FUSION_NOTE,
        'domain_gap': 'Empty placeholder: no domain-gap samples supplied. Add only test inputs with source, split, original filename and licence recorded in ../README.md. No training or synthetic stand-ins.',
    }
    for folder, description in readmes.items():
        (output/folder/'README.md').write_text(description+'\n')
        record(folder+'/README.md', 'Generated documentation', 'not applicable', 'scripts/prepare_samples.py; no dataset payload')
    # Save a full source tree locally; do not flood the terminal with ~40k names.
    tree = []
    for dataset in sorted(datasets.iterdir()):
        if dataset.is_dir():
            tree.append(dataset.name+'/')
            for file in sorted(dataset.rglob('*')):
                tree.append(str(file.relative_to(datasets))+('/' if file.is_dir() else f' ({file.stat().st_size} bytes)'))
    (output/'source_tree.txt').write_text('\n'.join(tree)+'\n')
    record('source_tree.txt', 'Local dataset inventory', 'all installed splits listed; only validation/test copied', str(datasets)+' (inventory only)')
    report = dict(vqa_total=len(vqa), vqa_matched=len(matched_qa), vqa_unique_images=len({r['image_id'] for r in matched_qa}),
                  referring_total=len(referring), referring_matched=len(matched_ref), referring_unique_images=len({r['image_id'] for r in matched_ref}),
                  polygons_outside_image=sum(any(n<0 or n>1 for n in row['obj_corner']) for row in matched_ref),
                  box_encoding_consistency='all matched rows: floor(100 * clipped polygon envelope) == ground_truth',
                  levir_test_annotations=len(test_rows), levir_test_pairs_present=len(paired),
                  selected_change=[{'filename': row['filename'], 'changeflag': row['changeflag'], 'references': len(row['sentences'])} for row in change_rows])
    (output/'preparation_summary.json').write_text(json.dumps(report, indent=2)+'\n')
    record('preparation_summary.json', 'Generated audit', 'VRSBench validation / LEVIR-CC test', 'Counts from EVAL JSONs and LevirCCcaptions.json; scripts/prepare_samples.py')
    box_check.parent.mkdir(parents=True, exist_ok=True)
    sample = boxes[0]
    with Image.open(images/sample['image']) as original:
        image = original.convert('RGB')
        canvas = Image.new('RGB', (image.width, image.height+100), 'white')
        canvas.paste(image)
        draw = ImageDraw.Draw(canvas)
        draw.rectangle(sample['boxes'][0], outline='red', width=3)
        draw.multiline_text((8, image.height+5), '\n'.join(textwrap.wrap(sample['phrase'], 70))+ '\n'+str(sample['boxes'][0]), fill='black')
        canvas.save(box_check)
    lines = ['# Evaluation samples (local only)', '',
             f'Dataset root: `{datasets}`. These copies are **test inputs only**. Nothing is trained here.', '',
             'Academic-use licences apply. Do not redistribute or commit dataset images/annotations or the derived box overlay. This note does not grant any additional rights; retain and follow the source dataset licences. Source data is untouched. `.gitignore` keeps sample payloads and `docs/box_check.png` out of Git while permitting these README files.', '',
             '## Inspected formats and selection', '',
             f'VRSBench EVAL QA: {len(matched_qa):,}/{len(vqa):,} entries match Images_val ({report["vqa_unique_images"]:,} distinct images). Referring: {len(matched_ref):,}/{len(referring):,} entries match ({report["referring_unique_images"]:,} distinct images). Selection follows EVAL order, one row per distinct image, without using missing images.', '',
             'Original box format: obj_corner = [x1,y1,x2,y2,x3,y3,x4,y4], normalized polygon vertices. Convert to absolute axis-aligned xyxy = [clip(min(x),0,1)*width, clip(min(y),0,1)*height, clip(max(x),0,1)*width, clip(max(y),0,1)*height]. No resized coordinates, axis swaps, masks, or cxcywh assumption. Retain pixel precision instead of using quantized ground_truth strings. Every EVAL ground_truth exactly equals floor(100 × the clipped normalized envelope).', '',
             f'{report["polygons_outside_image"]:,} EVAL polygons extend beyond image boundaries; clipping matches the EVAL encoding. Do not join objects using obj_ids: repeated IDs in EVAL rows are not a reliable per-image object identity. Use each referring row’s own phrase and polygon.', '',
             f'LEVIR-CC: top-level images; filename, split, filepath, sentences[].raw. {len(paired):,}/{len(test_rows):,} annotated test rows have both A and B locally. Only test is sampled. All references are retained as separate manifest lines. First pair {change_rows[0]["filename"]} has changeflag=0 and explicit no-change captions; the other four have changeflag=1.', '',
             f'Box check outside this folder: `{box_check}` derives from VRSBench validation `Images_val/{sample["image"]}` and its EVAL referring phrase/polygon. The original image occupies the top part; the caption strip adds no coordinate offset.', '',
             '## Every prepared file', '', '| Local file | Source dataset | Split | Original filename / source |', '|---|---|---|---|']
    for file, dataset, split, original in sorted(provenance):
        lines.append(f'| `{file}` | {dataset} | {split} | {original} |')
    lines.append('| `README.md` | Generated provenance documentation | not applicable | scripts/prepare_samples.py |')
    (output/'README.md').write_text('\n'.join(lines)+'\n')
    print('\nMATCH COUNTS\n'+json.dumps(report, indent=2))
    print('\n| Model | Images / pairs | Reference records |\n|---|---:|---:|')
    print('| Model 1 | 20 QA + 3 caption images | 20 QA + 3 captions |')
    print('| Model 2 | 10 images | 10 phrases / boxes |')
    print(f'| Model 3 | 5 pairs (10 images) | {len(manifest)} captions |')
    print('| Model 4 | 0 | pending |\n| Domain gap | 0 | pending |')
    print(f'Full dataset tree: {output/"source_tree.txt"}\nBox visual check: {box_check}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--datasets', type=Path, default=Path.home()/'Datasets')
    parser.add_argument('--output', type=Path, default=ROOT/'data_samples')
    parser.add_argument('--box-check', type=Path, default=ROOT/'docs/box_check.png')
    args = parser.parse_args()
    try:
        prepare(args.datasets.resolve(), args.output.resolve(), args.box_check.resolve())
    except (ValueError, KeyError, OSError, TypeError) as exc:
        parser.exit(1, f'Preparation stopped: {exc}\n')


if __name__ == '__main__':
    main()
