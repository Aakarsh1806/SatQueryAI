"""Evaluate intent routing and strict fusion input rejection without loading models."""
import argparse
import json
from pathlib import Path
import sys
import tempfile
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.eval_common import ROOT, cell
from backend.app.agent.routing import detect_task, AgentInputError
from backend.app.preprocessing.fusion import read_fusion_pair, FusionInputError

# Expected labels are explicit, independent of router regexes. "error" means a
# human-readable input rejection, not a specialist invocation.
CASES = [
    (['optical'], 'What is visible?', 'qa', None),
    (['optical'], 'How many buildings are visible?', 'qa', None),
    (['optical'], 'Is there water?', 'qa', None),
    (['optical'], 'What colour is the roof?', 'qa', None),
    (['optical'], 'Describe this image', 'caption', None),
    (['optical'], 'Caption this scene', 'caption', None),
    (['optical'], 'Summarize this area', 'caption', None),
    (['optical'], 'Summarise the image', 'caption', None),
    (['optical'], 'show me a summary', 'caption', None),
    (['optical'], 'Describe where the road is', 'caption', None),
    (['optical'], 'SHOW ME A SUMMARY', 'caption', None),
    (['optical'], 'where does the road go', 'grounding', None),
    (['optical'], 'Where are storage tanks?', 'grounding', None),
    (['optical'], 'Locate the bridge', 'grounding', None),
    (['optical'], 'Find ships', 'grounding', None),
    (['optical'], 'Highlight buildings', 'grounding', None),
    (['optical'], 'Show roads', 'grounding', None),
    (['optical'], 'Detect vehicles', 'grounding', None),
    (['optical'], 'DETECT water', 'grounding', None),
    (['optical'], 'Wherever possible, count buildings', 'qa', None),
    (['optical'], 'Is this a showcase building?', 'qa', None),
    (['optical'], 'Is smoke detectable?', 'qa', None),
    (['optical'], 'What about this?', 'qa', None),
    (['optical'], 'Help', 'qa', None),
    (['optical'], 'Classify the land cover', 'qa', None),
    (['optical','optical'], 'What changed?', 'change_caption', None),
    (['optical','optical'], 'How many buildings were added?', 'change_caption', None),
    (['optical','optical'], 'Describe differences', 'change_caption', None),
    (['optical','optical'], 'Find new roads', 'change_caption', None),
    (['optical','optical'], 'Generate a change map', 'change_caption', None),
    (['optical','optical'], 'Describe this image', 'change_caption', None),
    (['optical','sar'], 'Classify land cover', 'classification', 'valid'),
    (['sar','optical'], 'What is this?', 'classification', 'valid'),
    (['optical','sar'], 'Where is the forest?', 'classification', 'valid'),
    (['optical','sar'], 'Describe the area', 'classification', 'valid'),
    (['optical','sar'], 'Classify land cover', 'error', 'png'),
    (['optical','sar'], 'Classify land cover', 'error', 'rgb'),
    (['optical','sar'], 'Classify land cover', 'error', 'wrong_size'),
    (['optical','sar'], 'Classify land cover', 'error', 'mismatched_grid'),
    (['optical','sar'], 'Classify land cover', 'error', 'no_crs'),
    (['sar'], 'Describe this image', 'error', None),
    (['sar'], 'Classify land cover', 'error', None),
    (['sar','sar'], 'Compare these images', 'error', None),
    (['optical','optical','optical'], 'Describe these images', 'error', None),
    (['optical','sar','optical'], 'Classify land cover', 'error', None),
    ([], 'Describe this image', 'error', None),
    (['optical'], 'What changed?', 'error', None),
    (['optical'], 'Compare before and after', 'error', None),
    (['optical'], 'What is the difference?', 'error', None),
    (['optical'], '', 'error', None),
    (['optical','optical'], '   ', 'error', None),
    (['optical','sar'], '', 'error', None),
    (['unknown'], 'Describe this image', 'error', None),
]


def fusion_fixtures(directory):
    import numpy as np
    import rasterio
    from PIL import Image
    from rasterio.transform import from_origin
    paths = {}
    for name, bands, width, crs, offset in [('valid',13,256,'EPSG:32643',0), ('sar',2,256,'EPSG:32643',0),
        ('rgb',3,256,'EPSG:32643',0), ('wrong_size',13,128,'EPSG:32643',0),
        ('mismatched_grid',13,256,'EPSG:32643',10000), ('no_crs',13,256,None,0)]:
        path = directory/(name+'.tif')
        with rasterio.open(path,'w',driver='GTiff',width=width,height=width,count=bands,dtype='float32',
                           crs=crs,transform=from_origin(500000+offset,2000000,10,10)) as target:
            target.write(np.full((bands,width,width), -15 if bands==2 else 1000, dtype='float32'))
        paths[name] = path
    paths['png'] = directory/'rgb.png'
    Image.new('RGB',(256,256)).save(paths['png'])
    return paths


def evaluate():
    results = []
    with tempfile.TemporaryDirectory(prefix='routing-eval-') as temporary:
        fixtures = fusion_fixtures(Path(temporary))
        for index,(modalities,query,expected,format_case) in enumerate(CASES,1):
            intent, model_id = None, None
            try:
                decision = detect_task(modalities,query)
                intent, model_id = decision.task, decision.model_id
                if format_case:
                    read_fusion_pair(fixtures[format_case], fixtures['sar'])
                actual, reason = decision.task, ' '.join(decision.reasons)
            except (AgentInputError,FusionInputError) as exc:
                actual, reason = 'error', str(exc)
            results.append(dict(index=index, inputs='+'.join(modalities) or 'none', query=query,
                format=format_case or 'not applicable', expected=expected, actual=actual, intent=intent,
                model_id=model_id, correct=actual==expected, reason=reason))
    return results


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=ROOT/'docs/routing_eval.md')
    args = parser.parse_args(argv)
    rows = evaluate()
    correct = sum(row['correct'] for row in rows)
    wrong = [row for row in rows if not row['correct']]
    lines = ['# Routing and input-validation evaluation','',
        f'Accuracy: **{correct}/{len(rows)} = {correct/len(rows):.1%}**. This is deterministic rule/input-contract coverage, not model accuracy or evidence of general routing quality.', '',
        'No specialist is loaded. Fusion format fixtures are generated in a temporary directory solely to exercise the production raw-input validator; they are not Model 4 test samples. Model 4 remains stub. This directly scores the real fusion input contract, not the permissive synthetic stub demonstration path. The intent router chooses fusion from optical+SAR, then strict input validation rejects PNG/RGB/wrong-size/unregistered pairs. Ambiguous single-optical queries use the documented default QA route; no LLM fallback is enabled. A road-location query yields boxes, not a road trajectory.', '',
        'Fixed during evaluation: "show me a summary" initially selected grounding because "show" matched but the caption rule omitted the noun "summary". Caption precedence now includes "summary" (including uppercase). The initial run had 51/53 matches; these two cases failed. No expected labels were changed.', '',
        '## Wrong routes in the final run', '']
    lines += [f'- Case {r["index"]}: expected {r["expected"]}, got {r["actual"]}. {r["reason"]}' for r in wrong] or ['None.']
    lines += ['', '## Expected versus actual', '', '| # | Inputs | Format | Query | Expected | Actual | Correct | Reason |', '|---|---|---|---|---|---|---|---|']
    for row in rows:
        lines.append('| '+' | '.join(cell(row[k]) for k in ['index','inputs','format','query','expected','actual','correct','reason'])+' |')
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text('\n'.join(lines)+'\n')
    args.output.with_suffix('.json').write_text(json.dumps(rows,indent=2)+'\n')
    print(f'Routing accuracy: {correct}/{len(rows)} = {correct/len(rows):.1%}')
    for row in wrong: print('WRONG:',row)
    return int(bool(wrong))


if __name__=='__main__':raise SystemExit(main())
