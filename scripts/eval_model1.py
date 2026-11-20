"""Evaluate 20 real VRSBench validation QA examples, not the full benchmark."""
import sys
import json
from pathlib import Path
from time import perf_counter
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.eval_common import arguments, read_rows, real_model, sample_path, matches, save_report, no_samples


def main(argv=None):
    args = arguments(__doc__, 'model1', 'model1_eval.md', argv)
    rows = read_rows(args.samples/'questions.json')
    if not rows: return no_samples(args, 'Model 1 small-sample QA evaluation')
    from backend.app.schemas import VLMInput
    model = real_model('rs_vlm', args.device)
    results = []
    for index, row in enumerate(rows):
        started = perf_counter()
        record = dict(index=index+1, image=row['image'], question=row['question'], reference=row['answer'])
        try:
            image = sample_path(args.samples, row['image'])
            result = model.run(VLMInput(image_path=str(image), task='qa'), row['question'])
            assert result.metadata.mode == 'real'
            exact, loose = matches(result.answer, row['answer'])
            record.update(predicted=result.answer, exact=exact, lenient=loose, device=result.metadata.device,
                          prompt=result.metadata.parameters['prompt'], metadata=result.metadata.model_dump(mode='json'))
        except Exception as exc:
            record.update(error=f'{type(exc).__name__}: {exc}', exact=False, lenient=False)
        finally:
            model.unload()
        record['seconds'] = round(perf_counter()-started, 3)
        results.append(record)
        print(json.dumps({key:value for key,value in record.items() if key!='metadata'}), flush=True)
    failures = sum('error' in row for row in results)
    save_report(args.output, 'Model 1: small-sample VRSBench QA',
        'This is a selected validation sanity check, not the full VRSBench benchmark. No training or weight changes. '
        'Exact match folds case, Unicode and whitespace only. Lenient match removes punctuation/articles, maps English '
        'zero–ten to digits and accepts a contiguous whole-token reference phrase within the answer. This heuristic can '
        'over-credit verbose/contradictory answers; inspect each row. Errors count as misses. The original question is '
        'passed unchanged; the adapter adds exactly one " Answer with a short phrase." suffix. '
        'Each request lazy-loads and unloads; timings include loading, inference and cleanup.',
        results, {'samples':len(rows), 'errors':failures,
                  'exact_match':f'{sum(r["exact"] for r in results)}/{len(rows)} ({sum(r["exact"] for r in results)/len(rows):.1%})',
                  'lenient_match':f'{sum(r["lenient"] for r in results)}/{len(rows)} ({sum(r["lenient"] for r in results)/len(rows):.1%})',
                  'total_seconds':round(sum(r['seconds'] for r in results),3)},
        [('index','#'),('image','Image'),('question','Question'),('reference','Reference'),('predicted','Prediction'),
         ('exact','Exact'),('lenient','Lenient'),('seconds','Seconds'),('error','Error')])
    return int(bool(failures))


if __name__ == '__main__': raise SystemExit(main())
