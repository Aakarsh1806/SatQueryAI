"""Real LEVIR-CC subset caption evaluation with optional Python COCO metrics."""
import json
from pathlib import Path
import re
import sys
from time import perf_counter
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.eval_common import arguments, real_model, save_report, no_samples
from scripts.sample_inputs import change_samples, caption_concerns


def grouped_pairs(directory):
    groups={}
    for before,after,reference in change_samples(directory):
        key=(before.resolve(),after.resolve())
        groups.setdefault(key,[]).append(reference)
    return [(a,b,refs) for (a,b),refs in groups.items()]


def caption_metrics(records):
    try:
        from pycocoevalcap.bleu.bleu import Bleu
        from pycocoevalcap.rouge.rouge import Rouge
        from pycocoevalcap.cider.cider import Cider
    except ImportError as exc:
        return {'status':'not computed: optional pycocoevalcap unavailable', 'reason':str(exc)}
    tokenize=lambda text:' '.join(re.findall(r"[a-z0-9]+(?:'[a-z0-9]+)?",text.lower()))
    references={i:[tokenize(text) for text in row['references']] for i,row in enumerate(records)}
    predictions={i:[tokenize(row.get('predicted',''))] for i,row in enumerate(records)}
    bleu,_=Bleu(4).compute_score(references,predictions)
    rouge,_=Rouge().compute_score(references,predictions)
    cider,_=Cider().compute_score(references,predictions)
    return {'BLEU-4':float(bleu[3]),'ROUGE-L':float(rouge),'CIDEr':float(cider),'METEOR':'not computed'}


def main(argv=None):
    args=arguments(__doc__,'change','model3_eval.md',argv)
    pairs=grouped_pairs(args.samples)
    if not pairs:return no_samples(args,'Model 3 small-subset change captions')
    from backend.app.schemas import ChangeInput
    model=real_model('change',args.device)
    results=[]
    for index,(before,after,refs) in enumerate(pairs,1):
        started=perf_counter();record=dict(index=index,before=before.name,after=after.name,references=refs,reference_text=' | '.join(refs))
        try:
            result=model.run(ChangeInput(before_path=str(before),after_path=str(after)))
            assert result.metadata.mode=='real'
            notes=caption_concerns(result.answer,' '.join(refs))
            record.update(predicted=result.answer,concerns='; '.join(notes),device=result.metadata.device,metadata=result.metadata.model_dump(mode='json'))
        except Exception as exc:
            record.update(error=f'{type(exc).__name__}: {exc}',predicted='')
        finally:model.unload()
        record['seconds']=round(perf_counter()-started,3);results.append(record)
        print(json.dumps({k:v for k,v in record.items() if k!='metadata'}),flush=True)
    metrics=caption_metrics(results)
    print('Subset metrics:',metrics)
    failures=sum('error' in row for row in results)
    repeated=len({row['predicted'] for row in results})==1 and len(results)>1
    if repeated or any(row.get('concerns') for row in results):
        print('WARNING: possible poor/repetitive captions. Compare references; try normalization: imagenet in a separate config experiment. Do not automatically choose normalization using this subset.')
    save_report(args.output,'Model 3: small LEVIR-CC test subset',
        'This selected five-pair subset is not the benchmark. Full-test-split scores supplied by the user: '
        '**BLEU-4 0.542, ROUGE-L 0.727, CIDEr 1.29; METEOR not computed**. Those full-split scores were not reproduced here. '
        'A=before, B=after. All raw reference captions are grouped per pair, so each pair is inferred once. '
        'Default upstream normalization remains UNCONFIRMED: RGB 0–255, resized 256×256, mean [100.679,99.5023,84.9932], '
        'std [50.982,48.4838,44.7057]; no percentile stretch or /255. Optional metrics use pycocoevalcap, '
        'lowercase alphanumeric/apostrophe regex tokens (not Java PTB tokenization). CIDEr document frequencies '
        'are fitted to these five reference groups only; all scores are sensitive to this tiny selection and are '
        'not directly comparable to the full-split numbers. Error outputs are empty captions in the denominator. '
        'No confidence or change map is inferred. Timings include loading and cleanup.',results,
        {'pairs':len(pairs),'references':sum(len(refs) for a,b,refs in pairs),'errors':failures,
         **metrics,'all_captions_identical':repeated,'total_seconds':round(sum(r['seconds'] for r in results),3)},
        [('index','#'),('before','Before'),('after','After'),('predicted','Prediction'),('reference_text','All references'),
         ('concerns','Concerns'),('seconds','Seconds'),('error','Error')])
    return int(bool(failures))


if __name__=='__main__':raise SystemExit(main())
