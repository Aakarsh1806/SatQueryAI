"""Evaluate aligned SEN12MS pairs, or cleanly skip when samples are absent."""
import json
from pathlib import Path
import sys
from time import perf_counter
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.eval_common import arguments, real_model, save_report, no_samples


def fusion_samples(directory):
    pairs=[]
    for optical in sorted(directory.glob('*_s2.tif')):
        identifier=optical.name.removesuffix('_s2.tif')
        sar=directory/(identifier+'_s1.tif')
        if sar.is_file():pairs.append((identifier,optical,sar))
    return pairs


def main(argv=None):
    args=arguments(__doc__,'fusion','model4_eval.md',argv)
    pairs=fusion_samples(args.samples)
    if not pairs:return no_samples(args,'Model 4 evaluation — pending real samples')
    manifest=args.samples/'expected_labels.json'
    if not manifest.is_file():return no_samples(args,'Model 4 evaluation — missing expected_labels.json')
    labels=json.loads(manifest.read_text())
    if not isinstance(labels,dict) or any(not isinstance(labels.get(identifier),str) for identifier,a,b in pairs):
        raise ValueError('expected_labels.json must map each pair id to its expected class string, e.g. {"pair01":"Forest"}.')
    model=real_model('fusion')  # CPU only, independent of --device.
    results=[]
    for identifier,optical,sar in pairs:
        started=perf_counter();record=dict(id=identifier,expected=labels[identifier])
        try:
            prediction=model.fuse_predict(str(optical),str(sar))
            record.update(predicted=prediction['label'],top3=prediction['top3'],correct=prediction['label']==labels[identifier])
        except Exception as exc:record.update(error=f'{type(exc).__name__}: {exc}',correct=False)
        finally:model.unload()
        record['seconds']=round(perf_counter()-started,3);results.append(record)
        print(json.dumps(record),flush=True)
    failures=sum('error' in r for r in results)
    save_report(args.output,'Model 4 local pair accuracy',
        'Local aligned pair test only, not a reproduced SEN12MS benchmark. Input contract: original 13-band S2 DNs '
        'and 2-band VV/VH S1 dB GeoTIFFs, 256×256, matching CRS/transform, no resampling. Top-3 probabilities '
        'are uncalibrated. No registry modes or weights are modified.',results,
        {'pairs':len(pairs),'errors':failures,'accuracy':sum(r['correct'] for r in results)/len(results)},
        [('id','Pair'),('expected','Expected'),('predicted','Predicted'),('top3','Top 3'),('seconds','Seconds'),('error','Error')])
    return int(bool(failures))


if __name__=='__main__':raise SystemExit(main())
