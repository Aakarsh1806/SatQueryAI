"""Compare real LEVIR-CC test-pair captions with local references."""
import argparse
import json
from pathlib import Path
import sys
from time import perf_counter

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.sample_inputs import change_samples, caption_concerns


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--samples',type=Path,default=ROOT/'data_samples/change')
    parser.add_argument('--normalization',choices=['upstream','imagenet'])
    parser.add_argument('--device',choices=['cpu','mps','cuda'])
    args=parser.parse_args(argv)
    samples=change_samples(args.samples)
    if not samples:
        print(f"no samples found: add LEVIR-CC before/after images and captions.txt to {args.samples}; no model was loaded.")
        return 0
    from backend.app.models.registry import ModelRegistry
    from backend.app.models.chg2cap import Chg2CapModel
    from backend.app.schemas import ChangeInput
    registry=ModelRegistry();entry=registry.entries['change'].model_copy(deep=True)
    if args.normalization:entry.preprocessing['normalization']=args.normalization
    if args.device:entry.settings['device']=args.device
    if not Chg2CapModel.available():print('ML packages missing: install backend/requirements-ml.txt.');return 2
    model=Chg2CapModel(entry=entry,weights_path=registry.project_root/entry.weights_path)
    print('PREPROCESSING UNCONFIRMED: judge predicted vs reference captions; no evaluation claim is made.')
    predictions=[];failures=0
    for a,b,reference in samples:
        started=perf_counter()
        try:
            result=model.run(ChangeInput(before_path=str(a),after_path=str(b)))
            predictions.append(result.answer)
            notes=caption_concerns(result.answer,reference)
            print(json.dumps(dict(before=str(a),after=str(b),predicted=result.answer,reference=reference,
                normalization=entry.preprocessing.get('normalization','upstream'),device=result.metadata.device,
                elapsed_seconds=perf_counter()-started,concerns=notes)))
            if notes:print('WARNING: possible poor/repetitive output. Inspect the reference; try --normalization imagenet as a comparison, not an automatic fix.')
        except Exception as exc:
            failures+=1;print(json.dumps(dict(before=str(a),after=str(b),error=str(exc),elapsed_seconds=perf_counter()-started)))
        finally:model.unload()
    if len(predictions)>1 and len(set(predictions))==1:
        print('WARNING: identical captions for all pairs may indicate repetition/collapse. Compare references; try --normalization imagenet (RGB/255 with ImageNet mean/std).')
    print('If captions look wrong, compare --normalization imagenet; never mark normalization confirmed without genuine-pair validation.')
    return 1 if failures else 0


if __name__=='__main__':raise SystemExit(main())
