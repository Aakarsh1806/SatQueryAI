"""Describe domain-gap inputs with real Models 1 and 2; no labels or accuracy invented."""
from pathlib import Path
import sys
from time import perf_counter
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.eval_common import arguments, real_model, save_report, no_samples


def main(argv=None):
    args=arguments(__doc__,'domain_gap','domain_gap.md',argv)
    images=sorted(path for path in args.samples.rglob('*') if path.is_file() and path.suffix.lower() in {'.png','.jpg','.jpeg','.tif','.tiff'})
    if not images:return no_samples(args,'Domain-gap test')
    from backend.app.preprocessing import inspect, domain_warnings
    from backend.app.schemas import VLMInput, GroundingInput
    records=[]
    # Finish and unload each specialist before constructing the next one.
    for model_id in ['rs_vlm','grounding']:
        model=real_model(model_id,args.device)
        for image in images:
            record=dict(image=str(image.relative_to(args.samples)),model=model_id)
            started=perf_counter()
            try:
                info=inspect(image)
                record['warnings']=domain_warnings(info)
                if info['modality']!='optical':
                    raise ValueError('These specialists require an optical RGB input; modality was guessed SAR. No silent override.')
                inputs=VLMInput(image_path=str(image),task='caption') if model_id=='rs_vlm' else GroundingInput(image_path=str(image),text_prompt='buildings')
                output=model.run(inputs,'Describe this satellite image in detail.' if model_id=='rs_vlm' else 'buildings')
                record.update(answer=output.answer,output=output.model_dump(mode='json'),
                              warnings=record['warnings']+output.metadata.warnings)
            except Exception as exc:record['error']=f'{type(exc).__name__}: {exc}'
            finally:model.unload()
            record['seconds']=round(perf_counter()-started,3);records.append(record)
            print(record,flush=True)
    save_report(args.output,'Domain-gap test',
        'These inputs differ from VRSBench training data. Outputs and warnings are observations, not validated answers. '
        'No reference labels, accuracy claim, calibrated scores, or model suitability guarantee. RS-VLM uses the exact '
        'caption prompt; grounding uses the fixed buildings probe. Multispectral raw DNs may be rejected by the RGB contract.',records,
        {'images':len(images),'errors':sum('error' in row for row in records)},
        [('image','Image'),('model','Model'),('answer','Output'),('warnings','Warnings'),('seconds','Seconds'),('error','Error')])
    return int(any('error' in row for row in records))


if __name__=='__main__':raise SystemExit(main())
