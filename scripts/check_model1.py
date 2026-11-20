"""Run 20 short real QA checks after optical samples are supplied."""
import argparse
import json
from pathlib import Path
import sys
from time import perf_counter

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.sample_inputs import optical_images

QUESTIONS = [
    "What is visible?", "Is there a road?", "Are there buildings?", "Is there water?",
    "Is there vegetation?", "How many buildings are visible?", "What is the main land use?",
    "Are there vehicles?", "Is there a river?", "Is there a bridge?", "Are there storage tanks?",
    "Is there a parking lot?", "Are there agricultural fields?", "Is there a forest?",
    "Is the area residential?", "Are the buildings dense?", "What shape are the buildings?",
    "Is there an airport?", "What is in the center?", "Is this an urban area?",
]


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--samples',type=Path,default=ROOT/'data_samples')
    parser.add_argument('--device',choices=['cpu','mps','cuda'])
    args=parser.parse_args(argv)
    images=optical_images(args.samples)
    if not images:
        print(f"no samples found: add optical images to {args.samples}; no model was loaded.")
        return 0
    from backend.app.models.registry import ModelRegistry
    from backend.app.models.rsvlm import RSVLMModel
    from backend.app.schemas import VLMInput
    registry=ModelRegistry(); entry=registry.entries['rs_vlm'].model_copy(deep=True)
    if args.device:entry.settings['device']=args.device
    if not RSVLMModel.available():
        print('ML packages missing: install backend/requirements-ml.txt.');return 2
    model=RSVLMModel(entry=entry,weights_path=registry.project_root/entry.weights_path)
    failures=0
    for index,question in enumerate(QUESTIONS):
        image=images[index % len(images)]; started=perf_counter()
        try:
            result=model.run(VLMInput(image_path=str(image)),question)
            memory={key:value for key,value in result.metadata.parameters.items() if 'bytes' in key}
            print(json.dumps(dict(index=index+1,image=str(image),question=question,answer=result.answer,
                elapsed_seconds=perf_counter()-started,model_runtime_ms=result.metadata.runtime_ms,
                device=result.metadata.device,memory=memory,confidence=result.confidence.model_dump() if result.confidence else None)))
        except Exception as exc:
            failures+=1;print(json.dumps(dict(index=index+1,error=str(exc),elapsed_seconds=perf_counter()-started)))
        finally:model.unload()
    return 1 if failures else 0


if __name__=='__main__': raise SystemExit(main())
