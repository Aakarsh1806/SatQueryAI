"""Record a fixed list of genuine samples for explicit pre-computed playback."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import sys
import tempfile
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.eval_common import ROOT, read_rows
from scripts.eval_model3 import grouped_pairs
from backend.app.agent import Agent
from backend.app.agent.demo import file_sha256
from backend.app.models.registry import ModelRegistry


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--samples',type=Path,default=ROOT/'data_samples')
    parser.add_argument('--output',type=Path,default=ROOT/'demo/manifest.json')
    parser.add_argument('--from-rehearsal',type=Path,help='Package already measured live API results, without re-running models')
    args=parser.parse_args();samples=[]
    if args.from_rehearsal:
        source=json.loads(args.from_rehearsal.read_text());entries=[]
        args.output.parent.mkdir(parents=True,exist_ok=True)
        for row in source['scenarios']:
            job=row.get('job') or {};result=job.get('result')
            if not result:continue
            if result['metadata']['mode']!='real' or result['metadata']['execution_type']!='live':
                raise ValueError('Rehearsal must contain measured live real-model results')
            if job['model_id']=='fusion':raise ValueError('Do not package unverified Model 4 results')
            paths=[args.samples/name for name in row['files']]
            result=json.loads(json.dumps(result))
            overlay=(result.get('evidence') or {}).get('overlay_path')
            if overlay:
                destination=args.output.parent/(row['scenario']+'_overlay.png')
                shutil.copyfile(overlay,destination);result['evidence']['overlay_path']=destination.name
            entries.append(dict(id=row['scenario'],model_id=job['model_id'],task=job['task'],query=row['query'],
                sha256=[file_sha256(path) for path in paths],modalities=['optical']*len(paths),
                recorded_at=source['timestamp'],result=result,placeholder=False,
                provenance='Previously measured live API rehearsal, including its observed limitations',sample_files=[str(path.resolve()) for path in paths]))
        if not entries:raise ValueError('No live results to package')
        args.output.write_text(json.dumps({'format_version':1,'entries':entries},indent=2)+'\n')
        print(f'Packaged {len(entries)} measured outputs as pre-computed playback; no model was loaded. Fusion: no samples found.')
        return 0
    qa=read_rows(args.samples/'model1/questions.json')
    if qa:samples.append(('rs_vlm',[args.samples/'model1'/qa[0]['image']],qa[0]['question']))
    captions=read_rows(args.samples/'model1/captions.json')
    if captions:samples.append(('rs_vlm',[args.samples/'model1'/captions[0]['image']],'Describe this image'))
    ground=read_rows(args.samples/'grounding/boxes.json')
    if ground:samples.append(('grounding',[args.samples/'grounding'/ground[0]['image']],'Detect vehicles'))
    changes=grouped_pairs(args.samples/'change')
    if changes:
        pair=next((pair for pair in changes if pair[0].name=='test_000004_A.png'),changes[0])
        samples.append(('change',[pair[0],pair[1]],'What changed between these images?'))
    args.output.parent.mkdir(parents=True,exist_ok=True)
    entries=[];failures=0
    for model_id,paths,query in samples:
        if not all(path.is_file() for path in paths):
            print(f'{model_id}: no samples found');continue
        registry=ModelRegistry();registry.entries[model_id].mode='real'
        registry.entries['rs_vlm'].settings['local_files_only']=True
        if registry.resolve(model_id)['mode']!='real':
            print(f'{model_id}: unavailable; no pre-computed entry recorded');continue
        with tempfile.TemporaryDirectory(prefix='satquery-record-demo-') as temporary:
            agent=Agent(registry,trace_dir=Path(temporary)/'traces',artifact_dir=Path(temporary)/'prepared')
            response=agent.run([str(path) for path in paths],query)
            if not response.success or response.result.metadata.mode!='real':
                failures+=1;print(f'{model_id}: recording failed: {response.error}');continue
            identifier=model_id+'-'+response.routing.task
            result=response.result.model_dump(mode='json')
            overlay=(result.get('evidence') or {}).get('overlay_path')
            if overlay:
                destination=args.output.parent/(identifier+'_overlay.png')
                shutil.copyfile(overlay,destination);result['evidence']['overlay_path']=destination.name
            entries.append(dict(id=identifier,model_id=model_id,task=response.routing.task,query=query,
                sha256=[file_sha256(path) for path in paths],modalities=['optical']*len(paths),
                recorded_at=datetime.now(timezone.utc).isoformat(),result=result,placeholder=False,
                provenance='Locally recorded real inference on prepared academic-use validation/test inputs',
                sample_files=[str(path.resolve()) for path in paths]))
            print(f'Recorded {identifier}: query={query!r}',flush=True)
    print('fusion: no samples found; no Model 4 result or fake input was created.')
    if not entries:
        print('no samples found or no real model available; existing catalogue was not replaced.');return int(bool(failures))
    args.output.write_text(json.dumps({'format_version':1,'entries':entries},indent=2)+'\n')
    print(f'{len(entries)} pre-computed entries saved to {args.output}; outputs are not live inference when replayed.')
    return int(bool(failures))


if __name__=='__main__':raise SystemExit(main())
