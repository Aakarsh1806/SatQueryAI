"""Run all live demo API scenarios against a fresh, isolated local backend."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time
from uuid import uuid4

import httpx

ROOT=Path(__file__).resolve().parents[1]
SCENARIOS=[
 ('a_question',['model1/P0003_0002.png'],'What color are the large vehicles seen in the image?','rs_vlm','qa'),
 ('b_caption',['model1/P0019_0021.png'],'Describe this image','rs_vlm','caption'),
 ('c_grounding',['grounding/P0003_0002.png'],'Detect vehicles','grounding','grounding'),
 ('d_change',['change/test_000004_A.png','change/test_000004_B.png'],'What changed between these images?','change','change_caption'),
 ('e_failure',['model1/P0003_0002.png'],'What changed between these images?',None,None),
]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--with-tests',action='store_true')
    args=parser.parse_args()
    run=ROOT/'.demo-rehearsal'/(datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'-'+uuid4().hex[:6]);run.mkdir(parents=True)
    env=dict(os.environ,DATABASE_URL=f'sqlite:///{run/"demo.db"}',SATQUERY_STORAGE_DIR=str(run/'uploads'),
             SATQUERY_TRACE_DIR=str(run/'traces'),DEMO_MODE='false',SATQUERY_STATIC_DIR='',
             SATQUERY_RS_VLM_MODE='real',SATQUERY_GROUNDING_MODE='real',SATQUERY_CHANGE_MODE='real',SATQUERY_FUSION_MODE='real')
    records=[];test_summary='not requested';process=None
    try:
        if args.with_tests:
            # Tests use their own fresh tmp_path DB/upload fixtures, not the rehearsal history.
            test_env=dict(env)
            for model in ['RS_VLM','GROUNDING','CHANGE']:test_env.pop('SATQUERY_'+model+'_MODE',None)
            tested=subprocess.run([sys.executable,'-m','pytest','-q'],cwd=ROOT,env=test_env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
            (run/'pytest.log').write_text(tested.stdout)
            test_summary=next((line for line in reversed(tested.stdout.splitlines()) if 'passed' in line or 'failed' in line),'No totals')
            print(test_summary,flush=True)
            if tested.returncode:raise RuntimeError('Full test suite failed; inspect the private run log')
        with socket.socket() as sock:sock.bind(('127.0.0.1',0));port=sock.getsockname()[1]
        base=f'http://127.0.0.1:{port}'
        with (run/'server.log').open('w') as output:
            process=subprocess.Popen([sys.executable,'-m','uvicorn','backend.app.api.main:app','--host','127.0.0.1','--port',str(port)],cwd=ROOT,env=env,stdout=output,stderr=subprocess.STDOUT)
            with httpx.Client(base_url=base,timeout=15) as client:
                for _ in range(150):
                    try:
                        if client.get('/api/health').status_code==200:break
                    except httpx.TransportError:pass
                    time.sleep(.2)
                else:raise RuntimeError('Fresh backend did not become healthy')
                assert client.get('/api/requests').json()['total']==0
                models=client.get('/api/models').json()['models'];assert next(m for m in models if m['id']=='fusion')['mode']=='real'
                for identifier,filenames,query,model_id,task in SCENARIOS:
                    paths=[ROOT/'data_samples'/name for name in filenames]
                    if not all(path.is_file() for path in paths):raise RuntimeError('Demo sample missing: '+str(paths))
                    started=time.perf_counter()
                    accepted=client.post('/api/analyze',data={'query':query},files=[('files',(path.name,path.read_bytes(),'image/png')) for path in paths]);accepted.raise_for_status()
                    request_id=accepted.json()['request_id'];seen=[]
                    for _ in range(12000):
                        job=client.get('/api/requests/'+request_id).json();seen.append(job['status'])
                        if job['status'] not in {'queued','running'}:break
                        time.sleep(.05)
                    elapsed=time.perf_counter()-started
                    if model_id:
                        assert job['status']=='done',job
                        assert job['model_id']==model_id and job['routing']['task']==task
                        assert job['execution_type']=='live' and job['result']['metadata']['mode']=='real',job
                        if model_id in {'rs_vlm','change'}:assert job['result']['evidence'] is None
                        if model_id=='change':assert job['result']['confidence'] is None
                        assert client.get('/api/report/'+request_id).status_code==200
                    else:
                        assert job['status']=='failed' and 'two images' in job['error'],job
                        assert job['result'] is None
                    trace=client.get(job['trace_url']).json()
                    if model_id:assert trace['lifecycle']['unloaded']
                    row=dict(scenario=identifier,files=filenames,query=query,seconds=round(elapsed,3),statuses=list(dict.fromkeys(seen)),job=job,trace=trace)
                    records.append(row)
                    print(identifier,round(elapsed,3),job['status'],job.get('execution_type'),(job.get('result') or {}).get('answer',job.get('error')),flush=True)
                records.append(dict(scenario='f_fusion_status',status='real adapter enabled; no samples found; no real pair verified',model=next(m for m in models if m['id']=='fusion')))
        report=dict(timestamp=datetime.now(timezone.utc).isoformat(),fresh_database=True,fresh_uploads=True,tests=test_summary,scenarios=records)
        (ROOT/'docs/demo_rehearsal.json').write_text(json.dumps(report,indent=2)+'\n')
        print('Saved docs/demo_rehearsal.json; private run directory:',run)
        return 0
    except Exception as exc:
        (run/'failure.txt').write_text(str(exc));print('Rehearsal failed:',str(exc),flush=True);return 1
    finally:
        if process:
            process.terminate()
            try:process.wait(timeout=30)
            except subprocess.TimeoutExpired:process.kill();process.wait()


if __name__=='__main__':raise SystemExit(main())
