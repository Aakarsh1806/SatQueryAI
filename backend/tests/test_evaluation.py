"""Metric correctness, no-sample behavior, and sample-backed real API coverage."""
import asyncio
import json
from pathlib import Path
from time import perf_counter

import httpx
import pytest

from scripts.eval_common import matches, iou, best_iou
from scripts.eval_routing import evaluate
from scripts.eval_model3 import grouped_pairs, caption_metrics
from backend.app.models.registry import ModelRegistry, PROJECT_ROOT
from backend.app.api.config import Settings
from backend.app.api.main import create_app


def test_matching_and_geometry_metrics():
    assert matches(' Yellow ', 'yellow') == (True, True)
    assert matches('There are two.', '2') == (False, True)
    assert not matches('20', '2')[1]
    assert not matches('north', 'no')[1]
    assert not matches('', '')[1]
    assert iou([0,0,10,10], [0,0,10,10]) == 1
    assert iou([0,0,10,10], [5,0,15,10]) == pytest.approx(1/3)
    assert best_iou([], [[0,0,10,10]]) == 0
    assert best_iou([[20,20,30,30],[0,0,10,10]], [[0,0,10,10]]) == 1


def test_extended_routing_with_strict_formats():
    rows = evaluate()
    assert len(rows) >= 40
    assert all(row['correct'] for row in rows), [row for row in rows if not row['correct']]
    assert all(row['actual']=='error' and row['intent']=='classification' for row in rows if row['format'] in {'png','rgb'})


def test_reference_captions_are_grouped_once(tmp_path):
    for name in ['one_A.png','one_B.png','two_A.png','two_B.png']:
        (tmp_path/name).touch()
    (tmp_path/'captions.txt').write_text('one_A.png\tone_B.png\tfirst reference\none_A.png\tone_B.png\tsecond reference\ntwo_A.png\ttwo_B.png\tthird reference\n')
    groups = grouped_pairs(tmp_path)
    assert len(groups)==2 and groups[0][2]==['first reference','second reference']


def test_optional_caption_metrics():
    pytest.importorskip('pycocoevalcap')
    records=[{'predicted':'a new road was built here','references':['a new road was built here']},
             {'predicted':'the two scenes are identical','references':['the two scenes are identical']}]
    metrics=caption_metrics(records)
    assert metrics['BLEU-4']==pytest.approx(1,abs=1e-6)
    assert metrics['ROUGE-L']==pytest.approx(1)
    assert metrics['CIDEr']>0


@pytest.mark.parametrize('module_name',['eval_model1','eval_model2','eval_model3','eval_model4','domain_gap'])
def test_missing_samples_does_not_load_models(module_name, tmp_path, monkeypatch, capsys):
    import importlib
    module=importlib.import_module('scripts.'+module_name)
    def unexpected_load(*args,**kwargs):pytest.fail('A model was loaded without samples')
    monkeypatch.setattr(module,'real_model',unexpected_load)
    report=tmp_path/'report.md'
    assert module.main(['--samples',str(tmp_path/'empty'),'--output',str(report)])==0
    assert 'no samples found' in capsys.readouterr().out
    assert 'not been run yet' in report.read_text()


def sample_registry(model_id):
    registry=ModelRegistry()
    registry.entries[model_id].mode='real'
    registry.entries['fusion'].mode='stub'
    registry.entries['rs_vlm'].settings['local_files_only']=True
    if registry.resolve(model_id)['mode']!='real':pytest.skip(f'{model_id}: weights or ML packages missing')
    if model_id=='rs_vlm':
        from huggingface_hub import try_to_load_from_cache
        from backend.app.models.rsvlm import BASE_MODEL
        index=try_to_load_from_cache(BASE_MODEL,'model.safetensors.index.json')
        if not isinstance(index,str):pytest.skip('Base weights not cached; no downloads in tests')
        shards=set(json.loads(Path(index).read_text())['weight_map'].values())
        if not all(isinstance(try_to_load_from_cache(BASE_MODEL,name),str) for name in shards):
            pytest.skip('Base weights incomplete; no downloads in tests')
    return registry


@pytest.mark.parametrize('model_id',['rs_vlm','grounding','change'])
def test_real_samples_through_api(model_id,tmp_path):
    samples=PROJECT_ROOT/'data_samples'
    if model_id=='change':
        pairs=grouped_pairs(samples/'change')
        if not pairs:pytest.skip('No genuine change pairs supplied')
        paths=[pairs[0][0],pairs[0][1]];query='What changed between these images?'
    else:
        folder=samples/('model1' if model_id=='rs_vlm' else 'grounding')
        manifest=folder/('questions.json' if model_id=='rs_vlm' else 'boxes.json')
        if not manifest.is_file():pytest.skip(f'No {model_id} sample manifest')
        rows=json.loads(manifest.read_text())
        if not rows:pytest.skip('No samples found')
        paths=[folder/rows[0]['image']]
        query=rows[0]['question'] if model_id=='rs_vlm' else 'Locate '+rows[0]['phrase']
    if not all(path.is_file() for path in paths):pytest.skip('Sample image files missing')
    registry=sample_registry(model_id)
    async def scenario():
        app=create_app(Settings(database_url=f'sqlite:///{tmp_path/"api.db"}',storage_dir=tmp_path/'uploads',trace_dir=tmp_path/'traces'),registry)
        async with app.router.lifespan_context(app):
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app),base_url='http://test') as client:
                assert (await client.get('/api/health')).json()['status']=='ok'
                listing=(await client.get('/api/models')).json()['models']
                assert next(row for row in listing if row['id']=='fusion')['mode']=='stub'
                inspected=await client.post('/api/inspect',files={'file':(paths[0].name,paths[0].read_bytes(),'image/png')})
                assert inspected.status_code==200 and inspected.json()['band_count']==3
                started=perf_counter()
                accepted=await client.post('/api/analyze',data={'query':query},files=[('files',(p.name,p.read_bytes(),'image/png')) for p in paths])
                assert accepted.status_code==202 and accepted.json()['status']=='queued'
                identifier=accepted.json()['request_id']
                for _ in range(6000):
                    job=(await client.get(f'/api/requests/{identifier}')).json()
                    if job['status'] not in {'queued','running'}:break
                    await asyncio.sleep(.05)
                elapsed=perf_counter()-started
                print(f'{model_id} genuine sample API: {elapsed:.3f}s status={job["status"]}, error={job["error"]}')
                assert job['status']=='done',job
                assert job['model_id']==model_id and job['result']['metadata']['mode']=='real'
                assert job['result']['metadata']['validation_status']==registry.entries[model_id].validation_status
                trace=(await client.get(job['trace_url'])).json()
                assert trace['lifecycle']['unloaded']
                for url in job['preview_urls']+job['evidence_overlay_urls']:
                    assert (await client.get(url)).status_code==200
                assert (await client.get('/api/requests')).json()['total']==1
                assert (await client.get(f'/api/report/{identifier}')).status_code==200
                assert (await client.post(f'/api/requests/{identifier}/cancel')).json()['status']=='done'
                (tmp_path/f'{model_id}_genuine_api.json').write_text(json.dumps({'seconds':elapsed,'job':job},indent=2))
    asyncio.run(scenario())
