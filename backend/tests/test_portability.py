import asyncio
from io import BytesIO
import json
from pathlib import Path

import httpx
from PIL import Image
import pytest

from backend.app.agent.demo import DemoCatalog, file_sha256
from backend.app.api.config import Settings
from backend.app.api.main import create_app
from backend.app.models.registry import ModelRegistry
from backend.app.models.stubs import VLMStub, FusionStub
from backend.app.schemas import VLMInput, FusionInput


def png(path, color=(40,80,120)):
    Image.new('RGB',(256,256),color).save(path)
    return path.read_bytes()


def demo_manifest(tmp_path, model_id='rs_vlm', placeholder=False):
    a=tmp_path/'a.png';png(a)
    files=[a]
    if model_id=='fusion':
        b=tmp_path/'b.png';png(b,(60,70,90));files.append(b)
        model=FusionStub('Fusion','test');model.load()
        result=model.run(FusionInput(optical_path=str(a),sar_path=str(b)))
        query='Classify land cover';task='classification';modalities=['optical','sar']
    else:
        model=VLMStub('VLM','test');model.load()
        result=model.run(VLMInput(image_path=str(a)), 'What is visible?')
        query='What is visible?';task='qa';modalities=['optical']
    result.metadata.mode='real' if model_id!='fusion' else 'stub'
    model.unload()
    entry=dict(id='fixed',model_id=model_id,task=task,query=query,sha256=[file_sha256(p) for p in files],
               modalities=modalities,recorded_at='2026-11-11T00:00:00Z',result=result.model_dump(mode='json'),placeholder=placeholder)
    manifest=tmp_path/'manifest.json';manifest.write_text(json.dumps({'format_version':1,'entries':[entry]}))
    return manifest,files,query


@pytest.mark.parametrize('model_id,placeholder',[('rs_vlm',False),('fusion',True)])
def test_demo_api_labels_and_never_loads_model(tmp_path,model_id,placeholder):
    manifest,files,query=demo_manifest(tmp_path,model_id,placeholder)
    settings=Settings(database_url=f'sqlite:///{tmp_path/"api.db"}',storage_dir=tmp_path/'uploads',trace_dir=tmp_path/'traces',demo_mode=True,demo_manifest=manifest)
    registry=ModelRegistry()
    for entry in registry.entries.values(): entry.mode='stub'
    def unexpected_load(*args):pytest.fail('Playback loaded a model')
    registry.get=unexpected_load
    async def scenario():
        app=create_app(settings,registry)
        async with app.router.lifespan_context(app):
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app),base_url='http://test') as client:
                data={'query':query}
                if model_id=='fusion':data['modality_overrides']='["optical","sar"]'
                accepted=await client.post('/api/analyze',data=data,files=[('files',(p.name,p.read_bytes(),'image/png')) for p in files])
                identifier=accepted.json()['request_id']
                for _ in range(500):
                    job=(await client.get(f'/api/requests/{identifier}')).json()
                    if job['status'] not in {'queued','running'}:break
                    await asyncio.sleep(.01)
                assert job['status']=='done',job
                assert job['execution_type']=='pre-computed'
                assert job['result']['metadata']['execution_type']=='pre-computed'
                assert job['result']['metadata']['demo_info']['placeholder']==placeholder
                assert 'Pre-computed' in job['summary']
                trace=(await client.get(job['trace_url'])).json()
                assert trace['execution_type']=='pre-computed'
                assert trace['chosen_model']['execution_type']=='pre-computed'
                assert not trace['lifecycle']['loaded']
                report=await client.get(f'/api/report/{identifier}')
                assert 'pre-computed output (not live inference)' in report.text
                if placeholder:assert 'PLACEHOLDER' in report.text
    asyncio.run(scenario())


def test_demo_exact_query_hash_and_modality_only(tmp_path):
    from backend.app.agent.routing import detect_task
    manifest,files,query=demo_manifest(tmp_path)
    catalog=DemoCatalog(manifest);images=[{'path':str(files[0]),'modality':'optical'}]
    decision=detect_task(['optical'],query)
    assert catalog.lookup(images,query,decision)
    assert catalog.lookup(images,query+' extra',decision) is None
    assert catalog.lookup([dict(images[0],modality='sar')],query,decision) is None
    png(files[0],(1,2,3))
    assert catalog.lookup(images,query,decision) is None


def test_fusion_demo_rejects_unmarked_real_pair_claim(tmp_path):
    manifest,_,_=demo_manifest(tmp_path,'fusion',False)
    with pytest.raises(ValueError,match='placeholder'):DemoCatalog(manifest)


def test_production_static_serving_does_not_shadow_api(tmp_path):
    static=tmp_path/'dist';static.mkdir();(static/'index.html').write_text('<h1>SatQuery</h1>')
    settings=Settings(database_url=f'sqlite:///{tmp_path/"api.db"}',storage_dir=tmp_path/'uploads',trace_dir=tmp_path/'traces',static_dir=static)
    async def scenario():
        app=create_app(settings)
        async with app.router.lifespan_context(app):
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app),base_url='http://test') as client:
                assert (await client.get('/')).text=='<h1>SatQuery</h1>'
                assert (await client.get('/api/health')).json()['status']=='ok'
                assert (await client.get('/backend/models.yaml')).status_code==404
    asyncio.run(scenario())


def test_portable_environment_overrides(monkeypatch,tmp_path):
    import shutil
    from backend.app.models.registry import DEFAULT_CONFIG
    copied=tmp_path/'models.yaml';shutil.copyfile(DEFAULT_CONFIG,copied)
    monkeypatch.setenv('SATQUERY_MODELS_CONFIG',str(copied))
    monkeypatch.setenv('SATQUERY_RS_VLM_MODE','real')
    monkeypatch.setenv('SATQUERY_RS_VLM_WEIGHTS_PATH',str(tmp_path/'missing'))
    monkeypatch.setenv('SATQUERY_DEVICE','cpu')
    registry=ModelRegistry()
    assert registry.entries['rs_vlm'].weights_path==str(tmp_path/'missing')
    assert registry.entries['rs_vlm'].settings['device']=='cpu'
    assert registry.resolve('rs_vlm')['fallback_reason']=='weights_missing'
    assert registry.entries['fusion'].mode in {'stub','real'}
    monkeypatch.setenv('SATQUERY_RETENTION_DAYS','3')
    assert Settings().retention_hours==72
    monkeypatch.setenv('SATQUERY_RS_VLM_MODE','invalid')
    with pytest.raises(ValueError):ModelRegistry()


def test_standalone_registry_reads_local_env_without_overriding_shell(tmp_path,monkeypatch):
    from backend.app.models.registry import DEFAULT_CONFIG
    (tmp_path/'.env').write_text('SATQUERY_DEVICE=cpu\nSATQUERY_FUSION_WEIGHTS_PATH=weights/from-local-env\n')
    monkeypatch.setenv('SATQUERY_FUSION_WEIGHTS_PATH','temporary-test-value')
    monkeypatch.delenv('SATQUERY_FUSION_WEIGHTS_PATH')
    monkeypatch.setenv('SATQUERY_DEVICE','auto')
    registry=ModelRegistry(DEFAULT_CONFIG,project_root=tmp_path)
    assert registry.entries['fusion'].weights_path=='weights/from-local-env'
    assert registry.entries['rs_vlm'].settings.get('device') is None
