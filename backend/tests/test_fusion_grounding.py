"""Strict input, in-process runtime and four-real-mode routing coverage."""
import asyncio
from io import BytesIO
import json
from pathlib import Path
from time import perf_counter

import httpx
import numpy as np
from PIL import Image
import pytest
import rasterio
from rasterio.transform import from_origin

from backend.app.models.registry import ModelRegistry,PROJECT_ROOT
from backend.app.preprocessing import prepare_for,FusionInputError
from backend.app.preprocessing.fusion import S2_BANDS,S1_BANDS,read_fusion_pair
from backend.app.schemas import GroundingOutput


@pytest.fixture
def pair(tmp_path):
    def write(name,bands,*,size=256,crs='EPSG:32643',transform=None,descriptions=None,nonfinite=False):
        path=tmp_path/name
        values=np.random.default_rng(7).uniform(700,1800,(bands,size,size)).astype('float32') if bands!=2 else np.random.default_rng(8).uniform(-27,-3,(bands,size,size)).astype('float32')
        if nonfinite:values[0,0,0]=np.nan
        with rasterio.open(path,'w',driver='GTiff',count=bands,width=size,height=size,dtype='float32',crs=crs,
                           transform=transform or from_origin(500000,2000000,10,10)) as dst:
            dst.write(values)
            if descriptions:dst.descriptions=descriptions
        return path,values
    a,optical=write('optical.tif',13,descriptions=S2_BANDS)
    b,sar=write('sar.tif',2,descriptions=S1_BANDS)
    return a,b,optical,sar,write


def test_fusion_native_exact_normalization_and_display_bands(pair,tmp_path):
    a,b,optical,sar,_=pair
    if not (PROJECT_ROOT/'weights/SatQuery_Model4_Handover/model/model4_normalization_stats.json').is_file():
        pytest.skip('Model 4 normalization artifacts missing')
    prepared=prepare_for('fusion',[a,b],output_dir=tmp_path/'prepared')
    stats=json.loads((PROJECT_ROOT/'weights/SatQuery_Model4_Handover/model/model4_normalization_stats.json').read_text())
    for key,values,sensor in [('optical',optical,'sentinel2'),('sar',sar,'sentinel1')]:
        expected=(values-np.asarray(stats[sensor]['mean'],np.float32)[:,None,None])/np.asarray(stats[sensor]['std'],np.float32)[:,None,None]
        np.testing.assert_array_equal(prepared.tensors[key][0],expected)
        assert prepared.tensors[key].dtype==np.float32
    np.testing.assert_array_equal(prepared.tensors['optical_raw'],optical)
    np.testing.assert_array_equal(prepared.tensors['sar_raw'],sar)
    rgb=np.asarray(Image.open(prepared.previews[0]))
    for channel,index in enumerate([3,2,1]):
        band=optical[index];lo,hi=np.percentile(band,[2,98])
        np.testing.assert_array_equal(rgb[...,channel],np.rint(np.clip((band-lo)/(hi-lo),0,1)*255).astype('uint8'))
    assert prepared.notes[0]['resampled'] is False
    assert prepared.native_paths==[str(a),str(b)]


@pytest.mark.parametrize('case,message',[
    ('optical_bands','13 bands'),('sar_bands','2 bands'),('size','256x256'),('crs','non-null CRS'),
    ('transform','identical CRS'),('different_crs','identical CRS'),('nonfinite','non-finite'),
    ('order','band order'),('png','GeoTIFF'),('renamed_png','GeoTIFF'),
])
def test_fusion_rejections(pair,tmp_path,case,message):
    a,b,_,_,write=pair
    if case=='optical_bands':a=write('wrong_opt.tif',3)[0]
    elif case=='sar_bands':b=write('wrong_sar.tif',1)[0]
    elif case=='size':a=write('small.tif',13,size=128)[0]
    elif case=='crs':a=write('no_crs.tif',13,crs=None)[0]
    elif case=='transform':b=write('offset.tif',2,transform=from_origin(500010,2000000,10,10))[0]
    elif case=='different_crs':b=write('other_crs.tif',2,crs='EPSG:3857')[0]
    elif case=='nonfinite':b=write('nan.tif',2,nonfinite=True)[0]
    elif case=='order':b=write('swapped.tif',2,descriptions=['VH','VV'])[0]
    elif case in {'png','renamed_png'}:
        a=tmp_path/('image.png' if case=='png' else 'image.tif');Image.new('RGB',(256,256)).save(a,format='PNG')
    with pytest.raises(FusionInputError,match=message):read_fusion_pair(a,b)


def test_fusion_no_nodata_filling(pair):
    a,b,_,_,_=pair
    with rasterio.open(a,'r+') as image:
        image.nodata=-9999;data=image.read(1);data[0,0]=-9999;image.write(data,1)
    with pytest.raises(FusionInputError,match='nodata'):read_fusion_pair(a,b)


def require_model(model_id):
    pytest.importorskip('torch')
    if model_id=='grounding':pytest.importorskip('transformers')
    registry=ModelRegistry();registry.entries[model_id].mode='real'
    if registry.resolve(model_id)['mode']!='real':pytest.skip(f'{model_id} weights/packages unavailable')
    return registry,registry.get(model_id)


def test_grounding_strict_load_same_runtime():
    registry,model=require_model('grounding')
    started=perf_counter()
    try:
        model.load();assert model.load_report['strict']
        assert model.load_report['missing_keys']==model.load_report['unexpected_keys']==[]
        assert model.processor.image_processor.size['height']==512
        print(f'Grounding strict load: {perf_counter()-started:.3f}s, {model.device}, same Transformers runtime')
    finally:model.unload()
    assert not model.is_loaded


def test_grounding_cpu_retry_records_failure(monkeypatch):
    _,model=require_model('grounding')
    import torch
    calls=[]
    class Network:
        def to(self,device):calls.append(str(device))
    model.model,model.processor,model.torch,model.device=Network(),object(),torch,torch.device('mps')
    def infer(image,prompt,threshold):
        calls.append((str(model.device),prompt,threshold))
        if model.device.type=='mps':raise RuntimeError('simulated unsupported MPS operator')
        return GroundingOutput(boxes=[(1,2,8,9)],labels=['tank'],scores=[.6])
    monkeypatch.setattr(model,'_infer',infer)
    try:
        evidence,prompt=model._ground_image(Image.new('RGB',(20,20)),'  TaNk  ',.3)
        assert prompt=='tank.' and evidence.scores==[.6]
        assert model.device.type=='cpu' and model.device_fallback['from']=='mps'
        assert 'simulated unsupported' in model.device_fallback['reason']
    finally:model.unload()


async def finish(client,request_id):
    for _ in range(2400):
        job=(await client.get(f'/api/requests/{request_id}')).json()
        if job['status'] in {'done','failed'}:return job
        await asyncio.sleep(.05)
    pytest.fail('Inference did not finish')


@pytest.mark.parametrize('model_id',['grounding','fusion'])
def test_real_api_generated_inputs(pair,tmp_path,model_id):
    registry,unused=require_model(model_id)
    from backend.app.api.main import create_app
    from backend.app.api.config import Settings
    async def scenario():
        settings=Settings(database_url=f'sqlite:///{tmp_path/"api.db"}',storage_dir=tmp_path/'uploads',trace_dir=tmp_path/'traces')
        app=create_app(settings,registry)
        async with app.router.lifespan_context(app):
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app),base_url='http://test') as client:
                if model_id=='fusion':
                    files=[('files',('optical.tif',pair[0].read_bytes())),('files',('sar.tif',pair[1].read_bytes()))];query='Classify the land cover'
                else:
                    buffer=BytesIO();Image.new('RGB',(384,256),(100,120,140)).save(buffer,format='PNG')
                    files=[('files',('generated.png',buffer.getvalue()))];query='Where are the storage tanks?'
                started=perf_counter()
                accepted=await client.post('/api/analyze',data={'query':query},files=files)
                assert accepted.status_code==202
                job=await finish(client,accepted.json()['request_id'])
                elapsed=perf_counter()-started
                print(f'{model_id} real POST /api/analyze: {elapsed:.3f}s status={job["status"]}, error={job["error"]}; generated inputs, not validation data')
                assert job['status']=='done',job['error']
                result=job['result'];assert result['metadata']['mode']=='real'
                for url in job['preview_urls']+job['evidence_overlay_urls']:
                    assert (await client.get(url)).status_code==200
                if model_id=='grounding':
                    assert result['metadata']['validation_status']==registry.entries['grounding'].validation_status
                    assert result['metadata']['parameters']['text_threshold']==.25
                    assert result['evidence']['score_calibration']=='uncalibrated'
                    assert job['evidence_overlay_urls']
                    assert result['confidence'] is None
                    for x1,y1,x2,y2 in result['evidence']['boxes']:
                        assert 0<=x1<x2<=384 and 0<=y1<y2<=256
                else:
                    assert result['metadata']['device']=='cpu'
                    assert result['confidence']['calibrated'] is False
                    assert len(result['evidence']['top3'])==3 and not job['evidence_overlay_urls']
                    assert any('76%' in warning for warning in result['metadata']['warnings'])
                    assert any('Grassland' in warning for warning in job['warnings'])
                trace=(await client.get(job['trace_url'])).json()
                assert trace['chosen_model']['mode']=='real'
                (tmp_path/f'{model_id}_smoke.json').write_text(json.dumps({'elapsed_seconds':elapsed,'job':job,'trace':trace},indent=2))
    asyncio.run(scenario())


def test_invalid_fusion_explains_before_loading(tmp_path):
    registry,_=require_model('fusion')
    from backend.app.agent import Agent
    a,b=tmp_path/'optical.png',tmp_path/'sar.png'
    Image.new('RGB',(256,256),(10,20,30)).save(a);Image.new('L',(256,256),10).save(b)
    agent=Agent(registry,trace_dir=tmp_path/'traces',artifact_dir=tmp_path/'prepared')
    registry.get=lambda _:pytest.fail('Invalid native fusion inputs must not load a model')
    response=agent.run([a,b],'Classify the land cover')
    assert not response.success and 'GeoTIFF' in response.input_explanation
    trace=json.loads(Path(response.trace_path).read_text())
    assert trace['validation']['code']=='unsupported_fusion_input'
    assert trace['chosen_model']['mode'] is None


def test_routing_evaluation_all_four_configured_real():
    from backend.tests.test_agent import ROUTING_CASES
    from backend.app.agent import detect_task,AgentInputError
    registry=ModelRegistry()
    for entry in registry.entries.values():entry.mode='real'
    assert all(entry.mode=='real' for entry in registry.entries.values())
    print('Four-model effective modes:', {key:registry.resolve(key)['mode'] for key in registry.entries})
    correct=0
    for modalities,query,expected in ROUTING_CASES:
        try:
            decision=detect_task(modalities,query);actual=decision.task
            assert registry.entries[decision.model_id].mode=='real'
        except AgentInputError:actual='error'
        correct+=actual==expected
    print(f'Routing evaluation, all four configured real: {correct}/{len(ROUTING_CASES)} = {correct/len(ROUTING_CASES):.1%} (intent routing, not inference accuracy)')
    assert correct==len(ROUTING_CASES)


def test_genuine_grounding_box_check_if_samples_exist():
    samples=[path for path in (PROJECT_ROOT/'data_samples/grounding').rglob('*') if path.suffix.lower() in {'.png','.jpg','.jpeg','.tif','.tiff'}]
    if not samples:pytest.skip('No genuine grounding image supplied; model remains unvalidated')
    _,model=require_model('grounding')
    result=model.ground(str(samples[0]),'building',.3)
    with Image.open(samples[0]) as image:width,height=image.size
    for x1,y1,x2,y2 in result['boxes']:assert 0<=x1<x2<=width and 0<=y1<y2<=height
    print('Genuine-image box geometry check passed; no accuracy metric or ground-truth agreement is claimed.')


def test_fusion_genuine_expected_prediction_parity_if_supplied():
    from scripts.eval_model4 import fusion_samples
    directory=PROJECT_ROOT/'data_samples/fusion'
    pairs=fusion_samples(directory)
    manifest=directory/'expected_labels.json'
    if not pairs or not manifest.is_file():
        print('no samples found: Model 4 parity requires aligned pairs and expected_labels.json')
        pytest.skip('no samples found: genuine Model 4 parity inputs are absent')
    expected=json.loads(manifest.read_text())
    _,model=require_model('fusion')
    for identifier,a,b in pairs:
        result=model.fuse_predict(str(a),str(b))
        assert result['label']==expected[identifier]


def test_grounding_strict_mismatch_stops(monkeypatch):
    _,model=require_model('grounding')
    import torch
    import transformers
    from types import SimpleNamespace
    monkeypatch.setattr(transformers.AutoConfig,'from_pretrained',lambda *args,**kwargs:SimpleNamespace())
    monkeypatch.setattr(transformers.AutoProcessor,'from_pretrained',lambda *args,**kwargs:object())
    monkeypatch.setattr(transformers.GroundingDinoForObjectDetection,'from_pretrained',
        lambda *args,**kwargs:(torch.nn.Linear(1,1),{'missing_keys':{'model.text_projection.weight'}}))
    with pytest.raises(RuntimeError,match='Strict Grounding DINO.*model.text_projection.weight'):model.load()
    assert not model.is_loaded


def test_real_fusion_api_rejection_is_a_ui_explanation(tmp_path):
    registry,_=require_model('fusion')
    from backend.app.api.main import create_app
    from backend.app.api.config import Settings
    async def scenario():
        app=create_app(Settings(database_url=f'sqlite:///{tmp_path/"jobs.db"}',storage_dir=tmp_path/'uploads',trace_dir=tmp_path/'traces'),registry)
        async with app.router.lifespan_context(app):
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app),base_url='http://test') as client:
                files=[]
                for mode,name in [('RGB','optical.png'),('L','sar.png')]:
                    buffer=BytesIO();Image.new(mode,(256,256),10).save(buffer,format='PNG');files.append(('files',(name,buffer.getvalue())))
                accepted=await client.post('/api/analyze',data={'query':'Classify the land cover'},files=files)
                job=await finish(client,accepted.json()['request_id'])
                assert job['status']=='failed' and job['result'] is None
                assert 'GeoTIFF' in job['input_explanation']
                assert job['routing']['model_id']=='fusion'
                trace=(await client.get(job['trace_url'])).json()
                assert trace['validation']['code']=='unsupported_fusion_input'
    asyncio.run(scenario())


def test_grounding_standalone_result_writes_overlay(tmp_path):
    _,model=require_model('grounding')
    from backend.app.schemas import GroundingInput
    image=tmp_path/'image.png';Image.new('RGB',(64,48),(100,120,140)).save(image)
    result=model.run(GroundingInput(image_path=str(image),text_prompt='building'),options={'artifact_dir':str(tmp_path/'artifacts')})
    assert result.metadata.validation_status=='unvalidated'
    assert Path(result.evidence.overlay_path).is_file()
    assert Image.open(result.evidence.overlay_path).size==(64,48)
    assert not model.is_loaded
