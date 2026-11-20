"""Real weights are optional; never download base weights as a side effect of tests."""
import asyncio
from io import BytesIO
from pathlib import Path
from time import perf_counter
from types import SimpleNamespace

import httpx
import numpy as np
from PIL import Image
import pytest

from backend.app.models.registry import ModelRegistry, PROJECT_ROOT
from backend.app.preprocessing import PreparedInputs


def ml(*packages):
    for package in packages: pytest.importorskip(package)


def entry(model_id):
    registry=ModelRegistry(); item=registry.entries[model_id].model_copy(deep=True)
    item.mode='real'
    path=registry.project_root/item.weights_path
    if not path.exists():pytest.skip(f'Weights missing: {path}')
    return item,path


def test_model3_strict_load_and_unload():
    ml('torch','torchvision','einops')
    from backend.app.models.chg2cap import Chg2CapModel
    item,path=entry('change');model=Chg2CapModel(entry=item,weights_path=path)
    started=perf_counter()
    try:
        model.load()
        assert [record['keys'] for record in model.load_report]==[624,80,52]
        assert all(record['strict'] for record in model.load_report)
        assert len(model.id_to_word)==436 and model.id_to_word[2]=='<START>'
        print(f'Chg2Cap strict load: {perf_counter()-started:.3f}s, device={model.device}, no missing/unexpected keys')
    finally:model.unload()
    assert not model.is_loaded


def test_strict_errors_are_explicit():
    ml('torch')
    import torch
    from backend.app.models.loading import strict_state_load
    module=torch.nn.Linear(2,3)
    with pytest.raises(RuntimeError,match='shape_mismatches=.*weight'):
        strict_state_load(module,{'weight':torch.zeros(1,2),'bias':torch.zeros(3)},'decoder_dict')
    with pytest.raises(RuntimeError,match='missing=.*bias'):
        strict_state_load(module,{'weight':torch.zeros(3,2)},'decoder_dict')


def test_model1_processor_and_adapter_against_meta_base():
    ml('torch','transformers','peft','accelerate')
    import torch
    from transformers import AutoConfig,AutoProcessor,Qwen2_5_VLForConditionalGeneration
    from accelerate import init_empty_weights
    from peft import PeftModel
    from backend.app.models.rsvlm import BASE_MODEL,verify_adapter
    item,path=entry('rs_vlm')
    try:config=AutoConfig.from_pretrained(BASE_MODEL,local_files_only=True)
    except OSError:pytest.skip('Base config not cached; tests do not download it')
    processor=AutoProcessor.from_pretrained(path,local_files_only=True,min_pixels=100352,max_pixels=313600)
    assert processor.chat_template and processor.image_processor.size['shortest_edge']==100352
    with init_empty_weights():base=Qwen2_5_VLForConditionalGeneration(config)
    model=PeftModel.from_pretrained(base,path,low_cpu_mem_usage=True,autocast_adapter_dtype=False)
    try:
        from peft import get_peft_model_state_dict
        from safetensors import safe_open
        actual=get_peft_model_state_dict(model)
        with safe_open(str(path/'adapter_model.safetensors'),framework='pt',device='cpu') as saved:
            assert set(actual)==set(saved.keys()) and len(actual)==696
            assert all(list(actual[key].shape)==saved.get_slice(key).get_shape() for key in actual)
        print('RS-VLM: all 696 adapter keys/shapes match the pinned meta base; full base inference and value checks await cached base weights')
    finally:del model,base,processor


def test_model1_generation_contract_without_base_weights(monkeypatch):
    ml('torch','qwen_vl_utils')
    import torch
    from backend.app.models.rsvlm import RSVLMModel,CAPTION_PROMPT
    item,path=entry('rs_vlm')
    calls={}
    class Batch(dict):
        @property
        def input_ids(self):return self['input_ids']
        def to(self,device):return self
    class Tokenizer:
        def apply_chat_template(self,messages,**kwargs):calls['prompt']=messages[0]['content'][1]['text'];return 'formatted'
        def batch_decode(self,tokens,**kwargs):
            assert tokens.tolist()==[[3,4]]
            return ['a road']
    class Processor:
        tokenizer=Tokenizer()
        def __call__(self,**kwargs):return Batch(input_ids=torch.tensor([[1,2]]))
    class Generator:
        dtype=torch.bfloat16
        def generate(self,**kwargs):
            calls['options']=kwargs
            return SimpleNamespace(sequences=torch.tensor([[1,2,3,4]]),scores=(torch.tensor([[0.,0.,0.,4.,0.]]),torch.tensor([[0.,0.,0.,0.,4.]])))
    model=RSVLMModel(entry=item,weights_path=path)
    model.model,model.processor,model.torch,model.device=Generator(),Processor(),torch,torch.device('cpu')
    model.load_report={'strict':True}
    result=model.run(PreparedInputs('rs_vlm',{'rgb':np.ones((256,256,3),dtype=np.float32)*100}), 'ignored', {'task':'caption'})
    assert result.answer=='a road' and result.evidence is None
    assert calls['prompt']==CAPTION_PROMPT
    assert not calls['options']['do_sample'] and calls['options']['max_new_tokens']==128
    assert result.confidence.value==pytest.approx(float(torch.softmax(torch.tensor([0.,0.,0.,4.,0.]),0)[3]))
    assert not result.confidence.calibrated and 'uncalibrated' in result.confidence.note
    assert not model.is_loaded


@pytest.mark.parametrize('model_id', ['change','rs_vlm'])
def test_real_api_generated_input_wiring(tmp_path,model_id):
    ml('torch','torchvision','einops')
    item,path=entry(model_id)
    if model_id=='rs_vlm':
        ml('transformers','peft','accelerate','qwen_vl_utils')
        from huggingface_hub import snapshot_download
        from backend.app.models.rsvlm import BASE_MODEL
        try:cached=Path(snapshot_download(BASE_MODEL,local_files_only=True,allow_patterns=['config.json','generation_config.json','model*.safetensors','model.safetensors.index.json']))
        except Exception:pytest.skip('Base model not cached; no automatic multi-GB download in tests')
        import json
        index=cached/'model.safetensors.index.json'
        expected=set(json.loads(index.read_text())['weight_map'].values()) if index.is_file() else {'model.safetensors'}
        if not all((cached/name).is_file() for name in expected):pytest.skip('Complete base inference weights not cached')
        item.settings['local_files_only']=True
    from backend.app.api.main import create_app
    from backend.app.api.config import Settings
    registry=ModelRegistry();registry.entries[model_id]=item
    async def scenario():
        settings=Settings(database_url=f'sqlite:///{tmp_path / "jobs.db"}',storage_dir=tmp_path/'uploads',trace_dir=tmp_path/'traces')
        app=create_app(settings,registry)
        async with app.router.lifespan_context(app):
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app),base_url='http://test') as client:
                buffer=BytesIO();Image.new('RGB',(256,256),(100,120,140)).save(buffer,format='PNG')
                count=2 if model_id=='change' else 1
                started=perf_counter()
                response=await client.post('/api/analyze',data={'query':'What changed?' if count==2 else 'What is visible?'},files=[('files',(f'{i}.png',buffer.getvalue(),'image/png')) for i in range(count)])
                assert response.status_code==202
                request_id=response.json()['request_id']
                for _ in range(6000):
                    job=(await client.get(f'/api/requests/{request_id}')).json()
                    if job['status'] in {'done','failed'}:break
                    await asyncio.sleep(.05)
                print(f'{model_id} POST /api/analyze generated-input wiring: {perf_counter()-started:.3f}s status={job["status"]} error={job["error"]}; not a caption-quality evaluation')
                assert job['status']=='done',job['error']
                assert job['result']['metadata']['mode']=='real'
                strict=job['result']['metadata']['parameters']['strict_load']
                if model_id=='rs_vlm':
                    assert strict=={'strict':True,'adapter_tensors':696,'verified_values':True}
                    assert job['result']['confidence']['calibrated'] is False
                else:
                    assert all(record['strict'] for record in strict)
                    assert job['result']['confidence'] is None
                print('Real wiring metadata:', model_id, job['result']['metadata']['parameters'])
                trace=(await client.get(job['trace_url'])).json()
                assert trace['chosen_model']['mode']=='real'
                assert job['result']['evidence'] is None
                assert job['preview_urls']
    asyncio.run(scenario())


def test_model1_strict_value_verification_errors(monkeypatch,tmp_path):
    ml('torch','peft','safetensors')
    import torch
    import peft
    import safetensors.torch
    from backend.app.models.rsvlm import verify_adapter
    expected={'adapter.a':torch.tensor([1.,2.])}
    actual={'adapter.a':torch.tensor([1.,2.])}
    monkeypatch.setattr(safetensors.torch,'load_file',lambda *args,**kwargs:expected)
    monkeypatch.setattr(peft,'get_peft_model_state_dict',lambda _:actual)
    assert verify_adapter(object(),tmp_path)['verified_values']
    actual['adapter.a']=torch.tensor([0.,2.])
    with pytest.raises(RuntimeError,match='not loaded exactly.*adapter.a'):
        verify_adapter(object(),tmp_path)
    actual['adapter.a']=torch.zeros(3)
    with pytest.raises(RuntimeError,match='shape_mismatches=.*adapter.a'):
        verify_adapter(object(),tmp_path)
    actual.clear()
    with pytest.raises(RuntimeError,match='missing=.*adapter.a'):
        verify_adapter(object(),tmp_path)


def test_device_priority_and_explicit_cpu(monkeypatch):
    ml('torch')
    import torch
    from backend.app.models.device import select_device
    monkeypatch.setattr(torch.cuda,'is_available',lambda:True)
    monkeypatch.setattr(torch.backends.mps,'is_available',lambda:True)
    assert select_device(torch).type=='cuda'
    assert select_device(torch,'cpu').type=='cpu'
    monkeypatch.setattr(torch.cuda,'is_available',lambda:False)
    assert select_device(torch).type=='mps'
    monkeypatch.setattr(torch.backends.mps,'is_available',lambda:False)
    assert select_device(torch).type=='cpu'
    with pytest.raises(RuntimeError,match='unavailable'):select_device(torch,'mps')


def test_cache_cleanup_calls_available_device_caches():
    from types import SimpleNamespace
    from backend.app.models.device import free_cache
    calls = []
    torch = SimpleNamespace(
        cuda=SimpleNamespace(is_available=lambda: True, empty_cache=lambda: calls.append('cuda')),
        backends=SimpleNamespace(mps=SimpleNamespace(is_available=lambda: True)),
        mps=SimpleNamespace(empty_cache=lambda: calls.append('mps')),
    )
    free_cache(torch)
    assert calls == ['cuda', 'mps']
