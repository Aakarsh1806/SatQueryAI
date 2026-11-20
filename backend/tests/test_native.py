import numpy as np
from PIL import Image
import pytest

from backend.app.preprocessing import prepare_for, normalize_change, PreprocessingError
from scripts.sample_inputs import change_samples, caption_concerns


def test_native_rgb_preserves_values_separate_from_display(tmp_path):
    path=tmp_path/'input.png'
    pixels=np.full((256,256,3),120,dtype=np.uint8)
    pixels[0,0]=[100,110,120]
    Image.fromarray(pixels).save(path)
    result=prepare_for('rs_vlm',[path],output_dir=tmp_path/'out')
    np.testing.assert_array_equal(result.tensors['rgb'],pixels)
    assert result.native_paths != result.previews
    np.testing.assert_array_equal(np.asarray(Image.open(result.native_paths[0])),pixels)
    assert not np.array_equal(np.asarray(Image.open(result.previews[0])),pixels)


def test_change_normalization_and_order(tmp_path):
    a,b=tmp_path/'a.png',tmp_path/'b.png'
    Image.new('RGB',(256,256),(100,120,140)).save(a)
    Image.new('RGB',(256,256),(150,160,170)).save(b)
    result=prepare_for('change',[a,b],output_dir=tmp_path/'prepared')
    assert result.tensors['before'].shape==(1,3,256,256)
    expected=(np.array([100,120,140])-np.array([100.6790,99.5023,84.9932]))/np.array([50.9820,48.4838,44.7057])
    np.testing.assert_allclose(result.tensors['before'][0,:,0,0],expected,atol=1e-6)
    assert not np.array_equal(result.tensors['before'],result.tensors['after'])
    assert any('UNCONFIRMED' in warning for warning in result.warnings)
    assert result.tensors['before'].dtype==np.float32
    assert Image.open(result.previews[0]).size==(256,256)
    alternative,_=normalize_change(np.full((1,1,3),255),{'normalization':'imagenet'})
    np.testing.assert_allclose(alternative[0,:,0,0],(np.ones(3)-[.485,.456,.406])/[.229,.224,.225],atol=1e-6)
    with pytest.raises(PreprocessingError,match='positive'):
        normalize_change(np.ones((1,1,3)),{'std':[0,1,1]})


def test_sample_discovery_without_loading_models(tmp_path):
    assert change_samples(tmp_path)==[]
    pair=tmp_path/'pair';pair.mkdir()
    (pair/'before.png').touch();(pair/'after.png').touch()
    (pair/'captions.txt').write_text('a road was added\n')
    assert change_samples(tmp_path)[0][2]=='a road was added'
    (pair/'captions.txt').write_text('before.png\tafter.png\tnew buildings appeared\n')
    assert change_samples(tmp_path)[0][2]=='new buildings appeared'
    assert 'repetitive' in caption_concerns('road '*10,'road was added')[0]


def test_correct_registry_paths_and_missing_ml_fallback(monkeypatch,tmp_path):
    from backend.app.models.registry import ModelRegistry
    from backend.app.models.rsvlm import RSVLMModel
    registry=ModelRegistry()
    expected={'rs_vlm':'weights/satquery_rsvlm_lora','grounding':'weights/SatQuery_Model2_Handover',
              'change':'weights/chg2cap_3h/run_01/best.pth','fusion':'weights/SatQuery_Model4_Handover/model'}
    assert {key:value.weights_path for key,value in registry.entries.items()}==expected
    assert registry.entries['fusion'].mode in {'stub','real'}
    assert all(value.mode in {'stub','real'} for value in registry.entries.values())
    registry.entries['rs_vlm'].mode='real'
    monkeypatch.setattr(RSVLMModel,'available',staticmethod(lambda:False))
    if (registry.project_root/expected['rs_vlm']).exists():
        assert registry.get('rs_vlm').fallback_reason=='ml_dependencies_missing'


def test_strict_fusion_path_does_not_replace_raw_data_with_preview(tmp_path):
    import rasterio
    from rasterio.transform import from_origin
    a,b=tmp_path/'optical.tif',tmp_path/'sar.tif'
    for path,count,value in ((a,13,1500),(b,2,-12)):
        with rasterio.open(path,'w',driver='GTiff',width=256,height=256,count=count,dtype='float32',
                           crs='EPSG:32643',transform=from_origin(500000,2000000,10,10)) as out:
            out.write(np.full((count,256,256),value,dtype='float32'))
    from backend.app.preprocessing.fusion import S2_BANDS,S1_BANDS
    import json
    artifacts=tmp_path/'stats';artifacts.mkdir()
    (artifacts/'model4_normalization_stats.json').write_text(json.dumps({'sentinel2':{'bands':S2_BANDS,'mean':[0]*13,'std':[1]*13},'sentinel1':{'bands':S1_BANDS,'mean':[0]*2,'std':[1]*2}}))
    (artifacts/'metadata.json').write_text(json.dumps({'num_classes':8,'class_names':[f'class{i}' for i in range(8)]}))
    prepared=prepare_for('fusion',[a,b],output_dir=tmp_path/'out',artifact_dir=artifacts)
    assert prepared.tensors['optical_raw'].shape==(13,256,256)
    np.testing.assert_array_equal(prepared.tensors['optical_raw'],1500)
    np.testing.assert_array_equal(prepared.tensors['sar_raw'],-12)
    assert set(prepared.native_paths).isdisjoint(prepared.previews)
