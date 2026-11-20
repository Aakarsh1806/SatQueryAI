"""Strict SEN12MS raw-data preparation; display rendering is independent."""
import json
from pathlib import Path
import re

import numpy as np
from PIL import Image
import rasterio

from .imagery import PreprocessingError, sar_prepare

S2_BANDS = ['B1','B2','B3','B4','B5','B6','B7','B8','B8A','B9','B10','B11','B12']
S1_BANDS = ['VV','VH']


class FusionInputError(PreprocessingError):
    """An expected input-contract rejection, suitable for UI explanation."""


def _band_name(description):
    value = (description or '').strip().upper()
    match = re.search(r'(?<![A-Z0-9])B(?:AND)?0*(\d+)(A?)(?![A-Z0-9])', value)
    if match: return 'B'+str(int(match[1]))+match[2]
    match = re.search(r'(?<![A-Z])(?:VV|VH)(?![A-Z])',value)
    return match[0] if match else value


def read_fusion_pair(optical_path, sar_path):
    """Validate original GeoTIFFs before reading raw float32; never align/resize."""
    paths = [Path(optical_path),Path(sar_path)]
    for path in paths:
        if path.suffix.lower() not in {'.tif','.tiff'}:
            raise FusionInputError('Fusion needs original GeoTIFF inputs: 13-band Sentinel-2 optical and 2-band VV/VH SAR. PNG/JPG and RGB previews are not supported.')
        if not path.is_file(): raise FusionInputError(f'Image file does not exist: {path.name}')
    try:
        with rasterio.open(paths[0]) as a, rasterio.open(paths[1]) as b:
            if a.driver != 'GTiff' or b.driver != 'GTiff':
                raise FusionInputError('Both fusion inputs must be GeoTIFF files, not another format renamed as TIFF.')
            if a.count != 13: raise FusionInputError(f'Optical GeoTIFF needs exactly 13 bands in Sentinel-2 order; received {a.count}.')
            if b.count != 2: raise FusionInputError(f'SAR GeoTIFF needs exactly 2 bands in VV,VH order; received {b.count}.')
            if (a.width,a.height)!=(256,256) or (b.width,b.height)!=(256,256):
                raise FusionInputError('Fusion needs two 256x256 patches. No automatic resizing or resampling is performed.')
            if a.crs is None or b.crs is None:
                raise FusionInputError('Both fusion GeoTIFFs need non-null CRS metadata and identical georeferencing.')
            if a.crs!=b.crs or a.transform!=b.transform or a.width!=b.width or a.height!=b.height:
                raise FusionInputError('Fusion inputs must have identical CRS, affine transform, width and height. Provide an already aligned pair; no reprojection or resampling is performed.')
            notes=[]
            for source,expected,label in ((a,S2_BANDS,'optical'),(b,S1_BANDS,'SAR')):
                names=[_band_name(value) for value in source.descriptions]
                if any(names):
                    if names!=expected:
                        raise FusionInputError(f'{label} band order must be {",".join(expected)}; descriptions are {names}. Bands are not automatically reordered.')
                else:
                    notes.append(f'{label} band descriptions are absent; file order is assumed to be {",".join(expected)} and cannot be independently verified.')
            raw_a,raw_b=a.read(out_dtype='float32',masked=True),b.read(out_dtype='float32',masked=True)
            if np.ma.getmaskarray(raw_a).any() or np.ma.getmaskarray(raw_b).any():
                raise FusionInputError('Fusion inputs contain nodata/masked pixels. Supply complete finite-valued patches; no nodata filling is performed.')
            optical,sar=np.asarray(raw_a,dtype=np.float32),np.asarray(raw_b,dtype=np.float32)
            if not np.isfinite(optical).all() or not np.isfinite(sar).all():
                raise FusionInputError('Fusion inputs contain non-finite values (NaN or infinity); supply finite raw patches.')
            return optical,sar,dict(crs=a.crs.to_string(),transform=list(a.transform),bounds=list(a.bounds),warnings=notes)
    except FusionInputError: raise
    except (rasterio.errors.RasterioError,OSError) as exc:
        raise FusionInputError(f'Unable to read fusion GeoTIFF inputs: {exc}') from exc


def load_fusion_artifacts(directory):
    directory=Path(directory)
    stats=json.loads((directory/'model4_normalization_stats.json').read_text())
    metadata=json.loads((directory/'metadata.json').read_text())
    if stats['sentinel2']['bands']!=S2_BANDS or stats['sentinel1']['bands']!=S1_BANDS:
        raise ValueError('Fusion normalization statistics band order does not match the required model contract.')
    if metadata['num_classes']!=8 or len(metadata['class_names'])!=8 or len(set(metadata['class_names']))!=8:
        raise ValueError('Fusion metadata must contain eight distinct classes.')
    for name,count in (('sentinel2',13),('sentinel1',2)):
        mean,std=np.asarray(stats[name]['mean'],np.float32),np.asarray(stats[name]['std'],np.float32)
        if mean.shape!=(count,) or std.shape!=(count,) or not np.isfinite(mean).all() or not np.isfinite(std).all() or (std<=0).any():
            raise ValueError(f'Invalid fusion normalization statistics for {name}.')
    return stats,metadata


def prepare_fusion(inputs, directory, artifact_dir):
    from .native import PreparedInputs
    optical,sar,geo=read_fusion_pair(*inputs)
    stats,_=load_fusion_artifacts(artifact_dir)
    tensors={'optical_raw':optical.copy(),'sar_raw':sar.copy()}
    for key,raw,sensor in (('optical',optical,'sentinel2'),('sar',sar,'sentinel1')):
        mean=np.asarray(stats[sensor]['mean'],np.float32)[:,None,None]
        std=np.asarray(stats[sensor]['std'],np.float32)[:,None,None]
        tensors[key]=np.ascontiguousarray(((raw-mean)/std)[None],dtype=np.float32)
        if not np.isfinite(tensors[key]).all(): raise FusionInputError('Fusion normalization produced non-finite values; check source values and normalization statistics.')
    # Fixed 13-band file order starts B1: RGB display is bands 4,3,2 only.
    rgb=np.zeros((256,256,3),dtype=np.uint8)
    for channel,band in enumerate(optical[[3,2,1]]):
        low,high=np.percentile(band,[2,98])
        if high>low: rgb[...,channel]=np.rint(np.clip((band-low)/(high-low),0,1)*255).astype(np.uint8)
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=True)
    preview=directory/'display_optical.png';Image.fromarray(rgb).save(preview)
    _,sar_preview=sar_prepare(inputs[1],directory/'display_sar.png',input_scale='db')
    note=dict(operation='fusion_raw_normalization',optical_bands=S2_BANDS,sar_bands=S1_BANDS,
        optical_shape=list(tensors['optical'].shape),sar_shape=list(tensors['sar'].shape),dtype='float32',
        optical_units='raw digital numbers; not divided by 10000',sar_units='dB; no log conversion or clipping',
        normalization='(raw - training mean) / training std',statistics_path=str(Path(artifact_dir)/'model4_normalization_stats.json'),
        resampled=False,display_rgb_bands=[4,3,2],**geo)
    return PreparedInputs('fusion',tensors,native_paths=list(map(str,inputs)),previews=[str(preview),str(sar_preview)],notes=[note],warnings=geo['warnings'])
