"""Separate UI previews from framework-independent model-native NumPy tensors."""
from dataclasses import dataclass, field
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
from PIL import Image
import rasterio

from .imagery import (_read, _rgb_indices, optical_to_rgb, sar_prepare,
                      check_pair, PreprocessingError)

UPSTREAM_MEAN = [100.6790, 99.5023, 84.9932]
UPSTREAM_STD = [50.9820, 48.4838, 44.7057]


@dataclass
class PreparedInputs:
    model_id: str
    tensors: dict[str, np.ndarray]  # Torch conversion happens only inside real adapters.
    native_paths: list[str] = field(default_factory=list)
    previews: list[str] = field(default_factory=list)
    notes: list[dict] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


def _select_rgb(data, info):
    indices = _rgb_indices(info) if data.shape[0] >= 3 else [0, 0, 0]
    rgb = np.moveaxis(data[indices], 0, -1).copy()
    if not np.isfinite(rgb).all():
        raise PreprocessingError("RGB inference inputs contain nodata/invalid pixels; supply valid imagery or an explicit nodata treatment.")
    if rgb.min() < 0 or rgb.max() > 255:
        raise PreprocessingError("RGB inference requires values in 0–255 units. Supply an 8-bit RGB image; display stretching is not applied to model-native data.")
    return rgb.astype(np.float32)


def normalize_change(rgb, config=None):
    config = config or {}
    method = config.get("normalization", "upstream")
    if method == "upstream":
        mean = np.asarray(config.get("mean", UPSTREAM_MEAN), dtype=np.float32)
        std = np.asarray(config.get("std", UPSTREAM_STD), dtype=np.float32)
        values = rgb.astype(np.float32)
    elif method == "imagenet":
        mean = np.asarray([.485, .456, .406], dtype=np.float32)
        std = np.asarray([.229, .224, .225], dtype=np.float32)
        values = rgb.astype(np.float32) / 255
    else:
        raise PreprocessingError("Change normalization must be upstream or imagenet.")
    if mean.shape != (3,) or std.shape != (3,) or not np.isfinite(mean).all() or not np.isfinite(std).all() or (std <= 0).any():
        raise PreprocessingError("Change normalization requires three finite means and three positive standard deviations.")
    tensor = np.ascontiguousarray(np.moveaxis((values - mean) / std, -1, 0)[None], dtype=np.float32)
    return tensor, dict(operation="normalize_change", normalization=method, mean=mean.tolist(), std=std.tolist(),
                        divided_by_255=method == "imagenet", confirmed=bool(config.get("confirmed", False)))


def _persist_raw(path, data, info, pair):
    kwargs = dict(crs=pair["crs"], transform=pair["transform"]) if pair["crs"] else dict(transform=rasterio.transform.from_origin(0,data.shape[1],1,1))
    with rasterio.open(path, "w", driver="GTiff", width=data.shape[2], height=data.shape[1], count=data.shape[0], dtype="float32", nodata=np.nan, **kwargs) as dst:
        dst.write(data.astype(np.float32)); dst.descriptions = tuple(info["band_descriptions"])


def prepare_for(model_id, inputs, *, output_dir=None, config=None, artifact_dir=None):
    """Return PreparedInputs(tensors, notes, previews, native_paths).

    `inputs` is an ordered sequence of source paths (A=before, B=after).
    NumPy tensors remain usable without ML packages. Without output_dir this
    is an in-memory operation; a temporary display directory is discarded.
    Model 4 uses strict original GeoTIFFs and model-specific bandwise statistics.
    """
    paths = [str(path) for path in inputs]
    expected = 2 if model_id in {"change", "fusion"} else 1
    if model_id not in {"rs_vlm", "grounding", "change", "fusion"}:
        raise PreprocessingError(f"Unknown model: {model_id}")
    if len(paths) != expected:
        raise PreprocessingError(f"{model_id} needs {expected} image(s).")
    if output_dir is None:
        with TemporaryDirectory(prefix="satquery-prepare-") as temp:
            result = prepare_for(model_id, paths, output_dir=temp, config=config, artifact_dir=artifact_dir)
        result.native_paths, result.previews = [], []
        return result
    directory = Path(output_dir); directory.mkdir(parents=True, exist_ok=True)
    if model_id == "fusion":
        from .fusion import prepare_fusion
        artifact_dir = artifact_dir or Path(__file__).resolve().parents[3] / "weights/SatQuery_Model4_Handover/model"
        return prepare_fusion(paths, directory, artifact_dir)
    prepared = PreparedInputs(model_id, {})
    if model_id in {"rs_vlm", "grounding"}:
        data, info, _, _ = _read(paths[0])
        _, preview = optical_to_rgb(paths[0], directory / "display_rgb.png")
        rgb = _select_rgb(data, info)
        prepared.tensors["rgb"] = rgb
        native = directory / "native_rgb.png"
        Image.fromarray(np.rint(rgb).astype(np.uint8)).save(native)
        prepared.native_paths = [str(native)]; prepared.previews = [str(preview)]
        prepared.notes = [dict(operation="prepare_for", model_id=model_id, model_native="RGB HWC, 0–255; no percentile stretch", source=paths[0], native_path=str(native), preview=str(preview), shape=list(rgb.shape))]
        return prepared
    pair = check_pair(*paths, mode="before/after", modality_a="optical", modality_b="optical")
    prepared.warnings.extend(pair["warnings"])
    prepared.notes.append(dict(operation="check_pair", mode="before/after" if model_id == "change" else "optical+sar",
                               source_a=paths[0], source_b=paths[1], bounds=pair["bounds"], crs=pair["crs"], warnings=pair["warnings"]))
    for index, (name, info) in enumerate((("before", pair["info_a"]),("after", pair["info_b"]))):
        raw = pair["array_a"] if index == 0 else pair["array_b"]
        path = directory / f"aligned_{index}.tif"; _persist_raw(path, raw, info, pair)
        rgb = _select_rgb(raw, info)
        tensor, note = normalize_change(rgb, config)
        prepared.tensors[name] = tensor
        native = directory / f"native_{name}.png"
        Image.fromarray(np.rint(rgb).astype(np.uint8)).save(native)
        _, preview = optical_to_rgb(path, directory / f"display_{name}.png")
        prepared.native_paths.append(str(native)); prepared.previews.append(str(preview))
        prepared.notes.append(dict(note, input=name, source=paths[index], native_path=str(native), preview=str(preview), shape=list(tensor.shape)))
    if model_id == "change" and not (config or {}).get("confirmed", False):
        prepared.warnings.append("Chg2Cap preprocessing is UNCONFIRMED for this checkpoint; using the configured normalization without per-image percentile stretching.")
    return prepared
