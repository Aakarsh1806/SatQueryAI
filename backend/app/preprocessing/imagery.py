"""Image inspection and preparation without ML dependencies.

Preparation returns (array, preview_path). Pair checking returns aligned raw
bands in (bands, height, width) order, with invalid samples represented by NaN.
"""
from pathlib import Path
import re
import warnings

import numpy as np
from PIL import Image
import rasterio
from rasterio.enums import Resampling
from rasterio.errors import NotGeoreferencedWarning
from rasterio.transform import from_bounds
from rasterio.warp import reproject, transform_bounds


class PreprocessingError(ValueError):
    """A user-readable input or alignment error."""


def _read(path):
    path = Path(path)
    if not path.is_file():
        raise PreprocessingError(f"Image file does not exist: {path}")
    if path.suffix.lower() not in {".tif", ".tiff", ".png", ".jpg", ".jpeg"}:
        raise PreprocessingError("Use GeoTIFF/TIFF, PNG, or JPG imagery.")
    try:
        # GDAL emits this for ordinary demo images; report it in our metadata.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", NotGeoreferencedWarning)
            with rasterio.open(path) as src:
                data = src.read(masked=True).astype("float32").filled(np.nan)
                data[~np.isfinite(data)] = np.nan
                geo = src.crs is not None and not src.transform.is_identity
                meta = dict(
                    path=str(path), band_count=src.count, dtype=src.dtypes[0],
                    dtypes=list(src.dtypes), width=src.width, height=src.height,
                    crs=src.crs.to_string() if src.crs else None,
                    resolution=list(src.res) if geo else None,
                    nodata=float(src.nodata) if src.nodata is not None and np.isfinite(src.nodata) else None,
                    bounds=list(src.bounds) if geo else None,
                    band_descriptions=list(src.descriptions), georeferenced=geo,
                    warnings=[] if geo else ["No georeferencing; CRS and footprint checks are skipped."],
                )
                return data, meta, src.transform, src.crs
    except (rasterio.errors.RasterioError, OSError) as exc:
        raise PreprocessingError(f"Cannot read image {path.name}: {exc}") from exc


def _describe(data, meta, modality_override=None):
    if modality_override not in {None, "optical", "sar"}:
        raise PreprocessingError("Modality override must be 'optical' or 'sar'.")
    ranges = []
    negative = False
    for band in data:
        valid = band[np.isfinite(band)]
        ranges.append({"min": float(valid.min()), "max": float(valid.max())} if valid.size else {"min": None, "max": None})
        # A negative median is a stronger dB hint than a single noisy pixel.
        negative |= bool(valid.size and -100 <= float(np.median(valid)) < 0)
    descriptions = " ".join(desc or "" for desc in meta["band_descriptions"])
    if re.search(r"(?<![a-z])(?:VV|VH)(?![a-z])", descriptions, re.I):
        guess, reason = "sar", "VV/VH polarization band descriptions suggest SAR."
    elif negative:
        guess, reason = "sar", "Negative median values in a dB-like range suggest SAR."
    elif meta["band_count"] <= 2:
        guess, reason = "sar", "One or two bands suggest SAR; grayscale optical images may need an override."
    else:
        guess, reason = "optical", "Three or more bands suggest optical imagery."
    return dict(meta, value_ranges=ranges, guessed_modality=guess,
                modality=modality_override or guess,
                modality_reason=f"User override: {modality_override}. Heuristic: {reason}" if modality_override else reason)


def inspect(path, modality_override=None):
    """Inspect valid pixels (excluding nodata/NaN); modality is only a guess."""
    data, meta, _, _ = _read(path)
    return _describe(data, meta, modality_override)


def _preview(array, source, output, suffix):
    target = Path(output) if output else Path(source).with_name(Path(source).stem + suffix + ".png")
    if target.resolve() == Path(source).resolve():
        raise PreprocessingError("Preview path must differ from the input image path.")
    target.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(array).save(target, format="PNG")
    return target


def _rgb_indices(meta):
    aliases = {
        "red": {"red", "r", "b4", "b04", "band4", "band04"},
        "green": {"green", "g", "b3", "b03", "band3", "band03"},
        "blue": {"blue", "b", "b2", "b02", "band2", "band02"},
    }
    found = {}
    for index, description in enumerate(meta["band_descriptions"]):
        # Accept descriptions such as "B04 (red)" or "Sentinel-2 B4".
        tokens = set(re.findall(r"[a-z]+\d*", (description or "").lower()))
        for color, names in aliases.items():
            if names & tokens:
                found[color] = index
    if len(found) == 3:
        return [found[color] for color in ("red", "green", "blue")]
    return [2, 1, 0] if meta["band_count"] >= 10 else [0, 1, 2]


def optical_to_rgb(path, preview_path=None, *, sentinel_full=False):
    """Return HWC uint8 RGB and PNG path using per-band 2–98% stretching.

    Undescribed 10+ band stacks are assumed to start B2,B3,B4 (not B1).
    Invalid pixels are black; constant bands become zero. Grayscale demo
    imagery is replicated into RGB, and a second band is not used as color.
    """
    data, meta, _, _ = _read(path)
    indices = [3, 2, 1] if sentinel_full and data.shape[0] == 13 else (_rgb_indices(meta) if data.shape[0] >= 3 else [0, 0, 0])
    rgb = np.zeros((meta["height"], meta["width"], 3), dtype=np.uint8)
    for channel, index in enumerate(indices):
        band = data[index]
        valid = np.isfinite(band)
        if not valid.any():
            continue
        low, high = np.percentile(band[valid], [2, 98])
        if high > low:
            stretched = np.clip((band[valid] - low) / (high - low), 0, 1)
            rgb[..., channel][valid] = np.rint(stretched * 255).astype(np.uint8)
    # Do not invent color where any selected RGB band has no valid data.
    rgb[~np.isfinite(data[indices]).all(axis=0)] = 0
    return rgb, _preview(rgb, path, preview_path, "_rgb")


def sar_prepare(path, preview_path=None, *, input_scale="auto", db_range=(-30.0, 5.0)):
    """Return normalized HWC float32 bands and grayscale PNG preview.

    Auto treats any negative valid value as dB; otherwise assumes linear
    power and applies 10*log10. Override input_scale for ambiguous data.
    Nonpositive linear power and nodata become zero in the normalized array.
    The preview is the first polarization; all bands remain in the array.
    """
    if input_scale not in {"auto", "linear", "db"}:
        raise PreprocessingError("SAR input_scale must be auto, linear, or db.")
    low, high = db_range
    if not np.isfinite([low, high]).all() or high <= low:
        raise PreprocessingError("SAR dB range must have finite increasing bounds.")
    data, _, _, _ = _read(path)
    valid = np.isfinite(data)
    if not valid.any():
        raise PreprocessingError("SAR image contains no valid pixels.")
    linear = input_scale == "linear" or (input_scale == "auto" and not (data[valid] < 0).any())
    db = data.copy()
    if linear:
        valid &= data > 0
        db[:] = np.nan
        db[valid] = 10 * np.log10(data[valid])
        if not valid.any():
            raise PreprocessingError("Linear SAR image contains no positive power values.")
    normalized = np.zeros_like(data, dtype=np.float32)
    normalized[valid] = (np.clip(db[valid], low, high) - low) / (high - low)
    array = np.moveaxis(normalized, 0, -1)
    preview = np.rint(array[..., 0] * 255).astype(np.uint8)
    return array, _preview(preview, path, preview_path, "_sar")


def _resize(data, width, height):
    """Resize band data and validity separately, preventing nodata bleed."""
    result = []
    for band in data:
        valid = np.isfinite(band)
        numerator = np.asarray(Image.fromarray(np.where(valid, band, 0).astype("float32")).resize((width, height), Image.Resampling.BILINEAR))
        denominator = np.asarray(Image.fromarray(valid.astype("float32")).resize((width, height), Image.Resampling.BILINEAR))
        result.append(np.divide(numerator, denominator, out=np.full_like(numerator, np.nan), where=denominator > 0))
    return np.stack(result)


def check_pair(path_a, path_b=None, mode="optical+sar", *, modality_a=None, modality_b=None):
    """Validate and align raw bands; never modify source images.

    Modes: optical+sar (A=optical, B=SAR), before/after (A=before, B=after).
    Georeferenced pairs are cropped to their bounding-box intersection and
    reprojected onto one grid. Different CRSs are transformed to A's CRS.
    Plain demo images are resized with an explicit unverified-footprint warning.
    """
    if mode not in {"optical+sar", "before/after", "change"}:
        raise PreprocessingError("Pair mode must be 'optical+sar' or 'before/after'.")
    if path_b is None:
        raise PreprocessingError("A single SAR image is not valid; provide an optical image and a SAR image. "
                                 "Before/after comparisons require two optical images.")
    a, ma, ta, ca = _read(path_a)
    b, mb, tb, cb = _read(path_b)
    if not np.isfinite(a).any() or not np.isfinite(b).any():
        raise PreprocessingError("One or both images contain no valid pixels; choose a pair with usable image data.")
    ia, ib = _describe(a, ma, modality_a), _describe(b, mb, modality_b)
    expected = ("optical", "sar") if mode == "optical+sar" else ("optical", "optical")
    if (ia["modality"], ib["modality"]) != expected:
        raise PreprocessingError(f"{mode} requires A={expected[0]} and B={expected[1]}; "
                                 "a SAR image alone is not valid. Override a mistaken modality guess explicitly.")
    messages = []
    geo = ma["georeferenced"] and mb["georeferenced"]
    target_transform, target_crs, overlap = None, None, None
    if geo:
        if ta.b != 0 or ta.d != 0 or tb.b != 0 or tb.d != 0 or ta.a <= 0 or ta.e >= 0 or tb.a <= 0 or tb.e >= 0:
            raise PreprocessingError("Pair alignment requires north-up rasters; reproject rotated or flipped imagery first.")
        ba = ma["bounds"]
        try:
            bb = transform_bounds(cb, ca, *mb["bounds"], densify_pts=21)
        except rasterio.errors.RasterioError as exc:
            raise PreprocessingError(f"Cannot compare pair footprints: {exc}") from exc
        overlap = [max(ba[0], bb[0]), max(ba[1], bb[1]), min(ba[2], bb[2]), min(ba[3], bb[3])]
        left, bottom, right, top = overlap
        if not np.isfinite(overlap).all() or left >= right or bottom >= top:
            raise PreprocessingError("The images do not overlap geographically; select imagery of the same area.")
        if not np.allclose(ba, overlap, rtol=0, atol=1e-8) or not np.allclose(bb, overlap, rtol=0, atol=1e-8):
            messages.append("Footprints only partially overlap; both images were cropped to their shared bounds.")
        if ca != cb:
            messages.append("CRS differs; image B was reprojected to image A's CRS.")
        width = max(1, round((right - left) / abs(ta.a)))
        height = max(1, round((top - bottom) / abs(ta.e)))
        if mode != "optical+sar":
            width = height = 256
        target_transform = from_bounds(*overlap, width, height)
        target_crs = ca.to_string()
        aligned = []
        for data, transform, crs in ((a, ta, ca), (b, tb, cb)):
            dest = np.full((data.shape[0], height, width), np.nan, dtype=np.float32)
            reproject(data, dest, src_transform=transform, src_crs=crs,
                      dst_transform=target_transform, dst_crs=ca,
                      src_nodata=np.nan, dst_nodata=np.nan, resampling=Resampling.bilinear)
            aligned.append(dest)
        a, b = aligned
        if not (np.isfinite(a).any(axis=0) & np.isfinite(b).any(axis=0)).any():
            raise PreprocessingError("Shared bounds contain no overlapping valid pixels.")
        if ma["width"] != width or ma["height"] != height or mb["width"] != width or mb["height"] != height or ta != tb:
            messages.append(f"Images were resampled onto a shared {width}x{height} grid.")
    else:
        messages.append("Georeferencing is missing for one or both images; CRS and overlap checks were skipped. Same-area correspondence is unverified.")
        height, width = a.shape[1:] if mode == "optical+sar" else (256, 256)
        for label, data in (("A", a), ("B", b)):
            if data.shape[1:] != (height, width):
                messages.append(f"Image {label} resized to {width}x{height}; geographic alignment is unverified.")
        a, b = _resize(a, width, height), _resize(b, width, height)
    return dict(array_a=a, array_b=b, info_a=ia, info_b=ib, warnings=messages,
                crs=target_crs, transform=target_transform, bounds=overlap,
                geospatial_checks_performed=geo)


def domain_warnings(info):
    """Model-domain caveats based on metadata, not a suitability guarantee."""
    result = list(info.get("warnings", []))
    if min(info["width"], info["height"]) < 256:
        result.append("Image is too small (under 256 pixels on a side); resizing cannot recover missing detail.")
    if max(info["width"], info["height"]) > 4096:
        result.append("Image is very large (over 4096 pixels on a side); tiling or downsampling may be needed.")
    resolution = info.get("resolution")
    if info.get("modality") == "optical" and resolution and info.get("crs"):
        crs = rasterio.crs.CRS.from_string(info["crs"])
        if crs.is_projected:
            _, meters_per_unit = crs.linear_units_factor
            if max(resolution) * meters_per_unit >= 10:
                result.append("Coarse optical resolution (10 m or coarser, such as Sentinel-2) may perform poorly with VRSBench/LEVIR-CC models trained mostly on high-resolution imagery.")
        else:
            result.append("Resolution is in angular units; metric ground resolution was not estimated.")
    result.append("Domain gap: models trained on VRSBench, LEVIR-CC, or Sentinel data may perform poorly on different imagery.")
    return result
