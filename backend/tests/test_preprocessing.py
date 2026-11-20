import json

import numpy as np
from PIL import Image
import pytest
import rasterio
from rasterio.transform import from_origin

from backend.app.preprocessing import (
    PreprocessingError, check_pair, domain_warnings, inspect,
    optical_to_rgb, sar_prepare,
)


@pytest.fixture
def raster(tmp_path):
    def write(name, data, *, transform=None, crs="EPSG:32643", nodata=None, descriptions=None):
        path = tmp_path / name
        with rasterio.open(path, "w", driver="GTiff", width=data.shape[2], height=data.shape[1],
                           count=data.shape[0], dtype=data.dtype, crs=crs,
                           transform=transform or from_origin(500000, 2000000, 10, 10), nodata=nodata) as dst:
            dst.write(data)
            if descriptions:
                dst.descriptions = descriptions
        return path
    return write


@pytest.fixture
def imagery(raster):
    ramp = np.arange(16 * 12, dtype="float32").reshape(12, 16)
    optical = raster("optical.tif", np.stack([ramp, ramp * 2, ramp[::-1]]))
    many = raster("many.tif", np.stack([ramp + index for index in range(13)]))
    sar = raster("sar.tif", np.stack([np.full_like(ramp, -10), np.full_like(ramp, -20)]), descriptions=("VV", "VH"))
    far = raster("far.tif", np.stack([ramp] * 3), transform=from_origin(600000, 2000000, 10, 10))
    return optical, many, sar, far


def test_inspect_and_override(imagery):
    optical, many, sar, _ = imagery
    info = inspect(optical)
    assert (info["band_count"], info["dtype"], info["width"], info["height"]) == (3, "float32", 16, 12)
    assert info["crs"] == "EPSG:32643"
    assert info["resolution"] == [10, 10]
    assert info["nodata"] is None
    assert info["value_ranges"][0] == {"min": 0, "max": 191}
    assert info["modality"] == "optical"
    assert inspect(many)["band_count"] == 13
    assert inspect(sar)["modality"] == "sar"
    assert "VV/VH" in inspect(sar)["modality_reason"]
    override = inspect(sar, modality_override="optical")
    assert override["modality"] == "optical" and override["guessed_modality"] == "sar"
    assert "User override" in override["modality_reason"]
    json.dumps(info, allow_nan=False)
    with pytest.raises(PreprocessingError, match="override"):
        inspect(optical, "radar")


def test_modality_heuristics_and_nodata(raster):
    values = np.full((3, 4, 4), -15, dtype="float32")
    values[:, 0, 0] = -9999
    path = raster("negative.tif", values, nodata=-9999)
    info = inspect(path)
    assert info["modality"] == "sar" and "Negative" in info["modality_reason"]
    assert info["value_ranges"][0] == {"min": -15, "max": -15}
    single = raster("one.tif", np.ones((1, 4, 4), dtype="uint8"))
    assert inspect(single)["modality"] == "sar"
    empty = raster("empty.tif", np.full((3, 4, 4), -9999, dtype="float32"), nodata=-9999)
    assert inspect(empty)["value_ranges"][0] == {"min": None, "max": None}
    rgb, _ = optical_to_rgb(empty)
    assert not rgb.any()
    with pytest.raises(PreprocessingError, match="no valid"):
        sar_prepare(empty)


def stretched(band):
    low, high = np.percentile(band, [2, 98])
    return np.rint(np.clip((band - low) / (high - low), 0, 1) * 255).astype("uint8")


def test_rgb_selection_stretch_preview(imagery, raster, tmp_path):
    optical, many, _, _ = imagery
    rgb, preview = optical_to_rgb(optical)
    assert rgb.shape == (12, 16, 3) and rgb.dtype == np.uint8
    assert preview.exists()
    np.testing.assert_array_equal(np.asarray(Image.open(preview)), rgb)
    with rasterio.open(optical) as src:
        np.testing.assert_array_equal(rgb[..., 0], stretched(src.read(1)))
    # Give bands distinct spatial patterns so band order is observable.
    rng = np.random.default_rng(4)
    data = rng.uniform(0, 1000, size=(13, 12, 16)).astype("float32")
    path = raster("sentinel.tif", data)
    rgb, _ = optical_to_rgb(path)
    for channel, index in enumerate([2, 1, 0]):
        np.testing.assert_array_equal(rgb[..., channel], stretched(data[index]))
    desc = [""] * 13
    desc[7], desc[5], desc[9] = "B04 (red)", "B03", "B02"
    described = raster("described.tif", data, descriptions=desc)
    rgb, preview = optical_to_rgb(described, tmp_path / "previews/custom.png")
    for channel, index in enumerate([7, 5, 9]):
        np.testing.assert_array_equal(rgb[..., channel], stretched(data[index]))
    assert preview.name == "custom.png"
    assert optical_to_rgb(many)[0].shape == (12, 16, 3)
    with pytest.raises(PreprocessingError, match="differ"):
        optical_to_rgb(optical, optical)


def test_sar_linear_db_and_invalid(raster, imagery):
    _, _, sar, _ = imagery
    db, preview = sar_prepare(sar)
    assert db.shape == (12, 16, 2) and db.dtype == np.float32
    np.testing.assert_allclose(db[0, 0], [20 / 35, 10 / 35])
    assert Image.open(preview).mode == "L"
    values = np.array([[[0.001, 0.01, 0.1, 1, 10, 0, -9999]]], dtype="float32")
    linear = raster("power.tif", values, nodata=-9999)
    normalized, _ = sar_prepare(linear)
    np.testing.assert_allclose(normalized[0, :, 0], [0, 10/35, 20/35, 30/35, 1, 0, 0], atol=1e-6)
    # Explicit scale handles negative noise or all-positive dB data.
    positive_db = raster("positive_db.tif", np.ones((1, 2, 2), dtype="float32"))
    normalized, _ = sar_prepare(positive_db, input_scale="db")
    np.testing.assert_allclose(normalized, 31/35)
    with pytest.raises(PreprocessingError, match="range"):
        sar_prepare(sar, db_range=(5, -30))
    with pytest.raises(PreprocessingError, match="input_scale"):
        sar_prepare(sar, input_scale="unknown")
    zeros = raster("zero.tif", np.zeros((1, 2, 2), dtype="float32"))
    with pytest.raises(PreprocessingError, match="positive"):
        sar_prepare(zeros)


def test_pair_same_area_and_resampling(imagery, raster):
    optical, _, sar, _ = imagery
    pair = check_pair(optical, sar, "optical+sar")
    assert pair["array_a"].shape == (3, 12, 16)
    assert pair["array_b"].shape == (2, 12, 16)
    assert pair["geospatial_checks_performed"]
    assert pair["crs"] == "EPSG:32643"
    small_sar = raster("small_sar.tif", np.full((2, 6, 8), -12, dtype="float32"),
                       transform=from_origin(500000, 2000000, 20, 20))
    aligned = check_pair(optical, small_sar, "optical+sar")
    assert aligned["array_b"].shape == (2, 12, 16)
    np.testing.assert_allclose(aligned["array_b"], -12)
    assert any("resampled" in warning for warning in aligned["warnings"])
    change = check_pair(optical, optical, "before/after")
    assert change["array_a"].shape == change["array_b"].shape == (3, 256, 256)
    assert any("256x256" in warning for warning in change["warnings"])


def test_pair_overlap_crop_and_reject(imagery, raster):
    optical, _, sar, far = imagery
    with pytest.raises(PreprocessingError, match="do not overlap"):
        check_pair(optical, far, "before/after")
    with pytest.raises(PreprocessingError, match="single SAR"):
        check_pair(sar)
    with pytest.raises(PreprocessingError, match="requires"):
        check_pair(sar, sar, "optical+sar")
    with pytest.raises(PreprocessingError, match="mode"):
        check_pair(optical, optical, "nonsense")
    offset = raster("offset.tif", np.full((2, 12, 16), -10, dtype="float32"),
                    transform=from_origin(500080, 2000000, 10, 10))
    pair = check_pair(optical, offset, "optical+sar")
    assert pair["array_a"].shape == (3, 12, 8)
    assert pair["array_b"].shape == (2, 12, 8)
    assert pair["bounds"] == [500080, 1999880, 500160, 2000000]
    assert any("cropped" in warning for warning in pair["warnings"])
    with rasterio.open(optical) as src:
        np.testing.assert_allclose(pair["array_a"], src.read()[:, :, 8:])
    # Touching edges do not constitute overlap.
    touching = raster("touching.tif", np.ones((3, 12, 16), dtype="float32"),
                      transform=from_origin(500160, 2000000, 10, 10))
    with pytest.raises(PreprocessingError, match="do not overlap"):
        check_pair(optical, touching, "before/after")


def test_different_crs_alignment(raster):
    from rasterio.warp import transform_bounds
    left, bottom, right, top = transform_bounds("EPSG:4326", "EPSG:3857", 75, 19, 75.01, 19.01)
    optical = raster("degrees.tif", np.ones((3, 10, 10), dtype="float32"),
                     crs="EPSG:4326", transform=from_origin(75, 19.01, .001, .001))
    sar = raster("meters.tif", np.full((2, 10, 10), -10, dtype="float32"),
                 crs="EPSG:3857", transform=from_origin(left, top, (right-left)/10, (top-bottom)/10))
    pair = check_pair(optical, sar)
    assert pair["array_b"].shape == (2, 10, 10)
    assert any("CRS differs" in warning for warning in pair["warnings"])
    assert np.isfinite(pair["array_b"]).all()


@pytest.mark.parametrize("extension", ["png", "jpg"])
def test_plain_images(tmp_path, extension):
    a, b = tmp_path / f"a.{extension}", tmp_path / f"b.{extension}"
    Image.new("RGB", (20, 16), color=(10, 20, 30)).save(a)
    Image.new("L", (10, 8), color=4).save(b)
    info = inspect(a)
    assert info["crs"] is None and info["resolution"] is None
    assert info["warnings"] and info["modality"] == "optical"
    assert optical_to_rgb(a)[0].shape == (16, 20, 3)
    assert sar_prepare(b)[0].shape == (8, 10, 1)
    pair = check_pair(a, b)
    assert pair["array_b"].shape == (1, 16, 20)
    assert not pair["geospatial_checks_performed"]
    assert any("skipped" in warning for warning in pair["warnings"])
    # A grayscale demo optical is supported through an explicit override.
    change = check_pair(a, b, "before/after", modality_b="optical")
    assert change["array_a"].shape[1:] == change["array_b"].shape[1:] == (256, 256)
    assert optical_to_rgb(b)[0].shape == (8, 10, 3)


def test_domain_warnings(imagery):
    info = inspect(imagery[0])
    messages = domain_warnings(info)
    assert any("Coarse" in message for message in messages)
    assert any("too small" in message for message in messages)
    assert any("Domain gap" in message for message in messages)
    info.update(width=5000, height=5000)
    assert any("very large" in message for message in domain_warnings(info))
    info.update(crs="EPSG:4326", resolution=[.0001, .0001])
    assert any("angular" in message for message in domain_warnings(info))
    assert not any("Coarse" in message for message in domain_warnings(info))


def test_missing_corrupt_and_unsupported(tmp_path):
    with pytest.raises(PreprocessingError, match="does not exist"):
        inspect(tmp_path / "missing.tif")
    invalid = tmp_path / "invalid.tif"
    invalid.write_text("not an image")
    with pytest.raises(PreprocessingError, match="Cannot read"):
        inspect(invalid)
    wrong = tmp_path / "wrong.gif"
    Image.new("RGB", (2, 2)).save(wrong)
    with pytest.raises(PreprocessingError, match="Use GeoTIFF"):
        inspect(wrong)


def test_pair_nodata_and_resize_masks(raster, imagery, tmp_path):
    optical = imagery[0]
    empty = raster("empty_pair.tif", np.full((2, 12, 16), -9999, dtype="float32"), nodata=-9999)
    with pytest.raises(PreprocessingError, match="no valid pixels"):
        check_pair(optical, empty)
    # Bounding boxes can overlap while the valid pixels do not.
    a = np.full((3, 12, 16), -9999, dtype="float32")
    b = np.full((2, 12, 16), -9999, dtype="float32")
    a[:, :, :4] = 10
    b[:, :, 12:] = -10
    pa = raster("left_valid.tif", a, nodata=-9999)
    pb = raster("right_valid.tif", b, nodata=-9999)
    with pytest.raises(PreprocessingError, match="no overlapping valid pixels"):
        check_pair(pa, pb)
    # Missing georeferencing uses mask-weighted resizing rather than blending nodata.
    rgb = tmp_path / "plain.png"
    Image.new("RGB", (32, 24), color=(10, 20, 30)).save(rgb)
    aligned = check_pair(rgb, pb)
    finite = aligned["array_b"][np.isfinite(aligned["array_b"])]
    np.testing.assert_allclose(finite, -10)
    assert np.isnan(aligned["array_b"]).any()


def test_constant_rgb_and_nonfinite(raster):
    data = np.ones((3, 4, 4), dtype="float32")
    data[:, 0, 0] = np.nan
    data[:, 0, 1] = np.inf
    path = raster("nonfinite.tif", data)
    assert inspect(path)["value_ranges"][0] == {"min": 1, "max": 1}
    rgb, _ = optical_to_rgb(path)
    assert not rgb.any()
