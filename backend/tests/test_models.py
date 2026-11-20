from pathlib import Path

import pytest
import yaml
from PIL import Image
from pydantic import ValidationError

from backend.app.models.registry import DEFAULT_CONFIG, ModelRegistry
from backend.app.schemas import Confidence, GroundingOutput, QueryRequest, QueryResponse, StandardResult


@pytest.fixture
def images(tmp_path):
    paths = {}
    for name, size in [("optical", (640, 320)), ("before", (256, 256)), ("after", (256, 256)), ("sar", (640, 320))]:
        path = tmp_path / f"{name}.png"
        Image.new("RGB", size).save(path)
        paths[name] = str(path)
    return paths


def stub_registry():
    registry = ModelRegistry()
    for entry in registry.entries.values(): entry.mode = "stub"
    return registry


def configured(tmp_path, **overrides):
    data = yaml.safe_load(DEFAULT_CONFIG.read_text())
    for entry in data["models"]: entry["mode"] = "stub"
    data["models"][0].update(overrides)
    path = tmp_path / "models.yaml"
    path.write_text(yaml.safe_dump(data))
    return path


def test_registry_loads():
    registry = stub_registry()
    assert set(registry.entries) == {"rs_vlm", "grounding", "change", "fusion"}
    assert all(not registry.get(key).is_loaded for key in registry.entries)


@pytest.mark.parametrize("model_id", ["rs_vlm", "grounding", "change", "fusion"])
def test_stub_contracts(model_id, images):
    inputs = {
        "rs_vlm": {"image_path": images["optical"]},
        "grounding": {"image_path": images["optical"], "text_prompt": "buildings"},
        "change": {"before_path": images["before"], "after_path": images["after"]},
        "fusion": {"optical_path": images["optical"], "sar_path": images["sar"]},
    }[model_id]
    request = QueryRequest(model_id=model_id, inputs=inputs, query="What is visible?")
    model = stub_registry().get(model_id)
    model.load()
    assert model.is_loaded
    result = model.run(request.inputs, request.query)
    response = QueryResponse(request_id="test", result=result)
    restored = StandardResult.model_validate_json(response.result.model_dump_json())
    assert restored.metadata.mode == "stub"
    assert restored.metadata.runtime_ms >= 0
    assert restored.metadata.device == "cpu"
    assert not model.is_loaded
    model.unload()
    if result.confidence is None:
        assert result.metadata.confidence_reason == "not available"
    else:
        assert not result.confidence.calibrated
        assert "synthetic stub" in result.confidence.note
    if model_id == "grounding":
        assert len(result.evidence.boxes) == 3
        assert result.evidence.boxes[0] == (64, 32, 256, 128)
        for x1, y1, x2, y2 in result.evidence.boxes:
            assert 0 <= x1 < x2 <= 640
            assert 0 <= y1 < y2 <= 320
    if model_id == "fusion":
        assert len(result.evidence.top3) == 3


def test_missing_weights_never_instantiates_real(tmp_path, images):
    class ForbiddenReal:
        def __init__(self, **kwargs):
            pytest.fail("Real class must not be instantiated without weights")
    registry = ModelRegistry(configured(tmp_path, mode="real"), project_root=tmp_path,
                             real_classes={"rs_vlm": ForbiddenReal})
    result = registry.get("rs_vlm").run({"image_path": images["optical"]}, "What is here?")
    assert result.metadata.mode == "stub"
    assert result.metadata.fallback_reason == "weights_missing"


def test_real_selection_requires_both_conditions(tmp_path):
    weights = tmp_path / "weights/satquery_rsvlm_lora"
    weights.mkdir(parents=True)
    class RealAdapter:
        def __init__(self, *, entry, weights_path):
            self.entry = entry
            self.weights_path = weights_path
    stub = ModelRegistry(configured(tmp_path), project_root=tmp_path,
                         real_classes={"rs_vlm": RealAdapter}).get("rs_vlm")
    assert not isinstance(stub, RealAdapter)
    real = ModelRegistry(configured(tmp_path, mode="real"), project_root=tmp_path,
                         real_classes={"rs_vlm": RealAdapter}).get("rs_vlm")
    assert isinstance(real, RealAdapter)
    assert real.weights_path == weights
    fallback = ModelRegistry(configured(tmp_path, mode="real"), project_root=tmp_path, real_classes={}).get("rs_vlm")
    assert fallback.fallback_reason == "real_implementation_unavailable"


def test_prompt_rules(images):
    model = stub_registry().get("rs_vlm")
    qa = model.run({"image_path": images["optical"]}, "What is here?")
    assert qa.metadata.parameters["prompt"] == "What is here? Answer with a short phrase."
    caption = model.run({"image_path": images["optical"], "task": "caption"})
    assert caption.metadata.parameters["prompt"] == "Describe this satellite image in detail."
    assert caption.evidence is None


def test_threshold_and_cleanup(images):
    model = stub_registry().get("grounding")
    result = model.run({"image_path": images["optical"], "text_prompt": "roads", "box_threshold": 0.8})
    assert len(result.evidence.boxes) == 1
    result = model.run({"image_path": images["optical"], "text_prompt": "roads", "box_threshold": 1})
    assert result.evidence.boxes == []
    with pytest.raises(FileNotFoundError):
        model.run({"image_path": "missing.png", "text_prompt": "roads"})
    assert not model.is_loaded


def test_invalid_inputs(images):
    registry = stub_registry()
    with pytest.raises(ValueError, match="256x256"):
        registry.get("change").run({"before_path": images["optical"], "after_path": images["after"]})
    with pytest.raises(ValidationError):
        registry.get("fusion").run({"sar_path": images["sar"]})
    with pytest.raises(ValidationError):
        GroundingOutput(boxes=[(2, 0, 1, 4)], labels=["road"], scores=[0.5])
    with pytest.raises(ValidationError):
        Confidence(value=0.9, calibrated=False, note="softmax")


def test_disabled_and_duplicate_entries(tmp_path):
    with pytest.raises(ValueError, match="disabled"):
        ModelRegistry(configured(tmp_path, enabled=False)).get("rs_vlm")
    path = configured(tmp_path)
    data = yaml.safe_load(path.read_text())
    data["models"].append(data["models"][0])
    path.write_text(yaml.safe_dump(data))
    with pytest.raises(ValueError, match="Duplicate"):
        ModelRegistry(path)
