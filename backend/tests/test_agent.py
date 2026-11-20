import json
from pathlib import Path

import numpy as np
from PIL import Image
import pytest
import rasterio
from rasterio.transform import from_origin

from backend.app.agent import Agent, AgentInputError, detect_task
from backend.app.models.registry import ModelRegistry
from backend.app.models.stubs import VLMStub


# 30 mixed cases: expected task (or a displayable rejection).
ROUTING_CASES = [
    (["optical"], "What is visible?", "qa"),
    (["optical"], "How many buildings are there?", "qa"),
    (["optical"], "Is there a river?", "qa"),
    (["optical"], "What color is the roof?", "qa"),
    (["optical"], "Describe this image", "caption"),
    (["optical"], "Caption the scene", "caption"),
    (["optical"], "Summarize the imagery", "caption"),
    (["optical"], "Summarise this scene", "caption"),
    (["optical"], "Where are the buildings?", "grounding"),
    (["optical"], "Locate the bridge", "grounding"),
    (["optical"], "Find roads", "grounding"),
    (["optical"], "Highlight houses", "grounding"),
    (["optical"], "Show the airport", "grounding"),
    (["optical"], "DETECT vehicles", "grounding"),
    (["optical"], "Describe where the river runs", "caption"),
    (["optical", "optical"], "What changed?", "change_caption"),
    (["optical", "optical"], "Describe the differences", "change_caption"),
    (["optical", "optical"], "How many buildings were added?", "change_caption"),
    (["optical", "optical"], "Find new roads", "change_caption"),
    (["optical", "optical"], "Produce a change map", "change_caption"),
    (["optical", "sar"], "What land cover is this?", "classification"),
    (["sar", "optical"], "Describe the area", "classification"),
    (["optical", "sar"], "Find the forest", "classification"),
    (["sar"], "Describe this image", "error"),
    (["sar", "sar"], "Compare these", "error"),
    (["optical", "optical", "optical"], "Caption these", "error"),
    (["optical"], "What changed?", "error"),
    (["optical"], "Compare before and after", "error"),
    ([], "Describe", "error"),
    (["optical"], " ", "error"),
]


def test_routing_accuracy():
    matches = []
    errors = []
    for modalities, query, expected in ROUTING_CASES:
        try:
            actual = detect_task(modalities, query).task
        except AgentInputError as exc:
            actual = "error"
            assert str(exc)
        matches.append(actual == expected)
        if actual != expected:
            errors.append((modalities, query, expected, actual))
    accuracy = sum(matches) / len(matches)
    print(f"Routing accuracy: {sum(matches)}/{len(matches)} = {accuracy:.1%}")
    assert len(ROUTING_CASES) == 30
    assert accuracy == 1, errors


@pytest.fixture
def files(tmp_path):
    optical, after, sar = [tmp_path / name for name in ("optical.png", "after.png", "sar.tif")]
    ramp = np.arange(20 * 16, dtype="uint16").reshape(16, 20).astype("uint8")
    Image.fromarray(np.stack([ramp, ramp[::-1], ramp], axis=-1)).save(optical)
    Image.open(optical).resize((24, 20)).save(after)
    with rasterio.open(sar, "w", driver="GTiff", width=20, height=16, count=2,
                       dtype="float32", transform=from_origin(10, 50, 1, 1)) as dst:
        dst.write(np.full((2, 16, 20), -12, dtype="float32"))
        dst.descriptions = ("VV", "VH")
    return optical, after, sar


@pytest.fixture
def agent(tmp_path):
    registry = ModelRegistry()
    for entry in registry.entries.values(): entry.mode = "stub"
    return Agent(registry, trace_dir=tmp_path / "traces", artifact_dir=tmp_path / "prepared")


@pytest.mark.parametrize("task,query,indices", [
    ("qa", "What is visible?", [0]),
    ("caption", "Describe the scene", [0]),
    ("grounding", "Where are roads?", [0]),
    ("change_caption", "How many buildings changed?", [0, 1]),
    ("classification", "What land cover?", [2, 0]),
])
def test_pipeline_success(task, query, indices, files, agent):
    calls = []
    original = agent.registry.get
    def get(model_id):
        calls.append(model_id)
        return original(model_id)
    agent.registry.get = get
    response = agent.run([files[index] for index in indices], query)
    assert response.success, response.error
    assert response.routing.task == task
    assert calls == [response.routing.model_id]
    assert response.result.metadata.mode == "stub"
    assert "Stub" in response.summary
    assert response.warnings
    assert all(Path(path).exists() for path in response.previews)
    trace = json.loads(Path(response.trace_path).read_text())
    assert trace["request_id"] == response.request_id
    assert trace["timestamp"].endswith("+00:00")
    assert trace["query"] == query
    assert trace["validation"]["valid"]
    assert len(trace["detected_modalities"]) == len(indices)
    assert trace["routing"]["model_id"] == calls[0]
    assert trace["chosen_model"]["mode"] == "stub"
    assert trace["preprocessing_steps"]
    assert trace["parameters"] == response.result.metadata.parameters
    assert trace["device"] == "cpu"
    assert trace["runtime_ms"] >= 0
    assert trace["output"]["success"] and not trace["errors"]
    if task == "qa":
        assert trace["parameters"]["prompt"] == query + " Answer with a short phrase."
    if task == "caption":
        assert trace["parameters"]["prompt"] == "Describe this satellite image in detail."
    if task == "grounding":
        assert response.result.evidence.boxes[0] == (2, 1.6, 8, 6.4)
    if task == "change_caption":
        assert all(Image.open(path).size == (256, 256) for path in response.previews)
        assert any("not a direct answer" in warning for warning in response.warnings)
    if task == "classification":
        assert response.result.evidence.top3[0][0] == response.result.answer
        assert "uncalibrated" in response.result.confidence.note
        step = trace["preprocessing_steps"][-1]
        with rasterio.open(step["normalized_path"]) as src:
            assert src.count == 2
            assert 0 <= src.read().min() <= src.read().max() <= 1


@pytest.mark.parametrize("indices,query,expected", [
    ([2], "Describe", "SAR-only"),
    ([0, 1, 0], "Describe", "at most two"),
    ([0], "Detect changes", "needs two images"),
    ([], "Describe", "one or two"),
    ([0], "", "Enter a question"),
])
def test_validation_failures_are_traced(indices, query, expected, files, agent):
    agent.registry.get = lambda _: pytest.fail("No model may be instantiated on validation failure")
    response = agent.run([files[index] for index in indices], query)
    assert not response.success and expected in response.error
    trace = json.loads(Path(response.trace_path).read_text())
    assert not trace["validation"]["valid"]
    assert trace["errors"][0]["stage"] == "validation"
    assert trace["chosen_model"] is None


def test_format_override_and_missing_file(files, agent, tmp_path):
    missing = agent.run([tmp_path / "missing.tif"], "Describe")
    assert not missing.success and "does not exist" in missing.error
    wrong = tmp_path / "image.gif"
    Image.new("RGB", (10, 10)).save(wrong)
    assert "GeoTIFF" in agent.run([wrong], "Describe").error
    grayscale = tmp_path / "gray.png"
    Image.new("L", (16, 16), color=15).save(grayscale)
    assert not agent.run([grayscale], "Describe").success
    response = agent.run([{"path": str(grayscale), "modality_override": "optical"}], "Describe")
    assert response.success
    trace = json.loads(Path(response.trace_path).read_text())
    assert trace["detected_modalities"][0]["guess"] == "sar"
    assert trace["detected_modalities"][0]["modality"] == "optical"


@pytest.mark.parametrize("fail_at", ["load", "run"])
def test_model_exception_unloads_and_traces(fail_at, files, agent):
    class Broken(VLMStub):
        def load(self):
            super().load()
            if fail_at == "load":
                raise RuntimeError("Demo model load failed")
        def run(self, *args, **kwargs):
            raise RuntimeError("Demo model execution failed")
    broken = Broken("broken")
    agent.registry.get = lambda _: broken
    response = agent.run([files[0]], "What is visible?")
    assert not response.success and "failed" in response.error
    assert not broken.is_loaded
    trace = json.loads(Path(response.trace_path).read_text())
    assert trace["errors"][0]["stage"] == ("model_load" if fail_at == "load" else "model")
    assert trace["chosen_model"]["mode"] == "stub"


def test_llm_hook_opt_in_and_rules_priority():
    calls = []
    def hook(query):
        calls.append(query)
        return "grounding"
    assert detect_task(["optical"], "Buildings", llm_fallback=hook).task == "qa"
    assert not calls
    assert detect_task(["optical"], "Buildings", llm_fallback=hook, enable_llm_fallback=True).task == "grounding"
    assert len(calls) == 1
    assert detect_task(["optical"], "Describe buildings", llm_fallback=hook, enable_llm_fallback=True).task == "caption"
    assert len(calls) == 1
    def broken(query):
        raise RuntimeError("offline")
    assert detect_task(["optical"], "Buildings", llm_fallback=broken, enable_llm_fallback=True).task == "qa"


def test_geographic_rejection_and_trace(agent, tmp_path):
    paths = []
    for index, x in enumerate([500000, 600000]):
        path = tmp_path / f"geo{index}.tif"
        with rasterio.open(path, "w", driver="GTiff", width=8, height=8, count=3, dtype="uint8",
                           crs="EPSG:32643", transform=from_origin(x, 2000000, 10, 10)) as dst:
            dst.write(np.ones((3, 8, 8), dtype="uint8"))
        paths.append(path)
    response = agent.run(paths, "Describe changes")
    assert not response.success and "do not overlap" in response.error
    trace = json.loads(Path(response.trace_path).read_text())
    assert trace["routing"]["model_id"] == "change"
    assert trace["errors"][0]["stage"] == "preprocessing"


def test_registry_fallback_recorded_in_trace(files, agent, tmp_path):
    import yaml
    from backend.app.models.registry import DEFAULT_CONFIG
    config = yaml.safe_load(DEFAULT_CONFIG.read_text())
    config["models"][0]["mode"] = "real"
    path = tmp_path / "models.yaml"
    path.write_text(yaml.safe_dump(config))
    agent.registry = ModelRegistry(path, project_root=tmp_path)
    response = agent.run([files[0]], "Describe")
    assert response.success
    assert response.result.metadata.fallback_reason == "weights_missing"
    trace = json.loads(Path(response.trace_path).read_text())
    assert trace["chosen_model"]["configured_mode"] == "real"
    assert trace["chosen_model"]["mode"] == "stub"
    assert trace["chosen_model"]["fallback_reason"] == "weights_missing"


def test_trace_storage_failure_is_explicit(files, agent, tmp_path):
    blocked = tmp_path / "blocked"
    blocked.write_text("not a directory")
    agent.trace_dir = blocked
    response = agent.run([files[0]], "Describe")
    assert not response.success
    assert "Could not save the execution trace" in response.error
    assert response.trace_path is None


def test_concurrent_requests_hold_only_one_model(files, agent):
    from concurrent.futures import ThreadPoolExecutor
    from threading import Barrier, Lock
    import time
    gate = Barrier(2)
    lock = Lock()
    active = 0
    maximum = 0
    class Observed(VLMStub):
        def load(self):
            nonlocal active, maximum
            with lock:
                if not self.is_loaded:
                    active += 1
                    maximum = max(maximum, active)
            super().load()
        def run(self, *args, **kwargs):
            time.sleep(.02)
            return super().run(*args, **kwargs)
        def unload(self):
            nonlocal active
            with lock:
                if self.is_loaded:
                    active -= 1
            super().unload()
    agent.registry.get = lambda _: Observed("observed")
    def request():
        gate.wait(timeout=5)
        return agent.run([files[0]], "What is visible?")
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(request) for _ in range(2)]
        results = [future.result(timeout=10) for future in futures]
    assert all(result.success for result in results)
    assert maximum == 1 and active == 0
    assert results[0].request_id != results[1].request_id
    assert all(Path(result.trace_path).exists() for result in results)


def test_cancel_after_loading_unloads_model(files, agent):
    from threading import Event
    event = Event()
    model = VLMStub('Test', 'test')
    original = model.load
    def load():
        original()
        event.set()
    model.load = load
    agent.registry.get = lambda model_id: model
    response = agent.run([files[0]], 'Describe this image', cancelled=event.is_set)
    assert not response.success
    assert 'cancelled' in response.error
    assert not model.is_loaded
    assert json.loads(Path(response.trace_path).read_text())['cancelled']


def test_only_one_model_loaded_across_agents(files, tmp_path):
    from concurrent.futures import ThreadPoolExecutor
    from threading import Event, Lock
    from time import sleep
    guard = Lock()
    state = {'active': 0, 'peak': 0}
    entered = Event()
    class ObservedStub(VLMStub):
        def load(self):
            with guard:
                state['active'] += 1
                state['peak'] = max(state['peak'], state['active'])
            super().load()
        def run(self, *args, **kwargs):
            entered.set()
            sleep(.04)  # overlap the concurrent requests, not mirrored implementation
            return super().run(*args, **kwargs)
        def unload(self):
            with guard:
                if self.is_loaded: state['active'] -= 1
            super().unload()
    agents = [Agent(trace_dir=tmp_path/'traces', artifact_dir=tmp_path/f'prepared{i}') for i in range(2)]
    for agent in agents:
        agent.registry.get = lambda model_id: ObservedStub('Observed', 'test')
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(agent.run, [files[0]], 'Describe this image') for agent in agents]
        responses = [future.result() for future in futures]
    assert all(response.success for response in responses)
    assert state == {'active': 0, 'peak': 1}
    assert all(json.loads(Path(response.trace_path).read_text())['lifecycle']['unloaded'] for response in responses)
