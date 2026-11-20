"""Deterministic demonstration outputs. No inference or ML dependencies."""
from time import perf_counter
from typing import Any

from PIL import Image
from pathlib import Path

import rasterio

from ..schemas import (
    ChangeInput, Confidence, FusionInput, FusionOutput, GroundingInput,
    GroundingOutput, Metadata, StandardResult, VLMInput,
)
from .base import SpecialistModel


class StubModel(SpecialistModel):
    input_schema = VLMInput

    def __init__(self, model_name: str, version: str = "stub-v1", fallback_reason: str | None = None):
        self.model_name = model_name
        self.version = version
        self.fallback_reason = fallback_reason
        self._loaded = False

    def load(self):
        self._loaded = True

    def unload(self):
        self._loaded = False

    @property
    def is_loaded(self):
        return self._loaded

    def run(self, inputs, query="", options=None):
        started = perf_counter()
        validated = self.input_schema.model_validate(inputs)
        parameters = dict(options or {})
        if not self.is_loaded:
            self.load()
        try:
            answer, evidence, confidence = self._predict(validated, query, parameters)
            return StandardResult(
                answer=answer, evidence=evidence, confidence=confidence,
                metadata=Metadata(
                    model_name=self.model_name, version=self.version,
                    runtime_ms=(perf_counter() - started) * 1000, device="cpu",
                    parameters=parameters, mode="stub", fallback_reason=self.fallback_reason,
                    confidence_reason="not available" if confidence is None else None,
                    warnings=["Synthetic stub output; no real model inference."] + (["Model 4 is a stub placeholder until real aligned pairs arrive; not yet verified on a real pair."] if self.input_schema is FusionInput else []),
                ),
            )
        finally:
            self.unload()

    def _predict(self, inputs, query, parameters):
        raise NotImplementedError


def image_size(path: str) -> tuple[int, int]:
    # Pillow cannot decode multi-band floating-point SAR TIFFs.
    if Path(path).suffix.lower() in {".tif", ".tiff"}:
        with rasterio.open(path) as image:
            return image.width, image.height
    with Image.open(path) as image:
        if image.format not in {"TIFF", "PNG", "JPEG"}:
            raise ValueError("Expected GeoTIFF/TIFF, PNG, or JPG imagery")
        return image.size


def fake_confidence(value: float) -> Confidence:
    return Confidence(value=value, calibrated=False, note="uncalibrated synthetic stub score; not model confidence")


class VLMStub(StubModel):
    def _predict(self, inputs, query, parameters):
        image_size(inputs.image_path)
        if inputs.task == "caption":
            parameters["prompt"] = "Describe this satellite image in detail."
            answer = "[STUB] A demonstration scene with buildings, roads, and vegetation."
        else:
            if not query.strip():
                raise ValueError("QA requires a nonempty question")
            suffix = " Answer with a short phrase."
            parameters["prompt"] = query if query.endswith(suffix) else query + suffix
            answer = "[STUB] Residential buildings."
        return answer, None, None


class GroundingStub(StubModel):
    input_schema = GroundingInput

    def _predict(self, inputs, query, parameters):
        width, height = image_size(inputs.image_path)
        parameters.update(text_prompt=inputs.text_prompt, box_threshold=inputs.box_threshold)
        templates = [(0.1, 0.1, 0.4, 0.4), (0.5, 0.2, 0.9, 0.6), (0.2, 0.65, 0.45, 0.95)]
        scores = [0.87, 0.72, 0.55]
        kept = [(box, score) for box, score in zip(templates, scores) if score >= inputs.box_threshold]
        evidence = GroundingOutput(
            boxes=[(a * width, b * height, c * width, d * height) for (a, b, c, d), _ in kept],
            labels=[inputs.text_prompt for _ in kept], scores=[score for _, score in kept],
        )
        return "[STUB] Synthetic demonstration boxes; no real object detection was performed.", evidence, None


class ChangeStub(StubModel):
    input_schema = ChangeInput

    def _predict(self, inputs, query, parameters):
        for path in (inputs.before_path, inputs.after_path):
            if image_size(path) != (256, 256):
                raise ValueError("Change imagery must be 256x256 (before=A, after=B)")
        return "[STUB] Synthetic example: several new buildings beside a road. This is not an observation of the uploaded pair.", None, None


class FusionStub(StubModel):
    input_schema = FusionInput

    def _predict(self, inputs, query, parameters):
        image_size(inputs.optical_path)
        image_size(inputs.sar_path)
        evidence = FusionOutput(label="Forest", confidence=0.68, top3=[("Forest", 0.68), ("Croplands", 0.22), ("Urban_Builtup", 0.10)])
        return evidence.label, evidence, fake_confidence(evidence.confidence)
