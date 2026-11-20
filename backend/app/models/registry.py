"""Validated model catalog with explicit, lazy real-model registration."""
import os
from pathlib import Path
from typing import Literal

import yaml
from dotenv import load_dotenv
from pydantic import Field

from ..schemas.contracts import Contract
from .base import SpecialistModel
from .stubs import ChangeStub, FusionStub, GroundingStub, VLMStub
from .rsvlm import RSVLMModel
from .chg2cap import Chg2CapModel
from .grounding import GroundingModel
from .fusion import FusionModel

DEFAULT_REAL_CLASSES = {"rs_vlm": RSVLMModel, "change": Chg2CapModel, "grounding": GroundingModel, "fusion": FusionModel}

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CONFIG = PROJECT_ROOT / "backend/models.yaml"
STUB_CLASSES = {"rs_vlm": VLMStub, "grounding": GroundingStub, "change": ChangeStub, "fusion": FusionStub}


class ModelEntry(Contract):
    id: Literal["rs_vlm", "grounding", "change", "fusion"]
    display_name: str
    accepted_inputs: list[Literal["1 optical", "2 optical", "optical+sar"]] = Field(min_length=1)
    tasks: list[str] = Field(min_length=1)
    output_type: Literal["text", "boxes", "class_top3"]
    version: str
    weights_path: str = Field(min_length=1)
    enabled: bool
    mode: Literal["stub", "real"]
    preprocessing: dict = Field(default_factory=dict)
    settings: dict = Field(default_factory=dict)
    validation_status: str = "not_evaluated"


class Catalog(Contract):
    models: list[ModelEntry]


class ModelRegistry:
    def __init__(self, config_path: str | Path | None = None, *, project_root: str | Path = PROJECT_ROOT,
                 real_classes: dict[str, type[SpecialistModel]] | None = None):
        self.project_root = Path(project_root)
        load_dotenv(self.project_root / ".env", override=False)
        config_path = config_path or os.getenv("SATQUERY_MODELS_CONFIG", str(DEFAULT_CONFIG))
        catalog = Catalog.model_validate(yaml.safe_load(Path(config_path).read_text()))
        for entry in catalog.models:
            prefix = "SATQUERY_" + entry.id.upper()
            if os.getenv(prefix + "_WEIGHTS_PATH"):
                entry.weights_path = os.environ[prefix + "_WEIGHTS_PATH"]
            if os.getenv(prefix + "_MODE"):
                entry.mode = os.environ[prefix + "_MODE"]
            device = os.getenv("SATQUERY_DEVICE", "auto")
            if device not in {"auto", "cuda", "mps", "cpu"}:
                raise ValueError("SATQUERY_DEVICE must be auto, cuda, mps or cpu")
            if entry.id != "fusion" and device != "auto":
                entry.settings["device"] = device
            # Revalidate environment overrides against the unchanged catalog contract.
            ModelEntry.model_validate(entry.model_dump())
        self.entries = {entry.id: entry for entry in catalog.models}
        if len(self.entries) != len(catalog.models):
            raise ValueError("Duplicate model IDs")
        self.real_classes = dict(DEFAULT_REAL_CLASSES if real_classes is None else real_classes)

    def resolve(self, model_id: str) -> dict:
        entry = self.entries[model_id]
        if not entry.enabled:
            raise ValueError(f"Model {model_id} is disabled")
        weights = self.project_root / entry.weights_path
        reason = entry.settings.get("activation_reason") if entry.mode == "stub" else None
        if entry.mode == "real":
            cls = self.real_classes.get(model_id)
            if not weights.exists(): reason = "weights_missing"
            elif cls is None: reason = "real_implementation_unavailable"
            elif hasattr(cls, "required_files") and not all((weights / name).is_file() for name in cls.required_files): reason = "weights_incomplete"
            elif hasattr(cls, "available") and not cls.available(): reason = "ml_dependencies_missing"
        return {"mode": "stub" if entry.mode == "stub" or reason else "real", "fallback_reason": reason}

    def get(self, model_id: str) -> SpecialistModel:
        entry = self.entries[model_id]
        resolution = self.resolve(model_id)
        if resolution["mode"] == "real":
            return self.real_classes[model_id](entry=entry, weights_path=self.project_root / entry.weights_path)
        return STUB_CLASSES[model_id](entry.display_name, entry.version, resolution["fallback_reason"])
