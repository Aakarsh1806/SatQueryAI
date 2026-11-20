"""Framework-independent specialist lifecycle."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, TYPE_CHECKING

from ..schemas import SpecialistInput, StandardResult

if TYPE_CHECKING:
    from ..preprocessing.native import PreparedInputs


class SpecialistModel(ABC):
    @abstractmethod
    def load(self) -> None:
        """Load resources lazily."""

    @abstractmethod
    def unload(self) -> None:
        """Release resources; safe to call repeatedly."""

    @property
    @abstractmethod
    def is_loaded(self) -> bool:
        """Whether resources are loaded."""

    @abstractmethod
    def run(
        self, inputs: SpecialistInput | dict[str, Any] | PreparedInputs, query: str = "",
        options: dict[str, Any] | None = None,
    ) -> StandardResult:
        """Return the standard result for one query."""
