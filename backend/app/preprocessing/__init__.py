"""Rasterio/NumPy/Pillow preprocessing; no ML dependencies."""
from .imagery import (
    PreprocessingError, check_pair, domain_warnings, inspect,
    optical_to_rgb, sar_prepare,
)

__all__ = ["PreprocessingError", "check_pair", "domain_warnings", "inspect", "optical_to_rgb", "sar_prepare"]
from .native import PreparedInputs, normalize_change, prepare_for

__all__ += ["PreparedInputs", "normalize_change", "prepare_for"]

from .fusion import FusionInputError, read_fusion_pair
__all__ += ["FusionInputError", "read_fusion_pair"]
