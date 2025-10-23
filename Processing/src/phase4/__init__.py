"""
Phase-4: sentiment-driven Cox intensity over Kou jump-diffusion.
Convenience exports so you can `from src.phase4 import ...`.
"""

from .cox_kou_mc import (
    DiffusionParams,
    KouJumpParams,
    IntensityDynamics,
    simulate_paths_mc,
    price_european_mc,
)
from .calibration_stage2 import (
    MarketOption,
    calibrate_intensity,
)
from .sentiment_signal import make_step_fn

__all__ = [
    "DiffusionParams",
    "KouJumpParams",
    "IntensityDynamics",
    "simulate_paths_mc",
    "price_european_mc",
    "MarketOption",
    "calibrate_intensity",
    "make_step_fn",
]
