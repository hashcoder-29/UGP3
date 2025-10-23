
"""
Placeholder for sentiment time series mapping.
Expose `make_step_fn(tgrid, values)` to build a callable sentiment_fn(t) used by the pricer/simulator.
"""
import numpy as np
from typing import Callable

def make_step_fn(tgrid: np.ndarray, values: np.ndarray) -> Callable[[np.ndarray], np.ndarray]:
    tgrid = np.asarray(tgrid, dtype=float)
    values = np.asarray(values, dtype=float)
    def fn(t: np.ndarray) -> np.ndarray:
        t = np.asarray(t, dtype=float)
        idx = np.minimum(np.searchsorted(tgrid, t, side="right")-1, len(values)-1)
        idx = np.maximum(idx, 0)
        return values[idx]
    return fn
