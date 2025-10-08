"""
Optional JAX backend switch.

- Default: NumPy (no extra deps).
- If env USE_JAX=1 and JAX imports, use JAX with XLA JIT and 64-bit enabled.

Exposes:
  - USE_JAX: bool
  - xp: array module (numpy or jax.numpy)
  - fft: FFT submodule (xp.fft)
  - maybe_jit: decorator (jit when JAX, identity when NumPy)
  - vmap: vectorizer (jax.vmap when JAX, identity when NumPy)
  - asnumpy: convert array -> numpy.ndarray (no-op for NumPy)
"""
from __future__ import annotations
import os

USE_JAX = os.getenv("USE_JAX", "0") == "1"

if USE_JAX:
    try:
        import jax
        import jax.numpy as jnp
        from jax.config import config as _cfg
        _cfg.update("jax_enable_x64", True)  # complex128 / float64 for numerical finance

        xp = jnp
        fft = jnp.fft
        maybe_jit = jax.jit
        vmap = jax.vmap

        # Safe converter to numpy for returns in public API (tests expect numpy arrays)
        import numpy as _np

        def asnumpy(x):
            return _np.asarray(x)

    except Exception:
        # Fallback to NumPy if JAX import fails
        USE_JAX = False

if not USE_JAX:
    import numpy as xp  # type: ignore
    fft = xp.fft

    def maybe_jit(f):
        return f

    def vmap(f, *a, **k):
        return f

    def asnumpy(x):
        # Already numpy
        return x
