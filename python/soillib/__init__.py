"""
soillib -- numerical geomorphology.
"""

import silt  # noqa: F401  -- load-bearing, see above

from .soillib import *  # noqa: F401,F403
from .util import *  # noqa: F401,F403
from . import flow  # noqa: F401
from . import erosion  # noqa: F401
from . import spl  # noqa: F401

from . import soillib as _ext  # noqa: F401


# Out-of-place counterparts of the in-place `_`-suffixed kernels the
# extension binds (torch convention: spl_rate_ mutates, spl_rate returns a
# new tensor) -- see silt.add/add_ for the same pattern.

def spl_rate(discharge, slope, K, m, n):
    """K * discharge^m * |slope|^n, as a new tensor (see spl_rate_)."""
    rate = silt.like(discharge)
    spl_rate_(rate, discharge, slope, K=K, m=m, n=n)
    return rate


def spl_apply(height, rate, dt):
    """height - dt * rate, as a new tensor, leaving height untouched (see spl_apply_)."""
    result = height.copy_to()
    spl_apply_(result, rate, dt)
    return result


def spl_diffusivity(discharge, slope, D0, k_D, m, n):
    """D0 + k_D * discharge^m * |slope|^n, as a new tensor (see spl_diffusivity_)."""
    diffusivity = silt.like(discharge)
    spl_diffusivity_(diffusivity, discharge, slope, D0=D0, k_D=k_D, m=m, n=n)
    return diffusivity


def spl_deposition(Qs, Qw, k_S):
    """k_S * Qs / Qw (concentration-based), as a new tensor (see spl_deposition_)."""
    deposition = silt.like(Qs)
    spl_deposition_(deposition, Qs, Qw, k_S=k_S)
    return deposition


def _public_names() -> list:
    return sorted(n for n in dir(_ext) if not n.startswith("_"))

__all__ = _public_names() + ["spl_rate", "spl_apply", "spl_diffusivity", "spl_deposition"]

def _resolve_version() -> str:
    """Version of the installed distribution.

    Read from installed package metadata rather than a shipped file: the
    root VERSION file is not part of the wheel, so it is unavailable at
    runtime.
    """
    from importlib.metadata import PackageNotFoundError, version

    try:
        return version("soillib")
    except PackageNotFoundError:
        # Imported from an uninstalled build tree.
        return "0+unknown"


__version__ = _resolve_version()