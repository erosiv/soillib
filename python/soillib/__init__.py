"""
soillib -- numerical geomorphology.
"""

import silt  # noqa: F401  -- load-bearing, see above

from .soillib import *  # noqa: F401,F403
from .util import *  # noqa: F401,F403

from . import soillib as _ext  # noqa: F401

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