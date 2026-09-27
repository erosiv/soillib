# Sphinx configuration.
#
# Build with:  pip install -e .[docs]  &&  sphinx-build doc build/html
# Doxygen must also be installed and on PATH -- it is not a pip package.

project = 'soillib'
copyright = '2026, Nicholas McDonald, erosiv Studio'
author = 'Nicholas McDonald, erosiv Studio'

# Read the version from the same root VERSION file CMake and pyproject.toml
# read, so the docs cannot drift from the build.
import pathlib as _pathlib

release = (_pathlib.Path(__file__).resolve().parent.parent
           / 'VERSION').read_text(encoding='utf-8').strip()
version = '.'.join(release.split('.')[:2])

extensions = [
    'breathe',             # renders Doxygen XML as Sphinx pages -- api_cpp.rst
    'myst_parser',         # lets .rst pull fragments out of README.md -- index.rst
    'sphinx.ext.autodoc',  # generates the Python reference from the built module
]

templates_path = ['_templates']
exclude_patterns = ['_build']

# -- C++ API reference (Doxygen + Breathe) -----------------------------------
#
# Breathe only *reads* Doxygen's XML; it does not run Doxygen. Running it here
# keeps `sphinx-build doc build/html` a single command instead of a documented
# two-step process -- and makes a missing Doxygen fail loudly rather than
# silently producing an empty API page.

import shutil
import subprocess

_doc_dir = _pathlib.Path(__file__).resolve().parent
_doxygen = shutil.which('doxygen')
if _doxygen is None:
    raise RuntimeError(
        "doxygen not found on PATH -- required to build the C++ API reference "
        "(see doc/Doxyfile, and the `docs` extra in pyproject.toml for the rest "
        "of the toolchain). If you just installed it, reopen your terminal: "
        "PATH changes do not reach a shell that was already running."
    )

# Doxygen creates at most one missing OUTPUT_DIRECTORY level, not nested ones,
# and _build/doxygen is two levels deep on a clean checkout. Pre-create it.
(_doc_dir / '_build' / 'doxygen').mkdir(parents=True, exist_ok=True)

_doxygen_result = subprocess.run(
    [_doxygen, 'Doxyfile'], cwd=_doc_dir,
    capture_output=True, text=True,
)
if _doxygen_result.returncode != 0:
    raise RuntimeError(
        f"doxygen exited with status {_doxygen_result.returncode}:\n"
        f"{_doxygen_result.stdout}{_doxygen_result.stderr}"
    )

breathe_projects = {'soillib': str(_doc_dir / '_build' / 'doxygen' / 'xml')}
breathe_default_project = 'soillib'

# -- HTML output -------------------------------------------------------------

html_theme = 'alabaster'
html_static_path = ['_static']
