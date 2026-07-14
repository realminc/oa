"""OA's public Python package.

The API mirrors the C++ module boundaries. Native implementation details live in
the private :mod:`oa._oa` extension and are never wildcard-exported here.
"""

from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

from . import audio, core, crypto, ml, runtime, vision
from .runtime import Context


def _read_version() -> str:
    try:
        return version("oapython")
    except PackageNotFoundError:
        pass

    try:
        return (Path(__file__).resolve().parents[3] / "VERSION").read_text(
            encoding="utf-8"
        ).strip()
    except OSError:
        return "0+unknown"


__version__ = _read_version()
__all__ = [
    "Context", "audio", "core", "crypto", "ml", "runtime", "vision", "__version__"
]
