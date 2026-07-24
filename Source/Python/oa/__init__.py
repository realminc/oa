"""OA's public C++-parity Python package.

The private :mod:`oa._oa` extension owns native registration. Public values
live at this package root, while C++ ``OaFn*`` and nested namespaces are exposed
as real Python modules. Legacy domain modules remain explicit compatibility
attributes and are not wildcard exports.
"""

from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

from . import audio, core, crypto, ml, plot, runtime, vision
from ._native import native as _native
from ._surface import install_surface
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
__all__ = install_surface(
	globals(),
	{
		"audio": audio,
		"core": core,
		"crypto": crypto,
		"ml": ml,
		"plot": plot,
		"runtime": runtime,
		"ui": _native.ui,
		"vision": vision,
	},
)

del install_surface, _native
