"""OA's public Python package.

Convention (v2):
  ``import oa``                    — full surface at root
  ``import oa.core as oac``        — submodule import, same objects
  ``import oa.ml as oaml``
  ``import oa.vision as oav``

  oa.Matrix([12, 23])              # canonical PascalCase type
  oac.Matrix([12, 23])             # identical — same object
  oa.FnMatrix.ones([2, 3])         # canonical stateless operation namespace
"""

from importlib.metadata import PackageNotFoundError, version
from pathlib import Path as _Path

from ._native import native as _native
from ._surface import installSurface


def _readVersion() -> str:
	try:
		return version("oapython")
	except PackageNotFoundError:
		pass

	try:
		return (_Path(__file__).resolve().parents[3] / "VERSION").read_text(
			encoding="utf-8"
		).strip()
	except OSError:
		return "0+unknown"


__version__ = _readVersion()
_sources = {
	"audio":   _native.audio,
	"core":    _native.core,
	"crypto":  _native.crypto,
	"ml":      _native.ml,
	"plot":    _native.plot,
	"runtime": _native.runtime,
	"ui":      _native.ui,
	"vision":  _native.vision,
}
for _owner in (
	"FnAudio",
	"FnAdvantage",
	"FnAutograd",
	"FnDetection",
	"FnEnvironment",
	"FnImage",
	"FnLoss",
	"FnMatrix",
	"FnMetric",
	"FnPolicy",
):
	_sources[_owner] = getattr(_native, _owner)
if hasattr(_native, "FnHash"):
	_sources["FnHash"] = _native.FnHash
__all__ = list(installSurface(globals(), _sources))

del _owner, _sources, installSurface, _native
