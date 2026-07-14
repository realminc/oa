"""OA audio codecs and GPU signal-processing operations."""

from ._module import export_native
from ._native import native

__all__ = export_native(globals(), native.audio)
del export_native, native
