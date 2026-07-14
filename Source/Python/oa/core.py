"""OA tensors, dtypes, factories, and functional matrix operations."""

from ._module import export_native
from ._native import native

__all__ = export_native(globals(), native.core)
del export_native, native
