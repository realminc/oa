"""OA cryptography and GPU hashing.

The module imports on all builds. ``available`` is false when OA was compiled
without its optional cryptography backend.
"""

from ._module import export_native
from ._native import native

__all__ = export_native(globals(), native.crypto)
available = getattr(native.crypto, "available", True)
if "available" not in __all__:
    __all__.append("available")
del export_native, native
