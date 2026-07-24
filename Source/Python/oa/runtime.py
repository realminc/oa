"""OA Vulkan runtime, engine lifecycle, and execution contexts.
"""

from ._module import export_native
from ._native import native

__all__ = export_native(globals(), native.runtime)


class Context:
	"""Execute and synchronize the default OA context on scope exit.
	"""

	def __enter__(self) -> "Context":
		self._ctx = OaContextGetDefault()
		return self

	def __exit__(self, exc_type, exc_value, traceback) -> None:
		self._ctx.Execute()
		self._ctx.Sync()


__all__.append("Context")
del export_native, native
