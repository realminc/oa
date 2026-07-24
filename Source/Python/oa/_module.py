"""Helpers for the thin public wrappers around native OA submodules.
"""

from types import ModuleType
from typing import Any


def export_native(namespace: dict[str, Any], module: ModuleType) -> list[str]:
	"""Expose a native submodule without leaking its private module object.
		"""

	names = [name for name in dir(module) if not name.startswith("_")]
	for name in names:
		value = getattr(module, name)
		namespace[name] = value
		if isinstance(value, type):
			try:
				value.__module__ = namespace["__name__"]
			except (AttributeError, TypeError):
				# Some extension types deliberately expose a read-only module.
				pass
	return names
