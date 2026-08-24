"""Shared filesystem contract for OA code generators."""

from __future__ import annotations

from pathlib import Path


def write_generated_text(path: Path, content: str) -> bool:
	"""Write UTF-8 output only when its bytes changed.

	Returning ``False`` preserves the file timestamp, so an idempotent generator
	run cannot invalidate CMake/Ninja dependencies or trigger shader embedding.
	"""
	data = content.encode("utf-8")
	if path.exists() and path.read_bytes() == data:
		return False
	path.parent.mkdir(parents=True, exist_ok=True)
	path.write_bytes(data)
	return True
