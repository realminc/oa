"""Shared source-tree bootstrap for OA's Python test profiles."""

from __future__ import annotations

from importlib.util import find_spec
import os
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[2]


def _addDevPaths() -> None:
	buildDir = os.getenv("OA_PYTHON_BUILD_DIR")
	if buildDir:
		candidates = [Path(buildDir).expanduser(), REPO_ROOT / "source" / "py"]
	elif find_spec("oa") is None:
		candidates = [
			REPO_ROOT / "build" / "Release",
			REPO_ROOT / "build" / "Debug",
			REPO_ROOT / "build",
			REPO_ROOT / "source" / "py",
		]
	else:
		candidates = []

	for path in reversed(candidates):
		if path.exists() and str(path) not in sys.path:
			sys.path.insert(0, str(path))


_addDevPaths()
