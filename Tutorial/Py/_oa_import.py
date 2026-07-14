"""Local import helper for OA Python tutorials.

The tutorials are normally run from a source checkout before OA has a packaged
wheel. This helper finds the repo root, adds the Python source tree and common
CMake build output directories, then imports the public OA package.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]


def _candidate_paths() -> list[Path]:
    paths: list[Path] = []

    build_dir = os.getenv("OA_PYTHON_BUILD_DIR")
    if build_dir:
        paths.append(Path(build_dir).expanduser())

    paths.extend(
        [
            REPO_ROOT / "Build" / "Release",
            REPO_ROOT / "Build" / "Debug",
            REPO_ROOT / "build",
            REPO_ROOT / "Source" / "Python",
        ]
    )
    return paths


for path in _candidate_paths():
    if path.exists():
        path_str = str(path)
        if path_str not in sys.path:
            sys.path.insert(0, path_str)


try:
    import oa
except ImportError as exc:
    searched = "\n  ".join(str(path) for path in _candidate_paths())
    raise ImportError(
        "Could not import OA Python bindings. Build with "
        "`cmake --preset release -DOA_BUILD_PYTHON=ON` and "
        "`cmake --build Build/Release --target _oa`, or set "
        "OA_PYTHON_BUILD_DIR to the directory containing `_oa`.\n"
        f"Searched:\n  {searched}"
    ) from exc


core = oa.core
ml = oa.ml
runtime = oa.runtime


def set_grad_enabled(enabled: bool) -> None:
    """Set autograd tracking."""

    setter = getattr(ml, "SetEnabled", None) or getattr(ml, "SetGradEnabled", None)
    if setter is None:
        raise AttributeError("oa.ml exposes neither SetEnabled nor SetGradEnabled")
    setter(enabled)
