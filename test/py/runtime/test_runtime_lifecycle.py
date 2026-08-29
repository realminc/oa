#!/usr/bin/env python3
"""Process-isolated tests for OA's lazy Python engine lifecycle."""

from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys

import oa_python_test  # noqa: F401 - bootstraps source builds


def test_explicit_development_build_wins_over_installed_extension() -> None:
	buildDir = os.getenv("OA_PYTHON_BUILD_DIR")
	if buildDir is None:
		return
	from oa._native import native

	assert Path(native.__file__).resolve().parent == Path(buildDir).resolve()


def test_import_does_not_initialize_compute_engine() -> None:
	probe = """
import oa
from oa._native import native

assert not native.runtime._pythonEngineInitialized()
assert not native.runtime._pythonEnginePresentationCapable()
assert oa.MatrixShape([2, 3]).numElements() == 6
assert not native.runtime._pythonEngineInitialized()
assert not native.runtime._pythonEnginePresentationCapable()
"""
	subprocess.run([sys.executable, "-c", probe], check=True)


def test_first_device_factory_initializes_compute_engine() -> None:
	environment = os.environ.copy()
	# Deterministically exercise the presentation-warning / compute-only fallback
	# even when the test host has a working desktop session.
	environment["OA_UI_BACKEND"] = "oa-invalid-test-backend"
	probe = """
import oa
from oa._native import native

assert not native.runtime._pythonEngineInitialized()
one = oa.FnMatrix.ones([2, 3])
two = oa.FnMatrix.full([2, 3], 2.0)
values = oa.FnMatrix.copyToHost(oa.FnMatrix.add(one, two))
assert values == [3.0] * 6
assert native.runtime._pythonEngineInitialized()
"""
	subprocess.run([sys.executable, "-c", probe], check=True, env=environment)
