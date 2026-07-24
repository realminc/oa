#!/usr/bin/env python3
"""Process-isolated checks for OA's lazy Python engine lifecycle."""

from __future__ import annotations

import subprocess
import sys


def test_import_does_not_initialize_compute_engine() -> None:
	probe = """
import oa
from oa._native import native

assert not native.runtime._OaPythonEngineInitialized()
assert not native.runtime._OaPythonEnginePresentationCapable()
assert oa.OaMatrixShape([2, 3]).NumElements() == 6
assert not native.runtime._OaPythonEngineInitialized()
assert not native.runtime._OaPythonEnginePresentationCapable()
"""
	subprocess.run([sys.executable, "-c", probe], check=True)
