#!/usr/bin/env python3
"""Process-isolated checks for maintained public Python examples."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

from oa_python_test import REPO_ROOT


def _run(relativePath: str, workingDirectory: Path) -> str:
	environment = os.environ.copy()
	buildDirectory = environment.get("OA_PYTHON_BUILD_DIR")
	if buildDirectory:
		buildPath = Path(buildDirectory)
		if not buildPath.is_absolute():
			buildPath = REPO_ROOT / buildPath
		environment["OA_PYTHON_BUILD_DIR"] = str(buildPath)
		pythonPaths = [str(buildPath), str(REPO_ROOT / "source" / "py")]
		if environment.get("PYTHONPATH"):
			pythonPaths.append(environment["PYTHONPATH"])
		environment["PYTHONPATH"] = os.pathsep.join(pythonPaths)

	result = subprocess.run(
		[sys.executable, str(REPO_ROOT / relativePath)],
		cwd=workingDirectory,
		env=environment,
		check=True,
		capture_output=True,
		text=True,
	)
	return result.stdout


@pytest.mark.oa_gpu
def test_core_matrix_add_example(tmp_path: Path):
	output = _run("sdk/py/examples/core/matrixAdd.py", tmp_path)
	assert "[3.0, 3.0, 3.0, 3.0, 3.0, 3.0]" in output


@pytest.mark.oa_gpu
def test_audio_clip_example(tmp_path: Path):
	output = _run("sdk/py/examples/audio/clip.py", tmp_path)
	assert "8 mono samples clipped to 0.5" in output


@pytest.mark.oa_gpu
def test_vision_resize_example(tmp_path: Path):
	output = _run("sdk/py/examples/vision/resize.py", tmp_path)
	assert "RGB NCHW image resized from 2x2 to 4x3" in output


@pytest.mark.oa_crypto
@pytest.mark.oa_gpu
def test_crypto_shake256_example(tmp_path: Path):
	output = _run("sdk/py/examples/crypto/shake256.py", tmp_path)
	assert "3 GPU SHAKE-256 digests match the CPU oracle" in output


@pytest.mark.oa_gpu
def test_plot_line_example(tmp_path: Path):
	output = _run("sdk/py/examples/plot/line.py", tmp_path)
	assert "rendered a 360x240 training-loss figure" in output
