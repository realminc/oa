#!/usr/bin/env python3
"""Process-isolated checks for maintained public Python examples."""

from __future__ import annotations

from pathlib import Path

import pytest

from example_runner import runExample


@pytest.mark.oa_gpu
def test_vision_resize_example(tmp_path: Path):
	output = runExample("sdk/py/examples/vision/resize.py", tmp_path)
	assert "RGB NCHW image resized from 2x2 to 4x3" in output


@pytest.mark.oa_crypto
@pytest.mark.oa_gpu
def test_crypto_shake256_example(tmp_path: Path):
	output = runExample("sdk/py/examples/crypto/shake256.py", tmp_path)
	assert "3 GPU SHAKE-256 digests match the CPU oracle" in output


@pytest.mark.oa_gpu
def test_plot_line_example(tmp_path: Path):
	output = runExample("sdk/py/examples/plot/line.py", tmp_path)
	assert "rendered a 360x240 training-loss figure" in output
