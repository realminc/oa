#!/usr/bin/env python3
"""Process-isolated tests for the public Python tutorials."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

from oa_python_test import REPO_ROOT

TUTORIAL_ROOT = REPO_ROOT / "sdk" / "py" / "tutorials"


def _run(relativePath: str, workingDirectory: Path, *arguments: str) -> str:
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
		[sys.executable, str(TUTORIAL_ROOT / relativePath), *arguments],
		cwd=workingDirectory,
		env=environment,
		check=True,
		capture_output=True,
		text=True,
	)
	return result.stdout


def test_tutorial_layout_uses_domain_directories():
	domains = {
		entry.name
		for entry in TUTORIAL_ROOT.iterdir()
		if entry.is_dir() and not entry.name.startswith("__")
	}
	assert {"core", "audio", "vision", "ml", "plot"} <= domains
	assert (TUTORIAL_ROOT / "ml" / "nlp").is_dir()
	assert (TUTORIAL_ROOT / "vision" / "tutorialVisionViewer.py").is_file()
	assert not list(TUTORIAL_ROOT.glob("*.py"))


def test_python_plot_gallery_mirrors_cpp_tutorial():
	assert (
		REPO_ROOT / "sdk" / "cpp" / "tutorials" / "plot" / "tutorialPlotGallery.cpp"
	).is_file()
	assert (TUTORIAL_ROOT / "plot" / "tutorialPlotGallery.py").is_file()


def test_tutorial_sources_use_only_the_public_root():
	for tutorial in TUTORIAL_ROOT.rglob("*.py"):
		source = tutorial.read_text(encoding="utf-8")
		assert "from oa import *" in source, tutorial
		assert "_oa_import" not in source, tutorial
		for lowercaseModule in (
			"oa.audio",
			"oa.core",
			"oa.crypto",
			"oa.ml",
			"oa.plot",
			"oa.runtime",
			"oa.vision",
		):
			assert lowercaseModule not in source, (
				tutorial,
				lowercaseModule,
			)


def test_python_nlp_suite_mirrors_cpp_executables_one_to_one():
	cppRoot = REPO_ROOT / "sdk" / "cpp" / "tutorials" / "ml" / "nlp"
	pythonRoot = TUTORIAL_ROOT / "ml" / "nlp"
	cppMembers = {
		path.stem for path in cppRoot.glob("tutorialNlp*Ag.cpp")
	}
	pythonMembers = {
		path.stem for path in pythonRoot.glob("tutorialNlp*Ag.py")
	}

	assert len(cppMembers) == 16
	assert pythonMembers == cppMembers
	assert not (pythonRoot / "tutorialNlpRnn.py").exists()
	assert not (pythonRoot / "TutorialNlpRnnAutograd.py").exists()


@pytest.mark.oa_gpu
def test_core_basics_tutorial(tmp_path):
	output = _run("core/tutorialCoreBasics.py", tmp_path)
	assert "[2, 3]" in output
	assert "[0.0, 0.5, 1.0, 1.5, 2.0, 2.5]" in output


@pytest.mark.oa_gpu
def test_audio_basics_tutorial(tmp_path):
	output = _run("audio/tutorialAudioBasics.py", tmp_path)
	wav = Path(output.strip().splitlines()[-1])
	assert wav.read_bytes()[:4] == b"RIFF"
	assert str(wav) in output


@pytest.mark.oa_gpu
def test_vision_basics_tutorial(tmp_path):
	output = _run("vision/tutorialVisionBasics.py", tmp_path)
	assert "[1, 3, 90, 160]" in output


@pytest.mark.oa_gpu
def test_vision_data_augmentation_tutorial(tmp_path):
	outputPath = tmp_path / "oa-vision-data-augmentation.png"
	output = _run(
		"vision/tutorialVisionDataAugmentation.py",
		tmp_path,
		"--output",
		str(outputPath),
	)
	image = outputPath.read_bytes()
	assert image[:8] == b"\x89PNG\r\n\x1a\n"
	assert int.from_bytes(image[16:20], "big") == 1200
	assert int.from_bytes(image[20:24], "big") == 760
	assert str(outputPath) in output


@pytest.mark.oa_gpu
def test_ml_basics_tutorial(tmp_path):
	output = _run("ml/tutorialMlBasics.py", tmp_path)
	assert "loss:" in output
	assert "[-3." in output


@pytest.mark.oa_gpu
def test_plot_gallery_tutorial(tmp_path):
	outputDirectory = tmp_path / "plot-gallery"
	output = _run(
		"plot/tutorialPlotGallery.py",
		tmp_path,
		"--headless",
		"--output",
		str(outputDirectory),
	)
	expected = {
		"oa-plot-intro-dark.png": (1280, 560),
		"oa-plot-gallery-dark.png": (1600, 1050),
		"oa-plot-gallery-light.png": (1600, 1050),
		"oa-plot-evaluation-dark.png": (1280, 560),
		"oa-plot-diagnostics-dark.png": (1200, 900),
		"oa-plot-landscapes-dark.png": (1280, 560),
	}
	assert {path.name for path in outputDirectory.iterdir()} == set(expected)
	for name, (width, height) in expected.items():
		image = (outputDirectory / name).read_bytes()
		assert image[:8] == b"\x89PNG\r\n\x1a\n"
		assert int.from_bytes(image[16:20], "big") == width
		assert int.from_bytes(image[20:24], "big") == height
		assert str(outputDirectory / name) in output
