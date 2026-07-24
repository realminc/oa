#!/usr/bin/env python3
"""Process-isolated checks for the minimal public Python tutorials."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
TUTORIAL_ROOT = REPO_ROOT / "Tutorial" / "Py"


def _run(relative_path: str, working_directory: Path) -> str:
	result = subprocess.run(
		[sys.executable, str(TUTORIAL_ROOT / relative_path)],
		cwd=working_directory,
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
	assert {"Core", "Audio", "Vision", "Ml"} <= domains
	assert (TUTORIAL_ROOT / "Ml" / "Nlp").is_dir()
	assert (TUTORIAL_ROOT / "Vision" / "TutorialVisionViewer.py").is_file()
	assert not list(TUTORIAL_ROOT.glob("*.py"))


def test_tutorial_sources_use_only_the_public_root():
	for tutorial in TUTORIAL_ROOT.rglob("*.py"):
		source = tutorial.read_text(encoding="utf-8")
		assert "from oa import *" in source, tutorial
		assert "_oa_import" not in source, tutorial
		for compatibility_name in (
			"oa.core",
			"oa.ml",
			"oa.audio",
			"oa.vision",
			"oa.runtime",
		):
			assert compatibility_name not in source, (
				tutorial,
				compatibility_name,
			)


def test_python_nlp_suite_mirrors_cpp_executables_one_to_one():
	cpp_root = REPO_ROOT / "Tutorial" / "Ml" / "Nlp"
	python_root = TUTORIAL_ROOT / "Ml" / "Nlp"
	cpp_members = {
		path.stem for path in cpp_root.glob("TutorialNlp*Ag.cpp")
	}
	python_members = {
		path.stem for path in python_root.glob("TutorialNlp*Ag.py")
	}

	assert len(cpp_members) == 16
	assert python_members == cpp_members
	assert not (python_root / "TutorialNlpRnn.py").exists()
	assert not (python_root / "TutorialNlpRnnAutograd.py").exists()


def test_core_basics_tutorial(tmp_path):
	output = _run("Core/TutorialCoreBasics.py", tmp_path)
	assert "[2, 3]" in output
	assert "[0.0, 0.5, 1.0, 1.5, 2.0, 2.5]" in output


def test_audio_basics_tutorial(tmp_path):
	output = _run("Audio/TutorialAudioBasics.py", tmp_path)
	wav = Path(output.strip().splitlines()[-1])
	assert wav.read_bytes()[:4] == b"RIFF"
	assert str(wav) in output


def test_vision_basics_tutorial(tmp_path):
	output = _run("Vision/TutorialVisionBasics.py", tmp_path)
	assert "[1, 3, 90, 160]" in output


def test_ml_basics_tutorial(tmp_path):
	output = _run("Ml/TutorialMlBasics.py", tmp_path)
	assert "loss:" in output
	assert "[-3." in output
