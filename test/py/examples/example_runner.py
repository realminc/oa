"""Process-isolated runner shared by generated and handwritten examples."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

from oa_python_test import REPO_ROOT


def runExample(relativePath: str, workingDirectory: Path) -> str:
	environment = os.environ.copy()
	environment.setdefault("OA_ASSET_DIR", str(REPO_ROOT / "sdk" / "asset"))
	environment["OA_VAR_DIR"] = str(workingDirectory / "var")
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
