"""Static typing contract for OA's generated Python surface."""

from __future__ import annotations

import ast
from pathlib import Path
import subprocess
import sys

import oa


REPO_ROOT = Path(__file__).resolve().parents[2]
PACKAGE_ROOT = REPO_ROOT / "Source" / "Python" / "oa"
STUB_GENERATOR = REPO_ROOT / "Tools" / "Python" / "oapystubgen.py"


def _declared_names(path: Path) -> set[str]:
	tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
	names: set[str] = set()
	for node in tree.body:
		if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
			names.add(node.name)
		elif isinstance(node, ast.ImportFrom):
			names.update(alias.asname or alias.name for alias in node.names)
		elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
			names.add(node.target.id)
		elif isinstance(node, ast.Assign):
			names.update(
				target.id for target in node.targets if isinstance(target, ast.Name)
			)
	return names


def test_generated_stubs_are_current() -> None:
	result = subprocess.run(
		[sys.executable, str(STUB_GENERATOR), "--check"],
		cwd=REPO_ROOT,
		check=False,
		capture_output=True,
		text=True,
	)
	assert result.returncode == 0, result.stdout + result.stderr


def test_root_stub_declares_the_runtime_wildcard_surface() -> None:
	declared = _declared_names(PACKAGE_ROOT / "__init__.pyi")
	assert set(oa.__all__) <= declared


def test_namespace_and_native_stubs_parse() -> None:
	assert (PACKAGE_ROOT / "py.typed").is_file()
	for namespace in (name for name in oa.__all__ if name.startswith("OaFn")):
		assert (PACKAGE_ROOT / f"{namespace}.pyi").is_file()
	assert (PACKAGE_ROOT / "_oa" / "vision.pyi").is_file()

	stubs = sorted(PACKAGE_ROOT.rglob("*.pyi"))
	assert stubs
	for stub in stubs:
		ast.parse(stub.read_text(encoding="utf-8"), filename=str(stub))
