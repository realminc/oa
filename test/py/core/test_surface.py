#!/usr/bin/env python3
"""Host-only tests for OA's canonical Python surface publisher."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType
import sys
import unittest

from oa_python_test import REPO_ROOT


SURFACE_PATH = REPO_ROOT / "source" / "py" / "oa" / "_surface.py"
TEST_PACKAGE = "_oa_test_package"
packageModule = ModuleType(TEST_PACKAGE)
packageModule.__path__ = [str(SURFACE_PATH.parent)]
sys.modules[TEST_PACKAGE] = packageModule
SPEC = importlib.util.spec_from_file_location(
	f"{TEST_PACKAGE}._surface", SURFACE_PATH
)
assert SPEC is not None and SPEC.loader is not None
surface = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(surface)


def _module(name: str, **attributes: object) -> ModuleType:
	module = ModuleType(f"_oa.{name}")
	for key, value in attributes.items():
		setattr(module, key, value)
	return module


def _sources(*, crypto: bool = True) -> dict[str, ModuleType]:
	result = {
		"audio": _module("audio", Audio=type("Audio", (), {})),
		"core": _module(
			"core",
			Matrix=type("Matrix", (), {}),
			MatrixShape=type("MatrixShape", (), {}),
		),
		"crypto": _module("crypto", available=crypto),
		"ml": _module("ml", Module=type("Module", (), {})),
		"plot": _module("plot", Figure=type("Figure", (), {})),
		"runtime": _module("runtime", initComputeEngine=lambda: True),
		"ui": _module("ui", Viewer=type("Viewer", (), {})),
		"vision": _module("vision", Image=type("Image", (), {})),
	}
	for key in surface._FUNCTION_KEYS:
		if key == "FnHash" and not crypto:
			continue
		result[key] = _module(key)

	fnMatrix = result["FnMatrix"]
	resultNames = {
		name
		for name, _ in surface.SCHEMA_ROOT_EXPORTS.get("FnMatrix", ())
	}
	resultNames.update({
		"LinearWeightBiasBwdResult",
		"Mamba3PreprocessConfig",
		"Quantization",
		"QuantMatrix",
		"SsmConfig",
	})
	for name in resultNames:
		setattr(fnMatrix, name, type(name, (), {}))
	return result


class TestSurface(unittest.TestCase):
	def setUp(self) -> None:
		self.previousModules = {
			name: module
			for name, module in sys.modules.items()
			if name.startswith("oa.")
		}

	def tearDown(self) -> None:
		for name in list(sys.modules):
			if name.startswith("oa."):
				del sys.modules[name]
		sys.modules.update(self.previousModules)

	def test_domains_namespaces_and_root_values_are_identical(self) -> None:
		package: dict[str, object] = {}
		sources = _sources()
		inventory = surface.installSurface(package, sources)

		self.assertIs(package["Matrix"], sources["core"].Matrix)
		self.assertIs(package["core"], sources["core"])
		self.assertIs(package["FnMatrix"], sources["FnMatrix"])
		self.assertIs(sys.modules["oa.core"], sources["core"])
		self.assertIs(sys.modules["oa.FnMatrix"], sources["FnMatrix"])
		self.assertIn("Figure", inventory)
		self.assertFalse(any(name.startswith("Oa") for name in inventory))

	def test_unavailable_optional_crypto_is_not_published(self) -> None:
		package: dict[str, object] = {}
		inventory = surface.installSurface(package, _sources(crypto=False))
		self.assertNotIn("crypto", inventory)
		self.assertNotIn("FnHash", inventory)

	def test_preexisting_collision_fails_import(self) -> None:
		with self.assertRaisesRegex(ImportError, "collision"):
			surface.installSurface({"Matrix": object()}, _sources())

	def test_missing_schema_owned_result_fails_import(self) -> None:
		sources = _sources()
		delattr(sources["FnMatrix"], "TopKResult")
		with self.assertRaisesRegex(ImportError, "TopKResult"):
			surface.installSurface({}, sources)


if __name__ == "__main__":
	unittest.main()
