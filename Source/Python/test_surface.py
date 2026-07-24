#!/usr/bin/env python3
"""Host-only tests for OA's root exposure manifest."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import re
from types import ModuleType
import sys
import unittest


SURFACE_PATH = Path(__file__).resolve().parent / "oa" / "_surface.py"
REPO_ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location("_oa_test_surface", SURFACE_PATH)
assert SPEC is not None and SPEC.loader is not None
surface = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(surface)


def _function(name: str):
	def value():
		return name

	value.__name__ = name
	return value


def _sources(*, crypto: bool = False) -> dict[str, ModuleType]:
	sources: dict[str, ModuleType] = {}
	for source_key in surface._ROOT_EXPORTS:
		module = ModuleType(f"oa.{source_key}")
		sources[source_key] = module
		for _, source_name in surface._ROOT_EXPORTS[source_key]:
			if not hasattr(module, source_name):
				setattr(module, source_name, type(source_name, (), {}))

	sources["plot"] = ModuleType("oa.plot")

	for exports in surface._NAMESPACE_EXPORTS.values():
		for source_key, mappings in exports.items():
			module = sources[source_key]
			for _, source_name in mappings:
				if not hasattr(module, source_name):
					setattr(module, source_name, _function(source_name))

	for _, (source_key, names) in surface._NESTED_EXPORTS.items():
		module = sources[source_key]
		for name in names:
			setattr(module, name, type(name, (), {}))

	crypto_module = ModuleType("oa.crypto")
	crypto_module.available = crypto
	sources["crypto"] = crypto_module
	if crypto:
		for _, source_name in surface._OPTIONAL_ROOT_EXPORTS["crypto"]:
			if not hasattr(crypto_module, source_name):
				setattr(crypto_module, source_name, type(source_name, (), {}))
		for exports in surface._OPTIONAL_NAMESPACE_EXPORTS.values():
			for _, mappings in exports.items():
				for _, source_name in mappings:
					setattr(crypto_module, source_name, _function(source_name))
	return sources


class TestSurface(unittest.TestCase):
	def setUp(self) -> None:
		self._previous_modules = {
			name: module
			for name, module in sys.modules.items()
			if name.startswith("oa.Oa")
		}

	def tearDown(self) -> None:
		for name in list(sys.modules):
			if name.startswith("oa.Oa"):
				del sys.modules[name]
		sys.modules.update(self._previous_modules)

	def test_required_root_and_namespaces(self) -> None:
		package: dict[str, object] = {}
		sources = _sources()
		inventory = surface.install_surface(package, sources)

		self.assertTrue(inventory)
		self.assertTrue(all(name.startswith("Oa") for name in inventory))
		self.assertIs(package["OaFilesystem"], sources["core"].OaFilesystem)
		self.assertIs(package["OaMatrix"], sources["core"].OaMatrix)
		self.assertIs(package["OaPath"], sources["core"].OaPath)
		self.assertIs(package["OaPaths"], sources["core"].OaPaths)
		self.assertIs(package["OaGradientTape"], sources["ml"].GradientTape)
		self.assertIs(package["OaNlpSuiteModel"], sources["ml"].OaNlpSuiteModel)
		self.assertIs(package["OaViewer"], sources["ui"].OaViewer)

		fn_audio = package["OaFnAudio"]
		self.assertIsInstance(fn_audio, ModuleType)
		self.assertEqual(fn_audio.__name__, "oa.OaFnAudio")
		self.assertIs(fn_audio.Normalize, sources["audio"].Normalize)
		self.assertIs(sys.modules["oa.OaFnAudio"], fn_audio)

		fn_metric = package["OaFnMetric"]
		self.assertIsInstance(fn_metric, ModuleType)
		self.assertIs(fn_metric.Accuracy, sources["ml"].MetricAccuracy)

		plot = package["OaPlot"]
		self.assertIsInstance(plot, ModuleType)
		self.assertIs(plot.Figure, sources["plot"].Figure)
		self.assertEqual(plot.Figure.__module__, "oa.OaPlot")

	def test_optional_crypto_does_not_mutate_default_inventory(self) -> None:
		default_package: dict[str, object] = {}
		default_inventory = surface.install_surface(default_package, _sources())
		self.assertNotIn("OaFnHash", default_inventory)

		crypto_package: dict[str, object] = {}
		crypto_inventory = surface.install_surface(
			crypto_package, _sources(crypto=True)
		)
		self.assertIn("OaFnHash", crypto_inventory)
		self.assertIn("OaSign", crypto_inventory)

	def test_missing_required_symbol_fails_import(self) -> None:
		sources = _sources()
		del sources["audio"].OaAudioDecoder
		with self.assertRaisesRegex(ImportError, "OaAudioDecoder"):
			surface.install_surface({}, sources)

	def test_preexisting_collision_fails_import(self) -> None:
		with self.assertRaisesRegex(ImportError, "collision"):
			surface.install_surface({"OaMatrix": object()}, _sources())

	def test_native_public_names_and_keywords_are_pascal_case(self) -> None:
		binding_root = REPO_ROOT / "Source" / "Python"
		lowercase_names: list[str] = []
		lowercase_keywords: list[str] = []
		for path in sorted(binding_root.rglob("*.cpp")):
			source = path.read_text(encoding="utf-8")
			relative = path.relative_to(REPO_ROOT)
			for name in re.findall(
				r'\.(?:def|def_static|def_prop|def_prop_ro|def_rw|def_ro)'
				r'\("([^"]+)"',
				source,
			):
				if not name.startswith("_") and not re.fullmatch(
					r"[A-Z][A-Za-z0-9]*", name
				):
					lowercase_names.append(f"{relative}:{name}")
			for name in re.findall(r'nb::arg\("([^"]+)"\)', source):
				if not re.fullmatch(r"[A-Z][A-Za-z0-9]*", name):
					lowercase_keywords.append(f"{relative}:{name}")

		self.assertEqual(lowercase_names, [])
		self.assertEqual(lowercase_keywords, [])


if __name__ == "__main__":
	unittest.main()
