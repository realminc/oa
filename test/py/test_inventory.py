"""Ensure the custom Rust layout cannot silently leave test files unregistered."""

import re
import tomllib
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

_PATH_ATTR = re.compile(r'#\[path\s*=\s*"([^"]+)"\]\s*mod\s+\w+\s*;')
_UNIT_PATH = re.compile(r'#\[cfg\(test\)\]\s*#\[path\s*=\s*"([^"]+)"\]\s*mod\s+\w+\s*;')


class InventoryTests(unittest.TestCase):
	def test_every_rust_test_file_is_reachable_from_a_cargo_suite(self):
		manifest = tomllib.loads((ROOT / "Cargo.toml").read_text())
		self.assertFalse(manifest["package"]["autotests"])
		self.assertFalse(list((ROOT / "tests").glob("*.rs")), "legacy test files remain")

		# Walk the [[test]] harness entries and follow every #[path] chain.
		pending = [ROOT / suite["path"] for suite in manifest["test"]]
		seen = set()
		while pending:
			path = pending.pop().resolve()
			if path in seen:
				relative = path.relative_to(ROOT)
				self.assertEqual(
					relative.parts[:3],
					("test", "rs", "support"),
					f"test source registered twice: {path}",
				)
				continue
			self.assertTrue(path.is_file(), f"registered source is missing: {path}")
			seen.add(path)
			for child in _PATH_ATTR.findall(path.read_text()):
				pending.append(path.parent / child)

		for source_root in (ROOT / "src/rs", ROOT / "sdk/rs"):
			for src in source_root.rglob("*.rs"):
				text = src.read_text()
				for child in _UNIT_PATH.findall(text):
					unit = (src.parent / child).resolve()
					self.assertEqual(unit.parent.parent, ROOT / "test/rs", f"unit test is outside test/rs: {unit}")
					self.assertTrue(unit.is_file(), f"unit test source is missing: {unit}")
					self.assertNotIn(unit, seen, f"test source registered twice: {unit}")
					seen.add(unit)
				text = _UNIT_PATH.sub("", text)
				for marker in ("#[cfg(test)]", "#[test]"):
					self.assertNotIn(
						marker,
						text,
						f"tests and test-only helpers belong under test/rs: {src}",
					)

		self.assertEqual(seen, {path.resolve() for path in (ROOT / "test/rs").rglob("*.rs")})
