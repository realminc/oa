"""Ensure the custom Rust layout cannot silently leave test files unregistered."""

import re
import tomllib
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


class InventoryTests(unittest.TestCase):
	def test_every_rust_test_file_is_reachable_from_a_cargo_suite(self):
		manifest = tomllib.loads((ROOT / "Cargo.toml").read_text())
		self.assertFalse(manifest["package"]["autotests"])
		self.assertFalse(list((ROOT / "tests").glob("*.rs")), "legacy test files remain")
		pending = [ROOT / suite["path"] for suite in manifest["test"]]
		seen = set()
		while pending:
			path = pending.pop().resolve()
			self.assertNotIn(path, seen, f"test source registered twice: {path}")
			self.assertTrue(path.is_file(), f"registered source is missing: {path}")
			seen.add(path)
			for child in re.findall(r'#\[path\s*=\s*"([^"]+)"\]\s*mod\s+\w+\s*;', path.read_text()):
				pending.append(path.parent / child)
		self.assertEqual(seen, {path.resolve() for path in (ROOT / "test/rs").rglob("*.rs")})
