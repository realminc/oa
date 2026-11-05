"""The oa dispatcher routes every subcommand to an existing tool."""

import importlib.util
import subprocess
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
OA = ROOT / "tool/oa"
LOADER = importlib.machinery.SourceFileLoader("oa_dispatch", str(OA))
SPEC = importlib.util.spec_from_loader("oa_dispatch", LOADER)
DISPATCHER = importlib.util.module_from_spec(SPEC)
LOADER.exec_module(DISPATCHER)


class DispatcherTests(unittest.TestCase):
	def test_every_route_targets_an_existing_script(self):
		for routes in (DISPATCHER.PYTHON_ROUTES, DISPATCHER.SHELL_ROUTES):
			for command, relative in routes.items():
				self.assertTrue((ROOT / "tool" / relative).is_file(), command)

	def test_every_test_scope_maps_to_a_directory(self):
		for scope, relative in DISPATCHER.TEST_SCOPES.items():
			self.assertTrue((ROOT / relative).is_dir(), scope)

	def test_routes_parse_longest_prefix(self):
		self.assertEqual(
			DISPATCHER.route(["audit", "crate", "x.crate", "1.0"]),
			(("audit", "crate"), ["x.crate", "1.0"]),
		)
		self.assertEqual(
			DISPATCHER.route(["build", "--profile", "release"]),
			(("build",), ["--profile", "release"]),
		)
		self.assertEqual(DISPATCHER.route(["bogus"]), ((), ["bogus"]))

	def test_help_lists_all_categories(self):
		result = subprocess.run(
			[sys.executable, str(OA), "--help"], capture_output=True, text=True
		)
		self.assertEqual(result.returncode, 0)
		for category in ("gen fn", "gen nn", "gen drift", "build", "stage", "audit", "profile", "release", "test"):
			self.assertIn(category, result.stdout)

	def test_unknown_command_is_rejected(self):
		result = subprocess.run(
			[sys.executable, str(OA), "bogus"], capture_output=True, text=True
		)
		self.assertEqual(result.returncode, 2)
		self.assertIn("usage", result.stderr)
