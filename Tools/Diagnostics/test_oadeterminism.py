#!/usr/bin/env python3

from __future__ import annotations

import json
import pathlib
import subprocess
import sys
import tempfile
import unittest

import oadeterminism


class OaDeterminismTest(unittest.TestCase):
	def setUp(self) -> None:
		self.temporary = tempfile.TemporaryDirectory(prefix="oa-determinism-test-")
		self.repo = pathlib.Path(self.temporary.name) / "repo"
		self.repo.mkdir()
		subprocess.run(("git", "init", "-q"), cwd=self.repo, check=True)
		subprocess.run(
			("git", "config", "user.email", "test@example.invalid"),
			cwd=self.repo,
			check=True,
		)
		subprocess.run(
			("git", "config", "user.name", "OA Test"), cwd=self.repo, check=True
		)
		(self.repo / "VERSION").write_text("0.7.5\n", encoding="utf-8")
		subprocess.run(("git", "add", "VERSION"), cwd=self.repo, check=True)
		subprocess.run(("git", "commit", "-qm", "baseline"), cwd=self.repo, check=True)

	def tearDown(self) -> None:
		self.temporary.cleanup()

	def test_compares_selected_trace_across_fresh_processes(self) -> None:
		output = self.repo / "determinism.json"
		args = oadeterminism.parse_args(
			(
				"--repo",
				str(self.repo),
				"--output",
				str(output),
				"--name",
				"unit.loss_curve",
				"--runs",
				"3",
				"--trace-regex",
				r"loss=(?P<value>[0-9.]+)",
				"--require-regex",
				"complete=yes",
				"--",
				sys.executable,
				"-c",
				"print('loss=1.0 loss=0.5 complete=yes')",
			)
		)
		actual, status = oadeterminism.run(args)
		self.assertEqual(actual, output)
		self.assertEqual(status, 0)
		document = json.loads(output.read_text(encoding="utf-8"))
		self.assertEqual(document["schema"], oadeterminism.SCHEMA)
		self.assertEqual(document["result"], "PASS")
		self.assertEqual(len(document["samples"]), 3)
		self.assertEqual(document["samples"][0]["trace_groups"], [["1.0", "0.5"]])
		self.assertTrue(all(run["matches_reference"] for run in document["samples"]))

	def test_fails_when_trace_changes(self) -> None:
		output = self.repo / "changing.json"
		counter = self.repo / "counter"
		program = (
			"from pathlib import Path; "
			f"p=Path({str(counter)!r}); "
			"n=int(p.read_text())+1 if p.exists() else 1; "
			"p.write_text(str(n)); print(f'value={n}')"
		)
		args = oadeterminism.parse_args(
			(
				"--repo",
				str(self.repo),
				"--output",
				str(output),
				"--name",
				"unit.changing",
				"--trace-regex",
				r"value=(?P<value>\d+)",
				"--",
				sys.executable,
				"-c",
				program,
			)
		)
		_, status = oadeterminism.run(args)
		self.assertEqual(status, 1)
		document = json.loads(output.read_text(encoding="utf-8"))
		self.assertEqual(document["result"], "FAIL")
		self.assertFalse(document["samples"][1]["matches_reference"])


if __name__ == "__main__":
	unittest.main()
