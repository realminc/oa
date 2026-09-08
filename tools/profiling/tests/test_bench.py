from __future__ import annotations

import re
import sys
import unittest
from pathlib import Path


PROFILING = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROFILING))

import bench  # noqa: E402
import suite  # noqa: E402


class BenchTests(unittest.TestCase):
	def test_statistics_preserve_raw_distribution(self) -> None:
		stats = bench.sample_statistics([1.0, 2.0, 3.0, 4.0, 10.0], "ms")
		self.assertEqual(stats["count"], 5)
		self.assertEqual(stats["median"], 3.0)
		self.assertEqual(stats["mad"], 1.0)
		self.assertEqual(stats["minimum"], 1.0)
		self.assertEqual(stats["maximum"], 10.0)

	def test_metric_extraction_uses_last_named_capture_and_strips_ansi(self) -> None:
		pattern = re.compile(r"metric=(?P<value>[0-9.]+)")
		value, matches = bench.extract_metric(
			pattern, "\x1b[32mmetric=1.25\x1b[0m\nmetric=2.5"
		)
		self.assertEqual(value, 2.5)
		self.assertEqual(matches, 2)

	def test_contract_rejects_duplicates(self) -> None:
		with self.assertRaisesRegex(ValueError, "duplicate"):
			bench.parse_contract(["dtype=f32", "dtype=i32"])

	def test_checked_in_suite_is_valid_and_complete(self) -> None:
		workloads = suite.load_suite(PROFILING / "suite.json")
		self.assertEqual(len(workloads), 6)
		self.assertEqual(
			{workload["contract"]["boundary"] for workload in workloads},
			{"captured_plan_gpu_timestamp"},
		)
		self.assertTrue(
			all(workload["contract"]["fallback"] == "none" for workload in workloads)
		)

	def test_command_expansion_is_explicit(self) -> None:
		command = suite.expand_command(
			["{binary_root}/bench", "--device", "{device_index}"],
			Path("/tmp/oars-bin"),
			3,
		)
		self.assertEqual(command, ["/tmp/oars-bin/bench", "--device", "3"])

	def test_command_redaction_hides_sensitive_values(self) -> None:
		self.assertEqual(
			bench.redact_command([
				"workload",
				"--api-key",
				"visible-no-more",
				"--password=hidden",
				"--shape",
				"32",
			]),
			[
				"workload",
				"--api-key",
				"[REDACTED]",
				"--password=[REDACTED]",
				"--shape",
				"32",
			],
		)

	def test_missing_tool_version_is_recorded(self) -> None:
		self.assertEqual(
			bench.command_version(("oars-tool-that-does-not-exist", "--version")),
			"unavailable",
		)


if __name__ == "__main__":
	unittest.main()
