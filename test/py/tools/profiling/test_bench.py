from __future__ import annotations

import json
import re
import sys
import tempfile
import unittest
from pathlib import Path


PROFILING = Path(__file__).resolve().parents[4] / "tools/profiling"
sys.path.insert(0, str(PROFILING))

import bench  # noqa: E402
import memory_compare  # noqa: E402
import suite  # noqa: E402
import vlm_compare  # noqa: E402


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

	def test_accepts_complete_fixed_clock_contract(self) -> None:
		snapshot = {
			"contract": {
				"cpu_fixed_khz": "2600000",
				"gpu_fixed_mhz": "1000",
				"cpu_governor": "performance",
				"power_profile": "performance",
			},
			"intel_pstate_no_turbo": "0",
			"cpu_policies": [{
				"policy": "policy0",
				"minimum_khz": "2600000",
				"maximum_khz": "2600000",
				"current_khz": "2599000",
				"governor": "performance",
			}],
			"gpu_engines": [{
				"engine": "freq0",
				"requested_minimum_mhz": "1000",
				"requested_maximum_mhz": "1000",
				"current_mhz": "1000",
				"actual_mhz": "1000",
				"throttle_status": "0",
				"throttle_reasons": "none",
			}],
			"power": {
				"power_profile": "performance",
				"performance_degraded": "",
			},
		}
		self.assertEqual(bench.clock_contract_violations(snapshot), [])

	def test_rejects_degraded_or_drifting_fixed_clock_contract(self) -> None:
		snapshot = {
			"contract": {
				"cpu_fixed_khz": "2600000",
				"gpu_fixed_mhz": "1000",
				"cpu_governor": "performance",
				"power_profile": "performance",
			},
			"intel_pstate_no_turbo": "0",
			"cpu_policies": [{
				"policy": "policy0",
				"minimum_khz": "400000",
				"maximum_khz": "2600000",
				"current_khz": "1800000",
				"governor": "powersave",
			}],
			"gpu_engines": [{
				"engine": "freq0",
				"requested_minimum_mhz": "500",
				"requested_maximum_mhz": "1000",
				"current_mhz": "800",
				"actual_mhz": "750",
				"throttle_status": "1",
				"throttle_reasons": "thermal",
			}],
			"power": {
				"power_profile": "performance",
				"performance_degraded": 's "lap-detected"',
			},
		}
		violations = bench.clock_contract_violations(snapshot)
		self.assertTrue(any("minimum_khz" in item for item in violations))
		self.assertTrue(any("governor" in item for item in violations))
		self.assertTrue(any("throttle" in item for item in violations))
		self.assertTrue(any("lap-detected" in item for item in violations))

	def test_memory_csv_parser_normalizes_matched_case(self) -> None:
		cpp = memory_compare.parse_rows(
			b"operation,chunk_bytes,working_set_bytes,implementation,passes,median_ns,p10_ns,p90_ns,GB_per_s\n"
			b"stream_copy,2048,268435456,oa_runtime,1,10,9,11,12.5\n"
		)
		rust = memory_compare.parse_rows(
			b"operation,chunk_bytes,working_set_bytes,implementation,passes,median_ns,p10_ns,p90_ns,GB_per_s\n"
			b"stream_copy,2048,268435456,oa_rust,1,9,8,10,13.5\n"
		)
		self.assertEqual(memory_compare.case_set(cpp), memory_compare.case_set(rust))

	def test_memory_csv_parser_rejects_unknown_implementation(self) -> None:
		with self.assertRaisesRegex(ValueError, "unknown memory implementation"):
			memory_compare.parse_rows(
				b"operation,size_bytes,src_offset,dst_offset,implementation,iterations,median_ns,p10_ns,p90_ns,GB_per_s\n"
				b"copy,8,0,0,mystery,1,1,1,1,8\n"
			)

	def test_vlm_parser_and_cross_language_oracle(self) -> None:
		cpp = vlm_compare.parse_cases(
			"oa_cpp",
			b"PAIR precision=f32 case=vec3_add contract=arithmetic oa_ns=2.0 glm_ns=1.9 ratio=1 delta_pct=0 oa_min=1 oa_max=3 glm_min=1 glm_max=3 checksum=12.5\n",
		)
		rust = vlm_compare.parse_cases(
			"oars",
			b"CASE precision=f32 case=vec3_add contract=arithmetic oars_ns=1.8 oars_min=1 oars_max=3 checksum=12.5001\n",
		)
		vlm_compare.validate_pair(cpp, rust)

	def test_vlm_cross_language_oracle_rejects_mismatch(self) -> None:
		cpp = vlm_compare.parse_cases(
			"oa_cpp",
			b"PAIR precision=f64 case=vec3_add contract=arithmetic oa_ns=2 glm_ns=2 ratio=1 delta_pct=0 oa_min=1 oa_max=3 glm_min=1 glm_max=3 checksum=12.5\n",
		)
		rust = vlm_compare.parse_cases(
			"oars",
			b"CASE precision=f64 case=vec3_add contract=arithmetic oars_ns=2 oars_min=1 oars_max=3 checksum=12.6\n",
		)
		with self.assertRaisesRegex(ValueError, "checksum differs"):
			vlm_compare.validate_pair(cpp, rust)

	def test_vlm_rejects_nonfinite_or_nonpositive_measurements(self) -> None:
		for latency, checksum in [("nan", "12"), ("inf", "12"), ("0", "12"), ("-1", "12"), ("2", "nan"), ("2", "inf")]:
			with self.subTest(latency=latency, checksum=checksum), self.assertRaises(ValueError):
				vlm_compare.parse_cases("oars", (
					f"CASE precision=f32 case=vec3_add contract=arithmetic oars_ns={latency} oars_min=1 oars_max=3 checksum={checksum}\n"
				).encode())
		row = {("f32", "vec3_add"): {"contract": "arithmetic", "checksum": float("nan")}}
		with self.assertRaisesRegex(ValueError, "non-finite"):
			vlm_compare.validate_pair(row, row)

	def test_vlm_baseline_requires_matching_hash_flags_and_clean_source(self) -> None:
		with tempfile.TemporaryDirectory() as directory:
			binary = Path(directory) / "baseline"
			binary.write_bytes(b"immutable test executable")
			source = {"dirty": False, "commit": "original-commit", "executable": {"path": str(binary), "sha256": bench.sha256(binary)}}
			document = {"result": "PASS", "sources": {"oars": source}, "build_contract": {"oars_rustflags": "native"}}
			record = Path(directory) / "baseline.json"
			record.write_text(json.dumps(document))
			self.assertEqual(vlm_compare.baseline_evidence(record, "native"), source)
			with self.assertRaisesRegex(ValueError, "flags differ"):
				vlm_compare.baseline_evidence(record, "portable")
			binary.write_bytes(b"changed")
			with self.assertRaisesRegex(ValueError, "hash changed"):
				vlm_compare.baseline_evidence(record, "native")
			source["dirty"] = True
			record.write_text(json.dumps(document))
			with self.assertRaisesRegex(ValueError, "clean-source"):
				vlm_compare.baseline_evidence(record, "native")


if __name__ == "__main__":
	unittest.main()
