#!/usr/bin/env python3
"""Run matched OA C++ and OARS memory benchmarks in alternating processes."""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import io
import json
import math
import os
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Sequence

import bench


SCHEMA = "oa.memory-comparison.v1"
MEASUREMENT_FIELDS = {"implementation", "median_ns", "p10_ns", "p90_ns", "GB_per_s"}
IMPLEMENTATION_NAMES = {
	"oa_runtime": "oa_cpp",
	"libc_runtime": "cpp_std",
	"oa_fixed": "oa_cpp_fixed",
	"compiler_fixed": "cpp_fixed",
	"oa_stream": "oa_cpp_stream",
	"oa_rust": "oa_rust",
	"rust_std": "rust_std",
	"oa_rust_fixed": "oa_rust_fixed",
	"rust_fixed": "rust_fixed",
	"oa_rust_stream": "oa_rust_stream",
}


def parse_rows(output: bytes) -> list[dict[str, str]]:
	text = output.decode("utf-8", errors="strict")
	reader = csv.DictReader(io.StringIO(text))
	if reader.fieldnames is None or not MEASUREMENT_FIELDS.issubset(reader.fieldnames):
		raise ValueError("memory benchmark emitted an incompatible CSV header")
	rows = list(reader)
	if not rows:
		raise ValueError("memory benchmark emitted no rows")
	for row in rows:
		implementation = row.get("implementation", "")
		if implementation not in IMPLEMENTATION_NAMES:
			raise ValueError(f"unknown memory implementation: {implementation}")
		for field in ("median_ns", "p10_ns", "p90_ns", "GB_per_s"):
			try:
				value = float(row[field])
			except (KeyError, TypeError, ValueError) as error:
				raise ValueError(f"invalid {field} for {implementation}") from error
			if not math.isfinite(value) or value <= 0.0:
				raise ValueError(f"non-finite or non-positive {field} for {implementation}")
	return rows


def case_key(row: dict[str, str]) -> tuple[tuple[str, str], ...]:
	return tuple(sorted(
		(key, value)
		for key, value in row.items()
		if key not in MEASUREMENT_FIELDS and value is not None
	))


def case_set(rows: Sequence[dict[str, str]]) -> set[tuple[tuple[str, str], ...]]:
	return {case_key(row) for row in rows}


def git(repo: Path, *arguments: str) -> str:
	result = subprocess.run(
		("git", *arguments), cwd=repo, text=True, capture_output=True, check=False
	)
	return result.stdout.strip() if result.returncode == 0 else ""


def source_evidence(repo: Path, executable: Path) -> dict[str, Any]:
	return {
		"repository": str(repo),
		"commit": git(repo, "rev-parse", "HEAD") or "unknown",
		"describe": git(repo, "describe", "--always", "--dirty", "--tags"),
		"dirty": bool(git(repo, "status", "--porcelain")),
		"executable": {"path": str(executable), "sha256": bench.sha256(executable)},
	}


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
	parser = argparse.ArgumentParser(description=__doc__)
	parser.add_argument("--oa-repo", type=Path, required=True)
	parser.add_argument("--oars-repo", type=Path, required=True)
	parser.add_argument("--oa-binary", type=Path, required=True)
	parser.add_argument("--oars-binary", type=Path, required=True)
	parser.add_argument("--oa-build-flags", default="-O3 -DNDEBUG -march=native")
	parser.add_argument("--oars-rustflags", default=os.environ.get("RUSTFLAGS", ""))
	parser.add_argument("--output", type=Path, required=True)
	parser.add_argument("--mode", choices=("copy", "streaming"), required=True)
	parser.add_argument("--quick", action="store_true")
	parser.add_argument("--warmup", type=int, default=2)
	parser.add_argument("--runs", type=int, default=7)
	parser.add_argument("--cooldown", type=float, default=3.0)
	parser.add_argument("--timeout", type=float, default=600.0)
	parser.add_argument("--thermal-limit-celsius", type=float, default=65.0)
	parser.add_argument("--thermal-sensor-regex", default="Package id 0")
	parser.add_argument("--thermal-timeout", type=float, default=300.0)
	parser.add_argument("--exploratory", action="store_true")
	return parser.parse_args(argv)


def run(args: argparse.Namespace) -> tuple[Path, int]:
	if (
		args.warmup < 0
		or args.runs < 7
		or args.cooldown < 0.0
		or args.timeout <= 0.0
		or args.thermal_limit_celsius <= 0.0
		or args.thermal_timeout <= 0.0
	):
		raise ValueError("requires non-negative warmup/cooldown, positive timeout, and >=7 runs")
	thermal_pattern = re.compile(args.thermal_sensor_regex)
	oa_repo = args.oa_repo.expanduser().resolve()
	oars_repo = args.oars_repo.expanduser().resolve()
	oa_binary = args.oa_binary.expanduser().resolve()
	oars_binary = args.oars_binary.expanduser().resolve()
	output = args.output.expanduser().resolve()
	logs = output.with_suffix(output.suffix + ".logs")
	if output.exists() or logs.exists():
		raise FileExistsError(output)
	for path in (oa_binary, oars_binary):
		if not path.is_file():
			raise ValueError(f"benchmark executable does not exist: {path}")
	sources = {
		"oa_cpp": source_evidence(oa_repo, oa_binary),
		"oars": source_evidence(oars_repo, oars_binary),
	}
	if not args.exploratory and any(source["dirty"] for source in sources.values()):
		raise ValueError("canonical comparison requires clean OA and OARS repositories")
	if not args.exploratory and "target-cpu=native" not in args.oars_rustflags:
		raise ValueError("canonical comparison requires recorded Rust target-cpu=native")
	clock_contract = {
		"cpu_fixed_khz": os.environ.get("OA_BENCH_CPU_FIXED_KHZ"),
		"gpu_fixed_mhz": os.environ.get("OA_BENCH_GPU_FIXED_MHZ"),
		"cpu_governor": os.environ.get("OA_BENCH_CPU_GOVERNOR"),
		"power_profile": os.environ.get("OA_BENCH_POWER_PROFILE"),
	}
	if not args.exploratory and any(value is None for value in clock_contract.values()):
		raise ValueError("canonical comparison must run inside stable_clocks.sh")

	output.parent.mkdir(parents=True, exist_ok=True)
	logs.mkdir()
	commands = {
		"oa_cpp": [str(oa_binary), f"--{args.mode}"],
		"oars": [str(oars_binary), f"--{args.mode}"],
	}
	if args.quick:
		for command in commands.values():
			command.insert(1, "--quick")

	total = args.warmup + args.runs
	samples: list[dict[str, Any]] = []
	aggregates: dict[tuple[tuple[tuple[str, str], ...], str], dict[str, list[float]]] = {}
	expected_cases: set[tuple[tuple[str, str], ...]] | None = None
	failure = False
	for pair_index in range(total):
		phase = "warmup" if pair_index < args.warmup else "measured"
		phase_index = pair_index if phase == "warmup" else pair_index - args.warmup
		order = ("oa_cpp", "oars") if pair_index % 2 == 0 else ("oars", "oa_cpp")
		for language in order:
			frequency_before = bench.frequency_snapshot()
			violations_before = bench.clock_contract_violations(frequency_before)
			if violations_before and not args.exploratory:
				raise ValueError(
					"fixed-clock admission failed before workload: "
					+ "; ".join(violations_before)
				)
			thermal_before, thermal_wait = bench.wait_for_thermal_limit(
				args.thermal_limit_celsius, thermal_pattern, args.thermal_timeout
			)
			started = time.monotonic()
			result = subprocess.run(
				commands[language],
				cwd=sources[language]["repository"],
				env=os.environ.copy(),
				stdout=subprocess.PIPE,
				stderr=subprocess.PIPE,
				timeout=args.timeout,
				check=False,
			)
			wall_seconds = time.monotonic() - started
			frequency_after = bench.frequency_snapshot()
			violations_after = bench.clock_contract_violations(frequency_after)
			thermal_after = bench.thermal_snapshot()
			stem = f"{phase}-{phase_index:02d}-{language}"
			stdout_path = logs / f"{stem}.csv"
			stderr_path = logs / f"{stem}.stderr.txt"
			stdout_path.write_bytes(result.stdout)
			stderr_path.write_bytes(result.stderr)
			rows = parse_rows(result.stdout) if result.returncode == 0 else []
			observed_cases = case_set(rows)
			if expected_cases is None and rows:
				expected_cases = observed_cases
			elif rows and observed_cases != expected_cases:
				raise ValueError("OA C++ and OARS emitted different memory case sets")
			sample = {
				"phase": phase,
				"index": phase_index,
				"language": language,
				"order": list(order),
				"command": commands[language],
				"exit_code": result.returncode,
				"wall_seconds": wall_seconds,
				"row_count": len(rows),
				"frequency_before": frequency_before,
				"frequency_after": frequency_after,
				"clock_contract_violations": violations_before + violations_after,
				"thermal_before": thermal_before,
				"thermal_after": thermal_after,
				"thermal_wait_seconds": thermal_wait,
				"stdout": str(stdout_path.relative_to(output.parent)),
				"stderr": str(stderr_path.relative_to(output.parent)),
			}
			samples.append(sample)
			if result.returncode != 0 or (violations_after and not args.exploratory):
				failure = True
				break
			if phase == "measured":
				for row in rows:
					implementation = IMPLEMENTATION_NAMES[row["implementation"]]
					key = (case_key(row), implementation)
					values = aggregates.setdefault(key, {"median_ns": [], "GB_per_s": []})
					values["median_ns"].append(float(row["median_ns"]))
					values["GB_per_s"].append(float(row["GB_per_s"]))
		if failure:
			break
		if pair_index + 1 < total and args.cooldown:
			time.sleep(args.cooldown)

	results = []
	for (case, implementation), values in sorted(aggregates.items()):
		if len(values["median_ns"]) != args.runs:
			failure = True
		results.append({
			"case": dict(case),
			"implementation": implementation,
			"latency": bench.sample_statistics(values["median_ns"], "ns"),
			"throughput": bench.sample_statistics(values["GB_per_s"], "GB/s"),
			"raw_process_median_ns": values["median_ns"],
			"raw_process_GB_per_s": values["GB_per_s"],
		})
	canonical_rejections = []
	if args.exploratory:
		canonical_rejections.append("exploratory_requested")
	if any(source["dirty"] for source in sources.values()):
		canonical_rejections.append("dirty_repository")
	if any(value is None for value in clock_contract.values()):
		canonical_rejections.append("fixed_clock_contract_missing")
	if "target-cpu=native" not in args.oars_rustflags:
		canonical_rejections.append("rust_target_cpu_native_unrecorded")
	document = {
		"schema": SCHEMA,
		"created_utc": dt.datetime.now(dt.timezone.utc).isoformat().replace("+00:00", "Z"),
		"result": "FAIL" if failure else "PASS",
		"canonical": not canonical_rejections and not failure,
		"canonical_rejections": canonical_rejections,
		"contract": {
			"mode": args.mode,
			"quick": args.quick,
			"warmup_process_pairs": args.warmup,
			"measured_process_pairs": args.runs,
			"cooldown_seconds": args.cooldown,
			"thermal_limit_celsius": args.thermal_limit_celsius,
			"thermal_sensor_regex": args.thermal_sensor_regex,
			"thermal_timeout_seconds": args.thermal_timeout,
			"inner_warmup_samples": 5,
			"inner_measured_samples": 21,
			"implementation_order": "rotated_within_process",
			"language_order": "alternated_between_process_pairs",
			"clock": {key: value for key, value in clock_contract.items() if value is not None},
		},
		"sources": sources,
		"build_contract": {
			"oa_cpp_flags": args.oa_build_flags,
			"oars_rustflags": args.oars_rustflags,
		},
		"host": {
			"power": bench.power_snapshot(),
			"system": os.uname().sysname,
			"release": os.uname().release,
			"machine": os.uname().machine,
		},
		"toolchain": {
			"clang": bench.command_version(("clang++", "--version")),
			"rustc": bench.command_version(("rustc", "-Vv")),
			"cargo": bench.command_version(("cargo", "--version")),
		},
		"samples": samples,
		"results": results,
	}
	with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=output.parent, delete=False) as stream:
		json.dump(document, stream, indent=2, sort_keys=True)
		stream.write("\n")
		temporary = Path(stream.name)
	temporary.replace(output)
	return output, 1 if failure else 0


def main(argv: Sequence[str] | None = None) -> int:
	try:
		output, status = run(parse_args(sys.argv[1:] if argv is None else argv))
		print(output)
		return status
	except (
		FileExistsError,
		OSError,
		UnicodeDecodeError,
		ValueError,
		subprocess.TimeoutExpired,
	) as error:
		print(f"memory-compare: {error}", file=sys.stderr)
		return 2


if __name__ == "__main__":
	raise SystemExit(main())
