#!/usr/bin/env python3
"""Run matched OA C++/GLM and OARS VLM benchmarks in alternating processes."""

from __future__ import annotations

import argparse
import datetime as dt
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


SCHEMA = "oa.vlm-comparison.v1"
CPP_CASE = re.compile(
	r"^PAIR precision=(?P<precision>\S+) case=(?P<case>\S+) contract=(?P<contract>\S+) "
	r"oa_ns=(?P<oa_ns>\S+) glm_ns=(?P<glm_ns>\S+).* checksum=(?P<checksum>\S+)$"
)
RUST_CASE = re.compile(
	r"^CASE precision=(?P<precision>\S+) case=(?P<case>\S+) contract=(?P<contract>\S+) "
	r"oars_ns=(?P<oars_ns>\S+).* checksum=(?P<checksum>\S+)$"
)


def parse_cases(language: str, output: bytes) -> dict[tuple[str, str], dict[str, Any]]:
	text = output.decode("utf-8", errors="strict")
	pattern = CPP_CASE if language == "oa_cpp" else RUST_CASE
	cases: dict[tuple[str, str], dict[str, Any]] = {}
	for line in text.splitlines():
		match = pattern.match(line)
		if match is None:
			continue
		fields = match.groupdict()
		key = (fields["precision"], fields["case"])
		if key in cases:
			raise ValueError(f"duplicate VLM result: {key}")
		row: dict[str, Any] = {
			"precision": fields["precision"],
			"case": fields["case"],
			"contract": fields["contract"],
			"checksum": float(fields["checksum"]),
		}
		if row["precision"] not in ("f32", "f64") or not math.isfinite(row["checksum"]):
			raise ValueError(f"invalid VLM precision or checksum: {key}")
		for metric in ("oa_ns", "glm_ns", "oars_ns"):
			if fields.get(metric) is not None:
				row[metric] = float(fields[metric])
				if not math.isfinite(row[metric]) or row[metric] <= 0:
					raise ValueError(f"invalid VLM latency: {key} {metric}")
		cases[key] = row
	if not cases:
		raise ValueError(f"{language} VLM benchmark emitted no case rows")
	return cases


def validate_pair(
	cpp: dict[tuple[str, str], dict[str, Any]],
	rust: dict[tuple[str, str], dict[str, Any]],
) -> None:
	if cpp.keys() != rust.keys():
		raise ValueError("OA C++ and OARS emitted different VLM case sets")
	for key in cpp:
		if cpp[key]["contract"] != rust[key]["contract"]:
			raise ValueError(f"VLM contract differs for {key}")
		left = float(cpp[key]["checksum"])
		right = float(rust[key]["checksum"])
		if not math.isfinite(left) or not math.isfinite(right):
			raise ValueError(f"non-finite VLM checksum: {key}")
		tolerance = 2.0e-4 if key[0] == "f32" else 2.0e-11
		scale = max(1.0, abs(left), abs(right))
		if abs(left - right) > scale * tolerance:
			raise ValueError(
				f"VLM checksum differs for {key}: OA C++={left:.17g}, OARS={right:.17g}"
			)


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


def baseline_evidence(record: Path, rustflags: str) -> dict[str, Any]:
	"""Admit a previously recorded, immutable Rust binary without relabeling its source."""
	document = json.loads(record.read_text())
	source = document["sources"]["oars"]
	if document["result"] != "PASS" or source["dirty"]:
		raise ValueError("Rust baseline must come from a passing clean-source record")
	if document["build_contract"]["oars_rustflags"] != rustflags:
		raise ValueError("Rust baseline and candidate flags differ")
	if bench.sha256(Path(source["executable"]["path"])) != source["executable"]["sha256"]:
		raise ValueError("Rust baseline executable hash changed")
	return source


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
	parser = argparse.ArgumentParser(description=__doc__)
	parser.add_argument("--oa-repo", type=Path, required=True)
	parser.add_argument("--oars-repo", type=Path, required=True)
	parser.add_argument("--oa-binary", type=Path, required=True)
	parser.add_argument("--oars-binary", type=Path, required=True)
	parser.add_argument("--baseline-oars-record", type=Path,
		help="also rerun the immutable Rust binary from a previous passing comparison")
	parser.add_argument("--oa-build-flags", default="-O3 -DNDEBUG -march=native")
	parser.add_argument("--oars-rustflags", default=os.environ.get("RUSTFLAGS", ""))
	parser.add_argument("--output", type=Path, required=True)
	parser.add_argument("--items", type=int, default=1 << 16)
	parser.add_argument("--inner-warmups", type=int, default=3)
	parser.add_argument("--inner-samples", type=int, default=11)
	parser.add_argument("--precision", choices=("f32", "f64", "both"), default="both")
	parser.add_argument("--filter", default="")
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
	if min(args.items, args.inner_warmups, args.inner_samples) <= 0:
		raise ValueError("items, inner warmups, and inner samples must be positive")
	if args.inner_samples < 3 or args.runs < 7:
		raise ValueError("requires >=3 inner samples and >=7 measured fresh processes")
	if (
		args.warmup < 0
		or args.cooldown < 0.0
		or args.timeout <= 0.0
		or args.thermal_limit_celsius <= 0.0
		or args.thermal_timeout <= 0.0
	):
		raise ValueError("warmup/cooldown must be non-negative and timeout positive")
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
	if args.baseline_oars_record is not None:
		sources["oars_baseline"] = baseline_evidence(args.baseline_oars_record, args.oars_rustflags)
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

	common = [
		"--items", str(args.items),
		"--warmups", str(args.inner_warmups),
		"--samples", str(args.inner_samples),
		"--precision", args.precision,
	]
	if args.filter:
		common.extend(("--filter", args.filter))
	commands = {
		"oa_cpp": [str(oa_binary), *common],
		"oars": [str(oars_binary), *common],
	}
	if "oars_baseline" in sources:
		commands["oars_baseline"] = [sources["oars_baseline"]["executable"]["path"], *common]
	output.parent.mkdir(parents=True, exist_ok=True)
	logs.mkdir()
	samples: list[dict[str, Any]] = []
	aggregates: dict[tuple[str, str], dict[str, list[float]]] = {}
	failure = False
	for pair_index in range(args.warmup + args.runs):
		phase = "warmup" if pair_index < args.warmup else "measured"
		phase_index = pair_index if phase == "warmup" else pair_index - args.warmup
		order = ("oa_cpp", "oars_baseline", "oars") if "oars_baseline" in sources else ("oa_cpp", "oars")
		if pair_index % 2:
			order = tuple(reversed(order))
		pair_rows: dict[str, dict[tuple[str, str], dict[str, Any]]] = {}
		for language in order:
			thermal_before, thermal_wait = bench.wait_for_thermal_limit(
				args.thermal_limit_celsius, thermal_pattern, args.thermal_timeout
			)
			frequency_before = bench.frequency_snapshot()
			violations_before = bench.clock_contract_violations(frequency_before)
			if violations_before and not args.exploratory:
				raise ValueError(
					"fixed-clock admission failed before workload: "
					+ "; ".join(violations_before)
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
			frequency_after = bench.frequency_snapshot()
			violations_after = bench.clock_contract_violations(frequency_after)
			thermal_after = bench.thermal_snapshot()
			stem = f"{phase}-{phase_index:02d}-{language}"
			stdout_path = logs / f"{stem}.stdout.txt"
			stderr_path = logs / f"{stem}.stderr.txt"
			stdout_path.write_bytes(result.stdout)
			stderr_path.write_bytes(result.stderr)
			rows = parse_cases(language, result.stdout) if result.returncode == 0 else {}
			pair_rows[language] = rows
			samples.append({
				"phase": phase,
				"index": phase_index,
				"language": language,
				"order": list(order),
				"exit_code": result.returncode,
				"wall_seconds": time.monotonic() - started,
				"case_count": len(rows),
				"frequency_before": frequency_before,
				"frequency_after": frequency_after,
				"clock_contract_violations": violations_before + violations_after,
				"thermal_before": thermal_before,
				"thermal_after": thermal_after,
				"thermal_wait_seconds": thermal_wait,
				"stdout": str(stdout_path.relative_to(output.parent)),
				"stderr": str(stderr_path.relative_to(output.parent)),
				"stdout_sha256": bench.sha256(stdout_path),
				"stderr_sha256": bench.sha256(stderr_path),
			})
			if result.returncode != 0 or (violations_after and not args.exploratory):
				failure = True
				break
		if failure:
			break
		validate_pair(pair_rows["oa_cpp"], pair_rows["oars"])
		if "oars_baseline" in sources:
			validate_pair(pair_rows["oa_cpp"], pair_rows["oars_baseline"])
			validate_pair(pair_rows["oars_baseline"], pair_rows["oars"])
		if phase == "measured":
			for key, cpp in pair_rows["oa_cpp"].items():
				rust = pair_rows["oars"][key]
				values = aggregates.setdefault(
					key, {"oa_cpp_ns": [], "glm_ns": [], "oars_ns": []}
				)
				values["oa_cpp_ns"].append(cpp["oa_ns"])
				values["glm_ns"].append(cpp["glm_ns"])
				values["oars_ns"].append(rust["oars_ns"])
				if "oars_baseline" in sources:
					values.setdefault("baseline_oars_ns", []).append(pair_rows["oars_baseline"][key]["oars_ns"])
		if pair_index + 1 < args.warmup + args.runs and args.cooldown:
			time.sleep(args.cooldown)

	results = []
	for (precision, case), values in sorted(aggregates.items()):
		if any(len(metric) != args.runs for metric in values.values()):
			failure = True
		medians = {
			name: bench.sample_statistics(metric, "ns/item")
			for name, metric in values.items()
		}
		results.append({
			"precision": precision,
			"case": case,
			"implementations": medians,
			"oars_to_oa_cpp_ratio": medians["oars_ns"]["median"] / medians["oa_cpp_ns"]["median"],
			"oars_to_glm_ratio": medians["oars_ns"]["median"] / medians["glm_ns"]["median"],
		})
		if "baseline_oars_ns" in values:
			results[-1]["baseline_to_oars_speedup"] = bench.sample_statistics([
				before / after for before, after in zip(values["baseline_oars_ns"], values["oars_ns"])
			], "baseline_ns/candidate_ns")
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
			"items": args.items,
			"precision": args.precision,
			"filter": args.filter,
			"inner_warmups": args.inner_warmups,
			"inner_samples": args.inner_samples,
			"warmup_process_pairs": args.warmup,
			"measured_process_pairs": args.runs,
			"cooldown_seconds": args.cooldown,
			"thermal_limit_celsius": args.thermal_limit_celsius,
			"thermal_sensor_regex": args.thermal_sensor_regex,
			"thermal_timeout_seconds": args.thermal_timeout,
			"language_order": "alternated_between_process_pairs",
			"cpu_affinity": sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None,
			"clock": {key: value for key, value in clock_contract.items() if value is not None},
		},
		"sources": sources,
		"build_contract": {
			"oa_cpp_flags": args.oa_build_flags,
			"oars_rustflags": args.oars_rustflags,
		},
		"host": {"power": bench.power_snapshot(), "uname": list(os.uname())},
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
		print(f"vlm-compare: {error}", file=sys.stderr)
		return 2


if __name__ == "__main__":
	raise SystemExit(main())
