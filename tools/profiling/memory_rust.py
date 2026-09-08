#!/usr/bin/env python3
"""Record OARS versus stock Rust using the repository's fresh-process runner."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import bench
import memory_compare


def summarize(streams: list[bytes], runs: int) -> list[dict]:
	"""Require every case/policy in every process; retain paired Rust ratios."""
	if len(streams) != runs:
		raise ValueError("incomplete measured process set")
	values: dict[tuple, list[float]] = {}
	expected = None
	for stream in streams:
		rows = memory_compare.parse_rows(stream)
		current = {}
		for row in rows:
			key = (memory_compare.case_key(row), row["implementation"])
			if key in current:
				raise ValueError("duplicate memory case/policy")
			current[key] = float(row["median_ns"])
		if expected is None:
			expected = current.keys()
		elif current.keys() != expected:
			raise ValueError("memory case/policy set changed between processes")
		for key, value in current.items():
			values.setdefault(key, []).append(value)
	results = []
	for (case, implementation), latencies in sorted(values.items()):
		baseline = "rust_fixed" if implementation.endswith("fixed") else "rust_std"
		if (case, baseline) not in values:
			raise ValueError("missing stock Rust baseline")
		# Ratios are paired inside each fresh process; a ratio above one is
		# faster. This avoids attributing between-process drift to the library.
		ratios = [stock / candidate for stock, candidate in zip(values[case, baseline], latencies)]
		results.append({
			"case": dict(case),
			"implementation": implementation,
			"baseline": baseline,
			"latency": bench.sample_statistics(latencies, "ns"),
			"speedup": bench.sample_statistics(ratios, "stock_ns/implementation_ns"),
			"raw_process_median_ns": latencies,
			"raw_paired_speedup": ratios,
		})
	return results


def main() -> int:
	parser = argparse.ArgumentParser(description=__doc__)
	parser.add_argument("--repo", type=Path, default=bench.ROOT)
	parser.add_argument("--binary", type=Path, required=True)
	parser.add_argument("--output", type=Path, required=True)
	parser.add_argument("--mode", choices=("copy", "streaming"), required=True)
	parser.add_argument("--quick", action="store_true")
	parser.add_argument("--runs", type=int, default=7)
	parser.add_argument("--cooldown", type=float, default=3.0)
	parser.add_argument("--observed-clocks", action="store_true",
		help="record development evidence without requiring the fixed-clock wrapper")
	parser.add_argument("--allow-dirty", action="store_true")
	args = parser.parse_args()
	if not args.observed_clocks and not all(os.environ.get(name) for name in (
		"OA_BENCH_CPU_FIXED_KHZ", "OA_BENCH_GPU_FIXED_MHZ",
		"OA_BENCH_CPU_GOVERNOR", "OA_BENCH_POWER_PROFILE",
	)):
		parser.error("use stable_clocks.sh or explicitly select --observed-clocks")
	command = [str(args.binary.resolve()), f"--{args.mode}"]
	if args.quick:
		command.append("--quick")
	# The scalar metric admits the CSV through the generic recorder. The
	# complete per-case distribution below is the memory acceptance surface.
	options = bench.parse_args([
		"--repo", str(args.repo), "--output", str(args.output),
		"--name", f"Rust memory {args.mode}", "--command-id", f"memory.{args.mode}.v2",
		"--profile", "release", "--warmup", "2", "--runs", str(args.runs),
		"--cooldown", str(args.cooldown), "--thermal-limit-celsius", "65",
		"--thermal-sensor-regex", "Package id 0",
		"--metric-regex", r"(?m)^.*,(?P<value>[0-9]+\.[0-9]+)$",
		"--metric-name", "last_csv_row", "--metric-unit", "GB/s",
		"--contract", "harness=opaque-slices-direct-policy-v2",
		*(["--allow-dirty"] if args.allow_dirty else []), "--", *command,
	])
	output, status = bench.run(options)
	document = json.loads(output.read_text())
	if args.observed_clocks:
		document["canonical_rejections"].append("observed_clocks_development_run")
	document["canonical"] = not document["canonical_rejections"] and status == 0
	try:
		streams = [
			(output.parent / sample["stdout"]["path"]).read_bytes()
			for sample in document["samples"] if sample["phase"] == "measured"
		]
		document["memory_results"] = summarize(streams, args.runs)
	except ValueError as error:
		document.update(result="FAIL", canonical=False, memory_error=str(error))
		status = 1
	output.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n")
	print(output)
	print(f"{document['result']}: {len(document.get('memory_results', []))} case/policy distributions")
	return status


if __name__ == "__main__":
	raise SystemExit(main())
