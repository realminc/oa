#!/usr/bin/env python3
"""Run OARS's checked-in benchmark workload suite."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Sequence

import bench


ROOT = Path(__file__).resolve().parents[2]
SCHEMA = "oars.benchmark_suite.v1"
DEFAULT_CONFIG = Path(__file__).with_name("suite.json")


def load_suite(path: Path) -> list[dict[str, Any]]:
	document = json.loads(path.read_text(encoding="utf-8"))
	if document.get("schema") != SCHEMA:
		raise ValueError("benchmark suite has an unsupported schema")
	workloads = document.get("workloads")
	if not isinstance(workloads, list) or not workloads:
		raise ValueError("benchmark suite has no workloads")
	names: set[str] = set()
	for workload in workloads:
		name = workload.get("name")
		if not isinstance(name, str) or not name or name in names:
			raise ValueError(f"invalid or duplicate workload name: {name!r}")
		for required in ("command_id", "command", "metric_regex", "metric_name", "metric_unit"):
			if not workload.get(required):
				raise ValueError(f"workload {name} has no {required}")
		names.add(name)
	return workloads


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
	parser = argparse.ArgumentParser(description=__doc__)
	parser.add_argument("--repo", type=Path, default=ROOT)
	parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
	parser.add_argument("--binary-root", dest="binary_root", type=Path)
	parser.add_argument("--output-dir", dest="output_dir", type=Path)
	parser.add_argument("--workload", action="append", default=[])
	parser.add_argument("--device-index", dest="device_index", type=int, default=0)
	parser.add_argument("--list", action="store_true")
	parser.add_argument("--allow-dirty", action="store_true")
	return parser.parse_args(argv)


def expand_command(items: Sequence[str], binary_root: Path, device_index: int) -> list[str]:
	return [
		item.replace("{binary_root}", str(binary_root)).replace("{device_index}", str(device_index))
		for item in items
	]


def run(args: argparse.Namespace) -> int:
	repo = args.repo.expanduser().resolve()
	config = args.config.expanduser().resolve()
	workloads = load_suite(config)
	if args.list:
		for workload in workloads:
			print(workload["name"])
		return 0
	selected = set(args.workload)
	known = {workload["name"] for workload in workloads}
	if unknown := selected - known:
		raise ValueError(f"unknown workloads: {', '.join(sorted(unknown))}")
	if selected:
		workloads = [workload for workload in workloads if workload["name"] in selected]
	if bench.git(repo, "status", "--porcelain") and not args.allow_dirty:
		raise ValueError(
			"repository is dirty; commit the exact tree or pass --allow-dirty for exploratory output"
		)
	if "VK_LAYER_KHRONOS_validation" in os.environ.get("VK_INSTANCE_LAYERS", ""):
		raise ValueError("validation instrumentation must be disabled for benchmark timing")
	binary_root = (
		args.binary_root.expanduser().resolve()
		if args.binary_root
		else repo / "bin/release"
	)
	for workload in workloads:
		bench.executable_evidence(
			expand_command(workload["command"], binary_root, args.device_index), repo
		)
	output_dir = (
		args.output_dir.expanduser().resolve()
		if args.output_dir
		else repo / "var/report/benchmark-suite" / dt.datetime.now().strftime("%Y%m%d-%H%M%S")
	)
	output_dir.mkdir(parents=True, exist_ok=False)

	results: list[dict[str, Any]] = []
	failures = 0
	for workload in workloads:
		output = output_dir / f"{workload['name']}.json"
		arguments = [
			"--repo", str(repo),
			"--output", str(output),
			"--name", workload["name"],
			"--command-id", workload["command_id"],
			"--device-index", str(args.device_index),
			"--profile", "release",
			"--warmup", str(workload.get("warmup", 2)),
			"--runs", str(workload.get("runs", 7)),
			"--cooldown", str(workload.get("cooldown", 2.0)),
			"--timeout", str(workload.get("timeout", 120.0)),
			"--metric-regex", workload["metric_regex"],
			"--metric-name", workload["metric_name"],
			"--metric-unit", workload["metric_unit"],
		]
		if args.allow_dirty:
			arguments.append("--allow-dirty")
		if "max_spread_percent" in workload:
			arguments.extend(("--max-spread-percent", str(workload["max_spread_percent"])))
		if "thermal_limit_celsius" in workload:
			arguments.extend((
				"--thermal-limit-celsius",
				str(workload["thermal_limit_celsius"]),
				"--thermal-timeout",
				str(workload.get("thermal_timeout", 300.0)),
			))
		if "thermal_sensor_regex" in workload:
			arguments.extend(("--thermal-sensor-regex", workload["thermal_sensor_regex"]))
		for key, value in sorted(workload.get("contract", {}).items()):
			arguments.extend(("--contract", f"{key}={value}"))
		for pattern in workload.get("require_regex", []):
			arguments.extend(("--require-regex", pattern))
		arguments.append("--")
		arguments.extend(expand_command(workload["command"], binary_root, args.device_index))
		path, status = bench.run(bench.parse_args(arguments))
		document = json.loads(path.read_text(encoding="utf-8"))
		stats = document["metric"]["statistics"]
		if stats:
			print(
				f"{workload['name']}: median={stats['median']:.9f} {stats['unit']} "
				f"mad={stats['mad']:.9f} spread={stats['spread_percent']:.2f}% "
				f"result={document['result']}"
			)
		else:
			print(f"{workload['name']}: metric unavailable result={document['result']}")
		results.append({
			"name": workload["name"],
			"result": document["result"],
			"canonical": document["canonical"],
			"artifact": path.name,
			"sha256": bench.sha256(path),
		})
		failures += int(status != 0)

	summary = {
		"schema": SCHEMA,
		"created_utc": dt.datetime.now(dt.timezone.utc).isoformat().replace("+00:00", "Z"),
		"config": {"path": str(config), "sha256": bench.sha256(config)},
		"results": results,
		"result": "FAIL" if failures else "PASS",
		"canonical": all(result["canonical"] for result in results),
	}
	(output_dir / "suite.json").write_text(
		json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
	)
	print(output_dir)
	return 1 if failures else 0


def main(argv: Sequence[str] | None = None) -> int:
	try:
		return run(parse_args(sys.argv[1:] if argv is None else argv))
	except (FileExistsError, OSError, ValueError, TimeoutError, json.JSONDecodeError, re.error) as error:
		print(f"oars-bench-suite: {error}", file=sys.stderr)
		return 2


if __name__ == "__main__":
	raise SystemExit(main())
