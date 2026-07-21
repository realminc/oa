#!/usr/bin/env python3
"""Run OA's checked-in hardware-scoped benchmark workload suite."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import pathlib
import re
import sys
from typing import Any, Sequence

import oabaseline
import oabench


SCHEMA = "oa.benchmark_suite.v1"
DEFAULT_CONFIG = pathlib.Path(__file__).with_name("benchmark_suite.json")


def _load_suite(path: pathlib.Path) -> list[dict[str, Any]]:
    document = json.loads(path.read_text(encoding="utf-8"))
    if document.get("schema") != SCHEMA:
        raise ValueError("benchmark suite has an unsupported schema")
    workloads = document.get("workloads")
    if not isinstance(workloads, list) or not workloads:
        raise ValueError("benchmark suite has no workloads")
    names: set[str] = set()
    for workload in workloads:
        name = workload.get("name")
        if not name or name in names:
            raise ValueError(f"invalid or duplicate workload name: {name!r}")
        if not workload.get("command") or not workload.get("command_id"):
            raise ValueError(f"workload {name} has no command identity")
        names.add(name)
    return workloads


def _slug(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")


def _platform_key(identity: dict[str, Any]) -> str:
    if not identity.get("available"):
        raise ValueError("no selected Vulkan device identity is available")
    return "-".join(
        _slug(str(identity[key]))
        for key in ("system", "machine", "vendor_id", "device_id", "driver_id")
    )


def _expanded_command(
    workload: dict[str, Any], *, binary_root: pathlib.Path
) -> list[str]:
    return [
        str(item).replace("{binary_root}", str(binary_root))
        for item in workload["command"]
    ]


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=pathlib.Path, default=pathlib.Path.cwd())
    parser.add_argument("--config", type=pathlib.Path, default=DEFAULT_CONFIG)
    parser.add_argument("--binary-root", type=pathlib.Path)
    parser.add_argument("--cmake-cache", type=pathlib.Path)
    parser.add_argument("--output-dir", type=pathlib.Path)
    parser.add_argument("--workload", action="append", default=[])
    parser.add_argument("--device-index", type=int, default=0)
    parser.add_argument("--list", action="store_true")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--record", action="store_true", help="run without comparison")
    mode.add_argument("--accept", action="store_true", help="accept new baselines")
    parser.add_argument("--accept-reason")
    parser.add_argument("--accepted-by", default="OA maintainers")
    return parser.parse_args(argv)


def run(args: argparse.Namespace) -> int:
    repo = args.repo.expanduser().resolve()
    config = args.config.expanduser().resolve()
    workloads = _load_suite(config)
    if args.list:
        for workload in workloads:
            print(workload["name"])
        return 0
    selected = set(args.workload)
    known = {workload["name"] for workload in workloads}
    unknown = selected - known
    if unknown:
        raise ValueError(f"unknown workloads: {', '.join(sorted(unknown))}")
    if selected:
        workloads = [item for item in workloads if item["name"] in selected]
    if args.accept and not args.accept_reason:
        raise ValueError("--accept requires --accept-reason")

    binary_root = (
        args.binary_root.expanduser().resolve()
        if args.binary_root
        else repo / "Bin/Release"
    )
    cmake_cache = (
        args.cmake_cache.expanduser().resolve()
        if args.cmake_cache
        else repo / "Build/Release/CMakeCache.txt"
    )
    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir
        else repo
        / "var/report/benchmark-suite"
        / dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    )
    output_dir.mkdir(parents=True, exist_ok=False)

    vulkan = oabench._vulkan_summary(repo)
    identity = oabench._platform_identity(vulkan, device_index=args.device_index)
    platform_key = _platform_key(identity)
    baseline_root = repo / "Tools/Diagnostics/Baselines" / platform_key
    completed: list[tuple[dict[str, Any], pathlib.Path, pathlib.Path]] = []
    failures = 0

    for workload in workloads:
        name = workload["name"]
        output = output_dir / f"{name}.json"
        baseline = baseline_root / f"{name}.json"
        if not args.record and not args.accept and not baseline.is_file():
            raise ValueError(f"accepted baseline does not exist: {baseline}")
        argv = [
            "--repo",
            str(repo),
            "--output",
            str(output),
            "--name",
            name,
            "--command-id",
            workload["command_id"],
            "--device-index",
            str(args.device_index),
            "--warmup",
            str(workload.get("warmup", 2)),
            "--runs",
            str(workload.get("runs", 7)),
            "--cooldown",
            str(workload.get("cooldown", 2.0)),
            "--timeout",
            str(workload.get("timeout", 120.0)),
            "--metric-regex",
            workload["metric_regex"],
            "--metric-name",
            workload["metric_name"],
            "--metric-unit",
            workload["metric_unit"],
            "--direction",
            workload.get("direction", "lower"),
            "--cmake-cache",
            str(cmake_cache),
        ]
        for key, value in sorted(workload.get("contract", {}).items()):
            argv.extend(("--contract", f"{key}={value}"))
        for pattern in workload.get("require_regex", []):
            argv.extend(("--require-regex", pattern))
        if not args.record and not args.accept:
            argv.extend(("--baseline", str(baseline)))
        argv.append("--")
        argv.extend(_expanded_command(workload, binary_root=binary_root))
        _, status = oabench.run(oabench.parse_args(argv))
        completed.append((workload, output, baseline))
        document = json.loads(output.read_text(encoding="utf-8"))
        stats = document["metric"]["statistics"]
        print(
            f"{name}: median={stats['median']:.6f} {stats['unit']} "
            f"mad={stats['mad']:.6f} result={document['result']}"
        )
        failures += int(status != 0)

    if failures:
        return 1
    # Accept only after every result was captured from the still-clean tree.
    if args.accept:
        for _, result, baseline in completed:
            oabaseline.accept(
                result,
                baseline,
                reason=args.accept_reason,
                accepted_by=args.accepted_by,
            )
            print(f"accepted {baseline}")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    try:
        return run(parse_args(sys.argv[1:] if argv is None else argv))
    except (
        FileExistsError,
        OSError,
        ValueError,
        json.JSONDecodeError,
        re.error,
    ) as error:
        print(f"oabenchsuite: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
