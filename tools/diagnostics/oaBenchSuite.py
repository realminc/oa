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

import oaBaseline
import oaBench


SCHEMA = "oa.benchmark_suite.v1"
ROOFLINE_SCHEMA = "oa.device_roofline.v1"
DEFAULT_CONFIG = pathlib.Path(__file__).with_name("benchmark_suite.json")
ROOFLINE_COPY = "runtime.roofline.copy"
ROOFLINE_COMPUTE = "runtime.roofline.fp32_gemm"


def _loadSuite(path: pathlib.Path) -> list[dict[str, Any]]:
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


def _platformKey(identity: dict[str, Any]) -> str:
    if not identity.get("available"):
        raise ValueError("no selected Vulkan device identity is available")
    return "-".join(
        _slug(str(identity[key]))
        for key in ("system", "machine", "vendor_id", "device_id", "driver_id")
    )


def _expandedCommand(
    workload: dict[str, Any], *, binaryRoot: pathlib.Path
) -> list[str]:
    return [
        str(item).replace("{binary_root}", str(binaryRoot))
        for item in workload["command"]
    ]


def _writeRooflineSummary(
    completed: Sequence[tuple[dict[str, Any], pathlib.Path, pathlib.Path]],
    *,
    platformIdentity: dict[str, Any],
    outputDir: pathlib.Path,
) -> pathlib.Path | None:
    results = {
        workload["name"]: result
        for workload, result, _ in completed
        if workload["name"] in (ROOFLINE_COPY, ROOFLINE_COMPUTE)
    }
    if set(results) != {ROOFLINE_COPY, ROOFLINE_COMPUTE}:
        return None
    documents = {
        name: json.loads(path.read_text(encoding="utf-8"))
        for name, path in results.items()
    }
    copyGbps = float(
        documents[ROOFLINE_COPY]["metric"]["statistics"]["median"]
    )
    computeGflops = float(
        documents[ROOFLINE_COMPUTE]["metric"]["statistics"]["median"]
    )
    if copyGbps <= 0.0 or computeGflops <= 0.0:
        raise ValueError("roofline probes must report positive medians")
    summary = {
        "schema": ROOFLINE_SCHEMA,
        "created_utc": dt.datetime.now(dt.timezone.utc).isoformat().replace(
            "+00:00", "Z"
        ),
        "platform": platformIdentity,
        "ceilings": {
            "delivered_fp32_gflops": computeGflops,
            "copy_effective_gbps": copyGbps,
            "ridge_point_flop_per_algorithmic_byte": computeGflops / copyGbps,
        },
        "interpretation": {
            "compute": "median delivered throughput of OA's production FP32 GEMM route",
            "bandwidth": "median Copy read+write algorithmic bytes per GPU timestamp; not a DRAM hardware counter",
            "hardware_counter_bandwidth": "unmeasured",
        },
        "probes": {
            name: {
                "result": path.name,
                "sha256": oaBench._sha256(path),
            }
            for name, path in results.items()
        },
    }
    path = outputDir / "device-roofline.json"
    path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def parseArgs(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=pathlib.Path, default=pathlib.Path.cwd())
    parser.add_argument("--config", type=pathlib.Path, default=DEFAULT_CONFIG)
    parser.add_argument("--binary-root", dest="binaryRoot", type=pathlib.Path)
    parser.add_argument("--cmake-cache", dest="cmakeCache", type=pathlib.Path)
    parser.add_argument("--output-dir", dest="outputDir", type=pathlib.Path)
    parser.add_argument("--workload", action="append", default=[])
    parser.add_argument("--device-index", dest="deviceIndex", type=int, default=0)
    parser.add_argument("--list", action="store_true")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--record", action="store_true", help="run without comparison")
    mode.add_argument("--accept", action="store_true", help="accept new baselines")
    parser.add_argument("--accept-reason", dest="acceptReason")
    parser.add_argument("--accepted-by", dest="acceptedBy", default="OA maintainers")
    return parser.parse_args(argv)


def run(args: argparse.Namespace) -> int:
    repo = args.repo.expanduser().resolve()
    config = args.config.expanduser().resolve()
    workloads = _loadSuite(config)
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
    if args.accept and not args.acceptReason:
        raise ValueError("--accept requires --accept-reason")

    binaryRoot = (
        args.binaryRoot.expanduser().resolve()
        if args.binaryRoot
        else repo / "bin/release"
    )
    cmakeCache = (
        args.cmakeCache.expanduser().resolve()
        if args.cmakeCache
        else repo / "build/release/CMakeCache.txt"
    )
    outputDir = (
        args.outputDir.expanduser().resolve()
        if args.outputDir
        else repo
        / "var/report/benchmark-suite"
        / dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    )
    outputDir.mkdir(parents=True, exist_ok=False)

    vulkan = oaBench._vulkanSummary(repo)
    identity = oaBench._platformIdentity(vulkan, device_index=args.deviceIndex)
    platformKey = _platformKey(identity)
    baselineRoot = repo / "tools/diagnostics/baselines" / platformKey
    completed: list[tuple[dict[str, Any], pathlib.Path, pathlib.Path]] = []
    failures = 0

    for workload in workloads:
        name = workload["name"]
        output = outputDir / f"{name}.json"
        baseline = baselineRoot / f"{name}.json"
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
            str(args.deviceIndex),
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
            str(cmakeCache),
        ]
        if "thermal_limit_celsius" in workload:
            argv.extend((
                "--thermal-limit-celsius",
                str(workload["thermal_limit_celsius"]),
                "--thermal-timeout",
                str(workload.get("thermal_timeout", 300.0)),
            ))
        if "thermal_sensor_regex" in workload:
            argv.extend((
                "--thermal-sensor-regex",
                workload["thermal_sensor_regex"],
            ))
        if "max_spread_percent" in workload:
            argv.extend((
                "--max-spread-percent",
                str(workload["max_spread_percent"]),
            ))
        for key, value in sorted(workload.get("contract", {}).items()):
            argv.extend(("--contract", f"{key}={value}"))
        for pattern in workload.get("require_regex", []):
            argv.extend(("--require-regex", pattern))
        if not args.record and not args.accept:
            argv.extend(("--baseline", str(baseline)))
        argv.append("--")
        argv.extend(_expandedCommand(workload, binaryRoot=binaryRoot))
        _, status = oaBench.run(oaBench.parseArgs(argv))
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
    roofline = _writeRooflineSummary(
        completed, platformIdentity=identity, outputDir=outputDir
    )
    if roofline is not None:
        print(f"device roofline: {roofline}")
    # Accept only after every result was captured from the still-clean tree.
    if args.accept:
        for _, result, baseline in completed:
            oaBaseline.accept(
                result,
                baseline,
                reason=args.acceptReason,
                accepted_by=args.acceptedBy,
            )
            print(f"accepted {baseline}")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    try:
        return run(parseArgs(sys.argv[1:] if argv is None else argv))
    except (
        FileExistsError,
        OSError,
        ValueError,
        json.JSONDecodeError,
        re.error,
    ) as error:
        print(f"oaBenchSuite: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
