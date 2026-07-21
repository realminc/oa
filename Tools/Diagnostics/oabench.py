#!/usr/bin/env python3
"""Run a reproducible fresh-process OA benchmark and emit canonical JSON."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import pathlib
import platform
import re
import shutil
import statistics
import subprocess
import sys
import tempfile
import time
from typing import Any, Sequence

import oaevidence


SCHEMA = "oa.benchmark.v1"
BASELINE_SCHEMA = "oa.benchmark_baseline.v1"
ANSI_ESCAPE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")
COMPARABLE_BUILD_KEYS = (
    "CMAKE_BUILD_TYPE",
    "OA_BUILD_SHARED",
    "OA_EMBED_SHADERS",
)


def _sha256(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _percentile(values: Sequence[float], fraction: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _statistics(values: Sequence[float], unit: str) -> dict[str, Any]:
    median = statistics.median(values)
    deviations = [abs(value - median) for value in values]
    mad = statistics.median(deviations)
    return {
        "count": len(values),
        "unit": unit,
        "median": median,
        "mad": mad,
        "relative_mad_percent": (100.0 * mad / median) if median else 0.0,
        "minimum": min(values),
        "maximum": max(values),
        "p10": _percentile(values, 0.10),
        "p90": _percentile(values, 0.90),
        "spread_percent":
            (100.0 * (max(values) - min(values)) / median) if median else 0.0,
    }


def _read_text(path: pathlib.Path) -> str | None:
    try:
        return path.read_text(encoding="utf-8", errors="replace").strip()
    except OSError:
        return None


def _power_metadata() -> dict[str, Any]:
    governors = {
        value
        for path in pathlib.Path("/sys/devices/system/cpu").glob(
            "cpu*/cpufreq/scaling_governor"
        )
        if (value := _read_text(path))
    }
    power_profile = None
    executable = shutil.which("powerprofilesctl")
    if executable:
        result = subprocess.run(
            (executable, "get"), text=True, capture_output=True, check=False
        )
        if result.returncode == 0:
            power_profile = result.stdout.strip()
    return {
        "cpu_governors": sorted(governors),
        "power_profile": power_profile,
    }


def _vulkan_summary(repo: pathlib.Path) -> dict[str, Any]:
    executable = shutil.which("vulkaninfo")
    if executable is None:
        return {"available": False, "reason": "vulkaninfo not found"}
    result = subprocess.run(
        (executable, "--summary"),
        cwd=repo,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    return {
        "available": result.returncode == 0,
        "exit_code": result.returncode,
        "summary": result.stdout,
        "diagnostics": result.stderr,
    }


def _platform_identity(
    vulkan: dict[str, Any], *, device_index: int
) -> dict[str, Any]:
    identity: dict[str, Any] = {
        "system": platform.system(),
        "machine": platform.machine(),
        "device_index": device_index,
    }
    if not vulkan.get("available"):
        return {**identity, "available": False}
    summary = str(vulkan.get("summary", ""))
    match = re.search(
        rf"(?ms)^GPU{device_index}:\s*$\n(?P<body>.*?)(?=^GPU\d+:\s*$|\Z)",
        summary,
    )
    if match is None:
        return {**identity, "available": False}
    fields: dict[str, str] = {}
    for line in match.group("body").splitlines():
        field = re.match(r"\s*([A-Za-z][A-Za-z0-9]*)\s*=\s*(.*?)\s*$", line)
        if field:
            fields[field.group(1)] = field.group(2)
    wanted = {
        "vendor_id": "vendorID",
        "device_id": "deviceID",
        "device_type": "deviceType",
        "device_name": "deviceName",
        "driver_id": "driverID",
        "driver_name": "driverName",
    }
    required = ("vendorID", "deviceID", "driverID")
    if any(not fields.get(name) for name in required):
        return {**identity, "available": False}
    identity["available"] = True
    identity.update({key: fields.get(source, "") for key, source in wanted.items()})
    return identity


def _driver_observation(
    vulkan: dict[str, Any], *, device_index: int
) -> dict[str, str]:
    if not vulkan.get("available"):
        return {}
    summary = str(vulkan.get("summary", ""))
    match = re.search(
        rf"(?ms)^GPU{device_index}:\s*$\n(?P<body>.*?)(?=^GPU\d+:\s*$|\Z)",
        summary,
    )
    if match is None:
        return {}
    fields: dict[str, str] = {}
    for line in match.group("body").splitlines():
        field = re.match(r"\s*([A-Za-z][A-Za-z0-9]*)\s*=\s*(.*?)\s*$", line)
        if field:
            fields[field.group(1)] = field.group(2)
    return {
        key: fields[source]
        for key, source in {
            "api_version": "apiVersion",
            "driver_version": "driverVersion",
            "driver_info": "driverInfo",
            "conformance_version": "conformanceVersion",
        }.items()
        if source in fields
    }


def _executable_evidence(command: Sequence[str], repo: pathlib.Path) -> dict[str, str]:
    requested = command[0]
    candidate = pathlib.Path(requested)
    if not candidate.is_absolute():
        discovered = shutil.which(requested)
        candidate = pathlib.Path(discovered) if discovered else repo / candidate
    try:
        resolved = candidate.resolve()
        if resolved.is_file():
            return {"path": str(resolved), "sha256": _sha256(resolved)}
    except OSError:
        pass
    return {"path": requested, "sha256": "unavailable"}


def _comparable_build(build: dict[str, Any]) -> dict[str, Any]:
    return {key: build.get(key) for key in COMPARABLE_BUILD_KEYS}


def _extract_metric(pattern: re.Pattern[str], text: str) -> tuple[float, int]:
    matches = list(pattern.finditer(ANSI_ESCAPE.sub("", text)))
    if not matches:
        raise ValueError("metric regex did not match workload output")
    match = matches[-1]
    value = match.groupdict().get("value")
    if value is None:
        if match.lastindex != 1:
            raise ValueError(
                "metric regex must contain one capture or a named 'value' capture"
            )
        value = match.group(1)
    return float(value), len(matches)


def _parse_contract(items: Sequence[str]) -> dict[str, str]:
    contract: dict[str, str] = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"contract entry must be key=value: {item}")
        key, value = item.split("=", 1)
        if not key or key in contract:
            raise ValueError(f"invalid or duplicate contract key: {key!r}")
        contract[key] = value
    return contract


def _comparison(
    candidate: dict[str, Any],
    baseline: dict[str, Any],
    *,
    threshold_percent: float,
    direction: str,
) -> dict[str, Any]:
    candidate_stats = candidate["metric"]["statistics"]
    baseline_stats = baseline["metric"]["statistics"]
    if candidate["workload"]["name"] != baseline["workload"]["name"]:
        raise ValueError("baseline workload name differs")
    if candidate["workload"].get("contract") != baseline["workload"].get("contract"):
        raise ValueError("baseline workload contract differs")
    if candidate["workload"].get("command_id") != baseline["workload"].get(
        "command_id"
    ):
        raise ValueError("baseline workload command identity differs")
    if candidate["metric"]["name"] != baseline["metric"]["name"]:
        raise ValueError("baseline metric name differs")
    if candidate_stats["unit"] != baseline_stats["unit"]:
        raise ValueError("baseline metric unit differs")
    if candidate.get("platform") != baseline.get("platform"):
        raise ValueError("baseline platform identity differs")
    if _comparable_build(candidate.get("build", {})) != _comparable_build(
        baseline.get("build", {})
    ):
        raise ValueError("baseline build identity differs")
    base = float(baseline_stats["median"])
    current = float(candidate_stats["median"])
    if base == 0.0:
        raise ValueError("baseline median is zero")
    signed_change = 100.0 * (current / base - 1.0)
    regression = signed_change if direction == "lower" else -signed_change
    noise_band = max(
        float(candidate_stats["relative_mad_percent"]),
        float(baseline_stats["relative_mad_percent"]),
    )
    actionable = regression > threshold_percent and regression > noise_band
    return {
        "baseline_commit": baseline.get("repository", {}).get(
            "commit", baseline.get("source", {}).get("repository_commit", "unknown")
        ),
        "baseline_median": base,
        "candidate_median": current,
        "signed_change_percent": signed_change,
        "regression_percent": regression,
        "noise_band_percent": noise_band,
        "threshold_percent": threshold_percent,
        "direction": direction,
        "actionable_regression": actionable,
        "result": "FAIL" if actionable else "PASS",
    }


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=pathlib.Path, default=pathlib.Path.cwd())
    parser.add_argument("--output", type=pathlib.Path, required=True)
    parser.add_argument("--name", required=True, help="stable workload name")
    parser.add_argument(
        "--command-id",
        help="stable logical command identity independent of the binary root",
    )
    parser.add_argument(
        "--device-index",
        type=int,
        default=int(os.environ.get("OA_DEVICE", "0"))
        if os.environ.get("OA_DEVICE", "0").isdigit()
        else 0,
    )
    parser.add_argument("--contract", action="append", default=[])
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--runs", type=int, default=7)
    parser.add_argument("--cooldown", type=float, default=2.0)
    parser.add_argument("--timeout", type=float, default=3600.0)
    parser.add_argument("--metric-regex")
    parser.add_argument("--metric-name", default="process_wall_ms")
    parser.add_argument("--metric-unit", default="ms")
    parser.add_argument("--require-regex", action="append", default=[])
    parser.add_argument("--baseline", type=pathlib.Path)
    parser.add_argument("--regression-threshold", type=float, default=3.0)
    parser.add_argument("--direction", choices=("lower", "higher"), default="lower")
    parser.add_argument(
        "--cmake-cache",
        type=pathlib.Path,
        default=pathlib.Path("Build/Release/CMakeCache.txt"),
    )
    parser.add_argument("command", nargs=argparse.REMAINDER)
    return parser.parse_args(argv)


def run(args: argparse.Namespace) -> tuple[pathlib.Path, int]:
    if args.runs < 7:
        raise ValueError("canonical short-workload protocol requires at least 7 runs")
    if args.warmup < 0 or args.cooldown < 0.0 or args.timeout <= 0.0:
        raise ValueError("warmup/cooldown must be non-negative and timeout positive")
    repo = args.repo.expanduser().resolve()
    output = args.output.expanduser().resolve()
    if output.exists():
        raise FileExistsError(output)
    logs = output.with_suffix(output.suffix + ".logs")
    if logs.exists():
        raise FileExistsError(logs)
    command = list(args.command)
    if command and command[0] == "--":
        command = command[1:]
    if not command:
        raise ValueError("workload command is required after --")
    if args.device_index < 0:
        raise ValueError("device index must be non-negative")
    contract = _parse_contract(args.contract)
    metric_pattern = re.compile(args.metric_regex) if args.metric_regex else None
    required_patterns = [re.compile(pattern) for pattern in args.require_regex]

    output.parent.mkdir(parents=True, exist_ok=True)
    logs.mkdir(parents=True)
    samples: list[dict[str, Any]] = []
    values: list[float] = []
    failure = False
    total = args.warmup + args.runs
    for index in range(total):
        phase = "warmup" if index < args.warmup else "measured"
        phase_index = index if phase == "warmup" else index - args.warmup
        started = time.monotonic()
        result = subprocess.run(
            command,
            cwd=repo,
            env=os.environ.copy(),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=args.timeout,
            check=False,
        )
        wall_ms = (time.monotonic() - started) * 1000.0
        stem = f"{phase}-{phase_index:02d}"
        stdout_path = logs / f"{stem}.stdout.txt"
        stderr_path = logs / f"{stem}.stderr.txt"
        stdout_path.write_bytes(result.stdout)
        stderr_path.write_bytes(result.stderr)
        combined = (
            result.stdout.decode("utf-8", errors="replace")
            + "\n"
            + result.stderr.decode("utf-8", errors="replace")
        )
        normalized = ANSI_ESCAPE.sub("", combined)
        required = [bool(pattern.search(normalized)) for pattern in required_patterns]
        metric_value: float | None = wall_ms
        match_count = 0
        if metric_pattern is not None and result.returncode == 0:
            metric_value, match_count = _extract_metric(metric_pattern, normalized)
        elif metric_pattern is not None:
            metric_value = None
        sample = {
            "phase": phase,
            "index": phase_index,
            "process_wall_ms": wall_ms,
            "metric_value": metric_value,
            "metric_match_count": match_count,
            "exit_code": result.returncode,
            "required_patterns_matched": required,
            "stdout": {
                "path": f"{logs.name}/{stdout_path.name}",
                "bytes": stdout_path.stat().st_size,
                "sha256": _sha256(stdout_path),
            },
            "stderr": {
                "path": f"{logs.name}/{stderr_path.name}",
                "bytes": stderr_path.stat().st_size,
                "sha256": _sha256(stderr_path),
            },
        }
        samples.append(sample)
        if result.returncode != 0 or not all(required):
            failure = True
            break
        if phase == "measured":
            assert metric_value is not None
            values.append(metric_value)
        if index + 1 < total and args.cooldown:
            time.sleep(args.cooldown)

    if len(values) != args.runs:
        failure = True
    stats = _statistics(values, args.metric_unit) if values else None
    cache = args.cmake_cache.expanduser()
    if not cache.is_absolute():
        cache = repo / cache
    commit = oaevidence._git(repo, "rev-parse", "HEAD") or "unknown"
    vulkan = _vulkan_summary(repo)
    command_id = args.command_id or " ".join(oaevidence._redact_command(command))
    document: dict[str, Any] = {
        "schema": SCHEMA,
        "created_utc": dt.datetime.now(dt.timezone.utc).isoformat().replace(
            "+00:00", "Z"
        ),
        "repository": {
            "commit": commit,
            "branch": oaevidence._git(repo, "branch", "--show-current"),
            "describe": oaevidence._git(repo, "describe", "--always", "--dirty", "--tags"),
            "dirty": bool(oaevidence._git(repo, "status", "--porcelain") or ""),
        },
        "host": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
            "python": platform.python_version(),
            "power": _power_metadata(),
        },
        "build": oaevidence._read_cache(cache.resolve()),
        "vulkan": vulkan,
        "platform": _platform_identity(vulkan, device_index=args.device_index),
        "driver": _driver_observation(vulkan, device_index=args.device_index),
        "workload": {
            "name": args.name,
            "contract": contract,
            "command_id": command_id,
            "command": oaevidence._redact_command(command),
            "executable": _executable_evidence(command, repo),
            "warmup_runs": args.warmup,
            "measured_runs": args.runs,
            "cooldown_seconds": args.cooldown,
            "timeout_seconds": args.timeout,
            "required_regexes": args.require_regex,
        },
        "metric": {
            "name": args.metric_name,
            "regex": args.metric_regex,
            "statistics": stats,
        },
        "samples": samples,
        "comparison": None,
        "result": "FAIL" if failure else "PASS",
    }
    if args.baseline and not failure:
        baseline = json.loads(args.baseline.read_text(encoding="utf-8"))
        if baseline.get("schema") not in (SCHEMA, BASELINE_SCHEMA):
            raise ValueError("baseline is not an OA benchmark baseline")
        document["comparison"] = _comparison(
            document,
            baseline,
            threshold_percent=args.regression_threshold,
            direction=args.direction,
        )
        if document["comparison"]["actionable_regression"]:
            document["result"] = "FAIL"
            failure = True

    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=output.parent, delete=False
    ) as stream:
        json.dump(document, stream, indent=2, sort_keys=True)
        stream.write("\n")
        temporary = pathlib.Path(stream.name)
    temporary.replace(output)
    return output, 1 if failure else 0


def main(argv: Sequence[str] | None = None) -> int:
    try:
        output, status = run(parse_args(sys.argv[1:] if argv is None else argv))
        document = json.loads(output.read_text(encoding="utf-8"))
        stats = document["metric"]["statistics"]
        print(output)
        if stats:
            print(
                f"{document['workload']['name']}: median={stats['median']:.6f} "
                f"{stats['unit']} mad={stats['mad']:.6f} "
                f"spread={stats['spread_percent']:.2f}% "
                f"result={document['result']}"
            )
        return status
    except (
        FileExistsError,
        ValueError,
        re.error,
        subprocess.TimeoutExpired,
        json.JSONDecodeError,
    ) as error:
        print(f"oabench: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
