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

import oaEvidence


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


def _readText(path: pathlib.Path) -> str | None:
    try:
        return path.read_text(encoding="utf-8", errors="replace").strip()
    except OSError:
        return None


def _powerMetadata() -> dict[str, Any]:
    governors = {
        value
        for path in pathlib.Path("/sys/devices/system/cpu").glob(
            "cpu*/cpufreq/scaling_governor"
        )
        if (value := _readText(path))
    }
    powerProfile = None
    profileDetails: dict[str, str] = {}
    executable = shutil.which("powerprofilesctl")
    if executable:
        result = subprocess.run(
            (executable, "get"), text=True, capture_output=True, check=False
        )
        if result.returncode == 0:
            powerProfile = result.stdout.strip()
        listed = subprocess.run(
            (executable, "list"), text=True, capture_output=True, check=False
        )
        if listed.returncode == 0 and powerProfile:
            currentProfile = None
            for line in listed.stdout.splitlines():
                header = re.match(r"^\s*(?:\*\s*)?([A-Za-z0-9_-]+):\s*$", line)
                if header:
                    currentProfile = header.group(1)
                    continue
                field = re.match(r"^\s+([A-Za-z][A-Za-z0-9]*):\s*(.*?)\s*$", line)
                if field and currentProfile == powerProfile:
                    profileDetails[field.group(1)] = field.group(2)
    return {
        "cpu_governors": sorted(governors),
        "power_profile": powerProfile,
        "power_profile_details": profileDetails,
    }


def _thermalSnapshot() -> list[dict[str, Any]]:
    """Read named Linux hwmon temperatures without requiring lm-sensors."""
    sensors: list[dict[str, Any]] = []
    for root in sorted(pathlib.Path("/sys/class/hwmon").glob("hwmon*")):
        device = _readText(root / "name") or root.name
        try:
            deviceIdentity = (root / "device").resolve(strict=True).name
        except OSError:
            deviceIdentity = root.name
        for inputPath in sorted(root.glob("temp*_input")):
            raw = _readText(inputPath)
            if raw is None:
                continue
            try:
                celsius = float(raw) / 1000.0
            except ValueError:
                continue
            prefix = inputPath.name.removesuffix("_input")
            label = _readText(root / f"{prefix}_label") or prefix
            sensors.append({
                "sensor": f"{device}:{deviceIdentity}:{label}",
                "celsius": celsius,
            })
    return sensors


def _thermalSummary(samples: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    observed: dict[str, list[float]] = {}
    for sample in samples:
        for key in ("thermal_before", "thermal_after"):
            for sensor in sample.get(key, []):
                observed.setdefault(str(sensor["sensor"]), []).append(
                    float(sensor["celsius"])
                )
    return [
        {
            "sensor": name,
            "minimum_celsius": min(values),
            "maximum_celsius": max(values),
        }
        for name, values in sorted(observed.items())
    ]


def _waitForThermalLimit(
    *,
    limitCelsius: float | None,
    sensorPattern: re.Pattern[str] | None,
    timeoutSeconds: float,
) -> tuple[list[dict[str, Any]], float, float | None]:
    started = time.monotonic()
    while True:
        snapshot = _thermalSnapshot()
        matching = [
            float(sensor["celsius"])
            for sensor in snapshot
            if float(sensor["celsius"]) > 0.0
            and (
                sensorPattern is None
                or sensorPattern.search(str(sensor["sensor"])) is not None
            )
        ]
        maximum = max(matching) if matching else None
        elapsed = time.monotonic() - started
        if limitCelsius is None:
            return snapshot, elapsed, maximum
        if maximum is None:
            raise ValueError("thermal gate matched no positive-temperature sensor")
        if maximum <= limitCelsius:
            return snapshot, elapsed, maximum
        if elapsed >= timeoutSeconds:
            raise TimeoutError(
                f"thermal gate timed out at {maximum:.1f} C "
                f"(limit {limitCelsius:.1f} C)"
            )
        time.sleep(min(5.0, timeoutSeconds - elapsed))


def _vulkanSummary(repo: pathlib.Path) -> dict[str, Any]:
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


def _platformIdentity(
    vulkan: dict[str, Any], *, deviceIndex: int
) -> dict[str, Any]:
    identity: dict[str, Any] = {
        "system": platform.system(),
        "machine": platform.machine(),
        "device_index": deviceIndex,
    }
    if not vulkan.get("available"):
        return {**identity, "available": False}
    summary = str(vulkan.get("summary", ""))
    match = re.search(
        rf"(?ms)^GPU{deviceIndex}:\s*$\n(?P<body>.*?)(?=^GPU\d+:\s*$|\Z)",
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


def _driverObservation(
    vulkan: dict[str, Any], *, deviceIndex: int
) -> dict[str, str]:
    if not vulkan.get("available"):
        return {}
    summary = str(vulkan.get("summary", ""))
    match = re.search(
        rf"(?ms)^GPU{deviceIndex}:\s*$\n(?P<body>.*?)(?=^GPU\d+:\s*$|\Z)",
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


def _executableEvidence(command: Sequence[str], repo: pathlib.Path) -> dict[str, str]:
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


def _comparableBuild(build: dict[str, Any]) -> dict[str, Any]:
    return {key: build.get(key) for key in COMPARABLE_BUILD_KEYS}


def _extractMetric(pattern: re.Pattern[str], text: str) -> tuple[float, int]:
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


def _parseContract(items: Sequence[str]) -> dict[str, str]:
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
    thresholdPercent: float,
    direction: str,
) -> dict[str, Any]:
    candidateStats = candidate["metric"]["statistics"]
    baselineStats = baseline["metric"]["statistics"]
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
    if candidateStats["unit"] != baselineStats["unit"]:
        raise ValueError("baseline metric unit differs")
    if candidate.get("platform") != baseline.get("platform"):
        raise ValueError("baseline platform identity differs")
    if _comparableBuild(candidate.get("build", {})) != _comparableBuild(
        baseline.get("build", {})
    ):
        raise ValueError("baseline build identity differs")
    base = float(baselineStats["median"])
    current = float(candidateStats["median"])
    if base == 0.0:
        raise ValueError("baseline median is zero")
    signedChange = 100.0 * (current / base - 1.0)
    regression = signedChange if direction == "lower" else -signedChange
    noiseBand = max(
        float(candidateStats["relative_mad_percent"]),
        float(baselineStats["relative_mad_percent"]),
    )
    actionable = regression > thresholdPercent and regression > noiseBand
    return {
        "baseline_commit": baseline.get("repository", {}).get(
            "commit", baseline.get("source", {}).get("repository_commit", "unknown")
        ),
        "baseline_median": base,
        "candidate_median": current,
        "signed_change_percent": signedChange,
        "regression_percent": regression,
        "noise_band_percent": noiseBand,
        "threshold_percent": thresholdPercent,
        "direction": direction,
        "actionable_regression": actionable,
        "result": "FAIL" if actionable else "PASS",
    }


def parseArgs(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=pathlib.Path, default=pathlib.Path.cwd())
    parser.add_argument("--output", type=pathlib.Path, required=True)
    parser.add_argument("--name", required=True, help="stable workload name")
    parser.add_argument(
        "--command-id", dest="commandId",
        help="stable logical command identity independent of the binary root",
    )
    parser.add_argument(
        "--device-index", dest="deviceIndex",
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
    parser.add_argument("--thermal-limit-celsius", dest="thermalLimitCelsius", type=float)
    parser.add_argument("--thermal-sensor-regex", dest="thermalSensorRegex")
    parser.add_argument("--thermal-timeout", dest="thermalTimeout", type=float, default=300.0)
    parser.add_argument("--metric-regex", dest="metricRegex")
    parser.add_argument("--metric-name", dest="metricName", default="process_wall_ms")
    parser.add_argument("--metric-unit", dest="metricUnit", default="ms")
    parser.add_argument("--require-regex", dest="requireRegex", action="append", default=[])
    parser.add_argument("--baseline", type=pathlib.Path)
    parser.add_argument("--regression-threshold", dest="regressionThreshold", type=float, default=3.0)
    parser.add_argument("--max-spread-percent", dest="maxSpreadPercent", type=float)
    parser.add_argument("--direction", choices=("lower", "higher"), default="lower")
    parser.add_argument(
        "--cmake-cache", dest="cmakeCache",
        type=pathlib.Path,
        default=pathlib.Path("build/release/CMakeCache.txt"),
    )
    parser.add_argument("command", nargs=argparse.REMAINDER)
    return parser.parse_args(argv)


def run(args: argparse.Namespace) -> tuple[pathlib.Path, int]:
    if args.runs < 7:
        raise ValueError("canonical short-workload protocol requires at least 7 runs")
    if args.warmup < 0 or args.cooldown < 0.0 or args.timeout <= 0.0:
        raise ValueError("warmup/cooldown must be non-negative and timeout positive")
    if args.thermalLimitCelsius is not None and args.thermalLimitCelsius <= 0.0:
        raise ValueError("thermal limit must be positive")
    if args.thermalTimeout <= 0.0:
        raise ValueError("thermal timeout must be positive")
    if args.maxSpreadPercent is not None and args.maxSpreadPercent <= 0.0:
        raise ValueError("maximum spread must be positive")
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
    if args.deviceIndex < 0:
        raise ValueError("device index must be non-negative")
    contract = _parseContract(args.contract)
    metricPattern = re.compile(args.metricRegex) if args.metricRegex else None
    requiredPatterns = [re.compile(pattern) for pattern in args.requireRegex]
    thermalPattern = (
        re.compile(args.thermalSensorRegex)
        if args.thermalSensorRegex else None
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    logs.mkdir(parents=True)
    samples: list[dict[str, Any]] = []
    values: list[float] = []
    failure = False
    total = args.warmup + args.runs
    for index in range(total):
        phase = "warmup" if index < args.warmup else "measured"
        phaseIndex = index if phase == "warmup" else index - args.warmup
        thermalBefore, thermalWaitSeconds, thermalStartMax = (
            _waitForThermalLimit(
                limitCelsius=args.thermalLimitCelsius,
                sensorPattern=thermalPattern,
                timeoutSeconds=args.thermalTimeout,
            )
        )
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
        wallMs = (time.monotonic() - started) * 1000.0
        thermalAfter = _thermalSnapshot()
        stem = f"{phase}-{phaseIndex:02d}"
        stdoutPath = logs / f"{stem}.stdout.txt"
        stderrPath = logs / f"{stem}.stderr.txt"
        stdoutPath.write_bytes(result.stdout)
        stderrPath.write_bytes(result.stderr)
        combined = (
            result.stdout.decode("utf-8", errors="replace")
            + "\n"
            + result.stderr.decode("utf-8", errors="replace")
        )
        normalized = ANSI_ESCAPE.sub("", combined)
        required = [bool(pattern.search(normalized)) for pattern in requiredPatterns]
        metricValue: float | None = wallMs
        matchCount = 0
        if metricPattern is not None and result.returncode == 0:
            metricValue, matchCount = _extractMetric(metricPattern, normalized)
        elif metricPattern is not None:
            metricValue = None
        sample = {
            "phase": phase,
            "index": phaseIndex,
            "process_wall_ms": wallMs,
            "metric_value": metricValue,
            "metric_match_count": matchCount,
            "exit_code": result.returncode,
            "required_patterns_matched": required,
            "thermal_before": thermalBefore,
            "thermal_after": thermalAfter,
            "thermal_wait_seconds": thermalWaitSeconds,
            "thermal_start_max_celsius": thermalStartMax,
            "stdout": {
                "path": f"{logs.name}/{stdoutPath.name}",
                "bytes": stdoutPath.stat().st_size,
                "sha256": _sha256(stdoutPath),
            },
            "stderr": {
                "path": f"{logs.name}/{stderrPath.name}",
                "bytes": stderrPath.stat().st_size,
                "sha256": _sha256(stderrPath),
            },
        }
        samples.append(sample)
        if result.returncode != 0 or not all(required):
            failure = True
            break
        if phase == "measured":
            assert metricValue is not None
            values.append(metricValue)
        if index + 1 < total and args.cooldown:
            time.sleep(args.cooldown)

    if len(values) != args.runs:
        failure = True
    stats = _statistics(values, args.metricUnit) if values else None
    if (
        stats is not None
        and args.maxSpreadPercent is not None
        and float(stats["spread_percent"]) > args.maxSpreadPercent
    ):
        failure = True
    cache = args.cmakeCache.expanduser()
    if not cache.is_absolute():
        cache = repo / cache
    commit = oaEvidence._git(repo, "rev-parse", "HEAD") or "unknown"
    vulkan = _vulkanSummary(repo)
    commandId = args.commandId or " ".join(oaEvidence._redactCommand(command))
    document: dict[str, Any] = {
        "schema": SCHEMA,
        "created_utc": dt.datetime.now(dt.timezone.utc).isoformat().replace(
            "+00:00", "Z"
        ),
        "repository": {
            "commit": commit,
            "branch": oaEvidence._git(repo, "branch", "--show-current"),
            "describe": oaEvidence._git(repo, "describe", "--always", "--dirty", "--tags"),
            "dirty": bool(oaEvidence._git(repo, "status", "--porcelain") or ""),
        },
        "host": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
            "python": platform.python_version(),
            "power": _powerMetadata(),
            "thermal": _thermalSummary(samples),
        },
        "build": oaEvidence._readCache(cache.resolve()),
        "vulkan": vulkan,
        "platform": _platformIdentity(vulkan, deviceIndex=args.deviceIndex),
        "driver": _driverObservation(vulkan, deviceIndex=args.deviceIndex),
        "workload": {
            "name": args.name,
            "contract": contract,
            "command_id": commandId,
            "command": oaEvidence._redactCommand(command),
            "executable": _executableEvidence(command, repo),
            "warmup_runs": args.warmup,
            "measured_runs": args.runs,
            "cooldown_seconds": args.cooldown,
            "timeout_seconds": args.timeout,
            "thermal_limit_celsius": args.thermalLimitCelsius,
            "thermal_sensor_regex": args.thermalSensorRegex,
            "thermal_timeout_seconds": args.thermalTimeout,
            "max_spread_percent": args.maxSpreadPercent,
            "required_regexes": args.requireRegex,
        },
        "metric": {
            "name": args.metricName,
            "regex": args.metricRegex,
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
            thresholdPercent=args.regressionThreshold,
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
        output, status = run(parseArgs(sys.argv[1:] if argv is None else argv))
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
        TimeoutError,
        re.error,
        subprocess.TimeoutExpired,
        json.JSONDecodeError,
    ) as error:
        print(f"oaBench: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
