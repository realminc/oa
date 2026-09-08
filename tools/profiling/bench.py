#!/usr/bin/env python3
"""Run one OARS workload in fresh processes and write a raw JSON artifact."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
import os
import platform
import re
import shutil
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Sequence


ROOT = Path(__file__).resolve().parents[2]
SCHEMA = "oars.benchmark.v1"
ANSI_ESCAPE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")
SENSITIVE_ARGUMENTS = ("token", "secret", "password", "api-key", "apikey")


def sha256(path: Path) -> str:
	digest = hashlib.sha256()
	with path.open("rb") as stream:
		for block in iter(lambda: stream.read(1024 * 1024), b""):
			digest.update(block)
	return digest.hexdigest()


def git(repo: Path, *arguments: str) -> str:
	result = subprocess.run(
		("git", *arguments), cwd=repo, text=True, capture_output=True, check=False
	)
	return result.stdout.strip() if result.returncode == 0 else ""


def percentile(values: Sequence[float], fraction: float) -> float:
	ordered = sorted(values)
	position = (len(ordered) - 1) * fraction
	lower = int(position)
	upper = min(lower + 1, len(ordered) - 1)
	weight = position - lower
	return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def sample_statistics(values: Sequence[float], unit: str) -> dict[str, Any]:
	if not values or not all(math.isfinite(value) for value in values):
		raise ValueError("statistics require finite samples")
	median = statistics.median(values)
	deviations = [abs(value - median) for value in values]
	scale = abs(median) or max(abs(value) for value in values)
	return {
		"count": len(values),
		"unit": unit,
		"median": median,
		"mad": statistics.median(deviations),
		"relative_mad_percent": 100.0 * statistics.median(deviations) / scale if scale else 0.0,
		"minimum": min(values),
		"maximum": max(values),
		"p10": percentile(values, 0.10),
		"p90": percentile(values, 0.90),
		"spread_percent": 100.0 * (max(values) - min(values)) / scale if scale else 0.0,
	}


def parse_contract(items: Sequence[str]) -> dict[str, str]:
	contract: dict[str, str] = {}
	for item in items:
		if "=" not in item:
			raise ValueError(f"contract entry must be key=value: {item}")
		key, value = item.split("=", 1)
		if not key or key in contract:
			raise ValueError(f"invalid or duplicate contract key: {key!r}")
		contract[key] = value
	return contract


def redact_command(command: Sequence[str]) -> list[str]:
	redacted: list[str] = []
	hide_next = False
	for argument in command:
		if hide_next:
			redacted.append("[REDACTED]")
			hide_next = False
			continue
		lowered = argument.lower()
		if argument.startswith("-") and any(word in lowered for word in SENSITIVE_ARGUMENTS):
			if "=" in argument:
				redacted.append(argument.split("=", 1)[0] + "=[REDACTED]")
			else:
				redacted.append(argument)
				hide_next = True
			continue
		redacted.append(argument)
	return redacted


def extract_metric(pattern: re.Pattern[str], text: str) -> tuple[float, int]:
	matches = list(pattern.finditer(ANSI_ESCAPE.sub("", text)))
	if not matches:
		raise ValueError("metric regex did not match workload output")
	match = matches[-1]
	value = match.groupdict().get("value")
	if value is None:
		if match.lastindex != 1:
			raise ValueError("metric regex needs one capture or a named 'value' capture")
		value = match.group(1)
	return float(value), len(matches)


def read_text(path: Path) -> str | None:
	try:
		return path.read_text(encoding="utf-8", errors="replace").strip()
	except OSError:
		return None


def thermal_snapshot() -> list[dict[str, Any]]:
	sensors: list[dict[str, Any]] = []
	for root in sorted(Path("/sys/class/hwmon").glob("hwmon*")):
		device = read_text(root / "name") or root.name
		for input_path in sorted(root.glob("temp*_input")):
			raw = read_text(input_path)
			if raw is None:
				continue
			try:
				celsius = float(raw) / 1000.0
			except ValueError:
				continue
			prefix = input_path.name.removesuffix("_input")
			label = read_text(root / f"{prefix}_label") or prefix
			sensors.append({"sensor": f"{device}:{label}", "celsius": celsius})
	return sensors


def wait_for_thermal_limit(
	limit: float | None, pattern: re.Pattern[str] | None, timeout: float
) -> tuple[list[dict[str, Any]], float]:
	started = time.monotonic()
	while True:
		snapshot = thermal_snapshot()
		values = [
			float(sensor["celsius"])
			for sensor in snapshot
			if float(sensor["celsius"]) > 0.0
			and (pattern is None or pattern.search(str(sensor["sensor"])))
		]
		elapsed = time.monotonic() - started
		if limit is None:
			return snapshot, elapsed
		if not values:
			raise ValueError("thermal gate matched no positive-temperature sensor")
		if max(values) <= limit:
			return snapshot, elapsed
		if elapsed >= timeout:
			raise TimeoutError(f"thermal gate timed out above {limit:.1f} C")
		time.sleep(min(5.0, timeout - elapsed))


def power_snapshot() -> dict[str, Any]:
	governors = sorted(
		{
			value
			for path in Path("/sys/devices/system/cpu").glob("cpu*/cpufreq/scaling_governor")
			if (value := read_text(path))
		}
	)
	power_profile = None
	performance_degraded = None
	powerprofilesctl = shutil.which("powerprofilesctl")
	if powerprofilesctl:
		result = subprocess.run(
			(powerprofilesctl, "get"), text=True, capture_output=True, check=False
		)
		if result.returncode == 0:
			power_profile = result.stdout.strip()
		busctl = shutil.which("busctl")
		if busctl:
			degraded = subprocess.run(
				(
					busctl,
					"get-property",
					"net.hadess.PowerProfiles",
					"/net/hadess/PowerProfiles",
					"net.hadess.PowerProfiles",
					"PerformanceDegraded",
				),
				text=True,
				capture_output=True,
				check=False,
			)
			if degraded.returncode == 0:
				value = degraded.stdout.strip()
				performance_degraded = "" if value == 's ""' else value
	return {
		"cpu_governors": governors,
		"intel_pstate_no_turbo": read_text(Path("/sys/devices/system/cpu/intel_pstate/no_turbo")),
		"power_profile": power_profile,
		"performance_degraded": performance_degraded,
	}


def frequency_snapshot(
	cpu_root: Path = Path("/sys/devices/system/cpu"),
	drm_root: Path = Path("/sys/class/drm"),
) -> dict[str, Any]:
	"""Capture the requested and observed Linux CPU/Xe clock state."""
	cpu_policies = []
	for root in sorted((cpu_root / "cpufreq").glob("policy*")):
		fields = {
			key: read_text(root / source)
			for key, source in {
				"minimum_khz": "scaling_min_freq",
				"maximum_khz": "scaling_max_freq",
				"current_khz": "scaling_cur_freq",
				"hardware_minimum_khz": "cpuinfo_min_freq",
				"hardware_maximum_khz": "cpuinfo_max_freq",
				"governor": "scaling_governor",
			}.items()
		}
		cpu_policies.append({
			"policy": root.name,
			**{key: value for key, value in fields.items() if value is not None},
		})

	gpu_engines = []
	for root in sorted(drm_root.glob("card[0-9]*/device/tile*/gt*/freq*")):
		fields = {
			key: read_text(root / source)
			for key, source in {
				"requested_minimum_mhz": "min_freq",
				"requested_maximum_mhz": "max_freq",
				"current_mhz": "cur_freq",
				"actual_mhz": "act_freq",
				"hardware_minimum_mhz": "rpn_freq",
				"hardware_maximum_mhz": "rp0_freq",
				"throttle_status": "throttle/status",
				"throttle_reasons": "throttle/reasons",
			}.items()
		}
		gpu_engines.append({
			"engine": str(root),
			**{key: value for key, value in fields.items() if value is not None},
		})

	return {
		"contract": {
			key: value
			for key, value in {
				"cpu_fixed_khz": os.environ.get("OA_BENCH_CPU_FIXED_KHZ")
				or os.environ.get("OA_BENCH_CPU_MAX_KHZ"),
				"gpu_fixed_mhz": os.environ.get("OA_BENCH_GPU_FIXED_MHZ"),
				"cpu_governor": os.environ.get("OA_BENCH_CPU_GOVERNOR"),
				"power_profile": os.environ.get("OA_BENCH_POWER_PROFILE"),
			}.items()
			if value is not None
		},
		"intel_pstate_no_turbo": read_text(cpu_root / "intel_pstate/no_turbo"),
		"cpu_policies": cpu_policies,
		"gpu_engines": gpu_engines,
		"power": power_snapshot(),
	}


def clock_contract_violations(snapshot: dict[str, Any]) -> list[str]:
	"""Return fail-closed violations for an advertised fixed-clock contract."""
	contract = snapshot.get("contract", {})
	if not contract:
		return []
	violations: list[str] = []

	def integer(value: Any, label: str) -> int | None:
		try:
			return int(str(value))
		except (TypeError, ValueError):
			violations.append(f"{label} is unavailable or non-numeric")
			return None

	fixed_cpu = (
		integer(contract["cpu_fixed_khz"], "CPU fixed-clock contract")
		if "cpu_fixed_khz" in contract else None
	)
	policies = snapshot.get("cpu_policies", [])
	if fixed_cpu is not None:
		if snapshot.get("intel_pstate_no_turbo") != "0":
			violations.append("Intel Turbo is unavailable or disabled")
		if not policies:
			violations.append("CPU frequency policies are unavailable")
		for policy in policies:
			name = policy.get("policy", "unknown")
			for field in ("minimum_khz", "maximum_khz"):
				observed = integer(policy.get(field), f"CPU {name} {field}")
				if observed is not None and observed != fixed_cpu:
					violations.append(
						f"CPU {name} {field} is {observed}, expected {fixed_cpu}"
					)
			current = integer(policy.get("current_khz"), f"CPU {name} current_khz")
			if current is not None and abs(current - fixed_cpu) > fixed_cpu * 0.02:
				violations.append(
					f"CPU {name} current_khz is {current}, expected {fixed_cpu} +/- 2%"
				)
	expected_governor = contract.get("cpu_governor")
	if expected_governor is not None:
		if not policies:
			violations.append("CPU governor state is unavailable")
		for policy in policies:
			if policy.get("governor") != expected_governor:
				violations.append(
					f"CPU {policy.get('policy', 'unknown')} governor is "
					f"{policy.get('governor', 'unavailable')}, expected {expected_governor}"
				)

	fixed_gpu = (
		integer(contract["gpu_fixed_mhz"], "GPU fixed-clock contract")
		if "gpu_fixed_mhz" in contract else None
	)
	engines = snapshot.get("gpu_engines", [])
	if fixed_gpu is not None:
		if not engines:
			violations.append("GPU frequency engines are unavailable")
		for engine in engines:
			name = engine.get("engine", "unknown")
			for field in ("requested_minimum_mhz", "requested_maximum_mhz"):
				observed = integer(engine.get(field), f"GPU {name} {field}")
				if observed is not None and observed != fixed_gpu:
					violations.append(
						f"GPU {name} {field} is {observed}, expected {fixed_gpu}"
					)
			for field in ("current_mhz", "actual_mhz"):
				observed = integer(engine.get(field), f"GPU {name} {field}")
				if observed is not None and abs(observed - fixed_gpu) > fixed_gpu * 0.02:
					violations.append(
						f"GPU {name} {field} is {observed}, expected {fixed_gpu} +/- 2%"
					)
			if engine.get("throttle_status") != "0":
				violations.append(
					f"GPU {name} throttle status is "
					f"{engine.get('throttle_status', 'unavailable')}"
				)
			if engine.get("throttle_reasons") != "none":
				violations.append(
					f"GPU {name} throttle reasons are "
					f"{engine.get('throttle_reasons', 'unavailable')}"
				)

	power = snapshot.get("power", {})
	expected_profile = contract.get("power_profile")
	if expected_profile is not None and power.get("power_profile") != expected_profile:
		violations.append(
			f"power profile is {power.get('power_profile', 'unavailable')}, "
			f"expected {expected_profile}"
		)
	degraded = power.get("performance_degraded")
	if expected_profile == "performance":
		if degraded is None:
			violations.append("performance degradation state is unavailable")
		elif degraded:
			violations.append(f"performance profile is degraded: {degraded}")
	return violations


def thermal_summary(samples: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
	observed: dict[str, list[float]] = {}
	for sample in samples:
		for key in ("thermal_before", "thermal_after"):
			for sensor in sample.get(key, []):
				observed.setdefault(str(sensor["sensor"]), []).append(float(sensor["celsius"]))
	return [
		{
			"sensor": sensor,
			"minimum_celsius": min(values),
			"maximum_celsius": max(values),
		}
		for sensor, values in sorted(observed.items())
	]


def vulkan_summary(repo: Path) -> dict[str, Any]:
	executable = shutil.which("vulkaninfo")
	if executable is None:
		return {"available": False, "reason": "vulkaninfo not found"}
	result = subprocess.run(
		(executable, "--summary"), cwd=repo, text=True, capture_output=True, check=False
	)
	return {
		"available": result.returncode == 0,
		"exit_code": result.returncode,
		"summary": result.stdout,
		"diagnostics": result.stderr,
	}


def find_vulkan_registry(repo: Path) -> Path | None:
	override = os.environ.get("OARS_VK_XML")
	if override:
		candidate = Path(override).expanduser()
		return candidate.resolve() if candidate.is_file() else None
	candidates = []
	for variable in ("VULKAN_SDK", "VK_SDK_PATH"):
		if sdk := os.environ.get(variable):
			candidates.append(Path(sdk) / "share/vulkan/registry/vk.xml")
	candidates.extend((repo / "vk.xml", Path("/usr/share/vulkan/registry/vk.xml")))
	return next((candidate.resolve() for candidate in candidates if candidate.is_file()), None)


def device_identity(vulkan: dict[str, Any], index: int) -> dict[str, Any]:
	identity: dict[str, Any] = {
		"system": platform.system(),
		"machine": platform.machine(),
		"device_index": index,
	}
	if not vulkan.get("available"):
		return {**identity, "available": False}
	match = re.search(
		rf"(?ms)^GPU{index}:\s*$\n(?P<body>.*?)(?=^GPU\d+:\s*$|\Z)",
		str(vulkan["summary"]),
	)
	if match is None:
		return {**identity, "available": False}
	fields: dict[str, str] = {}
	for line in match.group("body").splitlines():
		field = re.match(r"\s*([A-Za-z][A-Za-z0-9]*)\s*=\s*(.*?)\s*$", line)
		if field:
			fields[field.group(1)] = field.group(2)
	for required in ("vendorID", "deviceID", "driverID"):
		if not fields.get(required):
			return {**identity, "available": False}
	return {
		**identity,
		"available": True,
		"vendor_id": fields["vendorID"],
		"device_id": fields["deviceID"],
		"device_type": fields.get("deviceType", ""),
		"device_name": fields.get("deviceName", ""),
		"api_version": fields.get("apiVersion", ""),
		"driver_id": fields["driverID"],
		"driver_name": fields.get("driverName", ""),
		"driver_version": fields.get("driverVersion", ""),
		"driver_info": fields.get("driverInfo", ""),
	}


def command_version(command: Sequence[str]) -> str:
	try:
		result = subprocess.run(command, text=True, capture_output=True, check=False)
	except OSError:
		return "unavailable"
	text = (result.stdout or result.stderr).strip()
	return text if result.returncode == 0 else "unavailable"


def executable_evidence(command: Sequence[str], repo: Path) -> dict[str, str]:
	candidate = Path(command[0])
	if not candidate.is_absolute():
		discovered = shutil.which(command[0])
		candidate = Path(discovered) if discovered else repo / candidate
	resolved = candidate.resolve()
	if not resolved.is_file():
		raise ValueError(f"workload executable does not exist: {resolved}")
	return {"path": str(resolved), "sha256": sha256(resolved)}


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
	parser = argparse.ArgumentParser(description=__doc__)
	parser.add_argument("--repo", type=Path, default=ROOT)
	parser.add_argument("--output", type=Path, required=True)
	parser.add_argument("--name", required=True)
	parser.add_argument("--command-id", dest="command_id", required=True)
	parser.add_argument("--device-index", dest="device_index", type=int, default=0)
	parser.add_argument("--profile", choices=("debug", "release"), default="release")
	parser.add_argument("--contract", action="append", default=[])
	parser.add_argument("--warmup", type=int, default=2)
	parser.add_argument("--runs", type=int, default=7)
	parser.add_argument("--cooldown", type=float, default=2.0)
	parser.add_argument("--timeout", type=float, default=3600.0)
	parser.add_argument("--thermal-limit-celsius", dest="thermal_limit", type=float)
	parser.add_argument("--thermal-sensor-regex", dest="thermal_regex")
	parser.add_argument("--thermal-timeout", dest="thermal_timeout", type=float, default=300.0)
	parser.add_argument("--metric-regex", dest="metric_regex", required=True)
	parser.add_argument("--metric-name", dest="metric_name", required=True)
	parser.add_argument("--metric-unit", dest="metric_unit", required=True)
	parser.add_argument("--require-regex", dest="required_regexes", action="append", default=[])
	parser.add_argument("--max-spread-percent", dest="max_spread", type=float)
	parser.add_argument("--allow-dirty", action="store_true")
	parser.add_argument("command", nargs=argparse.REMAINDER)
	return parser.parse_args(argv)


def run(args: argparse.Namespace) -> tuple[Path, int]:
	repo = args.repo.expanduser().resolve()
	output = args.output.expanduser().resolve()
	if output.exists() or output.with_suffix(output.suffix + ".logs").exists():
		raise FileExistsError(output)
	if args.runs < 7:
		raise ValueError("canonical protocol requires at least seven measured fresh processes")
	if args.warmup < 0 or args.cooldown < 0.0 or args.timeout <= 0.0:
		raise ValueError("warmup/cooldown must be non-negative and timeout positive")
	if args.device_index < 0:
		raise ValueError("device index must be non-negative")
	if args.thermal_limit is not None and args.thermal_limit <= 0.0:
		raise ValueError("thermal limit must be positive")
	if args.thermal_timeout <= 0.0:
		raise ValueError("thermal timeout must be positive")
	if args.max_spread is not None and args.max_spread <= 0.0:
		raise ValueError("maximum spread must be positive")
	dirty = bool(git(repo, "status", "--porcelain"))
	if dirty and not args.allow_dirty:
		raise ValueError("repository is dirty; commit the exact tree or pass --allow-dirty for exploratory output")
	if "VK_LAYER_KHRONOS_validation" in os.environ.get("VK_INSTANCE_LAYERS", ""):
		raise ValueError("validation instrumentation must be disabled for benchmark timing")
	command = list(args.command)
	if command and command[0] == "--":
		command = command[1:]
	if not command:
		raise ValueError("workload command is required after --")
	contract = parse_contract(args.contract)
	for key, environment_name in {
		"cpu_fixed_khz": "OA_BENCH_CPU_FIXED_KHZ",
		"gpu_fixed_mhz": "OA_BENCH_GPU_FIXED_MHZ",
		"cpu_governor": "OA_BENCH_CPU_GOVERNOR",
		"power_profile": "OA_BENCH_POWER_PROFILE",
	}.items():
		value = os.environ.get(environment_name)
		if value is None:
			continue
		if key in contract and contract[key] != value:
			raise ValueError(
				f"workload contract {key}={contract[key]} conflicts with "
				f"{environment_name}={value}"
			)
		contract[key] = value
	if "cpu_fixed_khz" not in contract:
		legacy_cpu = os.environ.get("OA_BENCH_CPU_MAX_KHZ")
		if legacy_cpu is not None:
			contract["cpu_fixed_khz"] = legacy_cpu
	metric_pattern = re.compile(args.metric_regex)
	required_patterns = [re.compile(pattern) for pattern in args.required_regexes]
	thermal_pattern = re.compile(args.thermal_regex) if args.thermal_regex else None

	output.parent.mkdir(parents=True, exist_ok=True)
	logs = output.with_suffix(output.suffix + ".logs")
	logs.mkdir()
	samples: list[dict[str, Any]] = []
	values: list[float] = []
	failure = False
	for index in range(args.warmup + args.runs):
		phase = "warmup" if index < args.warmup else "measured"
		phase_index = index if phase == "warmup" else index - args.warmup
		thermal_before, thermal_wait = wait_for_thermal_limit(
			args.thermal_limit, thermal_pattern, args.thermal_timeout
		)
		frequency_before = frequency_snapshot()
		clock_violations_before = clock_contract_violations(frequency_before)
		if clock_violations_before:
			raise ValueError(
				"fixed-clock admission failed before workload: "
				+ "; ".join(clock_violations_before)
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
		process_wall_ms = (time.monotonic() - started) * 1000.0
		frequency_after = frequency_snapshot()
		clock_violations_after = clock_contract_violations(frequency_after)
		thermal_after = thermal_snapshot()
		stem = f"{phase}-{phase_index:02d}"
		stdout_path = logs / f"{stem}.stdout.txt"
		stderr_path = logs / f"{stem}.stderr.txt"
		stdout_path.write_bytes(result.stdout)
		stderr_path.write_bytes(result.stderr)
		combined = result.stdout.decode(errors="replace") + "\n" + result.stderr.decode(errors="replace")
		normalized = ANSI_ESCAPE.sub("", combined)
		required = [bool(pattern.search(normalized)) for pattern in required_patterns]
		metric_value: float | None = None
		match_count = 0
		metric_error = None
		if result.returncode == 0:
			try:
				metric_value, match_count = extract_metric(metric_pattern, normalized)
			except ValueError as error:
				metric_error = str(error)
		sample = {
			"phase": phase,
			"index": phase_index,
			"process_wall_ms": process_wall_ms,
			"metric_value": metric_value,
			"metric_match_count": match_count,
			"metric_error": metric_error,
			"exit_code": result.returncode,
			"required_patterns_matched": required,
			"thermal_before": thermal_before,
			"thermal_after": thermal_after,
			"thermal_wait_seconds": thermal_wait,
			"frequency_before": frequency_before,
			"frequency_after": frequency_after,
			"clock_contract_violations": clock_violations_after,
			"stdout": {
				"path": str(stdout_path.relative_to(output.parent)),
				"bytes": stdout_path.stat().st_size,
				"sha256": sha256(stdout_path),
			},
			"stderr": {
				"path": str(stderr_path.relative_to(output.parent)),
				"bytes": stderr_path.stat().st_size,
				"sha256": sha256(stderr_path),
			},
		}
		samples.append(sample)
		if (
			result.returncode != 0
			or not all(required)
			or metric_value is None
			or clock_violations_after
		):
			failure = True
			break
		if phase == "measured":
			values.append(metric_value)
		if index + 1 < args.warmup + args.runs and args.cooldown:
			time.sleep(args.cooldown)

	if len(values) != args.runs:
		failure = True
	stats = sample_statistics(values, args.metric_unit) if values else None
	if stats and args.max_spread is not None and stats["spread_percent"] > args.max_spread:
		failure = True
	vulkan = vulkan_summary(repo)
	manifest = repo / "Cargo.toml"
	lock = repo / "Cargo.lock"
	registry = find_vulkan_registry(repo)
	platform_identity = device_identity(vulkan, args.device_index)
	canonical_rejections = []
	if dirty:
		canonical_rejections.append("repository_dirty")
	if args.profile != "release":
		canonical_rejections.append("non_release_profile")
	if not platform_identity.get("available"):
		canonical_rejections.append("vulkan_device_identity_unavailable")
	if registry is None:
		canonical_rejections.append("vulkan_registry_unavailable")
	if os.environ.get("VK_INSTANCE_LAYERS"):
		canonical_rejections.append("explicit_vulkan_layers")
	document = {
		"schema": SCHEMA,
		"created_utc": dt.datetime.now(dt.timezone.utc).isoformat().replace("+00:00", "Z"),
		"result": "FAIL" if failure else "PASS",
		"canonical": not canonical_rejections,
		"canonical_rejections": canonical_rejections,
		"repository": {
			"commit": git(repo, "rev-parse", "HEAD") or "unknown",
			"branch": git(repo, "branch", "--show-current"),
			"describe": git(repo, "describe", "--always", "--dirty", "--tags"),
			"dirty": dirty,
		},
		"host": {
			"system": platform.system(),
			"release": platform.release(),
			"machine": platform.machine(),
			"python": platform.python_version(),
			"power": power_snapshot(),
			"thermal": thermal_summary(samples),
		},
		"toolchain": {
			"rustc": command_version(("rustc", "-Vv")),
			"cargo": command_version(("cargo", "--version")),
			"slangc": command_version((os.environ.get("SLANGC", "slangc"), "-version")),
			"spirv_val": command_version(
				(os.environ.get("SPIRV_VAL", "spirv-val"), "--version")
			),
		},
		"build": {
			"profile": args.profile,
			"rustflags": os.environ.get("RUSTFLAGS"),
			"cargo_encoded_rustflags": os.environ.get("CARGO_ENCODED_RUSTFLAGS"),
			"cargo_manifest_sha256": sha256(manifest),
			"cargo_lock_sha256": sha256(lock) if lock.is_file() else "unavailable",
			"vk_instance_layers": os.environ.get("VK_INSTANCE_LAYERS"),
		},
		"vulkan": vulkan,
		"vulkan_registry": {
			"path": str(registry) if registry else "unavailable",
			"sha256": sha256(registry) if registry else "unavailable",
		},
		"platform": platform_identity,
		"workload": {
			"name": args.name,
			"command_id": args.command_id,
			"contract": contract,
			"command": redact_command(command),
			"executable": executable_evidence(command, repo),
			"warmup_processes": args.warmup,
			"measured_processes": args.runs,
			"cooldown_seconds": args.cooldown,
			"timeout_seconds": args.timeout,
			"thermal_limit_celsius": args.thermal_limit,
			"thermal_sensor_regex": args.thermal_regex,
			"thermal_timeout_seconds": args.thermal_timeout,
			"max_spread_percent": args.max_spread,
			"required_regexes": args.required_regexes,
		},
		"metric": {
			"name": args.metric_name,
			"regex": args.metric_regex,
			"statistics": stats,
		},
		"samples": samples,
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
		document = json.loads(output.read_text(encoding="utf-8"))
		stats = document["metric"]["statistics"]
		print(output)
		if stats:
			print(
				f"{document['workload']['name']}: median={stats['median']:.9f} {stats['unit']} "
				f"mad={stats['mad']:.9f} spread={stats['spread_percent']:.2f}% "
				f"result={document['result']} canonical={int(document['canonical'])}"
			)
		return status
	except (FileExistsError, OSError, ValueError, TimeoutError, subprocess.TimeoutExpired, re.error) as error:
		print(f"oars-bench: {error}", file=sys.stderr)
		return 2


if __name__ == "__main__":
	raise SystemExit(main())
