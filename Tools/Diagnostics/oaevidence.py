#!/usr/bin/env python3
"""Create one self-describing OA diagnostic evidence bundle.

The collector has no third-party Python dependencies. It may either snapshot
the current machine or wrap one workload after ``--``. Wrapped workloads get a
private OA_GRAPH_REPORT destination and optional Vulkan validation request.
"""

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
import subprocess
import sys
import tempfile
import time
from typing import Any, Iterable, Sequence


SCHEMA = "oa.diagnostic_evidence.v1"
ANSI_ESCAPE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")
GEMM_SELECTION = re.compile(
    r"GemmRouter: M=(?P<m>\d+) N=(?P<n>\d+) K=(?P<k>\d+) "
    r"requested=(?P<requested>\S+) actual=(?P<actual>\S+) "
    r"kernel=(?P<kernel>\S+) path=(?P<path>\S+) "
    r"fallback=(?P<fallback>\S+) grid=(?P<gx>\d+),(?P<gy>\d+),(?P<gz>\d+)"
)
RELEVANT_ENV = (
    "OA_DEVICE",
    "OA_PRECISION",
    "OA_NUMERIC_MODE",
    "OA_VK_VALIDATION",
    "OA_VK_VALIDATION_MODE",
    "OA_LOG_GEMM_ROUTER",
    "VK_DRIVER_FILES",
    "VK_ICD_FILENAMES",
    "VK_LAYER_PATH",
    "VK_ADD_LAYER_PATH",
)

VALIDATION_LAYER = re.compile(
    r"^VK_LAYER_KHRONOS_validation\s+.*?\s+"
    r"(?P<api>\d+\.\d+\.\d+)\s+version\s+(?P<implementation>\d+)\s*$",
    re.MULTILINE,
)


def _validation_layer_record(summary: str) -> dict[str, Any] | None:
    match = VALIDATION_LAYER.search(summary)
    if match is None:
        return None
    return {
        "name": "VK_LAYER_KHRONOS_validation",
        "api_version": match.group("api"),
        "implementation_version": int(match.group("implementation")),
    }


def _run(
    command: Sequence[str],
    *,
    cwd: pathlib.Path,
    env: dict[str, str] | None = None,
    timeout: int = 120,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        list(command),
        cwd=cwd,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=timeout,
        check=False,
    )


def _git(repo: pathlib.Path, *args: str) -> str | None:
    result = _run(("git", *args), cwd=repo, timeout=30)
    return result.stdout.strip() if result.returncode == 0 else None


def _sha256(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _artifact(path: pathlib.Path, root: pathlib.Path) -> dict[str, Any]:
    return {
        "path": path.relative_to(root).as_posix(),
        "bytes": path.stat().st_size,
        "sha256": _sha256(path),
    }


def _read_cache(path: pathlib.Path) -> dict[str, str]:
    wanted = {
        "CMAKE_BUILD_TYPE",
        "CMAKE_C_COMPILER",
        "CMAKE_CXX_COMPILER",
        "OA_BUILD_SHARED",
        "OA_EMBED_SHADERS",
        "OA_VULKAN_VALIDATION",
        "VCPKG_INSTALLED_DIR",
    }
    values: dict[str, str] = {}
    if not path.is_file():
        return values
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if not line or line.startswith(("#", "//")) or "=" not in line:
            continue
        key_and_type, value = line.split("=", 1)
        key = key_and_type.split(":", 1)[0]
        if key in wanted:
            values[key] = value
    return values


def _registry_candidates(repo: pathlib.Path) -> Iterable[pathlib.Path]:
    sdk = os.environ.get("VULKAN_SDK")
    if sdk:
        yield pathlib.Path(sdk) / "share/vulkan/registry/vk.xml"
        yield pathlib.Path(sdk) / "registry/vk.xml"
    yield repo / "Build/Release/vcpkg_installed/x64-linux/share/vulkan/registry/vk.xml"
    installed = os.environ.get("VCPKG_INSTALLED_DIR")
    if installed:
        yield pathlib.Path(installed) / "x64-linux/share/vulkan/registry/vk.xml"
    yield (
        pathlib.Path.home()
        / ".vcpkg/oa/release/installed/x64-linux/share/vulkan/registry/vk.xml"
    )
    yield pathlib.Path.home() / ".vcpkg/installed/x64-linux/share/vulkan/registry/vk.xml"
    yield pathlib.Path("/usr/share/vulkan/registry/vk.xml")


def _find_registry(repo: pathlib.Path) -> pathlib.Path | None:
    override = os.environ.get("OA_VK_XML")
    if override:
        candidate = pathlib.Path(override).expanduser()
        return candidate.resolve() if candidate.is_file() else None
    for candidate in _registry_candidates(repo):
        if candidate.is_file():
            return candidate.resolve()
    return None


def _redact_command(command: Sequence[str]) -> list[str]:
    redacted: list[str] = []
    hide_next = False
    sensitive = ("token", "secret", "password", "api-key", "apikey")
    for argument in command:
        if hide_next:
            redacted.append("[REDACTED]")
            hide_next = False
            continue
        lowered = argument.lower()
        if argument.startswith("-") and any(word in lowered for word in sensitive):
            if "=" in argument:
                redacted.append(argument.split("=", 1)[0] + "=[REDACTED]")
            else:
                redacted.append(argument)
                hide_next = True
            continue
        redacted.append(argument)
    return redacted


def _snapshot_vulkan(staging: pathlib.Path, repo: pathlib.Path) -> dict[str, Any]:
    result: dict[str, Any] = {"available": False, "artifacts": []}
    vulkaninfo = shutil.which("vulkaninfo")
    if vulkaninfo is None:
        result["reason"] = "vulkaninfo not found"
        return result

    result["available"] = True
    device_dir = staging / "device"
    device_dir.mkdir(parents=True, exist_ok=True)
    summary = _run((vulkaninfo, "--summary"), cwd=repo)
    summary_path = device_dir / "vulkaninfo-summary.txt"
    summary_path.write_text(summary.stdout + summary.stderr, encoding="utf-8")
    result["summary_exit_code"] = summary.returncode
    result["validation_layer"] = _validation_layer_record(summary.stdout)
    result["artifacts"].append(_artifact(summary_path, staging))

    with tempfile.TemporaryDirectory(prefix="oa-vulkaninfo-") as temp_name:
        temp = pathlib.Path(temp_name)
        profile = _run((vulkaninfo, "--json"), cwd=temp)
        result["profile_exit_code"] = profile.returncode
        generated = sorted(temp.glob("VP_VULKANINFO_*.json"))
        if generated:
            profile_path = device_dir / "vulkan-profile.json"
            shutil.copy2(generated[0], profile_path)
            result["artifacts"].append(_artifact(profile_path, staging))
        elif profile.stderr:
            error_path = device_dir / "vulkan-profile-error.txt"
            error_path.write_text(profile.stderr, encoding="utf-8")
            result["artifacts"].append(_artifact(error_path, staging))
    return result


def _copy_inputs(
    paths: Sequence[pathlib.Path],
    destination: pathlib.Path,
    staging: pathlib.Path,
    *,
    schema_prefix: str | None = None,
) -> list[dict[str, Any]]:
    artifacts: list[dict[str, Any]] = []
    names: set[str] = set()
    for source in paths:
        source = source.expanduser().resolve()
        if not source.is_file():
            raise FileNotFoundError(source)
        name = source.name
        if name in names:
            name = f"{source.stem}-{len(names)}{source.suffix}"
        names.add(name)
        target = destination / name
        destination.mkdir(parents=True, exist_ok=True)
        if schema_prefix is not None:
            data = json.loads(source.read_text(encoding="utf-8"))
            schema = data.get("schema") if isinstance(data, dict) else None
            if not isinstance(schema, str) or not schema.startswith(schema_prefix):
                raise ValueError(f"{source}: unexpected schema {schema!r}")
        shutil.copy2(source, target)
        artifacts.append(_artifact(target, staging))
    return artifacts


def _run_workload(
    command: Sequence[str],
    *,
    repo: pathlib.Path,
    staging: pathlib.Path,
    request_validation: bool,
    request_selection: bool,
    allowed_fallbacks: set[str],
) -> dict[str, Any]:
    logs = staging / "logs"
    graphs = staging / "graphs"
    logs.mkdir(parents=True, exist_ok=True)
    graphs.mkdir(parents=True, exist_ok=True)
    graph_path = graphs / "execution-graph.json"
    environment = os.environ.copy()
    environment["OA_GRAPH_REPORT"] = str(graph_path)
    if request_validation:
        environment["OA_VK_VALIDATION"] = "1"
    if request_selection:
        environment["OA_LOG_GEMM_ROUTER"] = "1"

    started = time.monotonic()
    result = _run(command, cwd=repo, env=environment, timeout=24 * 60 * 60)
    duration = time.monotonic() - started
    stdout_path = logs / "workload.stdout.txt"
    stderr_path = logs / "workload.stderr.txt"
    stdout_path.write_text(result.stdout, encoding="utf-8")
    stderr_path.write_text(result.stderr, encoding="utf-8")
    combined = ANSI_ESCAPE.sub("", result.stdout + "\n" + result.stderr)
    validation_observed = "Validation layers: ON" in combined
    validation_features: list[str] = []
    feature_match = re.search(r"Validation features:\s*([^\r\n]+)", combined)
    if feature_match:
        validation_features = [
            item.strip() for item in feature_match.group(1).split(",") if item.strip()
        ]
    validation_error_lines = [
        line
        for line in combined.splitlines()
        if "Validation Error" in line or "VUID-" in line
    ]
    selection_records = [match.groupdict() for match in GEMM_SELECTION.finditer(combined)]
    kernel_counts: dict[str, int] = {}
    fallback_counts: dict[str, int] = {}
    for record in selection_records:
        kernel = record["kernel"]
        fallback = record["fallback"]
        kernel_counts[kernel] = kernel_counts.get(kernel, 0) + 1
        if fallback != "none":
            fallback_counts[fallback] = fallback_counts.get(fallback, 0) + 1
    unexpected_fallbacks = sum(
        count
        for fallback, count in fallback_counts.items()
        if fallback not in allowed_fallbacks
    )
    selection_passed = (
        not request_selection
        or (bool(selection_records) and unexpected_fallbacks == 0)
    )

    graph_artifact: dict[str, Any] | None = None
    if graph_path.is_file():
        data = json.loads(graph_path.read_text(encoding="utf-8"))
        schema = data.get("schema") if isinstance(data, dict) else None
        if not isinstance(schema, str) or not schema.startswith("oa.execution_graph."):
            raise ValueError(f"workload produced unexpected graph schema {schema!r}")
        graph_artifact = _artifact(graph_path, staging)

    return {
        "command": _redact_command(command),
        "exit_code": result.returncode,
        "duration_seconds": round(duration, 6),
        "artifacts": [
            _artifact(stdout_path, staging),
            _artifact(stderr_path, staging),
        ],
        "graph": graph_artifact,
        "validation": {
            "requested": request_validation,
            "observed_enabled": validation_observed,
            "observed_features": validation_features,
            "reported_error_count": len(validation_error_lines),
        },
        "selection": {
            "requested": request_selection,
            "record_count": len(selection_records),
            "kernel_counts": dict(sorted(kernel_counts.items())),
            "fallback_counts": dict(sorted(fallback_counts.items())),
            "allowed_fallbacks": sorted(allowed_fallbacks),
            "unexpected_fallback_count": unexpected_fallbacks,
            "gate_passed": selection_passed,
        },
    }


def collect(args: argparse.Namespace) -> tuple[pathlib.Path, int]:
    repo = args.repo.expanduser().resolve()
    if not (repo / ".git").exists():
        raise ValueError(f"not an OA Git checkout: {repo}")

    commit = _git(repo, "rev-parse", "HEAD") or "unknown"
    timestamp = dt.datetime.now(dt.timezone.utc).replace(microsecond=0)
    output = args.output
    if output is None:
        stamp = timestamp.strftime("%Y%m%dT%H%M%SZ")
        output = repo / "var/report" / f"evidence-{stamp}-{commit[:8]}"
    output = output.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"output already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)

    workload_exit = 0
    with tempfile.TemporaryDirectory(prefix=f".{output.name}-", dir=output.parent) as temp_name:
        staging = pathlib.Path(temp_name)
        workload: dict[str, Any] | None = None
        command = list(args.command)
        if command and command[0] == "--":
            command = command[1:]
        if command:
            workload = _run_workload(
                command,
                repo=repo,
                staging=staging,
                request_validation=args.validation,
                request_selection=args.selection_trace,
                allowed_fallbacks=set(args.allow_fallback),
            )
            workload_exit = int(workload["exit_code"])
            if not workload["selection"]["gate_passed"] and workload_exit == 0:
                workload_exit = 1

        graph_artifacts = _copy_inputs(
            args.graph,
            staging / "graphs",
            staging,
            schema_prefix="oa.execution_graph.",
        )
        validation_artifacts = _copy_inputs(
            args.validation_log, staging / "validation", staging
        )
        benchmark_artifacts = _copy_inputs(
            args.benchmark,
            staging / "benchmarks",
            staging,
            schema_prefix="oa.benchmark.",
        )

        registry = _find_registry(repo)
        registry_record: dict[str, Any] = {"available": registry is not None}
        if registry is not None:
            registry_record.update(
                {
                    "source_path": str(registry),
                    "bytes": registry.stat().st_size,
                    "sha256": _sha256(registry),
                }
            )

        version_path = repo / "VERSION"
        dirty = _git(repo, "status", "--porcelain") or ""
        cache_path = args.cmake_cache.expanduser()
        if not cache_path.is_absolute():
            cache_path = repo / cache_path
        manifest: dict[str, Any] = {
            "schema": SCHEMA,
            "created_utc": timestamp.isoformat().replace("+00:00", "Z"),
            "repository": {
                "commit": commit,
                "branch": _git(repo, "branch", "--show-current"),
                "describe": _git(repo, "describe", "--always", "--dirty", "--tags"),
                "dirty": bool(dirty),
                "version": version_path.read_text(encoding="utf-8").strip()
                if version_path.is_file()
                else None,
            },
            "host": {
                "system": platform.system(),
                "release": platform.release(),
                "machine": platform.machine(),
                "python": platform.python_version(),
            },
            "environment": {
                key: os.environ[key] for key in RELEVANT_ENV if key in os.environ
            },
            "build": _read_cache(cache_path.resolve()),
            "vulkan_registry": registry_record,
            "vulkan_device": _snapshot_vulkan(staging, repo),
            "workload": workload,
            "graphs": graph_artifacts,
            "validation_logs": validation_artifacts,
            "benchmarks": benchmark_artifacts,
            "capture_references": [str(path) for path in args.capture_reference],
        }
        manifest_path = staging / "manifest.json"
        manifest_path.write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        staging.rename(output)

    return output, workload_exit


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo", type=pathlib.Path, default=pathlib.Path.cwd(), help="OA checkout"
    )
    parser.add_argument("--output", type=pathlib.Path, help="new bundle directory")
    parser.add_argument(
        "--cmake-cache",
        type=pathlib.Path,
        default=pathlib.Path("Build/Release/CMakeCache.txt"),
    )
    parser.add_argument("--graph", type=pathlib.Path, action="append", default=[])
    parser.add_argument(
        "--validation-log", type=pathlib.Path, action="append", default=[]
    )
    parser.add_argument("--benchmark", type=pathlib.Path, action="append", default=[])
    parser.add_argument(
        "--capture-reference", type=pathlib.Path, action="append", default=[]
    )
    parser.add_argument(
        "--validation",
        action="store_true",
        help="request OA Vulkan validation for a wrapped workload",
    )
    parser.add_argument(
        "--selection-trace",
        action="store_true",
        help="collect GEMM selection records and reject unexpected fallbacks",
    )
    parser.add_argument(
        "--allow-fallback",
        action="append",
        default=[],
        help="allowed fallback category (precision, layout, or naive)",
    )
    parser.add_argument(
        "command", nargs=argparse.REMAINDER, help="workload command after --"
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    try:
        args = parse_args(sys.argv[1:] if argv is None else argv)
        output, workload_exit = collect(args)
        print(output)
        return workload_exit
    except (FileNotFoundError, FileExistsError, ValueError, json.JSONDecodeError) as error:
        print(f"oaevidence: {error}", file=sys.stderr)
        return 2
    except subprocess.TimeoutExpired as error:
        print(f"oaevidence: command timed out: {error.cmd}", file=sys.stderr)
        return 124


if __name__ == "__main__":
    raise SystemExit(main())
