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
    "LD_LIBRARY_PATH",
)

VALIDATION_LAYER = re.compile(
    r"^VK_LAYER_KHRONOS_validation\s+.*?\s+"
    r"(?P<api>\d+\.\d+\.\d+)\s+version\s+(?P<implementation>\d+)\s*$",
    re.MULTILINE,
)


def _validationLayerRecord(summary: str) -> dict[str, Any] | None:
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


def _externalFileRecord(path: pathlib.Path) -> dict[str, Any]:
    path = path.expanduser().resolve()
    return {
        "path": str(path),
        "bytes": path.stat().st_size,
        "sha256": _sha256(path),
    }


def _explicitValidationLayerProvenance() -> dict[str, Any] | None:
    """Resolve an explicitly selected VVL manifest and its exact library.

    A layer manifest may contain only a library soname.  In that case the
    manifest API version says nothing about which library the dynamic loader
    will select, so an explicit LD_LIBRARY_PATH is required for reproducible
    evidence.
    """
    searchRoots: list[pathlib.Path] = []
    for variable in ("VK_LAYER_PATH", "VK_ADD_LAYER_PATH"):
        for rawPath in os.environ.get(variable, "").split(os.pathsep):
            if rawPath:
                searchRoots.append(pathlib.Path(rawPath).expanduser().resolve())
    if not searchRoots:
        return None

    matches: list[tuple[pathlib.Path, dict[str, Any]]] = []
    seen: set[pathlib.Path] = set()
    for root in searchRoots:
        if not root.is_dir():
            continue
        for manifest in sorted(root.glob("*.json")):
            manifest = manifest.resolve()
            if manifest in seen:
                continue
            seen.add(manifest)
            try:
                document = json.loads(manifest.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            entries: list[Any] = []
            if isinstance(document, dict):
                entries.append(document.get("layer"))
                layers = document.get("layers")
                if isinstance(layers, list):
                    entries.extend(layers)
            for entry in entries:
                if (
                    isinstance(entry, dict)
                    and entry.get("name") == "VK_LAYER_KHRONOS_validation"
                ):
                    matches.append((manifest, entry))

    if not matches:
        return None
    if len(matches) != 1:
        paths = ", ".join(str(path) for path, _ in matches)
        raise RuntimeError(
            "explicit Vulkan layer paths contain multiple "
            f"VK_LAYER_KHRONOS_validation manifests: {paths}"
        )

    manifest, entry = matches[0]
    libraryName = entry.get("library_path")
    if not isinstance(libraryName, str) or not libraryName:
        raise RuntimeError(f"{manifest}: validation layer has no library_path")

    declared = pathlib.Path(libraryName)
    if declared.is_absolute():
        library = declared
    elif "/" in libraryName:
        library = manifest.parent / declared
    else:
        library = next(
            (
                pathlib.Path(rawPath).expanduser() / declared
                for rawPath in os.environ.get("LD_LIBRARY_PATH", "").split(
                    os.pathsep
                )
                if rawPath
                and (pathlib.Path(rawPath).expanduser() / declared).is_file()
            ),
            None,
        )
        if library is None:
            raise RuntimeError(
                f"{manifest}: library_path {libraryName!r} is a bare soname; "
                "set LD_LIBRARY_PATH to the directory containing the matching "
                "validation-layer library"
            )
    library = library.resolve()
    if not library.is_file():
        raise RuntimeError(
            f"{manifest}: validation-layer library does not exist: {library}"
        )

    return {
        "explicit_manifest": _externalFileRecord(manifest),
        "explicit_library": _externalFileRecord(library),
    }


def _artifact(path: pathlib.Path, root: pathlib.Path) -> dict[str, Any]:
    return {
        "path": path.relative_to(root).as_posix(),
        "bytes": path.stat().st_size,
        "sha256": _sha256(path),
    }


def _readCache(path: pathlib.Path) -> dict[str, str]:
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
        keyAndType, value = line.split("=", 1)
        key = keyAndType.split(":", 1)[0]
        if key in wanted:
            values[key] = value
    return values


def _registryCandidates(repo: pathlib.Path) -> Iterable[pathlib.Path]:
    sdk = os.environ.get("VULKAN_SDK")
    if sdk:
        yield pathlib.Path(sdk) / "share/vulkan/registry/vk.xml"
        yield pathlib.Path(sdk) / "registry/vk.xml"
    yield repo / "build/release/vcpkg_installed/x64-linux/share/vulkan/registry/vk.xml"
    installed = os.environ.get("VCPKG_INSTALLED_DIR")
    if installed:
        yield pathlib.Path(installed) / "x64-linux/share/vulkan/registry/vk.xml"
    yield (
        pathlib.Path.home()
        / ".vcpkg/oa/release/installed/x64-linux/share/vulkan/registry/vk.xml"
    )
    yield pathlib.Path.home() / ".vcpkg/installed/x64-linux/share/vulkan/registry/vk.xml"
    yield pathlib.Path("/usr/share/vulkan/registry/vk.xml")


def _findRegistry(repo: pathlib.Path) -> pathlib.Path | None:
    override = os.environ.get("OA_VK_XML")
    if override:
        candidate = pathlib.Path(override).expanduser()
        return candidate.resolve() if candidate.is_file() else None
    for candidate in _registryCandidates(repo):
        if candidate.is_file():
            return candidate.resolve()
    return None


def _redactCommand(command: Sequence[str]) -> list[str]:
    redacted: list[str] = []
    hideNext = False
    sensitive = ("token", "secret", "password", "api-key", "apikey")
    for argument in command:
        if hideNext:
            redacted.append("[REDACTED]")
            hideNext = False
            continue
        lowered = argument.lower()
        if argument.startswith("-") and any(word in lowered for word in sensitive):
            if "=" in argument:
                redacted.append(argument.split("=", 1)[0] + "=[REDACTED]")
            else:
                redacted.append(argument)
                hideNext = True
            continue
        redacted.append(argument)
    return redacted


def _snapshotVulkan(staging: pathlib.Path, repo: pathlib.Path) -> dict[str, Any]:
    result: dict[str, Any] = {"available": False, "artifacts": []}
    vulkaninfo = shutil.which("vulkaninfo")
    if vulkaninfo is None:
        result["reason"] = "vulkaninfo not found"
        return result

    result["available"] = True
    deviceDir = staging / "device"
    deviceDir.mkdir(parents=True, exist_ok=True)
    summary = _run((vulkaninfo, "--summary"), cwd=repo)
    summaryPath = deviceDir / "vulkaninfo-summary.txt"
    summaryPath.write_text(summary.stdout + summary.stderr, encoding="utf-8")
    result["summary_exit_code"] = summary.returncode
    validationLayer = _validationLayerRecord(summary.stdout)
    if validationLayer is not None:
        try:
            provenance = _explicitValidationLayerProvenance()
        except RuntimeError as error:
            validationLayer["explicit_provenance_error"] = str(error)
        else:
            if provenance is not None:
                validationLayer.update(provenance)
    result["validation_layer"] = validationLayer
    result["artifacts"].append(_artifact(summaryPath, staging))

    with tempfile.TemporaryDirectory(prefix="oa-vulkaninfo-") as tempName:
        temp = pathlib.Path(tempName)
        profile = _run((vulkaninfo, "--json"), cwd=temp)
        result["profile_exit_code"] = profile.returncode
        generated = sorted(temp.glob("VP_VULKANINFO_*.json"))
        if generated:
            profilePath = deviceDir / "vulkan-profile.json"
            shutil.copy2(generated[0], profilePath)
            result["artifacts"].append(_artifact(profilePath, staging))
        elif profile.stderr:
            errorPath = deviceDir / "vulkan-profile-error.txt"
            errorPath.write_text(profile.stderr, encoding="utf-8")
            result["artifacts"].append(_artifact(errorPath, staging))
    return result


def _copyInputs(
    paths: Sequence[pathlib.Path],
    destination: pathlib.Path,
    staging: pathlib.Path,
    *,
    schemaPrefix: str | None = None,
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
        if schemaPrefix is not None:
            data = json.loads(source.read_text(encoding="utf-8"))
            schema = data.get("schema") if isinstance(data, dict) else None
            if not isinstance(schema, str) or not schema.startswith(schemaPrefix):
                raise ValueError(f"{source}: unexpected schema {schema!r}")
        shutil.copy2(source, target)
        artifacts.append(_artifact(target, staging))
    return artifacts


def _runWorkload(
    command: Sequence[str],
    *,
    repo: pathlib.Path,
    staging: pathlib.Path,
    requestValidation: bool,
    requestSelection: bool,
    allowedFallbacks: set[str],
) -> dict[str, Any]:
    logs = staging / "logs"
    graphs = staging / "graphs"
    logs.mkdir(parents=True, exist_ok=True)
    graphs.mkdir(parents=True, exist_ok=True)
    graphPath = graphs / "execution-graph.json"
    environment = os.environ.copy()
    environment["OA_GRAPH_REPORT"] = str(graphPath)
    if requestValidation:
        environment["OA_VK_VALIDATION"] = "1"
    if requestSelection:
        environment["OA_LOG_GEMM_ROUTER"] = "1"

    started = time.monotonic()
    result = _run(command, cwd=repo, env=environment, timeout=24 * 60 * 60)
    duration = time.monotonic() - started
    stdoutPath = logs / "workload.stdout.txt"
    stderrPath = logs / "workload.stderr.txt"
    stdoutPath.write_text(result.stdout, encoding="utf-8")
    stderrPath.write_text(result.stderr, encoding="utf-8")
    combined = ANSI_ESCAPE.sub("", result.stdout + "\n" + result.stderr)
    validationObserved = "Validation layers: ON" in combined
    validationFeatures: list[str] = []
    featureMatch = re.search(r"Validation features:\s*([^\r\n]+)", combined)
    if featureMatch:
        validationFeatures = [
            item.strip() for item in featureMatch.group(1).split(",") if item.strip()
        ]
    validationErrorLines = [
        line
        for line in combined.splitlines()
        if "Validation Error" in line or "VUID-" in line
    ]
    selectionRecords = [match.groupdict() for match in GEMM_SELECTION.finditer(combined)]
    kernelCounts: dict[str, int] = {}
    fallbackCounts: dict[str, int] = {}
    for record in selectionRecords:
        kernel = record["kernel"]
        fallback = record["fallback"]
        kernelCounts[kernel] = kernelCounts.get(kernel, 0) + 1
        if fallback != "none":
            fallbackCounts[fallback] = fallbackCounts.get(fallback, 0) + 1
    unexpectedFallbacks = sum(
        count
        for fallback, count in fallbackCounts.items()
        if fallback not in allowedFallbacks
    )
    selectionPassed = (
        not requestSelection
        or (bool(selectionRecords) and unexpectedFallbacks == 0)
    )

    graphArtifact: dict[str, Any] | None = None
    if graphPath.is_file():
        data = json.loads(graphPath.read_text(encoding="utf-8"))
        schema = data.get("schema") if isinstance(data, dict) else None
        if not isinstance(schema, str) or not schema.startswith("oa.execution_graph."):
            raise ValueError(f"workload produced unexpected graph schema {schema!r}")
        graphArtifact = _artifact(graphPath, staging)

    return {
        "command": _redactCommand(command),
        "exit_code": result.returncode,
        "duration_seconds": round(duration, 6),
        "artifacts": [
            _artifact(stdoutPath, staging),
            _artifact(stderrPath, staging),
        ],
        "graph": graphArtifact,
        "validation": {
            "requested": requestValidation,
            "observed_enabled": validationObserved,
            "observed_features": validationFeatures,
            "reported_error_count": len(validationErrorLines),
        },
        "selection": {
            "requested": requestSelection,
            "record_count": len(selectionRecords),
            "kernel_counts": dict(sorted(kernelCounts.items())),
            "fallback_counts": dict(sorted(fallbackCounts.items())),
            "allowed_fallbacks": sorted(allowedFallbacks),
            "unexpected_fallback_count": unexpectedFallbacks,
            "gate_passed": selectionPassed,
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

    workloadExit = 0
    with tempfile.TemporaryDirectory(prefix=f".{output.name}-", dir=output.parent) as tempName:
        staging = pathlib.Path(tempName)
        workload: dict[str, Any] | None = None
        command = list(args.command)
        if command and command[0] == "--":
            command = command[1:]
        if command:
            workload = _runWorkload(
                command,
                repo=repo,
                staging=staging,
                requestValidation=args.validation,
                requestSelection=args.selectionTrace,
                allowedFallbacks=set(args.allowFallback),
            )
            workloadExit = int(workload["exit_code"])
            if not workload["selection"]["gate_passed"] and workloadExit == 0:
                workloadExit = 1

        graphArtifacts = _copyInputs(
            args.graph,
            staging / "graphs",
            staging,
            schemaPrefix="oa.execution_graph.",
        )
        validationArtifacts = _copyInputs(
            args.validationLog, staging / "validation", staging
        )
        benchmarkArtifacts = _copyInputs(
            args.benchmark,
            staging / "benchmarks",
            staging,
            schemaPrefix="oa.benchmark.",
        )

        registry = _findRegistry(repo)
        registryRecord: dict[str, Any] = {"available": registry is not None}
        if registry is not None:
            registryRecord.update(
                {
                    "source_path": str(registry),
                    "bytes": registry.stat().st_size,
                    "sha256": _sha256(registry),
                }
            )

        versionPath = repo / "VERSION"
        dirty = _git(repo, "status", "--porcelain") or ""
        cachePath = args.cmakeCache.expanduser()
        if not cachePath.is_absolute():
            cachePath = repo / cachePath
        manifest: dict[str, Any] = {
            "schema": SCHEMA,
            "created_utc": timestamp.isoformat().replace("+00:00", "Z"),
            "repository": {
                "commit": commit,
                "branch": _git(repo, "branch", "--show-current"),
                "describe": _git(repo, "describe", "--always", "--dirty", "--tags"),
                "dirty": bool(dirty),
                "version": versionPath.read_text(encoding="utf-8").strip()
                if versionPath.is_file()
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
            "build": _readCache(cachePath.resolve()),
            "vulkan_registry": registryRecord,
            "vulkan_device": _snapshotVulkan(staging, repo),
            "workload": workload,
            "graphs": graphArtifacts,
            "validation_logs": validationArtifacts,
            "benchmarks": benchmarkArtifacts,
            "capture_references": [str(path) for path in args.captureReference],
        }
        manifestPath = staging / "manifest.json"
        manifestPath.write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        staging.rename(output)

    return output, workloadExit


def parseArgs(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo", type=pathlib.Path, default=pathlib.Path.cwd(), help="OA checkout"
    )
    parser.add_argument("--output", type=pathlib.Path, help="new bundle directory")
    parser.add_argument(
        "--cmake-cache", dest="cmakeCache",
        type=pathlib.Path,
        default=pathlib.Path("build/release/CMakeCache.txt"),
    )
    parser.add_argument("--graph", type=pathlib.Path, action="append", default=[])
    parser.add_argument(
        "--validation-log", dest="validationLog", type=pathlib.Path, action="append", default=[]
    )
    parser.add_argument("--benchmark", type=pathlib.Path, action="append", default=[])
    parser.add_argument(
        "--capture-reference", dest="captureReference", type=pathlib.Path, action="append", default=[]
    )
    parser.add_argument(
        "--validation",
        action="store_true",
        help="request OA Vulkan validation for a wrapped workload",
    )
    parser.add_argument(
        "--selection-trace", dest="selectionTrace",
        action="store_true",
        help="collect GEMM selection records and reject unexpected fallbacks",
    )
    parser.add_argument(
        "--allow-fallback", dest="allowFallback",
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
        args = parseArgs(sys.argv[1:] if argv is None else argv)
        output, workloadExit = collect(args)
        print(output)
        return workloadExit
    except (FileNotFoundError, FileExistsError, ValueError, json.JSONDecodeError) as error:
        print(f"oaEvidence: {error}", file=sys.stderr)
        return 2
    except subprocess.TimeoutExpired as error:
        print(f"oaEvidence: command timed out: {error.cmd}", file=sys.stderr)
        return 124


if __name__ == "__main__":
    raise SystemExit(main())
