#!/usr/bin/env python3
"""Run an OA workload under a named Vulkan validation profile."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import pathlib
import re
import shutil
import subprocess
import sys
from typing import Sequence

import oaEvidence


EXPECTED_FEATURES = {
    "core": {"core"},
    "sync": {"core", "synchronization"},
    "gpu": {"core", "gpu-assisted"},
    "all": {"core", "synchronization", "gpu-assisted"},
}

_VVL_SHARED_RMW_FIX_API = (1, 4, 354)
_VVL_SHARED_RMW_FIX_COMMIT = "4cd431278be1e3dd074af9989e956306e4f1a2a6"
_VVL_SHARED_RMW_ISSUE = (
    "https://github.com/KhronosGroup/Vulkan-ValidationLayers/issues/12415"
)
_VVL_SHARED_RMW_PATTERN = re.compile(
    r"SharedMemoryDataRace-RaceOnStore.*?"
    r"local invocation index (?P<index>\d+).*?"
    r"\(Likely against local invocation index (?P=index)\).*?"
    r"SPIR-V Instruction: %\d+ = OpLoad.*?"
    r"SPIR-V Instruction: OpStore",
    re.DOTALL,
)


def _versionTuple(version: str) -> tuple[int, int, int]:
    parts = version.split(".")
    if len(parts) != 3 or any(not part.isdigit() for part in parts):
        raise RuntimeError(f"unrecognized validation-layer API version: {version!r}")
    return int(parts[0]), int(parts[1]), int(parts[2])


def _knownVvlSharedRmwFalsePositive(
    bundle: pathlib.Path, layer: dict[str, object]
) -> bool:
    version = layer.get("api_version")
    if not isinstance(version, str) or _versionTuple(version) >= _VVL_SHARED_RMW_FIX_API:
        return False
    stdout = bundle / "logs/workload.stdout.txt"
    if not stdout.is_file():
        return False
    return _VVL_SHARED_RMW_PATTERN.search(
        stdout.read_text(encoding="utf-8", errors="replace")
    ) is not None


def _vulkanPreflight(repo: pathlib.Path) -> dict[str, object]:
    provenance = oaEvidence._explicitValidationLayerProvenance()
    vulkaninfo = shutil.which("vulkaninfo")
    if vulkaninfo is None:
        raise RuntimeError("vulkaninfo not found")
    result = subprocess.run(
        (vulkaninfo, "--summary"),
        cwd=repo,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError("vulkaninfo --summary failed")
    if "VK_LAYER_KHRONOS_validation" not in result.stdout:
        raise RuntimeError("VK_LAYER_KHRONOS_validation is not installed")
    layer = oaEvidence._validationLayerRecord(result.stdout)
    if layer is None:
        raise RuntimeError("cannot determine VK_LAYER_KHRONOS_validation version")
    if provenance is not None:
        layer.update(provenance)
    return layer


def parseArgs(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo", type=pathlib.Path, default=pathlib.Path(__file__).parents[2]
    )
    parser.add_argument("--output", type=pathlib.Path)
    parser.add_argument(
        "--mode", choices=tuple(EXPECTED_FEATURES), default="sync"
    )
    parser.add_argument(
        "--cmake-cache", dest="cmakeCache",
        type=pathlib.Path,
        default=pathlib.Path("build/release/CMakeCache.txt"),
    )
    parser.add_argument("command", nargs=argparse.REMAINDER)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parseArgs(sys.argv[1:] if argv is None else argv)
    repo = args.repo.expanduser().resolve()
    command = list(args.command)
    if command and command[0] == "--":
        command = command[1:]
    if not command:
        command = [
            str(repo / "bin/release/test/runtime/graph/testExecutableGraph")
        ]
    executable = pathlib.Path(command[0]).expanduser()
    if not executable.is_absolute() and executable.parent == pathlib.Path("."):
        discovered = shutil.which(command[0])
        executable = pathlib.Path(discovered) if discovered else repo / executable
    elif not executable.is_absolute():
        executable = repo / executable
    if not executable.is_file():
        print(f"runValidation: workload not found: {executable}", file=sys.stderr)
        return 2
    command[0] = str(executable.resolve())

    try:
        validationLayer = _vulkanPreflight(repo)
    except RuntimeError as error:
        print(f"runValidation: {error}", file=sys.stderr)
        return 2

    output = args.output
    if output is None:
        stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        output = repo / "var/report" / f"validation-{args.mode}-{stamp}"

    previousMode = os.environ.get("OA_VK_VALIDATION_MODE")
    os.environ["OA_VK_VALIDATION_MODE"] = args.mode
    try:
        evidenceArgs = oaEvidence.parseArgs(
            (
                "--repo",
                str(repo),
                "--output",
                str(output),
                "--cmake-cache",
                str(args.cmakeCache),
                "--validation",
                "--",
                *command,
            )
        )
        bundle, workloadExit = oaEvidence.collect(evidenceArgs)
    except (
        FileNotFoundError,
        FileExistsError,
        ValueError,
        RuntimeError,
        subprocess.TimeoutExpired,
    ) as error:
        print(f"runValidation: {error}", file=sys.stderr)
        return 2
    finally:
        if previousMode is None:
            os.environ.pop("OA_VK_VALIDATION_MODE", None)
        else:
            os.environ["OA_VK_VALIDATION_MODE"] = previousMode

    manifest = json.loads((bundle / "manifest.json").read_text(encoding="utf-8"))
    validation = manifest["workload"]["validation"]
    observed = set(validation["observed_features"])
    expected = EXPECTED_FEATURES[args.mode]
    passed = (
        workloadExit == 0
        and validation["observed_enabled"]
        and validation["reported_error_count"] == 0
        and expected <= observed
    )
    print(bundle)
    print(
        "validation "
        f"mode={args.mode} observed={','.join(sorted(observed)) or 'none'} "
        f"errors={validation['reported_error_count']} workload_exit={workloadExit} "
        f"result={'PASS' if passed else 'FAIL'}"
    )
    if not passed and _knownVvlSharedRmwFalsePositive(
        bundle, validationLayer
    ):
        version = validationLayer["api_version"]
        print(
            "runValidation: the reported same-invocation shared-memory RMW "
            f"race matches known VVL false positive #12415 in layer {version}; "
            f"rerun with a layer containing commit {_VVL_SHARED_RMW_FIX_COMMIT} "
            f"(API >= {'.'.join(map(str, _VVL_SHARED_RMW_FIX_API))}) — "
            f"{_VVL_SHARED_RMW_ISSUE}",
            file=sys.stderr,
        )
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
