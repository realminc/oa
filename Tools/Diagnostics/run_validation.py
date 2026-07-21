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

import oaevidence


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


def _version_tuple(version: str) -> tuple[int, int, int]:
    parts = version.split(".")
    if len(parts) != 3 or any(not part.isdigit() for part in parts):
        raise RuntimeError(f"unrecognized validation-layer API version: {version!r}")
    return int(parts[0]), int(parts[1]), int(parts[2])


def _known_vvl_shared_rmw_false_positive(
    bundle: pathlib.Path, layer: dict[str, object]
) -> bool:
    version = layer.get("api_version")
    if not isinstance(version, str) or _version_tuple(version) >= _VVL_SHARED_RMW_FIX_API:
        return False
    stdout = bundle / "logs/workload.stdout.txt"
    if not stdout.is_file():
        return False
    return _VVL_SHARED_RMW_PATTERN.search(
        stdout.read_text(encoding="utf-8", errors="replace")
    ) is not None


def _vulkan_preflight(repo: pathlib.Path) -> dict[str, object]:
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
    layer = oaevidence._validation_layer_record(result.stdout)
    if layer is None:
        raise RuntimeError("cannot determine VK_LAYER_KHRONOS_validation version")
    return layer


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo", type=pathlib.Path, default=pathlib.Path(__file__).parents[2]
    )
    parser.add_argument("--output", type=pathlib.Path)
    parser.add_argument(
        "--mode", choices=tuple(EXPECTED_FEATURES), default="sync"
    )
    parser.add_argument(
        "--cmake-cache",
        type=pathlib.Path,
        default=pathlib.Path("Build/Release/CMakeCache.txt"),
    )
    parser.add_argument("command", nargs=argparse.REMAINDER)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    repo = args.repo.expanduser().resolve()
    command = list(args.command)
    if command and command[0] == "--":
        command = command[1:]
    if not command:
        command = [
            str(repo / "Bin/Release/Test/Runtime/Graph/TestComputeGraph")
        ]
    executable = pathlib.Path(command[0]).expanduser()
    if not executable.is_absolute() and executable.parent == pathlib.Path("."):
        discovered = shutil.which(command[0])
        executable = pathlib.Path(discovered) if discovered else repo / executable
    elif not executable.is_absolute():
        executable = repo / executable
    if not executable.is_file():
        print(f"run_validation: workload not found: {executable}", file=sys.stderr)
        return 2
    command[0] = str(executable.resolve())

    try:
        validation_layer = _vulkan_preflight(repo)
    except RuntimeError as error:
        print(f"run_validation: {error}", file=sys.stderr)
        return 2

    output = args.output
    if output is None:
        stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        output = repo / "var/report" / f"validation-{args.mode}-{stamp}"

    previous_mode = os.environ.get("OA_VK_VALIDATION_MODE")
    os.environ["OA_VK_VALIDATION_MODE"] = args.mode
    try:
        evidence_args = oaevidence.parse_args(
            (
                "--repo",
                str(repo),
                "--output",
                str(output),
                "--cmake-cache",
                str(args.cmake_cache),
                "--validation",
                "--",
                *command,
            )
        )
        bundle, workload_exit = oaevidence.collect(evidence_args)
    except (
        FileNotFoundError,
        FileExistsError,
        ValueError,
        RuntimeError,
        subprocess.TimeoutExpired,
    ) as error:
        print(f"run_validation: {error}", file=sys.stderr)
        return 2
    finally:
        if previous_mode is None:
            os.environ.pop("OA_VK_VALIDATION_MODE", None)
        else:
            os.environ["OA_VK_VALIDATION_MODE"] = previous_mode

    manifest = json.loads((bundle / "manifest.json").read_text(encoding="utf-8"))
    validation = manifest["workload"]["validation"]
    observed = set(validation["observed_features"])
    expected = EXPECTED_FEATURES[args.mode]
    passed = (
        workload_exit == 0
        and validation["observed_enabled"]
        and validation["reported_error_count"] == 0
        and expected <= observed
    )
    print(bundle)
    print(
        "validation "
        f"mode={args.mode} observed={','.join(sorted(observed)) or 'none'} "
        f"errors={validation['reported_error_count']} workload_exit={workload_exit} "
        f"result={'PASS' if passed else 'FAIL'}"
    )
    if not passed and _known_vvl_shared_rmw_false_positive(
        bundle, validation_layer
    ):
        version = validation_layer["api_version"]
        print(
            "run_validation: the reported same-invocation shared-memory RMW "
            f"race matches known VVL false positive #12415 in layer {version}; "
            f"rerun with a layer containing commit {_VVL_SHARED_RMW_FIX_COMMIT} "
            f"(API >= {'.'.join(map(str, _VVL_SHARED_RMW_FIX_API))}) — "
            f"{_VVL_SHARED_RMW_ISSUE}",
            file=sys.stderr,
        )
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
