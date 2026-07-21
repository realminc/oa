#!/usr/bin/env python3
"""Accept a clean OA benchmark result as an immutable compact baseline."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import pathlib
import tempfile
from typing import Any, Sequence

import oabench


def _sha256(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _validate(result: dict[str, Any]) -> None:
    if result.get("schema") != oabench.SCHEMA:
        raise ValueError("result is not oa.benchmark.v1")
    if result.get("result") != "PASS":
        raise ValueError("only a passing result can become a baseline")
    repository = result.get("repository", {})
    if repository.get("dirty"):
        raise ValueError("result was captured from a dirty repository")
    if not repository.get("commit") or repository.get("commit") == "unknown":
        raise ValueError("result has no repository commit")
    platform = result.get("platform", {})
    if not platform.get("available"):
        raise ValueError("result has no usable Vulkan platform identity")
    workload = result.get("workload", {})
    if not workload.get("name") or not workload.get("command_id"):
        raise ValueError("result has no stable workload/command identity")
    statistics = result.get("metric", {}).get("statistics") or {}
    if int(statistics.get("count", 0)) < 7:
        raise ValueError("baseline requires at least seven measured samples")


def accept(
    result_path: pathlib.Path,
    output: pathlib.Path,
    *,
    reason: str,
    accepted_by: str,
) -> pathlib.Path:
    result_path = result_path.expanduser().resolve()
    output = output.expanduser().resolve()
    if output.exists():
        raise FileExistsError(output)
    if not reason.strip():
        raise ValueError("acceptance reason must not be empty")
    result = json.loads(result_path.read_text(encoding="utf-8"))
    _validate(result)
    baseline = {
        "schema": oabench.BASELINE_SCHEMA,
        "accepted_utc": dt.datetime.now(dt.timezone.utc)
        .isoformat()
        .replace("+00:00", "Z"),
        "acceptance": {
            "accepted_by": accepted_by,
            "reason": reason.strip(),
        },
        "source": {
            "repository_commit": result["repository"]["commit"],
            "result_sha256": _sha256(result_path),
        },
        "platform": result["platform"],
        "environment": {
            "driver": result.get("driver", {}),
            "power": result.get("host", {}).get("power", {}),
        },
        "build": result.get("build", {}),
        "workload": result["workload"],
        "metric": result["metric"],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=output.parent, delete=False
    ) as stream:
        json.dump(baseline, stream, indent=2, sort_keys=True)
        stream.write("\n")
        temporary = pathlib.Path(stream.name)
    temporary.replace(output)
    return output


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result", type=pathlib.Path, required=True)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    parser.add_argument("--reason", required=True)
    parser.add_argument("--accepted-by", default=os.environ.get("USER", "unknown"))
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    import sys

    try:
        args = parse_args(sys.argv[1:] if argv is None else argv)
        print(
            accept(
                args.result,
                args.output,
                reason=args.reason,
                accepted_by=args.accepted_by,
            )
        )
        return 0
    except (FileExistsError, OSError, ValueError, json.JSONDecodeError) as error:
        print(f"oabaseline: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
