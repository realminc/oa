#!/usr/bin/env python3
"""Compare selected output traces across fresh OA workload processes."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import pathlib
import platform
import re
import subprocess
import sys
import tempfile
import time
from typing import Any, Sequence

import oaevidence


SCHEMA = "oa.determinism.v1"
ANSI_ESCAPE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")


def _sha256(path: pathlib.Path) -> str:
	result = hashlib.sha256()
	with path.open("rb") as stream:
		for block in iter(lambda: stream.read(1024 * 1024), b""):
			result.update(block)
	return result.hexdigest()


def _extract_trace(pattern: re.Pattern[str], text: str) -> list[str]:
	values: list[str] = []
	for match in pattern.finditer(text):
		value = match.groupdict().get("value")
		if value is not None:
			values.append(value)
		elif match.lastindex is None:
			values.append(match.group(0))
		elif match.lastindex == 1:
			values.append(match.group(1))
		else:
			raise ValueError(
				"trace regex must contain no captures, one capture, or a named "
				"'value' capture"
			)
	return values


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
	parser = argparse.ArgumentParser(description=__doc__)
	parser.add_argument("--repo", type=pathlib.Path, default=pathlib.Path.cwd())
	parser.add_argument("--output", type=pathlib.Path, required=True)
	parser.add_argument("--name", required=True, help="stable workload name")
	parser.add_argument("--runs", type=int, default=2)
	parser.add_argument("--timeout", type=float, default=3600.0)
	parser.add_argument(
		"--trace-regex",
		action="append",
		required=True,
		help="trace to compare; repeat for independent ordered trace groups",
	)
	parser.add_argument("--require-regex", action="append", default=[])
	parser.add_argument(
		"--cmake-cache",
		type=pathlib.Path,
		default=pathlib.Path("Build/Release/CMakeCache.txt"),
	)
	parser.add_argument("command", nargs=argparse.REMAINDER)
	return parser.parse_args(argv)


def run(args: argparse.Namespace) -> tuple[pathlib.Path, int]:
	if args.runs < 2:
		raise ValueError("determinism requires at least two fresh-process runs")
	if args.timeout <= 0.0:
		raise ValueError("timeout must be positive")
	repo = args.repo.expanduser().resolve()
	output = args.output.expanduser().resolve()
	logs = output.with_suffix(output.suffix + ".logs")
	if output.exists():
		raise FileExistsError(output)
	if logs.exists():
		raise FileExistsError(logs)
	command = list(args.command)
	if command and command[0] == "--":
		command = command[1:]
	if not command:
		raise ValueError("workload command is required after --")

	trace_patterns = [re.compile(pattern) for pattern in args.trace_regex]
	required_patterns = [re.compile(pattern) for pattern in args.require_regex]
	output.parent.mkdir(parents=True, exist_ok=True)
	logs.mkdir(parents=True)

	samples: list[dict[str, Any]] = []
	reference: list[list[str]] | None = None
	failure = False
	for index in range(args.runs):
		started = time.monotonic()
		result = subprocess.run(
			command,
			cwd=repo,
			env=os.environ.copy(),
			text=True,
			stdout=subprocess.PIPE,
			stderr=subprocess.PIPE,
			timeout=args.timeout,
			check=False,
		)
		duration = time.monotonic() - started
		stdout_path = logs / f"run-{index:02d}.stdout.txt"
		stderr_path = logs / f"run-{index:02d}.stderr.txt"
		stdout_path.write_text(result.stdout, encoding="utf-8")
		stderr_path.write_text(result.stderr, encoding="utf-8")
		normalized = ANSI_ESCAPE.sub("", result.stdout + "\n" + result.stderr).replace(
			"\r\n", "\n"
		)
		traces = [_extract_trace(pattern, normalized) for pattern in trace_patterns]
		required = [bool(pattern.search(normalized)) for pattern in required_patterns]
		all_traces_present = all(bool(trace) for trace in traces)
		matches_reference = reference is None or traces == reference
		if reference is None:
			reference = traces
		passed = (
			result.returncode == 0
			and all(required)
			and all_traces_present
			and matches_reference
		)
		samples.append(
			{
				"index": index,
				"duration_seconds": duration,
				"exit_code": result.returncode,
				"required_patterns_matched": required,
				"all_traces_present": all_traces_present,
				"matches_reference": matches_reference,
				"trace_groups": traces,
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
				"result": "PASS" if passed else "FAIL",
			}
		)
		if not passed:
			failure = True

	cache = args.cmake_cache.expanduser()
	if not cache.is_absolute():
		cache = repo / cache
	document: dict[str, Any] = {
		"schema": SCHEMA,
		"created_utc": dt.datetime.now(dt.timezone.utc).isoformat().replace(
			"+00:00", "Z"
		),
		"repository": {
			"commit": oaevidence._git(repo, "rev-parse", "HEAD") or "unknown",
			"branch": oaevidence._git(repo, "branch", "--show-current"),
			"describe": oaevidence._git(
				repo, "describe", "--always", "--dirty", "--tags"
			),
			"dirty": bool(oaevidence._git(repo, "status", "--porcelain") or ""),
		},
		"host": {
			"system": platform.system(),
			"release": platform.release(),
			"machine": platform.machine(),
			"python": platform.python_version(),
		},
		"build": oaevidence._read_cache(cache.resolve()),
		"environment": {
			key: os.environ[key]
			for key in oaevidence.RELEVANT_ENV
			if key in os.environ
		},
		"workload": {
			"name": args.name,
			"command": oaevidence._redact_command(command),
			"runs": args.runs,
			"timeout_seconds": args.timeout,
			"trace_regexes": args.trace_regex,
			"required_regexes": args.require_regex,
		},
		"samples": samples,
		"result": "FAIL" if failure else "PASS",
	}
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
		print(output)
		print(
			f"{document['workload']['name']}: runs={len(document['samples'])} "
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
		print(f"oadeterminism: {error}", file=sys.stderr)
		return 2


if __name__ == "__main__":
	raise SystemExit(main())
