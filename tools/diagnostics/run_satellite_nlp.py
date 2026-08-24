#!/usr/bin/env python3
"""Run one standalone or true two-process OA satellite NLP benchmark."""

from __future__ import annotations

import argparse
import os
import pathlib
import socket
import subprocess
import sys
import tempfile
import time
from typing import Sequence


_WORKLOAD = "nlp-byte-transformer-step-v1"
_DEVELOPMENT_KEY = "".join(f"{value:02x}" for value in range(32))


def _parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo", type=pathlib.Path, default=pathlib.Path(__file__).parents[2]
    )
    parser.add_argument(
        "--binary", type=pathlib.Path, default=pathlib.Path("bin/release/oa-satellite")
    )
    parser.add_argument("--mode", choices=("standalone", "split-batch"), required=True)
    parser.add_argument("--checkpoint", type=pathlib.Path)
    parser.add_argument("--timeout", type=float, default=180.0)
    return parser.parse_args(argv)


def _endpoint() -> str:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.bind(("127.0.0.1", 0))
        return f"127.0.0.1:{probe.getsockname()[1]}"


def _client_command(
    binary: pathlib.Path,
    mode: str,
    checkpoint: pathlib.Path,
    endpoint: str | None = None,
) -> list[str]:
    command = [
        str(binary),
        "benchmark",
        "--workload",
        _WORKLOAD,
        "--mode",
        mode,
        "--checkpoint",
        str(checkpoint),
    ]
    if endpoint is not None:
        command.extend(("--peer", endpoint, "--auth-key-hex", _DEVELOPMENT_KEY))
    return command


def _run_standalone(
    repo: pathlib.Path,
    binary: pathlib.Path,
    checkpoint: pathlib.Path,
    timeout: float,
) -> int:
    completed = subprocess.run(
        _client_command(binary, "standalone", checkpoint),
        cwd=repo,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=timeout,
        check=False,
    )
    sys.stdout.buffer.write(completed.stdout)
    sys.stderr.buffer.write(completed.stderr)
    return completed.returncode


def _run_split(
    repo: pathlib.Path,
    binary: pathlib.Path,
    checkpoint: pathlib.Path,
    timeout: float,
) -> int:
    endpoint = _endpoint()
    server = subprocess.Popen(
        (
            str(binary),
            "serve",
            "--listen",
            endpoint,
            "--auth-key-hex",
            _DEVELOPMENT_KEY,
            "--once",
        ),
        cwd=repo,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    started = time.monotonic()
    try:
        # The server flushes this line only after bind and engine creation, so a
        # successful read is the readiness contract and needs no polling sleep.
        assert server.stdout is not None
        ready = server.stdout.readline()
        if not ready.startswith(b"oa-satellite: listening on "):
            server_stdout, server_stderr = server.communicate(timeout=5.0)
            sys.stdout.buffer.write(ready + server_stdout)
            sys.stderr.buffer.write(server_stderr)
            return server.returncode if server.returncode not in (None, 0) else 1
        remaining = max(1.0, timeout - (time.monotonic() - started))
        completed = subprocess.run(
            _client_command(binary, "split-batch", checkpoint, endpoint),
            cwd=repo,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=remaining,
            check=False,
        )
        sys.stdout.buffer.write(completed.stdout)
        sys.stderr.buffer.write(completed.stderr)
        remaining = max(1.0, timeout - (time.monotonic() - started))
        server_stdout, server_stderr = server.communicate(timeout=remaining)
        sys.stderr.buffer.write(server_stderr)
        if server_stdout:
            sys.stdout.buffer.write(server_stdout)
        if completed.returncode != 0:
            return completed.returncode
        return server.returncode or 0
    finally:
        if server.poll() is None:
            server.terminate()
            try:
                server.wait(timeout=5.0)
            except subprocess.TimeoutExpired:
                server.kill()
                server.wait()


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(sys.argv[1:] if argv is None else argv)
    repo = args.repo.expanduser().resolve()
    binary = args.binary.expanduser()
    if not binary.is_absolute():
        binary = repo / binary
    binary = binary.resolve()
    if not binary.is_file() or not os.access(binary, os.X_OK):
        print(f"run_satellite_nlp: binary is not executable: {binary}", file=sys.stderr)
        return 2
    checkpoint = args.checkpoint
    temporary: tempfile.TemporaryDirectory[str] | None = None
    if checkpoint is None:
        temporary = tempfile.TemporaryDirectory(prefix="oa-satellite-nlp-")
        checkpoint = pathlib.Path(temporary.name) / "checkpoint.oam"
    else:
        checkpoint = checkpoint.expanduser().resolve()
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
    try:
        if args.mode == "standalone":
            return _run_standalone(repo, binary, checkpoint, args.timeout)
        return _run_split(repo, binary, checkpoint, args.timeout)
    except subprocess.TimeoutExpired:
        print("run_satellite_nlp: workload timed out", file=sys.stderr)
        return 124
    finally:
        if temporary is not None:
            temporary.cleanup()


if __name__ == "__main__":
    raise SystemExit(main())
