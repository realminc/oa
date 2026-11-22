#!/usr/bin/env python3
"""Check fn and NN outputs, deterministic generation, and idempotent publication."""

from __future__ import annotations

import hashlib
import subprocess
import sys
import tempfile
from pathlib import Path


def snapshot(root: Path) -> dict[str, tuple[bytes, int]]:
	return {
		path.relative_to(root).as_posix(): (hashlib.sha256(path.read_bytes()).digest(), path.stat().st_mtime_ns)
		for path in sorted(root.rglob("*")) if path.is_file()
	}


def main() -> int:
	root = Path(__file__).resolve().parents[2]
	for authority in ("fn", "nn"):
		generator = root / f"tool/gen/{authority}/generate.py"
		command = [sys.executable, str(generator), "--root", str(root)]
		result = subprocess.run(command + ["--check"], check=False)
		if result.returncode:
			return result.returncode
		with tempfile.TemporaryDirectory(prefix=f"oa-{authority}-drift-") as directory:
			preview = Path(directory)
			for iteration in range(2):
				result = subprocess.run(command + ["--output-dir", str(preview)], check=False)
				if result.returncode:
					return result.returncode
				current = snapshot(preview)
				if iteration == 0:
					first = current
				elif current != first:
					print(f"{authority} generation changed bytes, output inventory, or timestamps on repetition", file=sys.stderr)
					return 1
		print(f"{authority} drift: {len(current)} outputs match; repeated generation preserves bytes and timestamps")
	return 0


if __name__ == "__main__":
	raise SystemExit(main())
