from __future__ import annotations

import sys


def fail(msg: str) -> None:
	print(f"fnGen: ERROR: {msg}", file=sys.stderr)
	sys.exit(1)
