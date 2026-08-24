from __future__ import annotations

import sys


def fail(msg: str) -> None:
	print(f"oafnautogen: ERROR: {msg}", file=sys.stderr)
	sys.exit(1)

