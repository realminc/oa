#!/usr/bin/env python3
"""Run OA's schema-owned source generators from one entry point.

Without a selector, every generator runs. Preview output is written below
``build/gen``; ``--live`` updates the checked-in generated source surfaces.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


RepoRoot = Path(__file__).resolve().parents[2]
Generators = (
	("type", "types and records", "tools/gen/type/generate.py"),
	("fn", "operations and bindings", "tools/gen/fn/generate.py"),
	("nn", "neural-network modules", "tools/gen/nn/generate.py"),
	("kernel", "standalone kernel authority", "tools/gen/kernel/generate.py"),
	("tile", "linear-algebra kernel lattice", "tools/gen/tile/generate.py"),
	("example", "paired SDK examples", "tools/gen/example/generate.py"),
)


def runGenerator(script: str, *, live: bool, dryRun: bool) -> int:
	command = [sys.executable, script]
	if live:
		command.append("--live")
	if dryRun:
		command.append("--dry-run")
	return subprocess.run(command, cwd=RepoRoot, check=False).returncode


def parseArgs() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description=__doc__)
	parser.add_argument(
		"--live",
		action="store_true",
		help="Update checked-in generated artifacts instead of build/gen previews",
	)
	parser.add_argument(
		"--dry-run", dest="dryRun",
		action="store_true",
		help="Validate and report outputs without writing files",
	)
	for name, description, _ in Generators:
		parser.add_argument(
			f"--{name}",
			action="store_true",
			help=f"Run only the {description} generator",
		)
	return parser.parse_args()


def main() -> int:
	args = parseArgs()
	selected = {name for name, _, _ in Generators if getattr(args, name)}
	if not selected:
		selected = {name for name, _, _ in Generators}

	status = 0
	for name, description, script in Generators:
		if name not in selected:
			continue
		print(f"gen: {name} — {description}", flush=True)
		status |= runGenerator(script, live=args.live, dryRun=args.dryRun)

	if status == 0:
		print("gen: all selected generators completed successfully")
	else:
		print("gen: one or more generators failed", file=sys.stderr)
	return status


if __name__ == "__main__":
	raise SystemExit(main())
