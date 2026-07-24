#!/usr/bin/env python3
"""Run all 16 Python NLP entries, or a filtered subset, in fresh processes."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess
import sys

# Keep every tutorial-side Python file on the same canonical public surface.
# pyright: reportWildcardImportFromLibrary=false
from oa import *
import _nlp_common as nlp


def main() -> None:
	parser = argparse.ArgumentParser(description=__doc__)
	parser.add_argument(
		"filter",
		nargs="*",
		help="case-insensitive substrings such as Byte, Bpe, Gru, or Mamba3",
	)
	parser.add_argument(
		"--smoke",
		action="store_true",
		help="run one step, batch two, and two generated source units",
	)
	args = parser.parse_args()

	selected = [
		name
		for name in nlp.SUITE_MEMBERS
		if not args.filter
		or all(part.lower() in name.lower() for part in args.filter)
	]
	if not selected:
		raise SystemExit("No NLP suite members matched the requested filters")

	environment = os.environ.copy()
	if args.smoke:
		environment["OA_TUTORIAL_STEPS"] = "1"
		environment["OA_TUTORIAL_BATCH"] = "2"
		environment["OA_TUTORIAL_GENERATION_UNITS"] = "2"

	root = Path(__file__).resolve().parent
	for index, name in enumerate(selected, 1):
		print(f"\n[{index}/{len(selected)}] {name}", flush=True)
		subprocess.run(
			[sys.executable, str(root / f"{name}.py")],
			check=True,
			env=environment,
		)

	print(f"\nPASS: {len(selected)} Python NLP suite members")


if __name__ == "__main__":
	main()
