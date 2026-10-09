#!/usr/bin/env python3
"""Run all NLP tutorial entries, or a filtered subset, in fresh processes.

Each entry is launched as an independent subprocess so engine construction,
RNG state, and GPU resources are fully isolated between runs.

Usage:
	python run_nlp_suite.py                  # run all 16 entries
	python run_nlp_suite.py char             # run all Char entries
	python run_nlp_suite.py byte mamba3      # run Byte × Mamba-3 only
	python run_nlp_suite.py --smoke          # one step, batch 2, two output bytes
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

# Canonical 16-entry suite in the same order as the C++ and oacpp Python suites.
# File names use the tu_nlp_ prefix matching the RS tutorial naming convention.
SUITE_MEMBERS = (
	"tu_nlp_byte_rnn",
	"tu_nlp_byte_gru",
	"tu_nlp_byte_transformer",
	"tu_nlp_byte_moe_transformer",
	"tu_nlp_byte_mamba3",
	"tu_nlp_byte_empyrealm",
	"tu_nlp_bpe_rnn",
	"tu_nlp_bpe_gru",
	"tu_nlp_bpe_transformer",
	"tu_nlp_bpe_moe_transformer",
	"tu_nlp_bpe_mamba3",
	"tu_nlp_char_rnn",
	"tu_nlp_char_gru",
	"tu_nlp_char_transformer",
	"tu_nlp_char_moe_transformer",
	"tu_nlp_char_mamba3",
)


def main() -> None:
	parser = argparse.ArgumentParser(description=__doc__)
	parser.add_argument(
		"filter",
		nargs="*",
		help=(
			"case-insensitive substrings; all provided terms must match. "
			"Examples: byte, bpe, char, rnn, gru, transformer, mamba3, moe, empyrealm"
		),
	)
	parser.add_argument(
		"--smoke",
		action="store_true",
		help="run one training step, batch size 2, and two generated output bytes",
	)
	args = parser.parse_args()

	selected = [
		name
		for name in SUITE_MEMBERS
		if not args.filter
		or all(part.lower() in name.lower() for part in args.filter)
	]
	if not selected:
		raise SystemExit(
			f"No NLP suite members matched the requested filters: {args.filter!r}\n"
			f"Available: {', '.join(SUITE_MEMBERS)}"
		)

	environment = os.environ.copy()
	if args.smoke:
		environment["OA_TUTORIAL_STEPS"] = "1"
		environment["OA_TUTORIAL_BATCH"] = "2"
		environment["OA_TUTORIAL_GENERATION_UNITS"] = "2"

	root = Path(__file__).resolve().parent
	passed = 0
	for index, name in enumerate(selected, 1):
		script = root / f"{name}.py"
		label = name.removeprefix("tu_nlp_")
		print(f"\n[{index}/{len(selected)}] {label}", flush=True)
		subprocess.run([sys.executable, str(script)], check=True, env=environment)
		passed += 1

	print(f"\nPASS: {passed}/{len(selected)} NLP tutorial entries")


if __name__ == "__main__":
	main()
