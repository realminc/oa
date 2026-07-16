#!/usr/bin/env python3
"""
OA Autogen — unified code generation runner.

Runs all OA code generators:
  - oatypeautogen: generates enums/structs from TOML schemas
  - oafnautogen: generates function operations from TOML schemas
  - oannautogen: generates neural network modules
  - oatile: generates the bounded linear-algebra kernel lattice

Usage:
  python3 Tools/oaautogen.py              # Generate all to Output/
  python3 Tools/oaautogen.py --live       # Generate all to Source/
  python3 Tools/oaautogen.py --types      # Generate types only
  python3 Tools/oaautogen.py --functions  # Generate functions only
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).parent.parent


def run_type_autogen(live: bool, dry_run: bool) -> int:
	"""Run oatypeautogen."""
	cmd = [sys.executable, "Tools/TypeAutogen/oatypeautogen.py"]
	if live:
		cmd.append("--live")
	if dry_run:
		cmd.append("--dry-run")
	
	print("Running oatypeautogen...")
	result = subprocess.run(cmd, cwd=REPO_ROOT)
	return result.returncode


def run_fn_autogen(live: bool, dry_run: bool) -> int:
	"""Run oafnautogen."""
	cmd = [sys.executable, "Tools/FnAutogen/oafnautogen.py"]
	if live:
		cmd.append("--live")
	if dry_run:
		cmd.append("--dry-run")
	
	print("Running oafnautogen...")
	result = subprocess.run(cmd, cwd=REPO_ROOT)
	return result.returncode


def run_nn_autogen(live: bool, dry_run: bool) -> int:
	"""Run oannautogen."""
	cmd = [sys.executable, "Tools/NnAutogen/oannautogen.py"]
	if live:
		cmd.append("--live")
	if dry_run:
		cmd.append("--dry-run")

	print("Running oannautogen...")
	result = subprocess.run(cmd, cwd=REPO_ROOT)
	return result.returncode


def run_tile_autogen(live: bool, dry_run: bool) -> int:
	"""Run OaTile generation."""
	cmd = [sys.executable, "Tools/OaTile/oatile.py"]
	if live:
		cmd.append("--live")
	if dry_run:
		cmd.append("--dry-run")

	print("Running oatile...")
	result = subprocess.run(cmd, cwd=REPO_ROOT)
	return result.returncode


def main() -> int:
	parser = argparse.ArgumentParser(
		description="Unified OA code generation runner"
	)
	parser.add_argument(
		"--live",
		action="store_true",
		help="Write to Source/ instead of Output/"
	)
	parser.add_argument(
		"--dry-run",
		action="store_true",
		help="Print what would be written without writing files"
	)
	parser.add_argument(
		"--types",
		action="store_true",
		help="Run oatypeautogen only"
	)
	parser.add_argument(
		"--functions",
		action="store_true",
		help="Run oafnautogen only"
	)
	parser.add_argument(
		"--nn",
		action="store_true",
		help="Run oannautogen only"
	)
	parser.add_argument(
		"--tile",
		action="store_true",
		help="Run OaTile generation only"
	)
	
	args = parser.parse_args()
	
	# Determine which generators to run
	run_types = args.types or not (args.functions or args.nn or args.tile)
	run_functions = args.functions or not (args.types or args.nn or args.tile)
	run_nn = args.nn or not (args.types or args.functions or args.tile)
	run_tile = args.tile or not (args.types or args.functions or args.nn)
	
	exit_code = 0
	
	if run_types:
		exit_code |= run_type_autogen(args.live, args.dry_run)
	
	if run_functions:
		exit_code |= run_fn_autogen(args.live, args.dry_run)
	
	if run_nn:
		exit_code |= run_nn_autogen(args.live, args.dry_run)

	if run_tile:
		exit_code |= run_tile_autogen(args.live, args.dry_run)
	
	if exit_code == 0:
		print("\nAll generators completed successfully.")
	else:
		print("\nSome generators failed.")
	
	return exit_code


if __name__ == "__main__":
	sys.exit(main())
