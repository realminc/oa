"""Command-line entry point."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from .paths import REPO_ROOT
from .pipeline import expected_outputs
from .publish import check, publish
from .registry import registry_entries
from .schema import SCHEMAS, load_schemas
from .util import SchemaError


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser()
	parser.add_argument("--root", type=Path, default=REPO_ROOT)
	for name, spec in SCHEMAS.items():
		parser.add_argument(spec.flag, dest=name, type=Path)
	mode = parser.add_mutually_exclusive_group()
	mode.add_argument("--check", action="store_true")
	mode.add_argument("--live", action="store_true", help="update checked-in outputs (default)")
	mode.add_argument("--output-dir", type=Path, help="write a disposable source-tree preview")
	return parser.parse_args()


def main() -> int:
	args = parse_args()
	root = args.root.resolve()
	overrides = {name: getattr(args, name) for name in SCHEMAS if getattr(args, name)}
	try:
		schemas = load_schemas(root, overrides)
		registry_entries(schemas)
	except (OSError, SchemaError) as error:
		print(f"operation generation failed: {error}", file=os.sys.stderr)
		return 1
	try:
		outputs = expected_outputs(root, schemas)
	except (OSError, SchemaError) as error:
		print(f"operation assembly failed: {error}", file=os.sys.stderr)
		return 1
	output_root = root
	if args.output_dir:
		output_root = args.output_dir.resolve()
		if output_root == root or root.is_relative_to(output_root) or output_root.is_relative_to(root / "src") or output_root.is_relative_to(root / "test"):
			print("preview must not overlap the repository source trees", file=os.sys.stderr)
			return 1
		outputs = {output_root / path.relative_to(root): content for path, content in outputs.items()}
	try:
		if args.check:
			errors = check(outputs, output_root)
			if errors:
				print("\n".join(errors), file=os.sys.stderr)
				return 1
		else:
			publish(outputs, output_root)
	except (OSError, ValueError) as error:
		print(f"operation publication failed: {error}", file=os.sys.stderr)
		return 1
	return 0
