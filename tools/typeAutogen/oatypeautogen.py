#!/usr/bin/env python3
"""
typeAutogen — schema-driven generator for OA C++ types.

Stdlib only (tomllib + pathlib + argparse). No pip deps.

Discovers every TOML schema under `schema/`, validates each schema,
and emits per-domain files at their configured public or private ownership:

  - <out>/source/cpp/include/oa/<domain>/type.gen.h for public types
  - <out>/source/cpp/lib/oa/<domain>/type.gen.h for private runtime types
  - <out>/source/cpp/lib/oa/<domain>/type.gen.cpp include anchors
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

TOOLS_ROOT = Path(__file__).resolve().parents[1]
if str(TOOLS_ROOT) not in sys.path:
	sys.path.insert(0, str(TOOLS_ROOT))

from autogen_io import write_generated_text

from oatypeautogen_lib.config import (
	DOMAIN_OUTPUT_PATHS,
	DEFAULT_OUTPUT,
	LIVE_SOURCE_ROOT,
	REPO_ROOT,
	SCHEMA_DIR,
)
from oatypeautogen_lib.emitters import emit_cpp_file, emit_header_file
from oatypeautogen_lib.schema import load_schema, validate_schema


def schema_source_name(schema_path: Path) -> str:
	"""Return repository schema provenance for generated-file markers."""
	try:
		return schema_path.resolve().relative_to(
			SCHEMA_DIR.resolve()
		).as_posix()
	except ValueError:
		return schema_path.name


def process_schema(schema_path: Path, out_root: Path, *, live: bool, dry_run: bool) -> None:
	"""Process a single schema file and generate C++ code."""
	schema = load_schema(schema_path)
	errors = validate_schema(schema)
	
	if errors:
		print(f"ERROR: {schema_path.name} validation failed:")
		for err in errors:
			print(f"  - {err}")
		sys.exit(1)
	
	print(f"oatypeautogen: {schema_path.name} — domain={schema.domain}, "
	      f"{len(schema.enums)} enums, {len(schema.structs)} structs")
	
	# Determine output paths
	if schema.domain in DOMAIN_OUTPUT_PATHS:
		paths = DOMAIN_OUTPUT_PATHS[schema.domain]
		if live:
			header_path = LIVE_SOURCE_ROOT / paths["header"]
			cpp_path = LIVE_SOURCE_ROOT / paths["cpp"]
		else:
			header_path = out_root / paths["header"]
			cpp_path = out_root / paths["cpp"]
	else:
		# Default to the canonical generated preview layout.
		header_path = out_root / f"source/cpp/include/oa/{schema.domain.lower()}/type.gen.h"
		cpp_path = out_root / f"source/cpp/lib/oa/{schema.domain.lower()}/type.gen.cpp"
	
	schema_name = schema_source_name(schema_path)
	
	# Generate header content
	header_content = emit_header_file(schema, schema_name)
	
	# Generate cpp content
	cpp_content = emit_cpp_file(schema, schema_name)
	
	if dry_run:
		print(f"  would write: {header_path}")
		print(f"  would write: {cpp_path}")
		return
	
	# Write files
	write_generated_text(header_path, header_content)
	print(f"  wrote: {header_path}")

	write_generated_text(cpp_path, cpp_content)
	print(f"  wrote: {cpp_path}")


def main() -> int:
	parser = argparse.ArgumentParser(
		description="Generate OA C++ types from TOML schemas"
	)
	parser.add_argument(
		"--schema",
		type=Path,
		default=None,
		help="Specific schema file to process (default: all schemas in schema/)"
	)
	parser.add_argument(
		"--out",
		type=Path,
		default=DEFAULT_OUTPUT,
		help="Output directory (default: tools/typeAutogen/output)"
	)
	parser.add_argument(
		"--live",
		action="store_true",
		help="Write to source/ instead of tools/typeAutogen/output/"
	)
	parser.add_argument(
		"--dry-run",
		action="store_true",
		help="Print what would be written without writing files"
	)
	
	args = parser.parse_args()
	
	# Find schemas to process
	if args.schema:
		schemas = [args.schema]
	else:
		schemas = sorted(SCHEMA_DIR.glob("**/*.toml"))
	
	if not schemas:
		print("No schemas found")
		return 0
	
	print(f"Processing {len(schemas)} schema(s)...")
	
	for schema_path in schemas:
		process_schema(schema_path, args.out, live=args.live, dry_run=args.dry_run)
	
	print("\nDone.")
	return 0


if __name__ == "__main__":
	sys.exit(main())
