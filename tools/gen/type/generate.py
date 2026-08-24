#!/usr/bin/env python3
"""
OA type generation from schema-owned C++ contracts.

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

REPO_ROOT_PATH = Path(__file__).resolve().parents[3]
if str(REPO_ROOT_PATH) not in sys.path:
	sys.path.insert(0, str(REPO_ROOT_PATH))

from tools.gen.io import writeGeneratedText

from tools.gen.type.config import (
 DOMAIN_OUTPUT_PATHS,
 DEFAULT_OUTPUT,
 LIVE_SOURCE_ROOT,
 REPO_ROOT,
 SCHEMA_DIR,
)
from tools.gen.type.emitters import emitCppFile, emitHeaderFile
from tools.gen.type.schema import loadSchema, validateSchema


def schemaSourceName(schemaPath: Path) -> str:
	"""Return repository schema provenance for generated-file markers."""
	try:
		return schemaPath.resolve().relative_to(
		 SCHEMA_DIR.resolve()
		).as_posix()
	except ValueError:
		return schemaPath.name


def processSchema(schemaPath: Path, outRoot: Path, *, live: bool, dryRun: bool) -> None:
	"""Process a single schema file and generate C++ code."""
	schema = loadSchema(schemaPath)
	errors = validateSchema(schema)

	if errors:
		print(f"ERROR: {schemaPath.name} validation failed:")
		for err in errors:
			print(f"  - {err}")
		sys.exit(1)

	print(f"typeGen: {schemaPath.name} — domain={schema.domain}, "
	      f"{len(schema.enums)} enums, {len(schema.structs)} structs")

	# Determine output paths
	if schema.domain in DOMAIN_OUTPUT_PATHS:
		paths = DOMAIN_OUTPUT_PATHS[schema.domain]
		if live:
			headerPath = LIVE_SOURCE_ROOT / paths["header"]
			cppPath = LIVE_SOURCE_ROOT / paths["cpp"]
		else:
			headerPath = outRoot / paths["header"]
			cppPath = outRoot / paths["cpp"]
	else:
	 # Default to the canonical generated preview layout.
		headerPath = outRoot / f"source/cpp/include/oa/{schema.domain.lower()}/type.gen.h"
		cppPath = outRoot / f"source/cpp/lib/oa/{schema.domain.lower()}/type.gen.cpp"

	schemaName = schemaSourceName(schemaPath)

	# Generate header content
	headerContent = emitHeaderFile(schema, schemaName)

	# Generate cpp content
	cppContent = emitCppFile(schema, schemaName)

	if dryRun:
		print(f"  would write: {headerPath}")
		print(f"  would write: {cppPath}")
		return

 # Write files
	writeGeneratedText(headerPath, headerContent)
	print(f"  wrote: {headerPath}")

	writeGeneratedText(cppPath, cppContent)
	print(f"  wrote: {cppPath}")


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
	 help="Preview output root (default: build/gen/type)"
	)
	parser.add_argument(
	 "--live",
	 action="store_true",
	 help="Update checked-in generated source artifacts"
	)
	parser.add_argument(
	 "--dry-run", dest="dryRun",
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

	for schemaPath in schemas:
		processSchema(schemaPath, args.out, live=args.live, dryRun=args.dryRun)

	print("\nDone.")
	return 0


if __name__ == "__main__":
	sys.exit(main())
