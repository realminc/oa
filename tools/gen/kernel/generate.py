#!/usr/bin/env python3
"""Generate OA's standalone fixed-kernel build and registry surfaces."""

from __future__ import annotations

import argparse
import json
import re
import sys
import tomllib
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_SCHEMA = Path(__file__).with_name("schema.toml")
DEFAULT_OUTPUT = REPO_ROOT / "build" / "gen" / "kernel"

PREFIXES = {"Ml", "Crypto", "Vision", "Ui", "Audio", "Render"}
CATEGORIES = PREFIXES
PROFILES = {"core", "ssm", "vision"}
STAGES = {"compute", "vertex", "fragment"}
CONDITIONS = {"always", "crypto"}


class SchemaError(ValueError):
	pass


def loadSchema(path: Path) -> tuple[list[dict], list[dict]]:
	with path.open("rb") as stream:
		data = tomllib.load(stream)
	if data.get("version") != 1:
		raise SchemaError("schema version must be 1")
	kernels = data.get("kernels")
	reservations = data.get("reservations")
	if not isinstance(kernels, list) or not isinstance(reservations, list):
		raise SchemaError("kernels and reservations must be arrays")
	return kernels, reservations


def normalizedRecords(
 kernels: list[dict], reservations: list[dict], sourceRoot: Path
) -> tuple[list[dict], list[dict]]:
	seenNames: set[str] = set()
	seenIds: set[tuple[str, int]] = set()
	normalized: list[dict] = []
	for index, raw in enumerate(kernels):
		where = f"kernels[{index}]"
		if not isinstance(raw, dict):
			raise SchemaError(f"{where} must be a table")
		row = {
		 "name": raw.get("name"),
		 "prefix": raw.get("prefix"),
		 "local": raw.get("local"),
		 "category": raw.get("category"),
		 "origin": raw.get("origin", "oa"),
		 "source": raw.get("source"),
		 "profile": raw.get("profile", "core"),
		 "stage": raw.get("stage", "compute"),
		 "condition": raw.get("condition", "always"),
		}
		for field in ("name", "prefix", "category", "origin", "source"):
			if not isinstance(row[field], str) or not row[field]:
				raise SchemaError(f"{where}.{field} must be a non-empty string")
		if row["prefix"] not in PREFIXES:
			raise SchemaError(f"{where}.prefix is unknown: {row['prefix']}")
		if row["category"] not in CATEGORIES:
			raise SchemaError(f"{where}.category is unknown: {row['category']}")
		if not isinstance(row["local"], int) or row["local"] <= 0:
			raise SchemaError(f"{where}.local must be a positive integer")
		if row["profile"] not in PROFILES:
			raise SchemaError(f"{where}.profile is unknown: {row['profile']}")
		if row["stage"] not in STAGES:
			raise SchemaError(f"{where}.stage is unknown: {row['stage']}")
		if row["condition"] not in CONDITIONS:
			raise SchemaError(f"{where}.condition is unknown: {row['condition']}")
		if row["stage"] != "compute" and row["category"] != "Render":
			raise SchemaError(f"{where}: only Render kernels may use graphics stages")
		if row["category"] == "Render" and row["stage"] == "compute":
			raise SchemaError(f"{where}: Render kernels require an explicit graphics stage")
		if row["condition"] == "crypto" and row["category"] != "Crypto":
			raise SchemaError(f"{where}: only Crypto kernels may be conditional")
		if not row["source"].endswith(".slang"):
			raise SchemaError(f"{where}.source must name a .slang source")
		source = sourceRoot / row["source"]
		if not source.is_file():
			raise SchemaError(f"{where}.source does not exist: {row['source']}")
		packed = (row["prefix"], row["local"])
		if row["name"] in seenNames:
			raise SchemaError(f"duplicate kernel name: {row['name']}")
		if packed in seenIds:
			raise SchemaError(f"duplicate kernel id: {row['prefix']}:{row['local']}")
		seenNames.add(row["name"])
		seenIds.add(packed)
		normalized.append(row)

	normalizedReservations: list[dict] = []
	seenRanges: set[tuple[str, int, int]] = set()
	for index, raw in enumerate(reservations):
		where = f"reservations[{index}]"
		if not isinstance(raw, dict):
			raise SchemaError(f"{where} must be a table")
		prefix = raw.get("prefix")
		first = raw.get("first")
		last = raw.get("last")
		if prefix not in PREFIXES:
			raise SchemaError(f"{where}.prefix is unknown: {prefix}")
		if not isinstance(first, int) or not isinstance(last, int) or first <= 0 or last < first:
			raise SchemaError(f"{where} must be a positive inclusive range")
		key = (prefix, first, last)
		if key in seenRanges:
			raise SchemaError(f"duplicate reservation: {prefix}:{first}-{last}")
		for activePrefix, activeLocal in seenIds:
			if activePrefix == prefix and first <= activeLocal <= last:
				raise SchemaError(
				 f"reservation {prefix}:{first}-{last} overlaps active id {activeLocal}"
				)
		seenRanges.add(key)
		normalizedReservations.append({"prefix": prefix, "first": first, "last": last})
	return normalized, normalizedReservations


def writeIfChanged(path: Path, content: str, *, dryRun: bool = False) -> None:
	if dryRun:
		print(f"would write: {path}")
		return
	encoded = content.encode("utf-8")
	if path.exists() and path.read_bytes() == encoded:
		return
	path.parent.mkdir(parents=True, exist_ok=True)
	path.write_bytes(encoded)


def emitRegistry(
	outRoot: Path,
	kernels: list[dict],
	reservations: list[dict],
	*,
	dryRun: bool = False,
) -> None:
	runtime = outRoot / "lib" / "oa" / "runtime" / "gen"
	for category in ("Ml", "Vision", "Ui", "Audio", "Render", "Crypto"):
		lines = [
		 "// AUTO-GENERATED by tools/gen/kernel/generate.py — DO NOT EDIT.",
		 "// Source of truth: tools/gen/kernel/schema.toml.",
		]
		for row in kernels:
			if row["category"] != category:
				continue
			lines.append(
			 f'{{ "{row["name"]}", OA_COMPUTE_KERNEL_ID(oa::computeKernelPrefix::{row["prefix"]}, '
			 f'{row["local"]}), oa::ComputeKernelCategory::{row["category"]}, "{row["origin"]}" }},'
			)
		writeIfChanged(
			runtime / f"kernelRegistryStandalone{category}.inl",
			"\n".join(lines) + "\n",
			dryRun=dryRun,
		)

	lines = [
	 "// AUTO-GENERATED by tools/gen/kernel/generate.py — DO NOT EDIT.",
	 "// Source of truth: tools/gen/kernel/schema.toml.",
	]
	for row in reservations:
		lines.append(
		 f'{{ oa::computeKernelPrefix::{row["prefix"]}, {row["first"]}, {row["last"]} }},'
		)
	writeIfChanged(
		runtime / "kernelReservations.inl",
		"\n".join(lines) + "\n",
		dryRun=dryRun,
	)

	lines = [
	 "// AUTO-GENERATED by tools/gen/kernel/generate.py — DO NOT EDIT.",
	 "// Source of truth: tools/gen/kernel/schema.toml.",
	]
	for row in kernels:
		if re.fullmatch(r"[A-Za-z_]\w*", row["name"]) is None:
			continue
		lines.append(
		 f"static constexpr oa::U64 {row['name']} = "
		 f"OA_COMPUTE_KERNEL_ID(oa::computeKernelPrefix::{row['prefix']}, {row['local']});"
		)
	writeIfChanged(
		outRoot / "include" / "oa" / "runtime" / "gen" / "kernelIdsStandalone.inl",
		"\n".join(lines) + "\n",
		dryRun=dryRun,
	)


def emitCmake(outRoot: Path, kernels: list[dict], *, dryRun: bool = False) -> None:
	path = (
	 REPO_ROOT / "cmake" / "gen" / "kernelManifest.cmake"
	 if outRoot == REPO_ROOT / "source" / "cpp"
	 else outRoot / "cmake" / "gen" / "kernelManifest.cmake"
	)
	lines = [
	 "# AUTO-GENERATED by tools/gen/kernel/generate.py -- DO NOT EDIT.",
	 "# Source of truth: tools/gen/kernel/schema.toml.",
	]
	fields = (
	 ("OA_STANDALONE_KERNEL_NAMES", "name"),
	 ("OA_STANDALONE_KERNEL_SOURCES", "source"),
	 ("OA_STANDALONE_KERNEL_PROFILES", "profile"),
	 ("OA_STANDALONE_KERNEL_STAGES", "stage"),
	 ("OA_STANDALONE_KERNEL_CONDITIONS", "condition"),
	)
	for variable, field in fields:
		lines.append(f"set({variable}")
		lines.extend(f'\t"{row[field]}"' for row in kernels)
		lines.append(")")
	writeIfChanged(path, "\n".join(lines) + "\n", dryRun=dryRun)


def emitJson(
	outRoot: Path,
	kernels: list[dict],
	reservations: list[dict],
	*,
	dryRun: bool = False,
) -> None:
	payload = {
	 "generator": "tools/gen/kernel/generate.py",
	 "schema": "tools/gen/kernel/schema.toml",
	 "version": 1,
	 "kernels": kernels,
	 "reservations": reservations,
	}
	writeIfChanged(
		outRoot / "lib" / "oa" / "runtime" / "gen" / "kernelManifest.json",
		json.dumps(payload, indent=2, sort_keys=True) + "\n",
		dryRun=dryRun,
	)


def generate(
	schema: Path,
	outRoot: Path,
	sourceRoot: Path,
	*,
	dryRun: bool = False,
) -> tuple[int, int]:
	kernels, reservations = loadSchema(schema)
	kernels, reservations = normalizedRecords(kernels, reservations, sourceRoot)
	emitRegistry(outRoot, kernels, reservations, dryRun=dryRun)
	emitCmake(outRoot, kernels, dryRun=dryRun)
	emitJson(outRoot, kernels, reservations, dryRun=dryRun)
	return len(kernels), len(reservations)


def main() -> int:
	parser = argparse.ArgumentParser(description=__doc__)
	parser.add_argument("--schema", type=Path, default=DEFAULT_SCHEMA)
	parser.add_argument("--out", type=Path, default=DEFAULT_OUTPUT)
	parser.add_argument("--live", action="store_true", help="Write generated files into source/")
	parser.add_argument(
		"--dry-run", dest="dryRun",
		action="store_true",
		help="Validate and print outputs without writing files",
	)
	args = parser.parse_args()
	if args.live and args.out != DEFAULT_OUTPUT:
		parser.error("--live and --out are mutually exclusive")
	outRoot = REPO_ROOT / "source" / "cpp" if args.live else args.out
	try:
		kernelCount, reservationCount = generate(
			args.schema,
			outRoot,
			REPO_ROOT,
			dryRun=args.dryRun,
		)
	except (OSError, tomllib.TOMLDecodeError, SchemaError) as error:
		print(f"kernelGen: ERROR: {error}", file=sys.stderr)
		return 1
	print(f"kernelGen: {kernelCount} standalone kernels, {reservationCount} reserved ranges")
	return 0


if __name__ == "__main__":
	raise SystemExit(main())
