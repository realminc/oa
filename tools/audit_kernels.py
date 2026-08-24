#!/usr/bin/env python3
"""Audit OA's configured shader manifest against stable IDs and live consumers.

The configured core ``spirv_shader_list.txt`` and SDK-owned shader manifests
are the exact lists consumed by their embedded-SPIR-V generators.  This tool
deliberately does not try to re-parse CMake source syntax: doing so used to miss
generated variants, flatten logical names, and mistake registry declarations
for production dispatches.

Strict mode rejects:

* duplicate or malformed configured manifest entries;
* a fixed kernel whose configured build does not compile it;
* a generated matmul variant whose configured build does not compile it;
* a configured shader whose SPIR-V output is absent;
* a configured shader without a fixed or generated matmul-variant identity;
* a fixed kernel with no production runtime, extension, or SDK reference; and
* a configured shader with neither a fixed identity nor a production reference.

Tests are excluded from liveness.  A shader used only by a raw kernel test is
not a shipped operation.  Generated runtime/planner ``.inc`` and ``.inl`` files
are included because they are production selection tables.
"""

from __future__ import annotations

import argparse
import re
import sys
import tomllib
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
SOURCE_SUFFIXES = {".cpp", ".h", ".inc", ".inl"}
PRODUCTION_ROOTS = ("source", "extensions", "sdk")
MANIFEST_NAME = "spirv_shader_list.txt"
SDK_ENVIRONMENT_MANIFEST = Path("sdk/cpp/environment_shader_list.txt")

REGISTRY_ROW_RE = re.compile(
	r'\{\s*"([^"]+)"\s*,\s*'
	r'OA_COMPUTE_KERNEL_ID\(oa::computeKernelPrefix::(\w+),\s*(\d+)\),\s*'
	r'oa::ComputeKernelCategory::(\w+),\s*"([^"]+)"\s*\}'
)
KERNEL_ID_RE = re.compile(
	r'(?:static\s+)?constexpr\s+oa::U64\s+(\w+)\s*=\s*'
	r'OA_COMPUTE_KERNEL_ID\(oa::computeKernelPrefix::(\w+),\s*(\d+)\)'
)
RESERVED_RANGE_RE = re.compile(
	r'\{\s*oa::computeKernelPrefix::(\w+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\}'
)
STRING_RE = re.compile(r'"([^"\\]+)"')
TYPED_ID_USE_RE = re.compile(r'oa::computeKernelId::([A-Za-z_]\w*)')
MATMUL_VARIANT_ROW_RE = re.compile(
	r'oa::matmulVariantIdFromName\("([^"]+)"\)\s*,\s*"([^"]+)"'
)


@dataclass(frozen=True)
class FixedKernel:
	name: str
	prefix: str
	local_id: int
	category: str
	origin: str
	source: Path


def registry_sources(root: Path) -> list[Path]:
	runtime = root / "source/cpp/lib/oa/runtime"
	paths = [runtime / "kernelRegistry.h"]
	paths.extend(sorted(runtime.glob("kernelRegistry*.gen.inl")))
	paths.append(runtime / "oaTileKernelRegistry.gen.inc")
	return list(dict.fromkeys(path for path in paths if path.exists()))


def kernel_id_sources(root: Path) -> list[Path]:
	public = root / "source/cpp/include/oa/runtime"
	return [
		path for path in (
			public / "computeKernel.h",
			public / "kernelIdsStandalone.gen.inl",
			public / "computeKernelIds.gen.inl",
			public / "oaTileComputeKernel.gen.h",
		) if path.exists()
	]


def matmul_variant_sources(root: Path) -> list[Path]:
	gemm = root / "source/cpp/lib/oa/runtime/gemm"
	return sorted(gemm.glob("oaTile*Variants.gen.inc"))


def load_matmul_variants(root: Path) -> tuple[dict[str, Path], list[str]]:
	variants: dict[str, Path] = {}
	errors: list[str] = []
	for path in matmul_variant_sources(root):
		for match in MATMUL_VARIANT_ROW_RE.finditer(path.read_text()):
			identity_name = match.group(1)
			kernel_name = match.group(2)
			relative = path.relative_to(root)
			if identity_name != kernel_name:
				errors.append(
					f"matmul variant identity {identity_name} names kernel "
					f"{kernel_name} in {relative}"
				)
				continue
			if previous := variants.get(identity_name):
				errors.append(
					f"duplicate matmul variant identity {identity_name}: "
					f"{previous}, {relative}"
				)
				continue
			variants[identity_name] = relative
	return variants, errors


def load_manifest_file(path: Path) -> tuple[list[str], list[str]]:
	errors: list[str] = []
	if not path.exists():
		return [], [f"configured shader manifest not found: {path}"]
	names = [line.strip() for line in path.read_text().splitlines() if line.strip()]
	seen: set[str] = set()
	for name in names:
		if name.endswith(".spv"):
			errors.append(f"manifest key must omit .spv suffix: {name}")
		if name.startswith("/") or name.startswith("../") or "/../" in name:
			errors.append(f"manifest key escapes the SPIR-V root: {name}")
		if name in seen:
			errors.append(f"duplicate manifest key: {name}")
		seen.add(name)
	return names, errors


def load_manifest(build_dir: Path) -> tuple[list[str], list[str]]:
	return load_manifest_file(build_dir / MANIFEST_NAME)


def load_fixed_registry(root: Path) -> tuple[dict[str, FixedKernel], list[str]]:
	by_name: dict[str, FixedKernel] = {}
	by_id: dict[tuple[str, int], FixedKernel] = {}
	errors: list[str] = []
	for path in registry_sources(root):
		for match in REGISTRY_ROW_RE.finditer(path.read_text()):
			row = FixedKernel(
				name=match.group(1),
				prefix=match.group(2),
				local_id=int(match.group(3)),
				category=match.group(4),
				origin=match.group(5),
				source=path.relative_to(root),
			)
			if previous := by_name.get(row.name):
				errors.append(
					f"duplicate fixed name {row.name}: {previous.source}, {row.source}"
				)
			else:
				by_name[row.name] = row
			packed = (row.prefix, row.local_id)
			if previous := by_id.get(packed):
				errors.append(
					f"duplicate fixed ID {row.prefix}:{row.local_id}: "
					f"{previous.name}, {row.name}"
				)
			else:
				by_id[packed] = row
	return by_name, errors


def load_reserved_registry(root: Path) -> tuple[dict[str, set[int]], list[str]]:
	runtime = root / "source/cpp/lib/oa/runtime"
	paths = [runtime / "kernelRegistry.h", runtime / "kernelReservations.gen.inl"]
	reserved: dict[str, set[int]] = defaultdict(set)
	errors: list[str] = []
	if not any(path.exists() for path in paths):
		return {}, [f"kernel reservation ledger not found below: {runtime}"]
	for path in paths:
		if not path.exists():
			continue
		for match in RESERVED_RANGE_RE.finditer(path.read_text()):
			prefix = match.group(1)
			first = int(match.group(2))
			last = int(match.group(3))
			if first <= 0 or last < first:
				errors.append(f"invalid reserved kernel range {prefix}:{first}-{last}")
				continue
			for local in range(first, last + 1):
				if local in reserved[prefix]:
					errors.append(f"overlapping reserved kernel ID {prefix}:{local}")
					continue
				reserved[prefix].add(local)
	return dict(reserved), errors


def format_local_ranges(values: set[int]) -> str:
	ordered = sorted(values)
	if not ordered:
		return ""
	ranges: list[str] = []
	first = previous = ordered[0]
	for value in ordered[1:]:
		if value == previous + 1:
			previous = value
			continue
		ranges.append(str(first) if first == previous else f"{first}-{previous}")
		first = previous = value
	ranges.append(str(first) if first == previous else f"{first}-{previous}")
	return ",".join(ranges)


def validate_reserved_coverage(
	fixed: dict[str, FixedKernel], reserved: dict[str, set[int]]
) -> list[str]:
	active: dict[str, set[int]] = defaultdict(set)
	for row in fixed.values():
		active[row.prefix].add(row.local_id)
	errors: list[str] = []
	for prefix in sorted(set(active) | set(reserved)):
		active_ids = active.get(prefix, set())
		reserved_ids = reserved.get(prefix, set())
		for local in sorted(active_ids & reserved_ids):
			errors.append(f"active kernel reuses reserved ID {prefix}:{local}")
		if not active_ids and not reserved_ids:
			continue
		high_water = max(active_ids | reserved_ids)
		unclassified = set(range(1, high_water + 1)) - active_ids - reserved_ids
		if unclassified:
			errors.append(
				f"unclassified kernel ID gap {prefix}:"
				f"{format_local_ranges(unclassified)} below high-water {high_water}"
			)
	return errors


def load_typed_ids(
	root: Path, fixed: dict[str, FixedKernel]
) -> tuple[dict[str, str], list[str]]:
	by_packed = {(row.prefix, row.local_id): row.name for row in fixed.values()}
	typed: dict[str, str] = {}
	errors: list[str] = []
	for path in kernel_id_sources(root):
		for match in KERNEL_ID_RE.finditer(path.read_text()):
			identifier = match.group(1)
			packed = (match.group(2), int(match.group(3)))
			name = by_packed.get(packed)
			if name is None:
				errors.append(
					f"typed kernel ID {identifier} references missing fixed row "
					f"{packed[0]}:{packed[1]} in {path.relative_to(root)}"
				)
				continue
			if identifier != name:
				errors.append(
					f"typed kernel ID {identifier} resolves to fixed row {name} "
					f"at {packed[0]}:{packed[1]} in {path.relative_to(root)}"
				)
				continue
			if identifier in typed and typed[identifier] != name:
				errors.append(
					f"typed kernel ID {identifier} maps to both {typed[identifier]} and {name}"
				)
			else:
				typed[identifier] = name
	return typed, errors


def load_schema_kernel_references(
	root: Path, owner: str = "core"
) -> dict[str, list[str]]:
	references: dict[str, list[str]] = defaultdict(list)
	for path in sorted((root / "tools/fnAutogen/schema").rglob("*.toml")):
		data = tomllib.loads(path.read_text())
		if data.get("owner", "core") != owner:
			continue
		for operation in data.get("ops", []):
			for field, name in operation.items():
				if not field.startswith("kernel_") or not isinstance(name, str):
					continue
				references[name].append(
					f"{path.relative_to(root)}:{operation.get('name', '<unnamed>')}:{field}"
				)
	return dict(references)


def is_declaration_source(path: Path, root: Path) -> bool:
	return path in set(registry_sources(root) + kernel_id_sources(root)) or (
		"kernelregistry" in path.name.lower()
	)


def load_production_references(
	root: Path,
	known_names: set[str],
	typed_ids: dict[str, str],
) -> dict[str, set[Path]]:
	references: dict[str, set[Path]] = defaultdict(set)
	for relative_root in PRODUCTION_ROOTS:
		base = root / relative_root
		if not base.exists():
			continue
		for path in base.rglob("*"):
			if path.suffix not in SOURCE_SUFFIXES or is_declaration_source(path, root):
				continue
			text = path.read_text(errors="ignore")
			for literal in STRING_RE.findall(text):
				if literal in known_names:
					references[literal].add(path.relative_to(root))
			for identifier in TYPED_ID_USE_RE.findall(text):
				if name := typed_ids.get(identifier):
					references[name].add(path.relative_to(root))
	return dict(references)


def output_path(build_dir: Path, name: str) -> Path:
	return build_dir / "spirv" / f"{name}.spv"


def sdk_environment_output_path(build_dir: Path, name: str) -> Path:
	return build_dir / "sdk/cpp/spirv/environment" / f"{name}.spv"


def print_group(title: str, values: list[str]) -> None:
	print(f"\n=== {title} ({len(values)}) ===")
	for value in values:
		print(f"  {value}")


def parse_args(argv: list[str]) -> argparse.Namespace:
	parser = argparse.ArgumentParser(description=__doc__)
	parser.add_argument(
		"--build-dir", type=Path, default=Path("build/release"),
		help="configured build directory containing spirv_shader_list.txt",
	)
	parser.add_argument(
		"--strict", action="store_true",
		help="also fail for shaders/fixed IDs without production references",
	)
	return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
	args = parse_args(argv or sys.argv[1:])
	build_dir = args.build_dir
	if not build_dir.is_absolute():
		build_dir = REPO_ROOT / build_dir

	manifest_order, errors = load_manifest(build_dir)
	manifest = set(manifest_order)
	sdk_manifest_order, sdk_manifest_errors = load_manifest_file(
		build_dir / SDK_ENVIRONMENT_MANIFEST
	)
	errors.extend(sdk_manifest_errors)
	sdk_manifest = set(sdk_manifest_order)
	fixed, fixed_errors = load_fixed_registry(REPO_ROOT)
	errors.extend(fixed_errors)
	matmul_variants, matmul_variant_errors = load_matmul_variants(REPO_ROOT)
	errors.extend(matmul_variant_errors)
	reserved, reserved_errors = load_reserved_registry(REPO_ROOT)
	errors.extend(reserved_errors)
	errors.extend(validate_reserved_coverage(fixed, reserved))
	typed_ids, typed_errors = load_typed_ids(REPO_ROOT, fixed)
	errors.extend(typed_errors)
	schema_refs = load_schema_kernel_references(REPO_ROOT, "core")
	sdk_schema_refs = load_schema_kernel_references(REPO_ROOT, "sdk")
	production_refs = load_production_references(
		REPO_ROOT,
		manifest | sdk_manifest | set(fixed) | set(matmul_variants),
		typed_ids,
	)

	fixed_missing_build = sorted(set(fixed) - manifest)
	matmul_variant_missing_build = sorted(set(matmul_variants) - manifest)
	missing_outputs = sorted(
		name for name in manifest if not output_path(build_dir, name).is_file()
	)
	fixed_without_production = sorted(set(fixed) - set(production_refs))
	compiled_without_identity_or_production = sorted(
		manifest - set(fixed) - set(production_refs)
	)
	schema_compiled_without_production = sorted(
		(manifest & set(schema_refs)) - set(production_refs)
	)
	schema_missing_build = sorted(set(schema_refs) - manifest)
	sdk_schema_missing_build = sorted(set(sdk_schema_refs) - sdk_manifest)
	sdk_compiled_without_schema = sorted(sdk_manifest - set(sdk_schema_refs))
	sdk_missing_outputs = sorted(
		name for name in sdk_manifest
		if not sdk_environment_output_path(build_dir, name).is_file()
	)
	sdk_compiled_without_production = sorted(sdk_manifest - set(production_refs))
	compiled_with_matmul_variant_id_only = sorted(
		(manifest & set(matmul_variants)) - set(fixed)
	)
	compiled_without_stable_identity = sorted(
		manifest - set(fixed) - set(matmul_variants)
	)

	print(
		f"manifest={len(manifest_order)} unique={len(manifest)} "
		f"fixed={len(fixed)} reserved={sum(len(ids) for ids in reserved.values())} "
		f"matmul_variants={len(matmul_variants)} "
		f"variant_only={len(compiled_with_matmul_variant_id_only)} "
		f"production_refs={len(production_refs)} "
		f"schema_kernel_refs={len(schema_refs)} "
		f"sdk_manifest={len(sdk_manifest_order)} "
		f"sdk_schema_kernel_refs={len(sdk_schema_refs)}"
	)
	print_group("ERRORS", sorted(errors))
	print_group("FIXED_MISSING_BUILD", fixed_missing_build)
	print_group("MATMUL_VARIANT_MISSING_BUILD", matmul_variant_missing_build)
	print_group("MISSING_SPIRV_OUTPUT", missing_outputs)
	print_group("FIXED_WITHOUT_PRODUCTION_REFERENCE", fixed_without_production)
	print_group(
		"COMPILED_WITHOUT_IDENTITY_OR_PRODUCTION_REFERENCE",
		compiled_without_identity_or_production,
	)
	print_group(
		"SCHEMA_COMPILED_WITHOUT_PRODUCTION_REFERENCE",
		schema_compiled_without_production,
	)
	print_group(
		"SCHEMA_MISSING_BUILD",
		schema_missing_build,
	)
	print_group("SDK_SCHEMA_MISSING_BUILD", sdk_schema_missing_build)
	print_group("SDK_COMPILED_WITHOUT_SCHEMA", sdk_compiled_without_schema)
	print_group("SDK_MISSING_SPIRV_OUTPUT", sdk_missing_outputs)
	print_group(
		"SDK_COMPILED_WITHOUT_PRODUCTION_REFERENCE",
		sdk_compiled_without_production,
	)
	print_group(
		"COMPILED_WITH_MATMUL_VARIANT_ID_ONLY",
		compiled_with_matmul_variant_id_only,
	)
	print_group(
		"COMPILED_WITHOUT_STABLE_IDENTITY",
		compiled_without_stable_identity,
	)

	hard_errors = bool(
		errors or fixed_missing_build or matmul_variant_missing_build
		or missing_outputs or schema_missing_build
		or sdk_schema_missing_build or sdk_compiled_without_schema
		or sdk_missing_outputs
		or compiled_without_stable_identity
	)
	if args.strict:
		hard_errors = hard_errors or bool(
			fixed_without_production or compiled_without_identity_or_production
			or sdk_compiled_without_production
		)
	return 1 if hard_errors else 0


if __name__ == "__main__":
	raise SystemExit(main())
