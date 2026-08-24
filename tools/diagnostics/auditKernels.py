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


REPO_ROOT = Path(__file__).resolve().parent.parent.parent
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
	localId: int
	category: str
	origin: str
	source: Path


def registrySources(root: Path) -> list[Path]:
	runtime = root / "source/cpp/lib/oa/runtime"
	paths = [runtime / "kernelRegistry.h"]
	paths.extend(sorted((runtime / "gen").glob("kernelRegistry*.inl")))
	paths.append(runtime / "gen" / "tileKernelRegistry.inc")
	return list(dict.fromkeys(path for path in paths if path.exists()))


def kernelIdSources(root: Path) -> list[Path]:
	public = root / "source/cpp/include/oa/runtime"
	return [
	 path for path in (
	  public / "computeKernel.h",
	  public / "gen" / "kernelIdsStandalone.inl",
	  public / "gen" / "computeKernelIds.inl",
	  public / "gen" / "tileComputeKernel.h",
	 ) if path.exists()
	]


def matmulVariantSources(root: Path) -> list[Path]:
	gemm = root / "source/cpp/lib/oa/runtime/gemm"
	return sorted((gemm / "gen").glob("tile*Variants.inc"))


def loadMatmulVariants(root: Path) -> tuple[dict[str, Path], list[str]]:
	variants: dict[str, Path] = {}
	errors: list[str] = []
	for path in matmulVariantSources(root):
		for match in MATMUL_VARIANT_ROW_RE.finditer(path.read_text()):
			identityName = match.group(1)
			kernelName = match.group(2)
			relative = path.relative_to(root)
			if identityName != kernelName:
				errors.append(
				 f"matmul variant identity {identityName} names kernel "
				 f"{kernelName} in {relative}"
				)
				continue
			if previous := variants.get(identityName):
				errors.append(
				 f"duplicate matmul variant identity {identityName}: "
				 f"{previous}, {relative}"
				)
				continue
			variants[identityName] = relative
	return variants, errors


def loadManifestFile(path: Path) -> tuple[list[str], list[str]]:
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


def loadManifest(buildDir: Path) -> tuple[list[str], list[str]]:
	return loadManifestFile(buildDir / MANIFEST_NAME)


def loadFixedRegistry(root: Path) -> tuple[dict[str, FixedKernel], list[str]]:
	byName: dict[str, FixedKernel] = {}
	byId: dict[tuple[str, int], FixedKernel] = {}
	errors: list[str] = []
	for path in registrySources(root):
		for match in REGISTRY_ROW_RE.finditer(path.read_text()):
			row = FixedKernel(
			 name=match.group(1),
			 prefix=match.group(2),
			 localId=int(match.group(3)),
			 category=match.group(4),
			 origin=match.group(5),
			 source=path.relative_to(root),
			)
			if previous := byName.get(row.name):
				errors.append(
				 f"duplicate fixed name {row.name}: {previous.source}, {row.source}"
				)
			else:
				byName[row.name] = row
			packed = (row.prefix, row.localId)
			if previous := byId.get(packed):
				errors.append(
				 f"duplicate fixed ID {row.prefix}:{row.localId}: "
				 f"{previous.name}, {row.name}"
				)
			else:
				byId[packed] = row
	return byName, errors


def loadReservedRegistry(root: Path) -> tuple[dict[str, set[int]], list[str]]:
	runtime = root / "source/cpp/lib/oa/runtime"
	paths = [runtime / "kernelRegistry.h", runtime / "gen" / "kernelReservations.inl"]
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


def formatLocalRanges(values: set[int]) -> str:
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


def validateReservedCoverage(
 fixed: dict[str, FixedKernel], reserved: dict[str, set[int]]
) -> list[str]:
	active: dict[str, set[int]] = defaultdict(set)
	for row in fixed.values():
		active[row.prefix].add(row.localId)
	errors: list[str] = []
	for prefix in sorted(set(active) | set(reserved)):
		activeIds = active.get(prefix, set())
		reservedIds = reserved.get(prefix, set())
		for local in sorted(activeIds & reservedIds):
			errors.append(f"active kernel reuses reserved ID {prefix}:{local}")
		if not activeIds and not reservedIds:
			continue
		highWater = max(activeIds | reservedIds)
		unclassified = set(range(1, highWater + 1)) - activeIds - reservedIds
		if unclassified:
			errors.append(
			 f"unclassified kernel ID gap {prefix}:"
			 f"{formatLocalRanges(unclassified)} below high-water {highWater}"
			)
	return errors


def loadTypedIds(
 root: Path, fixed: dict[str, FixedKernel]
) -> tuple[dict[str, str], list[str]]:
	byPacked = {(row.prefix, row.localId): row.name for row in fixed.values()}
	typed: dict[str, str] = {}
	errors: list[str] = []
	for path in kernelIdSources(root):
		for match in KERNEL_ID_RE.finditer(path.read_text()):
			identifier = match.group(1)
			packed = (match.group(2), int(match.group(3)))
			name = byPacked.get(packed)
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


def loadSchemaKernelReferences(
 root: Path, owner: str = "core"
) -> dict[str, list[str]]:
	references: dict[str, list[str]] = defaultdict(list)
	for path in sorted((root / "tools/gen/fn/schema").rglob("*.toml")):
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


def isDeclarationSource(path: Path, root: Path) -> bool:
	return path in set(registrySources(root) + kernelIdSources(root)) or (
	 "kernelregistry" in path.name.lower()
	)


def loadProductionReferences(
 root: Path,
 knownNames: set[str],
 typedIds: dict[str, str],
) -> dict[str, set[Path]]:
	references: dict[str, set[Path]] = defaultdict(set)
	for relativeRoot in PRODUCTION_ROOTS:
		base = root / relativeRoot
		if not base.exists():
			continue
		for path in base.rglob("*"):
			if path.suffix not in SOURCE_SUFFIXES or isDeclarationSource(path, root):
				continue
			text = path.read_text(errors="ignore")
			for literal in STRING_RE.findall(text):
				if literal in knownNames:
					references[literal].add(path.relative_to(root))
			for identifier in TYPED_ID_USE_RE.findall(text):
				if name := typedIds.get(identifier):
					references[name].add(path.relative_to(root))
	return dict(references)


def outputPath(buildDir: Path, name: str) -> Path:
	return buildDir / "spirv" / f"{name}.spv"


def sdkEnvironmentOutputPath(buildDir: Path, name: str) -> Path:
	return buildDir / "sdk/cpp/spirv/environment" / f"{name}.spv"


def printGroup(title: str, values: list[str]) -> None:
	print(f"\n=== {title} ({len(values)}) ===")
	for value in values:
		print(f"  {value}")


def parseArgs(argv: list[str]) -> argparse.Namespace:
	parser = argparse.ArgumentParser(description=__doc__)
	parser.add_argument(
	 "--build-dir", dest="buildDir", type=Path, default=Path("build/release"),
	 help="configured build directory containing spirv_shader_list.txt",
	)
	parser.add_argument(
	 "--strict", action="store_true",
	 help="also fail for shaders/fixed IDs without production references",
	)
	return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
	args = parseArgs(argv or sys.argv[1:])
	buildDir = args.buildDir
	if not buildDir.is_absolute():
		buildDir = REPO_ROOT / buildDir

	manifestOrder, errors = loadManifest(buildDir)
	manifest = set(manifestOrder)
	sdkManifestOrder, sdkManifestErrors = loadManifestFile(
	 buildDir / SDK_ENVIRONMENT_MANIFEST
	)
	errors.extend(sdkManifestErrors)
	sdkManifest = set(sdkManifestOrder)
	fixed, fixedErrors = loadFixedRegistry(REPO_ROOT)
	errors.extend(fixedErrors)
	matmulVariants, matmulVariantErrors = loadMatmulVariants(REPO_ROOT)
	errors.extend(matmulVariantErrors)
	reserved, reservedErrors = loadReservedRegistry(REPO_ROOT)
	errors.extend(reservedErrors)
	errors.extend(validateReservedCoverage(fixed, reserved))
	typedIds, typedErrors = loadTypedIds(REPO_ROOT, fixed)
	errors.extend(typedErrors)
	schemaRefs = loadSchemaKernelReferences(REPO_ROOT, "core")
	sdkSchemaRefs = loadSchemaKernelReferences(REPO_ROOT, "sdk")
	productionRefs = loadProductionReferences(
	 REPO_ROOT,
	 manifest | sdkManifest | set(fixed) | set(matmulVariants),
	 typedIds,
	)

	fixedMissingBuild = sorted(set(fixed) - manifest)
	matmulVariantMissingBuild = sorted(set(matmulVariants) - manifest)
	missingOutputs = sorted(
	 name for name in manifest if not outputPath(buildDir, name).is_file()
	)
	fixedWithoutProduction = sorted(set(fixed) - set(productionRefs))
	compiledWithoutIdentityOrProduction = sorted(
	 manifest - set(fixed) - set(productionRefs)
	)
	schemaCompiledWithoutProduction = sorted(
	 (manifest & set(schemaRefs)) - set(productionRefs)
	)
	schemaMissingBuild = sorted(set(schemaRefs) - manifest)
	sdkSchemaMissingBuild = sorted(set(sdkSchemaRefs) - sdkManifest)
	sdkCompiledWithoutSchema = sorted(sdkManifest - set(sdkSchemaRefs))
	sdkMissingOutputs = sorted(
	 name for name in sdkManifest
	 if not sdkEnvironmentOutputPath(buildDir, name).is_file()
	)
	sdkCompiledWithoutProduction = sorted(sdkManifest - set(productionRefs))
	compiledWithMatmulVariantIdOnly = sorted(
	 (manifest & set(matmulVariants)) - set(fixed)
	)
	compiledWithoutStableIdentity = sorted(
	 manifest - set(fixed) - set(matmulVariants)
	)

	print(
	 f"manifest={len(manifestOrder)} unique={len(manifest)} "
	 f"fixed={len(fixed)} reserved={sum(len(ids) for ids in reserved.values())} "
	 f"matmul_variants={len(matmulVariants)} "
	 f"variant_only={len(compiledWithMatmulVariantIdOnly)} "
	 f"production_refs={len(productionRefs)} "
	 f"schema_kernel_refs={len(schemaRefs)} "
	 f"sdk_manifest={len(sdkManifestOrder)} "
	 f"sdk_schema_kernel_refs={len(sdkSchemaRefs)}"
	)
	printGroup("ERRORS", sorted(errors))
	printGroup("FIXED_MISSING_BUILD", fixedMissingBuild)
	printGroup("MATMUL_VARIANT_MISSING_BUILD", matmulVariantMissingBuild)
	printGroup("MISSING_SPIRV_OUTPUT", missingOutputs)
	printGroup("FIXED_WITHOUT_PRODUCTION_REFERENCE", fixedWithoutProduction)
	printGroup(
	 "COMPILED_WITHOUT_IDENTITY_OR_PRODUCTION_REFERENCE",
	 compiledWithoutIdentityOrProduction,
	)
	printGroup(
	 "SCHEMA_COMPILED_WITHOUT_PRODUCTION_REFERENCE",
	 schemaCompiledWithoutProduction,
	)
	printGroup(
	 "SCHEMA_MISSING_BUILD",
	 schemaMissingBuild,
	)
	printGroup("SDK_SCHEMA_MISSING_BUILD", sdkSchemaMissingBuild)
	printGroup("SDK_COMPILED_WITHOUT_SCHEMA", sdkCompiledWithoutSchema)
	printGroup("SDK_MISSING_SPIRV_OUTPUT", sdkMissingOutputs)
	printGroup(
	 "SDK_COMPILED_WITHOUT_PRODUCTION_REFERENCE",
	 sdkCompiledWithoutProduction,
	)
	printGroup(
	 "COMPILED_WITH_MATMUL_VARIANT_ID_ONLY",
	 compiledWithMatmulVariantIdOnly,
	)
	printGroup(
	 "COMPILED_WITHOUT_STABLE_IDENTITY",
	 compiledWithoutStableIdentity,
	)

	hardErrors = bool(
	 errors or fixedMissingBuild or matmulVariantMissingBuild
	 or missingOutputs or schemaMissingBuild
	 or sdkSchemaMissingBuild or sdkCompiledWithoutSchema
	 or sdkMissingOutputs
	 or compiledWithoutStableIdentity
	)
	if args.strict:
		hardErrors = hardErrors or bool(
		 fixedWithoutProduction or compiledWithoutIdentityOrProduction
		 or sdkCompiledWithoutProduction
		)
	return 1 if hardErrors else 0


if __name__ == "__main__":
	raise SystemExit(main())
