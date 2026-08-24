#!/usr/bin/env python3
"""Gate complete schema-v2 ownership across contracts, lowering, and Python."""

from __future__ import annotations

import importlib.util
import re
import sys
import tomllib
from pathlib import Path


TOOL_DIR = Path(__file__).resolve().parent
REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
	sys.path.insert(0, str(REPO_ROOT))

from tools.gen.fn import generate as fnGenerator
from tools.gen.fn.config import DOMAIN_NAMESPACE
from tools.gen.fn.layout import inferDomain


CPP_SUFFIXES = frozenset({".cc", ".cpp", ".h", ".hpp", ".inl"})
CPP_SOURCE_ROOTS = (
 Path("source/cpp/include"),
 Path("source/cpp/lib"),
 Path("sdk/cpp"),
 Path("extensions"),
 Path("test/cpp"),
)
COMPRESSED_FN_NAMESPACE_RE = re.compile(
 r"^[ \t]*namespace[ \t]+oa::Fn[A-Za-z0-9_]*\b",
 re.MULTILINE,
)


def compressedFnNamespaceDeclarations(
 repoRoot: Path = REPO_ROOT,
) -> list[str]:
	"""Return C++ declarations which compress the oa/Fn namespace boundary."""
	violations: list[str] = []
	for relativeRoot in CPP_SOURCE_ROOTS:
		root = repoRoot / relativeRoot
		if not root.is_dir():
			continue
		for path in sorted(root.rglob("*")):
			if not path.is_file() or path.suffix not in CPP_SUFFIXES:
				continue
			text = path.read_text(encoding="utf-8", errors="replace")
			for match in COMPRESSED_FN_NAMESPACE_RE.finditer(text):
				line = text.count("\n", 0, match.start()) + 1
				violations.append(
				 f"{path.relative_to(repoRoot).as_posix()}:{line}"
				)
	return violations


def implementationText() -> str:
	parts: list[str] = []
	for root in (
	 REPO_ROOT / "source" / "cpp" / "lib",
	 REPO_ROOT / "sdk" / "cpp" / "tutorials",
	 REPO_ROOT / "sdk" / "cpp" / "lib",
	):
		if not root.is_dir():
			continue
		for path in sorted(root.rglob("*")):
			if (
			 path.is_file()
			 and path.suffix in {".cpp", ".h", ".inl"}
			 and path.name != "opRegistry.gen.h"
			):
				parts.append(path.read_text(encoding="utf-8", errors="replace"))
	return "\n".join(parts)


def main() -> int:
	implementation = implementationText()
	surfacePath = REPO_ROOT / "source/py/oa/_schemaSurface.py"
	surfaceSpec = importlib.util.spec_from_file_location(
	 "_oa_schemaSurface_coverage", surfacePath
	)
	if surfaceSpec is None or surfaceSpec.loader is None:
		print(f"ERROR: cannot load generated Python surface {surfacePath}")
		return 1
	surfaceModule = importlib.util.module_from_spec(surfaceSpec)
	surfaceSpec.loader.exec_module(surfaceModule)
	surfaceExports = surfaceModule.SCHEMA_NAMESPACE_EXPORTS
	errors: list[str] = []
	for location in compressedFnNamespaceDeclarations():
		errors.append(
		 f"{location}: split the oa and Fn* namespace declarations"
		)
	contracted = 0
	lowered = 0
	publicNonvoid = 0
	pythonOwned = 0
	pythonExposed = 0
	reverse = 0
	autogradOwned = 0

	schemaPaths = sorted((TOOL_DIR / "schema").rglob("*.toml"))
	if not schemaPaths:
		print(f"ERROR: no schemas discovered under {TOOL_DIR / 'schema'}")
		return 1

	for schemaPath in schemaPaths:
		with schemaPath.open("rb") as stream:
			data = tomllib.load(stream)
		domain = inferDomain(schemaPath)
		namespace = data.get(
		 "namespace",
		 DOMAIN_NAMESPACE.get(domain, "oa::FnMatrix"),
		)
		for operation in data.get("ops", []):
			operation.setdefault("surface", data.get("surface"))
			fnGenerator.applySchemaDefaults(data, operation)
			context = (
			 f"{schemaPath.relative_to(REPO_ROOT)}:{operation['name']}"
			)

			if "contract" in operation:
				contracted += 1
				identity = (
				 "oa::detail::opRegistry::"
				 f"{fnGenerator.namespaceLeaf(namespace)}::{operation['name']}"
				)
				if identity not in implementation:
					errors.append(
					 f"{context}: no lowering references {identity}"
					)
				else:
					lowered += 1

			if (
			 operation.get("surface")
			 in ("public_operation", "stable_composite")
			 and operation.get("api_return", "oa::Matrix") != "void"
			):
				publicNonvoid += 1
				if "python" in operation:
					pythonOwned += 1
					pythonName = operation["python"].get(
					 "name", operation["name"]
					)
					mapping = (pythonName, pythonName)
					domainExports = surfaceExports.get(
					 fnGenerator.namespaceLeaf(namespace), {}
					).get(domain.lower(), ())
					if mapping in domainExports:
						pythonExposed += 1
					else:
						errors.append(
						 f"{context}: generated Python exposure is missing "
						 f"{namespace}.{pythonName}"
						)
				else:
					errors.append(
					 f"{context}: non-void public operation has no "
					 "schema-owned Python binding"
					)

			if (
			 operation.get("contract", {}).get("differentiation")
			 == "reverse"
			):
				reverse += 1
				if "autograd" in operation:
					autogradOwned += 1
				else:
					errors.append(
					 f"{context}: reverse differentiation has no "
					 "schema-owned autograd policy"
					)

	print(
	 "schema coverage: "
	 f"contracts {lowered}/{contracted}, "
	 f"Python {pythonOwned}/{publicNonvoid}, "
	 f"exposure {pythonExposed}/{publicNonvoid}, "
	 f"autograd {autogradOwned}/{reverse}"
	)
	for error in errors:
		print(f"ERROR: {error}")
	print("PASS" if not errors else "FAIL")
	return 0 if not errors else 1


if __name__ == "__main__":
	raise SystemExit(main())
