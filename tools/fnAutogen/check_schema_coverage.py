#!/usr/bin/env python3
"""Gate complete schema-v2 ownership across contracts, lowering, and Python."""

from __future__ import annotations

import importlib.util
import re
import sys
import tomllib
from pathlib import Path


TOOL_DIR = Path(__file__).resolve().parent
REPO_ROOT = TOOL_DIR.parents[1]
if str(TOOL_DIR) not in sys.path:
	sys.path.insert(0, str(TOOL_DIR))

import oafnautogen
from oafnautogen_lib.config import DOMAIN_NAMESPACE
from oafnautogen_lib.layout import infer_domain


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


def compressed_fn_namespace_declarations(
	repo_root: Path = REPO_ROOT,
) -> list[str]:
	"""Return C++ declarations which compress the oa/Fn namespace boundary."""
	violations: list[str] = []
	for relative_root in CPP_SOURCE_ROOTS:
		root = repo_root / relative_root
		if not root.is_dir():
			continue
		for path in sorted(root.rglob("*")):
			if not path.is_file() or path.suffix not in CPP_SUFFIXES:
				continue
			text = path.read_text(encoding="utf-8", errors="replace")
			for match in COMPRESSED_FN_NAMESPACE_RE.finditer(text):
				line = text.count("\n", 0, match.start()) + 1
				violations.append(
					f"{path.relative_to(repo_root).as_posix()}:{line}"
				)
	return violations


def implementation_text() -> str:
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
	implementation = implementation_text()
	surface_path = REPO_ROOT / "source/py/oa/_schemaSurface.py"
	surface_spec = importlib.util.spec_from_file_location(
		"_oa_schemaSurface_coverage", surface_path
	)
	if surface_spec is None or surface_spec.loader is None:
		print(f"ERROR: cannot load generated Python surface {surface_path}")
		return 1
	surface_module = importlib.util.module_from_spec(surface_spec)
	surface_spec.loader.exec_module(surface_module)
	surface_exports = surface_module.SCHEMA_NAMESPACE_EXPORTS
	errors: list[str] = []
	for location in compressed_fn_namespace_declarations():
		errors.append(
			f"{location}: split the oa and Fn* namespace declarations"
		)
	contracted = 0
	lowered = 0
	public_nonvoid = 0
	python_owned = 0
	python_exposed = 0
	reverse = 0
	autograd_owned = 0

	schema_paths = sorted((TOOL_DIR / "schema").rglob("*.toml"))
	if not schema_paths:
		print(f"ERROR: no schemas discovered under {TOOL_DIR / 'schema'}")
		return 1

	for schema_path in schema_paths:
		with schema_path.open("rb") as stream:
			data = tomllib.load(stream)
		domain = infer_domain(schema_path)
		namespace = data.get(
			"namespace",
			DOMAIN_NAMESPACE.get(domain, "oa::FnMatrix"),
		)
		for operation in data.get("ops", []):
			operation.setdefault("surface", data.get("surface"))
			oafnautogen.apply_schema_defaults(data, operation)
			context = (
				f"{schema_path.relative_to(REPO_ROOT)}:{operation['name']}"
			)

			if "contract" in operation:
				contracted += 1
				identity = (
					"oa::detail::opRegistry::"
					f"{oafnautogen.namespace_leaf(namespace)}::{operation['name']}"
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
				public_nonvoid += 1
				if "python" in operation:
					python_owned += 1
					python_name = operation["python"].get(
						"name", operation["name"]
					)
					mapping = (python_name, python_name)
					domain_exports = surface_exports.get(
						oafnautogen.namespace_leaf(namespace), {}
					).get(domain.lower(), ())
					if mapping in domain_exports:
						python_exposed += 1
					else:
						errors.append(
							f"{context}: generated Python exposure is missing "
							f"{namespace}.{python_name}"
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
					autograd_owned += 1
				else:
					errors.append(
						f"{context}: reverse differentiation has no "
						"schema-owned autograd policy"
					)

	print(
		"schema coverage: "
		f"contracts {lowered}/{contracted}, "
		f"Python {python_owned}/{public_nonvoid}, "
		f"exposure {python_exposed}/{public_nonvoid}, "
		f"autograd {autograd_owned}/{reverse}"
	)
	for error in errors:
		print(f"ERROR: {error}")
	print("PASS" if not errors else "FAIL")
	return 0 if not errors else 1


if __name__ == "__main__":
	raise SystemExit(main())
