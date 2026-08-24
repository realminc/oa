#!/usr/bin/env python3
"""Compare live public ``oa::Fn*`` declarations with the schema inventory.

Unlike ``check_schema_coverage.py``, this gate derives its denominator from the
Clang AST of the public headers.  It therefore catches a handwritten public
operation that never entered a schema at all.

The configured build supplies the exact compiler, defines, and include paths.
Configure OA with ``CMAKE_EXPORT_COMPILE_COMMANDS=ON`` before running this
check (the top-level CMake project enables it by default).
"""

from __future__ import annotations

import argparse
import json
import re
import shlex
import subprocess
import sys
import tomllib
from collections import defaultdict
from pathlib import Path


TOOL_DIR = Path(__file__).resolve().parent
REPO_ROOT = TOOL_DIR.parents[1]
if str(TOOL_DIR) not in sys.path:
	sys.path.insert(0, str(TOOL_DIR))

from oafnautogen_lib.config import DOMAIN_NAMESPACE
from oafnautogen_lib.layout import infer_domain
from oafnautogen import namespace_leaf


FN_NAMESPACE_RE = re.compile(
	r"\bnamespace\s+(?:oa::)?(Fn[A-Za-z0-9_]*)\s*\{"
)
SOURCE_SUFFIXES = frozenset({".c", ".cc", ".cpp", ".cxx"})
DEFAULT_COMPILE_DATABASES = (
	Path("build/release/compile_commands.json"),
	Path("build/python/compile_commands.json"),
)


def find_compile_database(requested: Path | None) -> Path:
	if requested is not None:
		path = requested if requested.is_absolute() else REPO_ROOT / requested
		if not path.is_file():
			raise ValueError(f"compile database does not exist: {path}")
		return path
	for relative in DEFAULT_COMPILE_DATABASES:
		path = REPO_ROOT / relative
		if path.is_file():
			return path
	raise ValueError(
		"no compile_commands.json found; configure OA first or pass "
		"--compile-commands"
	)


def public_function_headers() -> list[tuple[Path, str]]:
	"""Return public headers which directly declare an oa::Fn namespace."""
	headers: list[tuple[Path, str]] = []
	for include_root in (
		REPO_ROOT / "source" / "cpp" / "include",
		REPO_ROOT / "extensions" / "public",
	):
		if not include_root.is_dir():
			continue
		for path in sorted(include_root.rglob("*.h")):
			text = path.read_text(encoding="utf-8", errors="replace")
			if FN_NAMESPACE_RE.search(text):
				headers.append((include_root, path.relative_to(include_root).as_posix()))
	return headers


def _command_arguments(row: dict) -> list[str]:
	arguments = row.get("arguments")
	if arguments:
		return list(arguments)
	command = row.get("command")
	if not command:
		raise ValueError("compile database row has neither arguments nor command")
	return shlex.split(command)


def clang_ast_command(compile_database: Path) -> list[str]:
	rows = json.loads(compile_database.read_text(encoding="utf-8"))
	row = next(
		(
			candidate
			for candidate in rows
			if Path(candidate.get("file", "")).suffix in SOURCE_SUFFIXES
		),
		None,
	)
	if row is None:
		raise ValueError(f"no C++ compile command in {compile_database}")

	arguments = _command_arguments(row)
	command = [arguments[0]]
	index = 1
	while index < len(arguments):
		argument = arguments[index]
		if argument in ("-o", "--output"):
			index += 2
			continue
		if argument in ("-c", "--compile"):
			index += 1
			continue
		if (
			not argument.startswith("-")
			and Path(argument).suffix in SOURCE_SUFFIXES
		):
			index += 1
			continue
		command.append(argument)
		index += 1

	command.extend(
		(
			"-Xclang",
			"-ast-dump=json",
			"-Xclang",
			"-ast-dump-filter=Fn",
			"-fsyntax-only",
			"-x",
			"c++",
			"-",
		)
	)
	return command


def decode_json_stream(text: str) -> list[dict]:
	"""Decode Clang's concatenated JSON documents."""
	documents: list[dict] = []
	decoder = json.JSONDecoder()
	offset = 0
	while True:
		while offset < len(text) and text[offset].isspace():
			offset += 1
		if offset == len(text):
			return documents
		document, offset = decoder.raw_decode(text, offset)
		documents.append(document)


def _function_name(node: dict) -> str | None:
	if node.get("kind") == "FunctionDecl":
		return node.get("name")
	if node.get("kind") == "FunctionTemplateDecl":
		for child in node.get("inner", ()):
			if child.get("kind") == "FunctionDecl":
				return child.get("name")
	return None


def public_identities(compile_database: Path) -> dict[str, set[str]]:
	headers = public_function_headers()
	source = "".join(f"#include <{include}>\n" for _, include in headers)
	process = subprocess.run(
		clang_ast_command(compile_database),
		cwd=REPO_ROOT,
		input=source,
		text=True,
		capture_output=True,
		check=False,
	)
	if process.returncode != 0:
		raise ValueError(
			"public-header AST audit failed:\n"
			+ process.stderr.rstrip()
		)

	identities: dict[str, set[str]] = defaultdict(set)
	for document in decode_json_stream(process.stdout):
		namespace = document.get("name", "")
		if (
			document.get("kind") != "NamespaceDecl"
			or not namespace.startswith("Fn")
		):
			continue
		for child in document.get("inner", ()):
			name = _function_name(child)
			if name:
				identities[namespace].add(name)
	return identities


def schema_identities() -> dict[str, set[str]]:
	identities: dict[str, set[str]] = defaultdict(set)
	for path in sorted((TOOL_DIR / "schema").rglob("*.toml")):
		with path.open("rb") as stream:
			data = tomllib.load(stream)
		domain = infer_domain(path)
		namespace = data.get(
			"namespace",
			DOMAIN_NAMESPACE.get(domain, "oa::FnMatrix"),
		)
		for operation in data.get("ops", ()):
			identities[namespace_leaf(namespace)].add(operation["name"])
	return identities


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description=__doc__)
	parser.add_argument(
		"--compile-commands",
		type=Path,
		help="configured compile_commands.json (auto-detected by default)",
	)
	return parser.parse_args()


def main() -> int:
	args = parse_args()
	try:
		compile_database = find_compile_database(args.compile_commands)
		public = public_identities(compile_database)
		schemas = schema_identities()
	except (OSError, ValueError, json.JSONDecodeError, tomllib.TOMLDecodeError) as error:
		print(f"ERROR: {error}")
		return 1

	missing_count = 0
	print(
		"public operation coverage: "
		f"{sum(len(public[namespace] & schemas[namespace]) for namespace in public)}"
		f"/{sum(map(len, public.values()))} identities"
	)
	for namespace in sorted(public):
		missing = sorted(public[namespace] - schemas[namespace])
		if not missing:
			continue
		missing_count += len(missing)
		print(f"ERROR: {namespace} missing {len(missing)} schema identities:")
		print(f"  {', '.join(missing)}")

	schema_only = sum(
		len(names - public.get(namespace, set()))
		for namespace, names in schemas.items()
	)
	print(
		f"schema-only internal/session/kernel identities: {schema_only}"
	)
	print("PASS" if missing_count == 0 else "FAIL")
	return 0 if missing_count == 0 else 1


if __name__ == "__main__":
	raise SystemExit(main())
