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
REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
	sys.path.insert(0, str(REPO_ROOT))

from tools.gen.fn.config import DOMAIN_NAMESPACE
from tools.gen.fn.layout import inferDomain
from tools.gen.fn.generate import namespaceLeaf


FN_NAMESPACE_RE = re.compile(
 r"\bnamespace\s+(?:oa::)?(Fn[A-Za-z0-9_]*)\s*\{"
)
SOURCE_SUFFIXES = frozenset({".c", ".cc", ".cpp", ".cxx"})
DEFAULT_COMPILE_DATABASES = (
 Path("build/release/compile_commands.json"),
 Path("build/python/compile_commands.json"),
)


def findCompileDatabase(requested: Path | None) -> Path:
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


def publicFunctionHeaders() -> list[tuple[Path, str]]:
	"""Return public headers which directly declare an oa::Fn namespace."""
	headers: list[tuple[Path, str]] = []
	for includeRoot in (
	 REPO_ROOT / "source" / "cpp" / "include",
	 REPO_ROOT / "extensions" / "public",
	):
		if not includeRoot.is_dir():
			continue
		for path in sorted(includeRoot.rglob("*.h")):
			text = path.read_text(encoding="utf-8", errors="replace")
			if FN_NAMESPACE_RE.search(text):
				headers.append((includeRoot, path.relative_to(includeRoot).as_posix()))
	return headers


def _commandArguments(row: dict) -> list[str]:
	arguments = row.get("arguments")
	if arguments:
		return list(arguments)
	command = row.get("command")
	if not command:
		raise ValueError("compile database row has neither arguments nor command")
	return shlex.split(command)


def clangAstCommand(compileDatabase: Path) -> list[str]:
	rows = json.loads(compileDatabase.read_text(encoding="utf-8"))
	row = next(
	 (
	  candidate
	  for candidate in rows
	  if Path(candidate.get("file", "")).suffix in SOURCE_SUFFIXES
	 ),
	 None,
	)
	if row is None:
		raise ValueError(f"no C++ compile command in {compileDatabase}")

	arguments = _commandArguments(row)
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


def decodeJsonStream(text: str) -> list[dict]:
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


def _functionName(node: dict) -> str | None:
	if node.get("kind") == "FunctionDecl":
		return node.get("name")
	if node.get("kind") == "FunctionTemplateDecl":
		for child in node.get("inner", ()):
			if child.get("kind") == "FunctionDecl":
				return child.get("name")
	return None


def publicIdentities(compileDatabase: Path) -> dict[str, set[str]]:
	headers = publicFunctionHeaders()
	source = "".join(f"#include <{include}>\n" for _, include in headers)
	process = subprocess.run(
	 clangAstCommand(compileDatabase),
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
	for document in decodeJsonStream(process.stdout):
		namespace = document.get("name", "")
		if (
		 document.get("kind") != "NamespaceDecl"
		 or not namespace.startswith("Fn")
		):
			continue
		for child in document.get("inner", ()):
			name = _functionName(child)
			if name:
				identities[namespace].add(name)
	return identities


def schemaIdentities() -> dict[str, set[str]]:
	identities: dict[str, set[str]] = defaultdict(set)
	for path in sorted((TOOL_DIR / "schema").rglob("*.toml")):
		with path.open("rb") as stream:
			data = tomllib.load(stream)
		domain = inferDomain(path)
		namespace = data.get(
		 "namespace",
		 DOMAIN_NAMESPACE.get(domain, "oa::FnMatrix"),
		)
		for operation in data.get("ops", ()):
			identities[namespaceLeaf(namespace)].add(operation["name"])
	return identities


def parseArgs() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description=__doc__)
	parser.add_argument(
	 "--compile-commands", dest="compileCommands",
	 type=Path,
	 help="configured compile_commands.json (auto-detected by default)",
	)
	return parser.parse_args()


def main() -> int:
	args = parseArgs()
	try:
		compileDatabase = findCompileDatabase(args.compileCommands)
		public = publicIdentities(compileDatabase)
		schemas = schemaIdentities()
	except (OSError, ValueError, json.JSONDecodeError, tomllib.TOMLDecodeError) as error:
		print(f"ERROR: {error}")
		return 1

	missingCount = 0
	print(
	 "public operation coverage: "
	 f"{sum(len(public[namespace] & schemas[namespace]) for namespace in public)}"
	 f"/{sum(map(len, public.values()))} identities"
	)
	for namespace in sorted(public):
		missing = sorted(public[namespace] - schemas[namespace])
		if not missing:
			continue
		missingCount += len(missing)
		print(f"ERROR: {namespace} missing {len(missing)} schema identities:")
		print(f"  {', '.join(missing)}")

	schemaOnly = sum(
	 len(names - public.get(namespace, set()))
	 for namespace, names in schemas.items()
	)
	print(
	 f"schema-only internal/session/kernel identities: {schemaOnly}"
	)
	print("PASS" if missingCount == 0 else "FAIL")
	return 0 if missingCount == 0 else 1


if __name__ == "__main__":
	raise SystemExit(main())
