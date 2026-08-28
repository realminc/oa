#!/usr/bin/env python3
"""Audit and ratchet OA's dependency on the hosted C++ standard library.

The checked policy records the exact standard namespace symbols, standard
headers, exception tokens, and RTTI tokens present in every shipped or vendored
C/C++ source file. A check permits removal but rejects a new token, a count
increase, or a dependency in a newly added file. This makes the migration
monotonic without pretending the current tree is already standard-library free.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent.parent
POLICY_PATH = Path("tools/diagnostics/oaStdPolicy.json")
SCHEMA = "oa.std_dependency_policy.v1"
SOURCE_SUFFIXES = {".c", ".cc", ".cpp", ".cxx", ".h", ".hh", ".hpp", ".inc", ".inl"}
TRACKED_ROOTS = (
	Path("source/cpp/include/oa"),
	Path("source/cpp/lib/oa"),
	Path("source/cpp/thirdparty"),
)

STD_SYMBOL_RE = re.compile(r"\bstd::([A-Za-z_][A-Za-z0-9_]*)")
EXCEPTION_RE = re.compile(r"\b(throw|try|catch)\b")
RTTI_RE = re.compile(r"\b(dynamic_cast|typeid)\b")
INCLUDE_RE = re.compile(r"^[ \t]*#[ \t]*include[ \t]*<([^>]+)>", re.MULTILINE)

# C++20 library headers. C spellings such as <stdint.h> remain an explicit C
# runtime boundary; C++ wrapper spellings such as <cstdint> are tracked here.
STD_HEADERS = {
	"algorithm", "any", "array", "atomic", "barrier", "bit", "bitset",
	"cassert", "cctype", "cerrno", "cfenv", "cfloat", "charconv", "chrono",
	"cinttypes", "ciso646", "climits", "clocale", "cmath", "codecvt",
	"compare", "complex", "concepts", "condition_variable", "coroutine",
	"csetjmp", "csignal", "cstdarg", "cstddef", "cstdint", "cstdio",
	"cstdlib", "cstring", "ctime", "cuchar", "cwchar", "cwctype", "deque",
	"exception", "execution", "filesystem", "format", "forward_list",
	"fstream", "functional", "future", "initializer_list", "iomanip", "ios",
	"iosfwd", "iostream", "istream", "iterator", "latch", "limits", "list",
	"locale", "map", "memory", "memory_resource", "mutex", "new", "numbers",
	"numeric", "optional", "ostream", "queue", "random", "ranges", "ratio",
	"regex", "scoped_allocator", "semaphore", "set", "shared_mutex",
	"source_location", "span", "sstream", "stack", "stdexcept", "stop_token",
	"streambuf", "string", "string_view", "strstream", "syncstream",
	"system_error", "thread", "tuple", "type_traits", "typeindex", "typeinfo",
	"unordered_map", "unordered_set", "utility", "valarray", "variant", "vector",
	"version",
}


def stripCommentsAndLiterals(text: str) -> str:
	"""Replace comments and quoted literals while preserving lines and offsets."""

	result = list(text)
	index = 0
	length = len(text)
	state = "code"
	quote = ""
	rawEnd = ""
	while index < length:
		if state == "code":
			if text.startswith("//", index):
				result[index] = result[index + 1] = " "
				index += 2
				state = "line"
				continue
			if text.startswith("/*", index):
				result[index] = result[index + 1] = " "
				index += 2
				state = "block"
				continue
			if text.startswith('R"', index):
				delimiterEnd = text.find("(", index + 2, min(length, index + 19))
				if delimiterEnd != -1:
					delimiter = text[index + 2:delimiterEnd]
					rawEnd = ")" + delimiter + '"'
					for position in range(index, delimiterEnd + 1):
						result[position] = " "
					index = delimiterEnd + 1
					state = "raw"
					continue
			if text[index] in {'"', "'"}:
				quote = text[index]
				result[index] = " "
				index += 1
				state = "quote"
				continue
			index += 1
			continue

		if state == "line":
			if text[index] == "\n":
				state = "code"
			else:
				result[index] = " "
			index += 1
			continue

		if state == "block":
			if text.startswith("*/", index):
				result[index] = result[index + 1] = " "
				index += 2
				state = "code"
			else:
				if text[index] != "\n":
					result[index] = " "
				index += 1
			continue

		if state == "raw":
			if text.startswith(rawEnd, index):
				for position in range(index, index + len(rawEnd)):
					result[position] = " "
				index += len(rawEnd)
				state = "code"
			else:
				if text[index] != "\n":
					result[index] = " "
				index += 1
			continue

		if text[index] == "\\" and index + 1 < length:
			result[index] = " "
			if text[index + 1] != "\n":
				result[index + 1] = " "
			index += 2
		elif text[index] == quote:
			result[index] = " "
			index += 1
			state = "code"
		else:
			if text[index] != "\n":
				result[index] = " "
			index += 1

	return "".join(result)


def sourcePaths(root: Path) -> list[Path]:
	paths: list[Path] = []
	for relativeRoot in TRACKED_ROOTS:
		absoluteRoot = root / relativeRoot
		if not absoluteRoot.exists():
			continue
		for path in absoluteRoot.rglob("*"):
			if path.is_file() and path.suffix in SOURCE_SUFFIXES:
				paths.append(path)
	return sorted(set(paths))


def classifyPath(relative: Path) -> list[str]:
	text = relative.as_posix()
	scopes = ["closure"]
	if text.startswith("source/cpp/include/oa/"):
		scopes.extend(("owned", "public"))
	elif text.startswith("source/cpp/lib/oa/"):
		scopes.append("owned")
	else:
		scopes.append("vendor")
	if text.startswith("source/cpp/include/oa/core/std/"):
		scopes.append("foundation")
	if (
		text.startswith("source/cpp/thirdparty/")
		or text.startswith("source/cpp/lib/oa/runtime/vma/")
	):
		scopes.append("vendor")
	if "/gen/" in text or ".gen." in text:
		scopes.append("generated")
	return sorted(set(scopes))


def countFile(path: Path, root: Path) -> dict[str, object]:
	text = path.read_text(encoding="utf-8", errors="replace")
	code = stripCommentsAndLiterals(text)
	stdSymbols = Counter(STD_SYMBOL_RE.findall(code))
	stdIncludes = Counter(
		header for header in INCLUDE_RE.findall(code) if header in STD_HEADERS
	)
	exceptions = Counter(EXCEPTION_RE.findall(code))
	rtti = Counter(RTTI_RE.findall(code))
	return {
		"scopes": classifyPath(path.relative_to(root)),
		"stdSymbols": dict(sorted(stdSymbols.items())),
		"stdIncludes": dict(sorted(stdIncludes.items())),
		"exceptions": dict(sorted(exceptions.items())),
		"rtti": dict(sorted(rtti.items())),
	}


def totalCounter(files: dict[str, dict[str, object]], metric: str, scope: str) -> Counter[str]:
	total: Counter[str] = Counter()
	for record in files.values():
		if scope in record["scopes"]:
			total.update(record[metric])
	return total


def buildReport(root: Path) -> dict[str, object]:
	files = {
		str(path.relative_to(root)): countFile(path, root)
		for path in sourcePaths(root)
	}
	summary: dict[str, object] = {}
	for scope in ("closure", "owned", "public", "foundation", "vendor", "generated"):
		metrics = {
			metric: dict(sorted(totalCounter(files, metric, scope).items()))
			for metric in ("stdSymbols", "stdIncludes", "exceptions", "rtti")
		}
		summary[scope] = {
			"files": sum(scope in record["scopes"] for record in files.values()),
			"totals": {name: sum(values.values()) for name, values in metrics.items()},
			"symbols": metrics,
		}
	return {
		"schema": SCHEMA,
		"trackedRoots": [str(path) for path in TRACKED_ROOTS],
		"summary": summary,
		"files": files,
	}


def buildPolicy(report: dict[str, object]) -> dict[str, object]:
	files: dict[str, object] = {}
	for path, record in report["files"].items():
		metrics = {
			metric: record[metric]
			for metric in ("stdSymbols", "stdIncludes", "exceptions", "rtti")
			if record[metric]
		}
		if metrics:
			files[path] = metrics
	return {
		"schema": SCHEMA,
		"trackedRoots": report["trackedRoots"],
		"files": files,
	}


def compareCounters(
	path: str,
	metric: str,
	current: dict[str, int],
	allowed: dict[str, int],
) -> list[str]:
	errors: list[str] = []
	for name, count in current.items():
		limit = allowed.get(name, 0)
		if count > limit:
			errors.append(f"{path}: {metric} {name} increased {limit} -> {count}")
	return errors


def checkPolicy(report: dict[str, object], policy: dict[str, object]) -> list[str]:
	if policy.get("schema") != SCHEMA:
		return [f"policy schema must be {SCHEMA}"]
	errors: list[str] = []
	allowedFiles = policy.get("files", {})
	for path, current in report["files"].items():
		allowed = allowedFiles.get(path, {})
		for metric in ("stdSymbols", "stdIncludes", "exceptions", "rtti"):
			errors.extend(compareCounters(
				path,
				metric,
				current.get(metric, {}),
				allowed.get(metric, {}),
			))
	return errors


def printSummary(report: dict[str, object]) -> None:
	for scope, record in report["summary"].items():
		totals = record["totals"]
		print(
			f"{scope}: files={record['files']} stdSymbols={totals['stdSymbols']} "
			f"stdIncludes={totals['stdIncludes']} exceptions={totals['exceptions']} "
			f"rtti={totals['rtti']}"
		)


def parseArgs() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description=__doc__)
	parser.add_argument("--root", type=Path, default=REPO_ROOT)
	parser.add_argument("--json", type=Path, help="Write the current report as JSON")
	parser.add_argument("--write-policy", action="store_true",
		help="Replace the policy with the current monotonic baseline")
	parser.add_argument("--check", action="store_true",
		help="Reject any dependency increase relative to the policy")
	return parser.parse_args()


def main() -> int:
	args = parseArgs()
	root = args.root.resolve()
	report = buildReport(root)
	printSummary(report)

	if args.json:
		args.json.parent.mkdir(parents=True, exist_ok=True)
		args.json.write_text(json.dumps(report, indent=2) + "\n")

	policyPath = root / POLICY_PATH
	if args.write_policy:
		policyPath.write_text(json.dumps(buildPolicy(report), indent=2) + "\n")
		print(f"wrote {policyPath.relative_to(root)}")

	if args.check:
		if not policyPath.exists():
			print(f"error: policy not found: {policyPath}", file=sys.stderr)
			return 2
		policy = json.loads(policyPath.read_text())
		errors = checkPolicy(report, policy)
		if errors:
			for error in errors:
				print(f"error: {error}", file=sys.stderr)
			return 1
		print("OA standard-library dependency ratchet: PASS")
	return 0


if __name__ == "__main__":
	raise SystemExit(main())
