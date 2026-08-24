#!/usr/bin/env python3
"""Generate the publishable OA example inventory from checked source markers."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Any


REPO = Path(__file__).resolve().parents[2]
MANIFEST = REPO / "sdk" / "examples.json"
OUTPUT = REPO / "docs" / "external" / "generated" / "examples.json"
CPP_REGISTRY = REPO / "sdk" / "cpp" / "examples" / "CMakeLists.txt"
PYTHON_REGISTRY = REPO / "test" / "py" / "examples" / "test_examples.py"
ID_PATTERN = re.compile(r"^[a-z][a-z0-9]*(?:-[a-z0-9]+)*$")


class ExampleError(ValueError):
    """A manifest or marked-source contract is invalid."""


def _require_string(owner: dict[str, Any], key: str, context: str) -> str:
    value = owner.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ExampleError(f"{context}: {key} must be a non-empty string")
    return value


def _require_strings(owner: dict[str, Any], key: str, context: str) -> list[str]:
    value = owner.get(key)
    if (
        not isinstance(value, list)
        or not value
        or any(not isinstance(item, str) or not item.strip() for item in value)
    ):
        raise ExampleError(f"{context}: {key} must be a non-empty string list")
    return value


def _source_path(relative: str, language: str, suffix: str, context: str) -> Path:
    relative_path = Path(relative)
    if relative_path.is_absolute() or ".." in relative_path.parts:
        raise ExampleError(f"{context}: unsafe source path {relative}")
    expected_root = ("sdk", language, "examples")
    if relative_path.suffix != suffix or relative_path.parts[:3] != expected_root:
        raise ExampleError(
            f"{context}: expected an sdk/{language}/examples/*{suffix} source, "
            f"got {relative}"
        )
    path = REPO / relative_path
    if not path.is_file():
        raise ExampleError(f"{context}: missing source {relative}")
    return path


def _display_path(path: Path) -> Path:
    try:
        return path.relative_to(REPO)
    except ValueError:
        return path


def _extract(path: Path, example_id: str, comment: str) -> str:
    text = path.read_text(encoding="utf-8")
    begin = f"{comment} OA_DOC_BEGIN: {example_id}"
    end = f"{comment} OA_DOC_END: {example_id}"
    if text.count(begin) != 1 or text.count(end) != 1:
        raise ExampleError(
            f"{_display_path(path)}: expected exactly one {begin!r}/{end!r} pair"
        )
    begin_offset = text.index(begin) + len(begin)
    end_offset = text.index(end)
    if end_offset <= begin_offset:
        raise ExampleError(f"{_display_path(path)}: reversed documentation markers")
    code = text[begin_offset:end_offset].strip()
    if not code:
        raise ExampleError(f"{_display_path(path)}: empty documentation region")
    return code


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def generate() -> dict[str, Any]:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    if manifest.get("schema") != 1:
        raise ExampleError("sdk/examples.json: unsupported schema")
    examples = manifest.get("examples")
    if not isinstance(examples, list) or not examples:
        raise ExampleError("sdk/examples.json: examples must be a non-empty list")

    cmake = CPP_REGISTRY.read_text(encoding="utf-8")
    python_tests = PYTHON_REGISTRY.read_text(encoding="utf-8")
    ids: set[str] = set()
    generated: list[dict[str, Any]] = []

    for index, raw in enumerate(examples):
        context = f"sdk/examples.json:examples[{index}]"
        if not isinstance(raw, dict):
            raise ExampleError(f"{context}: entry must be an object")
        example_id = _require_string(raw, "id", context)
        if not ID_PATTERN.fullmatch(example_id):
            raise ExampleError(f"{context}: invalid id {example_id!r}")
        if example_id in ids:
            raise ExampleError(f"{context}: duplicate id {example_id!r}")
        ids.add(example_id)

        cpp = raw.get("cpp")
        python = raw.get("python")
        if not isinstance(cpp, dict) or not isinstance(python, dict):
            raise ExampleError(f"{context}: cpp and python entries are required")

        cpp_relative = _require_string(cpp, "path", f"{context}.cpp")
        python_relative = _require_string(python, "path", f"{context}.python")
        cpp_path = _source_path(cpp_relative, "cpp", ".cpp", f"{context}.cpp")
        python_path = _source_path(python_relative, "py", ".py", f"{context}.python")
        target = _require_string(cpp, "target", f"{context}.cpp")
        profile = _require_string(python, "profile", f"{context}.python")

        if not re.search(rf"\boa_add_example\(\s*{re.escape(target)}\b", cmake):
            raise ExampleError(f"{context}: CMake target {target!r} is not registered")
        if python_relative not in python_tests:
            raise ExampleError(
                f"{context}: Python source is not registered in "
                f"{PYTHON_REGISTRY.relative_to(REPO)}"
            )

        generated.append(
            {
                "id": example_id,
                "title": _require_string(raw, "title", context),
                "summary": _require_string(raw, "summary", context),
                "capability": _require_string(raw, "capability", context),
                "cppModules": _require_strings(raw, "cppModules", context),
                "pythonModules": _require_strings(raw, "pythonModules", context),
                "symbols": _require_strings(raw, "symbols", context),
                "cpp": {
                    "path": cpp_relative,
                    "filename": cpp_path.name,
                    "target": target,
                    "sha256": _sha256(cpp_path),
                    "code": _extract(cpp_path, example_id, "//"),
                },
                "python": {
                    "path": python_relative,
                    "filename": python_path.name,
                    "profile": profile,
                    "sha256": _sha256(python_path),
                    "code": _extract(python_path, example_id, "#"),
                },
            }
        )

    return {"schema": 1, "examples": generated}


def render() -> str:
    return json.dumps(generate(), indent=2, ensure_ascii=False) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--check",
        action="store_true",
        help="fail when the committed generated inventory differs",
    )
    args = parser.parse_args()

    try:
        rendered = render()
    except (ExampleError, json.JSONDecodeError) as error:
        print(f"[examples] FAIL: {error}")
        return 1

    if args.check:
        current = OUTPUT.read_text(encoding="utf-8") if OUTPUT.is_file() else ""
        if current != rendered:
            print(
                "[examples] FAIL: generated inventory is stale: "
                f"{OUTPUT.relative_to(REPO)}"
            )
            return 1
        print("[examples] PASS: manifest, registration, markers and snapshot are current")
        return 0

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(rendered, encoding="utf-8")
    print(f"[examples] wrote {OUTPUT.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
