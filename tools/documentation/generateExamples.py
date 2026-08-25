#!/usr/bin/env python3
"""Generate the publishable OA example inventory from checked source markers."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import tomllib
from pathlib import Path
from typing import Any


REPO = Path(__file__).resolve().parents[2]
MANIFEST = REPO / "sdk" / "examples.json"
OUTPUT = REPO / "docs" / "external" / "generated" / "examples.json"
ASSET_MANIFEST = REPO / "sdk" / "asset" / "manifest.toml"
CPP_REGISTRIES = (
    REPO / "sdk" / "cpp" / "examples" / "CMakeLists.txt",
    REPO / "sdk" / "cpp" / "examples" / "gen" / "examples.cmake",
)
PYTHON_REGISTRIES = (
    REPO / "test" / "py" / "examples" / "test_generated_examples.py",
)
ID_PATTERN = re.compile(r"^[a-z][a-z0-9]*(?:-[a-z0-9]+)*$")
PUBLIC_MODULES = {"Audio", "Core", "Crypto", "Data", "Ml", "Ui", "Vision"}


class ExampleError(ValueError):
    """A manifest or marked-source contract is invalid."""


def _requireString(owner: dict[str, Any], key: str, context: str) -> str:
    value = owner.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ExampleError(f"{context}: {key} must be a non-empty string")
    return value


def _requireStrings(owner: dict[str, Any], key: str, context: str) -> list[str]:
    value = owner.get(key)
    if (
        not isinstance(value, list)
        or not value
        or any(not isinstance(item, str) or not item.strip() for item in value)
    ):
        raise ExampleError(f"{context}: {key} must be a non-empty string list")
    return value


def _sourcePath(relative: str, language: str, suffix: str, context: str) -> Path:
    relativePath = Path(relative)
    if relativePath.is_absolute() or ".." in relativePath.parts:
        raise ExampleError(f"{context}: unsafe source path {relative}")
    expectedRoot = ("sdk", language, "examples")
    if relativePath.suffix != suffix or relativePath.parts[:3] != expectedRoot:
        raise ExampleError(
            f"{context}: expected an sdk/{language}/examples/*{suffix} source, "
            f"got {relative}"
        )
    path = REPO / relativePath
    if not path.is_file():
        raise ExampleError(f"{context}: missing source {relative}")
    return path


def _displayPath(path: Path) -> Path:
    try:
        return path.relative_to(REPO)
    except ValueError:
        return path


def _extract(path: Path, exampleId: str, comment: str) -> str:
    text = path.read_text(encoding="utf-8")
    begin = f"{comment} OA_DOC_BEGIN: {exampleId}"
    end = f"{comment} OA_DOC_END: {exampleId}"
    if text.count(begin) != 1 or text.count(end) != 1:
        raise ExampleError(
            f"{_displayPath(path)}: expected exactly one {begin!r}/{end!r} pair"
        )
    beginOffset = text.index(begin) + len(begin)
    endOffset = text.index(end)
    if endOffset <= beginOffset:
        raise ExampleError(f"{_displayPath(path)}: reversed documentation markers")
    code = text[beginOffset:endOffset].strip()
    if not code:
        raise ExampleError(f"{_displayPath(path)}: empty documentation region")
    return code


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _assetInventory() -> dict[str, dict[str, Any]]:
    with ASSET_MANIFEST.open("rb") as stream:
        manifest = tomllib.load(stream)
    inventory: dict[str, dict[str, Any]] = {}
    for index, entry in enumerate(manifest.get("asset", [])):
        context = f"sdk/asset/manifest.toml:asset[{index}]"
        if not isinstance(entry, dict):
            raise ExampleError(f"{context}: entry must be a table")
        relative = _requireString(entry, "path", context)
        path = REPO / "sdk" / "asset" / relative
        if not path.is_file():
            raise ExampleError(f"{context}: missing asset {relative}")
        expectedBytes = entry.get("bytes")
        expectedSha = _requireString(entry, "sha256", context)
        if path.stat().st_size != expectedBytes or _sha256(path) != expectedSha:
            raise ExampleError(f"{context}: asset evidence does not match its manifest pin")
        inventory[relative] = entry
    return inventory


def _publishPresentation(
    value: Any,
    declaredAssets: set[str],
    inventory: dict[str, dict[str, Any]],
    context: str,
) -> list[dict[str, Any]]:
    if not isinstance(value, list) or not value:
        raise ExampleError(f"{context}: presentation must be a non-empty list")
    published = []
    for presentationIndex, presentation in enumerate(value):
        presentationContext = f"{context}[{presentationIndex}]"
        if not isinstance(presentation, dict):
            raise ExampleError(f"{presentationContext}: presentation must be an object")
        kind = _requireString(presentation, "kind", presentationContext)
        if kind == "terminalOutput":
            published.append({
                "kind": kind,
                "title": _requireString(presentation, "title", presentationContext),
                "description": _requireString(presentation, "description", presentationContext),
                "filename": _requireString(presentation, "filename", presentationContext),
                "language": _requireString(presentation, "language", presentationContext),
                "generatedBy": _requireString(presentation, "generatedBy", presentationContext),
                "code": _requireString(presentation, "code", presentationContext),
            })
            continue
        if kind == "viewerCapture":
            asset = _requireString(presentation, "asset", presentationContext)
            if asset not in declaredAssets or asset not in inventory:
                raise ExampleError(
                    f"{presentationContext}: Viewer capture must be a declared checked asset"
                )
            width = presentation.get("width")
            height = presentation.get("height")
            if (
                isinstance(width, bool)
                or not isinstance(width, int)
                or width <= 0
                or isinstance(height, bool)
                or not isinstance(height, int)
                or height <= 0
            ):
                raise ExampleError(
                    f"{presentationContext}: Viewer capture dimensions must be positive integers"
                )
            assetRecord = inventory[asset]
            published.append({
                "kind": kind,
                "title": _requireString(presentation, "title", presentationContext),
                "description": _requireString(presentation, "description", presentationContext),
                "asset": asset,
                "src": f"/media/oa/{asset}",
                "mimeType": _requireString(presentation, "mimeType", presentationContext),
                "sha256": assetRecord["sha256"],
                "bytes": assetRecord["bytes"],
                "width": width,
                "height": height,
                "alt": _requireString(presentation, "alt", presentationContext),
                "generatedBy": _requireString(presentation, "generatedBy", presentationContext),
            })
            continue
        if kind == "imageGallery":
            items = presentation.get("items")
            if not isinstance(items, list) or not items:
                raise ExampleError(
                    f"{presentationContext}: imageGallery requires at least one item"
                )
            publishedItems = []
            roles: set[str] = set()
            for itemIndex, item in enumerate(items):
                itemContext = f"{presentationContext}.items[{itemIndex}]"
                if not isinstance(item, dict):
                    raise ExampleError(f"{itemContext}: item must be an object")
                role = _requireString(item, "role", itemContext)
                if role in roles:
                    raise ExampleError(f"{itemContext}: image roles must be unique")
                roles.add(role)
                asset = _requireString(item, "asset", itemContext)
                if asset not in declaredAssets or asset not in inventory:
                    raise ExampleError(
                        f"{itemContext}: image must be a declared checked asset"
                    )
                width = item.get("width")
                height = item.get("height")
                if (
                    isinstance(width, bool)
                    or not isinstance(width, int)
                    or width <= 0
                    or isinstance(height, bool)
                    or not isinstance(height, int)
                    or height <= 0
                ):
                    raise ExampleError(
                        f"{itemContext}: image dimensions must be positive integers"
                    )
                assetRecord = inventory[asset]
                publishedItems.append({
                    "role": role,
                    "label": _requireString(item, "label", itemContext),
                    "description": _requireString(item, "description", itemContext),
                    "asset": asset,
                    "src": f"/media/oa/{asset}",
                    "mimeType": _requireString(item, "mimeType", itemContext),
                    "sha256": assetRecord["sha256"],
                    "bytes": assetRecord["bytes"],
                    "width": width,
                    "height": height,
                    "alt": _requireString(item, "alt", itemContext),
                    "generatedBy": _requireString(item, "generatedBy", itemContext),
                })
            published.append({
                "kind": kind,
                "title": _requireString(presentation, "title", presentationContext),
                "description": _requireString(presentation, "description", presentationContext),
                "items": publishedItems,
            })
            continue
        if kind != "audioComparison":
            raise ExampleError(f"{presentationContext}: unsupported kind {kind!r}")
        items = presentation.get("items")
        if not isinstance(items, list) or len(items) != 2:
            raise ExampleError(f"{presentationContext}: audioComparison requires two items")
        publishedItems = []
        roles: set[str] = set()
        for itemIndex, item in enumerate(items):
            itemContext = f"{presentationContext}.items[{itemIndex}]"
            if not isinstance(item, dict):
                raise ExampleError(f"{itemContext}: item must be an object")
            role = _requireString(item, "role", itemContext)
            if role not in {"source", "processed"} or role in roles:
                raise ExampleError(f"{itemContext}: roles must be unique source/processed values")
            roles.add(role)
            asset = _requireString(item, "asset", itemContext)
            fallbackAsset = _requireString(item, "fallbackAsset", itemContext)
            if asset not in declaredAssets or fallbackAsset not in declaredAssets:
                raise ExampleError(f"{itemContext}: presentation assets must be declared by the example")
            if asset not in inventory or fallbackAsset not in inventory:
                raise ExampleError(f"{itemContext}: presentation asset is absent from the checked inventory")
            assetRecord = inventory[asset]
            fallbackRecord = inventory[fallbackAsset]
            publishedItems.append({
                "role": role,
                "label": _requireString(item, "label", itemContext),
                "description": _requireString(item, "description", itemContext),
                "asset": asset,
                "src": f"/media/oa/{asset}",
                "mimeType": _requireString(item, "mimeType", itemContext),
                "sha256": assetRecord["sha256"],
                "bytes": assetRecord["bytes"],
                "fallbackAsset": fallbackAsset,
                "fallbackSrc": f"/media/oa/{fallbackAsset}",
                "fallbackMimeType": _requireString(item, "fallbackMimeType", itemContext),
                "fallbackSha256": fallbackRecord["sha256"],
                "sampleRate": item["sampleRate"],
                "channelCount": item["channelCount"],
                "sampleCount": item["sampleCount"],
                "durationSeconds": item["durationSeconds"],
                "generatedBy": _requireString(item, "generatedBy", itemContext),
            })
        if roles != {"source", "processed"}:
            raise ExampleError(f"{presentationContext}: source and processed roles are required")
        published.append({
            "kind": kind,
            "title": _requireString(presentation, "title", presentationContext),
            "description": _requireString(presentation, "description", presentationContext),
            "items": publishedItems,
        })
    return published


def generate() -> dict[str, Any]:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    if manifest.get("schema") != 2:
        raise ExampleError("sdk/examples.json: unsupported schema")
    examples = manifest.get("examples")
    if not isinstance(examples, list) or not examples:
        raise ExampleError("sdk/examples.json: examples must be a non-empty list")
    assetInventory = _assetInventory()

    cmake = "\n".join(path.read_text(encoding="utf-8") for path in CPP_REGISTRIES)
    pythonTests = "\n".join(
        path.read_text(encoding="utf-8") for path in PYTHON_REGISTRIES
    )
    ids: set[str] = set()
    generated: list[dict[str, Any]] = []

    for index, raw in enumerate(examples):
        context = f"sdk/examples.json:examples[{index}]"
        if not isinstance(raw, dict):
            raise ExampleError(f"{context}: entry must be an object")
        exampleId = _requireString(raw, "id", context)
        if not ID_PATTERN.fullmatch(exampleId):
            raise ExampleError(f"{context}: invalid id {exampleId!r}")
        if exampleId in ids:
            raise ExampleError(f"{context}: duplicate id {exampleId!r}")
        ids.add(exampleId)

        cpp = raw.get("cpp")
        python = raw.get("python")
        if not isinstance(cpp, dict) or not isinstance(python, dict):
            raise ExampleError(f"{context}: cpp and python entries are required")

        cppRelative = _requireString(cpp, "path", f"{context}.cpp")
        pythonRelative = _requireString(python, "path", f"{context}.python")
        cppPath = _sourcePath(cppRelative, "cpp", ".cpp", f"{context}.cpp")
        pythonPath = _sourcePath(pythonRelative, "py", ".py", f"{context}.python")
        target = _requireString(cpp, "target", f"{context}.cpp")
        profile = _requireString(python, "profile", f"{context}.python")
        module = _requireString(raw, "module", context)
        if module not in PUBLIC_MODULES:
            raise ExampleError(f"{context}: unsupported public module {module!r}")

        if not re.search(rf"\boa_add_example\(\s*{re.escape(target)}\b", cmake):
            raise ExampleError(f"{context}: CMake target {target!r} is not registered")
        if pythonRelative not in pythonTests:
            raise ExampleError(
                f"{context}: Python source is not registered in "
                "the Python example test registries"
            )

        generatedEntry = {
                "id": exampleId,
                "title": _requireString(raw, "title", context),
                "summary": _requireString(raw, "summary", context),
                "module": module,
                "capability": _requireString(raw, "capability", context),
                "cpp": {
                    "path": cppRelative,
                    "filename": cppPath.name,
                    "target": target,
                    "symbols": _requireStrings(cpp, "symbols", f"{context}.cpp"),
                    "sha256": _sha256(cppPath),
                    "code": _extract(cppPath, exampleId, "//"),
                },
                "python": {
                    "path": pythonRelative,
                    "filename": pythonPath.name,
                    "profile": profile,
                    "symbols": _requireStrings(
                        python, "symbols", f"{context}.python"
                    ),
                    "sha256": _sha256(pythonPath),
                    "code": _extract(pythonPath, exampleId, "#"),
                },
            }
        assets = raw.get("assets")
        if assets is not None:
            if (
                not isinstance(assets, list)
                or not assets
                or any(not isinstance(asset, str) or not asset for asset in assets)
            ):
                raise ExampleError(f"{context}: assets must be a non-empty string list")
            generatedEntry["assets"] = assets
        presentation = raw.get("presentation")
        if presentation is not None:
            generatedEntry["presentation"] = _publishPresentation(
                presentation,
                set(assets or []),
                assetInventory,
                f"{context}.presentation",
            )
        generated.append(generatedEntry)

    return {"schema": 2, "examples": generated}


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
