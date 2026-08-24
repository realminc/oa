#!/usr/bin/env python3
"""Verify the complete checked-in SDK asset inventory and provenance pins."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path, PurePosixPath
import sys
import tomllib


def _safeRelative(value: str) -> PurePosixPath:
	path = PurePosixPath(value)
	if path.is_absolute() or not path.parts or ".." in path.parts:
		raise ValueError(f"unsafe relative path: {value!r}")
	return path


def _sha256(path: Path) -> str:
	digest = hashlib.sha256()
	with path.open("rb") as stream:
		for chunk in iter(lambda: stream.read(1024 * 1024), b""):
			digest.update(chunk)
	return digest.hexdigest()


def verify(assetRoot: Path) -> list[str]:
	manifestPath = assetRoot / "manifest.toml"
	with manifestPath.open("rb") as stream:
		manifest = tomllib.load(stream)

	errors: list[str] = []
	if manifest.get("schemaVersion") != 1:
		errors.append("manifest.toml: schemaVersion must be 1")

	metadata = manifest.get("metadata", [])
	assets = manifest.get("asset", [])
	declared: set[str] = set()

	for value in metadata:
		try:
			relative = _safeRelative(value)
		except ValueError as error:
			errors.append(str(error))
			continue
		key = relative.as_posix()
		if key in declared:
			errors.append(f"duplicate inventory path: {key}")
		declared.add(key)
		if not (assetRoot / relative).is_file():
			errors.append(f"missing metadata file: {key}")

	assetPaths: list[str] = []
	for entry in assets:
		try:
			relative = _safeRelative(entry["path"])
		except (KeyError, TypeError, ValueError) as error:
			errors.append(f"invalid asset entry: {error}")
			continue
		key = relative.as_posix()
		assetPaths.append(key)
		if key in declared:
			errors.append(f"duplicate inventory path: {key}")
		declared.add(key)

		for field in ("license", "source", "role"):
			if not isinstance(entry.get(field), str) or not entry[field].strip():
				errors.append(f"{key}: missing {field}")

		path = assetRoot / relative
		if not path.is_file():
			errors.append(f"missing asset: {key}")
			continue
		expectedBytes = entry.get("bytes")
		if path.stat().st_size != expectedBytes:
			errors.append(
				f"{key}: size {path.stat().st_size}, expected {expectedBytes}")
		expectedHash = entry.get("sha256")
		actualHash = _sha256(path)
		if actualHash != expectedHash:
			errors.append(f"{key}: sha256 {actualHash}, expected {expectedHash}")

	if assetPaths != sorted(assetPaths):
		errors.append("manifest asset entries must be sorted by path")

	actual = {
		path.relative_to(assetRoot).as_posix()
		for path in assetRoot.rglob("*")
		if path.is_file() and path != manifestPath
	}
	missing = sorted(actual - declared)
	extra = sorted(declared - actual)
	if missing:
		errors.append("unlisted files: " + ", ".join(missing))
	if extra:
		errors.append("listed files absent from tree: " + ", ".join(extra))
	return errors


def main() -> int:
	parser = argparse.ArgumentParser(description=__doc__)
	parser.add_argument(
		"assetRoot",
		type=Path,
		nargs="?",
		default=Path(__file__).resolve().parents[2] / "sdk" / "asset",
	)
	args = parser.parse_args()
	errors = verify(args.assetRoot.resolve())
	if errors:
		for error in errors:
			print(f"asset inventory: {error}", file=sys.stderr)
		return 1
	print(f"asset inventory: verified {args.assetRoot.resolve()}")
	return 0


if __name__ == "__main__":
	raise SystemExit(main())
