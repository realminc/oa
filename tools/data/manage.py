#!/usr/bin/env python3
"""Explicitly list, fetch, verify, and locate OA SDK dataset packs."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import os
from pathlib import Path, PurePosixPath
import shutil
import sys
import tempfile
import tomllib
import urllib.error
import urllib.request


class DataError(RuntimeError):
	pass


def _safeRelative(value: str) -> PurePosixPath:
	path = PurePosixPath(value)
	if path.is_absolute() or not path.parts or ".." in path.parts:
		raise DataError(f"unsafe relative path: {value!r}")
	return path


def _sha256(path: Path) -> str:
	digest = hashlib.sha256()
	with path.open("rb") as stream:
		for chunk in iter(lambda: stream.read(1024 * 1024), b""):
			digest.update(chunk)
	return digest.hexdigest()


def _verifyFile(path: Path, expectedBytes: int, expectedHash: str) -> None:
	if not path.is_file():
		raise DataError(f"missing file: {path}")
	actualBytes = path.stat().st_size
	if actualBytes != expectedBytes:
		raise DataError(
			f"{path}: size {actualBytes}, expected {expectedBytes}")
	actualHash = _sha256(path)
	if actualHash != expectedHash:
		raise DataError(
			f"{path}: sha256 {actualHash}, expected {expectedHash}")


def loadManifest(path: Path) -> dict[str, dict]:
	with path.open("rb") as stream:
		manifest = tomllib.load(stream)
	if manifest.get("schemaVersion") != 1:
		raise DataError("dataset manifest schemaVersion must be 1")
	packs: dict[str, dict] = {}
	for pack in manifest.get("pack", []):
		packId = pack.get("id")
		if not isinstance(packId, str) or not packId:
			raise DataError("every dataset pack requires a non-empty id")
		_safeRelative(packId)
		if packId in packs:
			raise DataError(f"duplicate dataset pack: {packId}")
		for field in ("version", "license", "source", "description"):
			if not isinstance(pack.get(field), str) or not pack[field].strip():
				raise DataError(f"{packId}: missing {field}")
		for item in pack.get("file", []):
			_safeRelative(item.get("path", ""))
			if item.get("compression") not in ("none", "gzip"):
				raise DataError(
					f"{packId}/{item.get('path')}: unsupported compression")
		packs[packId] = pack
	return packs


def verifyPack(pack: dict, dataRoot: Path) -> None:
	packRoot = dataRoot / pack["id"]
	for item in pack.get("file", []):
		path = packRoot / _safeRelative(item["path"])
		_verifyFile(path, item["bytes"], item["sha256"])


def _download(item: dict, destination: Path) -> None:
	request = urllib.request.Request(
		item["url"], headers={"User-Agent": "OA-dataset-tool/1"})
	with urllib.request.urlopen(request) as response, destination.open("wb") as out:
		shutil.copyfileobj(response, out, length=1024 * 1024)
	_verifyFile(destination, item["downloadBytes"], item["downloadSha256"])


def fetchPack(pack: dict, dataRoot: Path) -> None:
	packRoot = dataRoot / pack["id"]
	packRoot.mkdir(parents=True, exist_ok=True)
	for item in pack.get("file", []):
		destination = packRoot / _safeRelative(item["path"])
		destination.parent.mkdir(parents=True, exist_ok=True)
		try:
			_verifyFile(destination, item["bytes"], item["sha256"])
			print(f"verified {destination}")
			continue
		except DataError:
			pass

		with tempfile.TemporaryDirectory(
			prefix="oa-data-", dir=packRoot) as tempDirectory:
			tempRoot = Path(tempDirectory)
			download = tempRoot / "download"
			output = tempRoot / "output"
			print(f"fetching {item['url']}")
			_download(item, download)
			if item["compression"] == "gzip":
				with gzip.open(download, "rb") as source, output.open("wb") as target:
					shutil.copyfileobj(source, target, length=1024 * 1024)
			else:
				shutil.copyfile(download, output)
			_verifyFile(output, item["bytes"], item["sha256"])
			os.replace(output, destination)
			print(f"installed {destination}")


def _defaultRepoRoot() -> Path:
	return Path(__file__).resolve().parents[2]


def _dataRoot(value: str | None) -> Path:
	if value:
		return Path(value).expanduser().resolve()
	environment = os.getenv("OA_DATA_DIR")
	if environment:
		return Path(environment).expanduser().resolve()
	return _defaultRepoRoot() / "var" / "data"


def main(argv: list[str] | None = None) -> int:
	parser = argparse.ArgumentParser(description=__doc__)
	parser.add_argument(
		"--manifest",
		type=Path,
		default=_defaultRepoRoot() / "sdk" / "data" / "packs.toml",
	)
	parser.add_argument("--data-root", help="override OA_DATA_DIR")
	subparsers = parser.add_subparsers(dest="command", required=True)
	subparsers.add_parser("list")
	for command in ("fetch", "verify", "path"):
		commandParser = subparsers.add_parser(command)
		commandParser.add_argument("pack")
	args = parser.parse_args(argv)

	try:
		packs = loadManifest(args.manifest.resolve())
		dataRoot = _dataRoot(args.data_root)
		if args.command == "list":
			for packId, pack in sorted(packs.items()):
				print(
					f"{packId}\t{pack['version']}\t{pack['license']}\t"
					f"{pack['description']}")
			return 0
		pack = packs.get(args.pack)
		if pack is None:
			raise DataError(f"unknown dataset pack: {args.pack}")
		if args.command == "path":
			print((dataRoot / pack["id"]).resolve())
			return 0
		if args.command == "fetch":
			fetchPack(pack, dataRoot)
			verifyPack(pack, dataRoot)
			print(f"verified {pack['id']} at {dataRoot / pack['id']}")
			return 0
		if args.command == "verify":
			verifyPack(pack, dataRoot)
			print(f"verified {pack['id']} at {dataRoot / pack['id']}")
			return 0
	except (DataError, OSError, KeyError, TypeError, urllib.error.URLError) as error:
		print(f"dataset: {error}", file=sys.stderr)
		return 1


if __name__ == "__main__":
	raise SystemExit(main())
