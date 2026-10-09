#!/usr/bin/env python3
"""Verify the registry archive's public inventory and build contract."""

from __future__ import annotations

import argparse
from pathlib import Path, PurePosixPath
import tarfile
import tomllib


def audit(path: Path, version: str) -> None:
	root = f"oarust-{version}"
	allowed = {
		"Cargo.toml", "Cargo.toml.orig", "Cargo.lock", ".cargo_vcs_info.json",
		".oa-public-snapshot", "build.rs", "rustfmt.toml", "rust-toolchain.toml",
		"README.md", "LICENSE", "NOTICE.md", "CHANGELOG.md", "RELEASE_NOTES.md",
	}
	prefixes = ("src/rs/", "src/slang/", "sdk/asset/font/", "test/rs/", "test/fixtures/", "tool/gen/")
	if path.stat().st_size > 10 * 1024 * 1024:
		raise ValueError("crate archive exceeds the standard 10 MiB limit")
	files: dict[str, bytes] = {}
	with tarfile.open(path, "r:gz") as archive:
		for member in archive:
			parts = PurePosixPath(member.name).parts
			if not parts or parts[0] != root or ".." in parts:
				raise ValueError(f"unsafe crate path: {member.name}")
			if member.isdir():
				continue
			if not member.isfile():
				raise ValueError(f"non-file crate entry: {member.name}")
			name = "/".join(parts[1:])
			if name in files or not (name in allowed or name.startswith(prefixes)):
				raise ValueError(f"unexpected crate path: {name}")
			if "__pycache__" in parts or name.endswith(".pyc"):
				raise ValueError(f"cached Python output: {name}")
			stream = archive.extractfile(member)
			assert stream is not None
			files[name] = stream.read()
	manifest = tomllib.loads(files["Cargo.toml"].decode())
	if any(manifest.get(target) for target in ("bin", "example", "test", "bench")):
		raise ValueError("SDK/integration targets must stay outside the library crate")
	if manifest["package"]["name"] != "oarust" or manifest["package"]["version"] != version:
		raise ValueError("crate identity does not match release")
	if manifest["lib"]["name"] != "oa":
		raise ValueError("crate must preserve the oa library namespace")
	tables = [manifest, *manifest.get("target", {}).values()]
	for table in tables:
		for section in ("dependencies", "build-dependencies", "dev-dependencies"):
			for name, dependency in table.get(section, {}).items():
				if isinstance(dependency, dict) and any(key in dependency for key in ("git", "path")):
					raise ValueError(f"non-registry dependency: {name}")
	marker = files[".oa-public-snapshot"].decode()
	if f"public_version={version}\n" not in marker:
		raise ValueError("crate was not built from the sanitized public release")
	for required in ("src/rs/lib.rs", "build.rs", "tool/gen/fn/generate.py", "rustfmt.toml", "LICENSE", "NOTICE.md", "Cargo.lock", "README.md"):
		if required not in files:
			raise ValueError(f"missing crate build input: {required}")
	print(f"Verified {path.name}: {len(files)} public files, oa namespace, registry dependencies")


if __name__ == "__main__":
	parser = argparse.ArgumentParser(description=__doc__)
	parser.add_argument("archive", type=Path)
	parser.add_argument("version")
	args = parser.parse_args()
	audit(args.archive, args.version)
