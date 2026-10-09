#!/usr/bin/env python3
"""Verify the registry archive's public inventory and build contract."""

from __future__ import annotations

import argparse
import json
from pathlib import Path, PurePosixPath
import posixpath
import re
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
			library_shader = name.startswith("sdk/rs/slang/ml/rl/") and name.endswith(".slang")
			library_support = name in ("sdk/rs/mod.rs", "sdk/rs/data.rs", "sdk/rs/ml.rs") or (
				name.startswith(("sdk/rs/data/", "sdk/rs/ml/")) and name.endswith(".rs")
			)
			if name in files or not (name in allowed or name.startswith(prefixes) or library_shader or library_support):
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
	# Check declared build inputs before invoking compilers. Donor provenance is
	# not a local input; only live src/ and sdk/ shader paths enter this contract.
	def check_sources(value, owner):
		if isinstance(value, dict):
			for key, child in value.items():
				if key == "source" and isinstance(child, str) and child.endswith(".slang"):
					if child.startswith(("src/", "sdk/")) and child not in files:
						raise ValueError(f"missing crate shader input: {child} (declared by {owner})")
				check_sources(child, owner)
		elif isinstance(value, list):
			for child in value:
				check_sources(child, owner)
	for name, contents in files.items():
		if name.startswith("tool/gen/fn/schema/") and name.endswith(".json"):
			check_sources(json.loads(contents), name)
	for source in re.findall(r'"((?:src|sdk)/[^"\n]+\.slang)"', files["build.rs"].decode()):
		if source not in files:
			raise ValueError(f"missing crate shader input: {source} (declared by build.rs)")
	# Literal Rust paths are resolved relative to the declaring file, including
	# cfg(test) paths. OUT_DIR-generated shader includes are verified by Cargo.
	for name, contents in files.items():
		if not name.endswith(".rs"):
			continue
		text = contents.decode()
		paths = re.findall(r'#\[path\s*=\s*"([^"\n]+)"\]', text)
		paths += re.findall(r'\binclude(?:_bytes|_str)?!\(\s*"([^"\n]+)"\s*\)', text)
		for relative in paths:
			source = posixpath.normpath(posixpath.join(posixpath.dirname(name), relative))
			if source not in files:
				raise ValueError(f"missing crate Rust input: {source} (declared by {name})")
	print(f"Verified {path.name}: {len(files)} public files, oa namespace, registry dependencies")


if __name__ == "__main__":
	parser = argparse.ArgumentParser(description=__doc__)
	parser.add_argument("archive", type=Path)
	parser.add_argument("version")
	args = parser.parse_args()
	audit(args.archive, args.version)
