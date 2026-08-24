#!/usr/bin/env python3
"""Host-only tests for explicit dataset acquisition and verification."""

from __future__ import annotations

import gzip
import hashlib
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import manage  # noqa: E402


class DatasetManagementTest(unittest.TestCase):
	def testFetchVerifyAndLocateLocalPack(self) -> None:
		with tempfile.TemporaryDirectory() as directory:
			root = Path(directory)
			payload = b"oa dataset fixture\n" * 4
			archive = root / "fixture.gz"
			with gzip.GzipFile(filename=str(archive), mode="wb", mtime=0) as stream:
				stream.write(payload)
			manifest = root / "packs.toml"
			manifest.write_text(
				"\n".join([
					"schemaVersion = 1",
					"[[pack]]",
					'id = "fixture"',
					'version = "1"',
					'license = "CC0-1.0"',
					'source = "local test"',
					'description = "local deterministic fixture"',
					"[[pack.file]]",
					'path = "payload.bin"',
					f'url = "{archive.as_uri()}"',
					f"downloadBytes = {archive.stat().st_size}",
					f'downloadSha256 = "{hashlib.sha256(archive.read_bytes()).hexdigest()}"',
					'compression = "gzip"',
					f"bytes = {len(payload)}",
					f'sha256 = "{hashlib.sha256(payload).hexdigest()}"',
				]),
				encoding="utf-8",
			)
			dataRoot = root / "data"
			self.assertEqual(manage.main([
				"--manifest", str(manifest),
				"--data-root", str(dataRoot),
				"fetch", "fixture",
			]), 0)
			self.assertEqual(
				(dataRoot / "fixture" / "payload.bin").read_bytes(), payload)
			self.assertEqual(manage.main([
				"--manifest", str(manifest),
				"--data-root", str(dataRoot),
				"verify", "fixture",
			]), 0)

	def testRejectsTraversal(self) -> None:
		with tempfile.TemporaryDirectory() as directory:
			manifest = Path(directory) / "packs.toml"
			manifest.write_text(
				"""schemaVersion = 1
[[pack]]
id = "fixture"
version = "1"
license = "CC0-1.0"
source = "local test"
description = "invalid path fixture"
[[pack.file]]
path = "../escape"
url = "file:///invalid"
downloadBytes = 0
downloadSha256 = ""
compression = "none"
bytes = 0
sha256 = ""
""",
				encoding="utf-8",
			)
			with self.assertRaises(manage.DataError):
				manage.loadManifest(manifest)


if __name__ == "__main__":
	unittest.main()
