#!/usr/bin/env python3
"""Host-only tests for OA path and filesystem bindings."""

from __future__ import annotations

import os
from pathlib import Path
import unittest
from unittest.mock import patch

import oa_python_test  # noqa: F401 - bootstraps source builds

from oa import Filesystem, Path, Paths


class TestFilesystem(unittest.TestCase):
	def setUp(self) -> None:
		self.work = Paths.temp() / f"oa_python_filesystem_{os.getpid()}"
		if Filesystem.exists(self.work):
			Filesystem.removeDirectory(self.work, recursive=True)
		Filesystem.createDirectories(self.work)

	def tearDown(self) -> None:
		if Filesystem.exists(self.work):
			Filesystem.removeDirectory(self.work, recursive=True)

	def test_path_value_and_pathlike_protocol(self) -> None:
		path = Path("one") / "two" / "file.txt"
		self.assertEqual(path.filename().string(), "file.txt")
		self.assertEqual(path.stem().string(), "file")
		self.assertEqual(path.extension().string(), ".txt")
		self.assertEqual(os.fspath(path), path.string())
		self.assertEqual(Path(Path("one") / "two"), Path("one/two"))
		self.assertEqual(Path(b"one/two"), Path("one/two"))
		with self.assertRaises(ValueError):
			Path("bad\0path")
		with self.assertRaises(TypeError):
			Path(42)

	def test_text_binary_and_listing_round_trip(self) -> None:
		textPath = self.work / "nested" / "sample.txt"
		Filesystem.writeText(textPath, "first\n")
		Filesystem.appendText(textPath, "second\n")
		self.assertEqual(Filesystem.readText(textPath), "first\nsecond\n")
		self.assertEqual(Filesystem.readLines(textPath), ["first", "second"])

		binaryPath = Path(os.fspath(self.work)) / "sample.bin"
		Filesystem.writeBinary(binaryPath, b"\x00\xffOA")
		self.assertEqual(Filesystem.readBinary(binaryPath), b"\x00\xffOA")

		files = Filesystem.listFiles(self.work, ".bin")
		self.assertEqual([path.filename().string() for path in files], ["sample.bin"])
		self.assertEqual(
			[path.filename().string() for path in Filesystem.glob(self.work, "*.bin")],
			["sample.bin"],
		)

	def test_named_asset_location(self) -> None:
		fixture = Paths.asset("image/visionTestPattern320x180.jpg")
		self.assertTrue(Filesystem.isFile(fixture))

	def test_named_var_environment_override(self) -> None:
		override = self.work / "custom-var"
		with patch.dict(os.environ, {"OA_VAR_DIR": os.fspath(override)}):
			self.assertEqual(
				Paths.var("report.json"),
				override / "report.json",
			)

	def test_named_data_environment_override(self) -> None:
		override = self.work / "custom-data"
		with patch.dict(os.environ, {"OA_DATA_DIR": os.fspath(override)}):
			self.assertEqual(
				Paths.data("fashionMnist"),
				override / "fashionMnist",
			)


if __name__ == "__main__":
	unittest.main()
