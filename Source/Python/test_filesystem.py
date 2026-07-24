#!/usr/bin/env python3
"""Host-only contract tests for OA path and filesystem bindings."""

from __future__ import annotations

import os
from pathlib import Path
import unittest
from unittest.mock import patch

from oa import OaFilesystem, OaPath, OaPaths


class TestFilesystem(unittest.TestCase):
	def setUp(self) -> None:
		self.work = OaPaths.Temp() / f"oa_python_filesystem_{os.getpid()}"
		if OaFilesystem.Exists(self.work):
			OaFilesystem.RemoveDirectory(self.work, Recursive=True)
		OaFilesystem.CreateDirectories(self.work)

	def tearDown(self) -> None:
		if OaFilesystem.Exists(self.work):
			OaFilesystem.RemoveDirectory(self.work, Recursive=True)

	def test_path_value_and_pathlike_protocol(self) -> None:
		path = OaPath("one") / "two" / "file.txt"
		self.assertEqual(path.Filename().String(), "file.txt")
		self.assertEqual(path.Stem().String(), "file")
		self.assertEqual(path.Extension().String(), ".txt")
		self.assertEqual(os.fspath(path), path.String())
		self.assertEqual(OaPath(Path("one") / "two"), OaPath("one/two"))
		self.assertEqual(OaPath(b"one/two"), OaPath("one/two"))
		with self.assertRaises(ValueError):
			OaPath("bad\0path")
		with self.assertRaises(TypeError):
			OaPath(42)

	def test_text_binary_and_listing_round_trip(self) -> None:
		text_path = self.work / "nested" / "sample.txt"
		OaFilesystem.WriteText(text_path, "first\n")
		OaFilesystem.AppendText(text_path, "second\n")
		self.assertEqual(OaFilesystem.ReadText(text_path), "first\nsecond\n")
		self.assertEqual(OaFilesystem.ReadLines(text_path), ["first", "second"])

		binary_path = Path(os.fspath(self.work)) / "sample.bin"
		OaFilesystem.WriteBinary(binary_path, b"\x00\xffOA")
		self.assertEqual(OaFilesystem.ReadBinary(binary_path), b"\x00\xffOA")

		files = OaFilesystem.ListFiles(self.work, ".bin")
		self.assertEqual([path.Filename().String() for path in files], ["sample.bin"])
		self.assertEqual(
			[path.Filename().String() for path in OaFilesystem.Glob(self.work, "*.bin")],
			["sample.bin"],
		)

	def test_named_asset_location(self) -> None:
		fixture = OaPaths.Asset("Image/VisionTestPattern320x180.jpg")
		self.assertTrue(OaFilesystem.IsFile(fixture))

	def test_named_var_environment_override(self) -> None:
		override = self.work / "custom-var"
		with patch.dict(os.environ, {"OA_VAR_DIR": os.fspath(override)}):
			self.assertEqual(
				OaPaths.Var("report.json"),
				override / "report.json",
			)


if __name__ == "__main__":
	unittest.main()
