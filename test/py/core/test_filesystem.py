"""Installed-wheel coverage for OA Path and Filesystem bindings."""

import os
import tempfile
import unittest
from pathlib import Path as NativePath

import oa


class FilesystemBindingTest(unittest.TestCase):
	def test_root_and_core_identities(self) -> None:
		self.assertIs(oa.Path, oa.core.Path)
		self.assertIs(oa.Filesystem, oa.core.Filesystem)

	def test_path_accepts_pathlike_and_preserves_oa_operations(self) -> None:
		path = oa.Path(NativePath("alpha") / "beta.txt")
		self.assertEqual(os.fspath(path), "alpha/beta.txt")
		self.assertEqual(path.name(), "beta.txt")
		self.assertEqual(path.stem(), "beta")
		self.assertEqual(path.suffix(), "txt")
		self.assertEqual(path.parent(), oa.Path("alpha"))
		self.assertEqual(path / "child", oa.Path("alpha/beta.txt/child"))
		self.assertEqual(oa.Path.empty(), oa.Path())

	def test_named_locations_return_oa_paths(self) -> None:
		for location in (oa.Path.asset(), oa.Path.var(), oa.Path.data(), oa.Path.home(), oa.Path.temp()):
			with self.subTest(location=location):
				self.assertIsInstance(location, oa.Path)

	def test_filesystem_round_trip_and_path_results(self) -> None:
		with tempfile.TemporaryDirectory() as directory:
			root = NativePath(directory)
			text = root / "nested" / "sample.txt"
			binary = root / "nested" / "sample.bin"

			oa.Filesystem.write_text(text, "first\n")
			oa.Filesystem.append_text(text, "second\n")
			self.assertEqual(oa.Filesystem.read_lines(text), ["first", "second"])
			self.assertTrue(oa.Filesystem.is_file(text))
			self.assertEqual(oa.Filesystem.get_file_size(text), len("first\nsecond\n"))

			oa.Filesystem.write_binary(binary, b"\x00\x01\xff")
			self.assertEqual(oa.Filesystem.read_binary(binary), b"\x00\x01\xff")

			files = oa.Filesystem.list_files(root / "nested")
			self.assertEqual([os.fspath(path) for path in files], sorted(os.fspath(path) for path in files))
			self.assertTrue(all(isinstance(path, oa.Path) for path in files))
			self.assertEqual([path.name() for path in oa.Filesystem.glob(root / "nested", "*.txt")], ["sample.txt"])

			absolute = oa.Filesystem.absolute(text)
			self.assertIsInstance(absolute, oa.Path)
			self.assertTrue(absolute.is_absolute())


if __name__ == "__main__":
	unittest.main()
