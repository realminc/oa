from __future__ import annotations

import copy
import importlib.util
import tempfile
import unittest
from pathlib import Path


SCRIPT = Path(__file__).with_name("generate.py")
SPEC = importlib.util.spec_from_file_location("kernelGenerate", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class KernelGenerationTest(unittest.TestCase):
	@classmethod
	def setUpClass(cls) -> None:
		kernels, reservations = MODULE.loadSchema(MODULE.DEFAULT_SCHEMA)
		cls.kernels, cls.reservations = MODULE.normalizedRecords(
		 kernels, reservations, MODULE.REPO_ROOT
		)

	def testLiveSchemaIsUniqueAndSourcesExist(self) -> None:
		self.assertEqual(222, len(self.kernels))
		self.assertEqual(36, len(self.reservations))
		self.assertEqual(len(self.kernels), len({row["name"] for row in self.kernels}))
		self.assertEqual(
		 len(self.kernels),
		 len({(row["prefix"], row["local"]) for row in self.kernels}),
		)

	def testEmitIsIdempotent(self) -> None:
		with tempfile.TemporaryDirectory() as directory:
			out = Path(directory)
			MODULE.generate(MODULE.DEFAULT_SCHEMA, out, MODULE.REPO_ROOT)
			before = {path.relative_to(out): path.read_bytes() for path in out.rglob("*") if path.is_file()}
			MODULE.generate(MODULE.DEFAULT_SCHEMA, out, MODULE.REPO_ROOT)
			after = {path.relative_to(out): path.read_bytes() for path in out.rglob("*") if path.is_file()}
			self.assertEqual(before, after)

	def testDuplicateNameIsRejected(self) -> None:
		kernels = copy.deepcopy(self.kernels)
		kernels[1]["name"] = kernels[0]["name"]
		with self.assertRaisesRegex(MODULE.SchemaError, "duplicate kernel name"):
			MODULE.normalizedRecords(kernels, self.reservations, MODULE.REPO_ROOT)

	def testActiveIdReservationOverlapIsRejected(self) -> None:
		reservations = copy.deepcopy(self.reservations)
		reservations.append({"prefix": "Ml", "first": 1, "last": 1})
		with self.assertRaisesRegex(MODULE.SchemaError, "overlaps active id"):
			MODULE.normalizedRecords(self.kernels, reservations, MODULE.REPO_ROOT)


if __name__ == "__main__":
	unittest.main()
