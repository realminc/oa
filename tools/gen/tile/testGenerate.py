#!/usr/bin/env python3
"""Focused schema-contract tests for the stdlib-only OaTile generator."""

from __future__ import annotations

import sys
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path


RepoRoot = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(RepoRoot))
from tools.gen.tile import generate as oatile  # noqa: E402


class OaTileSchemaTest(unittest.TestCase):
	def schema(self, variants: str) -> Path:
		tmp = tempfile.NamedTemporaryFile("w", suffix=".toml", delete=False)
		self.addCleanup(Path(tmp.name).unlink, missing_ok=True)
		with tmp:
			tmp.write("schema_version = 1\ngenerator_abi = 1\n\n")
			tmp.write(variants)
		return Path(tmp.name)

	def baseSchema(self, variants: str) -> Path:
		tmp = tempfile.NamedTemporaryFile("w", suffix=".toml", delete=False)
		self.addCleanup(Path(tmp.name).unlink, missing_ok=True)
		with tmp:
			tmp.write("schema_version = 1\ngenerator_abi = 1\n\n")
			tmp.write(variants)
		return Path(tmp.name)

	def testProductionSchemaIsBoundedAndValid(self) -> None:
		version, abi, variants, schemaHash = oatile.loadSchema(oatile.SCHEMA)
		self.assertEqual(version, 1)
		self.assertEqual(abi, 4)
		self.assertEqual(
		 [v.name for v in variants],
		 [
		  "GemmTiled", "GemmTiledSwizzle4", "GemmTiledSwizzle",
		  "GemmTiledK32", "GemmTiledN32", "GemmStridedTiled",
		 ],
		)
		compiled = oatile.compiledVariants(variants)
		self.assertEqual(len(compiled), 26)
		self.assertEqual(len({v.name for v in compiled}), len(compiled))
		self.assertEqual(len({v.localId for v in compiled}), len(compiled))
		self.assertEqual(
		 {v.epilogue for v in compiled},
		 {"none", "bias", "bias_relu", "bias_gelu", "bias_silu"},
		)
		self.assertEqual([v.name for v in variants if v.strided], ["GemmStridedTiled"])
		self.assertNotEqual(schemaHash, 0)

	def testBaseSchemaOwnsCompileIdsRoutesAndCapabilities(self) -> None:
		version, abi, variants, schemaHash = oatile.loadBaseSchema(oatile.BASE_SCHEMA)
		self.assertEqual((version, abi), (1, 1))
		self.assertEqual(
		 [v.name for v in variants],
		 ["GemmNaive", "GemmStrided", "GemmCoopVec"],
		)
		self.assertEqual(
		 {v.compileGroup for v in variants},
		 {"standard", "coopvec"},
		)
		self.assertEqual(len({v.localId for v in variants}), len(variants))
		self.assertEqual([v.name for v in variants if v.publicId], ["GemmNaive"])
		self.assertTrue(all(v.caps for v in variants))
		self.assertNotEqual(schemaHash, 0)

	def testSmallMSchemaIsBoundedCompleteAndPortable(self) -> None:
		version, abi, variants, schemaHash = oatile.loadSmallMSchema(
		 oatile.SMALL_M_SCHEMA)
		self.assertEqual((version, abi), (1, 1))
		self.assertEqual(
		 [v.name for v in variants],
		 ["GemmSmallM4R32"],
		)
		compiled = oatile.compiledSmallMVariants(variants)
		self.assertEqual(len(compiled), 5)
		self.assertEqual(len({v.name for v in compiled}), len(compiled))
		self.assertEqual(len({v.localId for v in compiled}), len(compiled))
		self.assertTrue(all(v.maxM == 4 for v in variants))
		self.assertTrue(all(
		 v.workgroup == v.outputsPerGroup * v.reductionWidth
		 for v in variants))
		self.assertNotEqual(schemaHash, 0)

	def testBaseSchemaRejectsDuplicateStableIds(self) -> None:
		path = self.baseSchema("""
[[variant]]
name = "A"
source = "gemm/a"
compile_group = "standard"
local_id = 72
kernel = "Naive"
path = "Standard"
precisions = ["Fp32", "Fp32", "Fp32", "Fp32"]
tile = [1, 1, 1]
workgroup = 1
caps = "kCapNaiveFp32"

[[variant]]
name = "B"
source = "gemm/b"
compile_group = "standard"
local_id = 72
kernel = "Naive"
path = "Standard"
precisions = ["Fp32", "Fp32", "Fp32", "Fp32"]
tile = [1, 1, 1]
workgroup = 1
caps = "kCapNaiveFp32"
""")
		with self.assertRaisesRegex(ValueError, "duplicate base local_id"):
			oatile.loadBaseSchema(path)

	def testManifestsRejectCrossSchemaStableIdCollision(self) -> None:
		_, _, baseVariants, _ = oatile.loadBaseSchema(oatile.BASE_SCHEMA)
		_, _, variants, _ = oatile.loadSchema(oatile.SCHEMA)
		_, _, smallMVariants, _ = oatile.loadSmallMSchema(
		 oatile.SMALL_M_SCHEMA)
		variants[0] = replace(variants[0], localId=baseVariants[0].localId)
		with self.assertRaisesRegex(ValueError, "duplicate OaTile local_id"):
			oatile.validateCombinedKernelIdentities(
			 baseVariants, variants, smallMVariants)

	def testBf16CoopSchemaGeneratesBothCapabilityFamilies(self) -> None:
		version, abi, families, schemaHash = oatile.loadCoopSchema(oatile.BF16_SCHEMA)
		self.assertEqual((version, abi), (1, 2))
		self.assertEqual([f.template for f in families], ["GemmCmSgBf16", "GemmCmWgBf16"])
		self.assertEqual(sum(len(f.variants) for f in families), 12)
		self.assertEqual(
		 {v.kind for f in families for v in f.variants},
		 {"none", "bias", "bias_relu", "bias_gelu", "silu_dual", "bias_silu"},
		)
		self.assertNotEqual(schemaHash, 0)

	def testDuplicateStableIdIsRejected(self) -> None:
		path = self.schema("""
[[variant]]
name = "A"
local_id = 240
tile = [64, 64, 16]
thread_tile = [4, 4]
workgroup = 256
group_m = 0

[[variant]]
name = "B"
local_id = 240
tile = [64, 64, 16]
thread_tile = [4, 4]
workgroup = 256
group_m = 0
""")
		with self.assertRaisesRegex(ValueError, "duplicate local_id"):
			oatile.loadSchema(path)

	def testInconsistentThreadGridIsRejected(self) -> None:
		path = self.schema("""
[[variant]]
name = "BadGrid"
local_id = 240
tile = [64, 64, 16]
thread_tile = [4, 4]
workgroup = 128
group_m = 0
""")
		with self.assertRaisesRegex(ValueError, "thread grid requires 256"):
			oatile.loadSchema(path)

	def testEpilogueStableIdCollisionIsRejected(self) -> None:
		path = self.schema("""
[[variant]]
name = "A"
local_id = 240
tile = [64, 64, 16]
thread_tile = [4, 4]
workgroup = 256
group_m = 0
epilogues = [
  { kind = "bias", name = "ABias", local_id = 240 },
]
""")
		with self.assertRaisesRegex(ValueError, "duplicate local_id"):
			oatile.loadSchema(path)

	def testUndistributableFloat4TileIsRejected(self) -> None:
		path = self.schema("""
[[variant]]
name = "BadLoads"
local_id = 240
tile = [48, 32, 12]
thread_tile = [6, 4]
workgroup = 64
group_m = 0
""")
		with self.assertRaisesRegex(ValueError, "cannot be distributed as float4"):
			oatile.loadSchema(path)


if __name__ == "__main__":
	unittest.main()
