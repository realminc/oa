#!/usr/bin/env python3
"""Focused schema-contract tests for the stdlib-only OaTile generator."""

from __future__ import annotations

import sys
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parent))
import oatile  # noqa: E402


class OaTileSchemaTest(unittest.TestCase):
	def schema(self, variants: str) -> Path:
		tmp = tempfile.NamedTemporaryFile("w", suffix=".toml", delete=False)
		self.addCleanup(Path(tmp.name).unlink, missing_ok=True)
		with tmp:
			tmp.write("schema_version = 1\ngenerator_abi = 1\n\n")
			tmp.write(variants)
		return Path(tmp.name)

	def base_schema(self, variants: str) -> Path:
		tmp = tempfile.NamedTemporaryFile("w", suffix=".toml", delete=False)
		self.addCleanup(Path(tmp.name).unlink, missing_ok=True)
		with tmp:
			tmp.write("schema_version = 1\ngenerator_abi = 1\n\n")
			tmp.write(variants)
		return Path(tmp.name)

	def test_production_schema_is_bounded_and_valid(self) -> None:
		version, abi, variants, schema_hash = oatile.load_schema(oatile.SCHEMA)
		self.assertEqual(version, 1)
		self.assertEqual(abi, 4)
		self.assertEqual(
			[v.name for v in variants],
			[
				"GemmTiled", "GemmTiledSwizzle4", "GemmTiledSwizzle",
				"GemmTiledK32", "GemmTiledN32", "GemmStridedTiled",
			],
		)
		compiled = oatile.compiled_variants(variants)
		self.assertEqual(len(compiled), 26)
		self.assertEqual(len({v.name for v in compiled}), len(compiled))
		self.assertEqual(len({v.local_id for v in compiled}), len(compiled))
		self.assertEqual(
			{v.epilogue for v in compiled},
			{"none", "bias", "bias_relu", "bias_gelu", "bias_silu"},
		)
		self.assertEqual([v.name for v in variants if v.strided], ["GemmStridedTiled"])
		self.assertNotEqual(schema_hash, 0)

	def test_base_schema_owns_compile_ids_routes_and_capabilities(self) -> None:
		version, abi, variants, schema_hash = oatile.load_base_schema(oatile.BASE_SCHEMA)
		self.assertEqual((version, abi), (1, 1))
		self.assertEqual(
			[v.name for v in variants],
			["GemmNaive", "GemmStrided", "GemmCoopVec"],
		)
		self.assertEqual(
			{v.compile_group for v in variants},
			{"standard", "coopvec"},
		)
		self.assertEqual(len({v.local_id for v in variants}), len(variants))
		self.assertEqual([v.name for v in variants if v.public_id], ["GemmNaive"])
		self.assertTrue(all(v.caps for v in variants))
		self.assertNotEqual(schema_hash, 0)

	def test_small_m_schema_is_bounded_complete_and_portable(self) -> None:
		version, abi, variants, schema_hash = oatile.load_small_m_schema(
			oatile.SMALL_M_SCHEMA)
		self.assertEqual((version, abi), (1, 1))
		self.assertEqual(
			[v.name for v in variants],
			["GemmSmallM4R32"],
		)
		compiled = oatile.compiled_small_m_variants(variants)
		self.assertEqual(len(compiled), 5)
		self.assertEqual(len({v.name for v in compiled}), len(compiled))
		self.assertEqual(len({v.local_id for v in compiled}), len(compiled))
		self.assertTrue(all(v.max_m == 4 for v in variants))
		self.assertTrue(all(
			v.workgroup == v.outputs_per_group * v.reduction_width
			for v in variants))
		self.assertNotEqual(schema_hash, 0)

	def test_base_schema_rejects_duplicate_stable_ids(self) -> None:
		path = self.base_schema("""
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
			oatile.load_base_schema(path)

	def test_manifests_reject_cross_schema_stable_id_collision(self) -> None:
		_, _, base_variants, _ = oatile.load_base_schema(oatile.BASE_SCHEMA)
		_, _, variants, _ = oatile.load_schema(oatile.SCHEMA)
		_, _, small_m_variants, _ = oatile.load_small_m_schema(
			oatile.SMALL_M_SCHEMA)
		variants[0] = replace(variants[0], local_id=base_variants[0].local_id)
		with self.assertRaisesRegex(ValueError, "duplicate OaTile local_id"):
			oatile.validate_combined_kernel_identities(
				base_variants, variants, small_m_variants)

	def test_bf16_coop_schema_generates_both_capability_families(self) -> None:
		version, abi, families, schema_hash = oatile.load_coop_schema(oatile.BF16_SCHEMA)
		self.assertEqual((version, abi), (1, 2))
		self.assertEqual([f.template for f in families], ["GemmCmSgBf16", "GemmCmWgBf16"])
		self.assertEqual(sum(len(f.variants) for f in families), 12)
		self.assertEqual(
			{v.kind for f in families for v in f.variants},
			{"none", "bias", "bias_relu", "bias_gelu", "silu_dual", "bias_silu"},
		)
		self.assertNotEqual(schema_hash, 0)

	def test_duplicate_stable_id_is_rejected(self) -> None:
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
			oatile.load_schema(path)

	def test_inconsistent_thread_grid_is_rejected(self) -> None:
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
			oatile.load_schema(path)

	def test_epilogue_stable_id_collision_is_rejected(self) -> None:
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
			oatile.load_schema(path)

	def test_undistributable_float4_tile_is_rejected(self) -> None:
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
			oatile.load_schema(path)


if __name__ == "__main__":
	unittest.main()
