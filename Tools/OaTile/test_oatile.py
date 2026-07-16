#!/usr/bin/env python3
"""Focused schema-contract tests for the stdlib-only OaTile generator."""

from __future__ import annotations

import sys
import tempfile
import unittest
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

	def test_production_schema_is_bounded_and_valid(self) -> None:
		version, abi, variants, schema_hash = oatile.load_schema(oatile.SCHEMA)
		self.assertEqual(version, 1)
		self.assertEqual(abi, 2)
		self.assertEqual(
			[v.name for v in variants],
			["GemmTiled", "GemmTiledSwizzle4", "GemmTiledSwizzle", "GemmTiledK32"],
		)
		compiled = oatile.compiled_variants(variants)
		self.assertEqual(len(compiled), 16)
		self.assertEqual(len({v.name for v in compiled}), len(compiled))
		self.assertEqual(len({v.local_id for v in compiled}), len(compiled))
		self.assertEqual(
			{v.epilogue for v in compiled},
			{"none", "bias", "bias_relu", "bias_gelu"},
		)
		self.assertNotEqual(schema_hash, 0)

	def test_bf16_coop_schema_generates_both_capability_families(self) -> None:
		version, abi, families, schema_hash = oatile.load_coop_schema(oatile.BF16_SCHEMA)
		self.assertEqual((version, abi), (1, 1))
		self.assertEqual([f.template for f in families], ["GemmCmSgBf16", "GemmCmWgBf16"])
		self.assertEqual(sum(len(f.variants) for f in families), 10)
		self.assertEqual(
			{v.kind for f in families for v in f.variants},
			{"none", "bias", "bias_relu", "bias_gelu", "silu_dual"},
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
