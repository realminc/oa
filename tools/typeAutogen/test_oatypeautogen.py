#!/usr/bin/env python3
"""Regression tests for TypeAutogen ownership and provenance."""

from __future__ import annotations

import tempfile
import unittest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import oatypeautogen
from oatypeautogen_lib.config import DOMAIN_OUTPUT_PATHS, SCHEMA_DIR
from oatypeautogen_lib.emitters import emit_cpp_file, emit_header_file
from oatypeautogen_lib.schema import (
	StructDef,
	TypeSchema,
	load_schema,
	validate_schema,
)


class TypeAutogenRegressionTest(unittest.TestCase):
	def test_nested_schema_provenance_is_domain_qualified(self) -> None:
		path = SCHEMA_DIR / "ml" / "mlTypes.toml"
		self.assertEqual(
			oatypeautogen.schema_source_name(path),
			"ml/mlTypes.toml",
		)

	def test_external_schema_provenance_falls_back_to_filename(self) -> None:
		with tempfile.TemporaryDirectory() as temporary:
			path = Path(temporary) / "CustomTypes.toml"
			self.assertEqual(
				oatypeautogen.schema_source_name(path),
				"CustomTypes.toml",
			)

	def test_ml_schema_matches_the_supported_layer_surface(self) -> None:
		schema = load_schema(SCHEMA_DIR / "ml" / "mlTypes.toml")
		values = {
			enum.name: [(value.name, value.value) for value in enum.values]
			for enum in schema.enums
		}
		self.assertEqual(
			values,
			{
				"Quantization": [
					("Q4", 0),
					("Q8", 1),
				],
				"Activation": [
					("None", 0),
					("Relu", 1),
					("Gelu", 2),
					("Silu", 3),
				],
				"UpsampleMode": [
					("Nearest", 0),
					("Bilinear", 1),
				],
			},
		)

	def test_runtime_schema_preserves_live_routing_identities(self) -> None:
		schema = load_schema(
			SCHEMA_DIR / "runtime" / "runtimeTypes.toml"
		)
		values = {
			enum.name: [(value.name, value.value) for value in enum.values]
			for enum in schema.enums
		}
		self.assertEqual(
			values,
			{
				"StoragePrecision": [("Fp32", 0), ("Bf16", 2)],
				"GemmKernel": [
					("Auto", 0),
					("TiledFp32", 4),
					("Naive", 5),
					("CoopVec", 6),
					("GemmCmSgBf16", 11),
					("GemmCmWgBf16", 12),
					("StridedFp32", 13),
					("SmallMFp32", 14),
					("StridedTiledFp32", 15),
				],
				"GemmPath": [("Standard", 0), ("CoopVec", 1)],
				"GemmPrecision": [
					("Auto", 0),
					("Fp32", 1),
					("Bf16", 2),
				],
			},
		)

	def test_runtime_lowering_types_are_private(self) -> None:
		self.assertEqual(
			DOMAIN_OUTPUT_PATHS["runtime"]["header"],
			"source/cpp/lib/oa/runtime/type.gen.h",
		)

	def test_enum_conversions_are_header_complete(self) -> None:
		schema = load_schema(SCHEMA_DIR / "core" / "coreTypes.toml")
		header = emit_header_file(schema, "core/coreTypes.toml")
		implementation = emit_cpp_file(schema, "core/coreTypes.toml")
		self.assertIn(
			"constexpr const char* scalarTypeToString(", header
		)
		self.assertIn("std::strcmp(inString, \"Float32\")", header)
		self.assertIn("if (inString == nullptr)", header)
		self.assertNotIn("scalarTypeToString(", implementation)

	def test_enum_serialized_names_are_independent_of_cpp_identifiers(self) -> None:
		schema = load_schema(SCHEMA_DIR / "vision" / "visionTypes.toml")
		header = emit_header_file(schema, "vision/visionTypes.toml")
		self.assertIn('case ImageCodec::Webp: return "webp";', header)
		self.assertIn('std::strcmp(inString, "webp")', header)

	def test_struct_defaults_survive_comments(self) -> None:
		schema = load_schema(SCHEMA_DIR / "core" / "coreTypes.toml")
		header = emit_header_file(schema, "core/coreTypes.toml")
		self.assertIn(
			"Precision mode = Precision::FP32;  // Computation precision",
			header,
		)
		self.assertIn(
			"DeterminismMode determinism = DeterminismMode::Stable;",
			header,
		)

	def test_placeholder_struct_methods_are_rejected(self) -> None:
		schema = TypeSchema(
			domain="core",
			namespace="oa",
			enums=[],
			structs=[
				StructDef(
					name="Example",
					fields=[],
					generate_serialize=True,
					generate_validate=True,
				)
			],
		)
		errors = validate_schema(schema)
		self.assertIn(
			"Struct Example requests unsupported generated serialization",
			errors,
		)
		self.assertIn(
			"Struct Example requests unsupported generated validation",
			errors,
		)


if __name__ == "__main__":
	unittest.main()
