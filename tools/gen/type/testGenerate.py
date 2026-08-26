#!/usr/bin/env python3
"""Regression tests for type-generation ownership and provenance."""

from __future__ import annotations

import tempfile
import unittest
import sys
from pathlib import Path

RepoRoot = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(RepoRoot))
from tools.gen.type import generate as typeGenerator
from tools.gen.type.config import DOMAIN_OUTPUT_PATHS, SCHEMA_DIR
from tools.gen.type.emitters import emitCppFile, emitHeaderFile
from tools.gen.type.schema import (
 StructDef,
 TypeSchema,
 loadSchema,
 validateSchema,
)


class TypeGenerationTest(unittest.TestCase):
	def testNestedSchemaProvenanceIsDomainQualified(self) -> None:
		path = SCHEMA_DIR / "ml" / "mlTypes.toml"
		self.assertEqual(
		 typeGenerator.schemaSourceName(path),
		 "ml/mlTypes.toml",
		)

	def testExternalSchemaProvenanceFallsBackToFilename(self) -> None:
		with tempfile.TemporaryDirectory() as temporary:
			path = Path(temporary) / "CustomTypes.toml"
			self.assertEqual(
			 typeGenerator.schemaSourceName(path),
			 "CustomTypes.toml",
			)

	def testMlSchemaMatchesTheSupportedLayerSurface(self) -> None:
		schema = loadSchema(SCHEMA_DIR / "ml" / "mlTypes.toml")
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

	def testRuntimeSchemaPreservesLiveRoutingIdentities(self) -> None:
		schema = loadSchema(
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

	def testRuntimeLoweringTypesArePrivate(self) -> None:
		self.assertEqual(
		 DOMAIN_OUTPUT_PATHS["runtime"]["header"],
		 "source/cpp/lib/oa/runtime/type.gen.h",
		)

	def testEnumConversionsAreHeaderComplete(self) -> None:
		schema = loadSchema(SCHEMA_DIR / "core" / "coreTypes.toml")
		header = emitHeaderFile(schema, "core/coreTypes.toml")
		implementation = emitCppFile(schema, "core/coreTypes.toml")
		self.assertIn(
		 "constexpr const char* scalarTypeToString(", header
		)
		self.assertIn("oa::strcmp(inString, \"Float32\")", header)
		self.assertIn("#include <oa/core/std/cString.h>", header)
		self.assertNotIn("#include <cstring>", header)
		self.assertIn("if (inString == nullptr)", header)
		self.assertNotIn("scalarTypeToString(", implementation)

	def testEnumSerializedNamesAreIndependentOfCppIdentifiers(self) -> None:
		schema = loadSchema(SCHEMA_DIR / "vision" / "visionTypes.toml")
		header = emitHeaderFile(schema, "vision/visionTypes.toml")
		self.assertIn('case ImageCodec::Webp: return "webp";', header)
		self.assertIn('oa::strcmp(inString, "webp")', header)

	def testStructDefaultsSurviveComments(self) -> None:
		schema = loadSchema(SCHEMA_DIR / "core" / "coreTypes.toml")
		header = emitHeaderFile(schema, "core/coreTypes.toml")
		self.assertIn(
		 "Precision mode = Precision::FP32;  // Computation precision",
		 header,
		)
		self.assertIn(
		 "DeterminismMode determinism = DeterminismMode::Stable;",
		 header,
		)

	def testPlaceholderStructMethodsAreRejected(self) -> None:
		schema = TypeSchema(
		 domain="core",
		 namespace="oa",
		 enums=[],
		 structs=[
		  StructDef(
		   name="Example",
		   fields=[],
		   generateSerialize=True,
		   generateValidate=True,
		  )
		 ],
		)
		errors = validateSchema(schema)
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
