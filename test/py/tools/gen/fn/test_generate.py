import copy
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


GENERATOR_ROOT = Path(__file__).resolve().parents[5] / "tools/gen/fn"
GENERATOR_PATH = GENERATOR_ROOT / "generate.py"
SPEC = importlib.util.spec_from_file_location("oa_fn_generate", GENERATOR_PATH)
assert SPEC is not None and SPEC.loader is not None
GENERATOR = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(GENERATOR)
SCHEMA_PATH = GENERATOR_ROOT / "schema/matrix_elemwise.json"
BLAS_SCHEMA_PATH = GENERATOR_ROOT / "schema/matrix_blas.json"


class GeneratorTests(unittest.TestCase):
	def setUp(self):
		self.schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
		self.blas_schema = json.loads(BLAS_SCHEMA_PATH.read_text(encoding="utf-8"))

	def test_schema_generates_every_operation_surface(self):
		GENERATOR.validate_schema(self.schema)
		GENERATOR.validate_blas_schema(self.blas_schema)
		with tempfile.TemporaryDirectory() as directory:
			root = Path(directory)
			outputs = GENERATOR.expected_outputs(
				root, self.schema, "0" * 64, self.blas_schema, "1" * 64
			)
			variant_count = sum(
				len(GENERATOR.operation_variants(self.schema, operation))
				for operation in self.schema["operations"]
			)
			self.assertEqual(len(outputs), variant_count + 6)
			api = outputs[root / "src/rs/matrix/elemwise.gen.rs"]
			blas_api = outputs[root / "src/rs/matrix/blas.gen.rs"]
			registry = outputs[root / "src/rs/runtime/shader/generated.rs"]
			self.assertIn(root / "test/rs/matrix/test_elemwise.gen.rs", outputs)
			self.assertIn(root / "test/rs/matrix/test_blas.gen.rs", outputs)
			for operation in self.schema["operations"]:
				self.assertIn(f"pub fn {operation['name']}", api)
				for variant in GENERATOR.operation_variants(self.schema, operation):
					self.assertIn(variant["kernel_id"], registry)
					shader_path = root / f"src/slang/matrix/elemwise/{variant['source_stem']}.gen.slang"
					self.assertIn(shader_path, outputs)
					self.assertIn("void main(", outputs[shader_path])
					self.assertEqual(outputs[shader_path].count('[shader("compute")]'), 1)
			self.assertNotIn("entry_point", registry)
			self.assertNotIn("push_constant_size:", registry)
			for operation in self.blas_schema["operations"]:
				self.assertIn(f"pub fn {operation['name']}", blas_api)
				self.assertIn(operation["kernel_id"], registry)
				shader_path = root / f"src/slang/matrix/blas/{operation['source_stem']}.gen.slang"
				self.assertIn(shader_path, outputs)
				self.assertIn("void main(", outputs[shader_path])
				self.assertIn('[variant("tiled")]', outputs[shader_path])
			self.assertIn("dispatch_tile_size: [64, 64, 1]", registry)

	def test_blas_tile_geometry_is_validated(self):
		invalid = copy.deepcopy(self.blas_schema)
		invalid["output_tile_size"] = [32, 64, 1]
		with self.assertRaisesRegex(GENERATOR.SchemaError, "output_tile_size"):
			GENERATOR.validate_blas_schema(invalid)

	def test_stable_ids_must_be_unique_across_schema_families(self):
		invalid = copy.deepcopy(self.blas_schema)
		invalid["operations"][0]["stable_id"] = self.schema["operations"][0]["stable_id"]
		with self.assertRaisesRegex(GENERATOR.SchemaError, "across schemas"):
			GENERATOR.registry_entries(self.schema, invalid)

	def test_duplicate_stable_id_is_rejected_before_publication(self):
		invalid = copy.deepcopy(self.schema)
		invalid["operations"][1]["stable_id"] = invalid["operations"][0]["stable_id"]
		with self.assertRaisesRegex(GENERATOR.SchemaError, "duplicate"):
			GENERATOR.validate_schema(invalid)

	def test_scalar_contract_mismatch_is_rejected(self):
		invalid = copy.deepcopy(self.schema)
		invalid["operations"][0]["scalar_name"] = "scalar"
		with self.assertRaisesRegex(GENERATOR.SchemaError, "unary_scalar"):
			GENERATOR.validate_schema(invalid)

	def test_integer_variant_rejects_float_oracle_values(self):
		invalid = copy.deepcopy(self.schema)
		invalid["operations"][0]["additional_dtype_variants"][0]["test"]["expected"][0] = 1.5
		with self.assertRaisesRegex(GENERATOR.SchemaError, "i32 values"):
			GENERATOR.validate_schema(invalid)


if __name__ == "__main__":
	unittest.main()
