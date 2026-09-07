import copy
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


GENERATOR_PATH = Path(__file__).resolve().parents[1] / "generate.py"
SPEC = importlib.util.spec_from_file_location("oa_fn_generate", GENERATOR_PATH)
assert SPEC is not None and SPEC.loader is not None
GENERATOR = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(GENERATOR)
SCHEMA_PATH = Path(__file__).resolve().parents[1] / "schema/matrix_elemwise.json"


class GeneratorTests(unittest.TestCase):
	def setUp(self):
		self.schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))

	def test_schema_generates_every_operation_surface(self):
		GENERATOR.validate_schema(self.schema)
		with tempfile.TemporaryDirectory() as directory:
			root = Path(directory)
			outputs = GENERATOR.expected_outputs(root, self.schema, "0" * 64)
			variant_count = sum(
				len(GENERATOR.operation_variants(self.schema, operation))
				for operation in self.schema["operations"]
			)
			self.assertEqual(len(outputs), variant_count + 3)
			api = outputs[root / "src/rs/matrix/elemwise.gen.rs"]
			registry = outputs[root / "src/rs/runtime/shader/generated.rs"]
			for operation in self.schema["operations"]:
				self.assertIn(f"pub fn {operation['name']}", api)
				for variant in GENERATOR.operation_variants(self.schema, operation):
					self.assertIn(variant["kernel_id"], registry)
					self.assertIn(
						root / f"src/slang/matrix/elemwise/{variant['source_stem']}.gen.slang",
						outputs,
					)

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
