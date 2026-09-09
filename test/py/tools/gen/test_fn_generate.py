from __future__ import annotations

import copy
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[4]
GENERATOR_PATH = ROOT / "tools/gen/fn/generate.py"
SPEC = importlib.util.spec_from_file_location("oars_fn_generate", GENERATOR_PATH)
if SPEC is None or SPEC.loader is None:
	raise RuntimeError("could not load operation generator")
generate = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(generate)


class MlSchemaTests(unittest.TestCase):
	def setUp(self) -> None:
		self.path = ROOT / "tools/gen/fn/schema/ml_training.json"
		self.schema, self.digest = generate.load_ml_schema(self.path)

	def validate_copy(self, schema: dict) -> None:
		with tempfile.TemporaryDirectory() as directory:
			path = Path(directory) / "ml_training.json"
			path.write_text(json.dumps(schema))
			generate.load_ml_schema(path)

	def test_checked_in_ml_schema_is_complete(self) -> None:
		self.assertEqual(len(self.digest), 64)
		self.assertEqual(
			[operation["name"] for operation in self.schema["operations"]],
			[
				"linear",
				"linear_backward",
				"linear_parameter_backward",
				"cross_entropy",
				"cross_entropy_backward",
				"adamw",
				"adamw_graph_advance",
				"adamw_graph",
				"embedding",
				"embedding_backward",
				"rnn",
				"rnn_backward",
				"layer_norm",
				"layer_norm_backward",
				"layer_norm_parameter_backward",
				"gelu",
				"gelu_backward",
				"scaled_dot_product_attention_causal",
				"scaled_dot_product_attention_causal_probability_backward",
				"scaled_dot_product_attention_causal_backward",
				"qkv_projection_bias",
				"swiglu",
				"swiglu_backward",
				"gate_up_swiglu_bias",
			],
		)
		covered = {
			operation
			for record in self.schema["port_provenance"]
			for operation in record["operations"]
		}
		self.assertEqual(
			covered,
			{operation["name"] for operation in self.schema["operations"]},
		)

	def test_ml_schema_rejects_duplicate_stable_identity(self) -> None:
		schema = copy.deepcopy(self.schema)
		schema["operations"][1]["stable_id"] = schema["operations"][0]["stable_id"]
		with self.assertRaisesRegex(generate.SchemaError, "duplicate"):
			self.validate_copy(schema)

	def test_ml_schema_rejects_unknown_backward_relation(self) -> None:
		schema = copy.deepcopy(self.schema)
		schema["operations"][0]["differentiation"] = "missing_backward"
		with self.assertRaisesRegex(generate.SchemaError, "differentiation"):
			self.validate_copy(schema)

	def test_ml_schema_rejects_push_blocks_above_vulkan_minimum(self) -> None:
		schema = copy.deepcopy(self.schema)
		schema["operations"][0]["push_fields"] = [
			[f"field_{index}", "uint32"] for index in range(33)
		]
		with self.assertRaisesRegex(generate.SchemaError, "minimum Vulkan limit"):
			self.validate_copy(schema)

	def test_lowering_only_kernel_cannot_claim_a_semantic_contract(self) -> None:
		schema = copy.deepcopy(self.schema)
		kernel = schema["operations"][-1]
		kernel["contract"] = copy.deepcopy(schema["operations"][0]["contract"])
		with self.assertRaisesRegex(generate.SchemaError, "cannot own a semantic contract"):
			self.validate_copy(schema)

	def test_ml_schema_requires_complete_port_provenance(self) -> None:
		schema = copy.deepcopy(self.schema)
		schema["port_provenance"][0]["operations"].remove("linear")
		with self.assertRaisesRegex(generate.SchemaError, "cover every operation"):
			self.validate_copy(schema)

	def test_new_ml_work_cannot_claim_an_oa_donor(self) -> None:
		schema = copy.deepcopy(self.schema)
		schema["port_provenance"][0]["classification"] = "new"
		with self.assertRaisesRegex(generate.SchemaError, "empty for new work"):
			self.validate_copy(schema)


if __name__ == "__main__":
	unittest.main()
