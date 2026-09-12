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
		self.assertEqual(self.schema["retired_stable_ids"], [39, 115])
		self.assertEqual(
			[operation["name"] for operation in self.schema["operations"]],
			[
				"ppo_clipped_policy",
				"ppo_clipped_policy_backward",
				"linear",
				"linear_backward",
				"linear_parameter_backward",
				"cross_entropy",
				"cross_entropy_sum",
				"cross_entropy_backward",
				"masked_cross_entropy",
				"masked_cross_entropy_backward",
				"smooth_l1",
				"smooth_l1_backward",
				"smooth_l1_mean",
				"mse",
				"mse_backward",
				"l1",
				"l1_backward",
				"bce",
				"bce_backward",
				"sgd",
				"clip_grad_norm",
				"clip_grad_norm_scale",
				"sgd_momentum",
				"adam",
				"muon",
				"muon_vector",
				"muon_normalize",
				"muon_mat_mul_axpby",
				"muon_linear_combination",
				"muon_transpose",
				"muon_apply",
				"adamw",
				"adamw_graph_advance",
				"adamw_graph",
				"embedding",
				"embedding_backward",
				"rnn_scan",
				"rnn_scan_backward",
				"layer_norm",
				"layer_norm_backward",
				"rms_norm",
				"rms_norm_backward",
				"gelu",
				"gelu_backward",
				"silu",
				"silu_backward",
				"relu",
				"relu_backward",
				"tanh",
				"tanh_backward",
				"sigmoid",
				"sigmoid_backward",
				"leaky_relu",
				"leaky_relu_backward",
				"elu",
				"elu_backward",
				"mish",
				"mish_backward",
				"softplus",
				"softplus_backward",
				"qkv_projection_bias",
				"swiglu",
				"swiglu_backward",
				"gate_up_swiglu_bias",
				"adamw_many4",
				"adamw_many4_graph",
				"rope",
				"rope_backward",
				"avg_pool_2d",
				"avg_pool_2d_backward",
				"max_pool_2d",
				"max_pool_2d_backward",
				"adaptive_avg_pool_2d",
				"adaptive_avg_pool_2d_backward",
				"batch_norm_2d",
				"batch_norm_2d_with_stats",
				"batch_norm_2d_running_update",
				"batch_norm_2d_backward",
				"batch_norm_2d_input_backward",
				"upsample_2d",
				"upsample_2d_bilinear",
				"upsample_2d_backward",
				"upsample_2d_bilinear_backward",
				"conv_2d",
				"conv_2d_backward",
				"conv_2d_parameter_backward",
				"conv_1d",
				"conv_1d_bias_add",
				"conv_1d_transpose",
				"conv_1d_backward",
				"conv_1d_parameter_backward",
				"conv_transpose_1d",
				"conv_transpose_1d_backward",
				"conv_transpose_2d",
				"conv_transpose_2d_bias_add",
				"conv_transpose_2d_backward",
				"conv_transpose_2d_parameter_backward",
				"gru_scan",
				"gru_scan_backward",
				"gru_cell",
				"gru_cell_backward",
				"rnn_cell",
				"rnn_cell_backward",
				"bmm",
				"bmm_tiled_16",
				"bmm_nt",
				"bmm_nt_tiled_16",
				"bmm_tn",
				"bmm_tn_tiled_16",
				"split_heads",
				"merge_heads",
				"softmax_scaled_masked",
				"softmax_scaled_masked_n32",
				"softmax_scaled_masked_backward",
				"scaled_dot_product_attention",
				"sdpa_softmax_n32",
				"flash_attention_causal",
				"scaled_dot_product_attention_backward",
				"flash_attention_causal_backward_kv",
				"moe_route_weights",
				"moe_route_weights_backward",
				"moe_gather",
				"moe_gather_backward",
				"moe_combine",
				"moe_combine_backward",
				"grouped_linear_m",
				"grouped_linear_m_backward",
				"grouped_linear_m_parameter_backward",
				"silu_mul",
				"silu_mul_backward",
				"grouped_gemm_m",
				"grouped_gemm_m_backward",
				"grouped_gemm_m_parameter_backward",
				"linear_match",
				"euler_step",
				"masked_mse",
				"masked_mse_backward",
				"gae",
				"append",
				"reset",
				"append_batch",
				"sample",
				"dqn_target",
				"sac_target",
				"cart_pole_reset",
				"cart_pole_step",
				"lunar_lander_reset",
				"lunar_lander_step",
				"mamba3_siso",
				"mamba3_siso_backward",
				"mamba3_siso_backward_state",
				"mamba3_siso_backward_reverse_p16",
				"mamba3_siso_backward_finalize",
				"mamba3_siso_backward_reduce",
				"mamba3_siso_backward_group_reduce",
				"vq_assign",
				"vq_lookup",
				"vq_ema_update",
			],
		)
		covered = {
			operation
			for record in self.schema["port_provenance"]
			for operation in record["operations"]
		}
		self.assertEqual(
			covered,
			{
				operation["name"]
				for family in ("operations", "composite_contracts")
				for operation in self.schema.get(family, [])
			},
		)

	def test_lunar_lander_preserves_donor_vector_contracts(self) -> None:
		operations = {
			operation["name"]: operation for operation in self.schema["operations"]
		}
		reset = operations["lunar_lander_reset"]
		step = operations["lunar_lander_step"]
		self.assertEqual((reset["stable_id"], step["stable_id"]), (406, 407))
		self.assertEqual(reset["test"]["oracle"], "independent_scalar_lunar_lander_reset")
		self.assertEqual(step["test"]["oracle"], "independent_scalar_lunar_lander_step")
		self.assertEqual(step["contract"]["input_kinds"], ["matrix"] * 5)
		self.assertEqual(step["contract"]["output_kinds"], [])
		self.assertEqual(
			step["contract"]["variadic_input"],
			{"kind": "matrix", "minimum": 8},
		)
		self.assertEqual(
			step["contract"]["variadic_output"],
			{"kind": "matrix", "minimum": 8},
		)
		self.assertTrue(step["contract"]["aligned_variadic_aliases"])
		provenance = next(
			record
			for record in self.schema["port_provenance"]
			if record["family"] == "lunar_lander"
		)
		self.assertEqual(provenance["classification"], "mechanical_adaptation")

	def test_ml_schema_rejects_duplicate_stable_identity(self) -> None:
		schema = copy.deepcopy(self.schema)
		schema["operations"][1]["stable_id"] = schema["operations"][0]["stable_id"]
		with self.assertRaisesRegex(generate.SchemaError, "duplicate"):
			self.validate_copy(schema)

	def test_avg_pool_2d_preserves_the_donor_contract(self) -> None:
		forward = next(
			operation
			for operation in self.schema["operations"]
			if operation["name"] == "avg_pool_2d"
		)
		backward = next(
			operation
			for operation in self.schema["operations"]
			if operation["name"] == "avg_pool_2d_backward"
		)
		self.assertEqual((forward["stable_id"], backward["stable_id"]), (272, 273))
		self.assertEqual(forward["differentiation"], "avg_pool_2d_backward")
		self.assertEqual(forward["autograd"], {"mode": "manual", "family": "matrix/pool"})
		self.assertEqual(forward["workgroup_size"], [16, 16, 1])
		self.assertEqual(backward["workgroup_size"], [16, 16, 1])
		self.assertEqual(
			forward["source"],
			"src/slang/ml/nn/pooling/avg_pool_2d.slang",
		)
		self.assertEqual(
			backward["source"],
			"src/slang/ml/nn/pooling/avg_pool_2d_backward.slang",
		)

	def test_max_pool_2d_uses_exact_u32_argmax_storage(self) -> None:
		forward = next(
			operation
			for operation in self.schema["operations"]
			if operation["name"] == "max_pool_2d"
		)
		backward = next(
			operation
			for operation in self.schema["operations"]
			if operation["name"] == "max_pool_2d_backward"
		)
		self.assertEqual((forward["stable_id"], backward["stable_id"]), (274, 275))
		self.assertEqual(forward["contract"]["output_kinds"], ["matrix", "matrix"])
		self.assertEqual(forward["contract"]["dtype_rule"], "f32_with_u32_indices")
		self.assertEqual(backward["contract"]["dtype_rule"], "f32_with_u32_indices")

	def test_adaptive_avg_pool_2d_records_the_donor_correction(self) -> None:
		forward = next(
			operation
			for operation in self.schema["operations"]
			if operation["name"] == "adaptive_avg_pool_2d"
		)
		self.assertEqual(forward["stable_id"], 276)
		self.assertEqual(forward["differentiation"], "adaptive_avg_pool_2d_backward")
		provenance = next(
			record
			for record in self.schema["port_provenance"]
			if record["family"] == "adaptive_avg_pool_2d"
		)
		self.assertEqual(provenance["classification"], "replacement")
		self.assertIn("both spatial axes from height", provenance["reason"])

	def test_batch_norm_2d_records_numerical_and_eval_corrections(self) -> None:
		operations = {
			operation["name"]: operation for operation in self.schema["operations"]
		}
		self.assertEqual(
			[operations[name]["stable_id"] for name in (
				"batch_norm_2d",
				"batch_norm_2d_with_stats",
				"batch_norm_2d_running_update",
				"batch_norm_2d_backward",
				"batch_norm_2d_input_backward",
			)],
			[278, 279, 280, 281, 282],
		)
		self.assertEqual(
			operations["batch_norm_2d"]["differentiation"],
			"batch_norm_2d_backward",
		)
		self.assertTrue(operations["batch_norm_2d_input_backward"]["lowering_only"])
		provenance = next(
			record
			for record in self.schema["port_provenance"]
			if record["family"] == "batch_norm_2d"
		)
		self.assertEqual(provenance["classification"], "replacement")
		self.assertIn("centered variance", provenance["reason"])
		self.assertIn("inference mode", provenance["reason"])

	def test_upsample_2d_variants_share_semantic_operations(self) -> None:
		operations = {
			operation["name"]: operation for operation in self.schema["operations"]
		}
		self.assertEqual(
			[operations[name]["stable_id"] for name in (
				"upsample_2d",
				"upsample_2d_bilinear",
				"upsample_2d_backward",
				"upsample_2d_bilinear_backward",
			)],
			[283, 284, 285, 286],
		)
		self.assertEqual(
			operations["upsample_2d_bilinear"]["semantic_operation"],
			"upsample_2d",
		)
		self.assertEqual(
			operations["upsample_2d_bilinear_backward"]["semantic_operation"],
			"upsample_2d_backward",
		)
		provenance = next(
			record
			for record in self.schema["port_provenance"]
			if record["family"] == "upsample_2d"
		)
		self.assertIn("deterministic gather", provenance["reason"])

	def test_conv_2d_uses_one_split_semantic_adjoint(self) -> None:
		operations = {
			operation["name"]: operation for operation in self.schema["operations"]
		}
		self.assertEqual(
			[operations[name]["stable_id"] for name in (
				"conv_2d", "conv_2d_backward", "conv_2d_parameter_backward"
			)],
			[287, 288, 289],
		)
		self.assertEqual(operations["conv_2d"]["differentiation"], "conv_2d_backward")
		self.assertTrue(operations["conv_2d_parameter_backward"]["lowering_only"])
		self.assertEqual(
			operations["conv_2d_parameter_backward"]["semantic_operation"],
			"conv_2d_backward",
		)
		provenance = next(
			record
			for record in self.schema["port_provenance"]
			if record["family"] == "conv_2d"
		)
		self.assertIn("invalid bias identifiers", provenance["reason"])

	def test_conv_1d_preserves_the_im2col_gemm_lowering(self) -> None:
		operations = {
			operation["name"]: operation for operation in self.schema["operations"]
		}
		self.assertEqual(
			[operations[name]["stable_id"] for name in (
				"conv_1d",
				"conv_1d_bias_add",
				"conv_1d_transpose",
				"conv_1d_backward",
				"conv_1d_parameter_backward",
			)],
			[290, 291, 292, 293, 294],
		)
		self.assertEqual(operations["conv_1d"]["differentiation"], "conv_1d_backward")
		for name in ("conv_1d_bias_add", "conv_1d_transpose"):
			self.assertEqual(operations[name]["semantic_operation"], "conv_1d")
		self.assertEqual(
			operations["conv_1d_parameter_backward"]["semantic_operation"],
			"conv_1d_backward",
		)
		provenance = next(
			record
			for record in self.schema["port_provenance"]
			if record["family"] == "conv_1d"
		)
		self.assertIn("64x64x16 tiled MatMulNt", provenance["reason"])

	def test_conv_transpose_1d_reuses_the_conv_1d_adjoint_and_gemm(self) -> None:
		operations = {
			operation["name"]: operation for operation in self.schema["operations"]
		}
		self.assertEqual(
			[
				operations["conv_transpose_1d"]["stable_id"],
				operations["conv_transpose_1d_backward"]["stable_id"],
			],
			[295, 296],
		)
		self.assertEqual(
			operations["conv_transpose_1d"]["differentiation"],
			"conv_transpose_1d_backward",
		)
		provenance = next(
			record
			for record in self.schema["port_provenance"]
			if record["family"] == "conv_transpose_1d"
		)
		self.assertIn("64x64x16 tiled MatMulNt", provenance["reason"])
		self.assertIn("rather than introducing a second", provenance["reason"])

	def test_conv_transpose_2d_preserves_data_adjoint_and_bias_lowering(self) -> None:
		operations = {
			operation["name"]: operation for operation in self.schema["operations"]
		}
		self.assertEqual(
			[operations[name]["stable_id"] for name in (
				"conv_transpose_2d",
				"conv_transpose_2d_bias_add",
				"conv_transpose_2d_backward",
				"conv_transpose_2d_parameter_backward",
			)],
			[305, 306, 307, 308],
		)
		self.assertEqual(
			operations["conv_transpose_2d"]["differentiation"],
			"conv_transpose_2d_backward",
		)
		self.assertEqual(
			operations["conv_transpose_2d_bias_add"]["semantic_operation"],
			"conv_transpose_2d",
		)
		self.assertEqual(
			operations["conv_transpose_2d_parameter_backward"]["semantic_operation"],
			"conv_transpose_2d_backward",
		)
		provenance = next(
			record
			for record in self.schema["port_provenance"]
			if record["family"] == "conv_transpose_2d"
		)
		self.assertIn("unsigned coordinate underflow", provenance["reason"])

	def test_ml_schema_rejects_reuse_of_retired_stable_identity(self) -> None:
		schema = copy.deepcopy(self.schema)
		schema["operations"][0]["stable_id"] = 115
		with self.assertRaisesRegex(generate.SchemaError, "duplicate"):
			self.validate_copy(schema)

	def test_gru_scan_preserves_the_donor_dispatch_collapse(self) -> None:
		operations = {
			operation["name"]: operation for operation in self.schema["operations"]
		}
		self.assertEqual(
			[
				operations["gru_scan"]["stable_id"],
				operations["gru_scan_backward"]["stable_id"],
			],
			[311, 312],
		)
		self.assertEqual(
			operations["gru_scan"]["differentiation"],
			"gru_scan_backward",
		)
		self.assertEqual(
			operations["gru_scan"]["autograd"],
			{"mode": "manual", "family": "matrix/recurrent"},
		)
		provenance = next(
			record
			for record in self.schema["port_provenance"]
			if record["family"] == "gru"
		)
		self.assertIn("one-dispatch BPTT", provenance["reason"])
		self.assertIn("canonical Linear", provenance["reason"])
		self.assertEqual(
			[
				operations["gru_cell"]["stable_id"],
				operations["gru_cell_backward"]["stable_id"],
			],
			[313, 314],
		)
		self.assertEqual(
			operations["gru_cell"]["differentiation"],
			"gru_cell_backward",
		)

	def test_rnn_scan_preserves_the_donor_dispatch_collapse(self) -> None:
		operations = {
			operation["name"]: operation for operation in self.schema["operations"]
		}
		self.assertEqual(
			[
				operations["rnn_scan"]["stable_id"],
				operations["rnn_scan_backward"]["stable_id"],
				operations["rnn_cell"]["stable_id"],
				operations["rnn_cell_backward"]["stable_id"],
			],
			[29, 30, 315, 316],
		)
		self.assertEqual(
			operations["rnn_scan"]["differentiation"],
			"rnn_scan_backward",
		)
		self.assertEqual(
			operations["rnn_scan"]["autograd"],
			{"mode": "manual", "family": "matrix/recurrent"},
		)
		self.assertEqual(
			operations["rnn_cell"]["differentiation"],
			"rnn_cell_backward",
		)
		provenance = next(
			record
			for record in self.schema["port_provenance"]
			if record["family"] == "rnn"
		)
		self.assertEqual(provenance["classification"], "mechanical_adaptation")
		self.assertIn("one-dispatch BPTT", provenance["reason"])
		self.assertIn("canonical Linear", provenance["reason"])

	def test_bmm_family_preserves_semantics_and_private_provider_rows(self) -> None:
		operations = {
			operation["name"]: operation for operation in self.schema["operations"]
		}
		for semantic, generic_id, tiled_id in [
			("bmm", 317, 318),
			("bmm_nt", 319, 320),
			("bmm_tn", 321, 322),
		]:
			tiled = operations[f"{semantic}_tiled_16"]
			self.assertEqual(operations[semantic]["stable_id"], generic_id)
			self.assertEqual(operations[semantic]["differentiation"], "reverse")
			self.assertEqual(
				operations[semantic]["autograd"],
				{"mode": "manual", "family": "matrix/attention"},
			)
			self.assertEqual(tiled["stable_id"], tiled_id)
			self.assertTrue(tiled["lowering_only"])
			self.assertEqual(tiled["semantic_operation"], semantic)
			self.assertEqual(tiled["variant"], "tiled_16")
			self.assertEqual(
				tiled["physical_write"]["writes"][0]["extent"],
				"tile_16x16",
			)
		provenance = next(
			record
			for record in self.schema["port_provenance"]
			if record["family"] == "batched_matrix_multiply"
		)
		self.assertEqual(provenance["classification"], "mechanical_adaptation")
		self.assertIn("threshold", provenance["reason"])

	def test_head_transforms_preserve_the_donor_inverse_pair(self) -> None:
		operations = {
			operation["name"]: operation for operation in self.schema["operations"]
		}
		self.assertEqual(operations["split_heads"]["stable_id"], 323)
		self.assertEqual(operations["merge_heads"]["stable_id"], 324)
		for name in ["split_heads", "merge_heads"]:
			self.assertEqual(operations[name]["differentiation"], "reverse")
			self.assertEqual(
				operations[name]["autograd"],
				{"mode": "manual", "family": "matrix/attention"},
			)
		provenance = next(
			record
			for record in self.schema["port_provenance"]
			if record["family"] == "attention_head_transform"
		)
		self.assertIn("zero-copy differentiable reshape", provenance["reason"])

	def test_scaled_masked_softmax_preserves_private_narrow_routing(self) -> None:
		operations = {
			operation["name"]: operation for operation in self.schema["operations"]
		}
		self.assertEqual(operations["softmax_scaled_masked"]["stable_id"], 325)
		self.assertEqual(
			operations["softmax_scaled_masked"]["differentiation"],
			"softmax_scaled_masked_backward",
		)
		narrow = operations["softmax_scaled_masked_n32"]
		self.assertEqual(narrow["stable_id"], 326)
		self.assertTrue(narrow["lowering_only"])
		self.assertEqual(narrow["semantic_operation"], "softmax_scaled_masked")
		self.assertEqual(
			operations["softmax_scaled_masked_backward"]["stable_id"],
			327,
		)
		provenance = next(
			record
			for record in self.schema["port_provenance"]
			if record["family"] == "scaled_masked_softmax"
		)
		self.assertIn("detached mask", provenance["reason"])

	def test_standard_sdpa_preserves_optional_mask_and_saved_state(self) -> None:
		operations = {
			operation["name"]: operation for operation in self.schema["operations"]
		}
		operation = operations["scaled_dot_product_attention"]
		self.assertEqual(operation["stable_id"], 328)
		self.assertEqual(operation["contract"]["optional_inputs"], [3])
		self.assertEqual(operation["contract"]["output_kinds"], ["matrix", "matrix"])
		self.assertEqual(operation["differentiation"], "reverse")
		narrow = operations["sdpa_softmax_n32"]
		self.assertEqual(narrow["stable_id"], 329)
		self.assertTrue(narrow["lowering_only"])
		self.assertEqual(narrow["semantic_operation"], "scaled_dot_product_attention")
		flash = operations["flash_attention_causal"]
		self.assertEqual(flash["stable_id"], 330)
		self.assertTrue(flash["lowering_only"])
		self.assertEqual(flash["semantic_operation"], "scaled_dot_product_attention")
		backward = operations["scaled_dot_product_attention_backward"]
		self.assertEqual(backward["stable_id"], 331)
		self.assertEqual(backward["contract"]["optional_inputs"], [3])
		backward_kv = operations["flash_attention_causal_backward_kv"]
		self.assertEqual(backward_kv["stable_id"], 332)
		self.assertTrue(backward_kv["lowering_only"])
		self.assertEqual(
			backward_kv["semantic_operation"],
			"scaled_dot_product_attention_backward",
		)

	def test_ml_schema_rejects_non_optional_absent_index(self) -> None:
		schema = copy.deepcopy(self.schema)
		operation = next(
			operation
			for operation in schema["operations"]
			if operation["name"] == "scaled_dot_product_attention"
		)
		operation["contract"]["optional_inputs"] = [4]
		with self.assertRaisesRegex(generate.SchemaError, "optional_inputs"):
			self.validate_copy(schema)

	def test_ml_schema_rejects_unknown_backward_relation(self) -> None:
		schema = copy.deepcopy(self.schema)
		schema["operations"][0]["differentiation"] = "missing_backward"
		with self.assertRaisesRegex(generate.SchemaError, "differentiation"):
			self.validate_copy(schema)

	def test_ml_lowering_alias_must_name_a_semantic_operation(self) -> None:
		schema = copy.deepcopy(self.schema)
		operation = next(operation for operation in schema["operations"] if operation["name"] == "muon_vector")
		operation["semantic_operation"] = "missing_operation"
		with self.assertRaisesRegex(generate.SchemaError, "semantic_operation"):
			self.validate_copy(schema)

	def test_ml_semantic_operation_cannot_alias_another_contract(self) -> None:
		schema = copy.deepcopy(self.schema)
		schema["operations"][0]["semantic_operation"] = "adam"
		with self.assertRaisesRegex(generate.SchemaError, "only valid for a lowering-only"):
			self.validate_copy(schema)

	def test_loss_shader_rejects_a_generic_direction_basename(self) -> None:
		schema = copy.deepcopy(self.schema)
		operation = next(
			operation
			for operation in schema["operations"]
			if operation["name"] == "smooth_l1"
		)
		operation["source"] = "src/slang/ml/loss/smooth_l1/forward.slang"
		with self.assertRaisesRegex(generate.SchemaError, "retain the loss name"):
			self.validate_copy(schema)

	def test_optimizer_shader_rejects_a_flat_source_path(self) -> None:
		schema = copy.deepcopy(self.schema)
		operation = next(
			operation for operation in schema["operations"] if operation["name"] == "adamw"
		)
		operation["source"] = "src/slang/ml/optim/adamw.slang"
		with self.assertRaisesRegex(generate.SchemaError, "optimizer-family directory"):
			self.validate_copy(schema)

	def test_gradient_clip_shader_rejects_a_flat_source_path(self) -> None:
		schema = copy.deepcopy(self.schema)
		operation = next(
			operation
			for operation in schema["operations"]
			if operation["name"] == "clip_grad_norm"
		)
		operation["source"] = "src/slang/ml/optim/clip_grad_norm_reduce.slang"
		with self.assertRaisesRegex(generate.SchemaError, "optimizer-family directory"):
			self.validate_copy(schema)

	def test_variadic_alias_contract_requires_matching_tails(self) -> None:
		schema = copy.deepcopy(self.schema)
		operation = next(
			operation
			for operation in schema["operations"]
			if operation["name"] == "clip_grad_norm"
		)
		operation["contract"]["variadic_output"]["minimum"] = 2
		with self.assertRaisesRegex(generate.SchemaError, "identical input/output tails"):
			self.validate_copy(schema)

	def test_differentiable_forward_requires_autograd_policy(self) -> None:
		schema = copy.deepcopy(self.schema)
		del schema["operations"][0]["autograd"]
		with self.assertRaisesRegex(generate.SchemaError, "must own attachment policy"):
			self.validate_copy(schema)

	def test_backward_operation_rejects_autograd_policy(self) -> None:
		schema = copy.deepcopy(self.schema)
		schema["operations"][1]["autograd"] = {
			"mode": "manual",
			"family": "matrix/linear",
		}
		with self.assertRaisesRegex(generate.SchemaError, "valid only"):
			self.validate_copy(schema)

	def test_autograd_family_must_match_semantic_owner(self) -> None:
		schema = copy.deepcopy(self.schema)
		linear = next(
			operation for operation in schema["operations"] if operation["name"] == "linear"
		)
		linear["autograd"]["family"] = "loss/linear"
		with self.assertRaisesRegex(generate.SchemaError, "match its semantic owner"):
			self.validate_copy(schema)

	def test_generated_autograd_policy_requires_saved_state(self) -> None:
		schema = copy.deepcopy(self.schema)
		gelu = next(
			operation for operation in schema["operations"] if operation["name"] == "gelu"
		)
		del gelu["autograd"]["saved_matrices"]
		with self.assertRaisesRegex(generate.SchemaError, "fields are incomplete"):
			self.validate_copy(schema)

	def test_generated_autograd_policy_requires_original_inputs(self) -> None:
		schema = copy.deepcopy(self.schema)
		relu = next(
			operation for operation in schema["operations"] if operation["name"] == "relu"
		)
		del relu["autograd"]["inputs"]
		with self.assertRaisesRegex(generate.SchemaError, "fields are incomplete"):
			self.validate_copy(schema)

	def test_generated_autograd_policy_rejects_scalar_matrix_collision(self) -> None:
		schema = copy.deepcopy(self.schema)
		leaky_relu = next(
			operation
			for operation in schema["operations"]
			if operation["name"] == "leaky_relu"
		)
		leaky_relu["autograd"]["saved_scalars"] = [["input", "f32"]]
		with self.assertRaisesRegex(generate.SchemaError, "collides with a saved matrix"):
			self.validate_copy(schema)

	def test_generated_autograd_policy_rejects_unsupported_scalar_type(self) -> None:
		schema = copy.deepcopy(self.schema)
		elu = next(
			operation for operation in schema["operations"] if operation["name"] == "elu"
		)
		elu["autograd"]["saved_scalars"] = [["alpha", "f64"]]
		with self.assertRaisesRegex(generate.SchemaError, "supported Rust scalar type"):
			self.validate_copy(schema)

	def test_generated_autograd_policy_accepts_usize_state(self) -> None:
		schema = copy.deepcopy(self.schema)
		mse = next(operation for operation in schema["operations"] if operation["name"] == "mse")
		mse["autograd"]["saved_scalars"] = [["selected_count", "usize"]]
		self.validate_copy(schema)

	def test_ml_schema_requires_an_operation_owner(self) -> None:
		schema = copy.deepcopy(self.schema)
		del schema["operations"][0]["semantic_domain"]
		with self.assertRaisesRegex(generate.SchemaError, "semantic_domain"):
			self.validate_copy(schema)

	def test_ml_schema_rejects_a_lowering_only_operation_owner(self) -> None:
		schema = copy.deepcopy(self.schema)
		kernel = next(
			operation
			for operation in schema["operations"]
			if operation["name"] == "adamw_many4_graph"
		)
		kernel["semantic_domain"] = "ml::optim"
		with self.assertRaisesRegex(generate.SchemaError, "cannot own a semantic domain"):
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
		kernel = next(
			operation
			for operation in schema["operations"]
			if operation["name"] == "adamw_many4_graph"
		)
		kernel["contract"] = copy.deepcopy(schema["operations"][0]["contract"])
		with self.assertRaisesRegex(generate.SchemaError, "cannot own a semantic contract"):
			self.validate_copy(schema)

	def test_ml_schema_requires_complete_port_provenance(self) -> None:
		schema = copy.deepcopy(self.schema)
		linear = next(
			record for record in schema["port_provenance"] if "linear" in record["operations"]
		)
		linear["operations"].remove("linear")
		with self.assertRaisesRegex(generate.SchemaError, "cover every operation"):
			self.validate_copy(schema)

	def test_new_ml_work_cannot_claim_an_oa_donor(self) -> None:
		schema = copy.deepcopy(self.schema)
		schema["port_provenance"][0]["classification"] = "new"
		with self.assertRaisesRegex(generate.SchemaError, "empty for new work"):
			self.validate_copy(schema)


if __name__ == "__main__":
	unittest.main()
