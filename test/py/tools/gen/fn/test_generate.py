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
REDUCE_SCHEMA_PATH = GENERATOR_ROOT / "schema/matrix_reduce.json"
RNG_SCHEMA_PATH = GENERATOR_ROOT / "schema/matrix_rng.json"
ML_SCHEMA_PATH = GENERATOR_ROOT / "schema/ml_training.json"
AUDIO_SCHEMA_PATH = GENERATOR_ROOT / "schema/audio.json"
CRYPTOGRAPHY_HASH_SCHEMA_PATH = GENERATOR_ROOT / "schema/cryptography_hash.json"
IMAGE_SCHEMA_PATH = GENERATOR_ROOT / "schema/image.json"
VISION_SCHEMA_PATH = GENERATOR_ROOT / "schema/vision_detection.json"
REPOSITORY_ROOT = GENERATOR_ROOT.parents[2]
SLANG_ROOT = REPOSITORY_ROOT / "src/slang"


class GeneratorTests(unittest.TestCase):
	def setUp(self):
		self.schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
		self.blas_schema = json.loads(BLAS_SCHEMA_PATH.read_text(encoding="utf-8"))
		self.reduce_schema = json.loads(REDUCE_SCHEMA_PATH.read_text(encoding="utf-8"))
		self.rng_schema = json.loads(RNG_SCHEMA_PATH.read_text(encoding="utf-8"))
		self.ml_schema = json.loads(ML_SCHEMA_PATH.read_text(encoding="utf-8"))
		self.audio_schema = json.loads(AUDIO_SCHEMA_PATH.read_text(encoding="utf-8"))
		self.cryptography_hash_schema = json.loads(
			CRYPTOGRAPHY_HASH_SCHEMA_PATH.read_text(encoding="utf-8")
		)
		self.image_schema = json.loads(IMAGE_SCHEMA_PATH.read_text(encoding="utf-8"))
		self.vision_schema = json.loads(VISION_SCHEMA_PATH.read_text(encoding="utf-8"))

	def test_schema_generates_every_operation_surface(self):
		GENERATOR.validate_schema(self.schema)
		GENERATOR.validate_blas_schema(self.blas_schema)
		GENERATOR.validate_reduce_schema(self.reduce_schema)
		GENERATOR.validate_rng_schema(self.rng_schema)
		GENERATOR.validate_ml_schema(self.ml_schema)
		GENERATOR.validate_audio_schema(self.audio_schema)
		GENERATOR.validate_cryptography_hash_schema(self.cryptography_hash_schema)
		GENERATOR.validate_image_schema(self.image_schema)
		GENERATOR.validate_vision_schema(self.vision_schema)
		with tempfile.TemporaryDirectory() as directory:
			root = Path(directory)
			outputs = GENERATOR.expected_outputs(
				root,
				self.schema,
				"0" * 64,
				self.blas_schema,
				"1" * 64,
				self.reduce_schema,
				"6" * 64,
				self.rng_schema,
				"2" * 64,
				self.ml_schema,
				"3" * 64,
				self.audio_schema,
				"4" * 64,
				self.cryptography_hash_schema,
				"5" * 64,
				self.image_schema,
				"7" * 64,
				self.vision_schema,
				"8" * 64,
			)
			variant_count = sum(
				len(GENERATOR.operation_variants(self.schema, operation))
				for operation in self.schema["operations"]
			)
			autograd_family_count = len({
				operation["autograd"]["family"]
				for operation in self.ml_schema["operations"]
				if operation.get("autograd", {}).get("mode") == "generated"
			})
			self.assertEqual(len(outputs), variant_count + 10 + autograd_family_count)
			api = outputs[root / "src/rs/matrix/elemwise.gen.rs"]
			blas_api = outputs[root / "src/rs/matrix/blas.gen.rs"]
			reduce_api = outputs[root / "src/rs/matrix/reduce.gen.rs"]
			registry = outputs[root / "src/rs/runtime/shader/registry.gen.rs"]
			operations = outputs[root / "src/rs/core/operation/generated.rs"]
			dnn_roles = outputs[root / "src/rs/runtime/dnn/generated.rs"]
			activation_autograd = outputs[
				root / "src/rs/ml/autograd/matrix/activation.gen.rs"
			]
			loss_autograd = outputs[root / "src/rs/ml/autograd/loss/core.gen.rs"]
			self.assertIn(root / "test/rs/matrix/test_elemwise.gen.rs", outputs)
			self.assertIn(root / "test/rs/matrix/test_blas.gen.rs", outputs)
			self.assertIn(root / "test/rs/matrix/test_reduce.gen.rs", outputs)
			for operation in self.schema["operations"]:
				self.assertIn(f"pub fn {operation['name']}", api)
				self.assertIn(
					f"pub const {GENERATOR.rust_const_name(operation['name'])}", operations
				)
				self.assertIn(
					f"crate::core::operation::matrix::{GENERATOR.rust_const_name(operation['name'])}",
					api,
				)
				for variant in GENERATOR.operation_variants(self.schema, operation):
					self.assertIn(variant["kernel_id"], registry)
					shader_path = root / f"src/slang/matrix/elemwise/{variant['source_stem']}.gen.slang"
					self.assertIn(shader_path, outputs)
					self.assertIn("void main(", outputs[shader_path])
					self.assertEqual(outputs[shader_path].count('[shader("compute")]'), 1)
			self.assertNotIn("entry_point", registry)
			self.assertNotIn("push_constant_size:", registry)
			self.assertIn('Self::MatrixAddF32 => "matrix.add.f32"', registry)
			self.assertIn('Self::MatrixAddI32 => "int32"', registry)
			for operation in self.blas_schema["operations"]:
				self.assertIn(f"pub fn {operation['name']}", blas_api)
				self.assertIn(
					f"pub const {GENERATOR.rust_const_name(operation['name'])}", operations
				)
				self.assertIn(operation["kernel_id"], registry)
				shader_path = root / f"src/slang/matrix/blas/{operation['source_stem']}.gen.slang"
				self.assertIn(shader_path, outputs)
				self.assertIn("void main(", outputs[shader_path])
				self.assertIn('[variant("tiled")]', outputs[shader_path])
			self.assertIn("dispatch_tile_size: [64, 64, 1]", registry)
			self.assertIn("physical_write: None", registry)
			for operation in self.reduce_schema["operations"]:
				self.assertIn(operation["kernel_id"], registry)
				if not operation.get("lowering_only", False):
					self.assertIn(
						f"pub const {GENERATOR.rust_const_name(operation['name'])}",
						operations,
					)
			self.assertIn("pub fn softmax", reduce_api)
			self.assertIn("pub(crate) fn softmax_backward", reduce_api)
			self.assertIn("pub fn log_softmax", reduce_api)
			self.assertIn("pub(crate) fn log_softmax_backward", reduce_api)
			self.assertIn("pub fn sum", reduce_api)
			self.assertIn("pub(crate) fn sum_backward", reduce_api)
			self.assertIn(
				"Self::MatrixSumAxisF32 => Some(crate::core::operation::matrix::SUM)",
				registry,
			)
			self.assertIn("domain: LogicalWriteDomain::OutputElements", registry)
			self.assertIn("partition: WritePartition::ExclusivePerInvocation", registry)
			self.assertIn("workspace: WorkspacePartition::ExclusivePerWorkgroup", registry)
			self.assertNotIn("pub const SUM_AXIS", operations)
			for operation in self.rng_schema["operations"]:
				self.assertIn(operation["kernel_id"], registry)
				if "contract" in operation:
					self.assertIn(
						f"pub const {GENERATOR.rust_const_name(operation['name'])}", operations
					)
			self.assertIn(
				"Self::MatrixPhiloxUniformF32 => TrainingReplayRole::FrozenRng",
				registry,
			)
			self.assertRegex(
				registry,
				r"Self::MatrixPhiloxUniformReplayF32 => (?:\{\s*)?Some\(crate::core::operation::matrix::PHILOX_UNIFORM\)",
			)
			for contract in self.cryptography_hash_schema["contracts"]:
				self.assertIn(
					f"pub const {GENERATOR.rust_const_name(contract['name'])}",
					operations,
				)
			for kernel in self.cryptography_hash_schema["kernels"]:
				self.assertIn(kernel["kernel_id"], registry)
			for contract in self.image_schema["contracts"]:
				self.assertIn(
					f"pub const {GENERATOR.rust_const_name(contract['name'])}",
					operations,
				)
			for kernel in self.image_schema["kernels"]:
				self.assertIn(kernel["kernel_id"], registry)
			self.assertIn('"oa::image::resize"', operations)
			self.assertIn("OpValueKind::Image", operations)
			self.assertIn('Self::ImageResizeBilinearF32 => "image.resize_bilinear.f32"', registry)
			for contract in self.vision_schema["contracts"]:
				self.assertIn(
					f"pub const {GENERATOR.rust_const_name(contract['name'])}",
					operations,
				)
			for kernel in self.vision_schema["kernels"]:
				self.assertIn(kernel["kernel_id"], registry)
			self.assertIn('"oa::vision::box_iou"', operations)
			self.assertIn('Self::VisionBoxIouF32 => "vision.box_iou.f32"', registry)
			self.assertIn('Self::CryptographyShake256U8 => "uint8"', registry)
			self.assertIn(
				"pub const DROPOUT: OperationContract",
				operations,
			)
			self.assertIn(
				".with_differentiation(OpDifferentiation::Reverse)",
				operations[operations.index("pub const DROPOUT: OperationContract"):],
			)
			for operation in self.ml_schema["operations"]:
				self.assertIn(operation["kernel_id"], registry)
				if not operation.get("lowering_only", False):
					self.assertIn(
						f"pub const {GENERATOR.rust_const_name(operation['name'])}", operations
					)
			self.assertIn("pub(crate) enum TrainingReplayRole", registry)
			self.assertIn(
				"Self::MlAdamWF32 => TrainingReplayRole::HostSteppedOptimizer",
				registry,
			)
			self.assertIn(
				"Self::MlAdamWGraphAdvanceU32 => TrainingReplayRole::OptimizerStateAdvance",
				registry,
			)
			self.assertIn(
				"Self::MlAdamWGraphF32 => TrainingReplayRole::OptimizerStateUpdate",
				registry,
			)
			self.assertIn('Self::MlAdamWGraphF32 => "ml.optim.adamw_graph.f32"', registry)
			self.assertIn(
				"Self::MlAdamWGraphF32 => Some(crate::core::operation::ml::ADAMW_GRAPH)",
				registry,
			)
			self.assertIn('Self::MlQkvProjectionBiasF32 => "ml.qkv_projection_bias.f32"', registry)
			self.assertIn("Self::MlQkvProjectionBiasF32 => None", registry)
			self.assertIn('Self::MlSwigluF32 => "ml.matrix.swiglu.f32"', registry)
			self.assertIn(
				'Self::MlGateUpSwigluBiasF32 => "ml.gate_up_swiglu_bias.f32"',
				registry,
			)
			self.assertIn("Self::MlGateUpSwigluBiasF32 => None", registry)
			self.assertNotIn("pub const QKV_PROJECTION_BIAS", operations)
			self.assertNotIn("pub const GATE_UP_SWIGLU_BIAS", operations)
			self.assertIn("DnnOpType::Add", dnn_roles)
			self.assertIn("DnnOpType::Multiply", dnn_roles)
			self.assertIn("DnnOpType::Matmul", dnn_roles)
			self.assertIn("DnnProvider::QkvProjectionGroup.bit()", dnn_roles)
			self.assertIn("DnnOpType::GatedMultiply", dnn_roles)
			self.assertIn("crate::core::operation::ml::LINEAR", dnn_roles)
			self.assertIn("DnnEpilogue::Bias", dnn_roles)
			self.assertIn('"oa::ml::matrix::linear"', operations)
			self.assertIn('"oa::ml::loss::cross_entropy"', operations)
			self.assertIn('"oa::ml::optim::adamw"', operations)
			self.assertIn("pub(in crate::ml) fn record_gelu", activation_autograd)
			self.assertIn("Node::Gelu", activation_autograd)
			self.assertIn("pub(in crate::ml) fn record_relu", activation_autograd)
			self.assertIn("input: &Matrix, output: &Matrix, result: &Matrix", activation_autograd)
			self.assertIn("Node::Relu", activation_autograd)
			self.assertIn("pub(in crate::ml) fn record_leaky_relu", activation_autograd)
			self.assertIn("alpha: f32, result: &Matrix", activation_autograd)
			self.assertIn("Node::LeakyRelu", activation_autograd)
			self.assertIn("pub(in crate::ml) fn record_swiglu", activation_autograd)
			self.assertIn("Node::Swiglu", activation_autograd)
			self.assertNotIn("record_linear", activation_autograd)
			self.assertIn("pub(in crate::ml) fn record_cross_entropy", loss_autograd)
			self.assertIn("pub(in crate::ml) fn record_masked_cross_entropy", loss_autograd)
			self.assertIn("valid_count: usize", loss_autograd)
			self.assertIn('"oa::ml::loss::masked_cross_entropy"', operations)
			for loss in ("smooth_l1", "mse", "l1", "bce"):
				self.assertIn(f"pub(in crate::ml) fn record_{loss}", loss_autograd)
				self.assertIn(f'"oa::ml::loss::{loss}"', operations)
			self.assertIn('Self::MlMseF32 => "ml.loss.mse.f32"', registry)
			self.assertIn(
				'Self::MlSmoothL1MeanF32 => "ml.smooth_l1_mean.f32"',
				registry,
			)
			self.assertIn(
				"Self::MlSmoothL1MeanF32 => Some(crate::core::operation::ml::SMOOTH_L1)",
				registry,
			)
			self.assertIn("Node::CrossEntropy", loss_autograd)

	def test_every_slang_entry_point_has_one_schema_owner(self):
		schema_sources = set()
		for schema in (
			self.reduce_schema,
			self.rng_schema,
			self.ml_schema,
			self.audio_schema,
			self.cryptography_hash_schema,
			self.image_schema,
			self.vision_schema,
		):
			for row in schema.get("operations", []) + schema.get("kernels", []):
				if source := row.get("source"):
					schema_sources.add(source)

		generated_sources = {
			f"src/slang/matrix/elemwise/{variant['source_stem']}.gen.slang"
			for operation in self.schema["operations"]
			for variant in GENERATOR.operation_variants(self.schema, operation)
		}
		generated_sources.update(
			f"src/slang/matrix/blas/{operation['source_stem']}.gen.slang"
			for operation in self.blas_schema["operations"]
		)
		expected = schema_sources | generated_sources
		actual = {
			path.relative_to(REPOSITORY_ROOT).as_posix()
			for path in SLANG_ROOT.rglob("*.slang")
			if "void main(" in path.read_text(encoding="utf-8")
		}

		self.assertEqual(actual, expected)
		self.assertTrue(all((REPOSITORY_ROOT / source).is_file() for source in expected))
		for path in SLANG_ROOT.rglob("*.slang"):
			text = path.read_text(encoding="utf-8")
			self.assertNotIn("TODO: Add common", text, path.as_posix())

	def test_blas_tile_geometry_is_validated(self):
		invalid = copy.deepcopy(self.blas_schema)
		invalid["output_tile_size"] = [32, 64, 1]
		with self.assertRaisesRegex(GENERATOR.SchemaError, "output_tile_size"):
			GENERATOR.validate_blas_schema(invalid)

	def test_reduce_schema_preserves_the_donor_axis_contract(self):
		invalid = copy.deepcopy(self.reduce_schema)
		invalid["operations"][0]["semantic_attributes"] = []
		with self.assertRaisesRegex(GENERATOR.SchemaError, "signed dim attribute"):
			GENERATOR.validate_reduce_schema(invalid)

	def test_reduce_schema_owns_log_softmax_and_its_adjoint(self):
		operations = self.reduce_schema["operations"]
		self.assertEqual(
			[operation["name"] for operation in operations[2:4]],
			["log_softmax", "log_softmax_backward"],
		)
		self.assertEqual(operations[2]["stable_id"], 270)
		self.assertEqual(operations[3]["stable_id"], 271)
		self.assertEqual(operations[2]["differentiation"], "reverse")
		self.assertEqual(operations[3]["differentiation"], "none")

	def test_reduce_schema_owns_categorical_accuracy_counters(self):
		operations = self.reduce_schema["operations"]
		self.assertEqual(
			[operation["name"] for operation in operations[-2:]],
			["categorical_accuracy_count", "masked_categorical_accuracy_count"],
		)
		self.assertEqual(operations[-2]["contract"]["input_kinds"], ["matrix", "class_indices"])
		self.assertEqual(
			operations[-1]["contract"]["input_kinds"],
			["matrix", "class_indices", "matrix"],
		)

	def test_reduce_schema_requires_complete_physical_write_ownership(self):
		missing = copy.deepcopy(self.reduce_schema)
		del missing["operations"][0]["physical_write"]["writes"][0]["collision"]
		with self.assertRaisesRegex(GENERATOR.SchemaError, "fields are incomplete"):
			GENERATOR.validate_reduce_schema(missing)

		contradictory = copy.deepcopy(self.reduce_schema)
		contradictory["operations"][0]["physical_write"]["writes"][0][
			"partition"
		] = "exclusive_per_invocation"
		with self.assertRaisesRegex(GENERATOR.SchemaError, "contradictory"):
			GENERATOR.validate_reduce_schema(contradictory)

		unsupported = copy.deepcopy(self.reduce_schema)
		unsupported["operations"][0]["physical_write"]["writes"][0][
			"collision"
		] = "magic"
		with self.assertRaisesRegex(GENERATOR.SchemaError, "collision is unsupported"):
			GENERATOR.validate_reduce_schema(unsupported)

		duplicate = copy.deepcopy(self.reduce_schema)
		duplicate_write = copy.deepcopy(
			duplicate["operations"][0]["physical_write"]["writes"][0]
		)
		duplicate["operations"][0]["physical_write"]["writes"].append(duplicate_write)
		with self.assertRaisesRegex(GENERATOR.SchemaError, "duplicate"):
			GENERATOR.validate_reduce_schema(duplicate)

	def test_normalization_candidates_require_physical_write_ownership(self):
		invalid = copy.deepcopy(self.ml_schema)
		layer_norm = next(
			operation for operation in invalid["operations"] if operation["name"] == "layer_norm"
		)
		del layer_norm["physical_write"]
		with self.assertRaisesRegex(GENERATOR.SchemaError, "physical_write is required"):
			GENERATOR.validate_ml_schema(invalid)

	def test_image_kernels_require_physical_write_ownership(self):
		invalid = copy.deepcopy(self.image_schema)
		del invalid["kernels"][0]["physical_write"]
		with self.assertRaisesRegex(GENERATOR.SchemaError, "physical_write is required"):
			GENERATOR.validate_image_schema(invalid)

	def test_vision_schema_owns_the_complete_detection_surface(self):
		self.assertEqual(
			[contract["name"] for contract in self.vision_schema["contracts"]],
			[
				"box_iou",
				"nms",
				"confusion_matrix",
				"binary_mask_counts",
				"evaluate",
				"evaluate_segmentation",
			],
		)
		self.assertEqual(len(self.vision_schema["kernels"]), 12)

	def test_atomic_writes_require_shared_u32_collision_contract(self):
		invalid = copy.deepcopy(self.vision_schema)
		invalid["kernels"][2]["physical_write"]["writes"][0]["collision"] = "exclusive"
		with self.assertRaisesRegex(GENERATOR.SchemaError, "shared contributors"):
			GENERATOR.validate_vision_schema(invalid)

		invalid = copy.deepcopy(self.vision_schema)
		invalid["kernels"][2]["physical_write"]["writes"][0][
			"partition"
		] = "exclusive_per_invocation"
		with self.assertRaisesRegex(GENERATOR.SchemaError, "atomic_u32 requires"):
			GENERATOR.validate_vision_schema(invalid)

	def test_unknown_dnn_provider_is_rejected(self):
		invalid = copy.deepcopy(self.blas_schema)
		invalid["operations"][0]["dnn"]["providers"] = ["magic_vendor"]
		with self.assertRaisesRegex(GENERATOR.SchemaError, "dnn.providers"):
			GENERATOR.validate_blas_schema(invalid)

	def test_invalid_semantic_attribute_is_rejected(self):
		invalid = copy.deepcopy(self.ml_schema)
		invalid["operations"][0]["semantic_attributes"] = [["epsilon", "pointer"]]
		with self.assertRaisesRegex(GENERATOR.SchemaError, "kind is unsupported"):
			GENERATOR.validate_ml_schema(invalid)

	def test_unknown_training_replay_role_is_rejected(self):
		invalid = copy.deepcopy(self.ml_schema)
		invalid["operations"][0]["training_replay_role"] = "host_magic"
		with self.assertRaisesRegex(GENERATOR.SchemaError, "training_replay_role"):
			GENERATOR.validate_ml_schema(invalid)

	def test_unknown_rng_differentiation_is_rejected(self):
		invalid = copy.deepcopy(self.rng_schema)
		next(
			operation for operation in invalid["operations"] if operation["name"] == "dropout"
		)["differentiation"] = "symbolic_magic"
		with self.assertRaisesRegex(GENERATOR.SchemaError, "differentiation"):
			GENERATOR.validate_rng_schema(invalid)

	def test_stable_ids_must_be_unique_across_schema_families(self):
		invalid = copy.deepcopy(self.blas_schema)
		invalid["operations"][0]["stable_id"] = self.schema["operations"][0]["stable_id"]
		with self.assertRaisesRegex(GENERATOR.SchemaError, "across schemas"):
			GENERATOR.registry_entries(
				self.schema,
				invalid,
				self.reduce_schema,
				self.rng_schema,
				self.ml_schema,
				self.audio_schema,
				self.cryptography_hash_schema,
				self.image_schema,
				self.vision_schema,
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
