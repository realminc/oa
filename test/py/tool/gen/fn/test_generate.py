import copy
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


GENERATOR_ROOT = Path(__file__).resolve().parents[5] / "tool/gen/fn"
GENERATOR_PATH = GENERATOR_ROOT / "generate.py"
SPEC = importlib.util.spec_from_file_location("oa_fn_generate", GENERATOR_PATH)
assert SPEC is not None and SPEC.loader is not None
GENERATOR = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(GENERATOR)
SCHEMA_PATH = GENERATOR_ROOT / "schema/matrix/matrix_elemwise.json"
BLAS_SCHEMA_PATH = GENERATOR_ROOT / "schema/matrix/matrix_blas.json"
REDUCE_SCHEMA_PATH = GENERATOR_ROOT / "schema/matrix/matrix_reduce.json"
RNG_SCHEMA_PATH = GENERATOR_ROOT / "schema/matrix/matrix_rng.json"
VIEW_SCHEMA_PATH = GENERATOR_ROOT / "schema/matrix/matrix_view.json"
INDEX_SCHEMA_PATH = GENERATOR_ROOT / "schema/matrix/matrix_index.json"
ML_SCHEMA_PATH = GENERATOR_ROOT / "schema/ml/ml_training.json"
ACTIVATION_SCHEMA_PATH = GENERATOR_ROOT / "schema/ml/ml_activation.json"
AUDIO_SCHEMA_PATH = GENERATOR_ROOT / "schema/audio/audio.json"
CRYPTOGRAPHY_HASH_SCHEMA_PATH = GENERATOR_ROOT / "schema/cryptography/cryptography_hash.json"
CRYPTOGRAPHY_PQC_SCHEMA_PATH = GENERATOR_ROOT / "schema/cryptography/cryptography_pqc.json"
IMAGE_SCHEMA_PATH = GENERATOR_ROOT / "schema/image/image.json"
VISION_SCHEMA_PATH = GENERATOR_ROOT / "schema/vision/vision_detection.json"
UI_SCHEMA_PATH = GENERATOR_ROOT / "schema/ui/ui.json"
REPOSITORY_ROOT = GENERATOR_ROOT.parents[2]
SLANG_ROOT = REPOSITORY_ROOT / "src/slang"
SDK_SLANG_ROOT = REPOSITORY_ROOT / "sdk/rs/slang"


class GeneratorTests(unittest.TestCase):
	def setUp(self):
		self.schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
		self.blas_schema = json.loads(BLAS_SCHEMA_PATH.read_text(encoding="utf-8"))
		self.reduce_schema = json.loads(REDUCE_SCHEMA_PATH.read_text(encoding="utf-8"))
		self.rng_schema = json.loads(RNG_SCHEMA_PATH.read_text(encoding="utf-8"))
		self.index_schema = json.loads(INDEX_SCHEMA_PATH.read_text(encoding="utf-8"))
		self.view_schema = json.loads(VIEW_SCHEMA_PATH.read_text(encoding="utf-8"))
		self.ml_schema = json.loads(ML_SCHEMA_PATH.read_text(encoding="utf-8"))
		self.activation_schema = json.loads(ACTIVATION_SCHEMA_PATH.read_text(encoding="utf-8"))
		self.audio_schema = json.loads(AUDIO_SCHEMA_PATH.read_text(encoding="utf-8"))
		self.cryptography_hash_schema = json.loads(
			CRYPTOGRAPHY_HASH_SCHEMA_PATH.read_text(encoding="utf-8")
		)
		self.cryptography_pqc_schema = json.loads(
			CRYPTOGRAPHY_PQC_SCHEMA_PATH.read_text(encoding="utf-8")
		)
		self.image_schema = json.loads(IMAGE_SCHEMA_PATH.read_text(encoding="utf-8"))
		self.vision_schema = json.loads(VISION_SCHEMA_PATH.read_text(encoding="utf-8"))
		self.ui_schema = json.loads(UI_SCHEMA_PATH.read_text(encoding="utf-8"))

	def test_ml_activation_templates_emit_complete_operations(self):
		api = GENERATOR.generate_ml_activation_api(self.activation_schema, "0" * 64)
		for helper in ("fn record(", "fn validate_f32(", "fn as_u32(", "\n\t\trecord("):
			self.assertNotIn(helper, api)
		self.assertNotIn("{{", api)
		for operation in self.activation_schema["operations"]:
			body = GENERATOR._ml_activation_fn(operation)
			self.assertIn("let contract = crate::core::operation::ml::", body)
			self.assertIn("engine.record_semantic(", body)
			self.assertIn("u32::try_from(", body)
			self.assertIn(f"KernelId::{operation['kernel_id']}", body)
			if "autograd" in operation:
				self.assertEqual(body.count(f"autograd::record_{operation['name']}("), 1)
			if operation["kind"] in ("unary_backward", "unary_scalar_backward"):
				self.assertIn(f"BufferBinding::read(saved_{operation['saved']}.storage())", body)
			if operation["kind"] != "silu_mul" and "backward" not in operation["kind"]:
				self.assertIn("if element_count != 0", body)

	def test_ml_activation_templates_reject_incomplete_substitution(self):
		with self.assertRaises(ValueError):
			GENERATOR._render_ml_activation_template("unary", {"name": "gelu"})

	def test_elementwise_module_is_complete_and_routes_are_schema_owned(self):
		api = GENERATOR.generate_api(self.schema, "0" * 64)
		self.assertIn("struct BroadcastLayout", api)
		self.assertIn("fn select_kernel(", api)
		self.assertNotIn("include!", api)
		self.assertNotIn("contract.hash()", api)
		self.assertNotIn("{{items}}", api)
		for operation in self.schema["operations"]:
			body = api.split(f"pub fn {operation['name']}(", 1)[1].split("\n///", 1)[0]
			for candidate in operation.get("additional_lowering_variants", []):
				self.assertIn(f"KernelId::{candidate['kernel_id']}", body)
		operation = self.schema["operations"][0]
		operation["additional_lowering_variants"][0]["kernel_id"] = "SchemaSelectedBroadcast"
		api = GENERATOR.generate_api(self.schema, "0" * 64)
		self.assertIn("KernelId::SchemaSelectedBroadcast", api)
		self.assertNotIn("KernelId::MatrixAddBroadcastF32", api)

	def test_elementwise_retains_exact_shape_only_schema_support(self):
		operation = self.schema["operations"][0]
		operation["shape_rule"] = "equal_no_broadcast"
		del operation["additional_lowering_variants"]
		GENERATOR.validate_schema(self.schema)
		api = GENERATOR.generate_api(self.schema, "0" * 64)
		self.assertIn("requires equal shapes", api)
		self.assertNotIn("KernelId::MatrixAddBroadcastF32", api)

	def test_elementwise_rejects_missing_or_contradictory_broadcast_routes(self):
		for mutate in (
			lambda op: op.pop("additional_lowering_variants"),
			lambda op: op["additional_lowering_variants"].pop(),
			lambda op: op.update(shape_rule="equal_no_broadcast"),
		):
			with self.subTest(mutate=mutate):
				schema = copy.deepcopy(self.schema)
				mutate(schema["operations"][0])
				with self.assertRaises(GENERATOR.SchemaError):
					GENERATOR.validate_schema(schema)

	def test_pqc_parameter_identity_rejects_mismatched_or_duplicate_sets(self):
		for field, value in (("parameter_set", 66), ("parameter_set", True),
				("name", "ml_dsa_44_verify"), ("semantic_operation", "ml_dsa_44_verify_batch")):
			with self.subTest(field=field, value=value):
				schema = copy.deepcopy(self.cryptography_pqc_schema)
				schema["kernels"][0][field] = value
				with self.assertRaises(ValueError):
					GENERATOR.validate_cryptography_pqc_schema(schema)
		schema = copy.deepcopy(self.cryptography_pqc_schema)
		del schema["kernels"][0]["parameter_set"]
		with self.assertRaises(ValueError):
			GENERATOR.validate_cryptography_pqc_schema(schema)
		schema = copy.deepcopy(self.cryptography_pqc_schema)
		schema["kernels"][1]["parameter_set"] = 65
		with self.assertRaises(ValueError):
			GENERATOR.validate_cryptography_pqc_schema(schema)

	def test_prehashed_verification_abi_and_framing_are_distinct(self):
		for index, kernel in enumerate(self.cryptography_pqc_schema["kernels"]):
			if not kernel["name"].endswith("_verify_prehashed"):
				continue
			shader = GENERATOR.generate_mldsa_verify_shader(kernel, "0" * 64)
			self.assertIn("mldsa_verify_prehashed(", shader)
			self.assertIn("push.context_length, push.algorithm)", shader)
			self.assertIn("storage_buffers[push.results_index].Store(output_base, 0u)", shader)
			for field, value in (("push_fields", kernel["push_fields"][:-1]),
				("semantic_operation", f"ml_dsa_{kernel['parameter_set']}_verify_batch"),
				("name", f"ml_dsa_{kernel['parameter_set']}_verify")):
				with self.subTest(parameter=kernel["parameter_set"], field=field):
					schema = copy.deepcopy(self.cryptography_pqc_schema)
					schema["kernels"][index][field] = value
					with self.assertRaises(ValueError):
						GENERATOR.validate_cryptography_pqc_schema(schema)

	def test_hash_message_verification_abi_and_framing_are_distinct(self):
		seen = set()
		for index, kernel in enumerate(self.cryptography_pqc_schema["kernels"]):
			if not kernel["name"].endswith("_verify_hash_message"):
				continue
			seen.add(kernel["parameter_set"])
			shader = GENERATOR.generate_mldsa_verify_shader(kernel, "0" * 64)
			self.assertIn("mldsa_verify_hash_message(", shader)
			self.assertNotIn("mldsa_verify_prehashed(", shader)
			self.assertIn("push.context_length, push.algorithm)", shader)
			self.assertIn("length > push.message_bytes - offset", shader)
			self.assertIn("storage_buffers[push.results_index].Store(output_base, 0u)", shader)
			for field, value in (("push_fields", kernel["push_fields"][:-1]),
				("semantic_operation", f"ml_dsa_{kernel['parameter_set']}_verify_prehashed_batch"),
				("name", f"ml_dsa_{kernel['parameter_set']}_verify_prehashed"),
				("source", "wrong.slang"), ("kernel_id", "WrongKernel")):
				with self.subTest(parameter=kernel["parameter_set"], field=field):
					schema = copy.deepcopy(self.cryptography_pqc_schema)
					schema["kernels"][index][field] = value
					with self.assertRaises(ValueError):
						GENERATOR.validate_cryptography_pqc_schema(schema)
		self.assertEqual(seen, {44, 65, 87})

	def test_private_mlkem_abi_rejects_drift(self):
		for field, value in (("workgroup_size", [2, 1, 1]), ("stable_id", 205),
			("push_fields", [["seed_index", "uint32"]]), ("kernel_id", "WrongKernel"),
			("source", "secret.slang"), ("secret_bindings", [])):
			with self.subTest(field=field):
				schema = copy.deepcopy(self.cryptography_pqc_schema)
				schema["private_kernels"][0][field] = value
				with self.assertRaises(ValueError):
					GENERATOR.validate_cryptography_pqc_schema(schema)

	def test_private_mlkem_kem_classification_and_abi(self):
		GENERATOR.validate_cryptography_pqc_schema(self.cryptography_pqc_schema)
		for index in (1, 2):
			for field, value in (("secret_bindings", [0]), ("push_fields", [["k", "uint32"]]),
				("workgroup_size", [32, 1, 1]), ("name", "unknown"),
				("physical_write", {"writes": [], "workspace": "none"})):
				with self.subTest(index=index, field=field):
					schema = copy.deepcopy(self.cryptography_pqc_schema)
					schema["private_kernels"][index][field] = value
					with self.assertRaises(ValueError):
						GENERATOR.validate_cryptography_pqc_schema(schema)
		schema = copy.deepcopy(self.cryptography_pqc_schema)
		schema["private_kernels"].append(copy.deepcopy(schema["private_kernels"][1]))
		with self.assertRaises(ValueError):
			GENERATOR.validate_cryptography_pqc_schema(schema)


	def test_private_mldsa_keygen_layout_and_classification(self):
		index = next(i for i, row in enumerate(self.cryptography_pqc_schema["private_kernels"])
			if row["name"] == "ml_dsa_keygen")
		for field, value in (("secret_bindings", [0]), ("parameter_layouts", []),
			("push_fields", [["k", "uint32"]]), ("workgroup_size", [2, 1, 1]),
			("source", "mldsa_keygen.slang")):
			with self.subTest(field=field):
				schema = copy.deepcopy(self.cryptography_pqc_schema)
				schema["private_kernels"][index][field] = value
				with self.assertRaises(ValueError):
					GENERATOR.validate_cryptography_pqc_schema(schema)

	def test_private_mldsa_sign_layout_and_secret_outputs(self):
		index = next(i for i, row in enumerate(self.cryptography_pqc_schema["private_kernels"])
			if row["name"] == "ml_dsa_sign")
		for field, value in (("secret_bindings", [0]), ("parameter_layouts", []),
			("push_fields", [["parameter_set", "uint32"]]), ("workgroup_size", [2, 1, 1]),
			("physical_write", self.cryptography_pqc_schema["private_kernels"][0]["physical_write"]),
			("source", "mldsa_sign.slang")):
			with self.subTest(field=field):
				schema = copy.deepcopy(self.cryptography_pqc_schema)
				schema["private_kernels"][index][field] = value
				with self.assertRaises(ValueError):
					GENERATOR.validate_cryptography_pqc_schema(schema)

	def test_private_mldsa_sign_workspace_contract(self):
		index = next(i for i, row in enumerate(self.cryptography_pqc_schema["private_kernels"])
			if row["name"] == "ml_dsa_sign")
		kernel = self.cryptography_pqc_schema["private_kernels"][index]
		for field, value in (("workspace_bytes", 23548), ("workspace_bytes", 23552.0),
			("secret_bindings", [0, 1]), ("physical_write", {
				"writes": kernel["physical_write"]["writes"], "workspace": "none"})):
			with self.subTest(field=field):
				schema = copy.deepcopy(self.cryptography_pqc_schema)
				schema["private_kernels"][index][field] = value
				with self.assertRaises(ValueError):
					GENERATOR.validate_cryptography_pqc_schema(schema)
		shader = GENERATOR.generate_mldsa_sign_shader(kernel, "0" * 64)
		self.assertIn("workspace_size != MLDSA_SIGN_CACHE_BYTES", shader)
		self.assertIn("mldsa_sign_mu_buffer(parameter", shader)
		self.assertNotIn("mldsa_sign_mu(parameter", shader)
		self.assertIn("storage_buffers[push.workspace_index]", shader)
		self.assertEqual(kernel["secret_bindings"], [0, 1, 5])
		self.assertEqual([row["binding"] for row in kernel["physical_write"]["writes"]], [3, 4, 5])

	def test_all_signing_workspaces_have_one_secret_extent(self):
		for index, kernel in enumerate(self.cryptography_pqc_schema["private_kernels"]):
			if not kernel["name"].startswith("ml_dsa_sign"):
				continue
			self.assertEqual(kernel["workspace_bytes"], 23552)
			for extent in (0, 23548, 23556, 23552.0, True):
				with self.subTest(kernel=kernel["name"], extent=extent):
					schema = copy.deepcopy(self.cryptography_pqc_schema)
					schema["private_kernels"][index]["workspace_bytes"] = extent
					with self.assertRaises(ValueError):
						GENERATOR.validate_cryptography_pqc_schema(schema)

	def test_private_mldsa_message_framing_abi(self):
		index = next(i for i, row in enumerate(self.cryptography_pqc_schema["private_kernels"])
			if row["name"] == "ml_dsa_sign_message")
		for field, value in (("secret_bindings", [0]), ("parameter_layouts", []),
			("push_fields", [["parameter_set", "uint32"]]), ("workgroup_size", [2, 1, 1]),
			("source", "mldsa_sign_dispatch.slang"), ("kernel_id", "CryptographyMlDsaSignU8")):
			with self.subTest(field=field):
				schema = copy.deepcopy(self.cryptography_pqc_schema)
				schema["private_kernels"][index][field] = value
				with self.assertRaises(ValueError):
					GENERATOR.validate_cryptography_pqc_schema(schema)
		shader = GENERATOR.generate_mldsa_sign_shader(self.cryptography_pqc_schema["private_kernels"][index], "0" * 64)
		self.assertIn("push.context_length > 255u", shader)
		self.assertIn("push.message_length > 0xfffffffcu", shader)
		self.assertIn("true, storage_buffers[push.workspace_index]", shader)
		self.assertNotIn("mldsa_sign_mu(parameter", shader)

	def test_private_prehashed_signing_abi_is_fixed(self):
		index = next(i for i, row in enumerate(self.cryptography_pqc_schema["private_kernels"])
			if row["name"] == "ml_dsa_sign_prehashed")
		for field, value in (("secret_bindings", [0]), ("parameter_layouts", []),
			("push_fields", [["algorithm", "uint32"]]), ("source", "mldsa_sign_dispatch.slang")):
			with self.subTest(field=field):
				schema = copy.deepcopy(self.cryptography_pqc_schema)
				schema["private_kernels"][index][field] = value
				with self.assertRaises(ValueError):
					GENERATOR.validate_cryptography_pqc_schema(schema)
		shader = GENERATOR.generate_mldsa_sign_shader(self.cryptography_pqc_schema["private_kernels"][index], "0" * 64)
		self.assertIn("mldsa_sign_prehashed_buffer(parameter", shader)
		self.assertIn("message_size != mldsa_prehash_bytes(push.algorithm)", shader)
		self.assertNotIn("mldsa_sign_message(parameter", shader)
		self.assertIn("import mldsa_prehash_layout;", shader)

	def test_private_hash_message_signing_abi_is_fixed(self):
		index = next(i for i, row in enumerate(self.cryptography_pqc_schema["private_kernels"])
			if row["name"] == "ml_dsa_sign_hash_message")
		for field, value in (("secret_bindings", [0, 1]), ("parameter_layouts", []),
			("push_fields", [["algorithm", "uint32"]]),
			("source", "src/slang/cryptography/pqc/mldsa/mldsa_sign_prehashed_dispatch.slang"),
			("kernel_id", "CryptographyMlDsaSignMessageU8"), ("workspace_bytes", 0)):
			with self.subTest(field=field):
				schema = copy.deepcopy(self.cryptography_pqc_schema)
				schema["private_kernels"][index][field] = value
				with self.assertRaises(ValueError):
					GENERATOR.validate_cryptography_pqc_schema(schema)
		shader = GENERATOR.generate_mldsa_sign_shader(self.cryptography_pqc_schema["private_kernels"][index], "0" * 64)
		self.assertIn("mldsa_sign_hash_message_buffer(parameter", shader)
		self.assertIn("push.message_length > 0xfffffffcu", shader)
		self.assertIn("mldsa_prehash_bytes(push.algorithm) == 0u", shader)
		self.assertIn("workspace_size != MLDSA_SIGN_CACHE_BYTES", shader)
		self.assertNotIn("push.digest_index", shader)
		self.assertIn("import mldsa_prehash;", shader)

	def test_prehash_schema_rejects_identifier_and_extent_drift(self):
		for field, value in (("oid_suffix", 17), ("oid_suffix", True), ("oid_suffix", 1.0),
			("digest_bytes", 63), ("digest_bytes", 32.0), ("name", "SHA-1")):
			with self.subTest(field=field):
				schema = copy.deepcopy(self.cryptography_pqc_schema)
				schema["prehashes"][0][field] = value
				with self.assertRaises(ValueError):
					GENERATOR.validate_cryptography_pqc_schema(schema)

	def test_schema_generates_every_operation_surface(self):
		GENERATOR.validate_schema(self.schema)
		GENERATOR.validate_blas_schema(self.blas_schema)
		GENERATOR.validate_reduce_schema(self.reduce_schema)
		GENERATOR.validate_rng_schema(self.rng_schema)
		GENERATOR.validate_index_schema(self.index_schema)
		GENERATOR.validate_ml_schema(self.ml_schema)
		GENERATOR.validate_activation_schema(self.activation_schema)
		GENERATOR.validate_audio_schema(self.audio_schema)
		GENERATOR.validate_cryptography_hash_schema(self.cryptography_hash_schema)
		GENERATOR.validate_cryptography_pqc_schema(self.cryptography_pqc_schema)
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
				self.index_schema,
				"9" * 64,
				self.ml_schema,
				"3" * 64,
				self.activation_schema,
				"b" * 64,
				self.audio_schema,
				"4" * 64,
				self.cryptography_hash_schema,
				"5" * 64,
				self.cryptography_pqc_schema,
				"a" * 64,
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
			body_family_count = len({op["rust_family"] for op in self.ml_schema["operations"] if "rust_family" in op})
			activation_family_count = len({op["rust_family"] for op in self.activation_schema["operations"]})
			matrix_autograd_family_count = len({
				op["autograd"]["family"]
				for schema in (self.schema, self.blas_schema, self.reduce_schema, self.rng_schema, self.index_schema, self.view_schema)
				for op in schema["operations"] if op.get("autograd", {}).get("mode") == "generated"
			})
			self.assertEqual(len(outputs), variant_count + 24 + 2 * matrix_autograd_family_count + 3 + 2 * autograd_family_count + 1 + body_family_count + activation_family_count - 1 + len(self.cryptography_pqc_schema["kernels"]) + len(self.cryptography_pqc_schema["private_kernels"]))
			self.assertIn(root / "src/slang/cryptography/pqc/mldsa/mldsa_prehash_digest.slang", outputs)
			api = outputs[root / "src/rs/matrix/elemwise.gen.rs"]
			blas_api = outputs[root / "src/rs/matrix/blas.gen.rs"]
			reduce_api = outputs[root / "src/rs/matrix/reduce.gen.rs"]
			# Generated APIs own recording/allocation, not forwarding-only wrappers.
			for family in (api, blas_api, reduce_api):
				self.assertIn("Matrix::allocate(", family)
				self.assertIn("record_semantic(", family)
				self.assertNotIn("_impl(", family)
			self.assertNotIn("binary(left", api)
			self.assertNotIn("unary(input", api)
			self.assertIn("autograd::record_mat_mul_nt", blas_api)
			self.assertIn("autograd::record_softmax", reduce_api)

			registry = outputs[root / "src/rs/runtime/shader/registry.gen.rs"]
			self.assertIn("Self::CryptographyMlKemKeygenU8 => None", registry)
			self.assertIn("mlkem_keygen_internal", outputs[root / "src/slang/cryptography/pqc/mlkem/mlkem_keygen.slang"])
			for suffix in ("encaps", "decaps"):
				self.assertIn(f"Self::CryptographyMlKem{suffix.title()}U8 => None", registry)
				shader = outputs[root / f"src/slang/cryptography/pqc/mlkem/mlkem_{suffix}.slang"]
				self.assertIn(f"mlkem_{suffix}_internal", shader)
			for parameter in (44, 65, 87):
				self.assertEqual(outputs[root / "src/rs/core/operation/operation.gen.rs"].count(f"const ML_DSA_{parameter}_PUBLIC_KEY_SIZE:"), 1)
			self.assertIn("Self::CryptographyMlDsaKeygenU8 => None", registry)
			self.assertIn("44 => Some((32, 1312, 2560))", registry)
			self.assertIn("mldsa_keygen_internal", outputs[root / "src/slang/cryptography/pqc/mldsa/mldsa_keygen_dispatch.slang"])
			operations = outputs[root / "src/rs/core/operation/operation.gen.rs"]
			dnn_roles = outputs[root / "src/rs/runtime/dnn/generated.rs"]
			activation_autograd = outputs[
				root / "src/rs/ml/autograd/matrix/activation.gen.rs"
			]
			loss_autograd = outputs[root / "src/rs/ml/autograd/loss.gen.rs"]
			self.assertIn(root / "test/rs/matrix/test_elemwise.gen.rs", outputs)
			self.assertIn(root / "test/rs/matrix/test_blas.gen.rs", outputs)
			self.assertIn(root / "test/rs/matrix/test_reduce.gen.rs", outputs)
			for operation in self.schema["operations"]:
				self.assertIn(f"pub fn {operation['name']}", api)
				body = api.split(f"pub fn {operation['name']}(", 1)[1].split("\npub fn ", 1)[0]
				self.assertIn("record_semantic(", body)
				self.assertIn("Matrix::allocate(", body)
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
			self.assertIn("GradNodeOperation::Gelu", activation_autograd)
			self.assertIn("pub(in crate::ml) fn record_relu", activation_autograd)
			self.assertIn("input: &Matrix, output: &Matrix, result: &Matrix", activation_autograd)
			self.assertIn("GradNodeOperation::Relu", activation_autograd)
			self.assertIn("pub(in crate::ml) fn record_leaky_relu", activation_autograd)
			self.assertIn("alpha: f32, result: &Matrix", activation_autograd)
			self.assertIn("GradNodeOperation::LeakyRelu", activation_autograd)
			swiglu_autograd = outputs[root / "src/rs/ml/autograd/matrix/swiglu.gen.rs"]
			self.assertIn("pub(in crate::ml) fn record_swiglu", swiglu_autograd)
			self.assertIn("GradNodeOperation::Swiglu", swiglu_autograd)
			self.assertNotIn("record_swiglu", activation_autograd)
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
			self.assertIn("GradNodeOperation::CrossEntropy", loss_autograd)

	def test_every_slang_entry_point_has_one_schema_owner(self):
		schema_sources = set()
		for schema in (
			self.reduce_schema,
			self.rng_schema,
			self.index_schema,
			self.ml_schema,
			self.audio_schema,
			self.cryptography_hash_schema,
			self.cryptography_pqc_schema,
			self.image_schema,
			self.vision_schema,
			self.ui_schema,
		):
			for row in schema.get("operations", []) + schema.get("kernels", []) + schema.get("private_kernels", []):
				if source := row.get("source"):
					schema_sources.add(source)

		generated_sources = {
			f"src/slang/matrix/elemwise/{variant['source_stem']}.gen.slang"
			for operation in self.schema["operations"]
			for variant in GENERATOR.operation_variants(self.schema, operation)
		}
		generated_sources.update(
			lowering["source"]
			for operation in self.schema["operations"]
			for lowering in GENERATOR.operation_lowering_variants(operation)
		)
		generated_sources.update(
			f"src/slang/matrix/blas/{operation['source_stem']}.gen.slang"
			for operation in self.blas_schema["operations"]
		)
		expected = schema_sources | generated_sources
		actual = {
			path.relative_to(REPOSITORY_ROOT).as_posix()
			for root in (SLANG_ROOT, SDK_SLANG_ROOT)
			for path in root.rglob("*.slang")
			if "void main(" in path.read_text(encoding="utf-8")
		}

		self.assertEqual(actual, expected)
		self.assertTrue(all((REPOSITORY_ROOT / source).is_file() for source in expected))
		for root in (SLANG_ROOT, SDK_SLANG_ROOT):
			for path in root.rglob("*.slang"):
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

	def test_elemwise_abs_preserves_reverse_differentiation(self):
		operation = next(
			operation
			for operation in self.schema["operations"]
			if operation["name"] == "abs"
		)
		self.assertEqual(operation["differentiation"], "reverse")

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

	def test_categorical_matrix_dependencies_have_stable_schema_ownership(self):
		gather = next(
			operation
			for operation in self.index_schema["operations"]
			if operation["name"] == "gather_last_dim"
		)
		gather_backward = next(
			operation
			for operation in self.index_schema["operations"]
			if operation["name"] == "gather_last_dim_backward"
		)
		self.assertEqual(gather["stable_id"], 388)
		self.assertEqual(gather_backward["stable_id"], 389)
		self.assertEqual(gather["differentiation"], "reverse")
		self.assertEqual(
			gather["contract"]["dtype_rule"], "f32_values_i32_indices"
		)

		sampling = {
			operation["name"]: operation
			for operation in self.rng_schema["operations"]
			if operation["name"].startswith("sample_logits")
		}
		self.assertEqual(
			list(sampling),
			["sample_logits", "sample_logits_dense", "sample_logits_sorted"],
		)
		self.assertEqual(
			[operation["stable_id"] for operation in sampling.values()],
			[390, 391, 392],
		)
		self.assertEqual(
			[operation["variant"] for operation in sampling.values()],
			["greedy", "dense", "top_k_top_p"],
		)
		self.assertNotIn("semantic_operation", sampling["sample_logits"])
		for operation in (
			sampling["sample_logits_dense"],
			sampling["sample_logits_sorted"],
		):
			self.assertEqual(operation["semantic_operation"], "sample_logits")

	def test_ml_composites_preserve_contracts_and_read_only_aliases(self):
		operations = {
			operation["name"]: operation
			for operation in self.ml_schema["composite_contracts"]
		}
		self.assertEqual(list(operations), [
			"normalize_observation",
			"scale_action",
			"clip_reward",
			"normalize",
			"ppo",
			"dqn",
			"sac_critic",
			"sac_actor",
			"sample_categorical",
			"evaluate_categorical",
			"sample_tanh_normal",
			"evaluate_tanh_normal",
		])
		for name in ("normalize_observation", "scale_action", "clip_reward"):
			self.assertEqual(operations[name]["semantic_domain"], "ml::environment")
			self.assertEqual(
				operations[name]["contract"]["output_alias_inputs"], [-1]
			)
		self.assertEqual(
			operations["normalize"]["semantic_domain"], "ml::advantage"
		)
		self.assertEqual(
			operations["normalize"]["contract"]["output_alias_inputs"], [-1]
		)
		self.assertEqual(operations["ppo"]["semantic_domain"], "ml::loss")
		self.assertEqual(
			operations["ppo"]["contract"]["output_alias_inputs"], [-1, -1, -1, -1]
		)
		self.assertEqual(operations["dqn"]["semantic_domain"], "ml::loss")
		self.assertEqual(
			operations["dqn"]["contract"]["output_alias_inputs"], [-1, -1, -1]
		)
		self.assertEqual(
			operations["sac_critic"]["contract"]["output_alias_inputs"],
			[-1, -1, -1, -1],
		)
		self.assertEqual(
			operations["sac_actor"]["contract"]["output_alias_inputs"], [-1]
		)
		self.assertEqual(
			operations["sample_categorical"]["contract"]["output_alias_inputs"],
			[-1, -1, -1, 1],
		)
		self.assertEqual(
			operations["evaluate_categorical"]["contract"]["output_alias_inputs"],
			[1, -1, -1, 2],
		)
		self.assertEqual(
			operations["sample_tanh_normal"]["contract"]["output_alias_inputs"],
			[-1, -1, -1, -1, 2],
		)
		self.assertEqual(
			operations["evaluate_tanh_normal"]["contract"]["output_alias_inputs"],
			[-1, 2, -1, -1, 3],
		)
		self.assertTrue(
			all(not operation["contract"]["mutated_inputs"] for operation in operations.values())
		)

	def test_stable_ids_must_be_unique_across_schema_families(self):
		invalid = copy.deepcopy(self.blas_schema)
		invalid["operations"][0]["stable_id"] = self.schema["operations"][0]["stable_id"]
		with self.assertRaisesRegex(GENERATOR.SchemaError, "across schemas"):
			GENERATOR.registry_entries(
				self.schema,
				invalid,
				self.reduce_schema,
				self.rng_schema,
				self.index_schema,
				self.ml_schema,
				self.audio_schema,
				self.cryptography_hash_schema,
				self.cryptography_pqc_schema,
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
