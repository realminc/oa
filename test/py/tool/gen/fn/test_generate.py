import copy
import importlib.util
import json
import re
import tempfile
import unittest
from pathlib import Path


GENERATOR_ROOT = Path(__file__).resolve().parents[5] / "tool/gen/fn"
SCHEMA_PATH = GENERATOR_ROOT / "schema/matrix/matrix_elemwise.json"
BLAS_SCHEMA_PATH = GENERATOR_ROOT / "schema/matrix/matrix_blas.json"
REDUCE_SCHEMA_PATH = GENERATOR_ROOT / "schema/matrix/matrix_reduce.json"
RNG_SCHEMA_PATH = GENERATOR_ROOT / "schema/matrix/matrix_rng.json"
VIEW_SCHEMA_PATH = GENERATOR_ROOT / "schema/matrix/matrix_view.json"
INDEX_SCHEMA_PATH = GENERATOR_ROOT / "schema/matrix/matrix_index.json"
ML_SCHEMA_PATH = GENERATOR_ROOT / "schema/ml/ml_training.json"
ACTIVATION_SCHEMA_PATH = GENERATOR_ROOT / "schema/ml/ml_activation.json"
AUDIO_SCHEMA_PATH = GENERATOR_ROOT / "schema/audio/audio.json"
CRYPTOGRAPHY_HASH_SCHEMA_PATH = GENERATOR_ROOT / "schema/crypto/crypto_hash.json"
CRYPTOGRAPHY_PQC_SCHEMA_PATH = GENERATOR_ROOT / "schema/crypto/crypto_pqc.json"
IMAGE_SCHEMA_PATH = GENERATOR_ROOT / "schema/image/image.json"
VISION_SCHEMA_PATH = GENERATOR_ROOT / "schema/vision/vision_detection.json"
UI_SCHEMA_PATH = GENERATOR_ROOT / "schema/ui/ui.json"
REPOSITORY_ROOT = GENERATOR_ROOT.parents[2]
SLANG_ROOT = REPOSITORY_ROOT / "src/slang"
SDK_SLANG_ROOT = REPOSITORY_ROOT / "sdk/rs/slang"

import sys
sys.path.insert(0, str(GENERATOR_ROOT))
import oafn.audio as oa_audio
import oafn.bigint as oa_bigint
import oafn.hash as oa_hash
import oafn.image as oa_image
import oafn.matrix as oa_matrix
import oafn.ml as oa_ml
import oafn.model as oa_model
import oafn.pipeline as oa_pipeline
import oafn.pqc as oa_pqc
import oafn.publish as oa_publish
import oafn.registry as oa_registry
import oafn.schema as oa_schema
import oafn.signature as oa_signature
import oafn.uistyle as oa_uistyle
import oafn.util as oa_util
import oafn.video as oa_video
import oafn.vision as oa_vision



class GeneratorTests(unittest.TestCase):
	def test_registry_embeds_one_artifact_per_candidate(self):
		schemas = oa_schema.load_schemas(REPOSITORY_ROOT)
		entries = oa_registry.registry_entries(schemas)
		manifest = oa_registry.generate_kernel_manifest(schemas)
		self.assertEqual(manifest.count("include_bytes!("), len(entries))
		self.assertNotIn("_bounded.spv", manifest)
		self.assertNotIn("bounded_artifact", manifest)
		self.assertIn('env!("OA_SHADER_DIR")', manifest)

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
		self.bigint_schema = json.loads((GENERATOR_ROOT / "schema/crypto/crypto_bigint.json").read_text())
		self.image_schema = json.loads(IMAGE_SCHEMA_PATH.read_text(encoding="utf-8"))
		self.vision_schema = json.loads(VISION_SCHEMA_PATH.read_text(encoding="utf-8"))
		self.ui_schema = json.loads(UI_SCHEMA_PATH.read_text(encoding="utf-8"))
		self.render_schema = json.loads((GENERATOR_ROOT / "schema/render/render.json").read_text(encoding="utf-8"))
		self.video_schema = json.loads((GENERATOR_ROOT / "schema/video/video.json").read_text(encoding="utf-8"))

	def test_ml_activation_templates_emit_complete_operations(self):
		api = oa_ml.generate_ml_activation_api(self.activation_schema, "0" * 64)
		for helper in ("fn record(", "fn validate_f32(", "fn as_u32(", "\n\t\trecord("):
			self.assertNotIn(helper, api)
		self.assertNotIn("{{", api)
		for operation in self.activation_schema["operations"]:
			body = oa_ml._ml_activation_fn(operation)
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
			oa_ml._render_ml_activation_template("unary", {"name": "gelu"})

	def test_elementwise_module_is_complete_and_routes_are_schema_owned(self):
		api = oa_matrix.generate_api(self.schema, "0" * 64)
		self.assertIn("struct BroadcastLayout", api)
		self.assertIn("fn select_kernel(", api)
		self.assertNotIn("include!", api)
		self.assertNotIn("contract.hash()", api)
		self.assertNotIn("{{items}}", api)
		for operation in self.schema["operations"]:
			if operation.get("lowering_only", False):
				continue
			body = api.split(f"pub fn {operation['name']}(", 1)[1].split("\n///", 1)[0]
			for candidate in operation.get("additional_lowering_variants", []):
				self.assertIn(f"KernelId::{candidate['kernel_id']}", body)
		operation = self.schema["operations"][0]
		operation["additional_lowering_variants"][0]["kernel_id"] = "SchemaSelectedBroadcast"
		api = oa_matrix.generate_api(self.schema, "0" * 64)
		self.assertIn("KernelId::SchemaSelectedBroadcast", api)
		self.assertNotIn("KernelId::MatrixAddBroadcastF32", api)

	def test_elementwise_retains_exact_shape_only_schema_support(self):
		operation = self.schema["operations"][0]
		operation["shape_rule"] = "equal_no_broadcast"
		del operation["additional_lowering_variants"]
		oa_matrix.validate_schema(self.schema)
		api = oa_matrix.generate_api(self.schema, "0" * 64)
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
				with self.assertRaises(oa_util.SchemaError):
					oa_matrix.validate_schema(schema)

	def test_pqc_parameter_identity_rejects_mismatched_or_duplicate_sets(self):
		index = next(i for i, row in enumerate(self.cryptography_pqc_schema["private_kernels"])
			if row["name"] == "ml_dsa_mu_batch")
		for field, value in (("parameter_layouts", []), ("parameter_layouts", "x"),
				("name", "ml_dsa_44_verify"), ("kernel_id", "WrongKernel"), ("stable_id", 70000),
				("semantic_operations", ["ml_dsa_44_verify_batch"])):
			with self.subTest(field=field, value=value):
				schema = copy.deepcopy(self.cryptography_pqc_schema)
				schema["private_kernels"][index][field] = value
				with self.assertRaises(ValueError):
					oa_pqc.validate_cryptography_pqc_schema(schema)
		for row, value in ((0, 66), (0, True), (1, 44), (2, 65)):
			with self.subTest(row=row, value=value):
				schema = copy.deepcopy(self.cryptography_pqc_schema)
				schema["private_kernels"][index]["parameter_layouts"][row]["parameter_set"] = value
				with self.assertRaises(ValueError):
					oa_pqc.validate_cryptography_pqc_schema(schema)
		schema = copy.deepcopy(self.cryptography_pqc_schema)
		del schema["private_kernels"][index]["parameter_layouts"][1]["public_bytes"]
		with self.assertRaises(ValueError):
			oa_pqc.validate_cryptography_pqc_schema(schema)

	def test_prehashed_verification_abi_and_framing_are_distinct(self):
		kernels = self.cryptography_pqc_schema["private_kernels"]
		index = next(i for i, row in enumerate(kernels) if row["name"] == "ml_dsa_verify_mu")
		kernel = kernels[index]
		self.assertEqual(len(kernel["semantic_operations"]), 9)
		shader = oa_pqc.generate_mldsa_verify_mu_shader(kernel, "0" * 64)
		self.assertIn("mldsa_verify_mu_par(parameter", shader)
		self.assertIn("storage_buffers[push.results_index].Store(idx * 4u, valid ? 1u : 0u)", shader)
		for field, value in (("push_fields", kernel["push_fields"][:-1]),
			("semantic_operations", ["ml_dsa_44_verify_batch"]),
			("semantic_operations", []), ("name", "ml_dsa_44_verify"),
			("source", "wrong.slang"), ("kernel_id", "WrongKernel")):
			with self.subTest(field=field):
				schema = copy.deepcopy(self.cryptography_pqc_schema)
				schema["private_kernels"][index][field] = value
				with self.assertRaises(ValueError):
					oa_pqc.validate_cryptography_pqc_schema(schema)

	def test_hash_message_verification_abi_and_framing_are_distinct(self):
		kernels = self.cryptography_pqc_schema["private_kernels"]
		index = next(i for i, row in enumerate(kernels) if row["name"] == "ml_dsa_ph_batch")
		kernel = kernels[index]
		shader = oa_pqc.generate_mldsa_ph_batch_shader(kernel, "0" * 64)
		self.assertIn("mldsa_ph_batch_par(push.algorithm", shader)
		self.assertIn("length > push.input_bytes - offset", shader)
		self.assertNotIn("mldsa_verify_prehashed(", shader)
		for field, value in (("push_fields", kernel["push_fields"][:-1]),
			("parameter_layouts", []), ("name", "ml_dsa_44_verify"),
			("source", "wrong.slang"), ("kernel_id", "WrongKernel")):
			with self.subTest(field=field):
				schema = copy.deepcopy(self.cryptography_pqc_schema)
				schema["private_kernels"][index][field] = value
				with self.assertRaises(ValueError):
					oa_pqc.validate_cryptography_pqc_schema(schema)
		verify = next(row for row in kernels if row["name"] == "ml_dsa_verify_mu")
		for parameter in (44, 65, 87):
			self.assertIn(f"ml_dsa_{parameter}_verify_hash_message_batch", verify["semantic_operations"])

	def test_private_mlkem_abi_rejects_drift(self):
		for field, value in (("workgroup_size", [2, 1, 1]), ("stable_id", 70000),
			("push_fields", [["seed_index", "uint32"]]), ("kernel_id", "WrongKernel"),
			("source", "secret.slang"), ("secret_bindings", [])):
			with self.subTest(field=field):
				schema = copy.deepcopy(self.cryptography_pqc_schema)
				schema["private_kernels"][0][field] = value
				with self.assertRaises(ValueError):
					oa_pqc.validate_cryptography_pqc_schema(schema)

	def test_private_mlkem_parallel_contract(self):
		kernel = self.cryptography_pqc_schema["private_kernels"][0]
		for field, value in (("workspace_bytes", 4096), ("secret_bindings", [0, 1, 3]),
			("workgroup_size", [1, 1, 1]), ("push_fields", kernel["push_fields"] + [["workspace_index", "uint32"]]),
			("physical_write", {
				"writes": kernel["physical_write"]["writes"], "workspace": "exclusive_per_workgroup"})):
			with self.subTest(field=field, value=value):
				schema = copy.deepcopy(self.cryptography_pqc_schema)
				schema["private_kernels"][0][field] = value
				with self.assertRaises(ValueError):
					oa_pqc.validate_cryptography_pqc_schema(schema)
		shader = oa_pqc.generate_mlkem_keygen_shader(kernel, "0" * 64)
		self.assertIn("[numthreads(128, 1, 1)]", shader)
		self.assertIn("mlkem_keygen_par(push.k, lane", shader)
		self.assertNotIn("workspace_index", shader)

	def test_private_mlkem_consumer_parallel_contract(self):
		for index, suffix in ((1, "encaps"), (2, "decaps")):
			kernel = self.cryptography_pqc_schema["private_kernels"][index]
			for field, value in (("workspace_bytes", 4096), ("workgroup_size", [1, 1, 1])):
				with self.subTest(suffix=suffix, field=field):
					schema = copy.deepcopy(self.cryptography_pqc_schema)
					schema["private_kernels"][index][field] = value
					with self.assertRaises(ValueError):
						oa_pqc.validate_cryptography_pqc_schema(schema)
			shader = oa_pqc.generate_mlkem_kem_shader(kernel, "0" * 64)
			self.assertIn("[numthreads(128, 1, 1)]", shader)
			self.assertIn(f"mlkem_{suffix}_par(push.k, lane", shader)
			self.assertNotIn("workspace_index", shader)

	def test_private_mlkem_kem_classification_and_abi(self):
		oa_pqc.validate_cryptography_pqc_schema(self.cryptography_pqc_schema)
		for index in (1, 2):
			for field, value in (("secret_bindings", [0]), ("push_fields", [["k", "uint32"]]),
				("workgroup_size", [32, 1, 1]), ("name", "unknown"),
				("physical_write", {"writes": [], "workspace": "none"})):
				with self.subTest(index=index, field=field):
					schema = copy.deepcopy(self.cryptography_pqc_schema)
					schema["private_kernels"][index][field] = value
					with self.assertRaises(ValueError):
						oa_pqc.validate_cryptography_pqc_schema(schema)
		schema = copy.deepcopy(self.cryptography_pqc_schema)
		schema["private_kernels"].append(copy.deepcopy(schema["private_kernels"][1]))
		with self.assertRaises(ValueError):
			oa_pqc.validate_cryptography_pqc_schema(schema)


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
					oa_pqc.validate_cryptography_pqc_schema(schema)

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
					oa_pqc.validate_cryptography_pqc_schema(schema)

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
					oa_pqc.validate_cryptography_pqc_schema(schema)
		shader = oa_pqc.generate_mldsa_sign_shader(kernel, "0" * 64)
		self.assertIn("workspace_size != MLDSA_SIGN_CACHE_BYTES", shader)
		self.assertIn("mldsa_sign_mu_par(parameter", shader)
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
						oa_pqc.validate_cryptography_pqc_schema(schema)

	def test_private_mldsa_message_framing_abi(self):
		# Every signing route derives the public FIPS 204 mu representative
		# through the shared ml_dsa_mu kernel; the M' head rides push words.
		index = next(i for i, row in enumerate(self.cryptography_pqc_schema["private_kernels"])
			if row["name"] == "ml_dsa_mu")
		kernel = self.cryptography_pqc_schema["private_kernels"][index]
		for field, value in (("secret_bindings", [0]), ("parameter_layouts", []),
			("push_fields", [["parameter_set", "uint32"]]), ("workgroup_size", [2, 1, 1]),
			("source", "mldsa_mu.slang"), ("kernel_id", "CryptographyMlDsaSignU8"),
			("semantic_operations", [])):
			with self.subTest(field=field):
				schema = copy.deepcopy(self.cryptography_pqc_schema)
				schema["private_kernels"][index][field] = value
				with self.assertRaises(ValueError):
					oa_pqc.validate_cryptography_pqc_schema(schema)
		shader = oa_pqc.generate_mldsa_mu_shader(kernel, "0" * 64)
		self.assertIn("push.context_length > 255u", shader)
		self.assertIn("mldsa_mu_par(parameter", shader)
		self.assertIn("push.head_length != 2u && push.head_length != 13u", shader)
		self.assertNotIn("mldsa_sign_mu(parameter", shader)

	def test_private_prehashed_signing_abi_is_fixed(self):
		# Prehashed signing carries the OID trailer in the mu head push words;
		# the kernel ABI is fixed so the engine can pack it.
		index = next(i for i, row in enumerate(self.cryptography_pqc_schema["private_kernels"])
			if row["name"] == "ml_dsa_mu")
		kernel = self.cryptography_pqc_schema["private_kernels"][index]
		names = [name for name, _ in kernel["push_fields"]]
		self.assertEqual(names, ["public_index", "payload_index", "context_index", "mu_index",
			"parameter_set", "payload_length", "context_length", "head_length",
			"head_w0", "head_w1", "head_w2", "head_w3"])
		for field, value in (("secret_bindings", [0]), ("parameter_layouts", []),
			("push_fields", names[:-1] and [[name, "uint32"] for name in names[:-1]]),
			("source", "mldsa_sign_dispatch.slang")):
			with self.subTest(field=field):
				schema = copy.deepcopy(self.cryptography_pqc_schema)
				schema["private_kernels"][index][field] = value
				with self.assertRaises(ValueError):
					oa_pqc.validate_cryptography_pqc_schema(schema)
		shader = oa_pqc.generate_mldsa_mu_shader(kernel, "0" * 64)
		self.assertIn("mldsa_mu_par(parameter", shader)
		self.assertNotIn("mldsa_prehash_digest", shader)
		self.assertNotIn("push.algorithm", shader)

	def test_private_hash_message_signing_abi_is_fixed(self):
		# Hash-message signing prehashes on the engine hash path; the mu kernel
		# only absorbs the finished digest with the OID-framed head.
		index = next(i for i, row in enumerate(self.cryptography_pqc_schema["private_kernels"])
			if row["name"] == "ml_dsa_mu")
		kernel = self.cryptography_pqc_schema["private_kernels"][index]
		for field, value in (("secret_bindings", [0, 1]), ("parameter_layouts", []),
			("push_fields", [["algorithm", "uint32"]]),
			("source", "src/slang/crypto/pqc/mldsa/mldsa_sign_prehashed_dispatch.slang"),
			("kernel_id", "CryptographyMlDsaSignMessageU8"), ("workspace_bytes", 0)):
			with self.subTest(field=field):
				schema = copy.deepcopy(self.cryptography_pqc_schema)
				schema["private_kernels"][index][field] = value
				with self.assertRaises(ValueError):
					oa_pqc.validate_cryptography_pqc_schema(schema)
		shader = oa_pqc.generate_mldsa_mu_shader(kernel, "0" * 64)
		self.assertIn("mldsa_mu_par(parameter", shader)
		self.assertIn("push.head_length != 2u && push.head_length != 13u", shader)
		self.assertNotIn("push.digest_index", shader)
		self.assertNotIn("push.algorithm", shader)

	def test_prehash_schema_rejects_identifier_and_extent_drift(self):
		for field, value in (("oid_suffix", 17), ("oid_suffix", True), ("oid_suffix", 1.0),
			("digest_bytes", 63), ("digest_bytes", 32.0), ("name", "SHA-1")):
			with self.subTest(field=field):
				schema = copy.deepcopy(self.cryptography_pqc_schema)
				schema["prehashes"][0][field] = value
				with self.assertRaises(ValueError):
					oa_pqc.validate_cryptography_pqc_schema(schema)

	def test_schema_generates_every_operation_surface(self):
		oa_matrix.validate_schema(self.schema)
		oa_matrix.validate_blas_schema(self.blas_schema)
		oa_matrix.validate_reduce_schema(self.reduce_schema)
		oa_matrix.validate_rng_schema(self.rng_schema)
		oa_matrix.validate_index_schema(self.index_schema)
		oa_ml.validate_ml_schema(self.ml_schema)
		oa_ml.validate_activation_schema(self.activation_schema)
		oa_audio.validate_audio_schema(self.audio_schema)
		oa_hash.validate_cryptography_hash_schema(self.cryptography_hash_schema)
		oa_pqc.validate_cryptography_pqc_schema(self.cryptography_pqc_schema)
		oa_image.validate_image_schema(self.image_schema)
		oa_vision.validate_vision_schema(self.vision_schema)
		with tempfile.TemporaryDirectory() as directory:
			root = Path(directory)
			outputs = oa_pipeline.expected_outputs(
				root,
				{
					"elemwise": (self.schema, "0" * 64),
					"blas": (self.blas_schema, "1" * 64),
					"reduce": (self.reduce_schema, "6" * 64),
					"rng": (self.rng_schema, "2" * 64),
					"index": (self.index_schema, "9" * 64),
					"ml": (self.ml_schema, "3" * 64),
					"ml_activation": (self.activation_schema, "b" * 64),
					"audio": (self.audio_schema, "4" * 64),
					"hash": (self.cryptography_hash_schema, "5" * 64),
					"pqc": (self.cryptography_pqc_schema, "a" * 64),
					"image": (self.image_schema, "7" * 64),
					"vision": (self.vision_schema, "8" * 64),
				},
			)
			variant_count = sum(
				len(oa_model.operation_variants(self.schema, operation))
				for operation in self.schema["operations"]
				if not operation.get("lowering_only", False)
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
			annex_schemas = [
				oa_schema.load_schema(name, GENERATOR_ROOT / f"schema/{filename}")[0]
				for name, filename in (("sessions", "sessions.json"), ("codecs", "codecs.json"), ("ml_parity", "ml_parity.json"))
			]
			annex_signatures = oa_signature.domain_signatures("sessions", annex_schemas)
			parity_domain_count = len({
				domain for domain in ("video", "audio", "data", "ml", "image", "runtime", "net", "media")
				if any(signature.get("target", "").startswith(f"crate::{domain}::") for signature in annex_signatures)
			})
			# The final constant is the fixed remainder: operation/shader/DNN
			# registries, domain API, type and binding surfaces, crypto Rust/slang
			# inventories, Python fragments, generated tests and the ui_style
			# token preset module.
			self.assertEqual(
				len(outputs),
				variant_count
				+ len(self.blas_schema["operations"])
				+ sum("source" not in candidate for candidate in self.blas_schema.get("candidate_kernels", []))
				+ 2 * matrix_autograd_family_count + 3
				+ 2 * autograd_family_count + 1
				+ body_family_count + activation_family_count + 1
				+ len(oa_image.IMAGE_CATEGORIES) + 1
				+ parity_domain_count
				+ len(self.cryptography_pqc_schema["kernels"]) + len(self.cryptography_pqc_schema["private_kernels"]) + 4
				+ sum("digest" in k or "poseidon2" in k or k["name"].startswith("merkle_proof_") for k in self.cryptography_hash_schema["kernels"])
				+ len(self.bigint_schema["kernels"])
				+ 88,
			)
			self.assertIn(root / "src/rs/lib/ui/style.gen.rs", outputs)
			self.assertIn(root / "src/slang/crypto/pqc/mldsa/mldsa_prehash_digest.slang", outputs)
			api = outputs[root / "src/rs/lib/matrix/elemwise.gen.rs"]
			blas_api = outputs[root / "src/rs/lib/matrix/blas.gen.rs"]
			reduce_api = outputs[root / "src/rs/lib/matrix/reduce.gen.rs"]
			# Generated APIs own recording/allocation, not forwarding-only wrappers.
			for family in (api, blas_api, reduce_api):
				self.assertIn("Matrix::allocate(", family)
				self.assertIn("record_semantic(", family)
				self.assertNotIn("_impl(", family)
			self.assertNotIn("binary(left", api)
			self.assertNotIn("unary(input", api)
			self.assertIn("autograd::record_mat_mul_nt", blas_api)
			self.assertIn("autograd::record_softmax", reduce_api)

			registry = outputs[root / "src/rs/lib/runtime/dna/kernel/kernel.gen.rs"]
			self.assertIn("Self::CryptographyMlKemKeygenU8 => None", registry)
			self.assertIn("mlkem_keygen_par", outputs[root / "src/slang/crypto/pqc/mlkem/mlkem_keygen.slang"])
			for suffix in ("encaps", "decaps"):
				self.assertIn(f"Self::CryptographyMlKem{suffix.title()}U8 => None", registry)
				shader = outputs[root / f"src/slang/crypto/pqc/mlkem/mlkem_{suffix}.slang"]
				self.assertIn(f"mlkem_{suffix}_par", shader)
			for parameter in (44, 65, 87):
				self.assertEqual(outputs[root / "src/rs/lib/core/operation/operation.gen.rs"].count(f"const ML_DSA_{parameter}_PUBLIC_KEY_SIZE:"), 1)
			self.assertIn("Self::CryptographyMlDsaKeygenU8 => None", registry)
			self.assertIn("44 => Some((32, 1312, 2560))", registry)
			self.assertIn("mldsa_keygen_par", outputs[root / "src/slang/crypto/pqc/mldsa/mldsa_keygen_dispatch.slang"])
			operations = outputs[root / "src/rs/lib/core/operation/operation.gen.rs"]
			dnn_roles = outputs[root / "src/rs/lib/runtime/dna/provider/dnn/generated.rs"]
			activation_autograd = outputs[
				root / "src/rs/lib/ml/autograd/matrix/activation.gen.rs"
			]
			loss_autograd = outputs[root / "src/rs/lib/ml/autograd/loss.gen.rs"]
			self.assertIn(root / "test/rs/matrix/test_elemwise.gen.rs", outputs)
			self.assertIn(root / "test/rs/matrix/test_blas.gen.rs", outputs)
			self.assertIn(root / "test/rs/matrix/test_reduce.gen.rs", outputs)
			for operation in self.schema["operations"]:
				if operation.get("lowering_only", False):
					self.assertIn(operation["kernel_id"], registry)
					continue
				self.assertIn(f"pub fn {operation['name']}", api)
				body = api.split(f"pub fn {operation['name']}(", 1)[1].split("\npub fn ", 1)[0]
				self.assertIn("record_semantic(", body)
				self.assertIn("Matrix::allocate(", body)
				self.assertIn(
					f"pub const {oa_util.rust_const_name(operation['name'])}", operations
				)
				self.assertIn(
					f"crate::core::operation::matrix::{oa_util.rust_const_name(operation['name'])}",
					api,
				)
				for variant in oa_model.operation_variants(self.schema, operation):
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
					f"pub const {oa_util.rust_const_name(operation['name'])}", operations
				)
				self.assertIn(operation["kernel_id"], registry)
				shader_path = root / f"src/slang/matrix/blas/{operation['source_stem']}.gen.slang"
				self.assertIn(shader_path, outputs)
				self.assertIn("void main(", outputs[shader_path])
				self.assertIn('[variant("tiled")]', outputs[shader_path])
			self.assertIn("dispatch_tile_size: [64, 64, 1]", registry)
			self.assertIn("physical_write: None", registry)
			for operation in self.reduce_schema["operations"]:
				if "kernel_id" in operation:
					self.assertIn(operation["kernel_id"], registry)
				if not operation.get("lowering_only", False):
					self.assertIn(
						f"pub const {oa_util.rust_const_name(operation['name'])}",
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
						f"pub const {oa_util.rust_const_name(operation['name'])}", operations
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
					f"pub const {oa_util.rust_const_name(contract['name'])}",
					operations,
				)
			for kernel in self.cryptography_hash_schema["kernels"]:
				self.assertIn(kernel["kernel_id"], registry)
			for contract in self.image_schema["contracts"]:
				self.assertIn(
					f"pub const {oa_util.rust_const_name(contract['name'])}",
					operations,
				)
			for kernel in self.image_schema["kernels"]:
				self.assertIn(kernel["kernel_id"], registry)
			self.assertIn('"oa::image::resize"', operations)
			self.assertIn("OpValueKind::Image", operations)
			self.assertIn('Self::ImageResizeBilinearF32 => "image.resize_bilinear.f32"', registry)
			for contract in self.vision_schema["contracts"]:
				self.assertIn(
					f"pub const {oa_util.rust_const_name(contract['name'])}",
					operations,
				)
			for kernel in self.vision_schema["kernels"]:
				self.assertIn(kernel["kernel_id"], registry)
			self.assertIn('"oa::FnDetection::boxIou"', operations)
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
						f"pub const {oa_util.rust_const_name(operation['name'])}", operations
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
			self.assertIn('"oa::FnMatrix::linear"', operations)
			self.assertIn('"oa::FnLoss::crossEntropy"', operations)
			self.assertIn('"oa::ml::optim::adamw"', operations)
			self.assertIn("pub(in crate::ml) fn record_gelu", activation_autograd)
			self.assertIn("GradNodeOperation::Gelu", activation_autograd)
			self.assertIn("pub(in crate::ml) fn record_relu", activation_autograd)
			self.assertIn("input: &Matrix, output: &Matrix, result: &Matrix", activation_autograd)
			self.assertIn("GradNodeOperation::Relu", activation_autograd)
			self.assertIn("pub(in crate::ml) fn record_leaky_relu", activation_autograd)
			self.assertIn("alpha: f32, result: &Matrix", activation_autograd)
			self.assertIn("GradNodeOperation::LeakyRelu", activation_autograd)
			swiglu_autograd = outputs[root / "src/rs/lib/ml/autograd/matrix/swiglu.gen.rs"]
			self.assertIn("pub(in crate::ml) fn record_swiglu", swiglu_autograd)
			self.assertIn("GradNodeOperation::Swiglu", swiglu_autograd)
			self.assertNotIn("record_swiglu", activation_autograd)
			self.assertNotIn("record_linear", activation_autograd)
			self.assertIn("pub(in crate::ml) fn record_cross_entropy", loss_autograd)
			self.assertIn("pub(in crate::ml) fn record_masked_cross_entropy", loss_autograd)
			self.assertIn("valid_count: usize", loss_autograd)
			self.assertIn('"oa::FnLoss::maskedCrossEntropy"', operations)
			for loss, donor in (("smooth_l1", "smoothL1"), ("mse", "mse"), ("l1", "l1"), ("bce", "bce")):
				self.assertIn(f"pub(in crate::ml) fn record_{loss}", loss_autograd)
				self.assertIn(f'"oa::FnLoss::{donor}"', operations)
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

	def test_fixed_hash_helpers_reject_metadata_and_write_abi_drift(self):
		for field, value in (("bytes", 32), ("bytes", 28.0), ("word_type", "uint2"), ("module", "sha3"), ("helper", "sha2_256")):
			invalid = copy.deepcopy(self.cryptography_hash_schema)
			kernel = next(k for k in invalid["kernels"] if k["name"] == "sha224")
			kernel["digest"][field] = value
			with self.subTest(field=field, value=value), self.assertRaisesRegex(oa_util.SchemaError, "helper ABI"):
				oa_hash.validate_cryptography_hash_schema(invalid)

		invalid = copy.deepcopy(self.cryptography_hash_schema)
		next(k for k in invalid["kernels"] if k["name"] == "sha224")["physical_write"]["writes"][0]["binding"] = 0
		with self.assertRaisesRegex(oa_util.SchemaError, "write ownership"):
			oa_hash.validate_cryptography_hash_schema(invalid)
		invalid = copy.deepcopy(self.cryptography_hash_schema)
		next(a for a in invalid["rust_api"] if a["name"] == "sha224")["values"]["digest_bytes"] = "32"
		with self.assertRaisesRegex(oa_util.SchemaError, "API extent"):
			oa_hash.validate_cryptography_hash_schema(invalid)

	def test_bigint_layout_abi_and_write_ownership_are_checked(self):
		oa_bigint.validate_bigint_schema(self.bigint_schema)
		for mutate in (
			lambda s: s["limb_layout"].update(max_words=129),
			lambda s: s["limb_layout"].update(word_bits=32.0),
			lambda s: s["kernels"][0]["push_fields"].reverse(),
			lambda s: s["kernels"][0]["outputs"][0].update(extent="wide"),
			lambda s: s["kernels"][0]["physical_write"]["writes"][0].update(binding=0),
			lambda s: s["contracts"][-3]["output_kinds"].pop(),
			lambda s: s["kernels"][0].update(stable_id=True),
			lambda s: s.update(schema_version=True),
			lambda s: s["kernels"][-1]["physical_write"]["writes"][0].update(binding=True),
		):
			with self.subTest(mutate=mutate):
				invalid = copy.deepcopy(self.bigint_schema)
				mutate(invalid)
				with self.assertRaises(oa_util.SchemaError):
					oa_bigint.validate_bigint_schema(invalid)

	def test_bigint_product_and_bit_output_contracts_reject_drift(self):
		for name in ("multiply_high", "square_wide", "bit_not", "leading_zeros"):
			for mutate in (
				lambda k: k["outputs"][0].update(extent="wrong"),
				lambda k: k["push_fields"].insert(1, ["unused_index", "uint32"]),
				lambda k: k["physical_write"]["writes"][0].update(extent="wrong"),
			):
				invalid = copy.deepcopy(self.bigint_schema)
				mutate(next(k for k in invalid["kernels"] if k["name"] == name))
				with self.subTest(name=name, mutate=mutate), self.assertRaises(oa_util.SchemaError):
					oa_bigint.validate_bigint_schema(invalid)
			invalid = copy.deepcopy(self.bigint_schema)
			next(c for c in invalid["contracts"] if c["name"] == name)["input_kinds"].append("matrix")
			with self.subTest(name=name), self.assertRaises(oa_util.SchemaError):
				oa_bigint.validate_bigint_schema(invalid)

	def test_bigint_modular_buffer_abi_rejects_drift(self):
		for name in ("reduce_mod", "pow_mod", "barrett_multiply"):
			for mutate in (
				lambda k: k["push_fields"].pop(),
				lambda k: k["outputs"][1].update(extent="row"),
				lambda k: k["physical_write"]["writes"][1].update(binding=0),
			):
				invalid = copy.deepcopy(self.bigint_schema)
				kernel = next(k for k in invalid["kernels"] if k["name"] == name)
				mutate(kernel)
				with self.subTest(name=name, mutate=mutate), self.assertRaises(oa_util.SchemaError):
					oa_bigint.validate_bigint_schema(invalid)
			invalid = copy.deepcopy(self.bigint_schema)
			next(c for c in invalid["contracts"] if c["name"] == name)["input_kinds"].pop()
			with self.subTest(name=name), self.assertRaises(oa_util.SchemaError):
				oa_bigint.validate_bigint_schema(invalid)

	def test_merkle_proof_layout_and_auxiliary_ownership_reject_drift(self):
		for field, value in (("header_bytes", 20), ("version", 1.0), ("max_depth", 33)):
			invalid = copy.deepcopy(self.cryptography_hash_schema)
			invalid["proof_layout"][field] = value
			with self.assertRaisesRegex(oa_util.SchemaError, "wire layout"):
				oa_hash.validate_cryptography_hash_schema(invalid)
		invalid = copy.deepcopy(self.cryptography_hash_schema)
		next(k for k in invalid["kernels"] if k["name"] == "merkle_reduce")["auxiliary_operations"] = ["sha256"]
		with self.assertRaisesRegex(oa_util.SchemaError, "auxiliary ownership"):
			oa_hash.validate_cryptography_hash_schema(invalid)
		invalid = copy.deepcopy(self.cryptography_hash_schema)
		next(a for a in invalid["rust_api"] if a["name"] == "merkle_proofs")["values"]["header_bytes"] = "20"
		with self.assertRaisesRegex(oa_util.SchemaError, "API wire layout"):
			oa_hash.validate_cryptography_hash_schema(invalid)

	def test_fixed_hash_bindings_and_adapters_have_schema_authority(self):
		schema = self.cryptography_hash_schema
		with tempfile.TemporaryDirectory() as directory:
			root = Path(directory)
			outputs = oa_hash.cryptography_hash_binding_outputs(root, schema, "0" * 64)
			oa_publish.publish(outputs, root)
			self.assertEqual(oa_publish.check(outputs, root), [])
			self.assertEqual(oa_publish.owned_outputs(root), set(outputs))
			bindings = outputs[root / "src/py/pyo3/crypto/hash.gen.rs"]
			facade = outputs[root / "src/py/oa/_gen/crypto_hash.py"]
			for api in schema["rust_api"]:
				self.assertIn(f"fn crypto_hash_{api['name']}(", bindings)
				self.assertIn(f"crypto_hash_{api['name']} as hash_{api['name']}", facade)
		for kernel in schema["kernels"]:
			if "digest" not in kernel:
				continue
			adapter = oa_hash.generate_fixed_hash_shader(kernel, "0" * 64)
			self.assertIn(f"import {kernel['digest']['module']};", adapter)
			self.assertIn(f"!{kernel['digest']['helper']}(storage_buffers[push.input_index]", adapter)
			self.assertIn("push.count > input_extent / push.message_len", adapter)

	def test_ui_shader_push_and_workgroup_inventory_matches_source(self) -> None:
		# UI raster stages and compute kernels have their own image/buffer
		# adapter. Keep this schema inventory exact rather than accepting stale
		# ABI descriptions.
		types = {"uint": "uint32", "int": "int32", "float": "float32"}
		for row in (
			self.ui_schema["kernels"] + self.render_schema["kernels"] + self.video_schema["kernels"]
		):
			with self.subTest(kernel=row["name"]):
				path = REPOSITORY_ROOT / row["source"]
				source = path.read_text()
				# Entry files may share the push ABI through a sibling include.
				abi_source = source + "\n".join(
					path.with_name(included).read_text()
					for included in re.findall(r'^#include "([\w.]+\.slang)"', source, re.MULTILINE)
				)
				declaration = re.search(r"\[\[vk::push_constant\]\]\s*(\w+)\s+\w+", abi_source)
				push_name = declaration.group(1) if declaration else "PushConstants"
				push = abi_source.split(f"struct {push_name} {{", 1)
				fields = [[name, types.get(kind, kind)] for kind, name in re.findall(
					r"\b(?:row_major\s+|column_major\s+)?((?:u?int|float)(?:\d(?:x\d)?)?)\s+(\w+)\s*;",
					push[1].split("};", 1)[0],
				)] if len(push) > 1 else []
				self.assertEqual(row["push_fields"], fields)
				if stage := row.get("stage"):
					self.assertIn(f'[shader("{stage}")]', source)
					continue
				workgroup = re.search(r"\[numthreads\((\d+),\s*(\d+),\s*(\d+)\)\]", source)
				self.assertIsNotNone(workgroup)
				self.assertEqual(row["workgroup_size"], [int(value) for value in workgroup.groups()])

	def test_every_slang_entry_point_has_one_schema_owner(self):
		schema_sources = set()
		for schema in (
			self.schema,
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
			self.render_schema,
			self.video_schema,
			self.bigint_schema,
			json.loads((GENERATOR_ROOT / "schema/crypto/crypto_merkle.json").read_text()),
			json.loads((GENERATOR_ROOT / "schema/crypto/crypto_ntt.json").read_text()),
		):
			for row in schema.get("operations", []) + schema.get("kernels", []) + schema.get("private_kernels", []):
				if source := row.get("source"):
					schema_sources.add(source)

		generated_sources = {
			f"src/slang/matrix/elemwise/{variant['source_stem']}.gen.slang"
			for operation in self.schema["operations"]
			if not operation.get("lowering_only", False)
			for variant in oa_model.operation_variants(self.schema, operation)
		}
		generated_sources.update(
			lowering["source"]
			for operation in self.schema["operations"]
			for lowering in oa_model.operation_lowering_variants(operation)
		)
		generated_sources.update(
			f"src/slang/matrix/blas/{operation['source_stem']}.gen.slang"
			for operation in self.blas_schema["operations"]
		)
		generated_sources.update(
			f"src/slang/matrix/blas/{candidate['source_stem']}.gen.slang"
			for candidate in self.blas_schema.get("candidate_kernels", [])
			if "source" not in candidate
		)
		schema_sources.update(
			candidate["source"] for candidate in self.blas_schema.get("candidate_kernels", [])
			if "source" in candidate
		)
		expected = schema_sources | generated_sources
		actual = {
			path.relative_to(REPOSITORY_ROOT).as_posix()
			for root in (SLANG_ROOT, SDK_SLANG_ROOT)
			for path in root.rglob("*.slang")
			if (text := path.read_text(encoding="utf-8"))
			and ("void main(" in text or '[shader("' in text)
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
		with self.assertRaisesRegex(oa_util.SchemaError, "output_tile_size"):
			oa_matrix.validate_blas_schema(invalid)

	def test_reduce_schema_preserves_the_donor_axis_contract(self):
		invalid = copy.deepcopy(self.reduce_schema)
		invalid["operations"][0]["semantic_attributes"] = []
		with self.assertRaisesRegex(oa_util.SchemaError, "signed dim attribute"):
			oa_matrix.validate_reduce_schema(invalid)

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
		with self.assertRaisesRegex(oa_util.SchemaError, "fields are incomplete"):
			oa_matrix.validate_reduce_schema(missing)

		contradictory = copy.deepcopy(self.reduce_schema)
		contradictory["operations"][0]["physical_write"]["writes"][0][
			"partition"
		] = "exclusive_per_invocation"
		with self.assertRaisesRegex(oa_util.SchemaError, "contradictory"):
			oa_matrix.validate_reduce_schema(contradictory)

		unsupported = copy.deepcopy(self.reduce_schema)
		unsupported["operations"][0]["physical_write"]["writes"][0][
			"collision"
		] = "magic"
		with self.assertRaisesRegex(oa_util.SchemaError, "collision is unsupported"):
			oa_matrix.validate_reduce_schema(unsupported)

		duplicate = copy.deepcopy(self.reduce_schema)
		duplicate_write = copy.deepcopy(
			duplicate["operations"][0]["physical_write"]["writes"][0]
		)
		duplicate["operations"][0]["physical_write"]["writes"].append(duplicate_write)
		with self.assertRaisesRegex(oa_util.SchemaError, "duplicate"):
			oa_matrix.validate_reduce_schema(duplicate)

	def test_normalization_candidates_require_physical_write_ownership(self):
		invalid = copy.deepcopy(self.ml_schema)
		layer_norm = next(
			operation for operation in invalid["operations"] if operation["name"] == "layer_norm"
		)
		del layer_norm["physical_write"]
		with self.assertRaisesRegex(oa_util.SchemaError, "physical_write is required"):
			oa_ml.validate_ml_schema(invalid)

	def test_image_kernels_require_physical_write_ownership(self):
		invalid = copy.deepcopy(self.image_schema)
		del invalid["kernels"][0]["physical_write"]
		with self.assertRaisesRegex(oa_util.SchemaError, "physical_write is required"):
			oa_image.validate_image_schema(invalid)

	def test_video_rejects_invalid_fusion_classes(self):
		for section in ("contracts", "kernels"):
			with self.subTest(section=section):
				invalid = copy.deepcopy(self.video_schema)
				invalid[section][0]["fusion_class"] = "unknown"
				with self.assertRaisesRegex(oa_util.SchemaError, "fusion_class is unsupported"):
					oa_video.validate_video_schema(invalid)

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
		with self.assertRaisesRegex(oa_util.SchemaError, "shared contributors"):
			oa_vision.validate_vision_schema(invalid)

		invalid = copy.deepcopy(self.vision_schema)
		invalid["kernels"][2]["physical_write"]["writes"][0][
			"partition"
		] = "exclusive_per_invocation"
		with self.assertRaisesRegex(oa_util.SchemaError, "atomic_u32 requires"):
			oa_vision.validate_vision_schema(invalid)

	def test_unknown_dnn_provider_is_rejected(self):
		invalid = copy.deepcopy(self.blas_schema)
		invalid["operations"][0]["dnn"]["providers"] = ["magic_vendor"]
		with self.assertRaisesRegex(oa_util.SchemaError, "dnn.providers"):
			oa_matrix.validate_blas_schema(invalid)

	def test_invalid_semantic_attribute_is_rejected(self):
		invalid = copy.deepcopy(self.ml_schema)
		invalid["operations"][0]["semantic_attributes"] = [["epsilon", "pointer"]]
		with self.assertRaisesRegex(oa_util.SchemaError, "kind is unsupported"):
			oa_ml.validate_ml_schema(invalid)

	def test_unknown_training_replay_role_is_rejected(self):
		invalid = copy.deepcopy(self.ml_schema)
		invalid["operations"][0]["training_replay_role"] = "host_magic"
		with self.assertRaisesRegex(oa_util.SchemaError, "training_replay_role"):
			oa_ml.validate_ml_schema(invalid)

	def test_unknown_rng_differentiation_is_rejected(self):
		invalid = copy.deepcopy(self.rng_schema)
		next(
			operation for operation in invalid["operations"] if operation["name"] == "dropout"
		)["differentiation"] = "symbolic_magic"
		with self.assertRaisesRegex(oa_util.SchemaError, "differentiation"):
			oa_matrix.validate_rng_schema(invalid)

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
			gather["contract"]["dtype_rule"], "match_input"
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
			"quantize",
			"dequantize",
			"mat_mul_nt_quantized",
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
		with self.assertRaisesRegex(oa_util.SchemaError, "across schemas"):
			oa_registry.registry_entries(
				{
					"elemwise": (self.schema, ""),
					"blas": (invalid, ""),
					"reduce": (self.reduce_schema, ""),
					"rng": (self.rng_schema, ""),
					"index": (self.index_schema, ""),
					"ml": (self.ml_schema, ""),
					"audio": (self.audio_schema, ""),
					"hash": (self.cryptography_hash_schema, ""),
					"pqc": (self.cryptography_pqc_schema, ""),
					"image": (self.image_schema, ""),
					"vision": (self.vision_schema, ""),
				},
			)

	def test_duplicate_stable_id_is_rejected_before_publication(self):
		invalid = copy.deepcopy(self.schema)
		invalid["operations"][1]["stable_id"] = invalid["operations"][0]["stable_id"]
		with self.assertRaisesRegex(oa_util.SchemaError, "duplicate"):
			oa_matrix.validate_schema(invalid)

	def test_scalar_contract_mismatch_is_rejected(self):
		invalid = copy.deepcopy(self.schema)
		invalid["operations"][0]["scalar_name"] = "scalar"
		with self.assertRaisesRegex(oa_util.SchemaError, "unary_scalar"):
			oa_matrix.validate_schema(invalid)

	def test_integer_variant_rejects_float_oracle_values(self):
		invalid = copy.deepcopy(self.schema)
		invalid["operations"][0]["additional_dtype_variants"][0]["test"]["expected"][0] = 1.5
		with self.assertRaisesRegex(oa_util.SchemaError, "i32 values"):
			oa_matrix.validate_schema(invalid)

	# ── UI style token presets ──────────────────────────────────────────────

	def test_ui_style_schema_emits_all_presets_with_resolved_tokens(self):
		style = json.loads(
			(GENERATOR_ROOT / "schema/ui/ui_style.json").read_text(encoding="utf-8")
		)
		oa_uistyle.validate_ui_style_schema(style)
		source = oa_uistyle.generate_ui_style(style, "0" * 64, "ui_style.json")
		self.assertIn("pub(crate) const REALM_DATA_PALETTE: [Color; 6]", source)
		for preset in style["presets"]:
			self.assertIn(f"fn ui_style_{preset['name']}() -> UiStyle", source)
		# Derived presets emit fully resolved fields, not partial overrides.
		viewer = source.split("fn ui_style_viewer_dark()")[1]
		for token in style["color_tokens"] + style["metric_tokens"]:
			self.assertIn(f"{token}:", viewer)
		self.assertIn("data_palette: REALM_DATA_PALETTE", viewer)
		# Hex colors keep the u8-exact constructor; float colors keep literals.
		self.assertIn("Color::new_from_u32(0x181818FF)", viewer)
		self.assertIn("corner_radius: 15.0", viewer)

	def test_ui_style_base_preset_must_resolve_every_token(self):
		style = json.loads(
			(GENERATOR_ROOT / "schema/ui/ui_style.json").read_text(encoding="utf-8")
		)
		invalid = copy.deepcopy(style)
		del invalid["presets"][0]["values"]["background"]
		with self.assertRaisesRegex(oa_util.SchemaError, "every token"):
			oa_uistyle.validate_ui_style_schema(invalid)

	def test_ui_style_derived_preset_must_extend_an_earlier_preset(self):
		style = json.loads(
			(GENERATOR_ROOT / "schema/ui/ui_style.json").read_text(encoding="utf-8")
		)
		invalid = copy.deepcopy(style)
		invalid["presets"][1]["extends"] = "viewer_dark"
		with self.assertRaisesRegex(oa_util.SchemaError, "earlier preset"):
			oa_uistyle.validate_ui_style_schema(invalid)

	def test_ui_style_rejects_unknown_tokens_and_bad_colors(self):
		style = json.loads(
			(GENERATOR_ROOT / "schema/ui/ui_style.json").read_text(encoding="utf-8")
		)
		invalid = copy.deepcopy(style)
		invalid["presets"][2]["values"]["mystery"] = "0x000000FF"
		with self.assertRaisesRegex(oa_util.SchemaError, "unknown tokens"):
			oa_uistyle.validate_ui_style_schema(invalid)
		invalid = copy.deepcopy(style)
		invalid["presets"][2]["values"]["background"] = "0xGG0000FF"
		with self.assertRaisesRegex(oa_util.SchemaError, "0xRRGGBBAA"):
			oa_uistyle.validate_ui_style_schema(invalid)


if __name__ == "__main__":
	unittest.main()
