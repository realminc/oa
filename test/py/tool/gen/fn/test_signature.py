import copy
import ast
import importlib.util
import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[5]
GENERATOR_ROOT = ROOT / "tool/gen/fn"

import sys
sys.path.insert(0, str(GENERATOR_ROOT))
import oafn.audio as oa_audio
import oafn.matrix as oa_matrix
import oafn.python as oa_python
import oafn.signature as oa_signature
import oafn.util as oa_util
import oafn.vision as oa_vision



def load_schema(name: str) -> dict:
	return json.loads((GENERATOR_ROOT / "schema" / name).read_text())


class SignatureModelTests(unittest.TestCase):
	def setUp(self):
		self.elemwise = load_schema("matrix/matrix_elemwise.json")
		self.blas = load_schema("matrix/matrix_blas.json")
		self.reduce = load_schema("matrix/matrix_reduce.json")
		self.audio = load_schema("audio/audio.json")
		self.image = load_schema("image/image.json")
		self.vision = load_schema("vision/vision_detection.json")
		self.cryptography = load_schema("crypto/crypto_hash.json")

	def signature(self, domain: str, operation: dict) -> dict:
		result = oa_signature.operation_signature(domain, operation, "test")
		self.assertIsNotNone(result)
		return result

	def test_kind_derived_matrix_signatures(self):
		operations = {op["name"]: op for op in self.elemwise["operations"]}
		self.assertEqual(
			oa_signature.signature_header(self.signature("matrix", operations["add"])),
			"pub fn add(left: &Matrix, right: &Matrix) -> Result<Matrix>",
		)
		self.assertEqual(
			oa_signature.signature_header(self.signature("matrix", operations["neg"])),
			"pub fn neg(input: &Matrix) -> Result<Matrix>",
		)
		self.assertEqual(
			oa_signature.signature_header(self.signature("matrix", operations["pow"])),
			"pub fn pow(input: &Matrix, exponent: f32) -> Result<Matrix>",
		)
		self.assertEqual(self.signature("matrix", operations["add"])["receiver"], "left")
		self.assertEqual(self.signature("matrix", operations["neg"])["receiver"], "input")

	def test_blas_signature_is_kind_derived(self):
		operation = self.blas["operations"][0]
		signature = self.signature("matrix", operation)
		self.assertEqual(
			oa_signature.signature_header(signature),
			"pub fn mat_mul_nt(left: &Matrix, right: &Matrix) -> Result<Matrix>",
		)
		self.assertEqual(signature["receiver"], "left")

	def test_reduce_signature_records(self):
		operations = {op["name"]: op for op in self.reduce["operations"]}
		self.assertEqual(
			oa_signature.signature_header(self.signature("matrix", operations["softmax"])),
			"pub fn softmax(input: &Matrix, dim: i32) -> Result<Matrix>",
		)
		self.assertEqual(
			oa_signature.signature_header(self.signature("matrix", operations["softmax_backward"])),
			"pub(crate) fn softmax_backward(forward_output: &Matrix, output_gradient: &Matrix, dim: i32) -> Result<Matrix>",
		)
		self.assertEqual(
			oa_signature.signature_header(self.signature("matrix", operations["categorical_accuracy_count"])),
			"pub fn categorical_accuracy_count(logits: &Matrix, labels: &Matrix) -> Result<Matrix>",
		)
		self.assertEqual(
			oa_signature.signature_header(self.signature("matrix", operations["masked_categorical_accuracy_count"])),
			"pub fn masked_categorical_accuracy_count(logits: &Matrix, labels: &Matrix, mask: &Matrix) -> Result<Matrix>",
		)
		self.assertEqual(
			self.signature("matrix", operations["categorical_accuracy_count"])["receiver"],
			"logits",
		)

	def test_lowering_only_operation_has_no_signature(self):
		operation = next(op for op in self.reduce["operations"] if op["name"] == "sum_axis")
		self.assertIsNone(oa_signature.operation_signature("matrix", operation, "test"))

	def test_audio_rust_record_signature(self):
		operations = {op["name"]: op for op in self.audio["contracts"]}
		gain = self.signature("audio", operations["gain"])
		self.assertEqual(
			oa_signature.signature_header(gain),
			"pub fn gain(input: &Audio, gain_db: f32) -> Result<Audio>",
		)
		self.assertEqual(gain["receiver"], "input")
		self.assertEqual(
			oa_signature.signature_header(self.signature("audio", operations["amplitude_to_db"])),
			"pub fn amplitude_to_db(input: &Audio, floor_db: f32) -> Result<Matrix>",
		)

	def test_image_rust_record_signature(self):
		operations = {op["name"]: op for op in self.image["contracts"]}
		resize = self.signature("image", operations["resize"])
		self.assertEqual(
			oa_signature.signature_header(resize),
			"pub fn resize(input: &Image, target_width: u32, target_height: u32, interpolation: InterpolationMode) -> Result<Image>",
		)
		self.assertEqual(resize["receiver"], "input")

	def test_vision_operations_have_no_receiver(self):
		operations = {op["name"]: op for op in self.vision["contracts"]}
		box_iou = self.signature("vision", operations["box_iou"])
		self.assertEqual(
			oa_signature.signature_header(box_iou),
			"pub fn box_iou(left: &Matrix, right: &Matrix) -> Result<Matrix>",
		)
		self.assertIsNone(box_iou["receiver"])

	def test_cryptography_api_signature(self):
		api = self.cryptography["rust_api"][0]
		signature = oa_signature.api_signature("crypto", api, "test")
		header = oa_signature.signature_header(signature)
		self.assertTrue(header.startswith(f"{api['visibility']} fn {api['name']}("))
		self.assertTrue(header.endswith("-> Result<Matrix>"))
		self.assertIsNone(signature["receiver"])

	def test_emitted_declarations_match_the_model(self):
		for domain, operation, emit in (
			("matrix", self.elemwise["operations"][0], lambda: oa_matrix.generate_api(self.elemwise, "0" * 64)),
			("matrix", self.blas["operations"][0], lambda: oa_matrix.generate_blas_api(self.blas, "0" * 64)),
			("audio", self.audio["contracts"][0], lambda: oa_audio.generate_audio_api(self.audio, "0" * 64, "signal")),
			("vision", self.vision["contracts"][0], lambda: oa_vision.generate_vision_api(self.vision, "0" * 64)),
		):
			header = oa_signature.signature_header(self.signature(domain, operation))
			self.assertIn(header + " {", emit())

	def test_signature_rejects_unknown_or_missing_fields(self):
		operation = {
			"name": "broken",
			"signature": {
				"parameters": [["input", "&Matrix"]],
				"returns": "Matrix",
				"unknown": True,
			},
		}
		with self.assertRaises(oa_util.SchemaError):
			oa_signature.operation_signature("matrix", operation, "test")
		operation["signature"].pop("unknown")
		operation["signature"].pop("returns")
		with self.assertRaises(oa_util.SchemaError):
			oa_signature.operation_signature("matrix", operation, "test")

	def test_signature_rejects_duplicate_and_invalid_parameters(self):
		for parameters in (
			[["input", "&Matrix"], ["input", "&Matrix"]],
			[["9bad", "&Matrix"]],
			[["input"]],
			[],
		):
			operation = {
				"name": "broken",
				"signature": {"parameters": parameters, "returns": "Matrix"},
			}
			with self.assertRaises(oa_util.SchemaError, msg=repr(parameters)):
				oa_signature.operation_signature("matrix", operation, "test")

	def test_receiver_policy(self):
		operation = {
			"name": "module_only",
			"signature": {
				"parameters": [["input", "&Matrix"]],
				"returns": "Matrix",
				"receiver": "none",
			},
		}
		self.assertIsNone(self.signature("matrix", operation)["receiver"])
		for receiver in ("left", "dim"):
			operation["signature"]["receiver"] = receiver
			operation["signature"]["parameters"] = [["input", "&Matrix"], ["dim", "i32"]]
			with self.assertRaises(oa_util.SchemaError, msg=receiver):
				oa_signature.operation_signature("matrix", operation, "test")
		operation["signature"]["parameters"] = [["input", "&Matrix"]]
		operation["signature"]["receiver"] = "input"
		self.assertEqual(self.signature("matrix", operation)["receiver"], "input")

	def test_reduce_schema_validates_signature_records(self):
		oa_matrix.validate_reduce_schema(self.reduce)
		broken = copy.deepcopy(self.reduce)
		softmax = next(op for op in broken["operations"] if op["name"] == "softmax")
		softmax["signature"]["parameters"] = [["input", "&Matrix"]]
		with self.assertRaises(oa_util.SchemaError):
			oa_matrix.validate_reduce_schema(broken)
		broken = copy.deepcopy(self.reduce)
		sum_axis = next(op for op in broken["operations"] if op["name"] == "sum_axis")
		sum_axis["signature"] = {"parameters": [["input", "&Matrix"]], "returns": "Matrix"}
		with self.assertRaises(oa_util.SchemaError):
			oa_matrix.validate_reduce_schema(broken)

	def test_matrix_kind_families_reject_duplicate_signature_records(self):
		broken = copy.deepcopy(self.elemwise)
		broken["operations"][0]["signature"] = {
			"parameters": [["left", "&Matrix"], ["right", "&Matrix"]],
			"returns": "Matrix",
		}
		with self.assertRaises(oa_util.SchemaError):
			oa_matrix.validate_schema(broken)
		broken_blas = copy.deepcopy(self.blas)
		broken_blas["operations"][0]["signature"] = {
			"parameters": [["left", "&Matrix"], ["right", "&Matrix"]],
			"returns": "Matrix",
		}
		with self.assertRaises(oa_util.SchemaError):
			oa_matrix.validate_blas_schema(broken_blas)

	def test_python_methods_cover_all_audio_and_image_operations(self):
		for domain, schema in (("audio", self.audio), ("image", self.image)):
			signatures = oa_signature.domain_signatures(domain, [schema])
			rust, stub, methods = oa_python.generate_python_domain(domain, signatures, "0" * 64, "test")
			for signature in signatures:
				self.assertIn(f"fn {domain}_{signature['name']}(", rust)
				self.assertIn(f"def {domain}_{signature['name']}(", stub)
				self.assertIn(f"def {signature['name']}(self", methods)
			self.assertIn('interpolation="bilinear"' if domain == "image" else 'target_db=-3.0, mode="peak"', rust)
			ast.parse(stub)

	def test_python_defaults_and_config_conversions_share_one_call(self):
		signatures = oa_signature.domain_signatures("audio", [self.audio])
		rust, stub, methods = oa_python.generate_python_domain("audio", signatures, "0" * 64, "test")
		self.assertEqual(rust.count("oa::audio::NormalizeAudioConfig {"), 2)
		self.assertEqual(rust.count("mode: super::normalize_mode(mode)?"), 2)
		self.assertEqual(rust.count("oa::audio::MfccConfig {"), 2)
		self.assertEqual(rust.count(".collect::<Vec<_>>()"), 2)
		self.assertIn("center: bool = True", stub)
		self.assertIn("normalize: bool = False", methods)
		self.assertNotIn("$", rust)

	def test_python_adaptations_reject_invalid_defaults_and_conversions(self):
		original = next(op for op in self.audio["contracts"] if op["name"] == "normalize")
		mutations = (
			lambda p: p["defaults"].update(unknown=1),
			lambda p: p["defaults"].update(target_db="bad"),
			lambda p: p["defaults"].update(mode=1),
			lambda p: p["defaults"].pop("mode"),
			lambda p: p["arguments"].update(config="$unknown"),
			lambda p: p["arguments"].update(config="$mode;"),
			lambda p: p["arguments"].update(input="$mode"),
			lambda p: p["arguments"].clear(),
			lambda p: p["parameters"].append(["unused", "f32"]),
			lambda p: p["parameters"].append(["mode", "f32"]),
			lambda p: p["parameters"][0].__setitem__(0, "other"),
			lambda p: p["parameters"][1].__setitem__(1, "UnknownType"),
		)
		for mutation in mutations:
			operation = copy.deepcopy(original)
			mutation(operation["python"])
			with self.assertRaises(oa_util.SchemaError):
				self.signature("audio", operation)

	def test_native_stub_methods_and_exports_follow_schema_changes(self):
		template = (GENERATOR_ROOT / "template/python/native.pyi.in").read_text()
		gain = copy.deepcopy(next(op for op in self.audio["contracts"] if op["name"] == "gain"))
		gain["rust"]["parameters"][1][0] = "amount"
		signature = self.signature("audio", gain)
		_, fragment, methods = oa_python.generate_python_domain("audio", [signature], "0" * 64, "test")
		stub = oa_python.generate_native_stub(template, {"matrix": "", "audio": methods, "image": ""}, fragment)
		self.assertIn("def gain(self, amount: float)", stub)
		self.assertNotIn("def gain(self, gain_db:", stub)
		self.assertIn("audio_gain as audio_gain", stub)
		ast.parse(stub)
		missing = template.replace("{{audio_methods}}", "")
		with self.assertRaises(oa_util.SchemaError):
			oa_python.generate_native_stub(missing, {"audio": methods}, fragment)

	def test_rust_methods_have_compile_time_free_method_signature_parity(self):
		signatures = oa_signature.domain_signatures("audio", [self.audio])
		text = oa_signature.generate_domain_methods("audio", signatures, "0" * 64, "test")
		self.assertIn("const _: fn(&Audio, f32) -> Result<Audio> = crate::audio::gain;", text)
		self.assertIn("const _: fn(&Audio, f32) -> Result<Audio> = Audio::gain;", text)


if __name__ == "__main__":
	unittest.main()
