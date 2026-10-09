import copy
import importlib.util
import json
import re
import unittest
from pathlib import Path
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[5]
SPEC = importlib.util.spec_from_file_location("oa_image_generate", ROOT / "tool/gen/fn/generate.py")
GENERATOR = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(GENERATOR)


class ImageGenerationTests(unittest.TestCase):
	def setUp(self):
		self.schema = json.loads((ROOT / "tool/gen/fn/schema/image/image.json").read_text())

	def operation(self, name):
		return next(op for op in self.schema["contracts"] if op["name"] == name)

	def test_complete_categories_are_directly_declared_with_one_owner(self):
		counts = {"pixel": 20, "filter": 20, "geometric": 9, "color": 4}
		facade = (ROOT / "src/rs/image.rs").read_text()
		for category, count in counts.items():
			text = GENERATOR.generate_image_api(self.schema, "0" * 64, category)
			names = [op["name"] for op in self.schema["contracts"] if op["rust"]["category"] == category]
			self.assertEqual(len(names), count)
			self.assertEqual(re.findall(r"pub fn (\w+)\(", text), names)
			self.assertEqual(text.count("record_image_semantic("), count)
			self.assertIn("DO NOT EDIT", text)
			self.assertNotIn("{{", text)
			self.assertNotRegex(text, r"\b\w+_(?:impl|dispatch)\(")
			self.assertIn(f'#[path = "image/{category}.gen.rs"]', facade)
			self.assertFalse((ROOT / f"src/rs/image/{category}.rs").exists())

	def test_schema_owns_docs_signature_and_semantic_attribute_expressions(self):
		op = self.operation("threshold_binary")
		op["rust"]["doc"] = ["A schema-owned summary.", "", "Full operation documentation."]
		op["rust"]["attribute_sources"]["threshold"]["value"] = "threshold * 0.5"
		text = GENERATOR.generate_image_api(self.schema, "0" * 64, "pixel")
		self.assertIn("/// A schema-owned summary.\n///\n/// Full operation documentation.", text)
		self.assertIn("pub fn threshold_binary(input: &Image, threshold: f32, max_value: f32)", text)
		self.assertIn('float("threshold", threshold * 0.5)', text)

	def test_malformed_generation_records_produce_schema_errors(self):
		original = copy.deepcopy(self.schema)
		for value in (None, [], {"category": []}):
			self.schema = copy.deepcopy(original)
			self.operation("threshold_binary")["rust"] = value
			with self.assertRaises(GENERATOR.SchemaError):
				GENERATOR.validate_image_schema(self.schema)
		self.schema = copy.deepcopy(original)
		self.schema["types"] = self.schema["types"][1:]
		with self.assertRaisesRegex(GENERATOR.SchemaError, "selector"):
			GENERATOR.validate_image_schema(self.schema)

	def test_named_wire_order_is_independent_of_expression_map_order(self):
		op = self.operation("threshold_binary")
		before = GENERATOR.generate_image_api(self.schema, "0" * 64, "pixel")
		for field in ("bindings", "push_constants"):
			values = op["rust"][field][0]["values"]
			op["rust"][field][0]["values"] = dict(reversed(list(values.items())))
		GENERATOR.validate_image_schema(self.schema)
		self.assertEqual(before, GENERATOR.generate_image_api(self.schema, "0" * 64, "pixel"))

	def test_named_binding_declarations_and_references_share_schema_identity(self):
		op = self.operation("alpha_blend")
		op["rust"]["bindings"][0]["name"] = "blend_bindings"
		GENERATOR.validate_image_schema(self.schema)
		text = GENERATOR.generate_image_api(self.schema, "0" * 64, "pixel")
		self.assertIn("let blend_bindings = [", text)
		self.assertIn("buffers: &blend_bindings", text)

	def test_push_limit_is_checked_for_every_kernel(self):
		kernel = self.schema["kernels"][0]
		kernel["push_fields"].extend([[f"extra_{i}", "uint32"] for i in range(33)])
		with self.assertRaisesRegex(GENERATOR.SchemaError, "minimum guarantee"):
			GENERATOR.validate_image_schema(self.schema)

	def test_wire_and_semantic_drift_are_rejected_before_publication(self):
		mutations = [
			lambda op: op["rust"]["push_constants"][0]["values"].pop("element_count"),
			lambda op: op["rust"]["push_constants"][0]["values"]["element_count"].__setitem__(0, "F32"),
			lambda op: op["rust"]["bindings"][0]["values"].update(extra_index=["read", "input"]),
			lambda op: op["rust"]["attribute_sources"]["threshold"].update(encoding="unsigned"),
			lambda op: op["rust"]["parameters"][0].__setitem__(1, "&Matrix"),
			lambda op: op["rust"]["dispatches"][0].update(kernels=["ImageCropF32"]),
		]
		original = copy.deepcopy(self.schema)
		for mutation in mutations:
			self.schema = copy.deepcopy(original)
			mutation(self.operation("threshold_binary"))
			with self.assertRaises(GENERATOR.SchemaError):
				GENERATOR.validate_image_schema(self.schema)

	def test_resize_preserves_both_variants_and_rejects_incomplete_coverage(self):
		op = self.operation("resize")
		self.assertEqual(op["rust"]["dispatches"][0]["kernels"], ["ImageResizeNearestF32", "ImageResizeBilinearF32"])
		op["rust"]["dispatches"][0].update(kernels=["ImageResizeNearestF32"], kernel="KernelId::ImageResizeNearestF32")
		with self.assertRaisesRegex(GENERATOR.SchemaError, "coverage"):
			GENERATOR.validate_image_schema(self.schema)

	def test_supporting_types_preserve_arrays_defaults_and_enum_mappings(self):
		color = GENERATOR.generate_image_api(self.schema, "0" * 64, "color")
		self.assertIn("pub mean: [f32; 3]", color)
		self.assertIn("mean: [0.0; 3]", color)
		self.assertIn("std: [1.0; 3]", color)
		geometric = GENERATOR.generate_image_api(self.schema, "0" * 64, "geometric")
		self.assertIn("#[default]\nBilinear", geometric)
		self.assertIn('Self::Reflect101 => "reflect_101"', geometric)
		self.assertIn("Self::Reflect101 => 3", geometric)
		self.assertIn("Self::Nearest => KernelId::ImageResizeNearestF32", geometric)
		self.assertIn("struct ImageExtent", geometric)
		self.assertIn("#[allow(clippy::too_many_arguments)]\npub fn warp_affine", geometric)

	def test_type_mapping_and_default_drift_are_rejected(self):
		original = copy.deepcopy(self.schema)
		for field, value in (("code", "4294967296"), ("token", "nearest"), ("kernel", "KernelId::Unknown"), ("kernel", "KernelId::ImageCropF32")):
			self.schema = copy.deepcopy(original)
			self.schema["types"][0]["variants"][1][field] = value
			with self.assertRaises(GENERATOR.SchemaError):
				GENERATOR.validate_image_schema(self.schema)
		self.schema = copy.deepcopy(original)
		del self.schema["types"][-1]["defaults"]["std"]
		with self.assertRaisesRegex(GENERATOR.SchemaError, "defaults"):
			GENERATOR.validate_image_schema(self.schema)

	def test_template_coverage_and_unknown_placeholders_are_rejected(self):
		with patch.object(GENERATOR, "image_templates", return_value={}):
			with self.assertRaisesRegex(GENERATOR.SchemaError, "coverage"):
				GENERATOR.validate_image_schema(self.schema)
		sections = GENERATOR.image_templates("pixel")
		sections["threshold_binary"] += "\n{{unknown}}"
		with patch.object(GENERATOR, "image_templates", return_value=sections):
			with self.assertRaisesRegex(GENERATOR.SchemaError, "bindings"):
				GENERATOR.render_image_body(self.operation("threshold_binary"), {k["kernel_id"]: k for k in self.schema["kernels"]})


if __name__ == "__main__":
	unittest.main()
