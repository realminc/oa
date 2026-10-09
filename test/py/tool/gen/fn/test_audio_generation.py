import copy
import importlib.util
import json
import re
import unittest
from pathlib import Path
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[5]
SPEC = importlib.util.spec_from_file_location("oa_audio_generate", ROOT / "tool/gen/fn/generate.py")
GENERATOR = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(GENERATOR)


class AudioGenerationTests(unittest.TestCase):
	def setUp(self):
		self.schema = json.loads((ROOT / "tool/gen/fn/schema/audio/audio.json").read_text())

	def test_all_schema_operations_have_one_complete_generated_body(self):
		for category in ("signal", "transform"):
			text = GENERATOR.generate_audio_api(self.schema, "0" * 64, category)
			expected = [op["name"] for op in self.schema["contracts"] if op["rust"]["category"] == category]
			self.assertEqual(re.findall(r"pub fn (\w+)\(", text), expected)
			self.assertIn("DO NOT EDIT", text)
			self.assertNotIn("{{", text)
			self.assertNotRegex(text, r"\b\w+_(?:impl|dispatch)\(")
			self.assertIn("record_audio_split_semantic", text)
			self.assertIn(f'#[path = "audio/{category}.gen.rs"]', (ROOT / "src/rs/audio.rs").read_text())
			self.assertFalse((ROOT / f"src/rs/audio/{category}.rs").exists())

	def test_signatures_and_attribute_values_are_schema_owned(self):
		op = next(op for op in self.schema["contracts"] if op["name"] == "gain")
		op["rust"]["doc"] = "Apply schema-owned scalar gain."
		op["rust"]["attribute_sources"]["gain_db"] = "gain_db * 0.5"
		text = GENERATOR.generate_audio_api(self.schema, "0" * 64, "signal")
		self.assertIn("pub fn gain(input: &Audio, gain_db: f32)", text)
		self.assertIn("/// Apply schema-owned scalar gain.", text)
		self.assertIn('float_attribute("gain_db", gain_db * 0.5)', text)

	def test_dispatch_packing_rejects_abi_length_and_scalar_type_drift(self):
		for field, replacement in (("bindings", {}), ("push_constants", {"count": ["U32", "count"]}), ("push_constants", {"count": ["F32", "count"], "gain_db": ["F32", "gain_db"]})):
			with self.subTest(field=field, replacement=replacement):
				schema = copy.deepcopy(self.schema)
				op = next(op for op in schema["contracts"] if op["name"] == "gain")
				op["rust"][field][0]["values"] = replacement
				with self.assertRaises(GENERATOR.SchemaError):
					GENERATOR.validate_audio_schema(schema)

	def test_invalid_api_and_template_coverage_are_rejected(self):
		for field, value in (("category", "other"), ("returns", "Matrix"), ("parameters", [["input", "&Audio"], ["input", "f32"]]), ("attribute_sources", []), ("doc", "bad\nsummary")):
			with self.subTest(field=field):
				schema = copy.deepcopy(self.schema)
				schema["contracts"][2]["rust"][field] = value
				with self.assertRaises(GENERATOR.SchemaError):
					GENERATOR.validate_audio_schema(schema)
		with patch.object(GENERATOR, "audio_templates", return_value={}):
			with self.assertRaisesRegex(GENERATOR.SchemaError, "coverage"):
				GENERATOR.validate_audio_schema(self.schema)

	def test_unknown_kernel_and_template_binding_are_rejected(self):
		schema = copy.deepcopy(self.schema)
		schema["contracts"][2]["rust"]["dispatches"][0]["kernel"] = "Missing"
		with self.assertRaisesRegex(GENERATOR.SchemaError, "unknown kernel"):
			GENERATOR.validate_audio_schema(schema)
		op = self.schema["contracts"][2]
		sections = GENERATOR.audio_templates("signal")
		sections["gain"] += "\n{{missing}}"
		with patch.object(GENERATOR, "audio_templates", return_value=sections):
			with self.assertRaisesRegex(GENERATOR.SchemaError, "bindings"):
				GENERATOR.render_audio_body(op, {k["kernel_id"]: k for k in self.schema["kernels"]})

	def test_wire_order_comes_from_kernel_schema_not_expression_map_order(self):
		op = next(op for op in self.schema["contracts"] if op["name"] == "clip")
		op["rust"]["push_constants"][0]["values"] = dict(reversed(list(op["rust"]["push_constants"][0]["values"].items())))
		GENERATOR.validate_audio_schema(self.schema)
		text = GENERATOR.generate_audio_api(self.schema, "0" * 64, "signal")
		clip = text.split("pub fn clip(", 1)[1].split("pub fn saturate(", 1)[0]
		self.assertLess(clip.index("PushConstant::U32(count)"), clip.index("PushConstant::F32(minimum)"))
		self.assertLess(clip.index("PushConstant::F32(minimum)"), clip.index("PushConstant::F32(maximum)"))

	def test_unknown_named_wire_field_is_rejected(self):
		op = next(op for op in self.schema["contracts"] if op["name"] == "gain")
		op["rust"]["push_constants"][0]["values"]["unexpected"] = ["U32", "count"]
		with self.assertRaisesRegex(GENERATOR.SchemaError, "ABI"):
			GENERATOR.validate_audio_schema(self.schema)

	def test_config_types_and_defaults_have_one_schema_owner(self):
		for record in self.schema["types"]:
			text = GENERATOR.generate_type(record)
			self.assertIn(f"{record['kind']} {record['name']} {{", text)
			if record["defaults"] is not None:
				self.assertIn(f"impl Default for {record['name']}", text)
				for name, expression in record["defaults"].items():
					self.assertIn(f"{name}: {expression},", text)

	def test_invalid_type_defaults_and_duplicate_type_identity_are_rejected(self):
		schema = copy.deepcopy(self.schema)
		schema["types"].append(copy.deepcopy(schema["types"][0]))
		with self.assertRaises(GENERATOR.SchemaError):
			GENERATOR.validate_audio_schema(schema)
		schema = copy.deepcopy(self.schema)
		record = next(record for record in schema["types"] if record["name"] == "MelConfig")
		del record["defaults"]["normalize"]
		with self.assertRaisesRegex(GENERATOR.SchemaError, "defaults"):
			GENERATOR.validate_audio_schema(schema)
