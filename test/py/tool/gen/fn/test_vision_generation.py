import copy
import importlib.util
import json
import re
import unittest
from pathlib import Path
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[5]

import sys
sys.path.insert(0, str(ROOT / "tool/gen/fn"))
import oafn.util as oa_util
import oafn.vision as oa_vision



class VisionGenerationTests(unittest.TestCase):
	def setUp(self):
		self.schema = json.loads((ROOT / "tool/gen/fn/schema/vision/vision_detection.json").read_text())

	def operation(self, name):
		return next(op for op in self.schema["contracts"] if op["name"] == name)

	def render(self):
		return oa_vision.generate_vision_api(self.schema, "0" * 64)

	def test_complete_bodies_types_and_umbrella_have_one_owner(self):
		oa_vision.validate_vision_schema(self.schema)
		text = self.render()
		self.assertEqual(re.findall(r"pub fn (\w+)\(", text), [op["name"] for op in self.schema["contracts"]])
		self.assertEqual(text.count("record_semantic("), 2)
		self.assertEqual(text.count("record_split_semantic("), 4)
		self.assertIn("DO NOT EDIT", text)
		self.assertNotIn("{{", text)
		self.assertNotRegex(text, r"\b\w+_(?:impl|dispatch)\(")
		self.assertIn('#[path = "vision/detection.gen.rs"]', (ROOT / "src/rs/vision.rs").read_text())
		self.assertNotIn("@generated", (ROOT / "src/rs/vision/detection.rs").read_text())
		for record in self.schema["types"]:
			self.assertEqual(text.count(f"pub struct {record['name']} {{"), 1)
		self.assertIn("pub max_detections: i32", text)
		self.assertIn("pub count: Matrix", text)
		self.assertIn("max_detections: 100", text)
		self.assertNotIn("#[derive()]", text)

	def test_all_stages_preserve_clear_before_atomic_accumulation(self):
		expected = {
			"confusion_matrix": ["VisionConfusionMatrixClearU32", "VisionConfusionMatrixI32"],
			"binary_mask_counts": ["VisionBinaryMaskCountsClearU32", "VisionBinaryMaskCountsU8"],
			"evaluate": ["VisionDetectionMetricCurvesF32", "VisionDetectionAveragePrecisionF32", "VisionDetectionMeanAveragePrecisionF32"],
			"evaluate_segmentation": ["VisionSegmentationConfusionClearU32", "VisionSegmentationConfusionI32", "VisionSegmentationMetricsF32"],
		}
		for name, kernels in expected.items():
			self.assertEqual([stage["kernels"][0] for stage in self.operation(name)["rust"]["dispatches"]], kernels)
		text = self.render()
		self.assertEqual(text.count("BufferBinding::read_write("), 4)

	def test_signature_docs_defaults_and_attributes_are_schema_owned(self):
		op = self.operation("nms")
		op["rust"]["doc"] = ["Schema summary."]
		op["rust"]["attribute_sources"]["scoreThreshold"]["value"] = "f64::from(config.score_threshold * 0.5)"
		self.schema["types"][0]["defaults"]["max_detections"] = "200"
		text = self.render()
		self.assertIn("/// Schema summary.", text)
		self.assertIn("f64::from(config.score_threshold * 0.5)", text)
		self.assertIn("max_detections: 200", text)

	def test_wire_map_order_does_not_change_output(self):
		before = self.render()
		for op in self.schema["contracts"]:
			for field in ("bindings", "push_constants"):
				for declaration in op["rust"][field]:
					declaration["values"] = dict(reversed(list(declaration["values"].items())))
		oa_vision.validate_vision_schema(self.schema)
		self.assertEqual(before, self.render())

	def test_declaration_names_and_references_share_schema_identity(self):
		op = self.operation("confusion_matrix")
		op["rust"]["bindings"][0]["name"] = "zero_bindings"
		op["rust"]["push_constants"][0]["name"] = "zero_push"
		oa_vision.validate_vision_schema(self.schema)
		text = self.render()
		self.assertIn("let zero_bindings = [", text)
		self.assertIn("buffers: &zero_bindings", text)
		self.assertIn("let zero_push = [", text)
		self.assertIn("push_constants: &zero_push", text)

	def test_missing_result_owner_and_field_mismatch_are_rejected(self):
		self.schema["types"].pop()
		with self.assertRaisesRegex(oa_util.SchemaError, "result fields"):
			oa_vision.validate_vision_schema(self.schema)
		self.setUp()
		self.schema["types"][1]["fields"].pop()
		with self.assertRaisesRegex(oa_util.SchemaError, "result fields"):
			oa_vision.validate_vision_schema(self.schema)

	def test_bad_signatures_attributes_abi_and_stage_coverage_are_rejected(self):
		mutations = [
			lambda op: op["rust"]["parameters"][0].__setitem__(1, "&Image"),
			lambda op: op["rust"].update(returns="NmsResult"),
			lambda op: op["rust"]["push_constants"][0]["values"]["rows_a"].__setitem__(0, "F32"),
			lambda op: op["rust"]["bindings"][0]["values"].pop("output_index"),
			lambda op: op["rust"]["dispatches"][0].update(kernels=["VisionNmsF32"]),
			lambda op: op["rust"]["attribute_sources"].update(extra={"encoding": "Float", "value": "0.0"}),
		]
		original = copy.deepcopy(self.schema)
		for mutation in mutations:
			self.schema = copy.deepcopy(original)
			mutation(self.operation("box_iou"))
			with self.assertRaises(oa_util.SchemaError):
				oa_vision.validate_vision_schema(self.schema)
		self.schema = copy.deepcopy(original)
		self.operation("evaluate")["rust"]["dispatches"].pop()
		with self.assertRaises(oa_util.SchemaError):
			oa_vision.validate_vision_schema(self.schema)

	def test_template_coverage_and_unresolved_tokens_are_rejected(self):
		with patch.object(oa_vision, "vision_templates", return_value={}):
			with self.assertRaisesRegex(oa_util.SchemaError, "coverage"):
				oa_vision.validate_vision_schema(self.schema)
		templates = oa_vision.vision_templates()
		templates["box_iou"] += "\n{{unknown}}"
		with patch.object(oa_vision, "vision_templates", return_value=templates):
			with self.assertRaisesRegex(oa_util.SchemaError, "bindings"):
				oa_vision.validate_vision_schema(self.schema)


if __name__ == "__main__":
	unittest.main()
