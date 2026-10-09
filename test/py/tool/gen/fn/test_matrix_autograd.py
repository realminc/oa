"""Contracts for schema-owned foundational Matrix differentiation."""

import copy
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from .test_generate import GENERATOR, GENERATOR_ROOT


class MatrixAutogradSchemaTests(unittest.TestCase):
	def setUp(self):
		loaders = {
			"elemwise": GENERATOR.load_schema, "blas": GENERATOR.load_blas_schema,
			"reduce": GENERATOR.load_reduce_schema, "rng": GENERATOR.load_rng_schema,
			"index": GENERATOR.load_index_schema, "view": GENERATOR.load_view_schema,
		}
		self.schemas = {name: loader(GENERATOR_ROOT / f"schema/matrix/matrix_{name}.json")[0] for name, loader in loaders.items()}

	def validate_mutation(self, change, family="elemwise"):
		schema = copy.deepcopy(self.schemas[family])
		change(schema)
		with self.assertRaises(GENERATOR.SchemaError):
			GENERATOR.validate_matrix_autograd_schema(schema)

	def test_existing_catalog_has_25_records_and_four_composition_aliases(self):
		rows = GENERATOR.matrix_autograd_rows(list(self.schemas.values()))
		self.assertEqual(len(rows), 25)
		self.assertEqual(len({op["autograd"]["node"] for op in rows}), 25)
		aliases = {op["name"]: op["autograd"]["target"] for op in self.schemas["elemwise"]["operations"] if op.get("autograd", {}).get("mode") == "alias"}
		self.assertEqual(aliases, {"neg": "scale", "add_scalar": "scale", "sub_scalar": "scale", "div_scalar": "scale"})

	def test_every_reverse_operation_requires_policy(self):
		self.validate_mutation(lambda s: s["operations"][0].pop("autograd"))

	def test_detached_operation_cannot_attach(self):
		self.validate_mutation(lambda s: s["operations"][0].update(differentiation="none"))

	def test_family_cannot_cross_domain_ownership(self):
		self.validate_mutation(lambda s: s["operations"][0]["autograd"].update(family="ml/matrix"))

	def test_unrecognized_policy_or_rust_type_is_rejected(self):
		for change in [lambda a: a.update(body="unsafe {}"), lambda a: a["saved_fields"].append(["extra", "*mut Matrix"])]:
			with self.subTest(change=change):
				self.validate_mutation(lambda s: change(s["operations"][0]["autograd"]))

	def test_saved_fields_and_node_identities_are_unique(self):
		self.validate_mutation(lambda s: s["operations"][0]["autograd"]["saved_fields"].append(["left", "matrix"]))
		self.validate_mutation(lambda s: s["operations"][1]["autograd"].update(node="Add"))

	def test_output_identity_and_output_argument_are_reserved(self):
		for field in [["output_id", "matrix"], ["other_id", "output_identity"], ["output", "usize"], ["gradients", "f32"], ["gen", "matrix"], ["_", "matrix"]]:
			with self.subTest(field=field):
				self.validate_mutation(lambda s: s["operations"][0]["autograd"]["saved_fields"].append(field))

	def test_alias_cannot_reference_missing_or_another_alias(self):
		for target in ["missing", "neg", [], "unsafe {}"]:
			with self.subTest(target=target):
				self.validate_mutation(lambda s: next(op for op in s["operations"] if op["name"] == "neg")["autograd"].update(target=target))

	def test_alias_policy_controls_the_forward_attachment(self):
		schema = copy.deepcopy(self.schemas["elemwise"])
		next(op for op in schema["operations"] if op["name"] == "neg")["autograd"]["target"] = "clamp_min"
		GENERATOR.validate_matrix_autograd_schema(schema)
		output = GENERATOR.generate_api(schema, "1" * 64)
		neg = output.split("pub fn neg(", 1)[1].split("pub fn abs(", 1)[0]
		self.assertIn("autograd::record_clamp_min(input, &output, -1.0)", neg)

	def test_concat_borrows_matrix_list_and_moves_size_state(self):
		rows = [op for op in GENERATOR.matrix_autograd_rows([self.schemas["index"]]) if op["name"] == "concat"]
		output = GENERATOR.generate_matrix_attachments(rows, "1" * 64, "index")
		self.assertIn("inputs: &[Matrix]", output)
		self.assertIn("sizes: Vec<usize>", output)
		self.assertIn("inputs: inputs.to_vec()", output)
		self.assertIn("dim, sizes", output)
		self.assertNotIn("sizes.clone()", output)

	def test_saved_forward_output_is_borrowed_and_retained_once(self):
		rows = [op for op in GENERATOR.matrix_autograd_rows([self.schemas["reduce"]]) if op["name"] == "softmax"]
		output = GENERATOR.generate_matrix_attachments(rows, "1" * 64, "reduce")
		self.assertEqual(output.count("output: &Matrix"), 1)
		self.assertEqual(output.count("output.clone()"), 1)
		self.assertIn("output_id: output.value_id()", output)

	def test_recipe_requires_exact_saved_field_interface(self):
		rows = copy.deepcopy(GENERATOR.matrix_autograd_rows([self.schemas["view"]]))
		rows[0]["autograd"]["saved_fields"].append(["extra", "usize"])
		with self.assertRaisesRegex(GENERATOR.SchemaError, "saved fields differ"):
			GENERATOR.matrix_autograd_recipes("view", rows)

	def test_missing_duplicate_and_extra_recipes_fail(self):
		rows = GENERATOR.matrix_autograd_rows([self.schemas["view"]])
		original = (GENERATOR_ROOT / "template/matrix/autograd/view.rs.in").read_text()
		for text in ["", original + original, original + "\n// @operation extra\n// @fields input,output_id\nOk(false)\n"]:
			with self.subTest(text=text), mock.patch.object(Path, "read_text", return_value=text):
				with self.assertRaises(GENERATOR.SchemaError):
					GENERATOR.matrix_autograd_recipes("view", rows)

	def test_routing_is_exhaustive_without_fallback(self):
		rows = GENERATOR.matrix_autograd_rows(list(self.schemas.values()))
		output = GENERATOR.generate_matrix_autograd_routing(rows, "1" * 64)
		for row in rows:
			self.assertIn(f"Self::{row['autograd']['node']}", output)
			self.assertIn(f"{row['autograd']['family']}::{row['name']}(", output)
		self.assertNotIn("_ =>", output)
		self.assertNotIn("unreachable!", output)

	def test_cross_schema_node_collision_is_rejected(self):
		schemas = copy.deepcopy(list(self.schemas.values()))
		schemas[1]["operations"][0]["autograd"]["node"] = "Add"
		with self.assertRaises(GENERATOR.SchemaError):
			GENERATOR.matrix_autograd_rows(schemas)

	def test_numerical_backward_generation_is_complete_and_tape_independent(self):
		for family in ("elemwise", "blas"):
			rows = GENERATOR.matrix_autograd_rows([self.schemas[family]])
			output = GENERATOR.generate_matrix_backward_providers(self.schemas[family], family)
			self.assertEqual(output.count("pub(crate) fn "), len(rows))
			for row in rows:
				self.assertIn(f"fn {row['name']}_backward(", output)
			self.assertNotIn("GradientContext", output)
			self.assertNotIn("gradients.", output)

	def test_numerical_backward_recipes_reject_missing_duplicate_and_tape_access(self):
		original = (GENERATOR_ROOT / "template/matrix/backward/blas.rs.in").read_text()
		for text in ("", original + original, original.replace("let [rows, inner]", "gradients.take_output(0);\nlet [rows, inner]")):
			with self.subTest(text=text), mock.patch.object(Path, "read_text", return_value=text):
				with self.assertRaises(GENERATOR.SchemaError):
					GENERATOR.generate_matrix_backward_providers(self.schemas["blas"], "blas")

	def test_reverse_adapters_reject_formulas_and_wrong_provider(self):
		rows = GENERATOR.matrix_autograd_rows([self.schemas["view"]])
		original = (GENERATOR_ROOT / "template/matrix/autograd/view.rs.in").read_text()
		for text in (original.replace("matrix::reshape_backward", "matrix::reshape"), original + "\nlet extra = matrix::mul(input, input)?;", original + "\nlet extra = input.reshape([])?;"):
			with self.subTest(text=text), mock.patch.object(Path, "read_text", return_value=text):
				with self.assertRaises(GENERATOR.SchemaError):
					GENERATOR.matrix_autograd_recipes("view", rows)

	def test_reshape_does_not_invent_kernel_or_registry_identity(self):
		for change in [{"stable_id": 900}, {"kernel_id": "Reshape"}, {"contract": {}}, {"metadata_only": False}]:
			with self.subTest(change=change):
				schema = copy.deepcopy(self.schemas["view"])
				schema["operations"][0].update(change)
				with self.assertRaises(GENERATOR.SchemaError):
					GENERATOR.validate_view_schema(schema)

	def test_cli_rejects_recipe_mismatch_before_publication(self):
		schema = copy.deepcopy(self.schemas["view"])
		schema["operations"][0]["autograd"]["saved_fields"].append(["extra", "usize"])
		with tempfile.TemporaryDirectory() as directory:
			root = Path(directory)
			path = root / "view.json"
			path.write_text(json.dumps(schema))
			preview = root / "preview"
			result = subprocess.run(
				[sys.executable, str(GENERATOR_ROOT / "generate.py"), "--view-schema", str(path), "--output-dir", str(preview)],
				capture_output=True, text=True, timeout=60,
			)
			self.assertEqual(result.returncode, 1)
			self.assertIn("saved fields differ from recipe", result.stderr)
			self.assertNotIn("Traceback", result.stderr)
			self.assertFalse(preview.exists())
