"""Complete-body crypto generation, route ownership and wire contract proofs."""
import copy
import importlib.util
import json
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[5]
SPEC = importlib.util.spec_from_file_location("oa_crypto_generate", ROOT / "tool/gen/fn/generate.py")
GENERATOR = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(GENERATOR)


class CryptographyGenerationTests(unittest.TestCase):
	def setUp(self):
		self.schemas = {family: json.loads((ROOT / f"tool/gen/fn/schema/cryptography/cryptography_{family}.json").read_text()) for family in ("hash", "pqc")}

	def render(self, family):
		return GENERATOR.generate_cryptography_api(self.schemas[family], "0" * 64, "hash" if family == "hash" else "verify")

	def test_complete_bodies_have_one_owner(self):
		for family, category in (("hash", "hash"), ("pqc", "verify")):
			GENERATOR.validate_cryptography_api(self.schemas[family], category)
			text = self.render(family)
			self.assertIn("DO NOT EDIT", text)
			self.assertNotIn("{{", text)
			self.assertNotIn("lowering::", text)
			self.assertFalse((ROOT / f"src/rs/cryptography/{family}/lowering.rs").exists())
			for api in self.schemas[family]["rust_api"]:
				self.assertEqual(text.count(f"fn {api['name']}("), 1)
		self.assertEqual(self.render("pqc").count("engine.record_semantic("), 3)
		self.assertIn('mod host;', (ROOT / "src/rs/cryptography/pqc.rs").read_text())

	def test_map_order_does_not_change_abi(self):
		for family, category in (("hash", "hash"), ("pqc", "verify")):
			before = self.render(family)
			for api in self.schemas[family]["rust_api"]:
				api["attributes"] = dict(reversed(list(api["attributes"].items())))
				for stage in api["stages"]:
					for key in ("bindings", "push_constants"):
						stage[key] = dict(reversed(list(stage[key].items())))
			GENERATOR.validate_cryptography_api(self.schemas[family], category)
			self.assertEqual(before, self.render(family))

	def test_signatures_docs_and_shake_configuration_are_schema_owned(self):
		api = self.schemas["hash"]["rust_api"][0]
		api["doc"] = ["Updated schema summary."]
		api["values"]["default_output_length"] = "24"
		self.assertIn("/// Updated schema summary.", self.render("hash"))
		self.assertIn("let default_output_length = 24;", self.render("hash"))

	def test_all_parameter_sets_and_modes_are_recorded(self):
		text = self.render("pqc")
		for api in self.schemas["pqc"]["rust_api"]:
			for contract in api["contracts"]:
				self.assertEqual(text.count(f"::{contract.upper()},"), 1)
		self.assertEqual(text.count("mldsa_prehash_bytes(algorithm)"), 2)
		self.assertEqual(text.count("if batch_size == 0"), 3)
		self.assertNotIn("algorithm.unwrap_or", text)
		self.assertNotIn("attributes[..", text)

	def test_merkle_retains_ordered_multilevel_recording_and_producer_allocation(self):
		text = self.render("hash")
		self.assertIn("record_split_semantic", text)
		self.assertIn("source = if index == 0", text)
		self.assertIn("&levels[index - 1]", text)
		self.assertIn("Matrix::allocate_output", text)
		self.assertNotIn("from_slice", text)
		self.assertNotRegex(text, r"\.wait\(|\.read\(")

	def test_bad_wire_and_signature_metadata_is_rejected(self):
		for mutation in (
			lambda api: api["stages"][0]["bindings"].pop("output_index"),
			lambda api: api.update(attributes=None),
			lambda api: api.update(visibility=[]),
			lambda api: api.update(contracts=[[]]),
			lambda api: api["stages"][0].update(kernel=[]),
			lambda api: api["stages"][0]["push_constants"].pop("count"),
			lambda api: api["parameters"][0].__setitem__(1, "&Image"),
			lambda api: api["attributes"].clear(),
			lambda api: api["stages"][0].update(kernel="CryptographyKeccakF1600U8"),
		):
			with self.subTest(mutation=mutation):
				schema = copy.deepcopy(self.schemas["hash"])
				mutation(schema["rust_api"][0])
				with self.assertRaises(GENERATOR.SchemaError):
					GENERATOR.validate_cryptography_api(schema, "hash")

	def test_missing_duplicate_or_mismatched_verifier_route_is_rejected(self):
		for mutation in (
			lambda api: api["stages"].pop(),
			lambda api: api["parameters"].pop(),
			lambda api: api["contracts"].pop(),
			lambda api: api["stages"][1]["push_constants"].update(batch_size="0"),
			lambda api: api["contracts"].__setitem__(1, api["contracts"][0]),
		):
			with self.subTest(mutation=mutation):
				schema = copy.deepcopy(self.schemas["pqc"])
				mutation(schema["rust_api"][0])
				with self.assertRaises(GENERATOR.SchemaError):
					GENERATOR.validate_cryptography_api(schema, "verify")


if __name__ == "__main__":
	unittest.main()
