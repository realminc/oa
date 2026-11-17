from __future__ import annotations

import copy
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(ROOT / "tool/gen/fn"))

from oafn.effects import FUSION_EFFECTS, kernel_fusion_effects, rust_fusion_effects
from oafn.registry import generate_kernel_manifest, registry_entries
from oafn.schema import load_schemas
from oafn.util import SchemaError


class FusionEffectsTests(unittest.TestCase):
	@classmethod
	def setUpClass(cls):
		cls.schemas = load_schemas(ROOT)
		cls.entries = registry_entries(cls.schemas)
		cls.by_id = {entry["kernel_id"]: entry for entry in cls.entries}

	def test_replay_safety_and_exclusive_writes_do_not_imply_purity(self):
		self.assertEqual(kernel_fusion_effects({"training_replay_role": "safe"}), ["unknown"])
		record = {"physical_write": {"writes": [{"collision": "exclusive"}]}}
		self.assertEqual(kernel_fusion_effects(record), ["unknown"])
		self.assertEqual(kernel_fusion_effects({"fusion_effects": []}), [])

	def test_existing_replay_mutation_and_atomic_facts_are_preserved(self):
		cases = [
			({"training_replay_role": "frozen_rng"}, "random"),
			({"training_replay_role": "replay_rng"}, "random"),
			({"training_replay_role": "rng_state_advance"}, "state_mutation"),
			({"training_replay_role": "host_stepped_optimizer"}, "state_mutation"),
			({"training_replay_role": "optimizer_state_advance"}, "state_mutation"),
			({"training_replay_role": "optimizer_state_update"}, "state_mutation"),
			({"contract": {"mutated_inputs": [0]}}, "state_mutation"),
			({"physical_write": {"writes": [{"collision": "atomic_u32"}]}}, "atomic"),
			({"physical_write": {"uniform_stores": {"partition": "shared_atomic_contributors"}}}, "atomic"),
		]
		for record, expected in cases:
			with self.subTest(record=record):
				self.assertEqual(kernel_fusion_effects(record), ["unknown", expected])
				with self.assertRaisesRegex(SchemaError, "declares purity"):
					kernel_fusion_effects({**record, "fusion_effects": []})

	def test_effects_combine_in_stable_order_without_erasing_evidence(self):
		record = {
			"fusion_effects": ["execution_boundary", "host"],
			"training_replay_role": "replay_rng",
			"contract": {"mutated_inputs": [0]},
			"physical_write": {"writes": [{"collision": "atomic_u32"}]},
		}
		expected = ["random", "state_mutation", "atomic", "host", "execution_boundary"]
		self.assertEqual(kernel_fusion_effects(record), expected)
		record["fusion_effects"].reverse()
		self.assertEqual(kernel_fusion_effects(record), expected)
		for effect, rust_name in FUSION_EFFECTS.items():
			self.assertEqual(rust_fusion_effects([effect]), f"KernelEffects::{rust_name}")
		self.assertEqual(rust_fusion_effects([]), "KernelEffects::empty()")

	def test_variants_require_independent_audits_and_keep_parent_effects(self):
		self.assertEqual(kernel_fusion_effects({}, inherited={"fusion_effects": []}), ["unknown"])
		self.assertEqual(kernel_fusion_effects({"fusion_effects": []}, inherited={}), [])
		for parent in [
			{"training_replay_role": "replay_rng"},
			{"contract": {"mutated_inputs": [0]}},
			{"fusion_effects": ["host"]},
		]:
			with self.subTest(parent=parent):
				with self.assertRaisesRegex(SchemaError, "declares purity"):
					kernel_fusion_effects({"fusion_effects": []}, inherited=parent)

	def test_invalid_declarations_are_schema_errors(self):
		for invalid in [None, "pure", {}, ["pure"], ["random", "random"], [1], [[]]]:
			with self.subTest(invalid=invalid):
				with self.assertRaisesRegex(SchemaError, "fusion_effects"):
					kernel_fusion_effects({"fusion_effects": invalid})

	def test_every_registered_kernel_has_a_classification_and_generated_arm(self):
		manifest = generate_kernel_manifest(self.schemas)
		method = manifest.split("const fn fusion_effects(self)", 1)[1].split("fn linear_workgroups", 1)[0]
		for entry in self.entries:
			with self.subTest(kernel=entry["kernel_id"]):
				self.assertIn("fusion_effects", entry)
				self.assertIn(
					f"Self::{entry['kernel_id']} => {rust_fusion_effects(entry['fusion_effects'])},", method
				)
		self.assertEqual(method.count("Self::"), len(self.entries))
		for kernel in ["MatrixAddF32", "MatrixMulBroadcastF32", "MlLinearF32", "MlGroupedBackwardPrologueF32"]:
			self.assertEqual(self.by_id[kernel]["fusion_effects"], [])
		for entry in self.entries:
			if entry["training_replay_role"] != "safe":
				self.assertTrue(entry["fusion_effects"], entry["kernel_id"])

	def test_dtype_and_lowering_variants_do_not_inherit_base_purity(self):
		self.assertEqual(self.by_id["MatrixAddI32"]["fusion_effects"], ["unknown"])
		self.assertEqual(self.by_id["MatrixAddBroadcastF32"]["fusion_effects"], ["unknown"])
		self.assertIn("random", self.by_id["MatrixPhiloxUniformF32"]["fusion_effects"])
		self.assertIn("state_mutation", self.by_id["MlAdamWGraphF32"]["fusion_effects"])
		self.assertIn("atomic", self.by_id["VisionConfusionMatrixI32"]["fusion_effects"])

	def test_registry_rejects_invalid_audit_before_publication(self):
		schemas = copy.deepcopy(self.schemas)
		schemas["elemwise"][0]["operations"][0]["fusion_effects"] = ["magic"]
		with self.assertRaisesRegex(SchemaError, "fusion_effects"):
			registry_entries(schemas)
		schemas = copy.deepcopy(self.schemas)
		next(op for op in schemas["ml"][0]["operations"] if op["name"] == "adamw")["fusion_effects"] = []
		with self.assertRaisesRegex(SchemaError, "declares purity"):
			registry_entries(schemas)
