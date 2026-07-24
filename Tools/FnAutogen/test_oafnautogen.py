from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import oafnautogen


class OaFnAutogenRegressionTest(unittest.TestCase):
	def test_loss_category_fragment_stays_private(self) -> None:
		with tempfile.TemporaryDirectory() as temp_dir:
			root = Path(temp_dir)
			schema_dir = root / "Ml"
			schema_dir.mkdir()
			schema = schema_dir / "MlFnLoss.toml"
			schema.write_text(
				"""
namespace = "OaFnLoss"
category = "Loss"
file_category = ""
file_prefix = "FnLoss"
cpp_subdir = "FnLoss"
[[ops]]
name = "Example"
kind = "binary"
body = "manual_context"
generate_forwarder = true
api_return = "OaMatrix"
api_params = ["const OaMatrix& InA", "const OaMatrix& InB"]
""".lstrip()
			)
			out = root / "Source"
			layouts = oafnautogen.process_schema(
				schema, set(), out, live=False, dry_run=False
			)
			self.assertIsInstance(layouts, list)
			layout = layouts[0]
			self.assertEqual(
				layout.header_path,
				out / "Private/Oa/Ml/FnLoss/Example/FnLossExample.gen.h",
			)
			self.assertEqual(
				layout.cpp_path,
				out / "Private/Oa/Ml/FnLoss/Example/FnLossExample.gen.cpp",
			)
			self.assertTrue(layout.header_path.is_file())
			self.assertTrue(layout.cpp_path.is_file())
			self.assertFalse((out / "Public/Oa/Ml/FnLoss/Example").exists())
			oafnautogen.write_manifest_files(layouts, out, dry_run=False)
			umbrella_fragment = out / "Private/Oa/Ml/FnLoss/FnLoss.gen.h"
			self.assertTrue(umbrella_fragment.is_file())
			self.assertIn(
				'#include "Example/FnLossExample.gen.h"',
				umbrella_fragment.read_text(encoding="utf-8"),
			)
			self.assertFalse((out / "Public/Oa/Ml/FnLoss.gen.h").exists())

	def test_manual_autograd_accepts_saved_only_tensor_and_lowercase_out(self) -> None:
		emitted = "\n".join(oafnautogen._emit_manual_autograd_attach({
			"name": "Example",
			"autograd": {
				"inputs": ["InLogits"],
				"saved": ["InLogits", "InTargets", "out"],
				"grad_class": "OaGradExample",
				"attach": "standard",
			},
		}))
		self.assertIn("const OaMatrix& InTargets", emitted)
		self.assertNotIn("const OaMatrix& out", emitted)
		self.assertIn(
			"GradFn->Saved_ = OaVec<OaMatrix>{InLogits, InTargets, Out};",
			emitted,
		)
		self.assertIn(
			"GradFn->SetGraphInputs(OaVec<OaMatrix>{InLogits});",
			emitted,
		)

	def test_python_keyword_arguments_require_pascal_case(self) -> None:
		op = {
			"name": "Example",
			"kind": "binary",
			"body": "manual_context",
			"api_return": "OaMatrix",
			"api_params": [
				"const OaMatrix& InInput",
				"OaI32 InBatchSize",
			],
			"python": {
				"args": ["Input", "BatchSize"],
			},
		}
		oafnautogen.validate_schema(
			Path("CoreFnMatrixExample.toml"), [op], set()
		)
		op["python"]["args"] = ["input", "batch_size"]
		with self.assertRaises(SystemExit):
			oafnautogen.validate_schema(
				Path("CoreFnMatrixExample.toml"), [op], set()
			)

	def test_session_contract_honors_frozen_eight_attribute_descriptor(self) -> None:
		op = {
			"name": "CartPoleStep",
			"kind": "session_command",
			"body": "manual_context",
			"generate_declaration": False,
			"kernel_forward": "RlCartPoleStep",
			"kernel": {
				"id_prefix": "Ml",
				"id_local": 275,
				"category": "Ml",
				"origin": "oa",
				"source": "Rl/Environment/RlCartPoleStep",
			},
			"contract": {
				"value_validation": "session_command",
				"input_kinds": ["matrix"] * 4,
				"output_kinds": ["matrix"] * 7,
				"shape_rule": "explicit",
				"dtype_rule": "match_input",
				"effects": ["read_inputs", "write_outputs"],
				"mutated_inputs": [1, 2, 3],
				"output_alias_inputs": [-1, 1, -1, -1, -1, 2, 3],
				"differentiation": "none",
				"lowering": "dispatch",
				"control_flow": "conditional",
				"attributes": [
					{"name": f"Attribute{index}", "kind": "float"}
					for index in range(8)
				],
			},
		}
		oafnautogen.validate_schema(
			Path("MlFnRlEnvironment.toml"), [op], {"RlCartPoleStep"}
		)
		emitted = oafnautogen.emit_operation_registry(
			[("OaFnRlEnvironment", op)]
		)
		self.assertIn("OaOperationDtypeRule::MatchInput", emitted)
		self.assertIn(".AttributeCount = 8U", emitted)
		op["contract"]["attributes"].append(
			{"name": "Attribute8", "kind": "float"}
		)
		with self.assertRaises(SystemExit):
			oafnautogen.validate_schema(
				Path("MlFnRlEnvironment.toml"), [op], {"RlCartPoleStep"}
			)
		op["contract"]["attributes"].pop()
		op["kind"] = "unary"
		with self.assertRaises(SystemExit):
			oafnautogen.validate_schema(
				Path("MlFnRlEnvironment.toml"), [op], {"RlCartPoleStep"}
			)

	def test_schema_kernel_owns_registry_row_and_parallel_build_pair(self) -> None:
		kernels = [{
			"name": "RlCartPoleStep",
			"id_prefix": "Ml",
			"id_local": 275,
			"category": "Ml",
			"origin": "oa",
			"source": "Rl/Environment/RlCartPoleStep",
		}]
		registry = oafnautogen.emit_schema_kernel_registry(kernels, "Ml")
		ids = oafnautogen.emit_schema_kernel_ids(kernels)
		cmake = oafnautogen.emit_schema_kernel_cmake(kernels)
		self.assertIn('"RlCartPoleStep"', registry)
		self.assertIn("OaComputeKernelPrefix::Ml, 275", registry)
		self.assertIn("OaKernelId RlCartPoleStep", ids)
		self.assertIn("OaComputeKernelPrefix::Ml, 275", ids)
		self.assertIn("OA_FN_AUTOGEN_ML_KERNEL_NAMES", cmake)
		self.assertIn("Rl/Environment/RlCartPoleStep", cmake)

	def test_operation_reference_distinguishes_handwritten_and_session_commands(self) -> None:
		contract = {
			"input_kinds": ["matrix", "matrix"],
			"output_kinds": ["matrix"],
			"shape_rule": "explicit",
			"dtype_rule": "match_input",
			"effects": ["read_inputs", "write_outputs"],
			"mutated_inputs": [0],
			"output_alias_inputs": [0],
			"differentiation": "none",
			"lowering": "dispatch",
			"control_flow": "straight_line",
		}
		handwritten = {
			"name": "AddInPlace",
			"kind": "binary",
			"generate_declaration": False,
			"api_return": "void",
			"api_params": [
				"OaMatrix& InSelf",
				"const OaMatrix& InOther",
			],
			"contract": contract,
		}
		session = {
			"name": "CartPoleStep",
			"kind": "session_command",
			"generate_declaration": False,
			"contract": contract,
		}
		reference = oafnautogen.emit_operation_reference([
			("OaFnMatrix", handwritten),
			("OaFnRlEnvironment", session),
		])
		self.assertIn(
			"- C++: `void AddInPlace(OaMatrix& InSelf, const OaMatrix& InOther)`",
			reference,
		)
		self.assertIn("- C++: internal session command", reference)

if __name__ == "__main__":
	unittest.main()
