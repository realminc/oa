from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import oafnautogen


class OaFnAutogenRegressionTest(unittest.TestCase):
	def test_loss_declaration_stays_in_private_generated_tree(self) -> None:
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
			self.assertFalse((out / "Public/Oa/Ml/FnLoss").exists())

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

if __name__ == "__main__":
	unittest.main()
