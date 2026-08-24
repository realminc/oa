from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import oafnautogen
import check_schema_coverage


class FnAutogenRegressionTest(unittest.TestCase):
	def test_schema_owners_partition_core_and_sdk_authorities(self) -> None:
		with tempfile.TemporaryDirectory() as temp_dir:
			root = Path(temp_dir)
			core = root / "core.toml"
			sdk = root / "sdk.toml"
			invalid = root / "invalid.toml"
			core.write_text('namespace = "oa::FnMatrix"\n', encoding="utf-8")
			sdk.write_text(
				'owner = "sdk"\nnamespace = "oa::FnEnvironment"\n',
				encoding="utf-8",
			)
			invalid.write_text(
				'owner = "plugin"\nnamespace = "oa::FnMatrix"\n',
				encoding="utf-8",
			)
			self.assertEqual(
				oafnautogen.partition_schema_owners([core, sdk]),
				([core], [sdk]),
			)
			with self.assertRaises(SystemExit):
				oafnautogen.partition_schema_owners([invalid])

	def test_sdk_kernel_manifest_has_no_core_registry_surface(self) -> None:
		kernel = {
			"name": "RlCartPoleStep",
			"source": "environment/rlCartPoleStep",
		}
		emitted = oafnautogen.emit_sdk_kernel_cmake([kernel])
		self.assertIn("OA_SDK_ENVIRONMENT_KERNEL_NAMES", emitted)
		self.assertIn("RlCartPoleStep", emitted)
		self.assertIn("environment/rlCartPoleStep", emitted)
		self.assertNotIn("OA_FN_AUTOGEN_ML_KERNEL_NAMES", emitted)

	def test_compressed_fn_namespace_declarations_are_rejected(self) -> None:
		with tempfile.TemporaryDirectory() as temp_dir:
			root = Path(temp_dir)
			header = root / "source/cpp/include/oa/audio/fnAudio.h"
			header.parent.mkdir(parents=True)
			header.write_text(
				"namespace oa::FnAudio {\n}\n",
				encoding="utf-8",
			)
			self.assertEqual(
				check_schema_coverage.compressed_fn_namespace_declarations(root),
				["source/cpp/include/oa/audio/fnAudio.h:1"],
			)
			header.write_text(
				"namespace oa {\n\nnamespace FnAudio {\n}\n\n}\n",
				encoding="utf-8",
			)
			self.assertEqual(
				check_schema_coverage.compressed_fn_namespace_declarations(root),
				[],
			)

	def test_header_fragment_remains_namespace_neutral(self) -> None:
		emitted = oafnautogen.emit_header_fragment(
			[{
				"name": "normalize",
				"api_return": "oa::Audio",
				"api_params": ["const oa::Audio& inAudio"],
			}],
			"audio/audioFnAudioSignal.toml",
			"Signal",
			"oa::FnAudio",
		)
		self.assertIn("// oa::FnAudio — Signal overloads.", emitted)
		self.assertNotIn("namespace oa::FnAudio", emitted)
		self.assertNotIn("namespace FnAudio", emitted)

	def test_python_surface_inventory_is_owned_by_schema_namespace(self) -> None:
		emitted = oafnautogen.emit_python_surface_inventory([
			("Core", "oa::FnMatrix", {
				"name": "add",
				"python": {"name": "add"},
			}),
			("Ml", "oa::FnMatrix", {
				"name": "relu",
				"python": {"name": "relu"},
			}),
			("Audio", "oa::FnAudio", {
				"name": "normalize",
				"python": {"name": "normalize"},
			}),
			("Vision", "oa::FnDetection", {
				"name": "evaluate",
				"python": {"name": "evaluate"},
			}),
		])
		self.assertIn('"FnMatrix": {', emitted)
		self.assertIn('"core": (', emitted)
		self.assertIn('("add", "add"),', emitted)
		self.assertIn('"ml": (', emitted)
		self.assertIn('("relu", "relu"),', emitted)
		self.assertIn('"FnAudio": {', emitted)
		self.assertIn('("evaluate", "evaluate"),', emitted)
		self.assertLess(emitted.index('"FnAudio"'), emitted.index('"FnMatrix"'))

	def test_loss_category_fragment_stays_private(self) -> None:
		with tempfile.TemporaryDirectory() as temp_dir:
			root = Path(temp_dir)
			schema_dir = root / "Ml"
			schema_dir.mkdir()
			schema = schema_dir / "MlFnLoss.toml"
			schema.write_text(
				"""
namespace = "oa::FnLoss"
category = "Loss"
file_category = ""
file_prefix = "FnLoss"
cpp_subdir = "FnLoss"
surface = "public_operation"
[[ops]]
name = "example"
kind = "binary"
body = "manual_session"
generate_forwarder = true
api_return = "oa::Matrix"
api_params = ["const oa::Matrix& inA", "const oa::Matrix& inB"]
[ops.contract]
input_kinds = ["matrix", "matrix"]
output_kinds = ["matrix"]
shape_rule = "match_input"
dtype_rule = "match_input"
effects = ["read_inputs", "write_outputs"]
mutated_inputs = []
output_alias_inputs = [-1]
control_flow = "straight_line"
differentiation = "none"
lowering = "dispatch"
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
					out / "cpp/lib/oa/ml/fnloss/example/fnLossExample.gen.h",
			)
			self.assertEqual(
				layout.cpp_path,
					out / "cpp/lib/oa/ml/fnloss/example/fnLossExample.gen.cpp",
			)
			self.assertTrue(layout.header_path.is_file())
			self.assertTrue(layout.cpp_path.is_file())
			self.assertFalse((out / "cpp/include/oa/ml/fnloss/example").exists())
			oafnautogen.write_manifest_files(layouts, out, dry_run=False)
			umbrella_fragment = out / "cpp/lib/oa/ml/fnloss/fnLoss.gen.h"
			self.assertTrue(umbrella_fragment.is_file())
			self.assertIn(
				'#include "example/fnLossExample.gen.h"',
				umbrella_fragment.read_text(encoding="utf-8"),
			)
			self.assertFalse((out / "cpp/include/oa/ml/fnLoss.gen.h").exists())

	def test_manual_autograd_accepts_saved_only_tensor_and_lowercase_out(self) -> None:
		emitted = "\n".join(oafnautogen._emit_manual_autograd_attach_definition({
			"name": "example",
			"autograd": {
				"inputs": ["inLogits"],
				"saved": ["inLogits", "inTargets", "out"],
				"grad_class": "oa::GradExample",
				"attach": "standard",
			},
		}))
		self.assertIn("const oa::Matrix& inTargets", emitted)
		self.assertNotIn("const oa::Matrix& out", emitted)
		self.assertIn(
			"gradFn->saveForBackward({inLogits, inTargets, out});",
			emitted,
		)
		self.assertIn(
			"gradFn->setGraphInputs(oa::Vec<oa::Matrix>{inLogits});",
			emitted,
		)

	def test_core_autograd_requires_exact_family_header(self) -> None:
		op = {
			"name": "copy",
			"kind": "unary",
			"kernel_forward": "Copy",
			"forward_op": "a",
			"autograd": {
				"grad_class": "oa::GradCopy",
				"inputs": ["inA"],
				"saved": [],
				"formula": "manual:Copy",
				"attach": "standard",
			},
		}
		with self.assertRaises(SystemExit):
			oafnautogen.emit_cpp_file(
				[op], "core/coreFnMatrixShape.toml", "Shape", "core",
			)

	def test_ml_autograd_requires_exact_family_header(self) -> None:
		op = {
			"name": "example",
			"kind": "unary",
			"kernel_forward": "Example",
			"forward_op": "a",
			"autograd": {
				"grad_class": "oa::GradExample",
				"inputs": ["inA"],
				"saved": [],
				"formula": "manual:Example",
				"attach": "standard",
			},
		}
		with self.assertRaises(SystemExit):
			oafnautogen.emit_cpp_file(
				[op], "ml/mlFnMatrixExample.toml", "Example", "ml",
			)

	def test_core_autograd_header_stays_in_core_and_has_no_ml_include(self) -> None:
		op = {
			"name": "add",
			"autograd": {
				"grad_class": "oa::GradAdd",
				"formula": "auto_binary_add",
			},
		}
		emitted = oafnautogen.emit_autograd_header(
			[op],
			"core/coreFnMatrixElemwise.toml",
			"Elemwise",
			"core",
		)
		self.assertIn(
			"source/cpp/lib/oa/core/autograd/matrix/"
			"autogradElemwise.gen.h",
			emitted,
		)
		self.assertIn("#include <oa/core/autograd.h>", emitted)
		self.assertNotIn("<oa/ml/", emitted)

	def test_autograd_attachment_and_manifest_are_domain_owned(self) -> None:
		op = {
			"name": "example",
			"autograd": {
				"grad_class": "oa::GradExample",
				"inputs": ["inA"],
				"saved": [],
				"attach": "standard",
				"node_header":
					"oa/core/autograd/matrix/autogradBlas.h",
			},
		}
		core_attach = oafnautogen.emit_manual_autograd_attach_header(
			[("oa::FnMatrix", op)], "core"
		)
		self.assertIn(
			"concrete nodes are private to the", core_attach
		)
		self.assertIn(
			"return detail::FnMatrix::exampleEnabled", core_attach
		)
		self.assertNotIn("oa::GradExample", core_attach)
		core_attach_cpp = oafnautogen.emit_manual_autograd_attach_cpp(
			[("oa::FnMatrix", op)], "core"
		)
		self.assertIn(
			"#include <oa/core/autograd/matrix/autogradBlas.h>",
			core_attach_cpp,
		)
		self.assertIn("oa::GradExample", core_attach_cpp)
		self.assertNotIn("/autograd/nodes.h>", core_attach_cpp)
		narrow_op = {
			"name": "narrowExample",
			"autograd": {
				"grad_class": "oa::GradNarrowExample",
				"inputs": ["inA"],
				"saved": [],
				"attach": "standard",
				"node_header":
					"oa/core/autograd/matrix/autogradShape.h",
			},
		}
		narrow_attach_cpp = oafnautogen.emit_manual_autograd_attach_cpp(
			[("oa::FnMatrix", narrow_op)], "core"
		)
		self.assertIn(
			"#include <oa/core/autograd/matrix/autogradShape.h>",
			narrow_attach_cpp,
		)
		self.assertNotIn(
			"#include <oa/core/autograd/nodes.h>", narrow_attach_cpp
		)

		with tempfile.TemporaryDirectory() as temp_dir:
			root = Path(temp_dir)
			layout = oafnautogen.SchemaLayout(
				domain="core",
				namespace="oa::FnMatrix",
				file_prefix="FnMatrix",
				cpp_subdir="FnMatrix",
				header_path=root / "cpp/lib/oa/core/fnmatrix/x.gen.h",
				cpp_path=root / "cpp/lib/oa/core/fnmatrix/x.gen.cpp",
				autograd_header_path=(
					root
					/ "cpp/lib/oa/core/autograd/matrix/autogradX.gen.h"
				),
				test_path=root / "Test.cpp",
				emit_autograd=True,
			)
			oafnautogen.write_autograd_manifest_files(
				[layout], root, dry_run=False
			)
			manifest = (
				root / "cpp/lib/oa/core/autograd/autograd.gen.h"
			).read_text(encoding="utf-8")
			self.assertIn('<oa/core/autograd.h>', manifest)
			self.assertIn('"matrix/autogradX.gen.h"', manifest)
			self.assertNotIn("<oa/ml/", manifest)

	def test_python_keyword_arguments_require_camel_case(self) -> None:
		op = {
			"name": "example",
			"kind": "binary",
			"surface": "public_operation",
			"body": "manual_session",
			"api_return": "oa::Matrix",
			"api_params": [
				"const oa::Matrix& inInput",
				"oa::I32 inBatchSize",
			],
			"python": {
				"args": ["input", "batchSize"],
			},
			"contract": {
				"input_kinds": ["matrix", "matrix"],
				"output_kinds": ["matrix"],
				"shape_rule": "match_input",
				"dtype_rule": "match_input",
				"effects": ["read_inputs", "write_outputs"],
				"mutated_inputs": [],
				"output_alias_inputs": [-1],
				"control_flow": "straight_line",
				"differentiation": "none",
				"lowering": "dispatch",
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

	def test_schema_level_python_generation_derives_uniform_signature(self) -> None:
		data = {
			"surface": "public_operation",
			"generate_python_bindings": True,
		}
		op = {
			"name": "pow",
			"kind": "unary_scalar",
			"surface": "public_operation",
			"kernel_forward": "Pow",
			"scalar_param": {
				"name": "exponent",
				"type": "oa::F32",
				"push_field": "Exponent",
			},
		}
		oafnautogen.apply_schema_defaults(data, op)
		self.assertEqual(
			op["api_params"],
			["const oa::Matrix& inA", "oa::F32 inExponent"],
		)
		self.assertEqual(op["python"]["args"], ["a", "exponent"])
		emitted = "\n".join(
			oafnautogen.emit_python_binding("oa::FnMatrix", op)
		)
		self.assertIn(
			"matrixPtr(oa::FnMatrix::pow(inA, inExponent))",
			emitted,
		)

	def test_schema_level_autograd_header_is_default_not_override(self) -> None:
		data = {
			"autograd_node_header":
				"oa/core/autograd/matrix/autogradElemwise.h",
		}
		defaulted = {
			"name": "neg",
			"autograd": {"grad_class": "oa::GradNeg"},
		}
		overridden = {
			"name": "copy",
			"autograd": {
				"grad_class": "oa::GradCopy",
					"node_header": "oa/core/autograd/matrix/autogradShape.h",
			},
		}
		oafnautogen.apply_schema_defaults(data, defaulted)
		oafnautogen.apply_schema_defaults(data, overridden)
		self.assertEqual(
			defaulted["autograd"]["node_header"],
			"oa/core/autograd/matrix/autogradElemwise.h",
		)
		self.assertEqual(
			overridden["autograd"]["node_header"],
			"oa/core/autograd/matrix/autogradShape.h",
		)

	def test_generated_audio_binding_owns_semantic_result(self) -> None:
		op = {
			"name": "gain",
			"kind": "unary",
			"surface": "public_operation",
			"api_return": "oa::Audio",
			"api_params": [
				"const oa::Audio& inAudio",
				"oa::F32 inGainDb",
			],
			"python": {"args": ["audio", "gainDb"]},
		}
		oafnautogen.validate_python_binding("AudioFnAudioSignal.toml:Gain", op)
		emitted = "\n".join(
			oafnautogen.emit_python_binding("oa::FnAudio", op)
		)
		self.assertIn(
			"return new oa::Audio(oa::FnAudio::gain(inAudio, inGainDb));",
			emitted,
		)

	def test_generated_const_span_binding_uses_python_vector_boundary(self) -> None:
		op = {
			"name": "sosFilter",
			"kind": "unary",
			"surface": "public_operation",
			"api_return": "oa::Audio",
			"api_params": [
				"const oa::Audio& inAudio",
				"oa::Span<const oa::BiquadCoefficients> inSections",
			],
			"python": {"args": ["audio", "sections"]},
		}
		oafnautogen.validate_python_binding(
			"AudioFnAudioSignal.toml:SosFilter", op
		)
		emitted = "\n".join(
			oafnautogen.emit_python_binding("oa::FnAudio", op)
		)
		self.assertIn(
			"std::vector<oa::BiquadCoefficients> inSections", emitted
		)
		self.assertIn(
			"oa::Span<const oa::BiquadCoefficients>(inSections.data(), inSections.size())",
			emitted,
		)

	def test_manual_python_binding_owns_surface_without_generated_body(self) -> None:
		op = {
			"name": "decodeFile",
			"kind": "unary",
			"surface": "host_utility",
			"api_return": "oa::Result<oa::Image>",
			"api_params": [
				"const oa::Path& inPath",
				"oa::ImageFormat inFormat = oa::ImageFormat::Rgb",
			],
			"python": {
				"binding": "manual",
				"args": ["path", "format"],
			},
		}
		oafnautogen.validate_python_binding(
			"VisionFnImageCodec.toml:DecodeFile", op
		)
		emitted = oafnautogen.emit_python_bindings([("oa::FnImage", op)])
		self.assertNotIn("decodeFile", emitted)
		surface = oafnautogen.emit_python_surface_inventory([
			("Vision", "oa::FnImage", op)
		])
		self.assertIn('(\"decodeFile\", \"decodeFile\")', surface)

	def test_generated_void_binding_returns_python_none(self) -> None:
		op = {
			"name": "update",
			"kind": "binary",
			"surface": "public_operation",
			"api_return": "void",
			"api_params": [
				"const oa::Matrix& inSource",
				"oa::Matrix& inOutState",
			],
			"python": {"args": ["source", "outState"]},
		}
		oafnautogen.validate_python_binding(
			"MlFnMatrixVq.toml:Update", op
		)
		emitted = "\n".join(
			oafnautogen.emit_python_binding("oa::FnMatrix", op)
		)
		self.assertIn(
			"oa::FnMatrix::update(inSource, inOutState);",
			emitted,
		)
		self.assertNotIn(
			"return oa::FnMatrix::update",
			emitted,
		)

	def test_generated_structured_result_binding_is_schema_owned(self) -> None:
		data = {
			"surface": "public_operation",
			"generate_python_bindings": True,
		}
		op = {
			"name": "maxPool2d",
			"kind": "unary",
			"surface": "public_operation",
			"api_return": "oa::MaxPool2dResult",
			"api_params": ["const oa::Matrix& inX"],
			"python_result": {
				"cpp_type": "oa::MaxPool2dResult",
				"members": ["out", "indices"],
			},
		}
		oafnautogen.apply_schema_defaults(data, op)
		oafnautogen.validate_python_binding(
			"MlFnMatrixPool.toml:MaxPool2d", op
		)
		emitted = oafnautogen.emit_python_bindings([("oa::FnMatrix", op)])
		self.assertIn(
			'nb::class_<oa::MaxPool2dResult>'
			'(m, "MaxPool2dResult")',
			emitted,
		)
		self.assertIn(
			"new oa::MaxPool2dResult("
			"oa::FnMatrix::maxPool2d(inX))",
			emitted,
		)
		surface = oafnautogen.emit_python_surface_inventory(
			[("Ml", "oa::FnMatrix", op)]
		)
		self.assertIn(
			'SCHEMA_ROOT_EXPORTS = {\n'
			'\t"FnMatrix": (\n'
			'\t\t("MaxPool2dResult", "MaxPool2dResult"),',
			surface,
		)

	def test_generated_structured_result_can_reuse_manual_type_binding(self) -> None:
		op = {
			"name": "nms",
			"kind": "unary",
			"surface": "public_operation",
			"api_return": "oa::NmsResult",
			"api_params": ["const oa::Matrix& inBoxes"],
			"python": {"args": ["boxes"]},
			"python_result": {
				"cpp_type": "oa::NmsResult",
				"members": ["indices", "count"],
				"bind_type": False,
			},
		}
		oafnautogen.validate_python_binding(
			"VisionFnDetection.toml:Nms", op
		)
		emitted = oafnautogen.emit_python_bindings([("oa::FnDetection", op)])
		self.assertNotIn("nb::class_<oa::NmsResult>", emitted)
		self.assertIn(
			"new oa::NmsResult(oa::FnDetection::nms(inBoxes))",
			emitted,
		)

	def test_generated_python_result_validation_preserves_error_boundary(self) -> None:
		op = {
			"name": "normalize",
			"kind": "unary",
			"surface": "public_operation",
			"api_return": "oa::Matrix",
			"api_params": ["const oa::Matrix& inAdvantage"],
			"python": {
				"args": ["advantage"],
				"result_validation": "not_empty",
				"error": "NormalizeAdvantages rejected its input",
			},
		}
		oafnautogen.validate_python_binding("mlFnAdvantage.toml:normalize", op)
		emitted = "\n".join(
			oafnautogen.emit_python_binding("oa::FnAdvantage", op)
		)
		self.assertIn(
			"auto result = oa::FnAdvantage::normalize(inAdvantage);",
			emitted,
		)
		self.assertIn("if (result.isEmpty())", emitted)
		self.assertIn("matrixPtr(std::move(result))", emitted)

	def test_generated_python_default_is_typed(self) -> None:
		op = {
			"name": "stft",
			"kind": "unary",
			"surface": "public_operation",
			"api_return": "oa::Matrix",
			"api_params": [
				"const oa::Audio& inAudio",
				"const oa::StftConfig& inConfig = {}",
			],
			"python": {"args": ["audio", "config"]},
		}
		emitted = "\n".join(
			oafnautogen.emit_python_binding("oa::FnAudio", op)
		)
		self.assertIn(
			'nb::arg("config") = oa::StftConfig{}',
			emitted,
		)

	def test_manual_multi_output_autograd_requires_provenance(self) -> None:
		op = {
			"name": "residualRmsNorm",
			"kind": "binary",
			"surface": "public_operation",
			"body": "manual_session",
			"api_return": "oa::ResidualRmsNormResult",
			"api_params": [
				"const oa::Matrix& inA",
				"const oa::Matrix& inB",
				"const oa::Matrix& inWeight",
			],
			"contract": {
				"input_kinds": ["matrix", "matrix", "matrix"],
				"output_kinds": ["matrix", "matrix"],
				"shape_rule": "match_input",
				"dtype_rule": "match_input",
				"effects": ["read_inputs", "write_outputs"],
				"mutated_inputs": [],
				"output_alias_inputs": [-1, -1],
				"control_flow": "straight_line",
				"differentiation": "reverse",
				"lowering": "dispatch",
			},
			"autograd": {
				"formula": "manual",
				"attach": "manual",
				"inputs": ["inA", "inB", "inWeight"],
				"outputs": ["out", "Residual"],
				"implementation": "source/cpp/lib/oa/ml/fnmatrix/norm/fnMatrixNorm.cpp",
			},
		}
		oafnautogen.validate_schema(
			Path("MlFnMatrixNorm.toml"), [op], set()
		)
		del op["autograd"]["implementation"]
		with self.assertRaises(SystemExit):
			oafnautogen.validate_schema(
				Path("MlFnMatrixNorm.toml"), [op], set()
			)

	def test_session_contract_honors_frozen_eight_attribute_descriptor(self) -> None:
		op = {
			"name": "cartPoleStep",
			"kind": "session_command",
			"surface": "session_command",
			"body": "manual_session",
			"generate_declaration": False,
			"kernel_forward": "RlCartPoleStep",
			"session": {
				"owners": ["oa::Environment"],
				"transition": "Ready -> Recorded",
				"effects": ["read_state", "write_state", "device_work"],
				"completion": "recorded",
			},
			"kernel": {
				"id_prefix": "Ml",
				"id_local": 275,
				"category": "Ml",
				"origin": "oa",
				"source": "environment/rlCartPoleStep",
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
			Path("mlFnEnvironmentSession.toml"), [op], {"RlCartPoleStep"}
		)
		emitted = oafnautogen.emit_operation_registry(
			[("oa::FnEnvironment", op)]
		)
		self.assertIn("namespace FnEnvironment", emitted)
		self.assertIn(
			'.name = "oa::FnEnvironment::cartPoleStep"',
			emitted,
		)
		self.assertIn("oa::OpDtypeRule::MatchInput", emitted)
		self.assertIn(".attributeCount = 8U", emitted)
		op["contract"]["attributes"].append(
			{"name": "attribute8", "kind": "float"}
		)
		with self.assertRaises(SystemExit):
			oafnautogen.validate_schema(
				Path("mlFnEnvironmentSession.toml"), [op], {"RlCartPoleStep"}
			)
		op["contract"]["attributes"].pop()
		op["kind"] = "unary"
		with self.assertRaises(SystemExit):
			oafnautogen.validate_schema(
				Path("mlFnEnvironmentSession.toml"), [op], {"RlCartPoleStep"}
			)

	def test_schema_set_rejects_cross_file_duplicate_identity(self) -> None:
		with tempfile.TemporaryDirectory() as temp_dir:
			root = Path(temp_dir) / "Core"
			root.mkdir()
			first = root / "CoreFnMatrixFirst.toml"
			second = root / "CoreFnMatrixSecond.toml"
			first.write_text('[[ops]]\nname = "Example"\n', encoding="utf-8")
			second.write_text('[[ops]]\nname = "Example"\n', encoding="utf-8")
			with self.assertRaises(SystemExit):
				oafnautogen.validate_schema_set([first, second])

	def test_registry_namespaces_same_spelling_without_collision(self) -> None:
		contract = {
			"input_kinds": ["matrix"],
			"output_kinds": ["matrix"],
			"shape_rule": "match_input",
			"dtype_rule": "match_input",
			"effects": ["read_inputs", "write_outputs"],
			"mutated_inputs": [],
			"output_alias_inputs": [-1],
			"differentiation": "none",
			"lowering": "dispatch",
			"control_flow": "straight_line",
		}
		emitted = oafnautogen.emit_operation_registry([
			("oa::FnAudio", {"name": "normalize", "contract": contract}),
			("oa::FnImage", {"name": "normalize", "contract": contract}),
		])
		self.assertIn("namespace FnAudio", emitted)
		self.assertIn("namespace FnImage", emitted)
		self.assertIn('.name = "oa::FnAudio::normalize"', emitted)
		self.assertIn('.name = "oa::FnImage::normalize"', emitted)

	def test_nullary_contract_has_zero_inputs_and_one_output(self) -> None:
		op = {
			"name": "fill",
			"kind": "nullary_scalar",
			"surface": "public_operation",
			"kernel_forward": "Fill",
			"scalar_param": {
				"name": "value",
				"type": "oa::F32",
				"push_field": "Value",
			},
			"api_return": "oa::Matrix",
			"api_params": [
				"const oa::MatrixShape& inShape",
				"oa::F32 inValue",
			],
			"contract": {
				"input_kinds": [],
				"output_kinds": ["matrix"],
				"shape_rule": "explicit",
				"dtype_rule": "match_input",
				"effects": ["write_outputs"],
				"mutated_inputs": [],
				"output_alias_inputs": [-1],
				"differentiation": "none",
				"lowering": "dispatch",
				"control_flow": "straight_line",
				"attributes": [
					{"name": "shape", "kind": "shape", "source": "inShape"},
				],
			},
		}
		oafnautogen.validate_schema(
			Path("CoreFnMatrixRng.toml"), [op], {"Fill"}
		)
		emitted = oafnautogen.emit_operation_registry([
			("oa::FnMatrix", op),
		])
		self.assertIn(".inputCount = 0U", emitted)
		self.assertIn(".outputCount = 1U", emitted)
		self.assertIn(".attributeCount = 2U", emitted)
		body = oafnautogen.emit_session_body(op)
		self.assertIn(
			"oa::Matrix oa::FnMatrix::fill("
			"const oa::MatrixShape& inShape, oa::F32 inValue)",
			body,
		)
		self.assertIn(
			"oa::OpAttribute::fromShape(\"shape\", inShape)",
			body,
		)

	def test_void_declaration_does_not_emit_nodiscard(self) -> None:
		lines = oafnautogen.header_fragment_for_op({
			"name": "fillInPlace",
			"notes": "Mutating fill.",
			"kind": "unary_scalar",
			"body": "manual_session",
			"api_return": "void",
			"api_params": ["oa::Matrix& inSelf", "oa::F32 inValue"],
		})
		declaration = "\n".join(lines)
		self.assertIn("void fillInPlace(", declaration)
		self.assertNotIn("[[nodiscard]]", declaration)

	def test_manual_contract_forwarder_groups_composed_lowering(self) -> None:
		op = {
			"name": "gaussianNoise",
			"kind": "unary",
			"surface": "stable_composite",
			"body": "manual_session",
			"generate_forwarder": True,
			"api_return": "oa::Matrix",
			"api_params": [
				"const oa::Matrix& inImage",
				"oa::F32 inMean",
				"oa::F32 inStddev",
				"oa::U64 inSeed",
			],
			"contract": {
				"input_kinds": ["matrix"],
				"output_kinds": ["matrix"],
				"shape_rule": "match_input",
				"dtype_rule": "match_input",
				"effects": ["read_inputs", "write_outputs"],
				"mutated_inputs": [],
				"output_alias_inputs": [-1],
				"differentiation": "none",
				"lowering": "dispatch",
				"control_flow": "straight_line",
				"attributes": [
					{"name": "mean", "kind": "float", "source": "inMean"},
					{"name": "stddev", "kind": "float", "source": "inStddev"},
					{"name": "seed", "kind": "unsigned_integer", "source": "inSeed"},
				],
			},
		}
		body = oafnautogen.emit_manual_forwarder(op, "oa::FnImage")
		self.assertIn("auto& session = oa::ExecutionSession::getActive();", body)
		self.assertIn(
			"oa::OpLoweringScope lowering(session);", body
		)
		self.assertIn("oa::FnImage::gaussianNoise(session.engine()", body)
		self.assertIn(
			"oa::detail::opRegistry::FnImage::gaussianNoise, {&inImage}, {&out}",
			body,
		)
		self.assertIn(
			'oa::OpAttribute::fromUnsignedInteger("seed", inSeed)',
			body,
		)

	def test_optional_input_contract_and_autograd_omit_absent_edge(self) -> None:
		op = {
			"name": "linear",
			"kind": "binary",
			"surface": "public_operation",
			"body": "manual_session",
			"generate_declaration": False,
			"api_return": "oa::Matrix",
			"api_params": [
				"const oa::Matrix& inX",
				"const oa::Matrix& inWeight",
				"const oa::Matrix& inBias = oa::Matrix{}",
			],
			"contract": {
				"input_kinds": ["matrix", "matrix", "matrix"],
				"optional_inputs": [2],
				"output_kinds": ["matrix"],
				"shape_rule": "explicit",
				"dtype_rule": "match_input",
				"effects": ["read_inputs", "write_outputs"],
				"mutated_inputs": [],
				"output_alias_inputs": [-1],
				"differentiation": "reverse",
				"lowering": "gemm",
				"control_flow": "straight_line",
			},
			"autograd": {
				"grad_class": "oa::GradLinear",
				"inputs": ["inX", "inWeight", "inBias"],
				"saved": ["inX", "inWeight"],
				"attach": "standard",
			},
			"dnn": {
				"role": "matmul",
				"epilogue": "bias",
				"epilogue_requires_input": 2,
			},
		}
		oafnautogen.validate_schema(
			Path("MlFnMatrixBlas.toml"), [op], {"Linear"}
		)
		registry = oafnautogen.emit_operation_registry([
			("oa::FnMatrix", op),
		])
		self.assertIn(".optionalInputMask = 0x04U", registry)
		attachment = "\n".join(
			oafnautogen._emit_manual_autograd_attach_definition(op)
		)
		self.assertIn("if (not inBias.isEmpty())", attachment)
		self.assertIn(
			"graphInputs.pushBack(inX);",
			attachment,
		)
		self.assertIn(
			"graphInputs.pushBack(inBias);",
			attachment,
		)
		dnn_roles = oafnautogen.emit_dnn_operation_roles([
			("oa::FnMatrix", op),
		])
		self.assertIn('.name = "oa::FnMatrix::linear"', dnn_roles)
		self.assertIn(".type = oa::DnnOpType::Matmul", dnn_roles)
		self.assertIn(".epilogue = oa::GemmEpilogue::Bias", dnn_roles)
		self.assertIn(".epilogueRequiredInput = 2U", dnn_roles)
		op["dnn"]["epilogue_requires_input"] = 1
		with self.assertRaises(SystemExit):
			oafnautogen.validate_schema(
				Path("MlFnMatrixBlas.toml"), [op], {"Linear"}
			)

	def test_void_mutation_contract_may_have_no_outputs(self) -> None:
		op = {
			"name": "update",
			"kind": "binary",
			"surface": "public_cpp_operation",
			"body": "manual_session",
			"generate_declaration": False,
			"api_return": "void",
			"api_params": [
				"const oa::Matrix& inSource",
				"oa::Matrix& inOutState",
			],
			"contract": {
				"input_kinds": ["matrix", "matrix"],
				"output_kinds": [],
				"shape_rule": "explicit",
				"dtype_rule": "explicit",
				"effects": ["read_inputs"],
				"mutated_inputs": [1],
				"output_alias_inputs": [],
				"differentiation": "none",
				"lowering": "dispatch",
				"control_flow": "straight_line",
			},
		}
		oafnautogen.validate_schema(
			Path("CoreFnMatrixMutation.toml"), [op], {"Update"}
		)
		registry = oafnautogen.emit_operation_registry([
			("oa::FnMatrix", op),
		])
		self.assertIn(".outputCount = 0U", registry)
		self.assertIn(".mutatedInputMask = 0x02U", registry)

		invalid = {
			**op,
			"name": "observe",
			"contract": {
				**op["contract"],
				"mutated_inputs": [],
			},
		}
		with self.assertRaises(SystemExit):
			oafnautogen.validate_schema(
				Path("CoreFnMatrixMutation.toml"), [invalid], {"Observe"}
			)

	def test_variadic_contract_emits_homogeneous_tail_descriptor(self) -> None:
		op = {
			"name": "concat",
			"kind": "unary",
			"surface": "public_operation",
			"body": "manual_session",
			"generate_declaration": False,
			"api_return": "oa::Matrix",
			"api_params": [
				"oa::Span<oa::Matrix> inInputs",
				"oa::I32 inDim = 0",
			],
			"python": {
				"args": ["inputs", "dim"],
			},
			"contract": {
				"input_kinds": [],
				"variadic_inputs": {"kind": "matrix", "minimum": 1},
				"output_kinds": ["matrix"],
				"shape_rule": "explicit",
				"dtype_rule": "match_input",
				"effects": ["read_inputs", "write_outputs"],
				"mutated_inputs": [],
				"output_alias_inputs": [-1],
				"differentiation": "none",
				"lowering": "dispatch",
				"control_flow": "straight_line",
			},
		}
		oafnautogen.validate_schema(
			Path("CoreFnMatrixIndex.toml"), [op], set()
		)
		registry = oafnautogen.emit_operation_registry([
			("oa::FnMatrix", op),
		])
		self.assertIn(
			".variadicInputKind = oa::OpValueKind::Matrix", registry
		)
		self.assertIn(".minimumVariadicInputCount = 1U", registry)
		self.assertIn(".inputCount = 0U", registry)
		binding = "\n".join(
			oafnautogen.emit_python_binding("oa::FnMatrix", op)
		)
		self.assertIn("std::vector<oa::Matrix> inInputs", binding)
		self.assertIn(
			"oa::Span<oa::Matrix>(inInputs.data(), inInputs.size())", binding
		)

		invalid = {
			**op,
			"contract": {
				**op["contract"],
				"variadic_inputs": {"kind": "matrix", "minimum": 0},
			},
		}
		with self.assertRaises(SystemExit):
			oafnautogen.validate_schema(
				Path("CoreFnMatrixIndex.toml"), [invalid], set()
			)

	def test_schema_kernel_owns_registry_row_and_parallel_build_pair(self) -> None:
		kernels = [{
			"name": "rlCartPoleStep",
			"id_prefix": "Ml",
			"id_local": 275,
			"category": "Ml",
			"origin": "oa",
			"source": "Rl/Environment/RlCartPoleStep",
		}]
		registry = oafnautogen.emit_schema_kernel_registry(kernels, "Ml")
		ids = oafnautogen.emit_schema_kernel_ids(kernels)
		cmake = oafnautogen.emit_schema_kernel_cmake(kernels)
		self.assertIn('"rlCartPoleStep"', registry)
		self.assertIn("oa::computeKernelPrefix::Ml, 275", registry)
		self.assertIn("oa::U64 rlCartPoleStep", ids)
		self.assertIn("oa::computeKernelPrefix::Ml, 275", ids)
		self.assertIn("OA_FN_AUTOGEN_ML_KERNEL_NAMES", cmake)
		self.assertIn("Rl/Environment/RlCartPoleStep", cmake)

	def test_schema_kernel_collision_reads_canonical_registry_fragments(self) -> None:
		with tempfile.TemporaryDirectory() as temp_dir:
			root = Path(temp_dir)
			registry = root / "kernelRegistry.h"
			registry.write_text(
				'#include <oa/runtime/kernelRegistryStandaloneMl.gen.inl>\n'
				'#include <oa/runtime/oaTileKernelRegistry.gen.inc>\n',
				encoding="utf-8",
			)
			(root / "kernelRegistryStandaloneMl.gen.inl").write_text(
				'{ "Standalone", OA_COMPUTE_KERNEL_ID('
				'oa::computeKernelPrefix::Ml, 1), category, "oa" },\n',
				encoding="utf-8",
			)
			(root / "oaTileKernelRegistry.gen.inc").write_text(
				'{ "Tile", OA_COMPUTE_KERNEL_ID('
				'oa::computeKernelPrefix::Ml, 2), category, "oa" },\n',
				encoding="utf-8",
			)

			with self.assertRaises(SystemExit):
				oafnautogen.validate_schema_kernel_ownership([{
					"name": "Tile",
					"id_prefix": "Ml",
					"id_local": 3,
					"source": "Ml/Tile",
					"schema": "Ml/Test.toml",
					"operation": "tile",
				}], registry)

			with self.assertRaises(SystemExit):
				oafnautogen.validate_schema_kernel_ownership([{
					"name": "Other",
					"id_prefix": "Ml",
					"id_local": 1,
					"source": "Ml/Other",
					"schema": "Ml/Test.toml",
					"operation": "other",
				}], registry)

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
			"name": "addInPlace",
			"kind": "binary",
			"surface": "public_operation",
			"generate_declaration": False,
			"api_return": "void",
			"api_params": [
				"oa::Matrix& inSelf",
				"const oa::Matrix& inOther",
			],
			"contract": contract,
		}
		session = {
			"name": "cartPoleStep",
			"kind": "session_command",
			"surface": "session_command",
			"generate_declaration": False,
			"session": {
				"owners": ["oa::Environment"],
				"transition": "Ready -> Recorded",
				"effects": ["read_state", "write_state", "device_work"],
				"completion": "recorded",
			},
			"contract": contract,
		}
		reference = oafnautogen.emit_operation_reference([
			("oa::FnMatrix", handwritten),
			("oa::FnEnvironment", session),
		])
		self.assertIn(
			"- C++: `void addInPlace(oa::Matrix& inSelf, const oa::Matrix& inOther)`",
			reference,
		)
		self.assertIn("- C++: internal session command", reference)
		self.assertIn("- Remaining session descriptors: `0`", reference)
		self.assertIn("- Owners: `oa::Environment`", reference)

	def test_host_utility_and_session_descriptor_are_not_fake_operations(self) -> None:
		host = {
			"name": "parseNalAnnexB",
			"kind": "unary",
			"surface": "host_utility",
			"body": "cpu_util",
		}
		session = {
			"name": "decodeFrame",
			"kind": "session_command",
			"surface": "session_command",
			"body": "manual_session",
			"session": {
				"owners": ["oa::VideoDecoder"],
				"transition": "Ready -> Ready | Failed",
				"effects": [
					"read_state",
					"write_state",
					"device_work",
					"host_input",
				],
				"completion": "output_event",
			},
		}
		oafnautogen.validate_schema(
			Path("VisionFnVideoNal.toml"), [host], set()
		)
		oafnautogen.validate_schema(
			Path("VisionVideoDecoder.toml"), [session], set()
		)
		reference = oafnautogen.emit_operation_reference([
			("oa::FnVideo", host),
			("oa::VideoDecoder", session),
		])
		self.assertIn(
			"- Rows requiring semantic operation contracts: `0`",
			reference,
		)
		self.assertIn("- Remaining required rows: `0`", reference)
		self.assertIn("- Remaining session descriptors: `0`", reference)
		self.assertNotIn("## `oa::FnVideo::parseNalAnnexB`", reference)
		self.assertIn("### `oa::VideoDecoder::decodeFrame`", reference)
		session["session"]["completion"] = "implicit_wait"
		with self.assertRaises(SystemExit):
			oafnautogen.validate_schema(
				Path("VisionVideoDecoder.toml"), [session], set()
			)

if __name__ == "__main__":
	unittest.main()
