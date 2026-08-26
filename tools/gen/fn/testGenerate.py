from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

RepoRoot = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(RepoRoot))
from tools.gen.fn import generate as fnGenerator
from tools.gen.fn import checkSchemaCoverage as check_schema_coverage


class FnGenerationTest(unittest.TestCase):
	def testSchemaOwnersPartitionCoreAndSdkAuthorities(self) -> None:
		with tempfile.TemporaryDirectory() as tempDir:
			root = Path(tempDir)
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
			 fnGenerator.partitionSchemaOwners([core, sdk]),
			 ([core], [sdk]),
			)
			with self.assertRaises(SystemExit):
				fnGenerator.partitionSchemaOwners([invalid])

	def testSdkKernelManifestHasNoCoreRegistrySurface(self) -> None:
		kernel = {
		 "name": "RlCartPoleStep",
		 "source": "environment/rlCartPoleStep",
		}
		emitted = fnGenerator.emitSdkKernelCmake([kernel])
		self.assertIn("OA_SDK_ENVIRONMENT_KERNEL_NAMES", emitted)
		self.assertIn("RlCartPoleStep", emitted)
		self.assertIn("environment/rlCartPoleStep", emitted)
		self.assertNotIn("OA_OPERATION_ML_KERNEL_NAMES", emitted)

	def testCompressedFnNamespaceDeclarationsAreRejected(self) -> None:
		with tempfile.TemporaryDirectory() as tempDir:
			root = Path(tempDir)
			header = root / "source/cpp/include/oa/audio/fnAudio.h"
			header.parent.mkdir(parents=True)
			header.write_text(
			 "namespace oa::FnAudio {\n}\n",
			 encoding="utf-8",
			)
			self.assertEqual(
			 check_schema_coverage.compressedFnNamespaceDeclarations(root),
			 ["source/cpp/include/oa/audio/fnAudio.h:1"],
			)
			header.write_text(
			 "namespace oa {\n\nnamespace FnAudio {\n}\n\n}\n",
			 encoding="utf-8",
			)
			self.assertEqual(
			 check_schema_coverage.compressedFnNamespaceDeclarations(root),
			 [],
			)

	def testHeaderFragmentRemainsNamespaceNeutral(self) -> None:
		emitted = fnGenerator.emitHeaderFragment(
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

	def testPythonSurfaceInventoryIsOwnedBySchemaNamespace(self) -> None:
		emitted = fnGenerator.emitPythonSurfaceInventory([
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

	def testLossCategoryFragmentStaysPrivate(self) -> None:
		with tempfile.TemporaryDirectory() as tempDir:
			root = Path(tempDir)
			schemaDir = root / "Ml"
			schemaDir.mkdir()
			schema = schemaDir / "MlFnLoss.toml"
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
			layouts = fnGenerator.processSchema(
			 schema, set(), out, live=False, dryRun=False
			)
			self.assertIsInstance(layouts, list)
			layout = layouts[0]
			self.assertEqual(
			 layout.headerPath,
			  out / "cpp/lib/oa/ml/fnloss/example/fnLossExample.gen.h",
			)
			self.assertEqual(
			 layout.cppPath,
			  out / "cpp/lib/oa/ml/fnloss/example/fnLossExample.gen.cpp",
			)
			self.assertTrue(layout.headerPath.is_file())
			self.assertTrue(layout.cppPath.is_file())
			self.assertFalse((out / "cpp/include/oa/ml/fnloss/example").exists())
			fnGenerator.writeManifestFiles(layouts, out, dryRun=False)
			umbrellaFragment = out / "cpp/lib/oa/ml/fnloss/fnLoss.gen.h"
			self.assertTrue(umbrellaFragment.is_file())
			self.assertIn(
			 '#include "example/fnLossExample.gen.h"',
			 umbrellaFragment.read_text(encoding="utf-8"),
			)
			self.assertFalse((out / "cpp/include/oa/ml/fnLoss.gen.h").exists())

	def testEmptySiblingGroupDoesNotDeleteSharedSourceManifest(self) -> None:
		with tempfile.TemporaryDirectory() as temporary:
			root = Path(temporary)
			generated = fnGenerator.SchemaLayout(
				domain="ml",
				namespace="oa::FnMatrix",
				filePrefix="FnMatrix",
				cppSubdir="FnMatrix",
				headerPath=root / "cpp/lib/oa/ml/fnmatrix/a.gen.h",
				cppPath=root / "cpp/lib/oa/ml/fnmatrix/a.gen.cpp",
				autogradHeaderPath=root / "unusedA.gen.h",
				testPath=root / "testA.gen.cpp",
				emitHeader=False,
				emitCpp=True,
			)
			empty = fnGenerator.SchemaLayout(
				domain="ml",
				namespace="oa::FnMatrix",
				filePrefix="FnMatrix",
				cppSubdir="Ssm",
				headerPath=root / "cpp/lib/oa/ml/ssm/b.gen.h",
				cppPath=root / "cpp/lib/oa/ml/ssm/b.gen.cpp",
				autogradHeaderPath=root / "unusedB.gen.h",
				testPath=root / "testB.gen.cpp",
				emitHeader=False,
				emitCpp=False,
			)
			fnGenerator.writeManifestFiles(
				[generated, empty], root, dryRun=False
			)
			manifest = root / "cmake/gen/mlFnMatrixSources.cmake"
			self.assertTrue(manifest.is_file())
			self.assertIn(
				"cpp/lib/oa/ml/fnmatrix/a.gen.cpp",
				manifest.read_text(encoding="utf-8"),
			)

	def testManualAutogradAcceptsSavedOnlyTensorAndLowercaseOut(self) -> None:
		emitted = "\n".join(fnGenerator._emitManualAutogradAttachDefinition({
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
		 "gradFn->saveForBackward(inLogits, inTargets, out);",
		 emitted,
		)
		self.assertIn(
		 "gradFn->setGraphInputs(oa::Vec<oa::Matrix>{inLogits});",
		 emitted,
		)

	def testCoreAutogradRequiresExactFamilyHeader(self) -> None:
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
			fnGenerator.emitCppFile(
			 [op], "core/coreFnMatrixShape.toml", "Shape", "core",
			)

	def testMlAutogradRequiresExactFamilyHeader(self) -> None:
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
			fnGenerator.emitCppFile(
			 [op], "ml/mlFnMatrixExample.toml", "Example", "ml",
			)

	def testCoreAutogradHeaderStaysInCoreAndHasNoMlInclude(self) -> None:
		op = {
		 "name": "add",
		 "autograd": {
		  "grad_class": "oa::GradAdd",
		  "formula": "auto_binary_add",
		 },
		}
		emitted = fnGenerator.emitAutogradHeader(
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

	def testAutogradAttachmentAndManifestAreDomainOwned(self) -> None:
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
		coreAttach = fnGenerator.emitManualAutogradAttachHeader(
		 [("oa::FnMatrix", op)], "core"
		)
		self.assertIn(
		 "concrete nodes are private to the", coreAttach
		)
		self.assertIn(
		 "return detail::FnMatrix::exampleEnabled", coreAttach
		)
		self.assertNotIn("oa::GradExample", coreAttach)
		coreAttachCpp = fnGenerator.emitManualAutogradAttachCpp(
		 [("oa::FnMatrix", op)], "core"
		)
		self.assertIn(
		 "#include <oa/core/autograd/matrix/autogradBlas.h>",
		 coreAttachCpp,
		)
		self.assertIn("oa::GradExample", coreAttachCpp)
		self.assertNotIn("/autograd/nodes.h>", coreAttachCpp)
		narrowOp = {
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
		narrowAttachCpp = fnGenerator.emitManualAutogradAttachCpp(
		 [("oa::FnMatrix", narrowOp)], "core"
		)
		self.assertIn(
		 "#include <oa/core/autograd/matrix/autogradShape.h>",
		 narrowAttachCpp,
		)
		self.assertNotIn(
		 "#include <oa/core/autograd/nodes.h>", narrowAttachCpp
		)

		with tempfile.TemporaryDirectory() as tempDir:
			root = Path(tempDir)
			layout = fnGenerator.SchemaLayout(
			 domain="core",
			 namespace="oa::FnMatrix",
			 filePrefix="FnMatrix",
			 cppSubdir="FnMatrix",
			 headerPath=root / "cpp/lib/oa/core/fnmatrix/x.gen.h",
			 cppPath=root / "cpp/lib/oa/core/fnmatrix/x.gen.cpp",
			 autogradHeaderPath=(
			  root
			  / "cpp/lib/oa/core/autograd/matrix/autogradX.gen.h"
			 ),
			 testPath=root / "Test.cpp",
			 emitAutograd=True,
			)
			fnGenerator.writeAutogradManifestFiles(
			 [layout], root, dryRun=False
			)
			manifest = (
			 root / "cpp/lib/oa/core/autograd/autograd.gen.h"
			).read_text(encoding="utf-8")
			self.assertIn('<oa/core/autograd.h>', manifest)
			self.assertIn('"matrix/autogradX.gen.h"', manifest)
			self.assertNotIn("<oa/ml/", manifest)

	def testPythonKeywordArgumentsRequireCamelCase(self) -> None:
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
		fnGenerator.validateSchema(
		 Path("CoreFnMatrixExample.toml"), [op], set()
		)
		op["python"]["args"] = ["input", "batch_size"]
		with self.assertRaises(SystemExit):
			fnGenerator.validateSchema(
			 Path("CoreFnMatrixExample.toml"), [op], set()
			)

	def testSchemaLevelPythonGenerationDerivesUniformSignature(self) -> None:
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
		fnGenerator.applySchemaDefaults(data, op)
		self.assertEqual(
		 op["api_params"],
		 ["const oa::Matrix& inA", "oa::F32 inExponent"],
		)
		self.assertEqual(op["python"]["args"], ["a", "exponent"])
		emitted = "\n".join(
		 fnGenerator.emitPythonBinding("oa::FnMatrix", op)
		)
		self.assertIn(
		 "matrixPtr(oa::FnMatrix::pow(inA, inExponent))",
		 emitted,
		)

	def testSchemaLevelAutogradHeaderIsDefaultNotOverride(self) -> None:
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
		fnGenerator.applySchemaDefaults(data, defaulted)
		fnGenerator.applySchemaDefaults(data, overridden)
		self.assertEqual(
		 defaulted["autograd"]["node_header"],
		 "oa/core/autograd/matrix/autogradElemwise.h",
		)
		self.assertEqual(
		 overridden["autograd"]["node_header"],
		 "oa/core/autograd/matrix/autogradShape.h",
		)

	def testGeneratedAudioBindingOwnsSemanticResult(self) -> None:
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
		fnGenerator.validatePythonBinding("AudioFnAudioSignal.toml:Gain", op)
		emitted = "\n".join(
		 fnGenerator.emitPythonBinding("oa::FnAudio", op)
		)
		self.assertIn(
		 "return new oa::Audio(oa::FnAudio::gain(inAudio, inGainDb));",
		 emitted,
		)

	def testGeneratedConstSpanBindingUsesPythonVectorBoundary(self) -> None:
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
		fnGenerator.validatePythonBinding(
		 "AudioFnAudioSignal.toml:SosFilter", op
		)
		emitted = "\n".join(
		 fnGenerator.emitPythonBinding("oa::FnAudio", op)
		)
		self.assertIn(
		 "std::vector<oa::BiquadCoefficients> inSections", emitted
		)
		self.assertIn(
		 "oa::Span<const oa::BiquadCoefficients>(inSections.data(), inSections.size())",
		 emitted,
		)

	def testManualPythonBindingOwnsSurfaceWithoutGeneratedBody(self) -> None:
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
		fnGenerator.validatePythonBinding(
		 "VisionFnImageCodec.toml:DecodeFile", op
		)
		emitted = fnGenerator.emitPythonBindings([("oa::FnImage", op)])
		self.assertNotIn("decodeFile", emitted)
		surface = fnGenerator.emitPythonSurfaceInventory([
		 ("Vision", "oa::FnImage", op)
		])
		self.assertIn('(\"decodeFile\", \"decodeFile\")', surface)

	def testGeneratedVoidBindingReturnsPythonNone(self) -> None:
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
		fnGenerator.validatePythonBinding(
		 "MlFnMatrixVq.toml:Update", op
		)
		emitted = "\n".join(
		 fnGenerator.emitPythonBinding("oa::FnMatrix", op)
		)
		self.assertIn(
		 "oa::FnMatrix::update(inSource, inOutState);",
		 emitted,
		)
		self.assertNotIn(
		 "return oa::FnMatrix::update",
		 emitted,
		)

	def testGeneratedStructuredResultBindingIsSchemaOwned(self) -> None:
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
		fnGenerator.applySchemaDefaults(data, op)
		fnGenerator.validatePythonBinding(
		 "MlFnMatrixPool.toml:MaxPool2d", op
		)
		emitted = fnGenerator.emitPythonBindings([("oa::FnMatrix", op)])
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
		surface = fnGenerator.emitPythonSurfaceInventory(
		 [("Ml", "oa::FnMatrix", op)]
		)
		self.assertIn(
		 'SCHEMA_ROOT_EXPORTS = {\n'
		 '\t"FnMatrix": (\n'
		 '\t\t("MaxPool2dResult", "MaxPool2dResult"),',
		 surface,
		)

	def testGeneratedStructuredResultCanReuseManualTypeBinding(self) -> None:
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
		fnGenerator.validatePythonBinding(
		 "VisionFnDetection.toml:Nms", op
		)
		emitted = fnGenerator.emitPythonBindings([("oa::FnDetection", op)])
		self.assertNotIn("nb::class_<oa::NmsResult>", emitted)
		self.assertIn(
		 "new oa::NmsResult(oa::FnDetection::nms(inBoxes))",
		 emitted,
		)

	def testGeneratedPythonResultValidationPreservesErrorBoundary(self) -> None:
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
		fnGenerator.validatePythonBinding("mlFnAdvantage.toml:normalize", op)
		emitted = "\n".join(
		 fnGenerator.emitPythonBinding("oa::FnAdvantage", op)
		)
		self.assertIn(
		 "auto result = oa::FnAdvantage::normalize(inAdvantage);",
		 emitted,
		)
		self.assertIn("if (result.isEmpty())", emitted)
		self.assertIn("matrixPtr(oa::move(result))", emitted)

	def testGeneratedPythonDefaultIsTyped(self) -> None:
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
		 fnGenerator.emitPythonBinding("oa::FnAudio", op)
		)
		self.assertIn(
		 'nb::arg("config") = oa::StftConfig{}',
		 emitted,
		)

	def testManualMultiOutputAutogradRequiresProvenance(self) -> None:
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
		fnGenerator.validateSchema(
		 Path("MlFnMatrixNorm.toml"), [op], set()
		)
		del op["autograd"]["implementation"]
		with self.assertRaises(SystemExit):
			fnGenerator.validateSchema(
			 Path("MlFnMatrixNorm.toml"), [op], set()
			)

	def testSessionContractHonorsFrozenEightAttributeDescriptor(self) -> None:
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
		fnGenerator.validateSchema(
		 Path("mlFnEnvironmentSession.toml"), [op], {"RlCartPoleStep"}
		)
		emitted = fnGenerator.emitOperationRegistry(
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
			fnGenerator.validateSchema(
			 Path("mlFnEnvironmentSession.toml"), [op], {"RlCartPoleStep"}
			)
		op["contract"]["attributes"].pop()
		op["kind"] = "unary"
		with self.assertRaises(SystemExit):
			fnGenerator.validateSchema(
			 Path("mlFnEnvironmentSession.toml"), [op], {"RlCartPoleStep"}
			)

	def testSchemaSetRejectsCrossFileDuplicateIdentity(self) -> None:
		with tempfile.TemporaryDirectory() as tempDir:
			root = Path(tempDir) / "Core"
			root.mkdir()
			first = root / "CoreFnMatrixFirst.toml"
			second = root / "CoreFnMatrixSecond.toml"
			first.write_text('[[ops]]\nname = "Example"\n', encoding="utf-8")
			second.write_text('[[ops]]\nname = "Example"\n', encoding="utf-8")
			with self.assertRaises(SystemExit):
				fnGenerator.validateSchemaSet([first, second])

	def testRegistryNamespacesSameSpellingWithoutCollision(self) -> None:
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
		emitted = fnGenerator.emitOperationRegistry([
		 ("oa::FnAudio", {"name": "normalize", "contract": contract}),
		 ("oa::FnImage", {"name": "normalize", "contract": contract}),
		])
		self.assertIn("namespace FnAudio", emitted)
		self.assertIn("namespace FnImage", emitted)
		self.assertIn('.name = "oa::FnAudio::normalize"', emitted)
		self.assertIn('.name = "oa::FnImage::normalize"', emitted)

	def testNullaryContractHasZeroInputsAndOneOutput(self) -> None:
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
		fnGenerator.validateSchema(
		 Path("CoreFnMatrixRng.toml"), [op], {"Fill"}
		)
		emitted = fnGenerator.emitOperationRegistry([
		 ("oa::FnMatrix", op),
		])
		self.assertIn(".inputCount = 0U", emitted)
		self.assertIn(".outputCount = 1U", emitted)
		self.assertIn(".attributeCount = 2U", emitted)
		body = fnGenerator.emitSessionBody(op)
		self.assertIn(
		 "oa::Matrix oa::FnMatrix::fill("
		 "const oa::MatrixShape& inShape, oa::F32 inValue)",
		 body,
		)
		self.assertIn(
		 "oa::OpAttribute::fromShape(\"shape\", inShape)",
		 body,
		)

	def testVoidDeclarationDoesNotEmitNodiscard(self) -> None:
		lines = fnGenerator.headerFragmentForOp({
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

	def testManualContractForwarderGroupsComposedLowering(self) -> None:
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
		body = fnGenerator.emitManualForwarder(op, "oa::FnImage")
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

	def testOptionalInputContractAndAutogradOmitAbsentEdge(self) -> None:
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
		fnGenerator.validateSchema(
		 Path("MlFnMatrixBlas.toml"), [op], {"Linear"}
		)
		registry = fnGenerator.emitOperationRegistry([
		 ("oa::FnMatrix", op),
		])
		self.assertIn(".optionalInputMask = 0x04U", registry)
		attachment = "\n".join(
		 fnGenerator._emitManualAutogradAttachDefinition(op)
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
		dnnRoles = fnGenerator.emitDnnOperationRoles([
		 ("oa::FnMatrix", op),
		])
		self.assertIn('.name = "oa::FnMatrix::linear"', dnnRoles)
		self.assertIn(".type = oa::DnnOpType::Matmul", dnnRoles)
		self.assertIn(".epilogue = oa::GemmEpilogue::Bias", dnnRoles)
		self.assertIn(".epilogueRequiredInput = 2U", dnnRoles)
		op["dnn"]["epilogue_requires_input"] = 1
		with self.assertRaises(SystemExit):
			fnGenerator.validateSchema(
			 Path("MlFnMatrixBlas.toml"), [op], {"Linear"}
			)

	def testVoidMutationContractMayHaveNoOutputs(self) -> None:
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
		fnGenerator.validateSchema(
		 Path("CoreFnMatrixMutation.toml"), [op], {"Update"}
		)
		registry = fnGenerator.emitOperationRegistry([
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
			fnGenerator.validateSchema(
			 Path("CoreFnMatrixMutation.toml"), [invalid], {"Observe"}
			)

	def testVariadicContractEmitsHomogeneousTailDescriptor(self) -> None:
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
		fnGenerator.validateSchema(
		 Path("CoreFnMatrixIndex.toml"), [op], set()
		)
		registry = fnGenerator.emitOperationRegistry([
		 ("oa::FnMatrix", op),
		])
		self.assertIn(
		 ".variadicInputKind = oa::OpValueKind::Matrix", registry
		)
		self.assertIn(".minimumVariadicInputCount = 1U", registry)
		self.assertIn(".inputCount = 0U", registry)
		binding = "\n".join(
		 fnGenerator.emitPythonBinding("oa::FnMatrix", op)
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
			fnGenerator.validateSchema(
			 Path("CoreFnMatrixIndex.toml"), [invalid], set()
			)

	def testSchemaKernelOwnsRegistryRowAndParallelBuildPair(self) -> None:
		kernels = [{
		 "name": "rlCartPoleStep",
		 "id_prefix": "Ml",
		 "id_local": 275,
		 "category": "Ml",
		 "origin": "oa",
		 "source": "Rl/Environment/RlCartPoleStep",
		}]
		registry = fnGenerator.emitSchemaKernelRegistry(kernels, "Ml")
		ids = fnGenerator.emitSchemaKernelIds(kernels)
		cmake = fnGenerator.emitSchemaKernelCmake(kernels)
		self.assertIn('"rlCartPoleStep"', registry)
		self.assertIn("oa::computeKernelPrefix::Ml, 275", registry)
		self.assertIn("oa::U64 rlCartPoleStep", ids)
		self.assertIn("oa::computeKernelPrefix::Ml, 275", ids)
		self.assertIn("OA_OPERATION_ML_KERNEL_NAMES", cmake)
		self.assertIn("Rl/Environment/RlCartPoleStep", cmake)

	def testSchemaKernelCollisionReadsCanonicalRegistryFragments(self) -> None:
		with tempfile.TemporaryDirectory() as tempDir:
			root = Path(tempDir)
			registry = root / "kernelRegistry.h"
			registry.write_text(
			 '#include <oa/runtime/gen/kernelRegistryStandaloneMl.inl>\n'
			 '#include <oa/runtime/gen/tileKernelRegistry.inc>\n',
			 encoding="utf-8",
			)
			(root / "gen").mkdir()
			(root / "gen/kernelRegistryStandaloneMl.inl").write_text(
			 '{ "Standalone", OA_COMPUTE_KERNEL_ID('
			 'oa::computeKernelPrefix::Ml, 1), category, "oa" },\n',
			 encoding="utf-8",
			)
			(root / "gen/tileKernelRegistry.inc").write_text(
			 '{ "Tile", OA_COMPUTE_KERNEL_ID('
			 'oa::computeKernelPrefix::Ml, 2), category, "oa" },\n',
			 encoding="utf-8",
			)

			with self.assertRaises(SystemExit):
				fnGenerator.validateSchemaKernelOwnership([{
				 "name": "Tile",
				 "id_prefix": "Ml",
				 "id_local": 3,
				 "source": "Ml/Tile",
				 "schema": "Ml/Test.toml",
				 "operation": "tile",
				}], registry)

			with self.assertRaises(SystemExit):
				fnGenerator.validateSchemaKernelOwnership([{
				 "name": "Other",
				 "id_prefix": "Ml",
				 "id_local": 1,
				 "source": "Ml/Other",
				 "schema": "Ml/Test.toml",
				 "operation": "other",
				}], registry)

	def testOperationReferenceDistinguishesHandwrittenAndSessionCommands(self) -> None:
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
		reference = fnGenerator.emitOperationReference([
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

	def testHostUtilityAndSessionDescriptorAreNotFakeOperations(self) -> None:
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
		fnGenerator.validateSchema(
		 Path("VisionFnVideoNal.toml"), [host], set()
		)
		fnGenerator.validateSchema(
		 Path("VisionVideoDecoder.toml"), [session], set()
		)
		reference = fnGenerator.emitOperationReference([
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
			fnGenerator.validateSchema(
			 Path("VisionVideoDecoder.toml"), [session], set()
			)

if __name__ == "__main__":
	unittest.main()
