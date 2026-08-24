#!/usr/bin/env python3
"""Regression tests for the configured shader/fixed-ID audit."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path


TOOLS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(TOOLS_DIR))

import auditKernels  # noqa: E402


class KernelAuditTest(unittest.TestCase):
	def testManifestRejectsDuplicatesSuffixesAndParentTraversal(self) -> None:
		with tempfile.TemporaryDirectory() as temp:
			build = Path(temp)
			(build / auditKernels.MANIFEST_NAME).write_text(
			 "Good\nGood\nBad.spv\n../Escape\nNested/../Escape\n"
			)
			names, errors = auditKernels.loadManifest(build)

		self.assertEqual(names[0], "Good")
		self.assertTrue(any("duplicate manifest key: Good" in error for error in errors))
		self.assertTrue(any("must omit .spv" in error for error in errors))
		self.assertEqual(
		 2,
		 sum("escapes the SPIR-V root" in error for error in errors),
		)

	def testSchemaKernelReferencesArePartitionedByOwner(self) -> None:
		with tempfile.TemporaryDirectory() as temp:
			root = Path(temp)
			schema = root / "tools/gen/fn/schema/ml"
			schema.mkdir(parents=True)
			(schema / "core.toml").write_text(
			 '[[ops]]\nname = "coreOp"\nkernel_main = "CoreKernel"\n'
			)
			(schema / "sdk.toml").write_text(
			 'owner = "sdk"\n[[ops]]\nname = "sdkOp"\n'
			 'kernel_main = "SdkKernel"\n'
			)

			core = auditKernels.loadSchemaKernelReferences(root, "core")
			sdk = auditKernels.loadSchemaKernelReferences(root, "sdk")

		self.assertEqual({"CoreKernel"}, set(core))
		self.assertEqual({"SdkKernel"}, set(sdk))

	def testRegistryRejectsDuplicateNamesAndPackedIds(self) -> None:
		with tempfile.TemporaryDirectory() as temp:
			root = Path(temp)
			registry = root / "source/cpp/lib/oa/runtime/kernelRegistry.h"
			registry.parent.mkdir(parents=True)
			registry.write_text(
			 '{ "First", OA_COMPUTE_KERNEL_ID(oa::computeKernelPrefix::Ml, 1), '
			 'oa::ComputeKernelCategory::Ml, "oa" },\n'
			 '{ "First", OA_COMPUTE_KERNEL_ID(oa::computeKernelPrefix::Ml, 2), '
			 'oa::ComputeKernelCategory::Ml, "oa" },\n'
			 '{ "Second", OA_COMPUTE_KERNEL_ID(oa::computeKernelPrefix::Ml, 1), '
			 'oa::ComputeKernelCategory::Ml, "oa" },\n'
			)
			fixed, errors = auditKernels.loadFixedRegistry(root)

		self.assertEqual({"First", "Second"}, set(fixed))
		self.assertTrue(any("duplicate fixed name First" in error for error in errors))
		self.assertTrue(any("duplicate fixed ID Ml:1" in error for error in errors))

	def testTypedIdRequiresAFixedRegistryRow(self) -> None:
		with tempfile.TemporaryDirectory() as temp:
			root = Path(temp)
			public = root / "source/cpp/include/oa/runtime"
			public.mkdir(parents=True)
			(public / "computeKernel.h").write_text(
			 "static constexpr oa::U64 Present = "
			 "OA_COMPUTE_KERNEL_ID(oa::computeKernelPrefix::Ml, 1);\n"
			 "static constexpr oa::U64 Missing = "
			 "OA_COMPUTE_KERNEL_ID(oa::computeKernelPrefix::Ml, 7);\n"
			)
			fixed = {
			 "Present": auditKernels.FixedKernel(
			  "Present", "Ml", 1, "Ml", "oa", Path("KernelRegistry.h")
			 )
			}
			typed, errors = auditKernels.loadTypedIds(root, fixed)

		self.assertEqual({"Present": "Present"}, typed)
		self.assertTrue(any("Missing" in error and "Ml:7" in error for error in errors))

	def testTypedIdNameMustMatchItsFixedRegistryRow(self) -> None:
		with tempfile.TemporaryDirectory() as temp:
			root = Path(temp)
			public = root / "source/cpp/include/oa/runtime"
			public.mkdir(parents=True)
			(public / "computeKernel.h").write_text(
			 "static constexpr oa::U64 Claimed = "
			 "OA_COMPUTE_KERNEL_ID(oa::computeKernelPrefix::Ml, 1);\n"
			)
			fixed = {
			 "Actual": auditKernels.FixedKernel(
			  "Actual", "Ml", 1, "Ml", "oa", Path("KernelRegistry.h")
			 )
			}
			typed, errors = auditKernels.loadTypedIds(root, fixed)

		self.assertEqual({}, typed)
		self.assertTrue(
		 any("Claimed resolves to fixed row Actual" in error for error in errors)
		)

	def testMatmulVariantIdentityRejectsNameMismatchAndDuplicates(self) -> None:
		with tempfile.TemporaryDirectory() as temp:
			root = Path(temp)
			gemm = root / "source/cpp/lib/oa/runtime/gemm"
			gemm.mkdir(parents=True)
			(gemm / "gen").mkdir()
			(gemm / "gen" / "tileFirstVariants.inc").write_text(
			 '{oa::matmulVariantIdFromName("Good"), "Good", 1},\n'
			 '{oa::matmulVariantIdFromName("Wrong"), "Other", 2},\n'
			)
			(gemm / "gen" / "tileSecondVariants.inc").write_text(
			 '{oa::matmulVariantIdFromName("Good"), "Good", 3},\n'
			)
			variants, errors = auditKernels.loadMatmulVariants(root)

		self.assertEqual({"Good"}, set(variants))
		self.assertTrue(
		 any("identity Wrong names kernel Other" in error for error in errors)
		)
		self.assertTrue(
		 any("duplicate matmul variant identity Good" in error for error in errors)
		)

	def testReservedRangesRejectOverlapActiveReuseAndGaps(self) -> None:
		with tempfile.TemporaryDirectory() as temp:
			root = Path(temp)
			registry = root / "source/cpp/lib/oa/runtime/kernelRegistry.h"
			registry.parent.mkdir(parents=True)
			registry.write_text(
			 "{ oa::computeKernelPrefix::Ml, 2, 3 },\n"
			 "{ oa::computeKernelPrefix::Ml, 3, 3 },\n"
			)
			reserved, parseErrors = auditKernels.loadReservedRegistry(root)

		self.assertEqual({"Ml": {2, 3}}, reserved)
		self.assertTrue(any("overlapping reserved kernel ID Ml:3" in error for error in parseErrors))

		fixed = {
		 "First": auditKernels.FixedKernel(
		  "First", "Ml", 1, "Ml", "oa", Path("KernelRegistry.h")
		 ),
		 "Third": auditKernels.FixedKernel(
		  "Third", "Ml", 3, "Ml", "oa", Path("KernelRegistry.h")
		 ),
		 "Fifth": auditKernels.FixedKernel(
		  "Fifth", "Ml", 5, "Ml", "oa", Path("KernelRegistry.h")
		 ),
		}
		coverageErrors = auditKernels.validateReservedCoverage(fixed, reserved)
		self.assertTrue(any("active kernel reuses reserved ID Ml:3" in error for error in coverageErrors))
		self.assertTrue(any("unclassified kernel ID gap Ml:4" in error for error in coverageErrors))

	def testProductionScanIncludesGeneratedDispatchButNotTestsOrDeclarations(self) -> None:
		with tempfile.TemporaryDirectory() as temp:
			root = Path(temp)
			production = root / "source/cpp/lib/oa/feature"
			production.mkdir(parents=True)
			(production / "direct.cpp").write_text('Dispatch("Direct");\n')
			(production / "planner.gen.inc").write_text(
			 "return oa::computeKernelId::TypedId;\n"
			)
			sdk = root / "sdk/cpp/lib/ml"
			sdk.mkdir(parents=True)
			(sdk / "sdkDispatch.cpp").write_text('Dispatch("SdkOnly");\n')
			declaration = root / "source/cpp/lib/oa/runtime/gen/kernelRegistryExtra.inl"
			declaration.parent.mkdir(parents=True)
			declaration.write_text('{ "DeclaredOnly" };\n')
			test = root / "test/feature/testOnly.cpp"
			test.parent.mkdir(parents=True)
			test.write_text('Dispatch("TestOnly");\n')

			references = auditKernels.loadProductionReferences(
			 root,
			 {"Direct", "Typed", "SdkOnly", "DeclaredOnly", "TestOnly"},
			 {"TypedId": "Typed"},
			)

		self.assertEqual({"Direct", "Typed", "SdkOnly"}, set(references))
		self.assertEqual(
		 {Path("sdk/cpp/lib/ml/sdkDispatch.cpp")},
		 references["SdkOnly"],
		)
		self.assertEqual(
		 {Path("source/cpp/lib/oa/feature/planner.gen.inc")},
		 references["Typed"],
		)

	def testLiveFixedRegistryAndTypedIdsAreConsistent(self) -> None:
		fixed, fixedErrors = auditKernels.loadFixedRegistry(auditKernels.REPO_ROOT)
		_, typedErrors = auditKernels.loadTypedIds(auditKernels.REPO_ROOT, fixed)
		reserved, reservedErrors = auditKernels.loadReservedRegistry(
		 auditKernels.REPO_ROOT
		)
		coverageErrors = auditKernels.validateReservedCoverage(fixed, reserved)
		variants, variantErrors = auditKernels.loadMatmulVariants(
		 auditKernels.REPO_ROOT
		)
		self.assertEqual([], fixedErrors)
		self.assertEqual([], typedErrors)
		self.assertEqual([], reservedErrors)
		self.assertEqual([], coverageErrors)
		self.assertEqual([], variantErrors)
		self.assertGreater(len(variants), 0)


if __name__ == "__main__":
	unittest.main()
