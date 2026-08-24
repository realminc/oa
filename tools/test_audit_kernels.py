#!/usr/bin/env python3
"""Regression tests for the configured shader/fixed-ID audit."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path


TOOLS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(TOOLS_DIR))

import audit_kernels  # noqa: E402


class KernelAuditTest(unittest.TestCase):
	def test_manifest_rejects_duplicates_suffixes_and_parent_traversal(self) -> None:
		with tempfile.TemporaryDirectory() as temp:
			build = Path(temp)
			(build / audit_kernels.MANIFEST_NAME).write_text(
				"Good\nGood\nBad.spv\n../Escape\nNested/../Escape\n"
			)
			names, errors = audit_kernels.load_manifest(build)

		self.assertEqual(names[0], "Good")
		self.assertTrue(any("duplicate manifest key: Good" in error for error in errors))
		self.assertTrue(any("must omit .spv" in error for error in errors))
		self.assertEqual(
			2,
			sum("escapes the SPIR-V root" in error for error in errors),
		)

	def test_schema_kernel_references_are_partitioned_by_owner(self) -> None:
		with tempfile.TemporaryDirectory() as temp:
			root = Path(temp)
			schema = root / "tools/fnAutogen/schema/ml"
			schema.mkdir(parents=True)
			(schema / "core.toml").write_text(
				'[[ops]]\nname = "coreOp"\nkernel_main = "CoreKernel"\n'
			)
			(schema / "sdk.toml").write_text(
				'owner = "sdk"\n[[ops]]\nname = "sdkOp"\n'
				'kernel_main = "SdkKernel"\n'
			)

			core = audit_kernels.load_schema_kernel_references(root, "core")
			sdk = audit_kernels.load_schema_kernel_references(root, "sdk")

		self.assertEqual({"CoreKernel"}, set(core))
		self.assertEqual({"SdkKernel"}, set(sdk))

	def test_registry_rejects_duplicate_names_and_packed_ids(self) -> None:
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
			fixed, errors = audit_kernels.load_fixed_registry(root)

		self.assertEqual({"First", "Second"}, set(fixed))
		self.assertTrue(any("duplicate fixed name First" in error for error in errors))
		self.assertTrue(any("duplicate fixed ID Ml:1" in error for error in errors))

	def test_typed_id_requires_a_fixed_registry_row(self) -> None:
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
				"Present": audit_kernels.FixedKernel(
					"Present", "Ml", 1, "Ml", "oa", Path("KernelRegistry.h")
				)
			}
			typed, errors = audit_kernels.load_typed_ids(root, fixed)

		self.assertEqual({"Present": "Present"}, typed)
		self.assertTrue(any("Missing" in error and "Ml:7" in error for error in errors))

	def test_typed_id_name_must_match_its_fixed_registry_row(self) -> None:
		with tempfile.TemporaryDirectory() as temp:
			root = Path(temp)
			public = root / "source/cpp/include/oa/runtime"
			public.mkdir(parents=True)
			(public / "computeKernel.h").write_text(
				"static constexpr oa::U64 Claimed = "
				"OA_COMPUTE_KERNEL_ID(oa::computeKernelPrefix::Ml, 1);\n"
			)
			fixed = {
				"Actual": audit_kernels.FixedKernel(
					"Actual", "Ml", 1, "Ml", "oa", Path("KernelRegistry.h")
				)
			}
			typed, errors = audit_kernels.load_typed_ids(root, fixed)

		self.assertEqual({}, typed)
		self.assertTrue(
			any("Claimed resolves to fixed row Actual" in error for error in errors)
		)

	def test_matmul_variant_identity_rejects_name_mismatch_and_duplicates(self) -> None:
		with tempfile.TemporaryDirectory() as temp:
			root = Path(temp)
			gemm = root / "source/cpp/lib/oa/runtime/gemm"
			gemm.mkdir(parents=True)
			(gemm / "oaTileFirstVariants.gen.inc").write_text(
				'{oa::matmulVariantIdFromName("Good"), "Good", 1},\n'
				'{oa::matmulVariantIdFromName("Wrong"), "Other", 2},\n'
			)
			(gemm / "oaTileSecondVariants.gen.inc").write_text(
				'{oa::matmulVariantIdFromName("Good"), "Good", 3},\n'
			)
			variants, errors = audit_kernels.load_matmul_variants(root)

		self.assertEqual({"Good"}, set(variants))
		self.assertTrue(
			any("identity Wrong names kernel Other" in error for error in errors)
		)
		self.assertTrue(
			any("duplicate matmul variant identity Good" in error for error in errors)
		)

	def test_reserved_ranges_reject_overlap_active_reuse_and_gaps(self) -> None:
		with tempfile.TemporaryDirectory() as temp:
			root = Path(temp)
			registry = root / "source/cpp/lib/oa/runtime/kernelRegistry.h"
			registry.parent.mkdir(parents=True)
			registry.write_text(
				"{ oa::computeKernelPrefix::Ml, 2, 3 },\n"
				"{ oa::computeKernelPrefix::Ml, 3, 3 },\n"
			)
			reserved, parse_errors = audit_kernels.load_reserved_registry(root)

		self.assertEqual({"Ml": {2, 3}}, reserved)
		self.assertTrue(any("overlapping reserved kernel ID Ml:3" in error for error in parse_errors))

		fixed = {
			"First": audit_kernels.FixedKernel(
				"First", "Ml", 1, "Ml", "oa", Path("KernelRegistry.h")
			),
			"Third": audit_kernels.FixedKernel(
				"Third", "Ml", 3, "Ml", "oa", Path("KernelRegistry.h")
			),
			"Fifth": audit_kernels.FixedKernel(
				"Fifth", "Ml", 5, "Ml", "oa", Path("KernelRegistry.h")
			),
		}
		coverage_errors = audit_kernels.validate_reserved_coverage(fixed, reserved)
		self.assertTrue(any("active kernel reuses reserved ID Ml:3" in error for error in coverage_errors))
		self.assertTrue(any("unclassified kernel ID gap Ml:4" in error for error in coverage_errors))

	def test_production_scan_includes_generated_dispatch_but_not_tests_or_declarations(self) -> None:
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
			declaration = root / "source/cpp/lib/oa/runtime/kernelRegistryExtra.gen.inl"
			declaration.parent.mkdir(parents=True)
			declaration.write_text('{ "DeclaredOnly" };\n')
			test = root / "test/feature/testOnly.cpp"
			test.parent.mkdir(parents=True)
			test.write_text('Dispatch("TestOnly");\n')

			references = audit_kernels.load_production_references(
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

	def test_live_fixed_registry_and_typed_ids_are_consistent(self) -> None:
		fixed, fixed_errors = audit_kernels.load_fixed_registry(audit_kernels.REPO_ROOT)
		_, typed_errors = audit_kernels.load_typed_ids(audit_kernels.REPO_ROOT, fixed)
		reserved, reserved_errors = audit_kernels.load_reserved_registry(
			audit_kernels.REPO_ROOT
		)
		coverage_errors = audit_kernels.validate_reserved_coverage(fixed, reserved)
		variants, variant_errors = audit_kernels.load_matmul_variants(
			audit_kernels.REPO_ROOT
		)
		self.assertEqual([], fixed_errors)
		self.assertEqual([], typed_errors)
		self.assertEqual([], reserved_errors)
		self.assertEqual([], coverage_errors)
		self.assertEqual([], variant_errors)
		self.assertGreater(len(variants), 0)


if __name__ == "__main__":
	unittest.main()
