// TestKernelRegistry.cpp — fixed kernel metadata and embedded SPIR-V registry tests.

#include <oa/runtime/kernelRegistry.h>
#include <oa/runtime/spirv.h>
#include <oa/runtime/computeKernel.h>
#include <oa/runtime/engine.h>
#include <oa/runtime/dispatch.h>
#include <oa/runtime/matmulTypes.h>
#include <oa/core/std/hashMap.h>
#include <oa/core/std/string.h>
#include <gtest/gtest.h>

#include <cstring>
#include <unordered_set>

// ============================================================================
// TEST FIXTURE
// ============================================================================

template <typename Fn>
void forEachBuiltInKernel(Fn&& inFn)
{
	for (const auto& kernel : oa::kernelRegistry::getMlKernels()) { inFn(kernel); }
	for (const auto& kernel : oa::kernelRegistry::getVisionKernels()) { inFn(kernel); }
	for (const auto& kernel : oa::kernelRegistry::getUiKernels()) { inFn(kernel); }
	for (const auto& kernel : oa::kernelRegistry::getAudioKernels()) { inFn(kernel); }
	for (const auto& kernel : oa::kernelRegistry::getRenderKernels()) { inFn(kernel); }
	for (const auto& kernel : oa::kernelRegistry::getCryptoKernels()) { inFn(kernel); }
}

static oa::Usize builtInKernelCount()
{
	return oa::kernelRegistry::getMlKernels().size() +
		oa::kernelRegistry::getVisionKernels().size() +
		oa::kernelRegistry::getUiKernels().size() +
		oa::kernelRegistry::getAudioKernels().size() +
		oa::kernelRegistry::getRenderKernels().size() +
		oa::kernelRegistry::getCryptoKernels().size();
}

class KernelRegistryTest : public ::testing::Test {};

// ============================================================================
// REGISTRY STRUCTURE TESTS
// ============================================================================

TEST_VK(KernelRegistryTest, RegistryNotEmpty) {
	auto ml_kernels = oa::kernelRegistry::getMlKernels();
	auto vision_kernels = oa::kernelRegistry::getVisionKernels();
	auto ui_kernels = oa::kernelRegistry::getUiKernels();
	auto audio_kernels = oa::kernelRegistry::getAudioKernels();
	auto render_kernels = oa::kernelRegistry::getRenderKernels();
	auto crypto_kernels = oa::kernelRegistry::getCryptoKernels();
	
	EXPECT_GT(ml_kernels.size(), 0) << "ML kernel registry should not be empty";
	EXPECT_GT(vision_kernels.size(), 0) << "Vision kernel registry should not be empty";
	EXPECT_GT(ui_kernels.size(), 0) << "UI kernel registry should not be empty";
	EXPECT_GT(audio_kernels.size(), 0) << "Audio kernel registry should not be empty";
	EXPECT_GT(render_kernels.size(), 0) << "Render kernel registry should not be empty";
	EXPECT_GT(crypto_kernels.size(), 0) << "Crypto kernel registry should not be empty";
	
	oa::Usize total = oa::kernelRegistry::getTotalKernelCount();
	EXPECT_EQ(total, builtInKernelCount())
		<< "total count should match sum of categories";
	
	std::printf("[KernelRegistry] total kernels: %zu (ML=%zu, Vision=%zu, UI=%zu, Audio=%zu, Render=%zu, Crypto=%zu)\n",
		total, ml_kernels.size(), vision_kernels.size(), ui_kernels.size(),
		audio_kernels.size(), render_kernels.size(), crypto_kernels.size());
}

TEST_VK(KernelRegistryTest, EmbeddedRegistryContainsKnownShader) {
	const oavk::SpirvEntry* entry = oavk::findSpirv("Add");
	ASSERT_NE(entry, nullptr);
	EXPECT_STREQ(entry->name, "Add");
	EXPECT_NE(entry->data, nullptr);
	EXPECT_GT(entry->size, 0U);
	EXPECT_NE(entry->contentHash, 0U);
}

TEST_VK(KernelRegistryTest, AllKernelsHaveValidIds) {
	forEachBuiltInKernel([](const oa::ComputeKernel& kernel) {
		EXPECT_TRUE(oa::computeKernelIdIsValid(kernel.id))
			<< "kernel '" << kernel.name << "' has invalid ID: " << kernel.id;
		EXPECT_NE(kernel.name, nullptr)
			<< "kernel has null name";
		EXPECT_GT(std::strlen(kernel.name), 0)
			<< "kernel has empty name";
	});
}

TEST_VK(KernelRegistryTest, NoIdCollisions) {
	std::unordered_set<oa::U64> seen_ids;
	oa::Usize collision_count = 0;
	
	forEachBuiltInKernel([&](const oa::ComputeKernel& kernel) {
		if (seen_ids.count(kernel.id) > 0) {
			std::printf("[ERROR] ID collision detected: 0x%016lX (%s)\n",
				kernel.id, kernel.name);
			collision_count++;
		}
		seen_ids.insert(kernel.id);
	});
	
	EXPECT_EQ(collision_count, 0) << "Found " << collision_count << " ID collisions";
}

TEST_VK(KernelRegistryTest, NoNameCollisions) {
	std::unordered_set<std::string> seen_names;
	oa::Usize collision_count = 0;
	
	forEachBuiltInKernel([&](const oa::ComputeKernel& kernel) {
		std::string name(kernel.name);
		if (seen_names.count(name) > 0) {
			std::printf("[ERROR] Name collision detected: %s\n", kernel.name);
			collision_count++;
		}
		seen_names.insert(name);
	});
	
	EXPECT_EQ(collision_count, 0) << "Found " << collision_count << " name collisions";
}

TEST_VK(KernelRegistryTest, PrefixesAreCorrect) {
	auto ml_kernels = oa::kernelRegistry::getMlKernels();
	auto crypto_kernels = oa::kernelRegistry::getCryptoKernels();
	
	// ML kernels should have ML prefix
	for (const auto& kernel : ml_kernels) {
		oa::U32 prefix = oa::computeKernelIdUnpackPrefix(kernel.id);
		EXPECT_EQ(prefix, oa::computeKernelPrefix::Ml) 
			<< "ML kernel '" << kernel.name << "' has wrong prefix: 0x" 
			<< std::hex << prefix;
	}
	
	// Vision kernels should have Vision prefix
	for (const auto& kernel : oa::kernelRegistry::getVisionKernels()) {
		oa::U32 prefix = oa::computeKernelIdUnpackPrefix(kernel.id);
		EXPECT_EQ(prefix, oa::computeKernelPrefix::Vision)
			<< "Vision kernel '" << kernel.name << "' has wrong prefix: 0x"
			<< std::hex << prefix;
	}

	// UI kernels should have UI prefix
	for (const auto& kernel : oa::kernelRegistry::getUiKernels()) {
		oa::U32 prefix = oa::computeKernelIdUnpackPrefix(kernel.id);
		EXPECT_EQ(prefix, oa::computeKernelPrefix::Ui)
			<< "UI kernel '" << kernel.name << "' has wrong prefix: 0x"
			<< std::hex << prefix;
	}

	for (const auto& kernel : oa::kernelRegistry::getAudioKernels()) {
		oa::U32 prefix = oa::computeKernelIdUnpackPrefix(kernel.id);
		EXPECT_EQ(prefix, oa::computeKernelPrefix::Audio)
			<< "Audio kernel '" << kernel.name << "' has wrong prefix: 0x"
			<< std::hex << prefix;
	}

	for (const auto& kernel : oa::kernelRegistry::getRenderKernels()) {
		oa::U32 prefix = oa::computeKernelIdUnpackPrefix(kernel.id);
		EXPECT_EQ(prefix, oa::computeKernelPrefix::Render)
			<< "Render kernel '" << kernel.name << "' has wrong prefix: 0x"
			<< std::hex << prefix;
	}

	// Crypto kernels should have Crypto prefix
	for (const auto& kernel : crypto_kernels) {
		oa::U32 prefix = oa::computeKernelIdUnpackPrefix(kernel.id);
		EXPECT_EQ(prefix, oa::computeKernelPrefix::Crypto) 
			<< "Crypto kernel '" << kernel.name << "' has wrong prefix: 0x" 
			<< std::hex << prefix;
	}
}

TEST_VK(KernelRegistryTest, LocalIdsAreValid) {
	// Check every fixed kernel has a plausible non-zero local ID.
	// Note: IDs don't need to be sequential in the array since registry
	// order may differ from ID order (e.g., for historical compatibility)
	forEachBuiltInKernel([](const oa::ComputeKernel& kernel) {
		oa::U32 local = oa::computeKernelIdUnpackLocal(kernel.id);
		EXPECT_GT(local, 0) << "kernel '" << kernel.name << "' has zero local ID";
		EXPECT_LT(local, 10000) << "kernel '" << kernel.name
			<< "' has unreasonably large local ID: " << local;
	});
}

// ============================================================================
// LOOKUP FUNCTION TESTS
// ============================================================================

TEST_VK(KernelRegistryTest, FindByPackedIdWorks) {
	auto ml_kernels = oa::kernelRegistry::getMlKernels();
	
	// Test finding each ML kernel by ID
	for (const auto& expected : ml_kernels) {
		const oa::ComputeKernel* found = oa::computeKernelFindByPackedId(expected.id);
		ASSERT_NE(found, nullptr) << "Failed to find kernel with ID: 0x" 
			<< std::hex << expected.id;
		EXPECT_EQ(found->id, expected.id);
		EXPECT_STREQ(found->name, expected.name);
	}
}

TEST_VK(KernelRegistryTest, FindByPackedIdReturnsNullForInvalidId) {
	// Test with completely invalid ID
	const oa::ComputeKernel* found = oa::computeKernelFindByPackedId(0xDEADBEEFDEADBEEF);
	EXPECT_EQ(found, nullptr) << "Should return null for invalid ID";
	
	// Test with zero ID
	found = oa::computeKernelFindByPackedId(0);
	EXPECT_EQ(found, nullptr) << "Should return null for zero ID";
}

TEST_VK(KernelRegistryTest, FindByPackedIdHandlesAllCategories) {
	auto ml_kernels = oa::kernelRegistry::getMlKernels();
	auto crypto_kernels = oa::kernelRegistry::getCryptoKernels();
	
	// Test ML kernel
	if (ml_kernels.size() > 0) {
		const auto& kernel = ml_kernels[0];
		const oa::ComputeKernel* found = oa::computeKernelFindByPackedId(kernel.id);
		ASSERT_NE(found, nullptr);
		EXPECT_EQ(found->category, oa::ComputeKernelCategory::Ml);
	}
	
	// Test crypto kernel
	if (crypto_kernels.size() > 0) {
		const auto& kernel = crypto_kernels[0];
		const oa::ComputeKernel* found = oa::computeKernelFindByPackedId(kernel.id);
		ASSERT_NE(found, nullptr);
		EXPECT_EQ(found->category, oa::ComputeKernelCategory::Crypto);
	}
}

TEST_VK(KernelRegistryTest, KnownKernelIdsAreValid) {
	// Test known kernel IDs from oa::computeKernelId namespace
	const oa::ComputeKernel* silu = oa::computeKernelFindByPackedId(oa::computeKernelId::Silu);
	ASSERT_NE(silu, nullptr) << "Failed to find Silu kernel";
	EXPECT_STREQ(silu->name, "Silu");
	
	const oa::ComputeKernel* gemmNaive = oa::computeKernelFindByPackedId(oa::computeKernelId::GemmNaive);
	ASSERT_NE(gemmNaive, nullptr) << "Failed to find GemmNaive kernel";
	EXPECT_STREQ(gemmNaive->name, "GemmNaive");
}

TEST_VK(KernelRegistryTest, SchemaOwnedLossIdsResolveExactNames) {
	struct ExpectedKernel {
		oa::U64 id;
		const char* name;
	};
	const ExpectedKernel expected[] = {
		{oa::computeKernelId::SmoothL1, "SmoothL1"},
		{oa::computeKernelId::SmoothL1Bwd, "SmoothL1Bwd"},
		{oa::computeKernelId::Mse, "Mse"},
		{oa::computeKernelId::MseBwd, "MseBwd"},
		{oa::computeKernelId::L1, "L1"},
		{oa::computeKernelId::L1Bwd, "L1Bwd"},
		{oa::computeKernelId::Bce, "Bce"},
		{oa::computeKernelId::BceBwd, "BceBwd"},
	};
	for (const auto& item : expected) {
		const oa::ComputeKernel* kernel = oa::computeKernelFindByPackedId(item.id);
		ASSERT_NE(kernel, nullptr) << item.name;
		EXPECT_STREQ(kernel->name, item.name);
	}
}

TEST_VK(KernelRegistryTest, ReassignedSchemaIdsResolveExactNames) {
	struct ExpectedKernel {
		oa::U64 id;
		const char* name;
	};
	const ExpectedKernel expected[] = {
		{oa::computeKernelId::CompactRows, "CompactRows"},
		{oa::computeKernelId::CompactRowsBwd, "CompactRowsBwd"},
		{oa::computeKernelId::ScatterRows, "ScatterRows"},
		{oa::computeKernelId::ScatterRowsBwd, "ScatterRowsBwd"},
		{oa::computeKernelId::GreaterEqual, "GreaterEqual"},
		{oa::computeKernelId::CategoricalAccuracyCount, "CategoricalAccuracyCount"},
		{oa::computeKernelId::MoeRoutingBiasUpdate, "MoeRoutingBiasUpdate"},
		{oa::computeKernelId::CausalMaskBwd, "CausalMaskBwd"},
		{oa::computeKernelId::SampleSortedLogits, "SampleSortedLogits"},
		{oa::computeKernelId::SampleDenseLogits, "SampleDenseLogits"},
		{oa::computeKernelId::MoeExpertPlan, "MoeExpertPlan"},
		{oa::computeKernelId::GroupedGemmM, "GroupedGemmM"},
		{oa::computeKernelId::GroupedGemmMDataBwd, "GroupedGemmMDataBwd"},
		{oa::computeKernelId::GroupedGemmMWeightBwd, "GroupedGemmMWeightBwd"},
		{oa::computeKernelId::GatherLastDim, "GatherLastDim"},
		{oa::computeKernelId::GatherLastDimBwd, "GatherLastDimBwd"},
		{oa::computeKernelId::GroupedLinearM, "GroupedLinearM"},
		{oa::computeKernelId::GroupedLinearMBiasBwd, "GroupedLinearMBiasBwd"},
		{oa::computeKernelId::MoeCombine, "MoeCombine"},
		{oa::computeKernelId::MoeCombineBwd, "MoeCombineBwd"},
	};
	oa::U32 expectedLocal = 305;
	for (const auto& item : expected) {
		EXPECT_EQ(oa::computeKernelIdUnpackLocal(item.id), expectedLocal++) << item.name;
		const oa::ComputeKernel* kernel = oa::computeKernelFindByPackedId(item.id);
		ASSERT_NE(kernel, nullptr) << item.name;
		EXPECT_STREQ(kernel->name, item.name);
	}
}

TEST_VK(KernelRegistryTest, SchemaOwnedOperationIdsResolveExactNames) {
	struct ExpectedKernel {
		oa::U64 id;
		const char* name;
	};
	const ExpectedKernel expected[] = {
		{oa::computeKernelId::Bmm, "Bmm"},
		{oa::computeKernelId::ChannelNorm, "ChannelNorm"},
		{oa::computeKernelId::ChannelNormBwd, "ChannelNormBwd"},
		{oa::computeKernelId::ChannelNormRelu, "ChannelNormRelu"},
		{oa::computeKernelId::ChannelNormReluBwd, "ChannelNormReluBwd"},
		{oa::computeKernelId::Col2Im1d, "Col2Im1d"},
		{oa::computeKernelId::Conv1dBwdData, "Conv1dBwdData"},
		{oa::computeKernelId::Conv1dBwdWeight, "Conv1dBwdWeight"},
		{oa::computeKernelId::Conv2dBwdData, "Conv2dBwdData"},
		{oa::computeKernelId::Conv2dBwdWeight, "Conv2dBwdWeight"},
		{oa::computeKernelId::ConvTranspose2dBwdWeight, "ConvTranspose2dBwdWeight"},
		{oa::computeKernelId::EmpyrealmSisoBwd, "EmpyrealmSisoBwd"},
		{oa::computeKernelId::EmpyrealmSisoFwd, "EmpyrealmSisoFwd"},
		{oa::computeKernelId::EmpyrealmSisoStep, "EmpyrealmSisoStep"},
		{oa::computeKernelId::GruCellLinear, "GruCellLinear"},
		{oa::computeKernelId::Im2Col1d, "Im2Col1d"},
		{oa::computeKernelId::Mamba3SisoBwd, "Mamba3SisoBwd"},
		{oa::computeKernelId::Mamba3SisoFwd, "Mamba3SisoFwd"},
		{oa::computeKernelId::Mamba3SisoStep, "Mamba3SisoStep"},
		{oa::computeKernelId::RepeatInterleave, "RepeatInterleave"},
		{oa::computeKernelId::RepeatInterleaveBwd, "RepeatInterleaveBwd"},
		{oa::computeKernelId::RmsNormGatedBwd, "RmsNormGatedBwd"},
		{oa::computeKernelId::RnnCellLinear, "RnnCellLinear"},
		{oa::computeKernelId::TopKMask, "TopKMask"},
		{oa::computeKernelId::VqAssign, "VqAssign"},
		{oa::computeKernelId::VqEmaUpdate, "VqEmaUpdate"},
	};
	oa::U32 expectedLocal = 325;
	for (const auto& item : expected) {
		EXPECT_EQ(oa::computeKernelIdUnpackLocal(item.id), expectedLocal++) << item.name;
		const oa::ComputeKernel* kernel = oa::computeKernelFindByPackedId(item.id);
		ASSERT_NE(kernel, nullptr) << item.name;
		EXPECT_STREQ(kernel->name, item.name);
	}
}

TEST_VK(KernelRegistryTest, DomainLoweringIdsResolveExactNames) {
	struct ExpectedKernel {
		oa::U64 id;
		const char* name;
		oa::U32 prefix;
		oa::U32 local;
	};
	const ExpectedKernel expected[] = {
		{oa::computeKernelId::ClipGradNormReduce, "ClipGradNormReduce", oa::computeKernelPrefix::Ml, 351},
		{oa::computeKernelId::ClipGradNormScale, "ClipGradNormScale", oa::computeKernelPrefix::Ml, 352},
		{oa::computeKernelId::MaskedMse, "MaskedMse", oa::computeKernelPrefix::Ml, 354},
		{oa::computeKernelId::MaskedMseBwd, "MaskedMseBwd", oa::computeKernelPrefix::Ml, 355},
		{oa::computeKernelId::RopeApply, "RopeApply", oa::computeKernelPrefix::Ml, 356},
		{oa::computeKernelId::RopeApplyBwd, "RopeApplyBwd", oa::computeKernelPrefix::Ml, 357},
		{oa::computeKernelId::SmoothL1Mean, "SmoothL1Mean", oa::computeKernelPrefix::Ml, 358},
		{oa::computeKernelId::SmoothL1MeanBwd, "SmoothL1MeanBwd", oa::computeKernelPrefix::Ml, 359},
		{oa::computeKernelId::VelSmoothL1, "VelSmoothL1", oa::computeKernelPrefix::Ml, 360},
		{oa::computeKernelId::VelSmoothL1Bwd, "VelSmoothL1Bwd", oa::computeKernelPrefix::Ml, 361},
		{oa::computeKernelId::Mamba3SisoBwdShortPrepare, "Mamba3SisoBwdShortPrepare", oa::computeKernelPrefix::Ml, 381},
		{oa::computeKernelId::Mamba3SisoBwdShortState, "Mamba3SisoBwdShortState", oa::computeKernelPrefix::Ml, 382},
		{oa::computeKernelId::Mamba3SisoBwdShortFinalize, "Mamba3SisoBwdShortFinalize", oa::computeKernelPrefix::Ml, 384},
		{oa::computeKernelId::ResizeBilinearBwd, "ResizeBilinearBwd", oa::computeKernelPrefix::Vision, 40},
		{oa::computeKernelId::ResizeNearestBwd, "ResizeNearestBwd", oa::computeKernelPrefix::Vision, 41},
		{oa::computeKernelId::ResizeNormalizeNchw, "ResizeNormalizeNchw", oa::computeKernelPrefix::Vision, 42},
		{oa::computeKernelId::ClearRgba8, "ClearRgba8", oa::computeKernelPrefix::Ui, 9},
		{oa::computeKernelId::DrawHeatmap, "DrawHeatmap", oa::computeKernelPrefix::Ui, 10},
		{oa::computeKernelId::DrawLine, "DrawLine", oa::computeKernelPrefix::Ui, 11},
		{oa::computeKernelId::DrawPlotLine, "DrawPlotLine", oa::computeKernelPrefix::Ui, 12},
		{oa::computeKernelId::MatrixToRgba8, "MatrixToRgba8", oa::computeKernelPrefix::Ui, 13},
	};
	for (const auto& item : expected) {
		EXPECT_EQ(oa::computeKernelIdUnpackPrefix(item.id), item.prefix) << item.name;
		EXPECT_EQ(oa::computeKernelIdUnpackLocal(item.id), item.local) << item.name;
		const oa::ComputeKernel* kernel = oa::computeKernelFindByPackedId(item.id);
		ASSERT_NE(kernel, nullptr) << item.name;
		EXPECT_STREQ(kernel->name, item.name);
	}
}

TEST_VK(KernelRegistryTest, ReservedIdsFailClosed) {
	for (const auto& range : oa::kernelRegistry::getReservedKernelIdRanges()) {
		for (oa::U32 local = range.firstLocal; local <= range.lastLocal; ++local) {
			const oa::U64 id = OA_COMPUTE_KERNEL_ID(range.prefix, local);
			EXPECT_EQ(oa::computeKernelFindByPackedId(id), nullptr)
				<< "reserved packed ID resolved: prefix=" << range.prefix
				<< " local=" << local;
		}
	}
}

// ============================================================================
// EMBEDDED SPIR-V REGISTRY INTEGRATION TESTS
// ============================================================================

TEST_VK(KernelRegistryTest, EmbeddedRegistryFindsRegisteredKernels) {
	oa::Usize checked_count = 0;
	forEachBuiltInKernel([&](const oa::ComputeKernel& kernel) {
		const oavk::SpirvEntry* entry = oavk::findSpirv(kernel.name);
		ASSERT_NE(entry, nullptr)
			<< "Fixed kernel has no embedded SPIR-V: " << kernel.name
			<< " (ID=0x" << std::hex << kernel.id << ")";
		EXPECT_STREQ(entry->name, kernel.name);
		EXPECT_NE(entry->data, nullptr) << kernel.name;
		EXPECT_GT(entry->size, 0U) << kernel.name;
		++checked_count;
	});
	EXPECT_EQ(checked_count, builtInKernelCount());
}

TEST_VK(KernelRegistryTest, StableIdsResolveEmbeddedShaders) {
	// Stable IDs resolve metadata; metadata names resolve the immutable build registry.
	forEachBuiltInKernel([](const oa::ComputeKernel& kernel) {
		const oa::ComputeKernel* metadata = oa::computeKernelFindByPackedId(kernel.id);
		ASSERT_NE(metadata, nullptr);
		const oavk::SpirvEntry* entry = oavk::findSpirv(metadata->name);
		ASSERT_NE(entry, nullptr) << metadata->name;
		EXPECT_STREQ(entry->name, kernel.name)
			<< "shader name mismatch for kernel ID: 0x" << std::hex << kernel.id;
	});
}

TEST_VK(KernelRegistryTest, EmbeddedRegistryIndexAndNameAgree) {
	const oa::U32 count = oavk::spirvCount();
	ASSERT_GT(count, 0U);
	std::unordered_set<std::string> names;
	for (oa::U32 index = 0; index < count; ++index) {
		const oavk::SpirvEntry* entry = oavk::findSpirvByIndex(index);
		ASSERT_NE(entry, nullptr) << "missing embedded entry at index " << index;
		ASSERT_NE(entry->name, nullptr);
		EXPECT_NE(entry->data, nullptr) << entry->name;
		EXPECT_GT(entry->size, 0U) << entry->name;
		EXPECT_NE(entry->contentHash, 0U) << entry->name;
		EXPECT_TRUE(names.insert(entry->name).second)
			<< "duplicate embedded shader name: " << entry->name;
		EXPECT_EQ(oavk::findSpirv(entry->name), entry);
	}
	EXPECT_EQ(oavk::findSpirvByIndex(count), nullptr);
}

TEST_VK(KernelRegistryTest, MatmulVariantsOwnExactEmbeddedCapabilityMetadata) {
	std::unordered_set<std::string> variantNames;
	for (const auto& variant : oa::matmulRegistry::all()) {
		ASSERT_NE(variant.kernelName, nullptr);
		EXPECT_TRUE(variantNames.insert(variant.kernelName).second)
			<< "duplicate matmul shader identity: " << variant.kernelName;
		EXPECT_EQ(oa::matmulRegistry::findByShaderName(variant.kernelName), &variant);
		EXPECT_NE(variant.requiredCapsMask, 0U)
			<< "matmul variant has no explicit capability contract: " << variant.kernelName;
		EXPECT_NE(oavk::findSpirv(variant.kernelName), nullptr)
			<< "matmul capability metadata refers to an unembedded shader: "
			<< variant.kernelName;
	}

	EXPECT_EQ(oa::matmulRegistry::findByShaderName(nullptr), nullptr);
	EXPECT_EQ(oa::matmulRegistry::findByShaderName(""), nullptr);
	EXPECT_EQ(oa::matmulRegistry::findByShaderName("GemmCmSgBf16_suffix"), nullptr)
		<< "generated shader identity must be exact, not a guessed prefix";

	const oa::U64 optionalCaps =
		oa::kCapCoopMat1Khr |
		oa::kCapCoopMat1Bf16Input |
		oa::kCapCoopMat1Bf16Acc |
		oa::kCapCoopMat1Fp16Input |
		oa::kCapCoopMat1Fp32Acc |
		oa::kCapCoopMat1WorkgroupBf16 |
		oa::kCapCoopVec;
	for (oa::U32 index = 0; index < oavk::spirvCount(); ++index) {
		const oavk::SpirvEntry* entry = oavk::findSpirvByIndex(index);
		ASSERT_NE(entry, nullptr);
		const std::string name(entry->name);
		const bool featureSensitive =
			name.find("CoopMat") != std::string::npos or
			name.find("CoopVec") != std::string::npos or
			name.find("CmSg") != std::string::npos or
			name.find("CmWg") != std::string::npos;
		if (!featureSensitive) continue;

		const auto* variant = oa::matmulRegistry::findByShaderName(entry->name);
		ASSERT_NE(variant, nullptr)
			<< "feature-sensitive embedded shader lacks exact capability metadata: "
			<< entry->name;
		EXPECT_NE(variant->requiredCapsMask & optionalCaps, 0U)
			<< "feature-sensitive embedded shader has only fallback capabilities: "
			<< entry->name;
	}
}

// ============================================================================
// EDGE CASES & ROBUSTNESS TESTS
// ============================================================================

TEST_VK(KernelRegistryTest, HandleNullPointers) {
	// FindByPackedId should handle invalid IDs gracefully
	const oa::ComputeKernel* result = oa::computeKernelFindByPackedId(0);
	EXPECT_EQ(result, nullptr);
	
	EXPECT_EQ(oavk::findSpirv(nullptr), nullptr);
}

TEST_VK(KernelRegistryTest, HandleEmptyStrings) {
	const oavk::SpirvEntry* entry = oavk::findSpirv("");
	EXPECT_EQ(entry, nullptr) << "Should return null for empty shader name";
}

TEST_VK(KernelRegistryTest, HandleNonExistentKernelNames) {
	const char* fake_names[] = {
		"NonExistentKernel",
		"FakeShader123",
		"ThisDoesNotExist",
		"InvalidKernelName"
	};
	
	for (const char* name : fake_names) {
		const oavk::SpirvEntry* entry = oavk::findSpirv(name);
		EXPECT_EQ(entry, nullptr) << "Should return null for non-existent kernel: " << name;
	}
}

TEST_VK(KernelRegistryTest, IdUnpackingIsConsistent) {
	auto ml_kernels = oa::kernelRegistry::getMlKernels();
	
	for (const auto& kernel : ml_kernels) {
		oa::U32 prefix = oa::computeKernelIdUnpackPrefix(kernel.id);
		oa::U32 local = oa::computeKernelIdUnpackLocal(kernel.id);
		
		// Reconstruct the ID
		oa::U64 reconstructed = OA_COMPUTE_KERNEL_ID(prefix, local);
		
		EXPECT_EQ(reconstructed, kernel.id) 
			<< "ID unpacking/repacking failed for kernel: " << kernel.name
			<< " (original=0x" << std::hex << kernel.id 
			<< ", reconstructed=0x" << reconstructed << ")";
	}
}

TEST_VK(KernelRegistryTest, CategoryEnumIsValid) {
	forEachBuiltInKernel([](const oa::ComputeKernel& kernel) {
		EXPECT_NE(kernel.category, oa::ComputeKernelCategory::None) 
			<< "kernel '" << kernel.name << "' has None category";
	});
}

TEST_VK(KernelRegistryTest, OriginTagsAreValid) {
	forEachBuiltInKernel([](const oa::ComputeKernel& kernel) {
		EXPECT_NE(kernel.origin, nullptr) 
			<< "kernel '" << kernel.name << "' has null origin";
		EXPECT_GT(std::strlen(kernel.origin), 0) 
			<< "kernel '" << kernel.name << "' has empty origin";
	});
}

// ============================================================================
// PERFORMANCE & STRESS TESTS
// ============================================================================

TEST_VK(KernelRegistryTest, LookupPerformance) {
	auto ml_kernels = oa::kernelRegistry::getMlKernels();
	
	// Perform many lookups to test performance
	constexpr int iterations = 10000;
	
	auto start = std::chrono::high_resolution_clock::now();
	
	for (int i = 0; i < iterations; i++) {
		for (const auto& kernel : ml_kernels) {
			const oa::ComputeKernel* found = oa::computeKernelFindByPackedId(kernel.id);
			(void)found; // Suppress unused warning
		}
	}
	
	auto end = std::chrono::high_resolution_clock::now();
	auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
	
	double lookups_per_sec = (iterations * ml_kernels.size() * 1000000.0) / duration.count();
	
	std::printf("[Performance] %d iterations × %zu kernels = %.2f M lookups/sec\n", 
		iterations, ml_kernels.size(), lookups_per_sec / 1000000.0);
	
	// Should be able to do at least 1M lookups per second
	EXPECT_GT(lookups_per_sec, 1000000.0) 
		<< "lookup performance is too slow: " << lookups_per_sec << " lookups/sec";
}

TEST_VK(KernelRegistryTest, ConcurrentLookups) {
	// Test that lookups are thread-safe (registry is const, so should be safe)
	auto ml_kernels = oa::kernelRegistry::getMlKernels();
	
	ASSERT_GT(ml_kernels.size(), 0U);
	
	std::atomic<int> success_count{0};
	std::atomic<int> failure_count{0};
	
	auto lookup_thread = [&]() {
		for (int i = 0; i < 1000; i++) {
			for (const auto& kernel : ml_kernels) {
				const oa::ComputeKernel* found = oa::computeKernelFindByPackedId(kernel.id);
				if (found != nullptr && found->id == kernel.id) {
					success_count++;
				} else {
					failure_count++;
				}
			}
		}
	};
	
	// Launch multiple threads
	std::vector<std::thread> threads;
	for (int i = 0; i < 4; i++) {
		threads.emplace_back(lookup_thread);
	}
	
	// wait for all threads
	for (auto& t : threads) {
		t.join();
	}
	
	std::printf("[Concurrency] Success: %d, Failures: %d\n", 
		success_count.load(), failure_count.load());
	
	EXPECT_EQ(failure_count.load(), 0) << "Concurrent lookups had failures";
	EXPECT_GT(success_count.load(), 0) << "No successful concurrent lookups";
}
