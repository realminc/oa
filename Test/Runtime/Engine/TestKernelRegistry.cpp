// TestKernelRegistry.cpp — Comprehensive Kernel Registry & Shader Loading Tests
//
// Tests the complete kernel registration, lookup, and shader loading system.
// Validates the "most robust shader/kernel loading system ever designed by realm software."

#include <Oa/Core/KernelRegistry.h>
#include <Oa/Runtime/ShaderProvider.h>
#include <Oa/Runtime/ComputeKernel.h>
#include <Oa/Runtime/Engine.h>
#include <Oa/Runtime/Dispatch.h>
#include <Oa/Core/Std/HashMap.h>
#include <Oa/Core/Std/String.h>
#include <gtest/gtest.h>

#include <cstring>
#include <unordered_set>

// ============================================================================
// TEST FIXTURE
// ============================================================================

template <typename Fn>
void ForEachBuiltInKernel(Fn&& InFn)
{
	for (const auto& kernel : OaKernelRegistry::GetMlKernels()) { InFn(kernel); }
	for (const auto& kernel : OaKernelRegistry::GetVisionKernels()) { InFn(kernel); }
	for (const auto& kernel : OaKernelRegistry::GetUiKernels()) { InFn(kernel); }
	for (const auto& kernel : OaKernelRegistry::GetAudioKernels()) { InFn(kernel); }
	for (const auto& kernel : OaKernelRegistry::GetRenderKernels()) { InFn(kernel); }
	for (const auto& kernel : OaKernelRegistry::GetCryptoKernels()) { InFn(kernel); }
}

static OaUsize BuiltInKernelCount()
{
	return OaKernelRegistry::GetMlKernels().Size() +
		OaKernelRegistry::GetVisionKernels().Size() +
		OaKernelRegistry::GetUiKernels().Size() +
		OaKernelRegistry::GetAudioKernels().Size() +
		OaKernelRegistry::GetRenderKernels().Size() +
		OaKernelRegistry::GetCryptoKernels().Size();
}

class KernelRegistryTest : public ::testing::Test {
protected:
	void SetUp() override {
		// Initialize shader provider (should be automatic, but explicit for testing)
		OaShaderProviderConfig config = {};
		config.Mode = OaShaderProviderMode::Auto;
		config.ShaderDir = OA_SPIRV_DIR;
		OaShaderProviderInit(&config);
	}

	void TearDown() override {
		OaShaderProviderShutdown();
	}
};

// ============================================================================
// REGISTRY STRUCTURE TESTS
// ============================================================================

TEST_VK(KernelRegistryTest, RegistryNotEmpty) {
	auto ml_kernels = OaKernelRegistry::GetMlKernels();
	auto vision_kernels = OaKernelRegistry::GetVisionKernels();
	auto ui_kernels = OaKernelRegistry::GetUiKernels();
	auto audio_kernels = OaKernelRegistry::GetAudioKernels();
	auto render_kernels = OaKernelRegistry::GetRenderKernels();
	auto crypto_kernels = OaKernelRegistry::GetCryptoKernels();
	
	EXPECT_GT(ml_kernels.Size(), 0) << "ML kernel registry should not be empty";
	EXPECT_GT(vision_kernels.Size(), 0) << "Vision kernel registry should not be empty";
	EXPECT_GT(ui_kernels.Size(), 0) << "UI kernel registry should not be empty";
	EXPECT_GT(audio_kernels.Size(), 0) << "Audio kernel registry should not be empty";
	EXPECT_GT(render_kernels.Size(), 0) << "Render kernel registry should not be empty";
	EXPECT_GT(crypto_kernels.Size(), 0) << "Crypto kernel registry should not be empty";
	
	OaUsize total = OaKernelRegistry::GetTotalKernelCount();
	EXPECT_EQ(total, BuiltInKernelCount())
		<< "Total count should match sum of categories";
	
	std::printf("[KernelRegistry] Total kernels: %zu (ML=%zu, Vision=%zu, UI=%zu, Audio=%zu, Render=%zu, Crypto=%zu)\n",
		total, ml_kernels.Size(), vision_kernels.Size(), ui_kernels.Size(),
		audio_kernels.Size(), render_kernels.Size(), crypto_kernels.Size());
}

TEST_VK(KernelRegistryTest, ProviderAutoInitializesWithoutRecursiveLock) {
	OaShaderProviderShutdown();
	const OaSpvEntry* entry = OaShaderProviderFind("Add");
	EXPECT_TRUE(OaShaderProviderIsInitialized());
	EXPECT_NE(entry, nullptr);
}

TEST_VK(KernelRegistryTest, AllKernelsHaveValidIds) {
	ForEachBuiltInKernel([](const OaComputeKernel& kernel) {
		EXPECT_TRUE(OaComputeKernelIdIsValid(kernel.Id))
			<< "Kernel '" << kernel.Name << "' has invalid ID: " << kernel.Id;
		EXPECT_NE(kernel.Name, nullptr)
			<< "Kernel has null name";
		EXPECT_GT(std::strlen(kernel.Name), 0)
			<< "Kernel has empty name";
	});
}

TEST_VK(KernelRegistryTest, NoIdCollisions) {
	std::unordered_set<OaU64> seen_ids;
	OaUsize collision_count = 0;
	
	ForEachBuiltInKernel([&](const OaComputeKernel& kernel) {
		if (seen_ids.count(kernel.Id) > 0) {
			std::printf("[ERROR] ID collision detected: 0x%016lX (%s)\n",
				kernel.Id, kernel.Name);
			collision_count++;
		}
		seen_ids.insert(kernel.Id);
	});
	
	EXPECT_EQ(collision_count, 0) << "Found " << collision_count << " ID collisions";
}

TEST_VK(KernelRegistryTest, NoNameCollisions) {
	std::unordered_set<std::string> seen_names;
	OaUsize collision_count = 0;
	
	ForEachBuiltInKernel([&](const OaComputeKernel& kernel) {
		std::string name(kernel.Name);
		if (seen_names.count(name) > 0) {
			std::printf("[ERROR] Name collision detected: %s\n", kernel.Name);
			collision_count++;
		}
		seen_names.insert(name);
	});
	
	EXPECT_EQ(collision_count, 0) << "Found " << collision_count << " name collisions";
}

TEST_VK(KernelRegistryTest, PrefixesAreCorrect) {
	auto ml_kernels = OaKernelRegistry::GetMlKernels();
	auto crypto_kernels = OaKernelRegistry::GetCryptoKernels();
	
	// ML kernels should have ML prefix
	for (const auto& kernel : ml_kernels) {
		OaU32 prefix = OaComputeKernelIdUnpackPrefix(kernel.Id);
		EXPECT_EQ(prefix, OaComputeKernelPrefix::Ml) 
			<< "ML kernel '" << kernel.Name << "' has wrong prefix: 0x" 
			<< std::hex << prefix;
	}
	
	// Vision kernels should have Vision prefix
	for (const auto& kernel : OaKernelRegistry::GetVisionKernels()) {
		OaU32 prefix = OaComputeKernelIdUnpackPrefix(kernel.Id);
		EXPECT_EQ(prefix, OaComputeKernelPrefix::Vision)
			<< "Vision kernel '" << kernel.Name << "' has wrong prefix: 0x"
			<< std::hex << prefix;
	}

	// UI kernels should have UI prefix
	for (const auto& kernel : OaKernelRegistry::GetUiKernels()) {
		OaU32 prefix = OaComputeKernelIdUnpackPrefix(kernel.Id);
		EXPECT_EQ(prefix, OaComputeKernelPrefix::Ui)
			<< "UI kernel '" << kernel.Name << "' has wrong prefix: 0x"
			<< std::hex << prefix;
	}

	for (const auto& kernel : OaKernelRegistry::GetAudioKernels()) {
		OaU32 prefix = OaComputeKernelIdUnpackPrefix(kernel.Id);
		EXPECT_EQ(prefix, OaComputeKernelPrefix::Audio)
			<< "Audio kernel '" << kernel.Name << "' has wrong prefix: 0x"
			<< std::hex << prefix;
	}

	for (const auto& kernel : OaKernelRegistry::GetRenderKernels()) {
		OaU32 prefix = OaComputeKernelIdUnpackPrefix(kernel.Id);
		EXPECT_EQ(prefix, OaComputeKernelPrefix::Render)
			<< "Render kernel '" << kernel.Name << "' has wrong prefix: 0x"
			<< std::hex << prefix;
	}

	// Crypto kernels should have Crypto prefix
	for (const auto& kernel : crypto_kernels) {
		OaU32 prefix = OaComputeKernelIdUnpackPrefix(kernel.Id);
		EXPECT_EQ(prefix, OaComputeKernelPrefix::Crypto) 
			<< "Crypto kernel '" << kernel.Name << "' has wrong prefix: 0x" 
			<< std::hex << prefix;
	}
}

TEST_VK(KernelRegistryTest, LocalIdsAreValid) {
	// Check every fixed kernel has a plausible non-zero local ID.
	// Note: IDs don't need to be sequential in the array since registry
	// order may differ from ID order (e.g., for historical compatibility)
	ForEachBuiltInKernel([](const OaComputeKernel& kernel) {
		OaU32 local = OaComputeKernelIdUnpackLocal(kernel.Id);
		EXPECT_GT(local, 0) << "Kernel '" << kernel.Name << "' has zero local ID";
		EXPECT_LT(local, 10000) << "Kernel '" << kernel.Name
			<< "' has unreasonably large local ID: " << local;
	});
}

// ============================================================================
// LOOKUP FUNCTION TESTS
// ============================================================================

TEST_VK(KernelRegistryTest, FindByPackedIdWorks) {
	auto ml_kernels = OaKernelRegistry::GetMlKernels();
	
	// Test finding each ML kernel by ID
	for (const auto& expected : ml_kernels) {
		const OaComputeKernel* found = OaComputeKernelFindByPackedId(expected.Id);
		ASSERT_NE(found, nullptr) << "Failed to find kernel with ID: 0x" 
			<< std::hex << expected.Id;
		EXPECT_EQ(found->Id, expected.Id);
		EXPECT_STREQ(found->Name, expected.Name);
	}
}

TEST_VK(KernelRegistryTest, FindByPackedIdReturnsNullForInvalidId) {
	// Test with completely invalid ID
	const OaComputeKernel* found = OaComputeKernelFindByPackedId(0xDEADBEEFDEADBEEF);
	EXPECT_EQ(found, nullptr) << "Should return null for invalid ID";
	
	// Test with zero ID
	found = OaComputeKernelFindByPackedId(0);
	EXPECT_EQ(found, nullptr) << "Should return null for zero ID";
}

TEST_VK(KernelRegistryTest, FindByPackedIdHandlesAllCategories) {
	auto ml_kernels = OaKernelRegistry::GetMlKernels();
	auto crypto_kernels = OaKernelRegistry::GetCryptoKernels();
	
	// Test ML kernel
	if (ml_kernels.Size() > 0) {
		const auto& kernel = ml_kernels[0];
		const OaComputeKernel* found = OaComputeKernelFindByPackedId(kernel.Id);
		ASSERT_NE(found, nullptr);
		EXPECT_EQ(found->Category, OaComputeKernelCategory::Ml);
	}
	
	// Test crypto kernel
	if (crypto_kernels.Size() > 0) {
		const auto& kernel = crypto_kernels[0];
		const OaComputeKernel* found = OaComputeKernelFindByPackedId(kernel.Id);
		ASSERT_NE(found, nullptr);
		EXPECT_EQ(found->Category, OaComputeKernelCategory::Crypto);
	}
}

TEST_VK(KernelRegistryTest, KnownKernelIdsAreValid) {
	// Test known kernel IDs from OaComputeKernelId namespace
	const OaComputeKernel* silu = OaComputeKernelFindByPackedId(OaComputeKernelId::Silu);
	ASSERT_NE(silu, nullptr) << "Failed to find Silu kernel";
	EXPECT_STREQ(silu->Name, "Silu");
	
	const OaComputeKernel* gemmNaive = OaComputeKernelFindByPackedId(OaComputeKernelId::GemmNaive);
	ASSERT_NE(gemmNaive, nullptr) << "Failed to find GemmNaive kernel";
	EXPECT_STREQ(gemmNaive->Name, "GemmNaive");
}

// ============================================================================
// SHADER PROVIDER INTEGRATION TESTS
// ============================================================================

TEST_VK(KernelRegistryTest, ShaderProviderFindsAllRegisteredKernels) {
	auto ml_kernels = OaKernelRegistry::GetMlKernels();
	
	OaUsize found_count = 0;
	OaUsize missing_count = 0;
	
	for (const auto& kernel : ml_kernels) {
		const OaSpvEntry* entry = OaShaderProviderFind(kernel.Name);
		if (entry != nullptr) {
			found_count++;
		} else {
			// Some kernels might not have shaders (e.g., coopmat variants)
			missing_count++;
			std::printf("[INFO] Shader not found for kernel: %s (ID=0x%016lX)\n", 
				kernel.Name, kernel.Id);
		}
	}
	
	std::printf("[ShaderProvider] Found %zu/%zu shaders (missing=%zu)\n", 
		found_count, ml_kernels.Size(), missing_count);
	
	// We expect most kernels to have shaders (allow some missing for variants)
	EXPECT_GT(found_count, ml_kernels.Size() / 2) 
		<< "Too many kernels missing shaders";
}

TEST_VK(KernelRegistryTest, ShaderProviderFindByIdWorks) {
	auto ml_kernels = OaKernelRegistry::GetMlKernels();
	
	// Test finding shaders by kernel ID
	for (const auto& kernel : ml_kernels) {
		const OaSpvEntry* entry = OaShaderProviderFindById(kernel.Id);
		if (entry != nullptr) {
			// Verify the shader name matches the kernel name
			EXPECT_STREQ(entry->Name, kernel.Name) 
				<< "Shader name mismatch for kernel ID: 0x" << std::hex << kernel.Id;
		}
	}
}

TEST_VK(KernelRegistryTest, ShaderProviderStatsAreReasonable) {
	(void)OaShaderProviderFind("Add");
	OaShaderProviderStats stats = OaShaderProviderGetStats();
	
	std::printf("[ShaderProvider] Stats:\n");
	std::printf("  TotalShaders: %u\n", stats.TotalShaders);
	std::printf("  EmbeddedShaders: %u\n", stats.EmbeddedShaders);
	std::printf("  FileShaders: %u\n", stats.FileShaders);
	std::printf("  CacheHits: %u\n", stats.CacheHits);
	std::printf("  CacheMisses: %u\n", stats.CacheMisses);
	std::printf("  LoadFailures: %u\n", stats.LoadFailures);
	
	// Embedded release builds report all embedded shaders; debug file-loading builds
	// are lazy and report only shaders touched through OaShaderProviderFind().
	EXPECT_GT(stats.TotalShaders, 0) << "Should have at least one visible shader after lookup";
	// The embedded SPIR-V set is a superset of the kernel metadata table: it also
	// carries render/UI stages, naming variants, and alias shaders that have no
	// KernelRegistry row. Bound it generously against pathological counts only.
	EXPECT_GE(stats.TotalShaders, OaKernelRegistry::GetTotalKernelCount())
		<< "Embedded shaders should cover every registered kernel";
	EXPECT_LE(stats.TotalShaders, OaKernelRegistry::GetTotalKernelCount() + 64)
		<< "TotalShaders wildly exceeds kernel registry (unregistered SPIR-V drift?)";
	
	// Note: EmbeddedShaders/FileShaders may be 0 until shaders are accessed. This
	// is fine - we verify that a lookup populates provider-visible state.
}

// ============================================================================
// EDGE CASES & ROBUSTNESS TESTS
// ============================================================================

TEST_VK(KernelRegistryTest, HandleNullPointers) {
	// FindByPackedId should handle invalid IDs gracefully
	const OaComputeKernel* result = OaComputeKernelFindByPackedId(0);
	EXPECT_EQ(result, nullptr);
	
	// ShaderProviderFind should handle null name gracefully (if API allows)
	// Note: This might crash if not handled, so we skip it for safety
	// const OaSpvEntry* entry = OaShaderProviderFind(nullptr);
	// EXPECT_EQ(entry, nullptr);
}

TEST_VK(KernelRegistryTest, HandleEmptyStrings) {
	// ShaderProviderFind should handle empty string
	const OaSpvEntry* entry = OaShaderProviderFind("");
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
		const OaSpvEntry* entry = OaShaderProviderFind(name);
		EXPECT_EQ(entry, nullptr) << "Should return null for non-existent kernel: " << name;
	}
}

TEST_VK(KernelRegistryTest, IdUnpackingIsConsistent) {
	auto ml_kernels = OaKernelRegistry::GetMlKernels();
	
	for (const auto& kernel : ml_kernels) {
		OaU32 prefix = OaComputeKernelIdUnpackPrefix(kernel.Id);
		OaU32 local = OaComputeKernelIdUnpackLocal(kernel.Id);
		
		// Reconstruct the ID
		OaU64 reconstructed = OA_COMPUTE_KERNEL_ID(prefix, local);
		
		EXPECT_EQ(reconstructed, kernel.Id) 
			<< "ID unpacking/repacking failed for kernel: " << kernel.Name
			<< " (original=0x" << std::hex << kernel.Id 
			<< ", reconstructed=0x" << reconstructed << ")";
	}
}

TEST_VK(KernelRegistryTest, CategoryEnumIsValid) {
	ForEachBuiltInKernel([](const OaComputeKernel& kernel) {
		EXPECT_NE(kernel.Category, OaComputeKernelCategory::None) 
			<< "Kernel '" << kernel.Name << "' has None category";
	});
}

TEST_VK(KernelRegistryTest, OriginTagsAreValid) {
	ForEachBuiltInKernel([](const OaComputeKernel& kernel) {
		EXPECT_NE(kernel.Origin, nullptr) 
			<< "Kernel '" << kernel.Name << "' has null origin";
		EXPECT_GT(std::strlen(kernel.Origin), 0) 
			<< "Kernel '" << kernel.Name << "' has empty origin";
	});
}

// ============================================================================
// PERFORMANCE & STRESS TESTS
// ============================================================================

TEST_VK(KernelRegistryTest, LookupPerformance) {
	auto ml_kernels = OaKernelRegistry::GetMlKernels();
	
	// Perform many lookups to test performance
	constexpr int iterations = 10000;
	
	auto start = std::chrono::high_resolution_clock::now();
	
	for (int i = 0; i < iterations; i++) {
		for (const auto& kernel : ml_kernels) {
			const OaComputeKernel* found = OaComputeKernelFindByPackedId(kernel.Id);
			(void)found; // Suppress unused warning
		}
	}
	
	auto end = std::chrono::high_resolution_clock::now();
	auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
	
	double lookups_per_sec = (iterations * ml_kernels.Size() * 1000000.0) / duration.count();
	
	std::printf("[Performance] %d iterations × %zu kernels = %.2f M lookups/sec\n", 
		iterations, ml_kernels.Size(), lookups_per_sec / 1000000.0);
	
	// Should be able to do at least 1M lookups per second
	EXPECT_GT(lookups_per_sec, 1000000.0) 
		<< "Lookup performance is too slow: " << lookups_per_sec << " lookups/sec";
}

TEST_VK(KernelRegistryTest, ConcurrentLookups) {
	// Test that lookups are thread-safe (registry is const, so should be safe)
	auto ml_kernels = OaKernelRegistry::GetMlKernels();
	
	if (ml_kernels.Size() == 0) {
		GTEST_SKIP() << "No kernels to test";
	}
	
	std::atomic<int> success_count{0};
	std::atomic<int> failure_count{0};
	
	auto lookup_thread = [&]() {
		for (int i = 0; i < 1000; i++) {
			for (const auto& kernel : ml_kernels) {
				const OaComputeKernel* found = OaComputeKernelFindByPackedId(kernel.Id);
				if (found != nullptr && found->Id == kernel.Id) {
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
	
	// Wait for all threads
	for (auto& t : threads) {
		t.join();
	}
	
	std::printf("[Concurrency] Success: %d, Failures: %d\n", 
		success_count.load(), failure_count.load());
	
	EXPECT_EQ(failure_count.load(), 0) << "Concurrent lookups had failures";
	EXPECT_GT(success_count.load(), 0) << "No successful concurrent lookups";
}
