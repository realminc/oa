#include <gtest/gtest.h>

#include <chrono>
#include <cstdlib>
#include <cstdio>
#include <cstring>

#include <Oa/Core/Memory.h>
#include <Oa/Core/FnMatrix.h>
#include <Oa/Runtime/Engine.h>
#include <Oa/Runtime/ExecutionMemory.h>
#include <Oa/Runtime/Allocator.h>
#include <Oa/Runtime/Spirv.h>
#include <Oa/Runtime/Pool.h>
#include <Oa/Runtime/Context.h>
#include <Oa/Runtime/ComputeGraph.h>

#include <Oa/Runtime/Vma/Slab.h>
#include <Oa/Core/Thread.h>
#include <Oa/Crypto/Buffer.h>

TEST(Preamble, SuiteInfo) {
	fprintf(stderr,
		"  [test_allocator] runtime: slab/spinlock (CPU) + OaVma/inference pool (Vulkan)\n");
	SUCCEED();
}

// Slab tests (pure CPU, no Vulkan)
TEST(Slab, AllocFree64Slots) {
	alignas(64) OaU8 backing[64 * 1024];
	OaVkSlab slab;
	slab.Init(backing, 1024, 64);

	EXPECT_EQ(slab.FreeCount(), 64u);
	EXPECT_FALSE(slab.IsFull());
	EXPECT_TRUE(slab.IsEmpty());

	OaVec<void*> ptrs;
	for (OaU32 i = 0; i < 64; ++i) {
		void* p = slab.Alloc();
		ASSERT_NE(p, nullptr) << "alloc failed at slot " << i;
		ptrs.PushBack(p);
	}

	EXPECT_EQ(slab.FreeCount(), 0u);
	EXPECT_TRUE(slab.IsFull());
	EXPECT_FALSE(slab.IsEmpty());

	EXPECT_EQ(slab.Alloc(), nullptr);

	for (void* p : ptrs) {
		slab.Free(p);
	}
	EXPECT_EQ(slab.FreeCount(), 64u);
	EXPECT_TRUE(slab.IsEmpty());
}

TEST(Slab, SlotPointersAreDistinct) {
	alignas(64) OaU8 backing[32 * 256];
	OaVkSlab slab;
	slab.Init(backing, 256, 32);

	OaVec<void*> ptrs;
	for (OaU32 i = 0; i < 32; ++i) {
		ptrs.PushBack(slab.Alloc());
	}

	for (OaU32 i = 0; i < 32; ++i) {
		for (OaU32 j = i + 1; j < 32; ++j) {
			EXPECT_NE(ptrs[i], ptrs[j]) << "slots " << i << " and " << j << " overlap";
		}
	}

	for (void* p : ptrs) slab.Free(p);
}

TEST(Slab, SmallCapacity) {
	alignas(64) OaU8 backing[8 * 64];
	OaVkSlab slab;
	slab.Init(backing, 64, 8);

	EXPECT_EQ(slab.FreeCount(), 8u);
	for (OaU32 i = 0; i < 8; ++i) {
		ASSERT_NE(slab.Alloc(), nullptr);
	}
	EXPECT_TRUE(slab.IsFull());
	EXPECT_EQ(slab.Alloc(), nullptr);
}

TEST(Slab, BenchmarkAllocFree) {
	alignas(64) OaU8 backing[64 * 1024];
	OaVkSlab slab;
	slab.Init(backing, 1024, 64);

	constexpr OaU32 kIters = 100000;
	void* ptrs[64];

	auto t0 = std::chrono::high_resolution_clock::now();
	for (OaU32 iter = 0; iter < kIters; ++iter) {
		for (OaU32 i = 0; i < 64; ++i) {
			ptrs[i] = slab.Alloc();
		}
		for (OaU32 i = 0; i < 64; ++i) {
			slab.Free(ptrs[i]);
		}
	}
	auto t1 = std::chrono::high_resolution_clock::now();
	auto us = std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count();
	OaF64 nsPerOp = static_cast<OaF64>(us * 1000) / static_cast<OaF64>(kIters * 128);
	fprintf(stderr, "  Slab alloc+free: %.1f ns/op (%u iters x 128 ops)\n", nsPerOp, kIters);
	EXPECT_LT(nsPerOp, 50.0);
}

// Spinlock tests (pure CPU)

TEST(Spinlock, LockUnlock) {
	OaSpinlock lock;
	lock.Lock();
	lock.Unlock();
	SUCCEED();
}

TEST(Spinlock, TryLock) {
	OaSpinlock lock;
	EXPECT_TRUE(lock.TryLock());
	EXPECT_FALSE(lock.TryLock());
	lock.Unlock();
	EXPECT_TRUE(lock.TryLock());
	lock.Unlock();
}

TEST(Spinlock, Guard) {
	OaSpinlock lock;
	{
		OaSpinlockGuard guard(lock);
		EXPECT_FALSE(lock.TryLock());
	}
	EXPECT_TRUE(lock.TryLock());
	lock.Unlock();
}

// Vulkan allocator tests (require GPU)

static OaUniquePtr<OaEngine> CreateTestEngine(OaBool InRegisterAsGlobal = true) {
	const char* validation = std::getenv("OA_VK_VALIDATION");
	const OaBool enableValidation = validation != nullptr && validation[0] == '1';
	auto result = OaEngine::Create({
		.EnableValidation = enableValidation,
		.AppName = "test_allocator",
		.MeshVulkanIndices = {},
		.RegisterAsGlobal = InRegisterAsGlobal,
	});
	return std::move(result.GetValue());   // move the owning pointer out (engine is pinned)
}

TEST(Allocator, DescriptorRangeExposesPaddedVkBufferTail) {
	auto rt = CreateTestEngine(false);
	auto result = rt->AllocBuffer(6);
	ASSERT_TRUE(result.IsOk());
	auto buffer = std::move(*result);
	EXPECT_EQ(buffer.Size, 6U);
	EXPECT_EQ(buffer.Capacity, 8U);
	EXPECT_EQ(buffer.DescriptorRange(), 8U);

	// Reuse must not expose the old allocation capacity through the descriptor.
	auto largeResult = rt->AllocBuffer(4096);
	ASSERT_TRUE(largeResult.IsOk());
	auto large = std::move(*largeResult);
	const auto reusedHandle = large.Buffer;
	rt->FreeBuffer(large);
	auto reusedResult = rt->AllocBuffer(6);
	ASSERT_TRUE(reusedResult.IsOk());
	auto reused = std::move(*reusedResult);
	EXPECT_EQ(reused.Buffer, reusedHandle);
	EXPECT_GE(reused.Capacity, 4096U);
	EXPECT_EQ(reused.Size, 6U);
	EXPECT_EQ(reused.DescriptorRange(), 8U);
	rt->FreeBuffer(reused);
	rt->FreeBuffer(buffer);

	OaVkBuffer legacy;
	legacy.Size = 6U;
	EXPECT_EQ(legacy.Capacity, 0U);
	EXPECT_EQ(legacy.DescriptorRange(), legacy.Size);
}

TEST(Allocator, BarePipelineCannotSatisfyDifferentStorageDtype) {
	auto rt = CreateTestEngine(false);
	const auto* spirv = OaSpvFindAny("Scale");
	ASSERT_NE(spirv, nullptr);
	OaPipelineSpec spec{
		.WgSize = 256,
		.NumBindings = 16,
		.PushConstantBytes = 128,
	};
	const OaStringView kBareName = "ScaleBareDtypeContract";
	ASSERT_TRUE(rt->EnsurePipeline(
		kBareName, OaSpan<const OaU8>(spirv->Data, spirv->Size), spec).IsOk());

	const auto& fp32 = rt->Pipelines.GetPipeline(kBareName, 0U);
	ASSERT_NE(fp32.Pipeline, nullptr);
	EXPECT_EQ(fp32.NativeDtype, 0U);
	const auto& bf16 = rt->Pipelines.GetPipeline(kBareName, 1U);
	EXPECT_EQ(bf16.Pipeline, nullptr);
	const auto& invalid = rt->Pipelines.GetPipeline("Scale", 2U);
	EXPECT_EQ(invalid.Pipeline, nullptr);
}

TEST(Allocator, EngineOwningPointerProvidesRaiiTeardown) {
	{
		auto rt = CreateTestEngine();
		ASSERT_NE(rt.get(), nullptr);
		EXPECT_EQ(OaEngine::GetGlobal(), rt.get());
		// No explicit Destroy(): the owning pointer must drain and release the
		// engine, including clearing its process-global registration.
	}
	EXPECT_EQ(OaEngine::GetGlobal(), nullptr);
}

TEST(Allocator, EngineInitializationIsOneShotAndStateful) {
	OaEngine engine;
	EXPECT_EQ(engine.GetState(), OaEngineState::Empty);
	EXPECT_FALSE(engine.HasCompute());

	OaEngineConfig config;
	config.AppName = "test_engine_state";
	config.RegisterAsGlobal = false;
	config.PreloadEmbeddedPipelines = false;
	ASSERT_TRUE(engine.InitInPlace(config).IsOk());
	EXPECT_EQ(engine.GetState(), OaEngineState::Ready);
	EXPECT_TRUE(engine.HasCompute());

	auto secondInit = engine.InitInPlace(config);
	EXPECT_FALSE(secondInit.IsOk());
	EXPECT_EQ(secondInit.GetCode(), OaStatusCode::FailedPrecondition);
	EXPECT_EQ(engine.GetState(), OaEngineState::Ready);

	EXPECT_TRUE(engine.Close().IsOk());
	EXPECT_EQ(engine.GetState(), OaEngineState::Destroyed);
	EXPECT_FALSE(engine.HasCompute());
	EXPECT_TRUE(engine.Close().IsOk());
	EXPECT_EQ(engine.GetState(), OaEngineState::Destroyed);
}

TEST(Allocator, EngineOwnsTheContextUsedByFnMatrix) {
	for (const OaBool registerGlobal : {true, false}) {
		auto result = OaEngine::Create(OaEngineConfig{
			.RegisterAsGlobal = registerGlobal,
		});
		ASSERT_TRUE(result.IsOk());
		auto engine = std::move(*result);
		EXPECT_EQ(OaEngine::GetGlobal() != nullptr, registerGlobal);
		OaContext::RecordingScope contextScope(engine->GetContext());
		auto input = OaFnMatrix::Ones(OaMatrixShape{16});
		auto output = OaFnMatrix::Scale(input, 3.0F);
		EXPECT_GT(engine->GetContext().NodeCount(), 0U);
		ASSERT_TRUE(engine->GetContext().Execute().IsOk());
		ASSERT_TRUE(engine->GetContext().Sync().IsOk());
		OaF32 values[16]{};
		ASSERT_TRUE(OaFnMatrix::CopyToHost(output, values, sizeof(values)).IsOk());
		for (OaF32 value : values) EXPECT_FLOAT_EQ(value, 3.0F);
	}
}

TEST(Allocator, StableMatrixStorageBelongsToExecutionSessionPolicy) {
	auto engine = CreateTestEngine(false);
	auto& context = engine->GetContext();

	OaExecutionMemory::BeginStableFrame(context);
	EXPECT_TRUE(OaExecutionMemory::IsStableFrameActive(context));
	auto first = OaExecutionMemory::AllocateMatrixBuffer(
		context, 4096, OaMemoryPlacement::DeviceLocal);
	OaExecutionMemory::SealStableInputs(context);
	EXPECT_TRUE(OaExecutionMemory::AreStableInputsSealed(context));
	auto second = OaExecutionMemory::AllocateMatrixBuffer(
		context, 8192, OaMemoryPlacement::DeviceLocal);
	EXPECT_EQ(OaExecutionMemory::StableExternalResourceCount(context), 1U);
	EXPECT_EQ(OaExecutionMemory::StableTransientResourceCount(context), 1U);
	OaExecutionMemory::EndStableFrame(context);
	ASSERT_TRUE(first);
	ASSERT_TRUE(second);
	EXPECT_FALSE(OaExecutionMemory::IsStableFrameActive(context));

	OaExecutionMemory::BeginStableFrame(context);
	auto firstReused = OaExecutionMemory::AllocateMatrixBuffer(
		context, 2048, OaMemoryPlacement::DeviceLocal);
	OaExecutionMemory::SealStableInputs(context);
	auto secondReused = OaExecutionMemory::AllocateMatrixBuffer(
		context, 8192, OaMemoryPlacement::DeviceLocal);
	OaExecutionMemory::EndStableFrame(context);

	EXPECT_EQ(firstReused.Get(), first.Get());
	EXPECT_EQ(secondReused.Get(), second.Get());
	EXPECT_EQ(firstReused->Size, 2048U);
	EXPECT_GE(firstReused->Capacity, 4096U);
}

TEST(Allocator, MatrixOwnerDoesNotCallDestroyedEngine) {
	OaMatrix retained;
	OaMatrix retainedView;
	auto engine = CreateTestEngine(false);
	ASSERT_NE(engine, nullptr);
	{
		OaContext::RecordingScope contextScope(engine->GetContext());
		retained = OaFnMatrix::Ones(OaMatrixShape{16});
		retainedView = retained.View(OaMatrixShape{4, 4});
		ASSERT_TRUE(retained.HasStorage());
		ASSERT_TRUE(retainedView.HasStorage());
		ASSERT_TRUE(engine->GetContext().Execute().IsOk());
		ASSERT_TRUE(engine->GetContext().Sync().IsOk());
	}

	ASSERT_TRUE(engine->Close().IsOk());
	// Close owns allocation teardown and mutates the shared wrapper seen by
	// every retained matrix/view. No stale descriptor or mapped pointer remains.
	EXPECT_FALSE(retained.HasStorage());
	EXPECT_FALSE(retainedView.HasStorage());
	EXPECT_EQ(retained.Data(), nullptr);
	EXPECT_EQ(retainedView.Data(), nullptr);
	EXPECT_EQ(retained.HeapSlot(), -1);
	EXPECT_EQ(retainedView.HeapSlot(), -1);
	EXPECT_EQ(retained.HostBlock().Ptr, nullptr);
	EXPECT_EQ(retainedView.HostBlock().Ptr, nullptr);
	EXPECT_EQ(retained.GetVkBuffer().Buffer, nullptr);
	EXPECT_EQ(retained.GetVkBuffer().Allocation, nullptr);

	// Dropping the wrappers after Close must not touch the dead allocator.
	retained = {};
	retainedView = {};
}

TEST(Allocator, EngineCloseDrainsAliasLeasesBeforeAllocatorTeardown) {
	auto engine = CreateTestEngine(false);
	ASSERT_NE(engine, nullptr);

	constexpr OaU32 N = 64;
	OaVec<OaMatrix> matrices;
	{
		OaContext::RecordingScope contextScope(engine->GetContext());
		for (OaU32 i = 0; i < 5U; ++i) {
			matrices.PushBack(OaFnMatrix::Empty(
				{static_cast<OaI64>(N)}, OaScalarType::Float32,
				OaMemoryPlacement::HostUpload));
			ASSERT_TRUE(matrices.Back().HasStorage());
		}
	}

	OaVec<OaVkBuffer> buffers;
	for (const auto& matrix : matrices) buffers.PushBack(matrix.GetVkBuffer());
	OaComputeGraph graph;
	struct PushConstants { OaU32 Count; OaF32 Scale; } push{N, 1.001F};
	for (OaU32 i = 0; i < 4U; ++i) {
		OaVkBuffer dispatchBuffers[] = {buffers[i], buffers[i + 1U]};
		OaBufferAccess access[] = {OaBufferAccess::Read, OaBufferAccess::Write};
		graph.Add("Scale", dispatchBuffers, access, &push, sizeof(push), 1);
	}
	OaMatrix* eligible[] = {&matrices[1], &matrices[3]};
	ASSERT_TRUE(graph.MaterializeAliases(*engine, eligible).IsOk());
	ASSERT_NE(matrices[1].GetVkBuffer().Buffer, nullptr);
	ASSERT_EQ(matrices[1].Data(), matrices[3].Data());

	ASSERT_TRUE(engine->Close().IsOk());
	for (const OaUsize index : {OaUsize{1}, OaUsize{3}}) {
		EXPECT_FALSE(matrices[index].HasStorage());
		EXPECT_EQ(matrices[index].Data(), nullptr);
		EXPECT_EQ(matrices[index].HeapSlot(), -1);
		EXPECT_EQ(matrices[index].GetVkBuffer().Buffer, nullptr);
		EXPECT_EQ(matrices[index].GetVkBuffer().Allocation, nullptr);
	}
	// The graph and matrices intentionally outlive Close(). Their eventual
	// shared-owner release must only delete the already-inert wrappers.
}

TEST(Allocator, BindlessExhaustionFailsAllocationWithoutLeakingStorage) {
	auto engine = CreateTestEngine(false);
	auto& rt = *engine;

	const OaU32 savedBufferCapacity = OA_BINDLESS_CAPACITY;
	const OaU32 savedImageCapacity = OA_BINDLESS_IMAGE_CAPACITY;
	const OaU32 savedSamplerCapacity = OA_BINDLESS_SAMPLER_CAPACITY;
	OA_BINDLESS_CAPACITY = 2;
	OA_BINDLESS_IMAGE_CAPACITY = 2;
	OA_BINDLESS_SAMPLER_CAPACITY = 2;
	auto tinyResult = OaBindlessHeap::Create(rt.Device);
	if (not tinyResult.IsOk()) {
		OA_BINDLESS_CAPACITY = savedBufferCapacity;
		OA_BINDLESS_IMAGE_CAPACITY = savedImageCapacity;
		OA_BINDLESS_SAMPLER_CAPACITY = savedSamplerCapacity;
		FAIL() << tinyResult.GetStatus().ToString();
		return;
	}

	auto originalHeap = std::move(rt.Bindless);
	rt.Bindless = std::move(*tinyResult);
	auto first = rt.AllocBufferDevice(4096);
	const OaVmaStats beforeFailure = rt.Allocator.GetStats();
	auto exhausted = rt.AllocBufferDevice(4096);
	const OaVmaStats afterFailure = rt.Allocator.GetStats();

	EXPECT_TRUE(first.IsOk()) << first.GetStatus().ToString();
	EXPECT_FALSE(exhausted.IsOk());
	if (not exhausted.IsOk()) {
		EXPECT_EQ(exhausted.GetStatus().GetCode(), OaStatusCode::ResourceExhausted);
	}
	EXPECT_EQ(afterFailure.AllocationCount, beforeFailure.AllocationCount);

	if (first.IsOk()) rt.FreeBuffer(*first);
	rt.Bindless.Destroy(rt.Device);
	rt.Bindless = std::move(originalHeap);
	OA_BINDLESS_CAPACITY = savedBufferCapacity;
	OA_BINDLESS_IMAGE_CAPACITY = savedImageCapacity;
	OA_BINDLESS_SAMPLER_CAPACITY = savedSamplerCapacity;
}

TEST(Allocator, AllocBarFallback) {
	auto rtP = CreateTestEngine(); OaEngine& rt = *rtP;
	{
		const OaStringView name = rt.DeviceName();
		fprintf(stderr, "  [test_allocator] device: %.*s\n",
			static_cast<int>(name.size()), name.data());
	}

	auto result = rt.Allocator.AllocBar(4096);
	ASSERT_TRUE(result.IsOk()) << result.GetStatus().ToString();
	auto& buf = result.GetValue();

	EXPECT_NE(buf.Buffer, nullptr);
	EXPECT_NE(buf.MappedPtr, nullptr);
	EXPECT_EQ(buf.Size, 4096u);

	if (rt.HasSAM()) {
		EXPECT_TRUE(buf.IsBar());
		fprintf(stderr, "  AllocBar: SAM detected, BAR flag set\n");
	} else {
		fprintf(stderr, "  AllocBar: no SAM, fell back to host-visible\n");
	}

	rt.Allocator.Free(buf);
	rt.Destroy();
}

TEST(Allocator, PlacementMetadataMatchesAllocationContract) {
	auto rtP = CreateTestEngine(); OaEngine& rt = *rtP;

	auto deviceResult = rt.Allocator.AllocDevice(4096);
	ASSERT_TRUE(deviceResult.IsOk()) << deviceResult.GetStatus().ToString();
	auto device = std::move(deviceResult).GetValue();
	EXPECT_EQ(device.Size, 4096u);
	EXPECT_EQ(device.Capacity, 4096u);
	EXPECT_EQ(device.Placement, OaMemoryPlacement::DeviceLocal);
	EXPECT_FALSE(device.IsHostVisible());

	auto uploadResult = rt.Allocator.AllocHostVisible(4096);
	ASSERT_TRUE(uploadResult.IsOk()) << uploadResult.GetStatus().ToString();
	auto upload = std::move(uploadResult).GetValue();
	EXPECT_EQ(upload.Placement, OaMemoryPlacement::HostUpload);
	EXPECT_TRUE(upload.IsHostVisible());

	auto readbackResult = rt.Allocator.AllocHostReadback(4096);
	ASSERT_TRUE(readbackResult.IsOk()) << readbackResult.GetStatus().ToString();
	auto readback = std::move(readbackResult).GetValue();
	EXPECT_EQ(readback.Placement, OaMemoryPlacement::HostReadback);
	EXPECT_TRUE(readback.IsHostVisible());

	rt.Allocator.Free(device);
	rt.Allocator.Free(upload);
	rt.Allocator.Free(readback);
	rt.Destroy();
}

TEST(Allocator, DeviceLocalUploadReadbackRoundTrip) {
	auto rtP = CreateTestEngine(); OaEngine& rt = *rtP;
	constexpr OaU32 kCount = 1024;
	constexpr OaU64 kBytes = kCount * sizeof(OaU32);

	auto result = rt.AllocBuffer(kBytes, OaMemoryPlacement::DeviceLocal);
	ASSERT_TRUE(result.IsOk()) << result.GetStatus().ToString();
	auto buffer = std::move(result).GetValue();
	ASSERT_EQ(buffer.Placement, OaMemoryPlacement::DeviceLocal);

	OaVec<OaU32> source(kCount);
	OaVec<OaU32> destination(kCount);
	for (OaU32 i = 0; i < kCount; ++i) source[i] = i * 2654435761U;

	ASSERT_TRUE(rt.UploadBuffer(buffer, 0, source.Data(), kBytes).IsOk());
	ASSERT_TRUE(rt.ReadbackBuffer(buffer, 0, destination.Data(), kBytes).IsOk());
	for (OaU32 i = 0; i < kCount; ++i) EXPECT_EQ(destination[i], source[i]);
	const OaVmaStats firstReadbackStats = rt.Allocator.GetStats();
	for (OaU32 repeat = 0; repeat < 8; ++repeat) {
		ASSERT_TRUE(rt.ReadbackBuffer(buffer, 0, destination.Data(), kBytes).IsOk());
	}
	const OaVmaStats repeatedReadbackStats = rt.Allocator.GetStats();
	EXPECT_EQ(repeatedReadbackStats.AllocationCount, firstReadbackStats.AllocationCount);
	EXPECT_EQ(repeatedReadbackStats.AllocationBytes, firstReadbackStats.AllocationBytes);

	rt.FreeBuffer(buffer);

	// Byte and BF16 matrices are not necessarily four-byte sized. The public
	// transfer contract remains byte-addressable even though Vulkan buffer-copy
	// commands operate on aligned words.
	auto byteResult = rt.AllocBuffer(7, OaMemoryPlacement::DeviceLocal);
	ASSERT_TRUE(byteResult.IsOk()) << byteResult.GetStatus().ToString();
	auto byteBuffer = std::move(byteResult).GetValue();
	const OaU8 initial[7] = {1, 2, 3, 4, 5, 6, 7};
	const OaU8 patch[3] = {9, 10, 11};
	OaU8 bytes[7]{};
	ASSERT_TRUE(rt.UploadBuffer(byteBuffer, 0, initial, sizeof(initial)).IsOk());
	ASSERT_TRUE(rt.UploadBuffer(byteBuffer, 1, patch, sizeof(patch)).IsOk());
	ASSERT_TRUE(rt.ReadbackBuffer(byteBuffer, 0, bytes, sizeof(bytes)).IsOk());
	const OaU8 expected[7] = {1, 9, 10, 11, 5, 6, 7};
	for (OaU32 i = 0; i < 7; ++i) EXPECT_EQ(bytes[i], expected[i]);
	rt.FreeBuffer(byteBuffer);
	rt.Destroy();
}

TEST(Allocator, UploadWeightsCorrectness) {
	auto rtP = CreateTestEngine(); OaEngine& rt = *rtP;

	constexpr OaU32 N = 1024;
	constexpr OaU64 size = N * sizeof(OaF32);

	auto result = rt.Allocator.AllocBar(size);
	ASSERT_TRUE(result.IsOk());
	auto& buf = result.GetValue();

	OaVec<OaF32> src(N);
	for (OaU32 i = 0; i < N; ++i) {
		src[i] = static_cast<OaF32>(i) * 0.5f;
	}

	auto status = rt.Allocator.UploadWeights(buf, src.Data(), size);
	ASSERT_TRUE(status.IsOk()) << status.ToString();

	auto* dst = static_cast<OaF32*>(buf.MappedPtr);
	for (OaU32 i = 0; i < N; ++i) {
		EXPECT_FLOAT_EQ(dst[i], src[i]) << "mismatch at " << i;
	}

	rt.Allocator.Free(buf);
	rt.Destroy();
}

TEST(Allocator, BudgetQuery) {
	auto rtP = CreateTestEngine(); OaEngine& rt = *rtP;

	auto stats = rt.Allocator.GetStats();
	fprintf(stderr, "  Budget: used=%llu MB, budget=%llu MB\n",
		(unsigned long long)(stats.UsedBytes / (1024 * 1024)),
		(unsigned long long)(stats.BudgetBytes / (1024 * 1024)));

	EXPECT_GT(stats.BudgetBytes, 0u);

	rt.Destroy();
}

TEST(Allocator, BenchmarkVmaVsSlab) {
	auto rtP = CreateTestEngine(); OaEngine& rt = *rtP;

	constexpr OaU32 kVmaIters = 1000;
	constexpr OaU64 kBufSize = 1024;

	auto t0 = std::chrono::high_resolution_clock::now();
	for (OaU32 i = 0; i < kVmaIters; ++i) {
		auto r = rt.Allocator.AllocHostVisible(kBufSize);
		if (r.IsOk()) {
			auto& b = r.GetValue();
			rt.Allocator.Free(b);
		}
	}
	auto t1 = std::chrono::high_resolution_clock::now();
	auto vmaUs = std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count();
	OaF64 vmaNsPerOp = static_cast<OaF64>(vmaUs * 1000) / static_cast<OaF64>(kVmaIters);

	alignas(64) OaU8 backing[64 * kBufSize];
	OaVkSlab slab;
	slab.Init(backing, kBufSize, 64);

	constexpr OaU32 kSlabIters = 100000;
	auto t2 = std::chrono::high_resolution_clock::now();
	for (OaU32 i = 0; i < kSlabIters; ++i) {
		void* p = slab.Alloc();
		slab.Free(p);
	}
	auto t3 = std::chrono::high_resolution_clock::now();
	auto slabUs = std::chrono::duration_cast<std::chrono::microseconds>(t3 - t2).count();
	OaF64 slabNsPerOp = static_cast<OaF64>(slabUs * 1000) / static_cast<OaF64>(kSlabIters);

	fprintf(stderr, "  VMA alloc+free: %.0f ns/op (%u iters)\n", vmaNsPerOp, kVmaIters);
	fprintf(stderr, "  Slab alloc+free: %.1f ns/op (%u iters)\n", slabNsPerOp, kSlabIters);
	if (slabNsPerOp > 0) {
		fprintf(stderr, "  Speedup: %.0fx\n", vmaNsPerOp / slabNsPerOp);
	}

	rt.Destroy();
}

// Inference pool tests (require GPU)

TEST(InferencePool, PreAllocatedStablePointers) {
	auto rtP = CreateTestEngine(); OaEngine& rt = *rtP;

	OaVkInferencePoolConfig cfg{};
	cfg.NumLayers = 4;
	cfg.ActivationSize = 512;
	cfg.MaxBatchSize = 2;
	cfg.OutputRingBytes = 1024;

	OaVkInferencePool pool;
	auto status = pool.Init(rt.Allocator, cfg);
	ASSERT_TRUE(status.IsOk()) << status.ToString();

	EXPECT_EQ(pool.NumLayers(), 4u);
	EXPECT_EQ(pool.MaxBatchSize(), 2u);
	EXPECT_EQ(pool.ActivationSize(), 512u);
	EXPECT_EQ(pool.TotalActivationBytes(), 4u * 2u * 512u);

	void* p00a = pool.GetActivation(0, 0);
	void* p00b = pool.GetActivation(0, 0);
	EXPECT_EQ(p00a, p00b);

	void* p31a = pool.GetActivation(3, 1);
	void* p31b = pool.GetActivation(3, 1);
	EXPECT_EQ(p31a, p31b);

	void* ptrs[8];
	for (OaU32 l = 0; l < 4; ++l) {
		for (OaU32 b = 0; b < 2; ++b) {
			ptrs[l * 2 + b] = pool.GetActivation(l, b);
		}
	}
	for (OaU32 i = 0; i < 8; ++i) {
		for (OaU32 j = i + 1; j < 8; ++j) {
			EXPECT_NE(ptrs[i], ptrs[j]) << "activation slots " << i << " and " << j << " overlap";
		}
	}

	pool.Destroy(rt.Allocator);
	rt.Destroy();
}

TEST(InferencePool, RingBufferWrapAround) {
	auto rtP = CreateTestEngine(); OaEngine& rt = *rtP;

	OaVkInferencePoolConfig cfg{};
	cfg.NumLayers = 1;
	cfg.ActivationSize = 64;
	cfg.MaxBatchSize = 1;
	cfg.OutputRingBytes = 64 * 4;

	OaVkInferencePool pool;
	auto status = pool.Init(rt.Allocator, cfg);
	ASSERT_TRUE(status.IsOk()) << status.ToString();

	OaVec<void*> ringPtrs;
	for (OaU32 i = 0; i < 4; ++i) {
		void* p = pool.NextOutputSlot();
		ASSERT_NE(p, nullptr);
		ringPtrs.PushBack(p);
		pool.ConsumeOutputSlot();
	}

	void* wrap = pool.NextOutputSlot();
	EXPECT_EQ(wrap, ringPtrs[0]);

	pool.Destroy(rt.Allocator);
	rt.Destroy();
}

TEST(InferencePool, ResetSessionZeros) {
	auto rtP = CreateTestEngine(); OaEngine& rt = *rtP;

	OaVkInferencePoolConfig cfg{};
	cfg.NumLayers = 2;
	cfg.ActivationSize = 128;
	cfg.MaxBatchSize = 1;
	cfg.OutputRingBytes = 0;

	OaVkInferencePool pool;
	auto status = pool.Init(rt.Allocator, cfg);
	ASSERT_TRUE(status.IsOk());

	auto* p = static_cast<OaF32*>(pool.GetActivation(0, 0));
	for (OaU32 i = 0; i < 32; ++i) p[i] = 1.0f;

	pool.ResetSession();

	for (OaU32 i = 0; i < 32; ++i) {
		EXPECT_FLOAT_EQ(p[i], 0.0f) << "not zeroed at " << i;
	}

	pool.Destroy(rt.Allocator);
	rt.Destroy();
}

TEST(InferencePool, WeightPinning) {
	auto rtP = CreateTestEngine(); OaEngine& rt = *rtP;

	auto bufResult = rt.Allocator.AllocHostVisible(4096);
	ASSERT_TRUE(bufResult.IsOk());
	auto& buf = bufResult.GetValue();

	OaVkInferencePoolConfig cfg{};
	cfg.NumLayers = 1;
	cfg.ActivationSize = 64;
	cfg.MaxBatchSize = 1;
	cfg.OutputRingBytes = 0;

	OaVkInferencePool pool;
	ASSERT_TRUE(pool.Init(rt.Allocator, cfg).IsOk());

	auto pinStatus = pool.PinWeights(buf);
	EXPECT_TRUE(pinStatus.IsOk());

	pool.Destroy(rt.Allocator);
	rt.Allocator.Free(buf);
	rt.Destroy();
}

// Secure buffer tests (pure CPU)

TEST(SecureBuffer, ZeroOnDestroy) {
	alignas(64) OaU8 backing[256];
	OaMemset(backing, 0xAA, 256);

	{
		OaSecureBuffer sec(backing, 256);
		EXPECT_TRUE(sec.IsValid());
	}
	// After OaSecureBuffer destructor, memory must be zeroed
	for (OaU32 i = 0; i < 256; ++i) {
		EXPECT_EQ(backing[i], 0u) << "byte " << i << " not zeroed";
	}
}

TEST(SecureBuffer, MoveSemantics) {
	alignas(64) OaU8 backing[128];
	OaMemset(backing, 0xBB, 128);

	OaSecureBuffer a(backing, 128);
	EXPECT_TRUE(a.IsValid());

	OaSecureBuffer b = std::move(a);
	EXPECT_FALSE(a.IsValid());
	EXPECT_TRUE(b.IsValid());
	EXPECT_EQ(b.Data(), backing);
	EXPECT_EQ(b.SizeBytes(), 128u);

	// Destroy b — backing should be zeroed
	b.Reset();
	EXPECT_FALSE(b.IsValid());
	for (OaU32 i = 0; i < 128; ++i) {
		EXPECT_EQ(backing[i], 0u);
	}
}

TEST(SecureBuffer, DefaultIsInvalid) {
	OaSecureBuffer sec;
	EXPECT_FALSE(sec.IsValid());
}

// PQC pool tests (require GPU)

TEST(PqcPool, NttSlabAllocFree) {
	auto rtP = CreateTestEngine(); OaEngine& rt = *rtP;

	OaVkPqcPoolConfig cfg{};
	cfg.MaxConcurrentSigns = 8;
	cfg.MaxConcurrentHashes = 8;
	cfg.NttSlotSize = 1024;
	cfg.HashSlotSize = 200;

	OaVkPqcPool pool;
	auto status = pool.Init(rt.Allocator, cfg);
	ASSERT_TRUE(status.IsOk()) << status.ToString();

	EXPECT_EQ(pool.NttSlotsUsed(), 0u);

	OaVec<void*> slots;
	for (OaU32 i = 0; i < 8; ++i) {
		void* p = pool.AllocNttSlot();
		ASSERT_NE(p, nullptr) << "NTT alloc failed at " << i;
		slots.PushBack(p);
	}
	EXPECT_EQ(pool.NttSlotsUsed(), 8u);

	// 9th should fail
	EXPECT_EQ(pool.AllocNttSlot(), nullptr);

	// Verify 256B alignment
	for (void* p : slots) {
		EXPECT_EQ(reinterpret_cast<OaUsize>(p) % 256, 0u) << "NTT slot not 256B-aligned";
	}

	for (void* p : slots) pool.FreeNttSlot(p);
	EXPECT_EQ(pool.NttSlotsUsed(), 0u);

	pool.Destroy(rt.Allocator);
	rt.Destroy();
}

TEST(PqcPool, HashStatePool) {
	auto rtP = CreateTestEngine(); OaEngine& rt = *rtP;

	OaVkPqcPoolConfig cfg{};
	cfg.MaxConcurrentSigns = 4;
	cfg.MaxConcurrentHashes = 4;
	cfg.NttSlotSize = 1024;
	cfg.HashSlotSize = 200;

	OaVkPqcPool pool;
	ASSERT_TRUE(pool.Init(rt.Allocator, cfg).IsOk());

	EXPECT_EQ(pool.HashSlotsUsed(), 0u);

	OaVec<void*> slots;
	for (OaU32 i = 0; i < 4; ++i) {
		void* p = pool.AllocHashSlot();
		ASSERT_NE(p, nullptr);
		slots.PushBack(p);
	}
	EXPECT_EQ(pool.HashSlotsUsed(), 4u);
	EXPECT_EQ(pool.AllocHashSlot(), nullptr);

	for (void* p : slots) pool.FreeHashSlot(p);
	EXPECT_EQ(pool.HashSlotsUsed(), 0u);

	pool.Destroy(rt.Allocator);
	rt.Destroy();
}

TEST(PqcPool, SecureKeyAlloc) {
	auto rtP = CreateTestEngine(); OaEngine& rt = *rtP;

	OaVkPqcPoolConfig cfg{};
	cfg.MaxConcurrentSigns = 4;
	cfg.MaxConcurrentHashes = 4;

	OaVkPqcPool pool;
	ASSERT_TRUE(pool.Init(rt.Allocator, cfg).IsOk());

	auto key = pool.AllocKeyBuffer(4096);
	EXPECT_TRUE(key.IsValid());
	EXPECT_EQ(key.SizeBytes(), 4096u);

	// Write to key buffer
	OaMemset(key.Data(), 0xFF, key.SizeBytes());

	// Move out — zeroes the old one on destruction
	{
		OaSecureBuffer key2 = std::move(key);
		EXPECT_TRUE(key2.IsValid());
		EXPECT_FALSE(key.IsValid());
		// key2 destructor will zero the memory
	}

	pool.Destroy(rt.Allocator);
	rt.Destroy();
}
