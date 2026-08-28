#include <gtest/gtest.h>
#include "../../oaTest.h"

#include <cstdlib>
#include <cstdio>
#include <cstring>

#include <oa/core/std/memory.h>
#include <oa/core/fnMatrix.h>
#include <oa/core/matrixAccess.h>
#include <oa/runtime/engine.h>
#include <oa/runtime/engine/allocatorAccess.h>
#include <oa/runtime/engine/bindlessAccess.h>
#include <oa/runtime/engine/pipelineAccess.h>
#include <oa/runtime/executionSession.h>
#include <oa/runtime/allocator.h>
#include <oa/runtime/spirv.h>
#include <oa/runtime/executionSession.h>
#include <oa/runtime/executableGraph.h>

#include <oa/core/thread.h>

TEST(Preamble, SuiteInfo) {
	fprintf(stderr,
		"  [test_allocator] runtime: spinlock + engine cache + VMA\n");
	SUCCEED();
}

// Spinlock tests (pure CPU)

TEST(Spinlock, LockUnlock) {
	oa::Spinlock lock;
	lock.lock();
	lock.unlock();
	SUCCEED();
}

TEST(Spinlock, tryLock) {
	oa::Spinlock lock;
	EXPECT_TRUE(lock.tryLock());
	EXPECT_FALSE(lock.tryLock());
	lock.unlock();
	EXPECT_TRUE(lock.tryLock());
	lock.unlock();
}

TEST(Spinlock, Guard) {
	oa::Spinlock lock;
	{
		oa::SpinlockGuard guard(lock);
		EXPECT_FALSE(lock.tryLock());
	}
	EXPECT_TRUE(lock.tryLock());
	lock.unlock();
}

// vulkan allocator tests (require GPU)

static oa::UniquePtr<oa::Engine> createTestEngine(oa::Bool inMakeDefaultContext = false) {
	const char* validation = std::getenv("OA_VK_VALIDATION");
	const oa::Bool enableValidation = validation != nullptr && validation[0] == '1';
	auto result = oa::Engine::create({
		.enableValidation = enableValidation,
		.appName = "test_allocator",
		.selectForThread = inMakeDefaultContext,
	});
	return std::move(result.getValue());   // move the owning pointer out (engine is pinned)
}

TEST(allocator, DescriptorRangeExposesPaddedVkBufferTail) {
	auto rt = createTestEngine(false);
	auto result = oa::EngineResourceAccess::allocBuffer(*rt, 6);
	ASSERT_TRUE(result.isOk());
	auto buffer = std::move(*result);
	EXPECT_EQ(buffer.size, 6U);
	EXPECT_EQ(buffer.capacity, 8U);
	EXPECT_EQ(buffer.descriptorRange(), 8U);

	// Reuse must not expose the old allocation capacity through the descriptor.
	auto largeResult = oa::EngineResourceAccess::allocBuffer(*rt, 4096);
	ASSERT_TRUE(largeResult.isOk());
	auto large = std::move(*largeResult);
	const auto reusedHandle = large.buffer;
	oa::EngineResourceAccess::freeBuffer(*rt, large);
	auto reusedResult = oa::EngineResourceAccess::allocBuffer(*rt, 6);
	ASSERT_TRUE(reusedResult.isOk());
	auto reused = std::move(*reusedResult);
	EXPECT_EQ(reused.buffer, reusedHandle);
	EXPECT_GE(reused.capacity, 4096U);
	EXPECT_EQ(reused.size, 6U);
	EXPECT_EQ(reused.descriptorRange(), 8U);
	oa::EngineResourceAccess::freeBuffer(*rt, reused);
	oa::EngineResourceAccess::freeBuffer(*rt, buffer);

	oavk::Buffer legacy;
	legacy.size = 6U;
	EXPECT_EQ(legacy.capacity, 0U);
	EXPECT_EQ(legacy.descriptorRange(), legacy.size);
}

TEST(allocator, HostUploadCacheReusesSmallestSufficientBuffer) {
	auto engine = createTestEngine(false);
	auto smallResult = oa::EngineResourceAccess::allocBuffer(*engine, 1024U);
	auto mediumResult = oa::EngineResourceAccess::allocBuffer(*engine, 2048U);
	auto largeResult = oa::EngineResourceAccess::allocBuffer(*engine, 4096U);
	ASSERT_TRUE(smallResult.isOk());
	ASSERT_TRUE(mediumResult.isOk());
	ASSERT_TRUE(largeResult.isOk());
	auto small = std::move(*smallResult);
	auto medium = std::move(*mediumResult);
	auto large = std::move(*largeResult);
	const void* mediumHandle = medium.buffer;

	// Free out of size order so reuse cannot depend on insertion position.
	oa::EngineResourceAccess::freeBuffer(*engine, large);
	oa::EngineResourceAccess::freeBuffer(*engine, small);
	oa::EngineResourceAccess::freeBuffer(*engine, medium);

	const auto statsBefore = oa::EngineAllocatorAccess::get(*engine).getStats();
	auto reuseResult = oa::EngineResourceAccess::allocBuffer(*engine, 1536U);
	ASSERT_TRUE(reuseResult.isOk());
	auto reused = std::move(*reuseResult);
	const auto statsAfter = oa::EngineAllocatorAccess::get(*engine).getStats();
	EXPECT_EQ(reused.buffer, mediumHandle);
	EXPECT_EQ(reused.capacity, 2048U);
	EXPECT_EQ(reused.size, 1536U);
	EXPECT_EQ(statsAfter.allocationCount, statsBefore.allocationCount);
	oa::EngineResourceAccess::freeBuffer(*engine, reused);
}

TEST(allocator, DescriptorAdmissionUsesLiveDeviceLimit) {
	auto rt = createTestEngine(false);
	auto rawResult = oa::EngineAllocatorAccess::get(*rt).allocHostVisible(16U);
	ASSERT_TRUE(rawResult.isOk());
	auto raw = std::move(*rawResult);
	auto registeredResult = oa::EngineResourceAccess::allocBuffer(*rt, 16U);
	ASSERT_TRUE(registeredResult.isOk());
	auto registered = std::move(*registeredResult);

	auto& maximum =
		oa::EngineDeviceAccess::get(*rt).info.hardware.maxStorageBufferRangeBytes;
	const oa::U64 savedMaximum = maximum;
	ASSERT_GT(savedMaximum, 8U);
	maximum = 8U;

	EXPECT_EQ(oa::EngineBindlessAccess::registerBuffer(*rt, raw), OA_BINDLESS_INVALID);
	EXPECT_EQ(raw.bindlessIndex, OA_BINDLESS_INVALID);
	const auto rejectedAllocation = oa::EngineResourceAccess::allocBuffer(*rt, 16U);
	EXPECT_FALSE(rejectedAllocation.isOk());
	EXPECT_EQ(rejectedAllocation.getStatus().getCode(),
		oa::StatusCode::OutOfRange);
	const auto rejectedUpdate =
		oa::EngineBindlessAccess::updateBufferDescriptor(*rt, registered);
	EXPECT_EQ(rejectedUpdate.getCode(), oa::StatusCode::OutOfRange);

	maximum = savedMaximum;
	EXPECT_TRUE(oa::EngineBindlessAccess::updateBufferDescriptor(*rt, registered).isOk());
	oa::EngineAllocatorAccess::get(*rt).free(raw);
	oa::EngineResourceAccess::freeBuffer(*rt, registered);
}

TEST(allocator, BarePipelineCannotSatisfyDifferentStorageDtype) {
	auto rt = createTestEngine(false);
	const auto* spirv = oavk::findSpirv("Scale");
	ASSERT_NE(spirv, nullptr);
	oa::PipelineSpec spec{
		.numBindings = 16,
		.pushConstantBytes = 128,
	};
	const oa::StringView kBareName = "ScaleBareDtypeContract";
	ASSERT_TRUE(oa::EnginePipelineAccess::ensure(
		*rt,
		kBareName,
		oa::Span<const oa::U8>(spirv->data, spirv->size),
		spec).isOk());

	const auto& fp32 = oa::EnginePipelineAccess::get(*rt).getPipeline(kBareName, 0U);
	ASSERT_NE(fp32.pipeline, nullptr);
	EXPECT_EQ(fp32.nativeDtype, 0U);
	const auto& bf16 = oa::EnginePipelineAccess::get(*rt).getPipeline(kBareName, 1U);
	EXPECT_EQ(bf16.pipeline, nullptr);
	const auto& invalid = oa::EnginePipelineAccess::get(*rt).getPipeline("Scale", 2U);
	EXPECT_EQ(invalid.pipeline, nullptr);
}

TEST(allocator, EngineOwningPointerProvidesRaiiTeardown) {
	oa::ExecutionSession* previous = oa::ExecutionSession::getActivePtr();
	{
		auto rt = createTestEngine(false);
		ASSERT_NE(rt.get(), nullptr);
		EXPECT_EQ(rt->getState(), oa::EngineState::Ready);
		EXPECT_EQ(oa::ExecutionSession::getActivePtr(), previous);
		// No explicit close(): the owning pointer must drain and release the
		// pinned engine without changing another engine's active context.
	}
	EXPECT_EQ(oa::ExecutionSession::getActivePtr(), previous);
}

TEST(allocator, FactoryEngineLifecycleIsStatefulAndCloseIsIdempotent) {
	oa::EngineConfig config;
	config.appName = "test_engine_state";
	config.selectForThread = false;
	config.preloadEmbeddedPipelines = false;
	auto result = oa::Engine::create(config);
	ASSERT_TRUE(result.isOk()) << result.getStatus().toString();
	auto engine = oa::move(*result);
	EXPECT_EQ(engine->getState(), oa::EngineState::Ready);
	EXPECT_TRUE(engine->hasCompute());

	EXPECT_TRUE(engine->close().isOk());
	EXPECT_EQ(engine->getState(), oa::EngineState::Destroyed);
	EXPECT_FALSE(engine->hasCompute());
	EXPECT_TRUE(engine->close().isOk());
	EXPECT_EQ(engine->getState(), oa::EngineState::Destroyed);
}

TEST(allocator, EngineOwnsTheContextUsedByFnMatrix) {
	oa::ExecutionSession* previous = oa::ExecutionSession::getActivePtr();
	for (const oa::Bool makeDefault : {true, false}) {
		oa::ExecutionSession::setActive(nullptr);
		auto result = oa::Engine::create(oa::EngineConfig{
			.selectForThread = makeDefault,
		});
		ASSERT_TRUE(result.isOk());
		auto engine = std::move(*result);
		auto& engineContext = oa::ExecutionSession::forEngine(*engine);
		EXPECT_EQ(oa::ExecutionSession::getActivePtr(),
			makeDefault ? &engineContext : nullptr);
		{
			oa::ExecutionSession::RecordingScope contextScope(engineContext);
			auto input = oa::FnMatrix::ones(oa::MatrixShape{16});
			auto output = oa::FnMatrix::scale(input, 3.0F);
			EXPECT_GT(engineContext.nodeCount(), 0U);
			ASSERT_TRUE(testSubmitAndWait(engineContext).isOk());
			oa::F32 values[16]{};
			ASSERT_TRUE(oa::FnMatrix::copyToHost(output, values, sizeof(values)).isOk());
			for (oa::F32 value : values) EXPECT_FLOAT_EQ(value, 3.0F);
		}
		EXPECT_TRUE(engine->close().isOk());
	}
	oa::ExecutionSession::setActive(previous);
}

TEST(allocator, StableMatrixStorageBelongsToExecutionSessionPolicy) {
	auto engine = createTestEngine(false);
	auto& context = oa::ExecutionSession::forEngine(*engine);

	context.beginStableResourceFrame();
	EXPECT_TRUE(context.isStableResourceFrameActive());
	auto first = context.allocateMatrixBuffer(4096, oa::MemoryPlacement::DeviceLocal);
	context.sealStableResourceInputs();
	EXPECT_TRUE(context.areStableResourceInputsSealed());
	auto second = context.allocateMatrixBuffer(8192, oa::MemoryPlacement::DeviceLocal);
	EXPECT_EQ(context.stableExternalResourceCount(), 1U);
	EXPECT_EQ(context.stableTransientResourceCount(), 1U);
	context.endStableResourceFrame();
	ASSERT_TRUE(first);
	ASSERT_TRUE(second);
	EXPECT_FALSE(context.isStableResourceFrameActive());

	context.beginStableResourceFrame();
	auto firstReused = context.allocateMatrixBuffer(2048, oa::MemoryPlacement::DeviceLocal);
	context.sealStableResourceInputs();
	auto secondReused = context.allocateMatrixBuffer(8192, oa::MemoryPlacement::DeviceLocal);
	context.endStableResourceFrame();

	EXPECT_EQ(firstReused.get(), first.get());
	EXPECT_EQ(secondReused.get(), second.get());
	EXPECT_EQ(firstReused->size, 2048U);
	EXPECT_GE(firstReused->capacity, 4096U);
}

TEST(allocator, MatrixOwnerDoesNotCallDestroyedEngine) {
	oa::Matrix retained;
	oa::Matrix retainedView;
	auto engine = createTestEngine(false);
	ASSERT_NE(engine, nullptr);
	{
		auto& engineContext = oa::ExecutionSession::forEngine(*engine);
		oa::ExecutionSession::RecordingScope contextScope(engineContext);
		retained = oa::FnMatrix::ones(oa::MatrixShape{16});
		retainedView = retained.view(oa::MatrixShape{4, 4});
		ASSERT_TRUE(retained.hasStorage());
		ASSERT_TRUE(retainedView.hasStorage());
		ASSERT_TRUE(testSubmitAndWait(engineContext).isOk());
	}

	ASSERT_TRUE(engine->close().isOk());
	// Close owns allocation teardown and mutates the shared wrapper seen by
	// every retained matrix/view. No stale descriptor or mapped pointer remains.
	EXPECT_FALSE(retained.hasStorage());
	EXPECT_FALSE(retainedView.hasStorage());
	EXPECT_EQ(retained.data(), nullptr);
	EXPECT_EQ(retainedView.data(), nullptr);
	EXPECT_EQ(retained.heapSlot(), -1);
	EXPECT_EQ(retainedView.heapSlot(), -1);
	EXPECT_EQ(retained.hostBlock().ptr, nullptr);
	EXPECT_EQ(retainedView.hostBlock().ptr, nullptr);
	EXPECT_EQ(oa::MatrixAccess::descriptor(retained).buffer, nullptr);
	EXPECT_EQ(oa::MatrixAccess::descriptor(retained).allocation, nullptr);

	// Dropping the wrappers after Close must not touch the dead allocator.
	retained = {};
	retainedView = {};
}

TEST(allocator, EngineCloseDrainsAliasLeasesBeforeAllocatorTeardown) {
	auto engine = createTestEngine(false);
	ASSERT_NE(engine, nullptr);

	constexpr oa::U32 N = 64;
	oa::Vector<oa::Matrix> matrices;
	{
		auto& engineContext = oa::ExecutionSession::forEngine(*engine);
		oa::ExecutionSession::RecordingScope contextScope(engineContext);
		for (oa::U32 i = 0; i < 5U; ++i) {
			matrices.pushBack(oa::FnMatrix::empty(
				{static_cast<oa::I64>(N)}, oa::ScalarType::Float32,
				oa::MemoryPlacement::HostUpload));
			ASSERT_TRUE(matrices.back().hasStorage());
		}
	}

	oa::Vector<oavk::Buffer> buffers;
	for (const auto& matrix : matrices) {
		buffers.pushBack(oa::MatrixAccess::descriptor(matrix));
	}
	oa::ExecutableGraph graph;
	struct PushConstants { oa::U32 Count; oa::F32 scale; } push{N, 1.001F};
	for (oa::U32 i = 0; i < 4U; ++i) {
		oavk::Buffer dispatchBuffers[] = {buffers[i], buffers[i + 1U]};
		oa::BufferAccess access[] = {oa::BufferAccess::Read, oa::BufferAccess::Write};
		graph.add("scale", dispatchBuffers, access, &push, sizeof(push), 1);
	}
	oa::Matrix* eligible[] = {&matrices[1], &matrices[3]};
	ASSERT_TRUE(graph.materializeAliases(*engine, eligible).isOk());
	ASSERT_NE(oa::MatrixAccess::descriptor(matrices[1]).buffer, nullptr);
	ASSERT_EQ(matrices[1].data(), matrices[3].data());

	ASSERT_TRUE(engine->close().isOk());
	for (const oa::Usize index : {oa::Usize{1}, oa::Usize{3}}) {
		EXPECT_FALSE(matrices[index].hasStorage());
		EXPECT_EQ(matrices[index].data(), nullptr);
		EXPECT_EQ(matrices[index].heapSlot(), -1);
		EXPECT_EQ(oa::MatrixAccess::descriptor(matrices[index]).buffer, nullptr);
		EXPECT_EQ(oa::MatrixAccess::descriptor(matrices[index]).allocation, nullptr);
	}
	// The graph and matrices intentionally outlive close(). Their eventual
	// shared-owner release must only delete the already-inert wrappers.
}

TEST(allocator, BindlessExhaustionFailsAllocationWithoutLeakingStorage) {
	auto engine = createTestEngine(false);
	auto& rt = *engine;

	auto tinyResult = oavk::BindlessHeap::create(
		oa::EngineDeviceAccess::get(rt),
		oavk::BindlessCapacities{.buffers = 2, .images = 2, .samplers = 2});
	if (not tinyResult.isOk()) {
		FAIL() << tinyResult.getStatus().toString();
		return;
	}

	auto& bindless = oa::EngineBindlessAccess::get(rt);
	auto originalHeap = std::move(bindless);
	bindless = std::move(*tinyResult);
	auto first = oa::EngineResourceAccess::allocBufferDevice(rt, 4096);
	const RuntimeAllocatorStats beforeFailure = oa::EngineAllocatorAccess::get(rt).getStats();
	auto exhausted = oa::EngineResourceAccess::allocBufferDevice(rt, 4096);
	const RuntimeAllocatorStats afterFailure = oa::EngineAllocatorAccess::get(rt).getStats();

	EXPECT_TRUE(first.isOk()) << first.getStatus().toString();
	EXPECT_FALSE(exhausted.isOk());
	if (not exhausted.isOk()) {
		EXPECT_EQ(exhausted.getStatus().getCode(), oa::StatusCode::ResourceExhausted);
	}
	EXPECT_EQ(afterFailure.allocationCount, beforeFailure.allocationCount);

	if (first.isOk()) oa::EngineResourceAccess::freeBuffer(rt, *first);
	bindless.destroy(oa::EngineDeviceAccess::get(rt));
	bindless = std::move(originalHeap);
}

TEST(allocator, AllocBarFallback) {
	auto rtP = createTestEngine(); oa::Engine& rt = *rtP;
	{
		const oa::StringView name = rt.deviceName();
		fprintf(stderr, "  [test_allocator] device: %.*s\n",
			static_cast<int>(name.size()), name.data());
	}

	auto result = oa::EngineAllocatorAccess::get(rt).allocBar(4096);
	ASSERT_TRUE(result.isOk()) << result.getStatus().toString();
	auto& buf = result.getValue();

	EXPECT_NE(buf.buffer, nullptr);
	EXPECT_NE(buf.mappedPtr, nullptr);
	EXPECT_EQ(buf.size, 4096u);
	EXPECT_TRUE(buf.supportsIndirectDispatch());

	if (oa::EngineDeviceAccess::get(rt).info.hardware.hasSAM) {
		EXPECT_TRUE(buf.isBar());
		fprintf(stderr, "  allocBar: SAM detected, BAR flag set\n");
	} else {
		fprintf(stderr, "  allocBar: no SAM, fell back to host-visible\n");
	}

	oa::EngineAllocatorAccess::get(rt).free(buf);
	EXPECT_TRUE(rt.close().isOk());
}

TEST(allocator, PlacementMetadataMatchesAllocationContract) {
	auto rtP = createTestEngine(); oa::Engine& rt = *rtP;

	auto deviceResult = oa::EngineAllocatorAccess::get(rt).allocDevice(4096);
	ASSERT_TRUE(deviceResult.isOk()) << deviceResult.getStatus().toString();
	auto device = std::move(deviceResult).getValue();
	EXPECT_EQ(device.size, 4096u);
	EXPECT_EQ(device.capacity, 4096u);
	EXPECT_EQ(device.placement, oa::MemoryPlacement::DeviceLocal);
	EXPECT_FALSE(device.isHostVisible());
	EXPECT_TRUE(device.supportsIndirectDispatch());

	auto uploadResult = oa::EngineAllocatorAccess::get(rt).allocHostVisible(4096);
	ASSERT_TRUE(uploadResult.isOk()) << uploadResult.getStatus().toString();
	auto upload = std::move(uploadResult).getValue();
	EXPECT_EQ(upload.placement, oa::MemoryPlacement::HostUpload);
	EXPECT_TRUE(upload.isHostVisible());
	EXPECT_TRUE(upload.supportsIndirectDispatch());

	auto readbackResult = oa::EngineAllocatorAccess::get(rt).allocHostReadback(4096);
	ASSERT_TRUE(readbackResult.isOk()) << readbackResult.getStatus().toString();
	auto readback = std::move(readbackResult).getValue();
	EXPECT_EQ(readback.placement, oa::MemoryPlacement::HostReadback);
	EXPECT_TRUE(readback.isHostVisible());
	EXPECT_TRUE(readback.supportsIndirectDispatch());

	oa::EngineAllocatorAccess::get(rt).free(device);
	oa::EngineAllocatorAccess::get(rt).free(upload);
	oa::EngineAllocatorAccess::get(rt).free(readback);
	EXPECT_TRUE(rt.close().isOk());
}

TEST(allocator, DeviceLocalUploadReadbackRoundTrip) {
	auto rtP = createTestEngine(); oa::Engine& rt = *rtP;
	constexpr oa::U32 kCount = 1024;
	constexpr oa::U64 kBytes = kCount * sizeof(oa::U32);

	auto result = oa::EngineResourceAccess::allocBuffer(rt, kBytes, oa::MemoryPlacement::DeviceLocal);
	ASSERT_TRUE(result.isOk()) << result.getStatus().toString();
	auto buffer = std::move(result).getValue();
	ASSERT_EQ(buffer.placement, oa::MemoryPlacement::DeviceLocal);

	oa::Vector<oa::U32> source(kCount);
	oa::Vector<oa::U32> destination(kCount);
	for (oa::U32 i = 0; i < kCount; ++i) source[i] = i * 2654435761U;

	ASSERT_TRUE(oa::EngineResourceAccess::uploadBuffer(rt, buffer, 0, source.data(), kBytes).isOk());
	ASSERT_TRUE(oa::EngineResourceAccess::readbackBuffer(rt, buffer, 0, destination.data(), kBytes).isOk());
	for (oa::U32 i = 0; i < kCount; ++i) EXPECT_EQ(destination[i], source[i]);
	const RuntimeAllocatorStats firstReadbackStats = oa::EngineAllocatorAccess::get(rt).getStats();
	for (oa::U32 repeat = 0; repeat < 8; ++repeat) {
		ASSERT_TRUE(oa::EngineResourceAccess::readbackBuffer(rt, buffer, 0, destination.data(), kBytes).isOk());
	}
	const RuntimeAllocatorStats repeatedReadbackStats = oa::EngineAllocatorAccess::get(rt).getStats();
	EXPECT_EQ(repeatedReadbackStats.allocationCount, firstReadbackStats.allocationCount);
	EXPECT_EQ(repeatedReadbackStats.allocationBytes, firstReadbackStats.allocationBytes);

	oa::EngineResourceAccess::freeBuffer(rt, buffer);

	// Byte and BF16 matrices are not necessarily four-byte sized. The public
	// transfer contract remains byte-addressable even though vulkan buffer-copy
	// commands operate on aligned words.
	auto byteResult = oa::EngineResourceAccess::allocBuffer(rt, 7, oa::MemoryPlacement::DeviceLocal);
	ASSERT_TRUE(byteResult.isOk()) << byteResult.getStatus().toString();
	auto byteBuffer = std::move(byteResult).getValue();
	const oa::U8 initial[7] = {1, 2, 3, 4, 5, 6, 7};
	const oa::U8 patch[3] = {9, 10, 11};
	oa::U8 bytes[7]{};
	ASSERT_TRUE(oa::EngineResourceAccess::uploadBuffer(rt, byteBuffer, 0, initial, sizeof(initial)).isOk());
	ASSERT_TRUE(oa::EngineResourceAccess::uploadBuffer(rt, byteBuffer, 1, patch, sizeof(patch)).isOk());
	ASSERT_TRUE(oa::EngineResourceAccess::readbackBuffer(rt, byteBuffer, 0, bytes, sizeof(bytes)).isOk());
	const oa::U8 expected[7] = {1, 9, 10, 11, 5, 6, 7};
	for (oa::U32 i = 0; i < 7; ++i) EXPECT_EQ(bytes[i], expected[i]);
	oa::EngineResourceAccess::freeBuffer(rt, byteBuffer);
	EXPECT_TRUE(rt.close().isOk());
}

TEST(allocator, MappedInPlaceUploadPublishesAndTracksMutation) {
	auto rtP = createTestEngine(); oa::Engine& rt = *rtP;
	constexpr oa::U64 kBytes = 4U * sizeof(oa::U32);
	auto result = oa::EngineResourceAccess::allocBuffer(rt, kBytes, oa::MemoryPlacement::HostUpload);
	ASSERT_TRUE(result.isOk()) << result.getStatus().toString();
	auto buffer = std::move(result).getValue();
	ASSERT_NE(buffer.mappedPtr, nullptr);

	auto* mapped = static_cast<oa::U32*>(buffer.mappedPtr);
	for (oa::U32 i = 0; i < 4U; ++i) mapped[i] = 0x10203040U + i;
	const oa::U64 before = buffer.currentMutationVersion();
	ASSERT_TRUE(oa::EngineResourceAccess::uploadBuffer(rt, buffer, 0, buffer.mappedPtr, kBytes).isOk());
	EXPECT_GT(buffer.currentMutationVersion(), before);

	oa::U32 readback[4]{};
	ASSERT_TRUE(oa::EngineResourceAccess::readbackBuffer(rt, buffer, 0, readback, kBytes).isOk());
	for (oa::U32 i = 0; i < 4U; ++i) EXPECT_EQ(readback[i], 0x10203040U + i);

	oa::EngineResourceAccess::freeBuffer(rt, buffer);
	EXPECT_TRUE(rt.close().isOk());
}

TEST(allocator, UploadWeightsCorrectness) {
	auto rtP = createTestEngine(); oa::Engine& rt = *rtP;

	constexpr oa::U32 N = 1024;
	constexpr oa::U64 size = N * sizeof(oa::F32);

	auto result = oa::EngineAllocatorAccess::get(rt).allocBar(size);
	ASSERT_TRUE(result.isOk());
	auto& buf = result.getValue();

	oa::Vector<oa::F32> src(N);
	for (oa::U32 i = 0; i < N; ++i) {
		src[i] = static_cast<oa::F32>(i) * 0.5f;
	}

	auto status = oa::EngineAllocatorAccess::get(rt).uploadWeights(buf, src.data(), size);
	ASSERT_TRUE(status.isOk()) << status.toString();

	auto* dst = static_cast<oa::F32*>(buf.mappedPtr);
	for (oa::U32 i = 0; i < N; ++i) {
		EXPECT_FLOAT_EQ(dst[i], src[i]) << "mismatch at " << i;
	}

	oa::EngineAllocatorAccess::get(rt).free(buf);
	EXPECT_TRUE(rt.close().isOk());
}

TEST(allocator, BudgetQuery) {
	auto rtP = createTestEngine(); oa::Engine& rt = *rtP;

	auto stats = oa::EngineAllocatorAccess::get(rt).getStats();
	const auto memory = rt.getMemoryUsage();
	fprintf(stderr, "  budget: used=%llu MB, budget=%llu MB\n",
		(unsigned long long)(stats.usedBytes / (1024 * 1024)),
		(unsigned long long)(stats.budgetBytes / (1024 * 1024)));

	EXPECT_GT(stats.budgetBytes, 0u);
	EXPECT_GT(memory.totalBytes, 0u);
	EXPECT_LE(memory.usedBytes, memory.totalBytes);
	EXPECT_EQ(memory.freeBytes, memory.totalBytes - memory.usedBytes);
	EXPECT_GE(memory.usedPercent, 0.0);
	EXPECT_LE(memory.usedPercent, 100.0);

	EXPECT_TRUE(rt.close().isOk());
}
