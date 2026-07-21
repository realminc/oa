// ═══════════════════════════════════════════════════════════════════════════════
// Test: OaVkStream — persistent async compute streams
// ═══════════════════════════════════════════════════════════════════════════════

#include "../../OaTest.h"
#include <Oa/Runtime/Stream.h>
#include <Oa/Runtime/UploadRing.h>
#include <Oa/Runtime/Prefetch.h>
#include <Oa/Runtime/Engine.h>
#include <Oa/Core/Simd.h>
#include <Oa/Core/Memory.h>
#include <thread>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <mutex>

// ─── Stream Lifecycle ─────────────────────────────────────────────────────────

TEST(VkStream, CreateDestroy) {
	auto* rt = OaEngine::GetGlobal();
	ASSERT_NE(rt, nullptr);

	auto res = OaVkStream::CreateCompute(rt->Device);
	ASSERT_TRUE(res.IsOk());
	auto stream = std::move(*res);

	EXPECT_NE(stream.CommandPool, nullptr);
	EXPECT_NE(stream.CommandBuffer, nullptr);
	EXPECT_NE(stream.TimelineSem.Semaphore, nullptr);
	EXPECT_EQ(stream.TimelineValue, 0u);
	EXPECT_FALSE(stream.Recording);
	EXPECT_FALSE(stream.Submitted);

	stream.Destroy(rt->Device);
	EXPECT_EQ(stream.CommandPool, nullptr);
}

// ─── Begin/Submit Cycle ───────────────────────────────────────────────────────

TEST(VkStream, BeginSubmitCycle) {
	auto* rt = OaEngine::GetGlobal();
	ASSERT_NE(rt, nullptr);

	auto res = OaVkStream::CreateCompute(rt->Device);
	ASSERT_TRUE(res.IsOk());
	auto stream = std::move(*res);

	ASSERT_TRUE(stream.Begin(rt->Device).IsOk());
	EXPECT_TRUE(stream.Recording);

	stream.RecordBufferBarrier();

	auto status = stream.SubmitAndWait(*rt);
	EXPECT_TRUE(status.IsOk());
	EXPECT_FALSE(stream.Recording);
	EXPECT_TRUE(stream.IsComplete(rt->Device));

	stream.Destroy(rt->Device);
}

// ─── Stream Reuse (multiple submit cycles) ───────────────────────────────────

TEST(VkStream, Reuse) {
	auto* rt = OaEngine::GetGlobal();
	ASSERT_NE(rt, nullptr);

	auto res = OaVkStream::CreateCompute(rt->Device);
	ASSERT_TRUE(res.IsOk());
	auto stream = std::move(*res);

	for (OaI32 i = 0; i < 5; ++i) {
		ASSERT_TRUE(stream.Begin(rt->Device).IsOk());
		stream.RecordBufferBarrier();
		ASSERT_TRUE(stream.SubmitAndWait(*rt).IsOk());
		EXPECT_TRUE(stream.IsComplete(rt->Device));
	}

	stream.Destroy(rt->Device);
}

// ─── Async Submit + Poll ──────────────────────────────────────────────────────

TEST(VkStream, AsyncSubmitPoll) {
	auto* rt = OaEngine::GetGlobal();
	ASSERT_NE(rt, nullptr);

	auto res = OaVkStream::CreateCompute(rt->Device);
	ASSERT_TRUE(res.IsOk());
	auto stream = std::move(*res);

	ASSERT_TRUE(stream.Begin(rt->Device).IsOk());
	stream.RecordBufferBarrier();
	ASSERT_TRUE(stream.Submit(*rt).IsOk());
	EXPECT_TRUE(stream.Submitted);

	ASSERT_TRUE(stream.Synchronize(rt->Device).IsOk());
	EXPECT_TRUE(stream.IsComplete(rt->Device));

	stream.Destroy(rt->Device);
}

// ─── Fence IsSignaled ─────────────────────────────────────────────────────────

TEST(VkFence, IsSignaled) {
	auto* rt = OaEngine::GetGlobal();
	ASSERT_NE(rt, nullptr);

	auto fRes = OaVkFence::Create(rt->Device, true);
	ASSERT_TRUE(fRes.IsOk());
	auto fence = std::move(*fRes);
	EXPECT_TRUE(fence.IsSignaled(rt->Device));

	fence.Reset(rt->Device);
	EXPECT_FALSE(fence.IsSignaled(rt->Device));

	fence.Destroy(rt->Device);
}

TEST(VkTimelineSemaphore, CompletionTokenSnapshotsWrapperHandle) {
	auto* rt = OaEngine::GetGlobal();
	ASSERT_NE(rt, nullptr);

	auto semaphoreResult = OaVkTimelineSemaphore::Create(rt->Device, 1U);
	ASSERT_TRUE(semaphoreResult.IsOk());
	auto semaphoreOwner = std::move(*semaphoreResult);
	OaCompletionToken completion(rt->Device, semaphoreOwner, 1U);
	OaCompletionToken completionCopy = completion;

	// Moving an owning facade may invalidate its wrapper object without
	// invalidating the Vulkan semaphore that backs an already-issued event.
	OaVkTimelineSemaphore movedOwner = semaphoreOwner;
	semaphoreOwner.Semaphore = nullptr;
	ASSERT_TRUE(completion.IsValid());
	ASSERT_TRUE(completionCopy.IsSameCompletion(completion));
	EXPECT_TRUE(completionCopy.IsComplete());
	EXPECT_TRUE(completionCopy.Wait().IsOk());

	movedOwner.Destroy(rt->Device);
}

// ─── Engine Stream Pool ───────────────────────────────────────────────────────

TEST(VkStream, PoolAcquireRelease) {
	auto* rt = OaEngine::GetGlobal();
	ASSERT_NE(rt, nullptr);

	OaVkStream* s1 = rt->AcquireStream();
	ASSERT_NE(s1, nullptr);
	EXPECT_NE(s1->CommandPool, nullptr);

	OaVkStream* s2 = rt->AcquireStream();
	ASSERT_NE(s2, nullptr);
	EXPECT_NE(s1, s2);

	rt->ReleaseStream(s1);
	OaVkStream* s3 = rt->AcquireStream();
	EXPECT_EQ(s3, s1);

	rt->ReleaseStream(s2);
	rt->ReleaseStream(s3);
}

// ─── RunOnce (single-shot dispatch) ──────────────────────────────────────────

TEST(VkStream, RunOnce) {
	auto* rt = OaEngine::GetGlobal();
	ASSERT_NE(rt, nullptr);

	auto srcRes = rt->Allocator.AllocHostVisible(256 * sizeof(OaF32));
	ASSERT_TRUE(srcRes.IsOk());
	auto src = srcRes.GetValue();
	rt->RegisterBuffer(src);

	auto dstRes = rt->Allocator.AllocHostVisible(256 * sizeof(OaF32));
	ASSERT_TRUE(dstRes.IsOk());
	auto dst = dstRes.GetValue();
	rt->RegisterBuffer(dst);

	OaF32* srcData = static_cast<OaF32*>(src.MappedPtr);
	for (OaI32 i = 0; i < 256; ++i) srcData[i] = static_cast<OaF32>(i);

	struct { OaU32 N; OaF32 Scale; } push{256, 2.0f};
	OaVkBuffer bufs[] = {src, dst};

	auto status = OaVkStream::RunOnce(*rt, "Scale", bufs, &push, sizeof(push), (256 + 255) / 256);

	if (status.IsOk()) {
		OaF32* out = static_cast<OaF32*>(dst.MappedPtr);
		for (OaI32 i = 0; i < 256; ++i) {
			EXPECT_NEAR(out[i], static_cast<OaF32>(i) * 2.0f, 1e-3f);
		}
	}

	rt->DeregisterBuffer(src);
	rt->Allocator.Free(src);
	rt->DeregisterBuffer(dst);
	rt->Allocator.Free(dst);
}

// ─── Dispatch Backward Compat ─────────────────────────────────────────────────

TEST(VkStream, DispatchBackwardCompat) {
	auto* rt = OaEngine::GetGlobal();
	ASSERT_NE(rt, nullptr);

	auto srcRes = rt->Allocator.AllocHostVisible(64 * sizeof(OaF32));
	ASSERT_TRUE(srcRes.IsOk());
	auto src = srcRes.GetValue();
	rt->RegisterBuffer(src);

	auto dstRes = rt->Allocator.AllocHostVisible(64 * sizeof(OaF32));
	ASSERT_TRUE(dstRes.IsOk());
	auto dst = dstRes.GetValue();
	rt->RegisterBuffer(dst);

	OaF32* data = static_cast<OaF32*>(src.MappedPtr);
	for (OaI32 i = 0; i < 64; ++i) data[i] = static_cast<OaF32>(i + 1);

	struct { OaU32 N; OaF32 Scale; } push{64, 0.5f};
	OaVkBuffer bufs[] = {src, dst};

	auto status = OaVkDispatch::Run(*rt, "Scale", bufs, &push, sizeof(push), 1);
	if (status.IsOk()) {
		OaF32* out = static_cast<OaF32*>(dst.MappedPtr);
		EXPECT_NEAR(out[0], 0.5f, 1e-3f);
		EXPECT_NEAR(out[63], 32.0f, 1e-3f);
	}

	rt->DeregisterBuffer(src);
	rt->Allocator.Free(src);
	rt->DeregisterBuffer(dst);
	rt->Allocator.Free(dst);
}

// ─── Batch Dispatch Backward Compat ───────────────────────────────────────────

TEST(VkStream, BatchDispatchCompat) {
	auto* rt = OaEngine::GetGlobal();
	ASSERT_NE(rt, nullptr);

	auto aRes = rt->Allocator.AllocHostVisible(64 * sizeof(OaF32));
	ASSERT_TRUE(aRes.IsOk());
	auto a = aRes.GetValue();
	rt->RegisterBuffer(a);

	auto bRes = rt->Allocator.AllocHostVisible(64 * sizeof(OaF32));
	ASSERT_TRUE(bRes.IsOk());
	auto b = bRes.GetValue();
	rt->RegisterBuffer(b);

	OaF32* data = static_cast<OaF32*>(a.MappedPtr);
	for (OaI32 i = 0; i < 64; ++i) data[i] = 1.0f;

	struct { OaU32 N; OaF32 Scale; } push{64, 3.0f};
	OaVkBuffer bufs[] = {a, b};

	auto batchRes = OaVkDispatch::BeginBatch(*rt);
	ASSERT_TRUE(batchRes.IsOk());
	auto batch = std::move(*batchRes);

	auto recSt = OaVkDispatch::Record(batch, *rt, "Scale", bufs, &push, sizeof(push), 1);
	if (!recSt.IsOk()) {
		rt->DeregisterBuffer(a);
		rt->Allocator.Free(a);
		rt->DeregisterBuffer(b);
		rt->Allocator.Free(b);
		GTEST_SKIP() << "scale shader not loaded";
	}

	ASSERT_TRUE(OaVkDispatch::Flush(batch, *rt).IsOk());

	OaF32* out = static_cast<OaF32*>(b.MappedPtr);
	EXPECT_NEAR(out[0], 3.0f, 1e-3f);

	rt->DeregisterBuffer(a);
	rt->Allocator.Free(a);
	rt->DeregisterBuffer(b);
	rt->Allocator.Free(b);
}

// ─── Indirect Dispatch ────────────────────────────────────────────────────────

TEST(VkDispatch, IndirectDispatch) {
	auto* rt = OaEngine::GetGlobal();
	ASSERT_NE(rt, nullptr);

	constexpr OaU32 N = 64;
	auto srcRes = rt->AllocBuffer(N * sizeof(OaF32));
	ASSERT_TRUE(srcRes.IsOk());
	auto src = std::move(*srcRes);

	auto dstRes = rt->AllocBuffer(N * sizeof(OaF32));
	ASSERT_TRUE(dstRes.IsOk());
	auto dst = std::move(*dstRes);

	OaF32* srcData = static_cast<OaF32*>(src.MappedPtr);
	for (OaU32 i = 0; i < N; ++i) srcData[i] = static_cast<OaF32>(i + 1);

	// Indirect buffer: VkDispatchIndirectCommand {groupsX, groupsY, groupsZ}
	auto indRes = rt->AllocBuffer(3 * sizeof(OaU32));
	ASSERT_TRUE(indRes.IsOk());
	auto indBuf = std::move(*indRes);
	OaU32* indData = static_cast<OaU32*>(indBuf.MappedPtr);
	indData[0] = (N + 255) / 256;
	indData[1] = 1;
	indData[2] = 1;

	struct { OaU32 N; OaF32 Scale; } push{N, 10.0f};
	OaVkBuffer bufs[] = {src, dst};

	auto status = OaVkDispatch::RunIndirect(
		*rt, "Scale", bufs, &push, sizeof(push), indBuf);

	if (status.IsOk()) {
		OaF32* out = static_cast<OaF32*>(dst.MappedPtr);
		for (OaU32 i = 0; i < N; ++i) {
			EXPECT_NEAR(out[i], static_cast<OaF32>(i + 1) * 10.0f, 1e-3f);
		}
	}

	rt->FreeBuffer(src);
	rt->FreeBuffer(dst);
	rt->FreeBuffer(indBuf);
}

// ─── Async Transfer ───────────────────────────────────────────────────────────

TEST(VkStream, AsyncTransfer) {
	auto* rt = OaEngine::GetGlobal();
	ASSERT_NE(rt, nullptr);

	constexpr OaU64 kSize = 1024;
	auto srcRes = rt->Allocator.AllocHostVisible(kSize);
	ASSERT_TRUE(srcRes.IsOk());
	auto src = srcRes.GetValue();
	rt->RegisterBuffer(src);

	auto dstRes = rt->Allocator.AllocHostVisible(kSize);
	ASSERT_TRUE(dstRes.IsOk());
	auto dst = dstRes.GetValue();
	rt->RegisterBuffer(dst);

	OaU8* srcData = static_cast<OaU8*>(src.MappedPtr);
	for (OaU32 i = 0; i < kSize; ++i) srcData[i] = static_cast<OaU8>(i & 0xFF);
	OaMemset(dst.MappedPtr, 0xA5, kSize);
	ASSERT_TRUE(rt->Allocator.FlushHostBuffer(src, 0U, kSize));

	auto firstCopy = rt->CopyBufferAsync(src, dst, kSize);
	ASSERT_TRUE(firstCopy.IsOk());
	ASSERT_TRUE(firstCopy->IsValid());
	ASSERT_TRUE(firstCopy->Wait().IsOk());
	EXPECT_TRUE(firstCopy->IsComplete());
	ASSERT_TRUE(rt->Allocator.InvalidateHostBuffer(dst, 0U, kSize));

	OaU8* dstData = static_cast<OaU8*>(dst.MappedPtr);
	for (OaU32 i = 0; i < kSize; ++i) {
		EXPECT_EQ(dstData[i], static_cast<OaU8>(i & 0xFF)) << "mismatch at " << i;
	}

	for (OaU32 i = 0; i < kSize; ++i) {
		srcData[i] = static_cast<OaU8>((i * 17U + 3U) & 0xFFU);
	}
	OaMemset(dst.MappedPtr, 0x5A, kSize);
	ASSERT_TRUE(rt->Allocator.FlushHostBuffer(src, 0U, kSize));
	auto secondCopy = rt->CopyBufferAsync(src, dst, kSize);
	ASSERT_TRUE(secondCopy.IsOk());
	ASSERT_TRUE(secondCopy->IsValid());
	EXPECT_EQ(secondCopy->Semaphore()->Semaphore,
		firstCopy->Semaphore()->Semaphore);
	EXPECT_GT(secondCopy->Value(), firstCopy->Value());
	ASSERT_TRUE(secondCopy->Wait().IsOk());
	ASSERT_TRUE(rt->Allocator.InvalidateHostBuffer(dst, 0U, kSize));
	for (OaU32 i = 0; i < kSize; ++i) {
		EXPECT_EQ(dstData[i], static_cast<OaU8>((i * 17U + 3U) & 0xFFU))
			<< "reuse mismatch at " << i;
	}

	rt->DeregisterBuffer(src);
	rt->Allocator.Free(src);
	rt->DeregisterBuffer(dst);
	rt->Allocator.Free(dst);
}

TEST(VkStream, AsyncTransferRejectsInvalidBufferContractsBeforeSubmission) {
	auto* rt = OaEngine::GetGlobal();
	ASSERT_NE(rt, nullptr);

	constexpr OaU64 kSize = 256;
	auto srcResult = rt->AllocBuffer(kSize);
	ASSERT_TRUE(srcResult.IsOk());
	auto src = std::move(*srcResult);
	auto dstResult = rt->AllocBuffer(kSize);
	ASSERT_TRUE(dstResult.IsOk());
	auto dst = std::move(*dstResult);

	OaVkBuffer nullBuffer;
	auto nullSource = rt->CopyBufferAsync(nullBuffer, dst, kSize);
	ASSERT_FALSE(nullSource.IsOk());
	EXPECT_EQ(nullSource.GetStatus().GetCode(), OaStatusCode::InvalidArgument);
	auto nullDestination = rt->CopyBufferAsync(src, nullBuffer, kSize);
	ASSERT_FALSE(nullDestination.IsOk());
	EXPECT_EQ(nullDestination.GetStatus().GetCode(), OaStatusCode::InvalidArgument);
	auto zeroSize = rt->CopyBufferAsync(src, dst, 0U);
	ASSERT_FALSE(zeroSize.IsOk());
	EXPECT_EQ(zeroSize.GetStatus().GetCode(), OaStatusCode::InvalidArgument);
	auto oversized = rt->CopyBufferAsync(src, dst, kSize + 1U);
	ASSERT_FALSE(oversized.IsOk());
	EXPECT_EQ(oversized.GetStatus().GetCode(), OaStatusCode::InvalidArgument);
	auto overlapping = rt->CopyBufferAsync(src, src, kSize);
	ASSERT_FALSE(overlapping.IsOk());
	EXPECT_EQ(overlapping.GetStatus().GetCode(), OaStatusCode::InvalidArgument);
	OaVkBuffer foreignSource = src;
	foreignSource.NodeIndex = 1U;
	auto foreign = rt->CopyBufferAsync(foreignSource, dst, kSize);
	ASSERT_FALSE(foreign.IsOk());
	EXPECT_EQ(foreign.GetStatus().GetCode(), OaStatusCode::InvalidArgument);
	OaVkBuffer importedSource = src;
	importedSource.Flags |= OA_VK_BUFFER_FLAG_IMPORTED;
	auto imported = rt->CopyBufferAsync(importedSource, dst, kSize);
	ASSERT_FALSE(imported.IsOk());
	EXPECT_EQ(imported.GetStatus().GetCode(), OaStatusCode::InvalidArgument);
	OaVkBuffer externalSource = src;
	externalSource.Allocation = nullptr;
	auto external = rt->CopyBufferAsync(externalSource, dst, kSize);
	ASSERT_FALSE(external.IsOk());
	EXPECT_EQ(external.GetStatus().GetCode(), OaStatusCode::InvalidArgument);
	OaVkBuffer otherEngineSource = src;
	otherEngineSource.AllocatorIdentity = &otherEngineSource;
	auto otherEngine = rt->CopyBufferAsync(otherEngineSource, dst, kSize);
	ASSERT_FALSE(otherEngine.IsOk());
	EXPECT_EQ(otherEngine.GetStatus().GetCode(), OaStatusCode::InvalidArgument);
	OaVkBuffer aliasDestination = dst;
	aliasDestination.AliasIdentity = dst.Allocation;
	auto alias = rt->CopyBufferAsync(src, aliasDestination, kSize);
	ASSERT_FALSE(alias.IsOk());
	EXPECT_EQ(alias.GetStatus().GetCode(), OaStatusCode::InvalidArgument);

	rt->FreeBuffer(src);
	rt->FreeBuffer(dst);
}

TEST(VkStream, AsyncTransferRejectsOverlappingInFlightSubmission) {
	auto* rt = OaEngine::GetGlobal();
	ASSERT_NE(rt, nullptr);

	constexpr OaU64 kSize = 4096;
	auto srcResult = rt->AllocBuffer(kSize);
	ASSERT_TRUE(srcResult.IsOk());
	auto src = std::move(*srcResult);
	auto firstDstResult = rt->AllocBuffer(kSize);
	ASSERT_TRUE(firstDstResult.IsOk());
	auto firstDst = std::move(*firstDstResult);
	auto secondDstResult = rt->AllocBuffer(kSize);
	ASSERT_TRUE(secondDstResult.IsOk());
	auto secondDst = std::move(*secondDstResult);
	OaMemset(src.MappedPtr, 0x6D, kSize);
	ASSERT_TRUE(rt->Allocator.FlushHostBuffer(src, 0U, kSize));

	auto gateResult = OaVkTimelineSemaphore::Create(rt->Device, 0U);
	ASSERT_TRUE(gateResult.IsOk());
	auto gate = std::move(*gateResult);
	OaVkStream* blocker = rt->AcquireStream();
	ASSERT_NE(blocker, nullptr);
	ASSERT_TRUE(blocker->Begin(rt->Device).IsOk());
	ASSERT_TRUE(blocker->SubmitWithDependency(*rt, gate, 1U).IsOk());

	// The compute queue cannot reach either copy until the host signals gate.
	// This makes the one-flight rejection deterministic rather than timing based.
	auto firstCopy = rt->CopyBufferAsync(src, firstDst, kSize);
	std::mutex watchdogMutex;
	std::condition_variable watchdogCondition;
	bool copyCallReturned = false;
	bool watchdogTimedOut = false;
	std::atomic<OaI32> gateSignalResult{static_cast<OaI32>(VK_NOT_READY)};
	auto signalGate = [&]() {
		VkSemaphoreSignalInfo signalInfo = {
			.sType = VK_STRUCTURE_TYPE_SEMAPHORE_SIGNAL_INFO,
			.pNext = nullptr,
			.semaphore = static_cast<VkSemaphore>(gate.Semaphore),
			.value = 1U,
		};
		gateSignalResult.store(static_cast<OaI32>(vkSignalSemaphore(
			static_cast<VkDevice>(rt->Device.Device), &signalInfo)),
			std::memory_order_release);
	};
	std::thread gateWatchdog([&]() {
		bool releaseGate = false;
		{
			std::unique_lock lock(watchdogMutex);
			releaseGate = not watchdogCondition.wait_for(lock,
				std::chrono::seconds(5), [&]() { return copyCallReturned; });
			watchdogTimedOut = releaseGate;
		}
		if (releaseGate) signalGate();
	});
	auto rejectedCopy = rt->CopyBufferAsync(src, secondDst, kSize);
	{
		std::lock_guard lock(watchdogMutex);
		copyCallReturned = true;
	}
	watchdogCondition.notify_one();
	gateWatchdog.join();
	if (not watchdogTimedOut) signalGate();
	EXPECT_FALSE(watchdogTimedOut)
		<< "CopyBufferAsync did not reject before the five-second watchdog timeout";
	EXPECT_EQ(gateSignalResult.load(std::memory_order_acquire),
		static_cast<OaI32>(VK_SUCCESS));
	if (firstCopy.IsOk()) EXPECT_TRUE(firstCopy->Wait().IsOk());
	if (rejectedCopy.IsOk()) EXPECT_TRUE(rejectedCopy->Wait().IsOk());
	EXPECT_TRUE(blocker->Synchronize(rt->Device).IsOk());
	rt->ReleaseStream(blocker);
	gate.Destroy(rt->Device);

	ASSERT_TRUE(firstCopy.IsOk());
	ASSERT_FALSE(rejectedCopy.IsOk());
	EXPECT_EQ(rejectedCopy.GetStatus().GetCode(),
		OaStatusCode::FailedPrecondition);
	rt->FreeBuffer(src);
	rt->FreeBuffer(firstDst);
	rt->FreeBuffer(secondDst);
}

TEST(VkStream, AsyncTransferEventChainsGpuCopyConsumer) {
	auto* rt = OaEngine::GetGlobal();
	ASSERT_NE(rt, nullptr);

	constexpr OaU64 kSize = 4096;
	auto srcResult = rt->AllocBuffer(kSize);
	ASSERT_TRUE(srcResult.IsOk());
	auto src = std::move(*srcResult);
	auto intermediateResult = rt->AllocBufferDevice(kSize);
	ASSERT_TRUE(intermediateResult.IsOk());
	auto intermediate = std::move(*intermediateResult);
	auto dstResult = rt->AllocBuffer(kSize);
	ASSERT_TRUE(dstResult.IsOk());
	auto dst = std::move(*dstResult);

	auto* source = static_cast<OaU8*>(src.MappedPtr);
	for (OaU64 index = 0; index < kSize; ++index) {
		source[index] = static_cast<OaU8>((index * 31U + 11U) & 0xFFU);
	}
	OaMemset(dst.MappedPtr, 0xC3, kSize);
	ASSERT_TRUE(rt->Allocator.FlushHostBuffer(src, 0U, kSize));

	auto producer = rt->CopyBufferAsync(src, intermediate, kSize);
	ASSERT_TRUE(producer.IsOk());
	OaVkStream* consumer = rt->AcquireStream();
	ASSERT_NE(consumer, nullptr);
	ASSERT_TRUE(consumer->Begin(rt->Device).IsOk());
	consumer->RecordCopyBuffer(intermediate, dst, kSize);
	consumer->RecordTransferWriteBarrier(dst, 0U, kSize);
	const OaVkTimelineWait wait = producer->TimelineWait();
	ASSERT_TRUE(consumer->SubmitWithDependencies(
		*rt, OaSpan<const OaVkTimelineWait>(&wait, 1)).IsOk());
	ASSERT_TRUE(consumer->Synchronize(rt->Device).IsOk());
	rt->ReleaseStream(consumer);
	EXPECT_TRUE(producer->IsComplete());
	ASSERT_TRUE(rt->Allocator.InvalidateHostBuffer(dst, 0U, kSize));
	EXPECT_TRUE(OaMemEqual(source, dst.MappedPtr, kSize));

	rt->FreeBuffer(src);
	rt->FreeBuffer(intermediate);
	rt->FreeBuffer(dst);
}

TEST(VkStream, MappedReadbackWaitsForPrimaryQueueProducer) {
	auto* rt = OaEngine::GetGlobal();
	ASSERT_NE(rt, nullptr);

	constexpr OaU64 kSize = 4096;
	auto srcResult = rt->AllocBuffer(kSize);
	ASSERT_TRUE(srcResult.IsOk());
	auto src = std::move(*srcResult);
	auto dstResult = rt->AllocBuffer(kSize);
	ASSERT_TRUE(dstResult.IsOk());
	auto dst = std::move(*dstResult);
	auto* source = static_cast<OaU8*>(src.MappedPtr);
	for (OaU64 index = 0; index < kSize; ++index) {
		source[index] = static_cast<OaU8>((index * 31U + 9U) & 0xFFU);
	}
	OaMemset(dst.MappedPtr, 0xD3, kSize);
	ASSERT_TRUE(rt->Allocator.FlushHostBuffer(src, 0U, kSize));
	ASSERT_TRUE(rt->Allocator.FlushHostBuffer(dst, 0U, kSize));

	auto gateResult = OaVkTimelineSemaphore::Create(rt->Device, 0U);
	ASSERT_TRUE(gateResult.IsOk());
	auto gate = std::move(*gateResult);
	OaVkStream* producer = rt->AcquireStream();
	ASSERT_NE(producer, nullptr);
	ASSERT_TRUE(producer->Begin(rt->Device).IsOk());
	producer->RecordCopyBuffer(src, dst, kSize);
	ASSERT_TRUE(producer->SubmitWithDependency(*rt, gate, 1U).IsOk());

	std::atomic<OaI32> gateSignalResult{static_cast<OaI32>(VK_NOT_READY)};
	std::thread gateWatchdog([&]() {
		std::this_thread::sleep_for(std::chrono::milliseconds(100));
		VkSemaphoreSignalInfo signalInfo = {
			.sType = VK_STRUCTURE_TYPE_SEMAPHORE_SIGNAL_INFO,
			.pNext = nullptr,
			.semaphore = static_cast<VkSemaphore>(gate.Semaphore),
			.value = 1U,
		};
		gateSignalResult.store(static_cast<OaI32>(vkSignalSemaphore(
			static_cast<VkDevice>(rt->Device.Device), &signalInfo)),
			std::memory_order_release);
	});
	OaU8 output[kSize] = {};
	const OaStatus readbackStatus = rt->ReadbackBuffer(
		dst, 0U, output, kSize);
	gateWatchdog.join();
	EXPECT_EQ(gateSignalResult.load(std::memory_order_acquire),
		static_cast<OaI32>(VK_SUCCESS));
	EXPECT_TRUE(readbackStatus.IsOk()) << readbackStatus.GetMessage();
	EXPECT_TRUE(OaMemEqual(source, output, kSize));
	EXPECT_TRUE(producer->Synchronize(rt->Device).IsOk());
	rt->ReleaseStream(producer);
	gate.Destroy(rt->Device);
	rt->FreeBuffer(src);
	rt->FreeBuffer(dst);
}

TEST(VkStream, AsyncTransferPublishesPriorPrimaryQueueWrites) {
	auto* rt = OaEngine::GetGlobal();
	ASSERT_NE(rt, nullptr);

	constexpr OaU64 kSize = 4096;
	auto srcResult = rt->AllocBuffer(kSize);
	ASSERT_TRUE(srcResult.IsOk());
	auto src = std::move(*srcResult);
	auto intermediateResult = rt->AllocBufferDevice(kSize);
	ASSERT_TRUE(intermediateResult.IsOk());
	auto intermediate = std::move(*intermediateResult);
	auto dstResult = rt->AllocBuffer(kSize);
	ASSERT_TRUE(dstResult.IsOk());
	auto dst = std::move(*dstResult);

	auto* source = static_cast<OaU8*>(src.MappedPtr);
	for (OaU64 index = 0; index < kSize; ++index) {
		source[index] = static_cast<OaU8>((index * 13U + 19U) & 0xFFU);
	}
	OaMemset(dst.MappedPtr, 0xB7, kSize);
	ASSERT_TRUE(rt->Allocator.FlushHostBuffer(src, 0U, kSize));

	OaVkStream* producer = rt->AcquireStream();
	ASSERT_NE(producer, nullptr);
	ASSERT_TRUE(producer->Begin(rt->Device).IsOk());
	// Deliberately omit a producer-side post-copy barrier. CopyBufferAsync must
	// publish this earlier primary-queue write before reading the source.
	producer->RecordCopyBuffer(src, intermediate, kSize);
	ASSERT_TRUE(producer->Submit(*rt).IsOk());

	auto copied = rt->CopyBufferAsync(intermediate, dst, kSize);
	ASSERT_TRUE(copied.IsOk()) << copied.GetStatus().GetMessage();
	ASSERT_TRUE(copied->Wait().IsOk());
	ASSERT_TRUE(rt->Allocator.InvalidateHostBuffer(dst, 0U, kSize));
	EXPECT_TRUE(OaMemEqual(source, dst.MappedPtr, kSize));
	EXPECT_TRUE(producer->Synchronize(rt->Device).IsOk());
	rt->ReleaseStream(producer);

	rt->FreeBuffer(src);
	rt->FreeBuffer(intermediate);
	rt->FreeBuffer(dst);
}

TEST(VkStream, DeviceLocalReadbackPublishesPriorPrimaryQueueWrites) {
	auto* rt = OaEngine::GetGlobal();
	ASSERT_NE(rt, nullptr);

	constexpr OaU64 kSize = 4096;
	auto srcResult = rt->AllocBuffer(kSize);
	ASSERT_TRUE(srcResult.IsOk());
	auto src = std::move(*srcResult);
	auto deviceResult = rt->AllocBufferDevice(kSize);
	ASSERT_TRUE(deviceResult.IsOk());
	auto device = std::move(*deviceResult);
	auto* source = static_cast<OaU8*>(src.MappedPtr);
	for (OaU64 index = 0; index < kSize; ++index) {
		source[index] = static_cast<OaU8>((index * 47U + 3U) & 0xFFU);
	}
	ASSERT_TRUE(rt->Allocator.FlushHostBuffer(src, 0U, kSize));

	OaVkStream* producer = rt->AcquireStream();
	ASSERT_NE(producer, nullptr);
	ASSERT_TRUE(producer->Begin(rt->Device).IsOk());
	// ReadbackBuffer owns the producer-write to transfer-read dependency.
	producer->RecordCopyBuffer(src, device, kSize);
	ASSERT_TRUE(producer->Submit(*rt).IsOk());

	OaU8 output[kSize] = {};
	const OaStatus readback = rt->ReadbackBuffer(device, 0U, output, kSize);
	ASSERT_TRUE(readback.IsOk()) << readback.GetMessage();
	EXPECT_TRUE(OaMemEqual(source, output, kSize));
	EXPECT_TRUE(producer->Synchronize(rt->Device).IsOk());
	rt->ReleaseStream(producer);
	rt->FreeBuffer(src);
	rt->FreeBuffer(device);
}

TEST(VkStream, ReadbackRejectsUnownedAndAliasedBuffers) {
	auto* rt = OaEngine::GetGlobal();
	ASSERT_NE(rt, nullptr);

	constexpr OaU64 kSize = 256;
	auto srcResult = rt->AllocBuffer(kSize);
	ASSERT_TRUE(srcResult.IsOk());
	auto src = std::move(*srcResult);
	OaU8 output[kSize] = {};

	OaVkBuffer foreignNode = src;
	foreignNode.NodeIndex = 1U;
	EXPECT_EQ(rt->ReadbackBuffer(foreignNode, 0U, output, kSize).GetCode(),
		OaStatusCode::InvalidArgument);
	OaVkBuffer imported = src;
	imported.Flags |= OA_VK_BUFFER_FLAG_IMPORTED;
	EXPECT_EQ(rt->ReadbackBuffer(imported, 0U, output, kSize).GetCode(),
		OaStatusCode::InvalidArgument);
	OaVkBuffer external = src;
	external.Allocation = nullptr;
	EXPECT_EQ(rt->ReadbackBuffer(external, 0U, output, kSize).GetCode(),
		OaStatusCode::InvalidArgument);
	OaVkBuffer otherEngine = src;
	otherEngine.AllocatorIdentity = &otherEngine;
	EXPECT_EQ(rt->ReadbackBuffer(otherEngine, 0U, output, kSize).GetCode(),
		OaStatusCode::InvalidArgument);
	OaVkBuffer alias = src;
	alias.AliasIdentity = src.Allocation;
	EXPECT_EQ(rt->ReadbackBuffer(alias, 0U, output, kSize).GetCode(),
		OaStatusCode::InvalidArgument);

	rt->FreeBuffer(src);
}

// ─── Concurrent Pool Acquire/Release ──────────────────────────────────────────

TEST(VkStream, ConcurrentPoolAccess) {
	auto* rt = OaEngine::GetGlobal();
	ASSERT_NE(rt, nullptr);

	constexpr OaI32 kThreads = 4;
	constexpr OaI32 kOps = 50;
	std::atomic<OaI32> counter{0};

	auto worker = [&]() {
		for (OaI32 i = 0; i < kOps; ++i) {
			OaVkStream* s = rt->AcquireStream();
			ASSERT_NE(s, nullptr);
			counter.fetch_add(1, std::memory_order_relaxed);
			rt->ReleaseStream(s);
		}
	};

	OaVec<std::thread> threads;
	for (OaI32 i = 0; i < kThreads; ++i) {
		threads.EmplaceBack(worker);
	}
	for (auto& t : threads) t.join();

	EXPECT_EQ(counter.load(), kThreads * kOps);
}

// ─── CopyBuffer Recording ────────────────────────────────────────────────────

TEST(VkStream, RecordCopyBuffer) {
	auto* rt = OaEngine::GetGlobal();
	ASSERT_NE(rt, nullptr);

	constexpr OaU64 kSize = 512;
	auto srcRes = rt->Allocator.AllocHostVisible(kSize);
	ASSERT_TRUE(srcRes.IsOk());
	auto src = srcRes.GetValue();
	rt->RegisterBuffer(src);

	auto dstRes = rt->Allocator.AllocHostVisible(kSize);
	ASSERT_TRUE(dstRes.IsOk());
	auto dst = dstRes.GetValue();
	rt->RegisterBuffer(dst);

	OaU8* data = static_cast<OaU8*>(src.MappedPtr);
	for (OaU32 i = 0; i < kSize; ++i) data[i] = static_cast<OaU8>(42);
	ASSERT_TRUE(rt->Allocator.FlushHostBuffer(src, 0U, kSize));

	OaVkStream* stream = rt->AcquireStream();
	ASSERT_NE(stream, nullptr);
	ASSERT_TRUE(stream->Begin(rt->Device).IsOk());
	stream->RecordCopyBuffer(src, dst, kSize);
	stream->RecordTransferWriteBarrier(dst, 0U, kSize);
	ASSERT_TRUE(stream->SubmitAndWait(*rt).IsOk());
	rt->ReleaseStream(stream);
	ASSERT_TRUE(rt->Allocator.InvalidateHostBuffer(dst, 0U, kSize));

	OaU8* out = static_cast<OaU8*>(dst.MappedPtr);
	for (OaU32 i = 0; i < kSize; ++i) {
		EXPECT_EQ(out[i], 42);
	}

	rt->DeregisterBuffer(src);
	rt->Allocator.Free(src);
	rt->DeregisterBuffer(dst);
	rt->Allocator.Free(dst);
}

static OaResult<OaUniquePtr<OaEngine>> CreateHeadlessGraphicsTestEngine() {
	auto config = OaTestEngineConfig(OaPrecision::FP32);
	config.PresentationMode = OaPresentationMode::Headless;
	config.RegisterAsGlobal = false;
	config.PreloadEmbeddedPipelines = false;
	config.EnablePipelineCache = false;
	return OaEngine::Create(config);
}

TEST(VkStream, PresenterDestructionCancelsUnsubmittedGraphicsBatch) {
	auto engineResult = CreateHeadlessGraphicsTestEngine();
	ASSERT_TRUE(engineResult.IsOk()) << engineResult.GetStatus().GetMessage();
	auto engine = std::move(*engineResult);
	ASSERT_NE(engine->Device.Queues.GraphicsQueue, nullptr);

	constexpr OaU64 kSize = 4096;
	auto srcResult = engine->AllocBuffer(kSize);
	ASSERT_TRUE(srcResult.IsOk());
	auto src = std::move(*srcResult);
	auto dstResult = engine->AllocBuffer(kSize);
	ASSERT_TRUE(dstResult.IsOk());
	auto dst = std::move(*dstResult);
	OaMemset(src.MappedPtr, 0x39, kSize);
	OaMemset(dst.MappedPtr, 0xC7, kSize);

	{
		OaPresenter presenter(*engine);
		ASSERT_TRUE(presenter.BeginGraphicsBatch().IsOk());
		auto* stream = presenter.ActiveGraphicsBatchStream();
		ASSERT_NE(stream, nullptr);
		stream->RecordCopyBuffer(src, dst, kSize);
	}

	ASSERT_TRUE(engine->Allocator.InvalidateHostBuffer(dst, 0, kSize));
	const auto* values = static_cast<const OaU8*>(dst.MappedPtr);
	for (OaU64 index = 0; index < kSize; ++index) {
		ASSERT_EQ(values[index], 0xC7);
	}
	engine->FreeBuffer(src);
	engine->FreeBuffer(dst);
	ASSERT_TRUE(engine->Close().IsOk());
}

TEST(VkStream, PresenterDestructionRetiresSubmittedGraphicsBatch) {
	auto engineResult = CreateHeadlessGraphicsTestEngine();
	ASSERT_TRUE(engineResult.IsOk()) << engineResult.GetStatus().GetMessage();
	auto engine = std::move(*engineResult);
	ASSERT_NE(engine->Device.Queues.GraphicsQueue, nullptr);

	constexpr OaU64 kSize = 1024 * 1024;
	auto srcResult = engine->AllocBuffer(kSize);
	ASSERT_TRUE(srcResult.IsOk());
	auto src = std::move(*srcResult);
	auto dstResult = engine->AllocBuffer(kSize);
	ASSERT_TRUE(dstResult.IsOk());
	auto dst = std::move(*dstResult);
	auto* source = static_cast<OaU8*>(src.MappedPtr);
	for (OaU64 index = 0; index < kSize; ++index) {
		source[index] = static_cast<OaU8>((index * 23U + 5U) & 0xFFU);
	}
	ASSERT_TRUE(engine->Allocator.FlushHostBuffer(src, 0, kSize));

	OaEvent completion;
	{
		OaPresenter presenter(*engine);
		ASSERT_TRUE(presenter.BeginGraphicsBatch().IsOk());
		auto* stream = presenter.ActiveGraphicsBatchStream();
		ASSERT_NE(stream, nullptr);
		stream->RecordCopyBuffer(src, dst, kSize);
		stream->RecordTransferWriteBarrier(dst, 0U, kSize);
		ASSERT_TRUE(presenter.FlushGraphicsBatch().IsOk());
		completion = stream->Completion(engine->Device);
		ASSERT_TRUE(completion.IsValid());
	}

	// The stream remains at its original heap address in engine retirement, so
	// the non-owning event remains valid after the presenter facade is gone.
	ASSERT_TRUE(completion.Wait().IsOk());
	ASSERT_TRUE(engine->Allocator.InvalidateHostBuffer(dst, 0, kSize));
	const auto* values = static_cast<const OaU8*>(dst.MappedPtr);
	for (OaU64 index = 0; index < kSize; index += 4093U) {
		EXPECT_EQ(values[index], source[index]);
	}
	engine->FreeBuffer(src);
	engine->FreeBuffer(dst);
	ASSERT_TRUE(engine->Close().IsOk());
}

// ─── Persistent Mapped Upload Ring ───────────────────────────────────────────

TEST(VkStream, UploadRingBatchesRegionsAndRecyclesFrames) {
	auto* rt = OaEngine::GetGlobal();
	ASSERT_NE(rt, nullptr);

	constexpr OaU64 kDstSize = 4096;
	auto dstAResult = rt->AllocBuffer(kDstSize);
	ASSERT_TRUE(dstAResult.IsOk());
	auto dstA = std::move(*dstAResult);
	auto dstBResult = rt->AllocBuffer(kDstSize);
	ASSERT_TRUE(dstBResult.IsOk());
	auto dstB = std::move(*dstBResult);

	auto* dstAData = static_cast<OaU8*>(dstA.MappedPtr);
	auto* dstBData = static_cast<OaU8*>(dstB.MappedPtr);
	ASSERT_NE(dstAData, nullptr);
	ASSERT_NE(dstBData, nullptr);
	OaMemset(dstAData, 0xCD, kDstSize);
	OaMemset(dstBData, 0xCD, kDstSize);

	auto ringResult = OaUploadRing::Create(*rt, OaUploadRingConfig{
		.CapacityBytes = 3 * 4096,
		.FramesInFlight = 3,
		.Alignment = 64,
	});
	ASSERT_TRUE(ringResult.IsOk()) << ringResult.GetStatus().GetMessage();
	auto ring = std::move(*ringResult);
	EXPECT_EQ(ring.FrameCapacityBytes(), 4096u);

	OaU8 first[256];
	OaU8 second[512];
	OaU8 third[128];
	for (OaU32 i = 0; i < 256; ++i) first[i] = static_cast<OaU8>(i);
	for (OaU32 i = 0; i < 512; ++i) second[i] = static_cast<OaU8>(255u - (i & 0xFFu));
	for (OaU32 i = 0; i < 128; ++i) third[i] = static_cast<OaU8>(0x80u + i);

	// Seven batches lap the three frame arenas twice. BeginBatch must only wait
	// when its own arena is recycled; earlier frames remain independently in flight.
	for (OaU32 batch = 0; batch < 7; ++batch) {
		ASSERT_TRUE(ring.BeginBatch().IsOk());
		ASSERT_TRUE(ring.Upload(dstA, 0, first, sizeof(first)).IsOk());
		ASSERT_TRUE(ring.Upload(dstA, 1024, second, sizeof(second)).IsOk());
		ASSERT_TRUE(ring.Upload(dstB, 256, third, sizeof(third)).IsOk());
		EXPECT_EQ(ring.PendingCopyCount(), 3u);
		EXPECT_LE(ring.BytesUsed(), ring.FrameCapacityBytes());
		auto completion = ring.Submit();
		ASSERT_TRUE(completion.IsOk()) << completion.GetStatus().GetMessage();
		ASSERT_TRUE(completion->IsValid());
	}
	ASSERT_TRUE(ring.Wait().IsOk());
	ASSERT_TRUE(rt->Allocator.InvalidateHostBuffer(dstA, 0, kDstSize));
	ASSERT_TRUE(rt->Allocator.InvalidateHostBuffer(dstB, 0, kDstSize));

	for (OaU32 i = 0; i < 256; ++i) EXPECT_EQ(dstAData[i], first[i]);
	for (OaU32 i = 0; i < 512; ++i) EXPECT_EQ(dstAData[1024 + i], second[i]);
	for (OaU32 i = 0; i < 128; ++i) EXPECT_EQ(dstBData[256 + i], third[i]);
	EXPECT_EQ(dstAData[512], 0xCD);
	EXPECT_EQ(dstBData[0], 0xCD);

	ring.Destroy();
	rt->FreeBuffer(dstA);
	rt->FreeBuffer(dstB);
}

TEST(VkStream, UploadRingRejectsInvalidRangesAndClosesEmptyBatch) {
	auto* rt = OaEngine::GetGlobal();
	ASSERT_NE(rt, nullptr);

	auto dstResult = rt->AllocBuffer(1024);
	ASSERT_TRUE(dstResult.IsOk());
	auto dst = std::move(*dstResult);
	auto ringResult = OaUploadRing::Create(*rt, OaUploadRingConfig{
		.CapacityBytes = 2 * 1024,
		.FramesInFlight = 2,
		.Alignment = 64,
	});
	ASSERT_TRUE(ringResult.IsOk());
	auto ring = std::move(*ringResult);

	ASSERT_TRUE(ring.BeginBatch().IsOk());
	EXPECT_FALSE(ring.Reserve(2048).IsOk());
	auto staleSlice = ring.Reserve(8);
	ASSERT_TRUE(staleSlice.IsOk());
	auto emptyCompletion = ring.Submit();
	ASSERT_TRUE(emptyCompletion.IsOk());
	EXPECT_TRUE(emptyCompletion->IsValid());
	EXPECT_FALSE(ring.IsBatchOpen());
	ASSERT_TRUE(emptyCompletion->Wait().IsOk());

	OaU8 data[8] = {};
	ASSERT_TRUE(ring.BeginBatch().IsOk());
	EXPECT_FALSE(ring.EnqueueCopy(*staleSlice, dst, 0).IsOk());
	EXPECT_FALSE(ring.Upload(dst, dst.Size - 4, data, sizeof(data)).IsOk());
	auto completion = ring.Submit();
	ASSERT_TRUE(completion.IsOk());
	ASSERT_TRUE(completion->Wait().IsOk());

	ring.Destroy();
	rt->FreeBuffer(dst);
}

TEST(VkStream, UploadRingPublishesWritesAcrossAliasHandles) {
	auto* rt = OaEngine::GetGlobal();
	ASSERT_NE(rt, nullptr);

	constexpr OaU64 kSize = 256;
	auto backingResult = rt->Allocator.AllocAliased(
		kSize, OaMemoryPlacement::DeviceLocal);
	ASSERT_TRUE(backingResult.IsOk());
	auto backing = std::move(*backingResult);
	auto siblingResult = rt->Allocator.CreateAliasingBuffer(backing, kSize);
	ASSERT_TRUE(siblingResult.IsOk());
	auto sibling = std::move(*siblingResult);
	auto readbackResult = rt->AllocBuffer(kSize);
	ASSERT_TRUE(readbackResult.IsOk());
	auto readback = std::move(*readbackResult);

	OaU8 source[kSize];
	for (OaU32 index = 0; index < kSize; ++index) {
		source[index] = static_cast<OaU8>((index * 29U + 7U) & 0xFFU);
	}
	auto ringResult = OaUploadRing::Create(*rt, OaUploadRingConfig{
		.CapacityBytes = 2 * 4096,
		.FramesInFlight = 2,
		.Alignment = 64,
	});
	ASSERT_TRUE(ringResult.IsOk());
	auto ring = std::move(*ringResult);
	ASSERT_TRUE(ring.BeginBatch().IsOk());
	ASSERT_TRUE(ring.Upload(backing, 0U, source, kSize).IsOk());
	auto uploaded = ring.Submit();
	ASSERT_TRUE(uploaded.IsOk());
	ASSERT_TRUE(uploaded->IsValid());

	OaVkStream* consumer = rt->AcquireStream();
	ASSERT_NE(consumer, nullptr);
	ASSERT_TRUE(consumer->Begin(rt->Device).IsOk());
	consumer->RecordCopyBuffer(sibling, readback, kSize);
	consumer->RecordTransferWriteBarrier(readback, 0U, kSize);
	// Both submissions use the primary compute queue. The upload's global
	// alias barrier, not a semaphore memory dependency, publishes the write to
	// this sibling-handle read.
	ASSERT_TRUE(consumer->Submit(*rt).IsOk());
	ASSERT_TRUE(consumer->Synchronize(rt->Device).IsOk());
	EXPECT_TRUE(uploaded->IsComplete());
	rt->ReleaseStream(consumer);
	ASSERT_TRUE(rt->Allocator.InvalidateHostBuffer(readback, 0U, kSize));
	EXPECT_TRUE(OaMemEqual(source, readback.MappedPtr, kSize));

	ring.Destroy();
	rt->Allocator.FreeAlias(sibling);
	rt->Allocator.FreeAlias(backing);
	rt->FreeBuffer(readback);
}

TEST(VkStream, UploadRingDestructionCancelsUnsubmittedBatch) {
	auto* rt = OaEngine::GetGlobal();
	ASSERT_NE(rt, nullptr);

	constexpr OaU64 kSize = 4096;
	auto dstResult = rt->AllocBuffer(kSize);
	ASSERT_TRUE(dstResult.IsOk());
	auto dst = std::move(*dstResult);
	ASSERT_NE(dst.MappedPtr, nullptr);
	OaMemset(dst.MappedPtr, 0xA5, kSize);

	OaU8 source[kSize];
	OaMemset(source, 0x3C, kSize);
	{
		auto ringResult = OaUploadRing::Create(*rt, OaUploadRingConfig{
			.CapacityBytes = 2 * kSize,
			.FramesInFlight = 2,
			.Alignment = 64,
		});
		ASSERT_TRUE(ringResult.IsOk());
		auto ring = std::move(*ringResult);
		ASSERT_TRUE(ring.BeginBatch().IsOk());
		ASSERT_TRUE(ring.Upload(dst, 0, source, kSize).IsOk());
		ASSERT_TRUE(ring.IsBatchOpen());
	}

	ASSERT_TRUE(rt->Allocator.InvalidateHostBuffer(dst, 0, kSize));
	const auto* values = static_cast<const OaU8*>(dst.MappedPtr);
	for (OaU64 index = 0; index < kSize; ++index) {
		ASSERT_EQ(values[index], 0xA5);
	}
	rt->FreeBuffer(dst);
}

TEST(VkStream, UploadRingDestructionRetiresSubmittedResourcesToEngine) {
	auto* rt = OaEngine::GetGlobal();
	ASSERT_NE(rt, nullptr);

	constexpr OaU64 kSize = 1024 * 1024;
	auto dstResult = rt->AllocBuffer(kSize);
	ASSERT_TRUE(dstResult.IsOk());
	auto dst = std::move(*dstResult);
	OaVec<OaU8> source(kSize);
	for (OaU64 index = 0; index < kSize; ++index) {
		source[index] = static_cast<OaU8>((index * 17U + 11U) & 0xFFU);
	}

	OaCompletionToken completion;
	{
		auto ringResult = OaUploadRing::Create(*rt, OaUploadRingConfig{
			.CapacityBytes = 2 * kSize,
			.FramesInFlight = 2,
			.Alignment = 256,
		});
		ASSERT_TRUE(ringResult.IsOk());
		auto ring = std::move(*ringResult);
		ASSERT_TRUE(ring.BeginBatch().IsOk());
		ASSERT_TRUE(ring.Upload(dst, 0, source.Data(), kSize).IsOk());
		auto submitted = ring.Submit();
		ASSERT_TRUE(submitted.IsOk());
		completion = *submitted;
		ASSERT_TRUE(completion.IsValid());
	}

	// The facade is gone, but the engine-retired frame stays at the same address
	// and owns its timeline until explicit engine close.
	ASSERT_TRUE(completion.Wait().IsOk());
	ASSERT_TRUE(rt->Allocator.InvalidateHostBuffer(dst, 0, kSize));
	const auto* values = static_cast<const OaU8*>(dst.MappedPtr);
	for (OaU64 index = 0; index < kSize; index += 4093U) {
		EXPECT_EQ(values[index], source[index]);
	}
	rt->FreeBuffer(dst);
}

TEST(VkStream, PrefetchDestructionCancelsPendingCopy) {
	auto* rt = OaEngine::GetGlobal();
	ASSERT_NE(rt, nullptr);

	constexpr OaU64 kSize = 4096;
	OaU8 source[kSize];
	OaMemset(source, 0x6D, kSize);
	{
		auto pipelineResult = OaPrefetchPipeline::Create(*rt, kSize);
		ASSERT_TRUE(pipelineResult.IsOk());
		auto pipeline = std::move(*pipelineResult);
		ASSERT_TRUE(pipeline.BeginCopy(source, kSize).IsOk());
		ASSERT_TRUE(pipeline.IsBusy());
	}

	// The abandoned recording was reset rather than submitted by destruction;
	// its destination has been released and the engine remains usable.
	auto probeResult = rt->AllocBuffer(kSize);
	ASSERT_TRUE(probeResult.IsOk());
	auto probe = std::move(*probeResult);
	rt->FreeBuffer(probe);
}

TEST(VkStream, PrefetchCompatibilityPathTransfersToDeviceBuffer) {
	auto* rt = OaEngine::GetGlobal();
	ASSERT_NE(rt, nullptr);

	constexpr OaU64 kSize = 1024;
	OaU8 source[kSize];
	for (OaU32 i = 0; i < kSize; ++i) {
		source[i] = static_cast<OaU8>((i * 29u + 7u) & 0xFFu);
	}
	auto pipelineResult = OaPrefetchPipeline::Create(*rt, kSize);
	ASSERT_TRUE(pipelineResult.IsOk());
	auto pipeline = std::move(*pipelineResult);
	ASSERT_TRUE(pipeline.BeginCopy(source, kSize).IsOk());
	ASSERT_TRUE(pipeline.IsBusy());
	auto deviceResult = pipeline.Wait();
	ASSERT_TRUE(deviceResult.IsOk());
	EXPECT_FALSE(pipeline.IsBusy());

	OaU8 readback[kSize] = {};
	ASSERT_TRUE(rt->ReadbackBuffer(
		*deviceResult, 0U, readback, kSize).IsOk());
	EXPECT_TRUE(OaMemEqual(source, readback, kSize));

	pipeline.Destroy(*rt);
}

// ─── Thread-Safe Pipeline Access ──────────────────────────────────────────────

TEST(VkStream, ConcurrentPipelineRead) {
	auto* rt = OaEngine::GetGlobal();
	ASSERT_NE(rt, nullptr);

	constexpr OaI32 kThreads = 4;
	constexpr OaI32 kReads = 100;
	std::atomic<OaI32> successCount{0};

	auto reader = [&]() {
		for (OaI32 i = 0; i < kReads; ++i) {
			try {
				auto& pipe = rt->GetPipeline("Scale");
				if (pipe.Pipeline != nullptr) {
					successCount.fetch_add(1, std::memory_order_relaxed);
				}
			} catch (...) {}
		}
	};

	OaVec<std::thread> threads;
	for (OaI32 i = 0; i < kThreads; ++i) {
		threads.EmplaceBack(reader);
	}
	for (auto& t : threads) t.join();

	EXPECT_GT(successCount.load(), 0);
}

// ─── SIMD Smoke Tests ─────────────────────────────────────────────────────────

TEST(Simd, Memzero) {
	OaU8 buf[256];
	for (auto& b : buf) b = 0xFF;
	OaMemzero(buf, 256);
	for (auto& b : buf) {
		EXPECT_EQ(b, 0);
	}
}

TEST(Simd, MemEqual) {
	OaU8 a[128], b[128];
	for (OaI32 i = 0; i < 128; ++i) {
		a[i] = static_cast<OaU8>(i);
		b[i] = static_cast<OaU8>(i);
	}
	EXPECT_TRUE(OaMemEqual(a, b, 128));
	b[64] = 0xFF;
	EXPECT_FALSE(OaMemEqual(a, b, 128));
}

TEST(Simd, DotProduct) {
	constexpr OaI64 N = 256;
	OaF32 a[N], b[N];
	OaF32 expected = 0.0f;
	for (OaI64 i = 0; i < N; ++i) {
		a[i] = static_cast<OaF32>(i) * 0.01f;
		b[i] = static_cast<OaF32>(N - i) * 0.01f;
		expected += a[i] * b[i];
	}
	OaF32 result = OaSimd::DotF32(a, b, N);
	EXPECT_NEAR(result, expected, 0.1f);
}

TEST(Simd, Scale) {
	constexpr OaI64 N = 128;
	OaF32 data[N];
	for (OaI64 i = 0; i < N; ++i) data[i] = static_cast<OaF32>(i);
	OaSimd::ScaleF32(data, 2.0f, N);
	for (OaI64 i = 0; i < N; ++i) {
		EXPECT_FLOAT_EQ(data[i], static_cast<OaF32>(i) * 2.0f);
	}
}

TEST(Simd, Add) {
	constexpr OaI64 N = 128;
	OaF32 a[N], b[N];
	for (OaI64 i = 0; i < N; ++i) {
		a[i] = static_cast<OaF32>(i);
		b[i] = 1.0f;
	}
	OaSimd::AddF32(a, b, N);
	for (OaI64 i = 0; i < N; ++i) {
		EXPECT_FLOAT_EQ(a[i], static_cast<OaF32>(i) + 1.0f);
	}
}
