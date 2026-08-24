// ═══════════════════════════════════════════════════════════════════════════════
// Test: oavk::Stream — persistent async compute streams
// ═══════════════════════════════════════════════════════════════════════════════

#include "../../oaTest.h"
#include <oa/runtime/stream.h>
#include <oa/runtime/uploadRing.h>
#include <oa/runtime/cmd.h>
#include <oa/runtime/engine.h>
#include <oa/runtime/eventAccess.h>
#include <oa/runtime/presenter.h>
#include <oa/runtime/engine/allocatorAccess.h>
#include <oa/runtime/engine/bindlessAccess.h>
#include <oa/runtime/engine/pipelineAccess.h>
#include <oa/runtime/engine/submissionAccess.h>
#include <oa/runtime/imageDispatch.h>
#include <oa/runtime/graphicsStream.h>
#include <oa/runtime/engine/queueSubmitRoute.h>
#include <oa/core/simd.h>
#include <oa/core/memory.h>
#include <thread>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <mutex>
#include <type_traits>

static_assert(std::is_trivially_copyable_v<oa::Event>);
static_assert(std::is_trivially_destructible_v<oa::Event>);
static_assert(sizeof(oa::Event) <= 32U);

// ─── stream Lifecycle ─────────────────────────────────────────────────────────

TEST(VkStream, CreateDestroy) {
	auto* rt = testEnginePtr();
	ASSERT_NE(rt, nullptr);

	auto res = oavk::Stream::createCompute(oa::EngineDeviceAccess::get(*rt));
	ASSERT_TRUE(res.isOk());
	auto stream = std::move(*res);

	EXPECT_NE(stream.commandPool, nullptr);
	EXPECT_NE(stream.commandBuffer, nullptr);
	EXPECT_NE(stream.timelineSem.semaphore, nullptr);
	EXPECT_EQ(stream.timelineValue, 0u);
	EXPECT_FALSE(stream.recording);
	EXPECT_FALSE(stream.submitted);

	stream.destroy(oa::EngineDeviceAccess::get(*rt));
	EXPECT_EQ(stream.commandPool, nullptr);
}

// ─── Begin/submit Cycle ───────────────────────────────────────────────────────

TEST(VkStream, BeginSubmitCycle) {
	auto* rt = testEnginePtr();
	ASSERT_NE(rt, nullptr);

	auto res = oavk::Stream::createCompute(oa::EngineDeviceAccess::get(*rt));
	ASSERT_TRUE(res.isOk());
	auto stream = std::move(*res);

	ASSERT_TRUE(stream.begin(oa::EngineDeviceAccess::get(*rt)).isOk());
	EXPECT_TRUE(stream.recording);

	stream.recordBufferBarrier();

	auto status = stream.submitAndWait(*rt);
	EXPECT_TRUE(status.isOk());
	EXPECT_FALSE(stream.recording);
	EXPECT_TRUE(stream.isComplete(oa::EngineDeviceAccess::get(*rt)));

	stream.destroy(oa::EngineDeviceAccess::get(*rt));
}

// ─── stream reuse (multiple submit cycles) ───────────────────────────────────

TEST(VkStream, Reuse) {
	auto* rt = testEnginePtr();
	ASSERT_NE(rt, nullptr);

	auto res = oavk::Stream::createCompute(oa::EngineDeviceAccess::get(*rt));
	ASSERT_TRUE(res.isOk());
	auto stream = std::move(*res);

	for (oa::I32 i = 0; i < 5; ++i) {
		ASSERT_TRUE(stream.begin(oa::EngineDeviceAccess::get(*rt)).isOk());
		stream.recordBufferBarrier();
		ASSERT_TRUE(stream.submitAndWait(*rt).isOk());
		EXPECT_TRUE(stream.isComplete(oa::EngineDeviceAccess::get(*rt)));
	}

	stream.destroy(oa::EngineDeviceAccess::get(*rt));
}

// ─── Async submit + poll ──────────────────────────────────────────────────────

TEST(VkStream, AsyncSubmitPoll) {
	auto* rt = testEnginePtr();
	ASSERT_NE(rt, nullptr);

	auto res = oavk::Stream::createCompute(oa::EngineDeviceAccess::get(*rt));
	ASSERT_TRUE(res.isOk());
	auto stream = std::move(*res);

	ASSERT_TRUE(stream.begin(oa::EngineDeviceAccess::get(*rt)).isOk());
	stream.recordBufferBarrier();
	ASSERT_TRUE(stream.submit(*rt).isOk());
	EXPECT_TRUE(stream.submitted);

	ASSERT_TRUE(stream.synchronize(oa::EngineDeviceAccess::get(*rt)).isOk());
	EXPECT_TRUE(stream.isComplete(oa::EngineDeviceAccess::get(*rt)));

	stream.destroy(oa::EngineDeviceAccess::get(*rt));
}

// ─── fence isSignaled ─────────────────────────────────────────────────────────

TEST(VkFence, isSignaled) {
	auto* rt = testEnginePtr();
	ASSERT_NE(rt, nullptr);

	auto fRes = oavk::Fence::create(oa::EngineDeviceAccess::get(*rt), true);
	ASSERT_TRUE(fRes.isOk());
	auto fence = std::move(*fRes);
	EXPECT_TRUE(fence.isSignaled(oa::EngineDeviceAccess::get(*rt)));

	fence.reset(oa::EngineDeviceAccess::get(*rt));
	EXPECT_FALSE(fence.isSignaled(oa::EngineDeviceAccess::get(*rt)));

	fence.destroy(oa::EngineDeviceAccess::get(*rt));
}

TEST(VkTimelineSemaphore, EventSnapshotsWrapperHandle) {
	auto* rt = testEnginePtr();
	ASSERT_NE(rt, nullptr);

	auto semaphoreResult = oavk::TimelineSemaphore::create(oa::EngineDeviceAccess::get(*rt), 1U);
	ASSERT_TRUE(semaphoreResult.isOk());
	auto semaphoreOwner = std::move(*semaphoreResult);
	oa::Event completion = oa::EventAccess::create(
		oa::EngineDeviceAccess::get(*rt), semaphoreOwner, 1U);
	oa::Event completionCopy = completion;

	// Moving an owning facade may invalidate its wrapper object without
	// invalidating the vulkan semaphore that backs an already-issued event.
	oavk::TimelineSemaphore movedOwner = semaphoreOwner;
	semaphoreOwner.semaphore = nullptr;
	ASSERT_TRUE(completion.isValid());
	ASSERT_TRUE(completionCopy.isSameCompletion(completion));
	EXPECT_TRUE(completionCopy.isComplete());
	EXPECT_TRUE(completionCopy.wait().isOk());

	movedOwner.destroy(oa::EngineDeviceAccess::get(*rt));
}

// ─── Engine stream pool ───────────────────────────────────────────────────────

TEST(VkStream, PoolAcquireRelease) {
	auto* rt = testEnginePtr();
	ASSERT_NE(rt, nullptr);

	oavk::Stream* s1 = oa::EngineSubmissionAccess::acquireStream(*rt);
	ASSERT_NE(s1, nullptr);
	EXPECT_NE(s1->commandPool, nullptr);

	oavk::Stream* s2 = oa::EngineSubmissionAccess::acquireStream(*rt);
	ASSERT_NE(s2, nullptr);
	EXPECT_NE(s1, s2);

	oa::EngineSubmissionAccess::releaseStream(*rt, s1);
	oavk::Stream* s3 = oa::EngineSubmissionAccess::acquireStream(*rt);
	EXPECT_EQ(s3, s1);

	oa::EngineSubmissionAccess::releaseStream(*rt, s2);
	oa::EngineSubmissionAccess::releaseStream(*rt, s3);
}

// ─── runOnce (single-shot dispatch) ──────────────────────────────────────────

TEST(VkStream, RunOnce) {
	auto* rt = testEnginePtr();
	ASSERT_NE(rt, nullptr);

	auto srcRes = oa::EngineAllocatorAccess::get(*rt).allocHostVisible(256 * sizeof(oa::F32));
	ASSERT_TRUE(srcRes.isOk());
	auto src = srcRes.getValue();
	ASSERT_NE(oa::EngineBindlessAccess::registerBuffer(*rt, src), OA_BINDLESS_INVALID);

	auto dstRes = oa::EngineAllocatorAccess::get(*rt).allocHostVisible(256 * sizeof(oa::F32));
	ASSERT_TRUE(dstRes.isOk());
	auto dst = dstRes.getValue();
	ASSERT_NE(oa::EngineBindlessAccess::registerBuffer(*rt, dst), OA_BINDLESS_INVALID);

	oa::F32* srcData = static_cast<oa::F32*>(src.mappedPtr);
	for (oa::I32 i = 0; i < 256; ++i) srcData[i] = static_cast<oa::F32>(i);

	struct { oa::U32 N; oa::F32 scale; } push{256, 2.0f};
	oavk::Buffer bufs[] = {src, dst};

	auto status = oavk::Stream::runOnce(
		*rt, "Scale", bufs, &push, sizeof(push),
		oa::ScalarType::Float32, (256 + 255) / 256);

	if (status.isOk()) {
		oa::F32* out = static_cast<oa::F32*>(dst.mappedPtr);
		for (oa::I32 i = 0; i < 256; ++i) {
			EXPECT_NEAR(out[i], static_cast<oa::F32>(i) * 2.0f, 1e-3f);
		}
	}

	oa::EngineBindlessAccess::deregisterBuffer(*rt, src);
	oa::EngineAllocatorAccess::get(*rt).free(src);
	oa::EngineBindlessAccess::deregisterBuffer(*rt, dst);
	oa::EngineAllocatorAccess::get(*rt).free(dst);
}

// ─── Explicit storage dtype ───────────────────────────────────────────────────

TEST(VkStream, ExplicitStorageDtypeUsesCachedExactPipeline) {
	auto* rt = testEnginePtr();
	ASSERT_NE(rt, nullptr);
	auto& warmedPipeline = oa::EnginePipelineAccess::get(*rt).getPipeline("Scale", 0U);
	ASSERT_NE(warmedPipeline.pipeline, nullptr);
	EXPECT_EQ(warmedPipeline.nativeDtype, 0U);
	const auto warmedHandle = warmedPipeline.pipeline;

	auto srcRes = oa::EngineAllocatorAccess::get(*rt).allocHostVisible(64 * sizeof(oa::F32));
	ASSERT_TRUE(srcRes.isOk());
	auto src = srcRes.getValue();
	ASSERT_NE(oa::EngineBindlessAccess::registerBuffer(*rt, src), OA_BINDLESS_INVALID);

	auto dstRes = oa::EngineAllocatorAccess::get(*rt).allocHostVisible(64 * sizeof(oa::F32));
	ASSERT_TRUE(dstRes.isOk());
	auto dst = dstRes.getValue();
	ASSERT_NE(oa::EngineBindlessAccess::registerBuffer(*rt, dst), OA_BINDLESS_INVALID);

	oa::F32* data = static_cast<oa::F32*>(src.mappedPtr);
	for (oa::I32 i = 0; i < 64; ++i) data[i] = static_cast<oa::F32>(i + 1);

	struct { oa::U32 N; oa::F32 scale; } push{64, 0.5f};
	oavk::Buffer bufs[] = {src, dst};

	auto status = oavk::Dispatch::run(
		*rt, "Scale", bufs, &push, sizeof(push), oa::ScalarType::Float32, 1);
	ASSERT_TRUE(status.isOk()) << status.getMessage();
	oa::F32* out = static_cast<oa::F32*>(dst.mappedPtr);
	EXPECT_NEAR(out[0], 0.5f, 1e-3f);
	EXPECT_NEAR(out[63], 32.0f, 1e-3f);

	// Dispatch performs an exact cached lookup after warmup. It must neither
	// switch to the engine's default precision nor replace the native pipeline.
	auto& afterDispatch = oa::EnginePipelineAccess::get(*rt).getPipeline("Scale", 0U);
	EXPECT_EQ(afterDispatch.pipeline, warmedHandle);
	EXPECT_EQ(afterDispatch.nativeDtype, 0U);

	const auto invalid = oavk::Dispatch::run(
		*rt, "Scale", bufs, &push, sizeof(push), oa::ScalarType::UInt32, 1);
	EXPECT_EQ(invalid.getCode(), oa::StatusCode::InvalidArgument);

	oa::EngineBindlessAccess::deregisterBuffer(*rt, src);
	oa::EngineAllocatorAccess::get(*rt).free(src);
	oa::EngineBindlessAccess::deregisterBuffer(*rt, dst);
	oa::EngineAllocatorAccess::get(*rt).free(dst);
}

TEST(VkDispatch, DescriptorAdmissionRejectsLiveRangeLimitAndRecoversStream) {
	auto* rt = testEnginePtr();
	ASSERT_NE(rt, nullptr);
	auto srcResult = oa::EngineResourceAccess::allocBuffer(*rt, 16U);
	auto dstResult = oa::EngineResourceAccess::allocBuffer(*rt, 16U);
	ASSERT_TRUE(srcResult.isOk() and dstResult.isOk());
	auto src = std::move(*srcResult);
	auto dst = std::move(*dstResult);
	auto* source = static_cast<oa::F32*>(src.mappedPtr);
	for (oa::U32 index = 0; index < 4U; ++index) {
		source[index] = static_cast<oa::F32>(index + 1U);
	}

	struct { oa::U32 Count; oa::F32 scale; } push{4U, 2.0F};
	oavk::Buffer buffers[] = {src, dst};
	auto& maximum =
		oa::EngineDeviceAccess::get(*rt).info.hardware.maxStorageBufferRangeBytes;
	const oa::U64 savedMaximum = maximum;
	ASSERT_GT(savedMaximum, 8U);
	maximum = 8U;
	const auto rejected = oavk::Dispatch::run(
		*rt, "Scale", buffers, &push, sizeof(push),
		oa::ScalarType::Float32, 1U);
	EXPECT_EQ(rejected.getCode(), oa::StatusCode::OutOfRange);

	auto rejectedBatch = oavk::Dispatch::beginBatch(*rt);
	ASSERT_TRUE(rejectedBatch.isOk());
	auto batch = std::move(*rejectedBatch);
	const auto rejectedRecord = oavk::Dispatch::record(
		batch, *rt, "Scale", buffers, &push, sizeof(push),
		oa::ScalarType::Float32, 1U);
	EXPECT_EQ(rejectedRecord.getCode(), oa::StatusCode::OutOfRange);
	EXPECT_EQ(batch.stream, nullptr);
	EXPECT_EQ(oavk::Dispatch::flush(batch, *rt).getCode(),
		oa::StatusCode::FailedPrecondition);

	maximum = savedMaximum;
	const auto accepted = oavk::Dispatch::run(
		*rt, "Scale", buffers, &push, sizeof(push),
		oa::ScalarType::Float32, 1U);
	ASSERT_TRUE(accepted.isOk()) << accepted.getMessage();
	const auto* destination = static_cast<const oa::F32*>(dst.mappedPtr);
	for (oa::U32 index = 0; index < 4U; ++index) {
		EXPECT_FLOAT_EQ(destination[index],
			static_cast<oa::F32>((index + 1U) * 2U));
	}

	oa::EngineResourceAccess::freeBuffer(*rt, src);
	oa::EngineResourceAccess::freeBuffer(*rt, dst);
}

TEST(VkDispatch, DirectGroupAdmissionUsesLiveThreeDimensionalLimit) {
	auto* rt = testEnginePtr();
	ASSERT_NE(rt, nullptr);
	auto sourceResult = oa::EngineResourceAccess::allocBuffer(*rt, 16U);
	auto destinationResult = oa::EngineResourceAccess::allocBuffer(*rt, 16U);
	ASSERT_TRUE(sourceResult.isOk() and destinationResult.isOk());
	auto source = std::move(*sourceResult);
	auto destination = std::move(*destinationResult);
	auto* sourceData = static_cast<oa::F32*>(source.mappedPtr);
	for (oa::U32 index = 0; index < 4U; ++index) {
		sourceData[index] = static_cast<oa::F32>(index + 1U);
	}

	auto& hardware = oa::EngineDeviceAccess::get(*rt).info.hardware;
	VkPhysicalDeviceProperties properties{};
	oa::EngineDeviceAccess::get(*rt).instanceDispatch.vkGetPhysicalDeviceProperties(
		static_cast<VkPhysicalDevice>(oa::EngineDeviceAccess::get(*rt).physicalDevice),
		&properties);
	EXPECT_EQ(hardware.maxComputeWorkGroupCountX,
		properties.limits.maxComputeWorkGroupCount[0]);
	EXPECT_EQ(hardware.maxComputeWorkGroupCountY,
		properties.limits.maxComputeWorkGroupCount[1]);
	EXPECT_EQ(hardware.maxComputeWorkGroupCountZ,
		properties.limits.maxComputeWorkGroupCount[2]);

	const oa::U32 savedMaximumY = hardware.maxComputeWorkGroupCountY;
	ASSERT_GT(savedMaximumY, 1U);
	struct RestoreLimit {
		oa::U32& Value;
		oa::U32 Saved;
		~RestoreLimit() { Value = Saved; }
	} restoreMaximumY{hardware.maxComputeWorkGroupCountY, savedMaximumY};
	hardware.maxComputeWorkGroupCountY = 1U;

	struct { oa::U32 Count; oa::F32 scale; } push{4U, 2.0F};
	oavk::Buffer buffers[] = {source, destination};
	const auto rejected = oavk::Dispatch::run(
		*rt, "Scale", buffers, &push, sizeof(push),
		oa::ScalarType::Float32, 1U, 2U, 1U);
	EXPECT_EQ(rejected.getCode(), oa::StatusCode::OutOfRange);

	auto batchResult = oavk::Dispatch::beginBatch(*rt);
	ASSERT_TRUE(batchResult.isOk());
	auto batch = std::move(*batchResult);
	const auto rejectedRecord = oavk::Dispatch::record(
		batch, *rt, "Scale", buffers, &push, sizeof(push),
		oa::ScalarType::Float32, 1U, 2U, 1U);
	EXPECT_EQ(rejectedRecord.getCode(), oa::StatusCode::OutOfRange);
	EXPECT_EQ(batch.stream, nullptr);
	EXPECT_EQ(oavk::Dispatch::flush(batch, *rt).getCode(),
		oa::StatusCode::FailedPrecondition);

	oavk::Command rawCommand;
	const auto rawRejected =
		rawCommand.dispatch(oa::EngineDeviceAccess::get(*rt), 1U, 2U, 1U);
	EXPECT_EQ(rawRejected.getCode(), oa::StatusCode::OutOfRange);

	const auto imageRejected = oavk::ImageDispatch::run(
		*rt, "UnusedForRejectedDispatch",
		oa::Span<const oavk::ImageDispatchBinding>{},
		nullptr, 0U, oa::ScalarType::Float32,
		1U, 2U, 1U);
	EXPECT_EQ(imageRejected.getCode(), oa::StatusCode::OutOfRange);

	auto noOpBatchResult = oavk::Dispatch::beginBatch(*rt);
	ASSERT_TRUE(noOpBatchResult.isOk());
	auto noOpBatch = std::move(*noOpBatchResult);
	const auto noOpRecord = oavk::Dispatch::record(
		noOpBatch, *rt, "Scale", buffers, &push, sizeof(push),
		oa::ScalarType::Float32, 0U, 0U, 0U);
	ASSERT_TRUE(noOpRecord.isOk()) << noOpRecord.getMessage();
	const auto noOpFlush = oavk::Dispatch::flush(noOpBatch, *rt);
	ASSERT_TRUE(noOpFlush.isOk()) << noOpFlush.getMessage();

	hardware.maxComputeWorkGroupCountY = savedMaximumY;
	const auto imagePastAdmission = oavk::ImageDispatch::run(
		*rt, "UnusedAfterAcceptedDispatch",
		oa::Span<const oavk::ImageDispatchBinding>{},
		nullptr, 0U, oa::ScalarType::Float32,
		1U, 1U, 1U);
	EXPECT_EQ(imagePastAdmission.getCode(), oa::StatusCode::InvalidArgument);

	const auto accepted = oavk::Dispatch::run(
		*rt, "Scale", buffers, &push, sizeof(push),
		oa::ScalarType::Float32, 1U, 1U, 1U);
	ASSERT_TRUE(accepted.isOk()) << accepted.getMessage();
	const auto* destinationData =
		static_cast<const oa::F32*>(destination.mappedPtr);
	for (oa::U32 index = 0; index < 4U; ++index) {
		EXPECT_FLOAT_EQ(destinationData[index],
			static_cast<oa::F32>((index + 1U) * 2U));
	}

	oa::EngineResourceAccess::freeBuffer(*rt, source);
	oa::EngineResourceAccess::freeBuffer(*rt, destination);
}

// ─── Batch Dispatch backward Compat ───────────────────────────────────────────

TEST(VkStream, BatchDispatchCompat) {
	auto* rt = testEnginePtr();
	ASSERT_NE(rt, nullptr);

	auto aRes = oa::EngineAllocatorAccess::get(*rt).allocHostVisible(64 * sizeof(oa::F32));
	ASSERT_TRUE(aRes.isOk());
	auto a = aRes.getValue();
	ASSERT_NE(oa::EngineBindlessAccess::registerBuffer(*rt, a), OA_BINDLESS_INVALID);

	auto bRes = oa::EngineAllocatorAccess::get(*rt).allocHostVisible(64 * sizeof(oa::F32));
	ASSERT_TRUE(bRes.isOk());
	auto b = bRes.getValue();
	ASSERT_NE(oa::EngineBindlessAccess::registerBuffer(*rt, b), OA_BINDLESS_INVALID);

	oa::F32* data = static_cast<oa::F32*>(a.mappedPtr);
	for (oa::I32 i = 0; i < 64; ++i) data[i] = 1.0f;

	struct { oa::U32 N; oa::F32 scale; } push{64, 3.0f};
	oavk::Buffer bufs[] = {a, b};

	auto batchRes = oavk::Dispatch::beginBatch(*rt);
	ASSERT_TRUE(batchRes.isOk());
	auto batch = std::move(*batchRes);

	auto recSt = oavk::Dispatch::record(
		batch, *rt, "Scale", bufs, &push, sizeof(push),
		oa::ScalarType::Float32, 1);
	if (!recSt.isOk()) {
		oa::EngineBindlessAccess::deregisterBuffer(*rt, a);
		oa::EngineAllocatorAccess::get(*rt).free(a);
		oa::EngineBindlessAccess::deregisterBuffer(*rt, b);
		oa::EngineAllocatorAccess::get(*rt).free(b);
		GTEST_SKIP() << "scale shader not loaded";
	}

	ASSERT_TRUE(oavk::Dispatch::flush(batch, *rt).isOk());

	oa::F32* out = static_cast<oa::F32*>(b.mappedPtr);
	EXPECT_NEAR(out[0], 3.0f, 1e-3f);

	oa::EngineBindlessAccess::deregisterBuffer(*rt, a);
	oa::EngineAllocatorAccess::get(*rt).free(a);
	oa::EngineBindlessAccess::deregisterBuffer(*rt, b);
	oa::EngineAllocatorAccess::get(*rt).free(b);
}

// ─── indirect Dispatch ────────────────────────────────────────────────────────

TEST(VkDispatch, IndirectDispatch) {
	auto* rt = testEnginePtr();
	ASSERT_NE(rt, nullptr);

	constexpr oa::U32 N = 64;
	auto srcRes = oa::EngineResourceAccess::allocBuffer(*rt, N * sizeof(oa::F32));
	ASSERT_TRUE(srcRes.isOk());
	auto src = std::move(*srcRes);

	auto dstRes = oa::EngineResourceAccess::allocBuffer(*rt, N * sizeof(oa::F32));
	ASSERT_TRUE(dstRes.isOk());
	auto dst = std::move(*dstRes);

	oa::F32* srcData = static_cast<oa::F32*>(src.mappedPtr);
	for (oa::U32 i = 0; i < N; ++i) srcData[i] = static_cast<oa::F32>(i + 1);

	// indirect buffer: VkDispatchIndirectCommand {groupsX, groupsY, groupsZ}
	auto indRes = oa::EngineResourceAccess::allocBuffer(*rt, 3 * sizeof(oa::U32));
	ASSERT_TRUE(indRes.isOk());
	auto indBuf = std::move(*indRes);
	ASSERT_TRUE(indBuf.supportsIndirectDispatch());
	oa::U32* indData = static_cast<oa::U32*>(indBuf.mappedPtr);
	indData[0] = (N + 255) / 256;
	indData[1] = 1;
	indData[2] = 1;

	struct { oa::U32 N; oa::F32 scale; } push{N, 10.0f};
	oavk::Buffer bufs[] = {src, dst};

	oavk::Buffer unsupported = indBuf;
	unsupported.flags &= ~OA_VK_BUFFER_FLAG_INDIRECT_DISPATCH;
	const auto unsupportedStatus = oavk::Dispatch::runIndirect(
		*rt, "Scale", bufs, &push, sizeof(push),
		oa::ScalarType::Float32, unsupported);
	EXPECT_EQ(unsupportedStatus.getCode(), oa::StatusCode::InvalidArgument);

	oavk::Buffer foreignAllocator = indBuf;
	foreignAllocator.allocatorIdentity = &foreignAllocator;
	const auto foreignAllocatorStatus = oavk::Dispatch::runIndirect(
		*rt, "Scale", bufs, &push, sizeof(push),
		oa::ScalarType::Float32, foreignAllocator);
	EXPECT_EQ(foreignAllocatorStatus.getCode(), oa::StatusCode::InvalidArgument);

	const auto overflowStatus = oavk::Dispatch::runIndirect(
		*rt, "Scale", bufs, &push, sizeof(push),
		oa::ScalarType::Float32, indBuf, ~oa::U64{0} - 3U);
	EXPECT_EQ(overflowStatus.getCode(), oa::StatusCode::OutOfRange);

	const auto status = oavk::Dispatch::runIndirect(
		*rt, "Scale", bufs, &push, sizeof(push),
		oa::ScalarType::Float32, indBuf);
	ASSERT_TRUE(status.isOk()) << status.getMessage();
	oa::F32* out = static_cast<oa::F32*>(dst.mappedPtr);
	for (oa::U32 i = 0; i < N; ++i) {
		EXPECT_NEAR(out[i], static_cast<oa::F32>(i + 1) * 10.0f, 1e-3f);
	}

	oa::EngineResourceAccess::freeBuffer(*rt, src);
	oa::EngineResourceAccess::freeBuffer(*rt, dst);
	oa::EngineResourceAccess::freeBuffer(*rt, indBuf);
}

// ─── Async Transfer ───────────────────────────────────────────────────────────

TEST(VkStream, AsyncTransfer) {
	auto* rt = testEnginePtr();
	ASSERT_NE(rt, nullptr);

	constexpr oa::U64 kSize = 1024;
	auto srcRes = oa::EngineAllocatorAccess::get(*rt).allocHostVisible(kSize);
	ASSERT_TRUE(srcRes.isOk());
	auto src = srcRes.getValue();
	ASSERT_NE(oa::EngineBindlessAccess::registerBuffer(*rt, src), OA_BINDLESS_INVALID);

	auto dstRes = oa::EngineAllocatorAccess::get(*rt).allocHostVisible(kSize);
	ASSERT_TRUE(dstRes.isOk());
	auto dst = dstRes.getValue();
	ASSERT_NE(oa::EngineBindlessAccess::registerBuffer(*rt, dst), OA_BINDLESS_INVALID);

	oa::U8* srcData = static_cast<oa::U8*>(src.mappedPtr);
	for (oa::U32 i = 0; i < kSize; ++i) srcData[i] = static_cast<oa::U8>(i & 0xFF);
	oa::memset(dst.mappedPtr, 0xA5, kSize);
	ASSERT_TRUE(oa::EngineAllocatorAccess::get(*rt).flushHostBuffer(src, 0U, kSize));

	auto firstCopy = oa::EngineResourceAccess::copyBufferAsync(*rt, src, dst, kSize);
	ASSERT_TRUE(firstCopy.isOk());
	ASSERT_TRUE(firstCopy->isValid());
	ASSERT_TRUE(firstCopy->wait().isOk());
	EXPECT_TRUE(firstCopy->isComplete());
	ASSERT_TRUE(oa::EngineAllocatorAccess::get(*rt).invalidateHostBuffer(dst, 0U, kSize));

	oa::U8* dstData = static_cast<oa::U8*>(dst.mappedPtr);
	for (oa::U32 i = 0; i < kSize; ++i) {
		EXPECT_EQ(dstData[i], static_cast<oa::U8>(i & 0xFF)) << "mismatch at " << i;
	}

	for (oa::U32 i = 0; i < kSize; ++i) {
		srcData[i] = static_cast<oa::U8>((i * 17U + 3U) & 0xFFU);
	}
	oa::memset(dst.mappedPtr, 0x5A, kSize);
	ASSERT_TRUE(oa::EngineAllocatorAccess::get(*rt).flushHostBuffer(src, 0U, kSize));
	auto secondCopy = oa::EngineResourceAccess::copyBufferAsync(*rt, src, dst, kSize);
	ASSERT_TRUE(secondCopy.isOk());
	ASSERT_TRUE(secondCopy->isValid());
	EXPECT_EQ(oa::EventAccess::semaphoreHandle(*secondCopy),
		oa::EventAccess::semaphoreHandle(*firstCopy));
	EXPECT_GT(secondCopy->value(), firstCopy->value());
	ASSERT_TRUE(secondCopy->wait().isOk());
	ASSERT_TRUE(oa::EngineAllocatorAccess::get(*rt).invalidateHostBuffer(dst, 0U, kSize));
	for (oa::U32 i = 0; i < kSize; ++i) {
		EXPECT_EQ(dstData[i], static_cast<oa::U8>((i * 17U + 3U) & 0xFFU))
			<< "reuse mismatch at " << i;
	}

	oa::EngineBindlessAccess::deregisterBuffer(*rt, src);
	oa::EngineAllocatorAccess::get(*rt).free(src);
	oa::EngineBindlessAccess::deregisterBuffer(*rt, dst);
	oa::EngineAllocatorAccess::get(*rt).free(dst);
}

TEST(VkStream, AsyncTransferRejectsInvalidBufferContractsBeforeSubmission) {
	auto* rt = testEnginePtr();
	ASSERT_NE(rt, nullptr);

	constexpr oa::U64 kSize = 256;
	auto srcResult = oa::EngineResourceAccess::allocBuffer(*rt, kSize);
	ASSERT_TRUE(srcResult.isOk());
	auto src = std::move(*srcResult);
	auto dstResult = oa::EngineResourceAccess::allocBuffer(*rt, kSize);
	ASSERT_TRUE(dstResult.isOk());
	auto dst = std::move(*dstResult);

	oavk::Buffer nullBuffer;
	auto nullSource = oa::EngineResourceAccess::copyBufferAsync(*rt, nullBuffer, dst, kSize);
	ASSERT_FALSE(nullSource.isOk());
	EXPECT_EQ(nullSource.getStatus().getCode(), oa::StatusCode::InvalidArgument);
	auto nullDestination = oa::EngineResourceAccess::copyBufferAsync(*rt, src, nullBuffer, kSize);
	ASSERT_FALSE(nullDestination.isOk());
	EXPECT_EQ(nullDestination.getStatus().getCode(), oa::StatusCode::InvalidArgument);
	auto zeroSize = oa::EngineResourceAccess::copyBufferAsync(*rt, src, dst, 0U);
	ASSERT_FALSE(zeroSize.isOk());
	EXPECT_EQ(zeroSize.getStatus().getCode(), oa::StatusCode::InvalidArgument);
	auto oversized = oa::EngineResourceAccess::copyBufferAsync(*rt, src, dst, kSize + 1U);
	ASSERT_FALSE(oversized.isOk());
	EXPECT_EQ(oversized.getStatus().getCode(), oa::StatusCode::InvalidArgument);
	auto overlapping = oa::EngineResourceAccess::copyBufferAsync(*rt, src, src, kSize);
	ASSERT_FALSE(overlapping.isOk());
	EXPECT_EQ(overlapping.getStatus().getCode(), oa::StatusCode::InvalidArgument);
	oavk::Buffer foreignSource = src;
	foreignSource.allocatorIdentity = &foreignSource;
	auto foreign = oa::EngineResourceAccess::copyBufferAsync(*rt, foreignSource, dst, kSize);
	ASSERT_FALSE(foreign.isOk());
	EXPECT_EQ(foreign.getStatus().getCode(), oa::StatusCode::InvalidArgument);
	oavk::Buffer externalSource = src;
	externalSource.allocation = nullptr;
	auto external = oa::EngineResourceAccess::copyBufferAsync(*rt, externalSource, dst, kSize);
	ASSERT_FALSE(external.isOk());
	EXPECT_EQ(external.getStatus().getCode(), oa::StatusCode::InvalidArgument);
	oavk::Buffer otherEngineSource = src;
	otherEngineSource.allocatorIdentity = &otherEngineSource;
	auto otherEngine = oa::EngineResourceAccess::copyBufferAsync(*rt, otherEngineSource, dst, kSize);
	ASSERT_FALSE(otherEngine.isOk());
	EXPECT_EQ(otherEngine.getStatus().getCode(), oa::StatusCode::InvalidArgument);
	oavk::Buffer aliasDestination = dst;
	aliasDestination.aliasIdentity = dst.allocation;
	auto alias = oa::EngineResourceAccess::copyBufferAsync(*rt, src, aliasDestination, kSize);
	ASSERT_FALSE(alias.isOk());
	EXPECT_EQ(alias.getStatus().getCode(), oa::StatusCode::InvalidArgument);

	oa::EngineResourceAccess::freeBuffer(*rt, src);
	oa::EngineResourceAccess::freeBuffer(*rt, dst);
}

TEST(VkStream, AsyncTransferRejectsOverlappingInFlightSubmission) {
	auto* rt = testEnginePtr();
	ASSERT_NE(rt, nullptr);

	constexpr oa::U64 kSize = 4096;
	auto srcResult = oa::EngineResourceAccess::allocBuffer(*rt, kSize);
	ASSERT_TRUE(srcResult.isOk());
	auto src = std::move(*srcResult);
	auto firstDstResult = oa::EngineResourceAccess::allocBuffer(*rt, kSize);
	ASSERT_TRUE(firstDstResult.isOk());
	auto firstDst = std::move(*firstDstResult);
	auto secondDstResult = oa::EngineResourceAccess::allocBuffer(*rt, kSize);
	ASSERT_TRUE(secondDstResult.isOk());
	auto secondDst = std::move(*secondDstResult);
	oa::memset(src.mappedPtr, 0x6D, kSize);
	ASSERT_TRUE(oa::EngineAllocatorAccess::get(*rt).flushHostBuffer(src, 0U, kSize));

	auto gateResult = oavk::TimelineSemaphore::create(oa::EngineDeviceAccess::get(*rt), 0U);
	ASSERT_TRUE(gateResult.isOk());
	auto gate = std::move(*gateResult);
	oavk::Stream* blocker = oa::EngineSubmissionAccess::acquireStream(*rt);
	ASSERT_NE(blocker, nullptr);
	ASSERT_TRUE(blocker->begin(oa::EngineDeviceAccess::get(*rt)).isOk());
	ASSERT_TRUE(blocker->submitWithDependency(*rt, gate, 1U).isOk());

	// The compute queue cannot reach either copy until the host signals gate.
	// This makes the one-flight rejection deterministic rather than timing based.
	auto firstCopy = oa::EngineResourceAccess::copyBufferAsync(*rt, src, firstDst, kSize);
	std::mutex watchdogMutex;
	std::condition_variable watchdogCondition;
	bool copyCallReturned = false;
	bool watchdogTimedOut = false;
	std::atomic<oa::I32> gateSignalResult{static_cast<oa::I32>(VK_NOT_READY)};
	auto signalGate = [&]() {
		VkSemaphoreSignalInfo signalInfo = {
			.sType = VK_STRUCTURE_TYPE_SEMAPHORE_SIGNAL_INFO,
			.pNext = nullptr,
			.semaphore = static_cast<VkSemaphore>(gate.semaphore),
			.value = 1U,
		};
		gateSignalResult.store(static_cast<oa::I32>(
			oa::EngineDeviceAccess::get(*rt).deviceDispatch.vkSignalSemaphore(
			static_cast<VkDevice>(oa::EngineDeviceAccess::get(*rt).device), &signalInfo)),
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
	auto rejectedCopy = oa::EngineResourceAccess::copyBufferAsync(*rt, src, secondDst, kSize);
	{
		std::lock_guard lock(watchdogMutex);
		copyCallReturned = true;
	}
	watchdogCondition.notify_one();
	gateWatchdog.join();
	if (not watchdogTimedOut) signalGate();
	EXPECT_FALSE(watchdogTimedOut)
		<< "copyBufferAsync did not reject before the five-second watchdog timeout";
	EXPECT_EQ(gateSignalResult.load(std::memory_order_acquire),
		static_cast<oa::I32>(VK_SUCCESS));
	if (firstCopy.isOk()) EXPECT_TRUE(firstCopy->wait().isOk());
	if (rejectedCopy.isOk()) EXPECT_TRUE(rejectedCopy->wait().isOk());
	EXPECT_TRUE(blocker->synchronize(oa::EngineDeviceAccess::get(*rt)).isOk());
	oa::EngineSubmissionAccess::releaseStream(*rt, blocker);
	gate.destroy(oa::EngineDeviceAccess::get(*rt));

	ASSERT_TRUE(firstCopy.isOk());
	ASSERT_FALSE(rejectedCopy.isOk());
	EXPECT_EQ(rejectedCopy.getStatus().getCode(),
		oa::StatusCode::FailedPrecondition);
	oa::EngineResourceAccess::freeBuffer(*rt, src);
	oa::EngineResourceAccess::freeBuffer(*rt, firstDst);
	oa::EngineResourceAccess::freeBuffer(*rt, secondDst);
}

TEST(VkStream, AsyncTransferEventChainsGpuCopyConsumer) {
	auto* rt = testEnginePtr();
	ASSERT_NE(rt, nullptr);

	constexpr oa::U64 kSize = 4096;
	auto srcResult = oa::EngineResourceAccess::allocBuffer(*rt, kSize);
	ASSERT_TRUE(srcResult.isOk());
	auto src = std::move(*srcResult);
	auto intermediateResult = oa::EngineResourceAccess::allocBufferDevice(*rt, kSize);
	ASSERT_TRUE(intermediateResult.isOk());
	auto intermediate = std::move(*intermediateResult);
	auto dstResult = oa::EngineResourceAccess::allocBuffer(*rt, kSize);
	ASSERT_TRUE(dstResult.isOk());
	auto dst = std::move(*dstResult);

	auto* source = static_cast<oa::U8*>(src.mappedPtr);
	for (oa::U64 index = 0; index < kSize; ++index) {
		source[index] = static_cast<oa::U8>((index * 31U + 11U) & 0xFFU);
	}
	oa::memset(dst.mappedPtr, 0xC3, kSize);
	ASSERT_TRUE(oa::EngineAllocatorAccess::get(*rt).flushHostBuffer(src, 0U, kSize));

	auto producer = oa::EngineResourceAccess::copyBufferAsync(*rt, src, intermediate, kSize);
	ASSERT_TRUE(producer.isOk());
	oavk::Stream* consumer = oa::EngineSubmissionAccess::acquireStream(*rt);
	ASSERT_NE(consumer, nullptr);
	ASSERT_TRUE(consumer->begin(oa::EngineDeviceAccess::get(*rt)).isOk());
	consumer->recordCopyBuffer(intermediate, dst, kSize);
	consumer->recordTransferWriteBarrier(dst, 0U, kSize);
	const oavk::TimelineWait wait = oa::EventAccess::timelineWait(*producer);
	ASSERT_TRUE(consumer->submitWithDependencies(
		*rt, oa::Span<const oavk::TimelineWait>(&wait, 1)).isOk());
	ASSERT_TRUE(consumer->synchronize(oa::EngineDeviceAccess::get(*rt)).isOk());
	oa::EngineSubmissionAccess::releaseStream(*rt, consumer);
	EXPECT_TRUE(producer->isComplete());
	ASSERT_TRUE(oa::EngineAllocatorAccess::get(*rt).invalidateHostBuffer(dst, 0U, kSize));
	EXPECT_TRUE(oa::memEqual(source, dst.mappedPtr, kSize));

	oa::EngineResourceAccess::freeBuffer(*rt, src);
	oa::EngineResourceAccess::freeBuffer(*rt, intermediate);
	oa::EngineResourceAccess::freeBuffer(*rt, dst);
}

TEST(VkStream, MappedReadbackWaitsForPrimaryQueueProducer) {
	auto* rt = testEnginePtr();
	ASSERT_NE(rt, nullptr);

	constexpr oa::U64 kSize = 4096;
	auto srcResult = oa::EngineResourceAccess::allocBuffer(*rt, kSize);
	ASSERT_TRUE(srcResult.isOk());
	auto src = std::move(*srcResult);
	auto dstResult = oa::EngineResourceAccess::allocBuffer(*rt, kSize);
	ASSERT_TRUE(dstResult.isOk());
	auto dst = std::move(*dstResult);
	auto* source = static_cast<oa::U8*>(src.mappedPtr);
	for (oa::U64 index = 0; index < kSize; ++index) {
		source[index] = static_cast<oa::U8>((index * 31U + 9U) & 0xFFU);
	}
	oa::memset(dst.mappedPtr, 0xD3, kSize);
	ASSERT_TRUE(oa::EngineAllocatorAccess::get(*rt).flushHostBuffer(src, 0U, kSize));
	ASSERT_TRUE(oa::EngineAllocatorAccess::get(*rt).flushHostBuffer(dst, 0U, kSize));

	auto gateResult = oavk::TimelineSemaphore::create(oa::EngineDeviceAccess::get(*rt), 0U);
	ASSERT_TRUE(gateResult.isOk());
	auto gate = std::move(*gateResult);
	oavk::Stream* producer = oa::EngineSubmissionAccess::acquireStream(*rt);
	ASSERT_NE(producer, nullptr);
	ASSERT_TRUE(producer->begin(oa::EngineDeviceAccess::get(*rt)).isOk());
	producer->recordCopyBuffer(src, dst, kSize);
	ASSERT_TRUE(producer->submitWithDependency(*rt, gate, 1U).isOk());

	std::atomic<oa::I32> gateSignalResult{static_cast<oa::I32>(VK_NOT_READY)};
	std::thread gateWatchdog([&]() {
		std::this_thread::sleep_for(std::chrono::milliseconds(100));
		VkSemaphoreSignalInfo signalInfo = {
			.sType = VK_STRUCTURE_TYPE_SEMAPHORE_SIGNAL_INFO,
			.pNext = nullptr,
			.semaphore = static_cast<VkSemaphore>(gate.semaphore),
			.value = 1U,
		};
		gateSignalResult.store(static_cast<oa::I32>(
			oa::EngineDeviceAccess::get(*rt).deviceDispatch.vkSignalSemaphore(
			static_cast<VkDevice>(oa::EngineDeviceAccess::get(*rt).device), &signalInfo)),
			std::memory_order_release);
	});
	oa::U8 output[kSize] = {};
	const oa::Status readbackStatus = oa::EngineResourceAccess::readbackBuffer(*rt,
		dst, 0U, output, kSize);
	gateWatchdog.join();
	EXPECT_EQ(gateSignalResult.load(std::memory_order_acquire),
		static_cast<oa::I32>(VK_SUCCESS));
	EXPECT_TRUE(readbackStatus.isOk()) << readbackStatus.getMessage();
	EXPECT_TRUE(oa::memEqual(source, output, kSize));
	EXPECT_TRUE(producer->synchronize(oa::EngineDeviceAccess::get(*rt)).isOk());
	oa::EngineSubmissionAccess::releaseStream(*rt, producer);
	gate.destroy(oa::EngineDeviceAccess::get(*rt));
	oa::EngineResourceAccess::freeBuffer(*rt, src);
	oa::EngineResourceAccess::freeBuffer(*rt, dst);
}

TEST(VkStream, AsyncTransferPublishesPriorPrimaryQueueWrites) {
	auto* rt = testEnginePtr();
	ASSERT_NE(rt, nullptr);

	constexpr oa::U64 kSize = 4096;
	auto srcResult = oa::EngineResourceAccess::allocBuffer(*rt, kSize);
	ASSERT_TRUE(srcResult.isOk());
	auto src = std::move(*srcResult);
	auto intermediateResult = oa::EngineResourceAccess::allocBufferDevice(*rt, kSize);
	ASSERT_TRUE(intermediateResult.isOk());
	auto intermediate = std::move(*intermediateResult);
	auto dstResult = oa::EngineResourceAccess::allocBuffer(*rt, kSize);
	ASSERT_TRUE(dstResult.isOk());
	auto dst = std::move(*dstResult);

	auto* source = static_cast<oa::U8*>(src.mappedPtr);
	for (oa::U64 index = 0; index < kSize; ++index) {
		source[index] = static_cast<oa::U8>((index * 13U + 19U) & 0xFFU);
	}
	oa::memset(dst.mappedPtr, 0xB7, kSize);
	ASSERT_TRUE(oa::EngineAllocatorAccess::get(*rt).flushHostBuffer(src, 0U, kSize));

	oavk::Stream* producer = oa::EngineSubmissionAccess::acquireStream(*rt);
	ASSERT_NE(producer, nullptr);
	ASSERT_TRUE(producer->begin(oa::EngineDeviceAccess::get(*rt)).isOk());
	// Deliberately omit a producer-side post-copy barrier. copyBufferAsync must
	// publish this earlier primary-queue write before reading the source.
	producer->recordCopyBuffer(src, intermediate, kSize);
	ASSERT_TRUE(producer->submit(*rt).isOk());

	auto copied = oa::EngineResourceAccess::copyBufferAsync(*rt, intermediate, dst, kSize);
	ASSERT_TRUE(copied.isOk()) << copied.getStatus().getMessage();
	ASSERT_TRUE(copied->wait().isOk());
	ASSERT_TRUE(oa::EngineAllocatorAccess::get(*rt).invalidateHostBuffer(dst, 0U, kSize));
	EXPECT_TRUE(oa::memEqual(source, dst.mappedPtr, kSize));
	EXPECT_TRUE(producer->synchronize(oa::EngineDeviceAccess::get(*rt)).isOk());
	oa::EngineSubmissionAccess::releaseStream(*rt, producer);

	oa::EngineResourceAccess::freeBuffer(*rt, src);
	oa::EngineResourceAccess::freeBuffer(*rt, intermediate);
	oa::EngineResourceAccess::freeBuffer(*rt, dst);
}

TEST(VkStream, DeviceLocalReadbackPublishesPriorPrimaryQueueWrites) {
	auto* rt = testEnginePtr();
	ASSERT_NE(rt, nullptr);

	constexpr oa::U64 kSize = 4096;
	auto srcResult = oa::EngineResourceAccess::allocBuffer(*rt, kSize);
	ASSERT_TRUE(srcResult.isOk());
	auto src = std::move(*srcResult);
	auto deviceResult = oa::EngineResourceAccess::allocBufferDevice(*rt, kSize);
	ASSERT_TRUE(deviceResult.isOk());
	auto device = std::move(*deviceResult);
	auto* source = static_cast<oa::U8*>(src.mappedPtr);
	for (oa::U64 index = 0; index < kSize; ++index) {
		source[index] = static_cast<oa::U8>((index * 47U + 3U) & 0xFFU);
	}
	ASSERT_TRUE(oa::EngineAllocatorAccess::get(*rt).flushHostBuffer(src, 0U, kSize));

	oavk::Stream* producer = oa::EngineSubmissionAccess::acquireStream(*rt);
	ASSERT_NE(producer, nullptr);
	ASSERT_TRUE(producer->begin(oa::EngineDeviceAccess::get(*rt)).isOk());
	// readbackBuffer owns the producer-write to transfer-read dependency.
	producer->recordCopyBuffer(src, device, kSize);
	ASSERT_TRUE(producer->submit(*rt).isOk());

	oa::U8 output[kSize] = {};
	const oa::Status readback = oa::EngineResourceAccess::readbackBuffer(*rt, device, 0U, output, kSize);
	ASSERT_TRUE(readback.isOk()) << readback.getMessage();
	EXPECT_TRUE(oa::memEqual(source, output, kSize));
	EXPECT_TRUE(producer->synchronize(oa::EngineDeviceAccess::get(*rt)).isOk());
	oa::EngineSubmissionAccess::releaseStream(*rt, producer);
	oa::EngineResourceAccess::freeBuffer(*rt, src);
	oa::EngineResourceAccess::freeBuffer(*rt, device);
}

TEST(VkStream, ReadbackRejectsUnownedAndAliasedBuffers) {
	auto* rt = testEnginePtr();
	ASSERT_NE(rt, nullptr);

	constexpr oa::U64 kSize = 256;
	auto srcResult = oa::EngineResourceAccess::allocBuffer(*rt, kSize);
	ASSERT_TRUE(srcResult.isOk());
	auto src = std::move(*srcResult);
	oa::U8 output[kSize] = {};

	oavk::Buffer foreignAllocator = src;
	foreignAllocator.allocatorIdentity = &foreignAllocator;
	EXPECT_EQ(oa::EngineResourceAccess::readbackBuffer(*rt, foreignAllocator, 0U, output, kSize).getCode(),
		oa::StatusCode::InvalidArgument);
	oavk::Buffer external = src;
	external.allocation = nullptr;
	EXPECT_EQ(oa::EngineResourceAccess::readbackBuffer(*rt, external, 0U, output, kSize).getCode(),
		oa::StatusCode::InvalidArgument);
	oavk::Buffer otherEngine = src;
	otherEngine.allocatorIdentity = &otherEngine;
	EXPECT_EQ(oa::EngineResourceAccess::readbackBuffer(*rt, otherEngine, 0U, output, kSize).getCode(),
		oa::StatusCode::InvalidArgument);
	oavk::Buffer alias = src;
	alias.aliasIdentity = src.allocation;
	EXPECT_EQ(oa::EngineResourceAccess::readbackBuffer(*rt, alias, 0U, output, kSize).getCode(),
		oa::StatusCode::InvalidArgument);

	oa::EngineResourceAccess::freeBuffer(*rt, src);
}

// ─── Concurrent pool acquire/release ──────────────────────────────────────────

TEST(VkStream, ConcurrentPoolAccess) {
	auto* rt = testEnginePtr();
	ASSERT_NE(rt, nullptr);

	constexpr oa::I32 kThreads = 4;
	constexpr oa::I32 kOps = 50;
	std::atomic<oa::I32> counter{0};

	auto worker = [&]() {
		for (oa::I32 i = 0; i < kOps; ++i) {
			oavk::Stream* s = oa::EngineSubmissionAccess::acquireStream(*rt);
			ASSERT_NE(s, nullptr);
			counter.fetch_add(1, std::memory_order_relaxed);
			oa::EngineSubmissionAccess::releaseStream(*rt, s);
		}
	};

	oa::Vec<std::thread> threads;
	for (oa::I32 i = 0; i < kThreads; ++i) {
		threads.emplaceBack(worker);
	}
	for (auto& t : threads) t.join();

	EXPECT_EQ(counter.load(), kThreads * kOps);
}

// ─── CopyBuffer Recording ────────────────────────────────────────────────────

TEST(VkStream, recordCopyBuffer) {
	auto* rt = testEnginePtr();
	ASSERT_NE(rt, nullptr);

	constexpr oa::U64 kSize = 512;
	auto srcRes = oa::EngineAllocatorAccess::get(*rt).allocHostVisible(kSize);
	ASSERT_TRUE(srcRes.isOk());
	auto src = srcRes.getValue();
	ASSERT_NE(oa::EngineBindlessAccess::registerBuffer(*rt, src), OA_BINDLESS_INVALID);

	auto dstRes = oa::EngineAllocatorAccess::get(*rt).allocHostVisible(kSize);
	ASSERT_TRUE(dstRes.isOk());
	auto dst = dstRes.getValue();
	ASSERT_NE(oa::EngineBindlessAccess::registerBuffer(*rt, dst), OA_BINDLESS_INVALID);

	oa::U8* data = static_cast<oa::U8*>(src.mappedPtr);
	for (oa::U32 i = 0; i < kSize; ++i) data[i] = static_cast<oa::U8>(42);
	ASSERT_TRUE(oa::EngineAllocatorAccess::get(*rt).flushHostBuffer(src, 0U, kSize));

	oavk::Stream* stream = oa::EngineSubmissionAccess::acquireStream(*rt);
	ASSERT_NE(stream, nullptr);
	ASSERT_TRUE(stream->begin(oa::EngineDeviceAccess::get(*rt)).isOk());
	stream->recordCopyBuffer(src, dst, kSize);
	stream->recordTransferWriteBarrier(dst, 0U, kSize);
	ASSERT_TRUE(stream->submitAndWait(*rt).isOk());
	oa::EngineSubmissionAccess::releaseStream(*rt, stream);
	ASSERT_TRUE(oa::EngineAllocatorAccess::get(*rt).invalidateHostBuffer(dst, 0U, kSize));

	oa::U8* out = static_cast<oa::U8*>(dst.mappedPtr);
	for (oa::U32 i = 0; i < kSize; ++i) {
		EXPECT_EQ(out[i], 42);
	}

	oa::EngineBindlessAccess::deregisterBuffer(*rt, src);
	oa::EngineAllocatorAccess::get(*rt).free(src);
	oa::EngineBindlessAccess::deregisterBuffer(*rt, dst);
	oa::EngineAllocatorAccess::get(*rt).free(dst);
}

static oa::Result<oa::UniquePtr<oa::Engine>> createHeadlessGraphicsTestEngine() {
	auto config = testEngineConfig(oa::Precision::FP32);
	config.presentationMode = oa::PresentationMode::Headless;
	config.selectForThread = false;
	config.preloadEmbeddedPipelines = false;
	config.enablePipelineCache = false;
	return oa::Engine::create(config);
}

TEST(GraphicsQueueRoute, ClassifiesMergedDistinctAndUnknownHandles) {
	auto* compute = reinterpret_cast<void*>(static_cast<std::uintptr_t>(0x1000U));
	auto* asyncCompute = reinterpret_cast<void*>(static_cast<std::uintptr_t>(0x2000U));
	auto* transfer = reinterpret_cast<void*>(static_cast<std::uintptr_t>(0x3000U));
	auto* graphics = reinterpret_cast<void*>(static_cast<std::uintptr_t>(0x4000U));
	auto* present = reinterpret_cast<void*>(static_cast<std::uintptr_t>(0x5000U));
	auto* unknown = reinterpret_cast<void*>(static_cast<std::uintptr_t>(0x6000U));

	oavk::Queues queues;
	queues.computeQueue = compute;
	queues.asyncComputeQueue = asyncCompute;
	queues.transferQueue = transfer;
	queues.graphicsQueue = compute;
	queues.presentQueue = compute;
	queues.hasAsyncCompute = true;
	EXPECT_EQ(oavk::classifyQueueSubmitRoute(queues, compute),
		oavk::QueueSubmitRoute::Compute);
	EXPECT_EQ(oavk::classifyQueueSubmitRoute(queues, asyncCompute),
		oavk::QueueSubmitRoute::AsyncCompute);
	EXPECT_EQ(oavk::classifyQueueSubmitRoute(queues, transfer),
		oavk::QueueSubmitRoute::Transfer);

	queues.graphicsQueue = graphics;
	queues.presentQueue = present;
	EXPECT_EQ(oavk::classifyQueueSubmitRoute(queues, graphics),
		oavk::QueueSubmitRoute::Graphics);
	EXPECT_EQ(oavk::classifyQueueSubmitRoute(queues, present),
		oavk::QueueSubmitRoute::Present);
	EXPECT_EQ(oavk::classifyQueueSubmitRoute(queues, unknown),
		oavk::QueueSubmitRoute::Unknown);
	EXPECT_EQ(oavk::classifyQueueSubmitRoute(queues, nullptr),
		oavk::QueueSubmitRoute::Unknown);
}

TEST(VkStream, HeadlessGraphicsLeaseReturnsExactGenerationSafeEvent) {
	auto engineResult = createHeadlessGraphicsTestEngine();
	ASSERT_TRUE(engineResult.isOk()) << engineResult.getStatus().getMessage();
	auto engine = std::move(*engineResult);
	if (oa::EngineDeviceAccess::get(*engine).queues.graphicsQueue
		!= oa::EngineDeviceAccess::get(*engine).queues.computeQueue) {
		ASSERT_TRUE(engine->close().isOk());
		GTEST_SKIP() << "merged graphics/compute queue required by this hardware proof";
	}

	auto firstResult = oa::GraphicsStreamLease::acquire(*engine);
	ASSERT_TRUE(firstResult.isOk()) << firstResult.getStatus().getMessage();
	auto first = oa::move(*firstResult);
	oavk::Stream* firstStream = first.getStream();
	ASSERT_NE(firstStream, nullptr);
	auto firstCompletion = first.submit();
	ASSERT_TRUE(firstCompletion.isOk())
		<< firstCompletion.getStatus().getMessage();
	EXPECT_TRUE(engine->ownsEvent(*firstCompletion));
	EXPECT_TRUE(firstCompletion->hasQueueFamily());
	EXPECT_EQ(firstCompletion->queueFamily(),
		oa::EngineDeviceAccess::get(*engine).queues.graphicsQueueFamily);
	ASSERT_TRUE(firstCompletion->wait().isOk());
	ASSERT_TRUE(first.recycle(*firstCompletion).isOk());
	EXPECT_FALSE(first.isValid());

	auto secondResult = oa::GraphicsStreamLease::acquire(*engine);
	ASSERT_TRUE(secondResult.isOk()) << secondResult.getStatus().getMessage();
	auto second = oa::move(*secondResult);
	EXPECT_EQ(second.getStream(), firstStream);
	auto secondCompletion = second.submit();
	ASSERT_TRUE(secondCompletion.isOk())
		<< secondCompletion.getStatus().getMessage();
	ASSERT_TRUE(secondCompletion->wait().isOk());
	const oa::Status staleRecycle = second.recycle(*firstCompletion);
	EXPECT_EQ(staleRecycle.getCode(), oa::StatusCode::InvalidArgument);
	EXPECT_TRUE(second.isValid());
	ASSERT_TRUE(second.recycle(*secondCompletion).isOk());

	VkSubmitInfo submitInfo = {.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO};
	VkSubmitInfo2 submitInfo2 = {.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO_2};
	void* unknown = reinterpret_cast<void*>(
		static_cast<std::uintptr_t>(0xBAD0U));
	EXPECT_EQ(oa::EngineSubmissionAccess::submit(
		*engine, unknown, &submitInfo, nullptr).getCode(),
		oa::StatusCode::InvalidArgument);
	EXPECT_EQ(oa::EngineSubmissionAccess::submit2(
		*engine, unknown, &submitInfo2).getCode(),
		oa::StatusCode::InvalidArgument);
	ASSERT_TRUE(engine->close().isOk());
}

TEST(VkStream, GraphicsLeaseRejectsForeignUnknownAndCrossFamilyDependencies) {
	auto engineResult = createHeadlessGraphicsTestEngine();
	ASSERT_TRUE(engineResult.isOk()) << engineResult.getStatus().getMessage();
	auto engine = std::move(*engineResult);
	auto gateResult = oavk::TimelineSemaphore::create(oa::EngineDeviceAccess::get(*engine), 0U);
	ASSERT_TRUE(gateResult.isOk()) << gateResult.getStatus().getMessage();
	auto gate = oa::move(*gateResult);

	oavk::Device foreignDevice;
	const oa::Event foreign = oa::EventAccess::create(
		foreignDevice,
		gate,
		1U,
		oa::EngineDeviceAccess::get(*engine).queues.graphicsQueueFamily);
	const oa::Event unknownFamily = oa::EventAccess::create(
		oa::EngineDeviceAccess::get(*engine), gate, 1U);
	oa::U32 otherFamily = oa::EngineDeviceAccess::get(*engine).queues.graphicsQueueFamily ^ 1U;
	if (otherFamily == oa::Event::UnknownQueueFamily) otherFamily = 0U;
	const oa::Event crossFamily = oa::EventAccess::create(
		oa::EngineDeviceAccess::get(*engine), gate, 1U, otherFamily);
	EXPECT_FALSE(engine->ownsEvent(foreign));
	EXPECT_TRUE(engine->ownsEvent(unknownFamily));
	EXPECT_TRUE(engine->ownsEvent(crossFamily));

	auto leaseResult = oa::GraphicsStreamLease::acquire(*engine);
	ASSERT_TRUE(leaseResult.isOk()) << leaseResult.getStatus().getMessage();
	auto lease = oa::move(*leaseResult);
	const oa::Event foreignDeps[] = {foreign};
	auto foreignSubmit = lease.submit(foreignDeps);
	EXPECT_FALSE(foreignSubmit.isOk());
	EXPECT_EQ(foreignSubmit.getStatus().getCode(), oa::StatusCode::InvalidArgument);
	ASSERT_NE(lease.getStream(), nullptr);

	const oa::Event unknownDeps[] = {unknownFamily};
	auto unknownSubmit = lease.submit(unknownDeps);
	EXPECT_FALSE(unknownSubmit.isOk());
	EXPECT_EQ(unknownSubmit.getStatus().getCode(), oa::StatusCode::InvalidArgument);
	ASSERT_NE(lease.getStream(), nullptr);

	const oa::Event crossDeps[] = {crossFamily};
	auto crossSubmit = lease.submit(crossDeps);
	EXPECT_FALSE(crossSubmit.isOk());
	EXPECT_EQ(crossSubmit.getStatus().getCode(),
		oa::StatusCode::FailedPrecondition);
	ASSERT_NE(lease.getStream(), nullptr);
	ASSERT_TRUE(lease.cancel().isOk());

	gate.destroy(oa::EngineDeviceAccess::get(*engine));
	ASSERT_TRUE(engine->close().isOk());
}

TEST(VkStream, GraphicsLeaseChainsSameFamilyDependencyOnGpu) {
	auto engineResult = createHeadlessGraphicsTestEngine();
	ASSERT_TRUE(engineResult.isOk()) << engineResult.getStatus().getMessage();
	auto engine = std::move(*engineResult);
	if (oa::EngineDeviceAccess::get(*engine).queues.graphicsQueueFamily
		!= oa::EngineDeviceAccess::get(*engine).queues.computeQueueFamily) {
		ASSERT_TRUE(engine->close().isOk());
		GTEST_SKIP() << "cross-family resource ownership is intentionally deferred";
	}

	constexpr oa::U64 kSize = 4096;
	auto srcResult = oa::EngineResourceAccess::allocBuffer(*engine, kSize);
	ASSERT_TRUE(srcResult.isOk());
	auto src = oa::move(*srcResult);
	auto intermediateResult = oa::EngineResourceAccess::allocBuffer(*engine, kSize);
	ASSERT_TRUE(intermediateResult.isOk());
	auto intermediate = oa::move(*intermediateResult);
	auto dstResult = oa::EngineResourceAccess::allocBuffer(*engine, kSize);
	ASSERT_TRUE(dstResult.isOk());
	auto dst = oa::move(*dstResult);
	auto* source = static_cast<oa::U8*>(src.mappedPtr);
	for (oa::U64 index = 0; index < kSize; ++index) {
		source[index] = static_cast<oa::U8>((index * 37U + 13U) & 0xFFU);
	}
	oa::memset(dst.mappedPtr, 0xB5, kSize);
	ASSERT_TRUE(oa::EngineAllocatorAccess::get(*engine).flushHostBuffer(src, 0U, kSize));
	ASSERT_TRUE(oa::EngineAllocatorAccess::get(*engine).flushHostBuffer(dst, 0U, kSize));

	oavk::Stream* producer = oa::EngineSubmissionAccess::acquireStream(*engine);
	ASSERT_NE(producer, nullptr);
	ASSERT_TRUE(producer->begin(oa::EngineDeviceAccess::get(*engine)).isOk());
	producer->recordCopyBuffer(src, intermediate, kSize);
	producer->recordTransferWriteBarrier(intermediate, 0U, kSize);
	ASSERT_TRUE(producer->submit(*engine).isOk());
	const oa::Event producerCompletion = producer->completion(oa::EngineDeviceAccess::get(*engine));
	ASSERT_TRUE(producerCompletion.isValid());
	ASSERT_EQ(producerCompletion.queueFamily(),
		oa::EngineDeviceAccess::get(*engine).queues.graphicsQueueFamily);

	auto consumerResult = oa::GraphicsStreamLease::acquire(*engine);
	ASSERT_TRUE(consumerResult.isOk()) << consumerResult.getStatus().getMessage();
	auto consumer = oa::move(*consumerResult);
	oavk::Stream* stream = consumer.getStream();
	ASSERT_NE(stream, nullptr);
	stream->recordCopyBuffer(intermediate, dst, kSize);
	stream->recordTransferWriteBarrier(dst, 0U, kSize);
	const oa::Event dependencies[] = {producerCompletion};
	auto consumed = consumer.submit(dependencies);
	ASSERT_TRUE(consumed.isOk()) << consumed.getStatus().getMessage();
	ASSERT_TRUE(consumed->wait().isOk());
	ASSERT_TRUE(consumer.recycle(*consumed).isOk());
	ASSERT_TRUE(producer->synchronize(oa::EngineDeviceAccess::get(*engine)).isOk());
	oa::EngineSubmissionAccess::releaseStream(*engine, producer);
	ASSERT_TRUE(oa::EngineAllocatorAccess::get(*engine).invalidateHostBuffer(dst, 0U, kSize));
	EXPECT_TRUE(oa::memEqual(source, dst.mappedPtr, kSize));

	oa::EngineResourceAccess::freeBuffer(*engine, src);
	oa::EngineResourceAccess::freeBuffer(*engine, intermediate);
	oa::EngineResourceAccess::freeBuffer(*engine, dst);
	ASSERT_TRUE(engine->close().isOk());
}

TEST(VkStream, GraphicsLeaseCancellationDoesNotSubmitRecordedWork) {
	auto engineResult = createHeadlessGraphicsTestEngine();
	ASSERT_TRUE(engineResult.isOk()) << engineResult.getStatus().getMessage();
	auto engine = std::move(*engineResult);
	constexpr oa::U64 kSize = 1024;
	auto srcResult = oa::EngineResourceAccess::allocBuffer(*engine, kSize);
	ASSERT_TRUE(srcResult.isOk());
	auto src = oa::move(*srcResult);
	auto dstResult = oa::EngineResourceAccess::allocBuffer(*engine, kSize);
	ASSERT_TRUE(dstResult.isOk());
	auto dst = oa::move(*dstResult);
	oa::memset(src.mappedPtr, 0x31, kSize);
	oa::memset(dst.mappedPtr, 0xC9, kSize);
	ASSERT_TRUE(oa::EngineAllocatorAccess::get(*engine).flushHostBuffer(src, 0U, kSize));
	ASSERT_TRUE(oa::EngineAllocatorAccess::get(*engine).flushHostBuffer(dst, 0U, kSize));

	auto leaseResult = oa::GraphicsStreamLease::acquire(*engine);
	ASSERT_TRUE(leaseResult.isOk()) << leaseResult.getStatus().getMessage();
	auto lease = oa::move(*leaseResult);
	oavk::Stream* cancelledStream = lease.getStream();
	ASSERT_NE(cancelledStream, nullptr);
	cancelledStream->recordCopyBuffer(src, dst, kSize);
	ASSERT_TRUE(lease.cancel().isOk());
	ASSERT_TRUE(oa::EngineAllocatorAccess::get(*engine).invalidateHostBuffer(dst, 0U, kSize));
	const auto* values = static_cast<const oa::U8*>(dst.mappedPtr);
	for (oa::U64 index = 0; index < kSize; ++index) {
		ASSERT_EQ(values[index], 0xC9);
	}

	auto reusedResult = oa::GraphicsStreamLease::acquire(*engine);
	ASSERT_TRUE(reusedResult.isOk()) << reusedResult.getStatus().getMessage();
	auto reused = oa::move(*reusedResult);
	EXPECT_EQ(reused.getStream(), cancelledStream);
	ASSERT_TRUE(reused.cancel().isOk());
	oa::EngineResourceAccess::freeBuffer(*engine, src);
	oa::EngineResourceAccess::freeBuffer(*engine, dst);
	ASSERT_TRUE(engine->close().isOk());
}

TEST(VkStream, GraphicsLeaseRetirementAndAcquireNeverWait) {
	auto engineResult = createHeadlessGraphicsTestEngine();
	ASSERT_TRUE(engineResult.isOk()) << engineResult.getStatus().getMessage();
	auto engine = std::move(*engineResult);
	auto gateResult = oavk::TimelineSemaphore::create(oa::EngineDeviceAccess::get(*engine), 0U);
	ASSERT_TRUE(gateResult.isOk()) << gateResult.getStatus().getMessage();
	auto gate = oa::move(*gateResult);
	const oa::Event gateEvent = oa::EventAccess::create(
		oa::EngineDeviceAccess::get(*engine),
		gate,
		1U,
		oa::EngineDeviceAccess::get(*engine).queues.graphicsQueueFamily);

	auto leaseResult = oa::GraphicsStreamLease::acquire(*engine);
	ASSERT_TRUE(leaseResult.isOk()) << leaseResult.getStatus().getMessage();
	auto lease = oa::move(*leaseResult);
	oavk::Stream* retiredStream = lease.getStream();
	ASSERT_NE(retiredStream, nullptr);
	const oa::Event dependencies[] = {gateEvent};
	auto submitted = lease.submit(dependencies);
	ASSERT_TRUE(submitted.isOk()) << submitted.getStatus().getMessage();

	auto signalGate = [&]() {
		VkSemaphoreSignalInfo signalInfo = {
			.sType = VK_STRUCTURE_TYPE_SEMAPHORE_SIGNAL_INFO,
			.semaphore = static_cast<VkSemaphore>(gate.semaphore),
			.value = 1U,
		};
		return oa::EngineDeviceAccess::get(*engine).deviceDispatch.vkSignalSemaphore(
			static_cast<VkDevice>(oa::EngineDeviceAccess::get(*engine).device), &signalInfo);
	};
	std::mutex watchdogMutex;
	std::condition_variable watchdogCondition;
	bool closeReturned = false;
	bool closeTimedOut = false;
	std::atomic<oa::I32> watchdogSignal{static_cast<oa::I32>(VK_NOT_READY)};
	std::thread closeWatchdog([&]() {
		std::unique_lock lock(watchdogMutex);
		if (not watchdogCondition.wait_for(lock, std::chrono::seconds(2),
			[&]() { return closeReturned; })) {
			closeTimedOut = true;
			lock.unlock();
			watchdogSignal.store(
				static_cast<oa::I32>(signalGate()), std::memory_order_release);
		}
	});
	const oa::Status closeStatus = lease.close();
	{
		std::lock_guard lock(watchdogMutex);
		closeReturned = true;
	}
	watchdogCondition.notify_one();
	closeWatchdog.join();
	EXPECT_TRUE(closeStatus.isOk()) << closeStatus.getMessage();
	EXPECT_FALSE(closeTimedOut) << "graphics lease Close waited for GPU completion";
	EXPECT_FALSE(submitted->isComplete());

	bool acquireReturned = false;
	bool acquireTimedOut = false;
	std::thread acquireWatchdog([&]() {
		std::unique_lock lock(watchdogMutex);
		if (not watchdogCondition.wait_for(lock, std::chrono::seconds(2),
			[&]() { return acquireReturned; })) {
			acquireTimedOut = true;
			lock.unlock();
			if (watchdogSignal.load(std::memory_order_acquire)
				== static_cast<oa::I32>(VK_NOT_READY)) {
				watchdogSignal.store(
					static_cast<oa::I32>(signalGate()), std::memory_order_release);
			}
		}
	});
	auto secondResult = oa::GraphicsStreamLease::acquire(*engine);
	{
		std::lock_guard lock(watchdogMutex);
		acquireReturned = true;
	}
	watchdogCondition.notify_one();
	acquireWatchdog.join();
	if (not secondResult.isOk()) {
		if (watchdogSignal.load(std::memory_order_acquire)
			== static_cast<oa::I32>(VK_NOT_READY)) {
			watchdogSignal.store(
				static_cast<oa::I32>(signalGate()), std::memory_order_release);
		}
		const oa::Status submittedStatus = submitted->wait();
		gate.destroy(oa::EngineDeviceAccess::get(*engine));
		const oa::Status engineStatus = engine->close();
		ADD_FAILURE()
			<< secondResult.getStatus().getMessage()
			<< "; pending completion cleanup: "
			<< submittedStatus.getMessage()
			<< "; engine cleanup: " << engineStatus.getMessage();
		return;
	}
	auto second = oa::move(*secondResult);
	EXPECT_FALSE(acquireTimedOut)
		<< "graphics stream acquisition waited for a retired slot";
	EXPECT_NE(second.getStream(), retiredStream);
	ASSERT_TRUE(second.cancel().isOk());

	if (watchdogSignal.load(std::memory_order_acquire)
		== static_cast<oa::I32>(VK_NOT_READY)) {
		watchdogSignal.store(
			static_cast<oa::I32>(signalGate()), std::memory_order_release);
	}
	ASSERT_EQ(watchdogSignal.load(std::memory_order_acquire),
		static_cast<oa::I32>(VK_SUCCESS));
	ASSERT_TRUE(submitted->wait().isOk());

	auto recycledResult = oa::GraphicsStreamLease::acquire(*engine);
	ASSERT_TRUE(recycledResult.isOk()) << recycledResult.getStatus().getMessage();
	auto recycled = oa::move(*recycledResult);
	EXPECT_EQ(recycled.getStream(), retiredStream);
	ASSERT_TRUE(recycled.cancel().isOk());
	gate.destroy(oa::EngineDeviceAccess::get(*engine));
	ASSERT_TRUE(engine->close().isOk());
}

TEST(VkStream, PresenterDestructionCancelsUnsubmittedGraphicsBatch) {
	auto engineResult = createHeadlessGraphicsTestEngine();
	ASSERT_TRUE(engineResult.isOk()) << engineResult.getStatus().getMessage();
	auto engine = std::move(*engineResult);
	ASSERT_NE(oa::EngineDeviceAccess::get(*engine).queues.graphicsQueue, nullptr);

	constexpr oa::U64 kSize = 4096;
	auto srcResult = oa::EngineResourceAccess::allocBuffer(*engine, kSize);
	ASSERT_TRUE(srcResult.isOk());
	auto src = std::move(*srcResult);
	auto dstResult = oa::EngineResourceAccess::allocBuffer(*engine, kSize);
	ASSERT_TRUE(dstResult.isOk());
	auto dst = std::move(*dstResult);
	oa::memset(src.mappedPtr, 0x39, kSize);
	oa::memset(dst.mappedPtr, 0xC7, kSize);

	{
		oa::Presenter presenter(*engine);
		ASSERT_TRUE(presenter.beginGraphicsBatch().isOk());
		auto* stream = presenter.activeGraphicsBatchStream();
		ASSERT_NE(stream, nullptr);
		stream->recordCopyBuffer(src, dst, kSize);
	}

	ASSERT_TRUE(oa::EngineAllocatorAccess::get(*engine).invalidateHostBuffer(dst, 0, kSize));
	const auto* values = static_cast<const oa::U8*>(dst.mappedPtr);
	for (oa::U64 index = 0; index < kSize; ++index) {
		ASSERT_EQ(values[index], 0xC7);
	}
	oa::EngineResourceAccess::freeBuffer(*engine, src);
	oa::EngineResourceAccess::freeBuffer(*engine, dst);
	ASSERT_TRUE(engine->close().isOk());
}

TEST(VkStream, PresenterDestructionRetiresSubmittedGraphicsBatch) {
	auto engineResult = createHeadlessGraphicsTestEngine();
	ASSERT_TRUE(engineResult.isOk()) << engineResult.getStatus().getMessage();
	auto engine = std::move(*engineResult);
	ASSERT_NE(oa::EngineDeviceAccess::get(*engine).queues.graphicsQueue, nullptr);

	constexpr oa::U64 kSize = 1024 * 1024;
	auto srcResult = oa::EngineResourceAccess::allocBuffer(*engine, kSize);
	ASSERT_TRUE(srcResult.isOk());
	auto src = std::move(*srcResult);
	auto dstResult = oa::EngineResourceAccess::allocBuffer(*engine, kSize);
	ASSERT_TRUE(dstResult.isOk());
	auto dst = std::move(*dstResult);
	auto* source = static_cast<oa::U8*>(src.mappedPtr);
	for (oa::U64 index = 0; index < kSize; ++index) {
		source[index] = static_cast<oa::U8>((index * 23U + 5U) & 0xFFU);
	}
	ASSERT_TRUE(oa::EngineAllocatorAccess::get(*engine).flushHostBuffer(src, 0, kSize));

	oa::Event completion;
	{
		oa::Presenter presenter(*engine);
		ASSERT_TRUE(presenter.beginGraphicsBatch().isOk());
		auto* stream = presenter.activeGraphicsBatchStream();
		ASSERT_NE(stream, nullptr);
		stream->recordCopyBuffer(src, dst, kSize);
		stream->recordTransferWriteBarrier(dst, 0U, kSize);
		ASSERT_TRUE(presenter.flushGraphicsBatch().isOk());
		completion = stream->completion(oa::EngineDeviceAccess::get(*engine));
		ASSERT_TRUE(completion.isValid());
	}

	// The stream remains at its original heap address in engine retirement, so
	// the non-owning event remains valid after the presenter facade is gone.
	ASSERT_TRUE(completion.wait().isOk());
	ASSERT_TRUE(oa::EngineAllocatorAccess::get(*engine).invalidateHostBuffer(dst, 0, kSize));
	const auto* values = static_cast<const oa::U8*>(dst.mappedPtr);
	for (oa::U64 index = 0; index < kSize; index += 4093U) {
		EXPECT_EQ(values[index], source[index]);
	}
	oa::EngineResourceAccess::freeBuffer(*engine, src);
	oa::EngineResourceAccess::freeBuffer(*engine, dst);
	ASSERT_TRUE(engine->close().isOk());
}

// ─── persistent mapped upload ring ───────────────────────────────────────────

TEST(VkStream, UploadRingBatchesRegionsAndRecyclesFrames) {
	auto* rt = testEnginePtr();
	ASSERT_NE(rt, nullptr);

	constexpr oa::U64 kDstSize = 4096;
	auto dstAResult = oa::EngineResourceAccess::allocBuffer(*rt, kDstSize);
	ASSERT_TRUE(dstAResult.isOk());
	auto dstA = std::move(*dstAResult);
	auto dstBResult = oa::EngineResourceAccess::allocBuffer(*rt, kDstSize);
	ASSERT_TRUE(dstBResult.isOk());
	auto dstB = std::move(*dstBResult);

	auto* dstAData = static_cast<oa::U8*>(dstA.mappedPtr);
	auto* dstBData = static_cast<oa::U8*>(dstB.mappedPtr);
	ASSERT_NE(dstAData, nullptr);
	ASSERT_NE(dstBData, nullptr);
	oa::memset(dstAData, 0xCD, kDstSize);
	oa::memset(dstBData, 0xCD, kDstSize);

	auto ringResult = oa::UploadRing::create(*rt, oa::UploadRingConfig{
		.capacityBytes = 3 * 4096,
		.framesInFlight = 3,
		.alignment = 64,
	});
	ASSERT_TRUE(ringResult.isOk()) << ringResult.getStatus().getMessage();
	auto ring = std::move(*ringResult);
	EXPECT_EQ(ring.frameCapacityBytes(), 4096u);

	oa::U8 first[256];
	oa::U8 second[512];
	oa::U8 third[128];
	for (oa::U32 i = 0; i < 256; ++i) first[i] = static_cast<oa::U8>(i);
	for (oa::U32 i = 0; i < 512; ++i) second[i] = static_cast<oa::U8>(255u - (i & 0xFFu));
	for (oa::U32 i = 0; i < 128; ++i) third[i] = static_cast<oa::U8>(0x80u + i);

	// Seven batches lap the three frame arenas twice. beginBatch must only wait
	// when its own arena is recycled; earlier frames remain independently in flight.
	for (oa::U32 batch = 0; batch < 7; ++batch) {
		ASSERT_TRUE(ring.beginBatch().isOk());
		ASSERT_TRUE(ring.upload(dstA, 0, first, sizeof(first)).isOk());
		ASSERT_TRUE(ring.upload(dstA, 1024, second, sizeof(second)).isOk());
		ASSERT_TRUE(ring.upload(dstB, 256, third, sizeof(third)).isOk());
		EXPECT_EQ(ring.pendingCopyCount(), 3u);
		EXPECT_LE(ring.bytesUsed(), ring.frameCapacityBytes());
		auto completion = ring.submit();
		ASSERT_TRUE(completion.isOk()) << completion.getStatus().getMessage();
		ASSERT_TRUE(completion->isValid());
	}
	ASSERT_TRUE(ring.wait().isOk());
	ASSERT_TRUE(oa::EngineAllocatorAccess::get(*rt).invalidateHostBuffer(dstA, 0, kDstSize));
	ASSERT_TRUE(oa::EngineAllocatorAccess::get(*rt).invalidateHostBuffer(dstB, 0, kDstSize));

	for (oa::U32 i = 0; i < 256; ++i) EXPECT_EQ(dstAData[i], first[i]);
	for (oa::U32 i = 0; i < 512; ++i) EXPECT_EQ(dstAData[1024 + i], second[i]);
	for (oa::U32 i = 0; i < 128; ++i) EXPECT_EQ(dstBData[256 + i], third[i]);
	EXPECT_EQ(dstAData[512], 0xCD);
	EXPECT_EQ(dstBData[0], 0xCD);

	ASSERT_TRUE(ring.close().isOk());
	oa::EngineResourceAccess::freeBuffer(*rt, dstA);
	oa::EngineResourceAccess::freeBuffer(*rt, dstB);
}

TEST(VkStream, UploadRingRejectsInvalidRangesAndClosesEmptyBatch) {
	auto* rt = testEnginePtr();
	ASSERT_NE(rt, nullptr);

	auto dstResult = oa::EngineResourceAccess::allocBuffer(*rt, 1024);
	ASSERT_TRUE(dstResult.isOk());
	auto dst = std::move(*dstResult);
	auto ringResult = oa::UploadRing::create(*rt, oa::UploadRingConfig{
		.capacityBytes = 2 * 1024,
		.framesInFlight = 2,
		.alignment = 64,
	});
	ASSERT_TRUE(ringResult.isOk());
	auto ring = std::move(*ringResult);

	ASSERT_TRUE(ring.beginBatch().isOk());
	EXPECT_FALSE(ring.reserve(2048).isOk());
	auto staleSlice = ring.reserve(8);
	ASSERT_TRUE(staleSlice.isOk());
	auto emptyCompletion = ring.submit();
	ASSERT_TRUE(emptyCompletion.isOk());
	EXPECT_TRUE(emptyCompletion->isValid());
	EXPECT_FALSE(ring.isBatchOpen());
	ASSERT_TRUE(emptyCompletion->wait().isOk());

	oa::U8 data[8] = {};
	ASSERT_TRUE(ring.beginBatch().isOk());
	EXPECT_FALSE(ring.enqueueCopy(*staleSlice, dst, 0).isOk());
	EXPECT_FALSE(ring.upload(dst, dst.size - 4, data, sizeof(data)).isOk());
	auto completion = ring.submit();
	ASSERT_TRUE(completion.isOk());
	ASSERT_TRUE(completion->wait().isOk());

	ASSERT_TRUE(ring.close().isOk());
	oa::EngineResourceAccess::freeBuffer(*rt, dst);
}

TEST(VkStream, UploadRingPublishesWritesAcrossAliasHandles) {
	auto* rt = testEnginePtr();
	ASSERT_NE(rt, nullptr);

	constexpr oa::U64 kSize = 256;
	auto backingResult = oa::EngineAllocatorAccess::get(*rt).allocAliased(
		kSize, oa::MemoryPlacement::DeviceLocal);
	ASSERT_TRUE(backingResult.isOk());
	auto backing = std::move(*backingResult);
	EXPECT_TRUE(backing.supportsIndirectDispatch());
	auto siblingResult = oa::EngineAllocatorAccess::get(*rt).createAliasingBuffer(backing, kSize);
	ASSERT_TRUE(siblingResult.isOk());
	auto sibling = std::move(*siblingResult);
	EXPECT_TRUE(sibling.supportsIndirectDispatch());
	auto readbackResult = oa::EngineResourceAccess::allocBuffer(*rt, kSize);
	ASSERT_TRUE(readbackResult.isOk());
	auto readback = std::move(*readbackResult);

	oa::U8 source[kSize];
	for (oa::U32 index = 0; index < kSize; ++index) {
		source[index] = static_cast<oa::U8>((index * 29U + 7U) & 0xFFU);
	}
	auto ringResult = oa::UploadRing::create(*rt, oa::UploadRingConfig{
		.capacityBytes = 2 * 4096,
		.framesInFlight = 2,
		.alignment = 64,
	});
	ASSERT_TRUE(ringResult.isOk());
	auto ring = std::move(*ringResult);
	ASSERT_TRUE(ring.beginBatch().isOk());
	ASSERT_TRUE(ring.upload(backing, 0U, source, kSize).isOk());
	auto uploaded = ring.submit();
	ASSERT_TRUE(uploaded.isOk());
	ASSERT_TRUE(uploaded->isValid());

	oavk::Stream* consumer = oa::EngineSubmissionAccess::acquireStream(*rt);
	ASSERT_NE(consumer, nullptr);
	ASSERT_TRUE(consumer->begin(oa::EngineDeviceAccess::get(*rt)).isOk());
	consumer->recordCopyBuffer(sibling, readback, kSize);
	consumer->recordTransferWriteBarrier(readback, 0U, kSize);
	// Both submissions use the primary compute queue. The upload's global
	// alias barrier, not a semaphore memory dependency, publishes the write to
	// this sibling-handle read.
	ASSERT_TRUE(consumer->submit(*rt).isOk());
	ASSERT_TRUE(consumer->synchronize(oa::EngineDeviceAccess::get(*rt)).isOk());
	EXPECT_TRUE(uploaded->isComplete());
	oa::EngineSubmissionAccess::releaseStream(*rt, consumer);
	ASSERT_TRUE(oa::EngineAllocatorAccess::get(*rt).invalidateHostBuffer(readback, 0U, kSize));
	EXPECT_TRUE(oa::memEqual(source, readback.mappedPtr, kSize));

	ASSERT_TRUE(ring.close().isOk());
	oa::EngineAllocatorAccess::get(*rt).freeAlias(sibling);
	oa::EngineAllocatorAccess::get(*rt).freeAlias(backing);
	oa::EngineResourceAccess::freeBuffer(*rt, readback);
}

TEST(VkStream, UploadRingDestructionCancelsUnsubmittedBatch) {
	auto* rt = testEnginePtr();
	ASSERT_NE(rt, nullptr);

	constexpr oa::U64 kSize = 4096;
	auto dstResult = oa::EngineResourceAccess::allocBuffer(*rt, kSize);
	ASSERT_TRUE(dstResult.isOk());
	auto dst = std::move(*dstResult);
	ASSERT_NE(dst.mappedPtr, nullptr);
	oa::memset(dst.mappedPtr, 0xA5, kSize);

	oa::U8 source[kSize];
	oa::memset(source, 0x3C, kSize);
	{
		auto ringResult = oa::UploadRing::create(*rt, oa::UploadRingConfig{
			.capacityBytes = 2 * kSize,
			.framesInFlight = 2,
			.alignment = 64,
		});
		ASSERT_TRUE(ringResult.isOk());
		auto ring = std::move(*ringResult);
		ASSERT_TRUE(ring.beginBatch().isOk());
		ASSERT_TRUE(ring.upload(dst, 0, source, kSize).isOk());
		ASSERT_TRUE(ring.isBatchOpen());
	}

	ASSERT_TRUE(oa::EngineAllocatorAccess::get(*rt).invalidateHostBuffer(dst, 0, kSize));
	const auto* values = static_cast<const oa::U8*>(dst.mappedPtr);
	for (oa::U64 index = 0; index < kSize; ++index) {
		ASSERT_EQ(values[index], 0xA5);
	}
	oa::EngineResourceAccess::freeBuffer(*rt, dst);
}

TEST(VkStream, UploadRingDestructionRetiresSubmittedResourcesToEngine) {
	auto* rt = testEnginePtr();
	ASSERT_NE(rt, nullptr);

	constexpr oa::U64 kSize = 1024 * 1024;
	auto dstResult = oa::EngineResourceAccess::allocBuffer(*rt, kSize);
	ASSERT_TRUE(dstResult.isOk());
	auto dst = std::move(*dstResult);
	oa::Vec<oa::U8> source(kSize);
	for (oa::U64 index = 0; index < kSize; ++index) {
		source[index] = static_cast<oa::U8>((index * 17U + 11U) & 0xFFU);
	}

	oa::Event completion;
	{
		auto ringResult = oa::UploadRing::create(*rt, oa::UploadRingConfig{
			.capacityBytes = 2 * kSize,
			.framesInFlight = 2,
			.alignment = 256,
		});
		ASSERT_TRUE(ringResult.isOk());
		auto ring = std::move(*ringResult);
		ASSERT_TRUE(ring.beginBatch().isOk());
		ASSERT_TRUE(ring.upload(dst, 0, source.data(), kSize).isOk());
		auto submitted = ring.submit();
		ASSERT_TRUE(submitted.isOk());
		completion = *submitted;
		ASSERT_TRUE(completion.isValid());
	}

	// The facade is gone, but the engine-retired frame stays at the same address
	// and owns its timeline until explicit engine close.
	ASSERT_TRUE(completion.wait().isOk());
	ASSERT_TRUE(oa::EngineAllocatorAccess::get(*rt).invalidateHostBuffer(dst, 0, kSize));
	const auto* values = static_cast<const oa::U8*>(dst.mappedPtr);
	for (oa::U64 index = 0; index < kSize; index += 4093U) {
		EXPECT_EQ(values[index], source[index]);
	}
	oa::EngineResourceAccess::freeBuffer(*rt, dst);
}

// ─── Thread-Safe pipeline access ──────────────────────────────────────────────

TEST(VkStream, ConcurrentPipelineRead) {
	auto* rt = testEnginePtr();
	ASSERT_NE(rt, nullptr);

	constexpr oa::I32 kThreads = 4;
	constexpr oa::I32 kReads = 100;
	std::atomic<oa::I32> successCount{0};

	auto reader = [&]() {
		for (oa::I32 i = 0; i < kReads; ++i) {
			try {
				auto& pipe = oa::EnginePipelineAccess::get(*rt).getPipeline("Scale", 0U);
				if (pipe.pipeline != nullptr) {
					successCount.fetch_add(1, std::memory_order_relaxed);
				}
			} catch (...) {}
		}
	};

	oa::Vec<std::thread> threads;
	for (oa::I32 i = 0; i < kThreads; ++i) {
		threads.emplaceBack(reader);
	}
	for (auto& t : threads) t.join();

	EXPECT_GT(successCount.load(), 0);
}

// ─── SIMD Smoke Tests ─────────────────────────────────────────────────────────

TEST(Simd, Memzero) {
	oa::U8 buf[256];
	for (auto& b : buf) b = 0xFF;
	oa::memzero(buf, 256);
	for (auto& b : buf) {
		EXPECT_EQ(b, 0);
	}
}

TEST(Simd, MemEqual) {
	oa::U8 a[128], b[128];
	for (oa::I32 i = 0; i < 128; ++i) {
		a[i] = static_cast<oa::U8>(i);
		b[i] = static_cast<oa::U8>(i);
	}
	EXPECT_TRUE(oa::memEqual(a, b, 128));
	b[64] = 0xFF;
	EXPECT_FALSE(oa::memEqual(a, b, 128));
}

TEST(Simd, DotProduct) {
	constexpr oa::I64 N = 256;
	oa::F32 a[N], b[N];
	oa::F32 expected = 0.0f;
	for (oa::I64 i = 0; i < N; ++i) {
		a[i] = static_cast<oa::F32>(i) * 0.01f;
		b[i] = static_cast<oa::F32>(N - i) * 0.01f;
		expected += a[i] * b[i];
	}
	oa::F32 result = oa::Simd::dotF32(a, b, N);
	EXPECT_NEAR(result, expected, 0.1f);
}

TEST(Simd, scale) {
	constexpr oa::I64 N = 128;
	oa::F32 data[N];
	for (oa::I64 i = 0; i < N; ++i) data[i] = static_cast<oa::F32>(i);
	oa::Simd::scaleF32(data, 2.0f, N);
	for (oa::I64 i = 0; i < N; ++i) {
		EXPECT_FLOAT_EQ(data[i], static_cast<oa::F32>(i) * 2.0f);
	}
}

TEST(Simd, Add) {
	constexpr oa::I64 N = 128;
	oa::F32 a[N], b[N];
	for (oa::I64 i = 0; i < N; ++i) {
		a[i] = static_cast<oa::F32>(i);
		b[i] = 1.0f;
	}
	oa::Simd::addF32(a, b, N);
	for (oa::I64 i = 0; i < N; ++i) {
		EXPECT_FLOAT_EQ(a[i], static_cast<oa::F32>(i) + 1.0f);
	}
}
