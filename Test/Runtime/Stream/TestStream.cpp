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

// ─── Stream Lifecycle ─────────────────────────────────────────────────────────

TEST(VkStream, CreateDestroy) {
	auto* rt = OaComputeEngine::GetGlobal();
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
	auto* rt = OaComputeEngine::GetGlobal();
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
	auto* rt = OaComputeEngine::GetGlobal();
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
	auto* rt = OaComputeEngine::GetGlobal();
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
	auto* rt = OaComputeEngine::GetGlobal();
	ASSERT_NE(rt, nullptr);

	auto fRes = OaVkFence::Create(rt->Device, true);
	ASSERT_TRUE(fRes.IsOk());
	auto fence = std::move(*fRes);
	EXPECT_TRUE(fence.IsSignaled(rt->Device));

	fence.Reset(rt->Device);
	EXPECT_FALSE(fence.IsSignaled(rt->Device));

	fence.Destroy(rt->Device);
}

// ─── Engine Stream Pool ───────────────────────────────────────────────────────

TEST(VkStream, PoolAcquireRelease) {
	auto* rt = OaComputeEngine::GetGlobal();
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
	auto* rt = OaComputeEngine::GetGlobal();
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
	auto* rt = OaComputeEngine::GetGlobal();
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
	auto* rt = OaComputeEngine::GetGlobal();
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
	auto* rt = OaComputeEngine::GetGlobal();
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
	auto* rt = OaComputeEngine::GetGlobal();
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

	auto status = rt->CopyBufferAsync(src, dst, kSize);
	ASSERT_TRUE(status.IsOk());
	ASSERT_TRUE(rt->WaitTransfer().IsOk());

	OaU8* dstData = static_cast<OaU8*>(dst.MappedPtr);
	for (OaU32 i = 0; i < kSize; ++i) {
		EXPECT_EQ(dstData[i], static_cast<OaU8>(i & 0xFF)) << "mismatch at " << i;
	}

	rt->DeregisterBuffer(src);
	rt->Allocator.Free(src);
	rt->DeregisterBuffer(dst);
	rt->Allocator.Free(dst);
}

// ─── Concurrent Pool Acquire/Release ──────────────────────────────────────────

TEST(VkStream, ConcurrentPoolAccess) {
	auto* rt = OaComputeEngine::GetGlobal();
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
	auto* rt = OaComputeEngine::GetGlobal();
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

	OaVkStream* stream = rt->AcquireStream();
	ASSERT_NE(stream, nullptr);
	ASSERT_TRUE(stream->Begin(rt->Device).IsOk());
	stream->RecordCopyBuffer(src, dst, kSize);
	stream->RecordBufferBarrier();
	ASSERT_TRUE(stream->SubmitAndWait(*rt).IsOk());
	rt->ReleaseStream(stream);

	OaU8* out = static_cast<OaU8*>(dst.MappedPtr);
	for (OaU32 i = 0; i < kSize; ++i) {
		EXPECT_EQ(out[i], 42);
	}

	rt->DeregisterBuffer(src);
	rt->Allocator.Free(src);
	rt->DeregisterBuffer(dst);
	rt->Allocator.Free(dst);
}

// ─── Persistent Mapped Upload Ring ───────────────────────────────────────────

TEST(VkStream, UploadRingBatchesRegionsAndRecyclesFrames) {
	auto* rt = OaComputeEngine::GetGlobal();
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
	auto* rt = OaComputeEngine::GetGlobal();
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

TEST(VkStream, PrefetchCompatibilityPathTransfersToDeviceBuffer) {
	auto* rt = OaComputeEngine::GetGlobal();
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

	auto readbackResult = rt->AllocBuffer(kSize);
	ASSERT_TRUE(readbackResult.IsOk());
	auto readback = std::move(*readbackResult);
	ASSERT_TRUE(rt->CopyBufferAsync(*deviceResult, readback, kSize).IsOk());
	ASSERT_TRUE(rt->WaitTransfer().IsOk());
	ASSERT_TRUE(rt->Allocator.InvalidateHostBuffer(readback, 0, kSize));
	EXPECT_TRUE(OaMemEqual(source, readback.MappedPtr, kSize));

	pipeline.Destroy(*rt);
	rt->FreeBuffer(readback);
}

// ─── Thread-Safe Pipeline Access ──────────────────────────────────────────────

TEST(VkStream, ConcurrentPipelineRead) {
	auto* rt = OaComputeEngine::GetGlobal();
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
