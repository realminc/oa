#include <gtest/gtest.h>
#include <Oa/Oa.h>
#include <Oa/Runtime/Collective.h>
#include <Oa/Runtime/Engine.h>
#include <Oa/Runtime/ExternalMemory.h>
#include <Oa/Runtime/Topology.h>
#include <Oa/Core/Memory.h>

#include <cmath>
#include <numeric>
#include <vector>

// ─── Helpers ──────────────────────────────────────────────────────────────

static OaVkBuffer MakeHostBuffer(OaComputeEngine& InRt, OaU64 InSize) {
	auto result = InRt.AllocBuffer(InSize);
	EXPECT_TRUE(result.IsOk()) << "AllocBuffer failed";
	return std::move(result.GetValue());
}

static void FillF32(OaVkBuffer& InBuf, OaF32 InValue, OaU64 InCount) {
	auto* data = static_cast<OaF32*>(InBuf.MappedPtr);
	for (OaU64 i = 0; i < InCount; ++i) data[i] = InValue;
}

static void FillSequence(OaVkBuffer& InBuf, OaF32 InStart, OaU64 InCount) {
	auto* data = static_cast<OaF32*>(InBuf.MappedPtr);
	for (OaU64 i = 0; i < InCount; ++i) data[i] = InStart + static_cast<OaF32>(i);
}

static OaF32 ReadF32(const OaVkBuffer& InBuf, OaU64 InIdx) {
	return static_cast<const OaF32*>(InBuf.MappedPtr)[InIdx];
}

// ─── Single-Device No-Op Tests ──────────────────────────────────────────

TEST(Collective, AllReduceSingleDeviceNoOp) {
	auto* rt = OaComputeEngine::GetGlobal();
	ASSERT_NE(rt, nullptr);

	OaU64 count = 64;
	OaU64 size = count * sizeof(OaF32);
	auto buf = MakeHostBuffer(*rt, size);
	FillF32(buf, 1.0f, count);

	OaVkBuffer bufs[] = {buf};
	OaSpan<OaVkBuffer> span(bufs, 1);
	auto status = OaCollective::AllReduce(*rt, span, OaReduceOp::Sum);
	EXPECT_TRUE(status.IsOk());

	for (OaU64 i = 0; i < count; ++i) {
		EXPECT_FLOAT_EQ(ReadF32(buf, i), 1.0f);
	}

	rt->FreeBuffer(buf);
}

TEST(Collective, BroadcastSingleDeviceNoOp) {
	auto* rt = OaComputeEngine::GetGlobal();
	ASSERT_NE(rt, nullptr);

	OaU64 count = 32;
	OaU64 size = count * sizeof(OaF32);
	auto buf = MakeHostBuffer(*rt, size);
	FillF32(buf, 42.0f, count);

	OaVkBuffer bufs[] = {buf};
	OaSpan<OaVkBuffer> span(bufs, 1);
	auto status = OaCollective::Broadcast(*rt, span, 0);
	EXPECT_TRUE(status.IsOk());

	rt->FreeBuffer(buf);
}

// ─── Two-Buffer AllReduce (simulates 2 devices) ────────────────────────

TEST(Collective, AllReduceTwoBuffersSum) {
	auto* rt = OaComputeEngine::GetGlobal();
	ASSERT_NE(rt, nullptr);

	OaU64 count = 256;
	OaU64 size = count * sizeof(OaF32);
	auto bufA = MakeHostBuffer(*rt, size);
	auto bufB = MakeHostBuffer(*rt, size);

	FillF32(bufA, 1.0f, count);
	FillF32(bufB, 2.0f, count);

	OaVkBuffer bufs[] = {bufA, bufB};
	OaSpan<OaVkBuffer> span(bufs, 2);
	auto status = OaCollective::AllReduce(*rt, span, OaReduceOp::Sum);
	EXPECT_TRUE(status.IsOk());

	for (OaU64 i = 0; i < count; ++i) {
		EXPECT_FLOAT_EQ(ReadF32(bufs[0], i), 3.0f) << "bufA[" << i << "]";
		EXPECT_FLOAT_EQ(ReadF32(bufs[1], i), 3.0f) << "bufB[" << i << "]";
	}

	rt->FreeBuffer(bufA);
	rt->FreeBuffer(bufB);
}

TEST(Collective, AllReduceTwoBuffersMax) {
	auto* rt = OaComputeEngine::GetGlobal();
	ASSERT_NE(rt, nullptr);

	OaU64 count = 128;
	OaU64 size = count * sizeof(OaF32);
	auto bufA = MakeHostBuffer(*rt, size);
	auto bufB = MakeHostBuffer(*rt, size);

	for (OaU64 i = 0; i < count; ++i) {
		static_cast<OaF32*>(bufA.MappedPtr)[i] = static_cast<OaF32>(i);
		static_cast<OaF32*>(bufB.MappedPtr)[i] = static_cast<OaF32>(count - i);
	}

	OaVkBuffer bufs[] = {bufA, bufB};
	OaSpan<OaVkBuffer> span(bufs, 2);
	auto status = OaCollective::AllReduce(*rt, span, OaReduceOp::Max);
	EXPECT_TRUE(status.IsOk());

	for (OaU64 i = 0; i < count; ++i) {
		OaF32 expected = std::fmax(static_cast<OaF32>(i), static_cast<OaF32>(count - i));
		EXPECT_FLOAT_EQ(ReadF32(bufs[0], i), expected) << "bufA[" << i << "]";
		EXPECT_FLOAT_EQ(ReadF32(bufs[1], i), expected) << "bufB[" << i << "]";
	}

	rt->FreeBuffer(bufA);
	rt->FreeBuffer(bufB);
}

// ─── Broadcast ──────────────────────────────────────────────────────────

TEST(Collective, BroadcastTwoBuffers) {
	auto* rt = OaComputeEngine::GetGlobal();
	ASSERT_NE(rt, nullptr);

	OaU64 count = 64;
	OaU64 size = count * sizeof(OaF32);
	auto bufA = MakeHostBuffer(*rt, size);
	auto bufB = MakeHostBuffer(*rt, size);

	FillSequence(bufA, 0.0f, count);
	FillF32(bufB, 0.0f, count);

	OaVkBuffer bufs[] = {bufA, bufB};
	OaSpan<OaVkBuffer> span(bufs, 2);
	auto status = OaCollective::Broadcast(*rt, span, 0);
	EXPECT_TRUE(status.IsOk());

	for (OaU64 i = 0; i < count; ++i) {
		EXPECT_FLOAT_EQ(ReadF32(bufs[1], i), static_cast<OaF32>(i));
	}

	rt->FreeBuffer(bufA);
	rt->FreeBuffer(bufB);
}

// ─── AllGather ──────────────────────────────────────────────────────────

TEST(Collective, AllGatherTwoBuffers) {
	auto* rt = OaComputeEngine::GetGlobal();
	ASSERT_NE(rt, nullptr);

	OaU64 partialCount = 32;
	OaU64 partialSize = partialCount * sizeof(OaF32);
	OaU64 fullSize = partialSize * 2;

	auto partA = MakeHostBuffer(*rt, partialSize);
	auto partB = MakeHostBuffer(*rt, partialSize);
	auto fullA = MakeHostBuffer(*rt, fullSize);
	auto fullB = MakeHostBuffer(*rt, fullSize);

	FillF32(partA, 1.0f, partialCount);
	FillF32(partB, 2.0f, partialCount);

	const OaVkBuffer partials[] = {partA, partB};
	OaVkBuffer fulls[] = {fullA, fullB};
	OaSpan<const OaVkBuffer> partialsSpan(partials, 2);
	OaSpan<OaVkBuffer> fullsSpan(fulls, 2);

	auto status = OaCollective::AllGather(*rt, partialsSpan, fullsSpan);
	EXPECT_TRUE(status.IsOk());

	for (OaU64 i = 0; i < partialCount; ++i) {
		EXPECT_FLOAT_EQ(ReadF32(fulls[0], i), 1.0f);
		EXPECT_FLOAT_EQ(ReadF32(fulls[0], partialCount + i), 2.0f);
		EXPECT_FLOAT_EQ(ReadF32(fulls[1], i), 1.0f);
		EXPECT_FLOAT_EQ(ReadF32(fulls[1], partialCount + i), 2.0f);
	}

	rt->FreeBuffer(partA);
	rt->FreeBuffer(partB);
	rt->FreeBuffer(fullA);
	rt->FreeBuffer(fullB);
}

// ─── Scatter ────────────────────────────────────────────────────────────

TEST(Collective, ScatterTwoBuffers) {
	auto* rt = OaComputeEngine::GetGlobal();
	ASSERT_NE(rt, nullptr);

	OaU64 fullCount = 64;
	OaU64 fullSize = fullCount * sizeof(OaF32);
	OaU64 partialSize = fullSize / 2;
	OaU64 partialCount = fullCount / 2;

	auto full = MakeHostBuffer(*rt, fullSize);
	auto partA = MakeHostBuffer(*rt, partialSize);
	auto partB = MakeHostBuffer(*rt, partialSize);

	FillSequence(full, 0.0f, fullCount);

	OaVkBuffer partials[] = {partA, partB};
	OaSpan<OaVkBuffer> partialsSpan(partials, 2);

	auto status = OaCollective::Scatter(*rt, full, partialsSpan);
	EXPECT_TRUE(status.IsOk());

	for (OaU64 i = 0; i < partialCount; ++i) {
		EXPECT_FLOAT_EQ(ReadF32(partials[0], i), static_cast<OaF32>(i));
		EXPECT_FLOAT_EQ(ReadF32(partials[1], i), static_cast<OaF32>(partialCount + i));
	}

	rt->FreeBuffer(full);
	rt->FreeBuffer(partA);
	rt->FreeBuffer(partB);
}

// ─── ReduceScatter ──────────────────────────────────────────────────────

TEST(Collective, ReduceScatterTwoBuffers) {
	auto* rt = OaComputeEngine::GetGlobal();
	ASSERT_NE(rt, nullptr);

	OaU64 count = 128;
	OaU64 size = count * sizeof(OaF32);
	auto bufA = MakeHostBuffer(*rt, size);
	auto bufB = MakeHostBuffer(*rt, size);

	FillF32(bufA, 1.0f, count);
	FillF32(bufB, 2.0f, count);

	OaVkBuffer bufs[] = {bufA, bufB};
	OaSpan<OaVkBuffer> span(bufs, 2);
	auto status = OaCollective::ReduceScatter(*rt, span, OaReduceOp::Sum);
	EXPECT_TRUE(status.IsOk());

	OaU64 chunkCount = count / 2;
	for (OaU64 i = 0; i < chunkCount; ++i) {
		EXPECT_FLOAT_EQ(ReadF32(bufs[0], i), 3.0f) << "bufA chunk[" << i << "]";
	}

	rt->FreeBuffer(bufA);
	rt->FreeBuffer(bufB);
}

// ─── Async AllReduce ────────────────────────────────────────────────────

TEST(Collective, AllReduceAsyncTwoBuffers) {
	auto* rt = OaComputeEngine::GetGlobal();
	ASSERT_NE(rt, nullptr);

	OaU64 count = 256;
	OaU64 size = count * sizeof(OaF32);
	auto bufA = MakeHostBuffer(*rt, size);
	auto bufB = MakeHostBuffer(*rt, size);

	FillF32(bufA, 3.0f, count);
	FillF32(bufB, 7.0f, count);

	OaVkBuffer bufs[] = {bufA, bufB};
	OaSpan<OaVkBuffer> span(bufs, 2);

	auto ticketResult = OaCollective::AllReduceAsync(*rt, span, OaReduceOp::Sum);
	EXPECT_TRUE(ticketResult.IsOk());

	auto ticket = std::move(ticketResult.GetValue());
	if (ticket.Semaphore.Semaphore) {
		auto waitStatus = ticket.Wait(rt->Device);
		EXPECT_TRUE(waitStatus.IsOk());
	}

	for (OaU64 i = 0; i < count; ++i) {
		EXPECT_FLOAT_EQ(ReadF32(bufs[0], i), 10.0f) << "bufA[" << i << "]";
		EXPECT_FLOAT_EQ(ReadF32(bufs[1], i), 10.0f) << "bufB[" << i << "]";
	}

	rt->FreeBuffer(bufA);
	rt->FreeBuffer(bufB);
}

// ─── Large Buffer Test ──────────────────────────────────────────────────

TEST(Collective, AllReduceLargeBuffer) {
	auto* rt = OaComputeEngine::GetGlobal();
	ASSERT_NE(rt, nullptr);

	OaU64 count = 1024 * 1024; // 4 MB per buffer
	OaU64 size = count * sizeof(OaF32);
	auto bufA = MakeHostBuffer(*rt, size);
	auto bufB = MakeHostBuffer(*rt, size);

	FillF32(bufA, 1.0f, count);
	FillF32(bufB, 1.0f, count);

	OaVkBuffer bufs[] = {bufA, bufB};
	OaSpan<OaVkBuffer> span(bufs, 2);
	auto status = OaCollective::AllReduce(*rt, span, OaReduceOp::Sum);
	EXPECT_TRUE(status.IsOk());

	// Spot-check a few values
	EXPECT_FLOAT_EQ(ReadF32(bufs[0], 0), 2.0f);
	EXPECT_FLOAT_EQ(ReadF32(bufs[0], count / 2), 2.0f);
	EXPECT_FLOAT_EQ(ReadF32(bufs[0], count - 1), 2.0f);
	EXPECT_FLOAT_EQ(ReadF32(bufs[1], 0), 2.0f);

	rt->FreeBuffer(bufA);
	rt->FreeBuffer(bufB);
}

// ─── Multi-device mesh (second VkDevice) ────────────────────────────────

TEST(Collective, MultiDeviceAllReduceTwoNodes) {
	auto* fixtureRt = OaComputeEngine::GetGlobal();
	ASSERT_NE(fixtureRt, nullptr);

	OaEngineConfig cfg{};
	cfg.EnableMultiDevice = true;
	cfg.RegisterAsGlobal = false;
	cfg.AppName = "OaTestMeshAllReduce";
	cfg.Precision = fixtureRt->GetPrecision();

	auto meshRes = OaComputeEngine::Create(cfg);
	if (!meshRes) {
		GTEST_SKIP() << "multi-device OaComputeEngine::Create failed";
	}
	OaComputeEngine& meshRt = *meshRes.GetValue();
	if (meshRt.DeviceCount() < 2) {
		meshRt.Destroy();
		GTEST_SKIP() << "fewer than two Vulkan physical devices";
	}

	const OaU64 count = 128;
	const OaU64 size = count * sizeof(OaF32);

	auto buf0Res = meshRt.AllocBufferOnNode(0, size);
	auto buf1Res = meshRt.AllocBufferOnNode(1, size);
	ASSERT_TRUE(buf0Res.IsOk());
	ASSERT_TRUE(buf1Res.IsOk());
	OaVkBuffer buf0 = std::move(*buf0Res);
	OaVkBuffer buf1 = std::move(*buf1Res);

	FillF32(buf0, 1.0f, count);
	FillF32(buf1, 4.0f, count);

	OaVkBuffer bufs[] = {buf0, buf1};
	OaSpan<OaVkBuffer> span(bufs, 2);
	auto st = OaCollective::AllReduce(meshRt, span, OaReduceOp::Sum);
	EXPECT_TRUE(st.IsOk());

	for (OaU64 i = 0; i < count; ++i) {
		EXPECT_FLOAT_EQ(ReadF32(buf0, i), 5.0f) << "node0[" << i << "]";
		EXPECT_FLOAT_EQ(ReadF32(buf1, i), 5.0f) << "node1[" << i << "]";
	}

	meshRt.FreeBufferOnNode(buf0);
	meshRt.FreeBufferOnNode(buf1);
	meshRt.Destroy();
}

#if defined(__linux__)
TEST(Collective, ExportImportFdSameVendorTwoNodes) {
	auto* fixtureRt = OaComputeEngine::GetGlobal();
	ASSERT_NE(fixtureRt, nullptr);

	OaEngineConfig cfg{};
	cfg.EnableMultiDevice = true;
	cfg.RegisterAsGlobal = false;
	cfg.AppName = "OaTestDmaBufCollective";
	cfg.Precision = fixtureRt->GetPrecision();

	auto meshRes = OaComputeEngine::Create(cfg);
	if (!meshRes) {
		GTEST_SKIP() << "multi-device OaComputeEngine::Create failed";
	}
	OaComputeEngine& meshRt = *meshRes.GetValue();
	if (meshRt.DeviceCount() < 2) {
		meshRt.Destroy();
		GTEST_SKIP() << "fewer than two Vulkan physical devices";
	}

	OaDeviceNode* n0 = meshRt.GetNode(0);
	OaDeviceNode* n1 = meshRt.GetNode(1);
	ASSERT_NE(n0, nullptr);
	ASSERT_NE(n1, nullptr);

	if (!OaCanUseDmaBuf(n0->Device, n1->Device)) {
		meshRt.Destroy();
		GTEST_SKIP() << "DMA-BUF requires same vendor and VK_KHR_external_memory_fd on both devices";
	}

	const OaU64 count = 64;
	const OaU64 size = count * sizeof(OaF32);

	auto buf0Res = meshRt.AllocBufferOnNode(0, size);
	if (!buf0Res.IsOk()) {
		meshRt.Destroy();
		GTEST_SKIP() << "AllocBufferOnNode(0) failed";
	}
	OaVkBuffer buf0 = std::move(*buf0Res);
	FillF32(buf0, 2.5f, count);

	auto extRes = OaExportBufferFd(meshRt.Device, meshRt.Allocator, buf0, 0);
	if (!extRes.IsOk()) {
		meshRt.FreeBufferOnNode(buf0);
		meshRt.Destroy();
		GTEST_SKIP() << "export fd failed: " << extRes.GetStatus().ToString();
	}
	OaExternalBuffer ext = std::move(extRes.GetValue());

	auto impRes = OaImportBufferFd(n1->Device, n1->Allocator, ext);
	if (!impRes.IsOk()) {
		ext.Close();
		meshRt.FreeBufferOnNode(buf0);
		meshRt.Destroy();
		GTEST_SKIP() << "import fd failed: " << impRes.GetStatus().ToString();
	}
	OaVkBuffer imported = std::move(impRes.GetValue());
	EXPECT_NE(imported.Buffer, nullptr);
	EXPECT_EQ(imported.Size, size);

	n1->Allocator.FreeImported(n1->Device, imported);
	meshRt.FreeBufferOnNode(buf0);
	meshRt.Destroy();
}
#endif

// ─── DMA-BUF Capability Check ──────────────────────────────────────────

TEST(Collective, DmaBufCapabilityCheck) {
	auto* rt = OaComputeEngine::GetGlobal();
	ASSERT_NE(rt, nullptr);

	bool hasDmaBuf = rt->Device.HasExternalMemoryFd;
	OA_LOG_INFO(OaLogComponent::Core, "HasExternalMemoryFd: %s", hasDmaBuf ? "yes" : "no");
}
