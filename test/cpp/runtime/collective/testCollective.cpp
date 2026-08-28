#include "../../oaTest.h"

#include <gtest/gtest.h>
#include <oa/oa.h>
#include "oa/runtime/collective/hostCollectiveOracle.h"
#include <oa/runtime/engine.h>
#include <oa/core/std/memory.h>

#include <cmath>
#include <numeric>
#include <vector>

// ─── Helpers ──────────────────────────────────────────────────────────────

static oavk::Buffer makeHostBuffer(oa::Engine& inRt, oa::U64 inSize) {
	auto result = oa::EngineResourceAccess::allocBuffer(inRt, inSize);
	EXPECT_TRUE(result.isOk()) << "allocBuffer failed";
	return std::move(result.getValue());
}

static void fillF32(oavk::Buffer& inBuf, oa::F32 inValue, oa::U64 inCount) {
	auto* data = static_cast<oa::F32*>(inBuf.mappedPtr);
	for (oa::U64 i = 0; i < inCount; ++i) data[i] = inValue;
}

static void fillSequence(oavk::Buffer& inBuf, oa::F32 inStart, oa::U64 inCount) {
	auto* data = static_cast<oa::F32*>(inBuf.mappedPtr);
	for (oa::U64 i = 0; i < inCount; ++i) data[i] = inStart + static_cast<oa::F32>(i);
}

static oa::F32 readF32(const oavk::Buffer& inBuf, oa::U64 inIdx) {
	return static_cast<const oa::F32*>(inBuf.mappedPtr)[inIdx];
}

static oavk::Buffer makeHostView(oa::F32* inData, oa::U64 inCount) {
	oavk::Buffer view;
	view.size = inCount * sizeof(oa::F32);
	view.capacity = view.size;
	view.mappedPtr = inData;
	return view;
}

// ─── Single-Device No-op Tests ──────────────────────────────────────────

TEST(Collective, AllReduceSingleDeviceNoOp) {
	auto* rt = testEnginePtr();
	ASSERT_NE(rt, nullptr);

	oa::U64 count = 64;
	oa::U64 size = count * sizeof(oa::F32);
	auto buf = makeHostBuffer(*rt, size);
	fillF32(buf, 1.0f, count);

	oavk::Buffer bufs[] = {buf};
	oa::Span<oavk::Buffer> span(bufs, 1);
	auto status = oa::HostCollectiveOracle::allReduce(span, oa::HostReduceOp::Sum);
	EXPECT_TRUE(status.isOk());

	for (oa::U64 i = 0; i < count; ++i) {
		EXPECT_FLOAT_EQ(readF32(buf, i), 1.0f);
	}

	oa::EngineResourceAccess::freeBuffer(*rt, buf);
}

TEST(Collective, BroadcastSingleDeviceNoOp) {
	auto* rt = testEnginePtr();
	ASSERT_NE(rt, nullptr);

	oa::U64 count = 32;
	oa::U64 size = count * sizeof(oa::F32);
	auto buf = makeHostBuffer(*rt, size);
	fillF32(buf, 42.0f, count);

	oavk::Buffer bufs[] = {buf};
	oa::Span<oavk::Buffer> span(bufs, 1);
	auto status = oa::HostCollectiveOracle::broadcast(span, 0);
	EXPECT_TRUE(status.isOk());

	oa::EngineResourceAccess::freeBuffer(*rt, buf);
}

// ─── Two-Buffer allReduce (simulates 2 devices) ────────────────────────

TEST(Collective, AllReduceTwoBuffersSum) {
	auto* rt = testEnginePtr();
	ASSERT_NE(rt, nullptr);

	oa::U64 count = 256;
	oa::U64 size = count * sizeof(oa::F32);
	auto bufA = makeHostBuffer(*rt, size);
	auto bufB = makeHostBuffer(*rt, size);

	fillF32(bufA, 1.0f, count);
	fillF32(bufB, 2.0f, count);

	oavk::Buffer bufs[] = {bufA, bufB};
	oa::Span<oavk::Buffer> span(bufs, 2);
	auto status = oa::HostCollectiveOracle::allReduce(span, oa::HostReduceOp::Sum);
	EXPECT_TRUE(status.isOk());

	for (oa::U64 i = 0; i < count; ++i) {
		EXPECT_FLOAT_EQ(readF32(bufs[0], i), 3.0f) << "bufA[" << i << "]";
		EXPECT_FLOAT_EQ(readF32(bufs[1], i), 3.0f) << "bufB[" << i << "]";
	}

	oa::EngineResourceAccess::freeBuffer(*rt, bufA);
	oa::EngineResourceAccess::freeBuffer(*rt, bufB);
}

TEST(Collective, AllReduceTwoBuffersMax) {
	auto* rt = testEnginePtr();
	ASSERT_NE(rt, nullptr);

	oa::U64 count = 128;
	oa::U64 size = count * sizeof(oa::F32);
	auto bufA = makeHostBuffer(*rt, size);
	auto bufB = makeHostBuffer(*rt, size);

	for (oa::U64 i = 0; i < count; ++i) {
		static_cast<oa::F32*>(bufA.mappedPtr)[i] = static_cast<oa::F32>(i);
		static_cast<oa::F32*>(bufB.mappedPtr)[i] = static_cast<oa::F32>(count - i);
	}

	oavk::Buffer bufs[] = {bufA, bufB};
	oa::Span<oavk::Buffer> span(bufs, 2);
	auto status = oa::HostCollectiveOracle::allReduce(span, oa::HostReduceOp::Max);
	EXPECT_TRUE(status.isOk());

	for (oa::U64 i = 0; i < count; ++i) {
		oa::F32 expected = std::fmax(static_cast<oa::F32>(i), static_cast<oa::F32>(count - i));
		EXPECT_FLOAT_EQ(readF32(bufs[0], i), expected) << "bufA[" << i << "]";
		EXPECT_FLOAT_EQ(readF32(bufs[1], i), expected) << "bufB[" << i << "]";
	}

	oa::EngineResourceAccess::freeBuffer(*rt, bufA);
	oa::EngineResourceAccess::freeBuffer(*rt, bufB);
}

TEST(Collective, AllReduceThreeBuffersSum) {
	auto* rt = testEnginePtr();
	ASSERT_NE(rt, nullptr);

	constexpr oa::U64 count = 12;
	constexpr oa::U64 size = count * sizeof(oa::F32);
	auto bufA = makeHostBuffer(*rt, size);
	auto bufB = makeHostBuffer(*rt, size);
	auto bufC = makeHostBuffer(*rt, size);
	fillF32(bufA, 1.0F, count);
	fillF32(bufB, 2.0F, count);
	fillF32(bufC, 4.0F, count);

	oavk::Buffer bufs[] = {bufA, bufB, bufC};
	auto status = oa::HostCollectiveOracle::allReduce(
		oa::Span<oavk::Buffer>(bufs, 3), oa::HostReduceOp::Sum);
	ASSERT_TRUE(status.isOk()) << status.toString();
	for (const oavk::Buffer& buffer : bufs) {
		for (oa::U64 i = 0; i < count; ++i) {
			EXPECT_FLOAT_EQ(readF32(buffer, i), 7.0F);
		}
	}

	oa::EngineResourceAccess::freeBuffer(*rt, bufA);
	oa::EngineResourceAccess::freeBuffer(*rt, bufB);
	oa::EngineResourceAccess::freeBuffer(*rt, bufC);
}

// ─── Broadcast ──────────────────────────────────────────────────────────

TEST(Collective, BroadcastTwoBuffers) {
	auto* rt = testEnginePtr();
	ASSERT_NE(rt, nullptr);

	oa::U64 count = 64;
	oa::U64 size = count * sizeof(oa::F32);
	auto bufA = makeHostBuffer(*rt, size);
	auto bufB = makeHostBuffer(*rt, size);

	fillSequence(bufA, 0.0f, count);
	fillF32(bufB, 0.0f, count);

	oavk::Buffer bufs[] = {bufA, bufB};
	oa::Span<oavk::Buffer> span(bufs, 2);
	auto status = oa::HostCollectiveOracle::broadcast(span, 0);
	EXPECT_TRUE(status.isOk());

	for (oa::U64 i = 0; i < count; ++i) {
		EXPECT_FLOAT_EQ(readF32(bufs[1], i), static_cast<oa::F32>(i));
	}

	oa::EngineResourceAccess::freeBuffer(*rt, bufA);
	oa::EngineResourceAccess::freeBuffer(*rt, bufB);
}

// ─── AllGather ──────────────────────────────────────────────────────────

TEST(Collective, AllGatherTwoBuffers) {
	auto* rt = testEnginePtr();
	ASSERT_NE(rt, nullptr);

	oa::U64 partialCount = 32;
	oa::U64 partialSize = partialCount * sizeof(oa::F32);
	oa::U64 fullSize = partialSize * 2;

	auto partA = makeHostBuffer(*rt, partialSize);
	auto partB = makeHostBuffer(*rt, partialSize);
	auto fullA = makeHostBuffer(*rt, fullSize);
	auto fullB = makeHostBuffer(*rt, fullSize);

	fillF32(partA, 1.0f, partialCount);
	fillF32(partB, 2.0f, partialCount);

	const oavk::Buffer partials[] = {partA, partB};
	oavk::Buffer fulls[] = {fullA, fullB};
	oa::Span<const oavk::Buffer> partialsSpan(partials, 2);
	oa::Span<oavk::Buffer> fullsSpan(fulls, 2);

	auto status = oa::HostCollectiveOracle::allGather(partialsSpan, fullsSpan);
	EXPECT_TRUE(status.isOk());

	for (oa::U64 i = 0; i < partialCount; ++i) {
		EXPECT_FLOAT_EQ(readF32(fulls[0], i), 1.0f);
		EXPECT_FLOAT_EQ(readF32(fulls[0], partialCount + i), 2.0f);
		EXPECT_FLOAT_EQ(readF32(fulls[1], i), 1.0f);
		EXPECT_FLOAT_EQ(readF32(fulls[1], partialCount + i), 2.0f);
	}

	oa::EngineResourceAccess::freeBuffer(*rt, partA);
	oa::EngineResourceAccess::freeBuffer(*rt, partB);
	oa::EngineResourceAccess::freeBuffer(*rt, fullA);
	oa::EngineResourceAccess::freeBuffer(*rt, fullB);
}

// ─── Scatter ────────────────────────────────────────────────────────────

TEST(Collective, ScatterTwoBuffers) {
	auto* rt = testEnginePtr();
	ASSERT_NE(rt, nullptr);

	oa::U64 fullCount = 64;
	oa::U64 fullSize = fullCount * sizeof(oa::F32);
	oa::U64 partialSize = fullSize / 2;
	oa::U64 partialCount = fullCount / 2;

	auto full = makeHostBuffer(*rt, fullSize);
	auto partA = makeHostBuffer(*rt, partialSize);
	auto partB = makeHostBuffer(*rt, partialSize);

	fillSequence(full, 0.0f, fullCount);

	oavk::Buffer partials[] = {partA, partB};
	oa::Span<oavk::Buffer> partialsSpan(partials, 2);

	auto status = oa::HostCollectiveOracle::scatter(full, partialsSpan);
	EXPECT_TRUE(status.isOk());

	for (oa::U64 i = 0; i < partialCount; ++i) {
		EXPECT_FLOAT_EQ(readF32(partials[0], i), static_cast<oa::F32>(i));
		EXPECT_FLOAT_EQ(readF32(partials[1], i), static_cast<oa::F32>(partialCount + i));
	}

	oa::EngineResourceAccess::freeBuffer(*rt, full);
	oa::EngineResourceAccess::freeBuffer(*rt, partA);
	oa::EngineResourceAccess::freeBuffer(*rt, partB);
}

// ─── ReduceScatter ──────────────────────────────────────────────────────

TEST(Collective, ReduceScatterTwoBuffers) {
	auto* rt = testEnginePtr();
	ASSERT_NE(rt, nullptr);

	oa::U64 count = 128;
	oa::U64 size = count * sizeof(oa::F32);
	auto bufA = makeHostBuffer(*rt, size);
	auto bufB = makeHostBuffer(*rt, size);

	fillF32(bufA, 1.0f, count);
	fillF32(bufB, 2.0f, count);

	oavk::Buffer bufs[] = {bufA, bufB};
	oa::Span<oavk::Buffer> span(bufs, 2);
	auto status = oa::HostCollectiveOracle::reduceScatter(span, oa::HostReduceOp::Sum);
	EXPECT_TRUE(status.isOk());

	oa::U64 chunkCount = count / 2;
	for (oa::U64 i = 0; i < chunkCount; ++i) {
		EXPECT_FLOAT_EQ(readF32(bufs[0], i), 3.0f) << "bufA chunk[" << i << "]";
	}

	oa::EngineResourceAccess::freeBuffer(*rt, bufA);
	oa::EngineResourceAccess::freeBuffer(*rt, bufB);
}

TEST(Collective, ReduceScatterThreeBuffers) {
	auto* rt = testEnginePtr();
	ASSERT_NE(rt, nullptr);

	constexpr oa::U64 count = 12;
	constexpr oa::U64 chunkCount = count / 3;
	constexpr oa::U64 size = count * sizeof(oa::F32);
	auto bufA = makeHostBuffer(*rt, size);
	auto bufB = makeHostBuffer(*rt, size);
	auto bufC = makeHostBuffer(*rt, size);
	fillF32(bufA, 1.0F, count);
	fillF32(bufB, 2.0F, count);
	fillF32(bufC, 4.0F, count);

	oavk::Buffer bufs[] = {bufA, bufB, bufC};
	auto status = oa::HostCollectiveOracle::reduceScatter(
		oa::Span<oavk::Buffer>(bufs, 3), oa::HostReduceOp::Sum);
	ASSERT_TRUE(status.isOk()) << status.toString();
	for (const oavk::Buffer& buffer : bufs) {
		for (oa::U64 i = 0; i < chunkCount; ++i) {
			EXPECT_FLOAT_EQ(readF32(buffer, i), 7.0F);
		}
	}

	oa::EngineResourceAccess::freeBuffer(*rt, bufA);
	oa::EngineResourceAccess::freeBuffer(*rt, bufB);
	oa::EngineResourceAccess::freeBuffer(*rt, bufC);
}

TEST(Collective, RejectsMalformedContractsBeforeMutation) {
	auto* rt = testEnginePtr();
	ASSERT_NE(rt, nullptr);

	oa::Span<oavk::Buffer> empty;
	EXPECT_FALSE(oa::HostCollectiveOracle::allReduce(
		empty, oa::HostReduceOp::Sum).isOk());

	auto src = makeHostBuffer(*rt, 4 * sizeof(oa::F32));
	auto shortDst = makeHostBuffer(*rt, 2 * sizeof(oa::F32));
	fillF32(src, 3.0F, 4);
	fillF32(shortDst, 9.0F, 2);
	oavk::Buffer broadcastBuffers[] = {src, shortDst};
	EXPECT_FALSE(oa::HostCollectiveOracle::broadcast(
		oa::Span<oavk::Buffer>(broadcastBuffers, 2), 0).isOk());
	EXPECT_FLOAT_EQ(readF32(shortDst, 0), 9.0F);
	EXPECT_FLOAT_EQ(readF32(shortDst, 1), 9.0F);

	oavk::Buffer reduceBuffers[] = {src, src};
	EXPECT_FALSE(oa::HostCollectiveOracle::allReduce(
		oa::Span<oavk::Buffer>(reduceBuffers, 2),
		static_cast<oa::HostReduceOp>(255)).isOk());

	auto oddFull = makeHostBuffer(*rt, 10);
	auto partA = makeHostBuffer(*rt, 4);
	auto partB = makeHostBuffer(*rt, 4);
	auto partC = makeHostBuffer(*rt, 4);
	oavk::Buffer scatterOutputs[] = {partA, partB, partC};
	EXPECT_FALSE(oa::HostCollectiveOracle::scatter(
		oddFull,
		oa::Span<oavk::Buffer>(scatterOutputs, 3)).isOk());

	const oavk::Buffer gatherInputs[] = {src, src};
	oavk::Buffer gatherOutputs[] = {shortDst, shortDst};
	EXPECT_FALSE(oa::HostCollectiveOracle::allGather(
		oa::Span<const oavk::Buffer>(gatherInputs, 2),
		oa::Span<oavk::Buffer>(gatherOutputs, 2)).isOk());

	oa::EngineResourceAccess::freeBuffer(*rt, src);
	oa::EngineResourceAccess::freeBuffer(*rt, shortDst);
	oa::EngineResourceAccess::freeBuffer(*rt, oddFull);
	oa::EngineResourceAccess::freeBuffer(*rt, partA);
	oa::EngineResourceAccess::freeBuffer(*rt, partB);
	oa::EngineResourceAccess::freeBuffer(*rt, partC);
}

TEST(Collective, StagesOverlappingHostViews) {
	oa::F32 broadcastStorage[] = {1.0F, 2.0F, 3.0F, 4.0F, 0.0F};
	oavk::Buffer broadcastBuffers[] = {
		makeHostView(broadcastStorage, 4),
		makeHostView(broadcastStorage + 1, 4),
	};
	ASSERT_TRUE(oa::HostCollectiveOracle::broadcast(
		oa::Span<oavk::Buffer>(broadcastBuffers, 2), 0).isOk());
	for (oa::U32 i = 0; i < 4; ++i) {
		EXPECT_FLOAT_EQ(broadcastStorage[i + 1], static_cast<oa::F32>(i + 1));
	}

	oa::F32 gatherStorage[] = {1.0F, 2.0F, 3.0F, 4.0F, 0.0F};
	oa::F32 gatherOutput[4] = {};
	const oavk::Buffer gatherInputs[] = {
		makeHostView(gatherStorage, 2),
		makeHostView(gatherStorage + 2, 2),
	};
	oavk::Buffer gatherOutputs[] = {
		makeHostView(gatherStorage + 1, 4),
		makeHostView(gatherOutput, 4),
	};
	ASSERT_TRUE(oa::HostCollectiveOracle::allGather(
		oa::Span<const oavk::Buffer>(gatherInputs, 2),
		oa::Span<oavk::Buffer>(gatherOutputs, 2)).isOk());
	for (const oavk::Buffer& output : gatherOutputs) {
		for (oa::U32 i = 0; i < 4; ++i) {
			EXPECT_FLOAT_EQ(readF32(output, i), static_cast<oa::F32>(i + 1));
		}
	}

	oa::F32 scatterStorage[] = {1.0F, 2.0F, 3.0F, 4.0F, 5.0F, 6.0F};
	oa::F32 scatterOutput[3] = {};
	const oavk::Buffer scatterInput = makeHostView(scatterStorage, 6);
	oavk::Buffer scatterOutputs[] = {
		makeHostView(scatterStorage + 3, 3),
		makeHostView(scatterOutput, 3),
	};
	ASSERT_TRUE(oa::HostCollectiveOracle::scatter(
		scatterInput,
		oa::Span<oavk::Buffer>(scatterOutputs, 2)).isOk());
	for (oa::U32 i = 0; i < 3; ++i) {
		EXPECT_FLOAT_EQ(readF32(scatterOutputs[0], i), static_cast<oa::F32>(i + 1));
		EXPECT_FLOAT_EQ(readF32(scatterOutputs[1], i), static_cast<oa::F32>(i + 4));
	}
}

// ─── Large Buffer Test ──────────────────────────────────────────────────

TEST(Collective, AllReduceLargeBuffer) {
	auto* rt = testEnginePtr();
	ASSERT_NE(rt, nullptr);

	oa::U64 count = 1024 * 1024; // 4 MB per buffer
	oa::U64 size = count * sizeof(oa::F32);
	auto bufA = makeHostBuffer(*rt, size);
	auto bufB = makeHostBuffer(*rt, size);

	fillF32(bufA, 1.0f, count);
	fillF32(bufB, 1.0f, count);

	oavk::Buffer bufs[] = {bufA, bufB};
	oa::Span<oavk::Buffer> span(bufs, 2);
	auto status = oa::HostCollectiveOracle::allReduce(span, oa::HostReduceOp::Sum);
	EXPECT_TRUE(status.isOk());

	// Spot-check a few values
	EXPECT_FLOAT_EQ(readF32(bufs[0], 0), 2.0f);
	EXPECT_FLOAT_EQ(readF32(bufs[0], count / 2), 2.0f);
	EXPECT_FLOAT_EQ(readF32(bufs[0], count - 1), 2.0f);
	EXPECT_FLOAT_EQ(readF32(bufs[1], 0), 2.0f);

	oa::EngineResourceAccess::freeBuffer(*rt, bufA);
	oa::EngineResourceAccess::freeBuffer(*rt, bufB);
}
