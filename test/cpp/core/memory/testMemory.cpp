// memory: oa::memcpy, oa::memzero, oa::memEqual, aligned alloc.

#include "../../oaTest.h"
#include <oa/core/std/memory.h>
#include <algorithm>
#include <array>
#include <cstring>
#include <vector>

TEST(CoreMemory, CopySmall) {
	oa::U8 src[16] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};
	oa::U8 dst[16] = {};
	oa::memcpy(dst, src, 16);
	EXPECT_EQ(std::memcmp(src, dst, 16), 0);
}

TEST(CoreMemory, MoveOverlappingRanges) {
	oa::U8 moveRight[] = {0, 1, 2, 3, 4, 5, 6, 7};
	EXPECT_EQ(oa::memmove(moveRight + 2, moveRight, 6), moveRight + 2);
	const oa::U8 expectedRight[] = {0, 1, 0, 1, 2, 3, 4, 5};
	EXPECT_EQ(std::memcmp(moveRight, expectedRight, sizeof(moveRight)), 0);

	oa::U8 moveLeft[] = {0, 1, 2, 3, 4, 5, 6, 7};
	EXPECT_EQ(oa::memmove(moveLeft, moveLeft + 2, 6), moveLeft);
	const oa::U8 expectedLeft[] = {2, 3, 4, 5, 6, 7, 6, 7};
	EXPECT_EQ(std::memcmp(moveLeft, expectedLeft, sizeof(moveLeft)), 0);
}

TEST(CoreMemory, CopyMedium) {
	void* src = oa::alignedAlloc(512, 64);
	void* dst = oa::alignedAlloc(512, 64);
	std::memset(src, 0xAB, 512);
	oa::memcpy(dst, src, 512);
	EXPECT_EQ(std::memcmp(src, dst, 512), 0);
	oa::alignedFree(src);
	oa::alignedFree(dst);
}

TEST(CoreMemory, CopyLarge) {
	oa::Usize size = 4 * 1024 * 1024;  // 4MB
	void* src = oa::alignedAlloc(size, 64);
	void* dst = oa::alignedAlloc(size, 64);
	std::memset(src, 0xCD, size);
	oa::memcpy(dst, src, size);
	EXPECT_EQ(std::memcmp(src, dst, size), 0);
	oa::alignedFree(src);
	oa::alignedFree(dst);
}

TEST(CoreMemory, CopyEverySmallSizeAndAlignment) {
	constexpr oa::Usize MaxSize = 1024;
	constexpr oa::Usize Guard = 64;
	constexpr oa::U8 Sentinel = 0xA5;
	std::array<oa::U8, MaxSize + Guard * 3> src{};
	std::array<oa::U8, MaxSize + Guard * 3> actual{};
	std::array<oa::U8, MaxSize + Guard * 3> expected{};

	for (oa::Usize index = 0; index < src.size(); ++index) {
		src[index] = static_cast<oa::U8>((index * 131U + 17U) & 0xFFU);
	}
	for (oa::Usize size = 0; size <= MaxSize; ++size) {
		for (oa::Usize srcOffset = 0; srcOffset < 64; srcOffset += 7) {
			for (oa::Usize dstOffset = 0; dstOffset < 64; dstOffset += 5) {
				actual.fill(Sentinel);
				expected.fill(Sentinel);
				std::memcpy(expected.data() + Guard + dstOffset,
					src.data() + Guard + srcOffset, size);
				EXPECT_EQ(oa::memcpy(actual.data() + Guard + dstOffset,
					src.data() + Guard + srcOffset, size),
					actual.data() + Guard + dstOffset);
				ASSERT_EQ(actual, expected)
					<< "size=" << size << " srcOffset=" << srcOffset
					<< " dstOffset=" << dstOffset;
			}
		}
	}
}

TEST(CoreMemory, StreamingCopyEveryTailAndAlignment) {
	constexpr oa::Usize MaxSize = 2048;
	constexpr oa::Usize Guard = 64;
	constexpr oa::U8 Sentinel = 0x5A;
	std::array<oa::U8, MaxSize + Guard * 3> src{};
	std::array<oa::U8, MaxSize + Guard * 3> actual{};
	std::array<oa::U8, MaxSize + Guard * 3> expected{};

	for (oa::Usize index = 0; index < src.size(); ++index) {
		src[index] = static_cast<oa::U8>((index * 67U + 29U) & 0xFFU);
	}
	for (oa::Usize size = 0; size <= MaxSize; ++size) {
		for (oa::Usize srcOffset : {oa::Usize{0}, oa::Usize{1}, oa::Usize{31}, oa::Usize{63}}) {
			for (oa::Usize dstOffset : {oa::Usize{0}, oa::Usize{1}, oa::Usize{17}, oa::Usize{63}}) {
				actual.fill(Sentinel);
				expected.fill(Sentinel);
				std::memcpy(expected.data() + Guard + dstOffset,
					src.data() + Guard + srcOffset, size);
				EXPECT_EQ(oa::memcpyStream(actual.data() + Guard + dstOffset,
					src.data() + Guard + srcOffset, size),
					actual.data() + Guard + dstOffset);
				ASSERT_EQ(actual, expected)
					<< "size=" << size << " srcOffset=" << srcOffset
					<< " dstOffset=" << dstOffset;
			}
		}
	}
}

TEST(CoreMemory, StreamingCopyAdmissionBoundaries) {
	constexpr oa::Usize Guard = 64;
	constexpr oa::U8 Sentinel = 0xA6;
	constexpr std::array sizes{
		oa::detail::MemcpyStreamMinBytes - 1U,
		oa::detail::MemcpyStreamMinBytes,
		oa::detail::MemcpyStreamMinBytes + 1U,
		oa::detail::MemcpyStreamMaxBytes - 1U,
		oa::detail::MemcpyStreamMaxBytes,
		oa::detail::MemcpyStreamMaxBytes + 1U,
	};
	const oa::Usize allocation = sizes.back() + Guard * 3U;
	std::vector<oa::U8> src(allocation);
	std::vector<oa::U8> actual(allocation);
	std::vector<oa::U8> expected(allocation);
	for (oa::Usize index = 0; index < allocation; ++index) {
		src[index] = static_cast<oa::U8>((index * 97U + 11U) & 0xFFU);
	}

	for (oa::Usize size : sizes) {
		for (oa::Usize srcOffset : {
			oa::Usize{0}, oa::Usize{1}, oa::Usize{63}})
		{
			for (oa::Usize dstOffset : {
				oa::Usize{0}, oa::Usize{17}, oa::Usize{63}})
			{
				std::fill(actual.begin(), actual.end(), Sentinel);
				std::fill(expected.begin(), expected.end(), Sentinel);
				std::memcpy(expected.data() + Guard + dstOffset,
					src.data() + Guard + srcOffset, size);
				EXPECT_EQ(oa::memcpyStream(
					actual.data() + Guard + dstOffset,
					src.data() + Guard + srcOffset, size),
					actual.data() + Guard + dstOffset);
				ASSERT_EQ(std::memcmp(
					actual.data(), expected.data(), allocation), 0)
					<< "size=" << size << " srcOffset=" << srcOffset
					<< " dstOffset=" << dstOffset;
			}
		}
	}
}

TEST(CoreMemory, Memzero) {
	void* buf = oa::alignedAlloc(256, 64);
	std::memset(buf, 0xFF, 256);
	oa::memzero(buf, 256);
	oa::U8* bytes = static_cast<oa::U8*>(buf);
	for (int i = 0; i < 256; ++i) EXPECT_EQ(bytes[i], 0);
	oa::alignedFree(buf);
}

TEST(CoreMemory, MemEqual) {
	oa::U8 a[32] = {};
	oa::U8 b[32] = {};
	std::memset(a, 0x42, 32);
	std::memset(b, 0x42, 32);
	EXPECT_TRUE(oa::memEqual(a, b, 32));
	b[15] = 0x99;
	EXPECT_FALSE(oa::memEqual(a, b, 32));
}

TEST(CoreMemory, SecureZeroAndConstantTimeEquality) {
	constexpr oa::Usize MaxSize = 1024;
	constexpr oa::Usize Guard = 64;
	constexpr oa::U8 Sentinel = 0xA7;
	std::array<oa::U8, MaxSize + Guard * 2> a{};
	std::array<oa::U8, MaxSize + Guard * 2> b{};

	EXPECT_TRUE(oa::memEqualConstantTime(nullptr, nullptr, 0));
	EXPECT_NO_FATAL_FAILURE(oa::memzeroSecure(nullptr, 0));
	for (oa::Usize size = 0; size <= MaxSize; ++size) {
		for (oa::Usize offset : {
			oa::Usize{0}, oa::Usize{1}, oa::Usize{17}, oa::Usize{63}})
		{
			a.fill(Sentinel);
			b.fill(Sentinel);
			EXPECT_TRUE(oa::memEqualConstantTime(
				a.data() + Guard + offset,
				b.data() + Guard + offset,
				size));
			if (size > 0) {
				for (const oa::Usize position : {
					oa::Usize{0}, size / 2U, size - 1U})
				{
					b[Guard + offset + position] ^= 0xFFU;
					EXPECT_FALSE(oa::memEqualConstantTime(
						a.data() + Guard + offset,
						b.data() + Guard + offset,
						size));
					b[Guard + offset + position] ^= 0xFFU;
				}
			}

			oa::memzeroSecure(a.data() + Guard + offset, size);
			for (oa::Usize index = 0; index < a.size(); ++index) {
				const bool inside = index >= Guard + offset
					and index < Guard + offset + size;
				ASSERT_EQ(a[index], inside ? 0U : Sentinel)
					<< "size=" << size << " offset=" << offset
					<< " index=" << index;
			}
		}
	}
}

TEST(CoreMemoryDeath, SecurePrimitivesRejectNullNonemptyRanges) {
	EXPECT_DEATH((void)oa::memEqual(nullptr, nullptr, 1), "");
	EXPECT_DEATH((void)oa::memEqualConstantTime(nullptr, "x", 1), "");
	EXPECT_DEATH((void)oa::memEqualConstantTime("x", nullptr, 1), "");
	EXPECT_DEATH(oa::memzeroSecure(nullptr, 1), "");
}

TEST(CoreMemory, CompareEverySizeAndMismatchPosition) {
	constexpr oa::Usize MaxSize = 1024;
	std::array<oa::U8, MaxSize> a{};
	std::array<oa::U8, MaxSize> b{};
	for (oa::Usize index = 0; index < MaxSize; ++index) {
		a[index] = static_cast<oa::U8>((index * 73U + 19U) & 0xFFU);
	}
	b = a;
	for (oa::Usize size = 0; size <= MaxSize; ++size) {
		EXPECT_EQ(oa::memcmp(a.data(), b.data(), size), 0);
		if (size == 0) continue;
		for (const oa::Usize position : {
			oa::Usize{0}, size / 2U, size - 1U})
		{
			b[position] ^= 0x80U;
			const int oaResult = oa::memcmp(a.data(), b.data(), size);
			const int stdResult = std::memcmp(a.data(), b.data(), size);
			EXPECT_EQ((oaResult > 0) - (oaResult < 0),
				(stdResult > 0) - (stdResult < 0));
			b[position] ^= 0x80U;
		}
	}
}

TEST(CoreMemory, MoveEverySizeAlignmentAndDirection) {
	constexpr oa::Usize MaxSize = 512;
	constexpr oa::Usize Guard = 64;
	std::array<oa::U8, MaxSize + Guard * 3> actual{};
	std::array<oa::U8, MaxSize + Guard * 3> expected{};

	for (oa::Usize size = 0; size <= MaxSize; ++size) {
		for (oa::Usize offset : {
			oa::Usize{0}, oa::Usize{1}, oa::Usize{17}, oa::Usize{63}})
		{
			for (oa::Usize distance : {
				oa::Usize{0}, oa::Usize{1}, oa::Usize{7}, oa::Usize{31}})
			{
				for (oa::Usize index = 0; index < actual.size(); ++index) {
					actual[index] = static_cast<oa::U8>(index * 31U + 7U);
				}
				expected = actual;
				oa::U8* const actualBase = actual.data() + Guard + offset;
				oa::U8* const expectedBase = expected.data() + Guard + offset;
				std::memmove(expectedBase + distance, expectedBase, size);
				EXPECT_EQ(oa::memmove(actualBase + distance, actualBase, size),
					actualBase + distance);
				ASSERT_EQ(actual, expected)
					<< "right size=" << size << " offset=" << offset
					<< " distance=" << distance;

				for (oa::Usize index = 0; index < actual.size(); ++index) {
					actual[index] = static_cast<oa::U8>(index * 31U + 7U);
				}
				expected = actual;
				std::memmove(expectedBase, expectedBase + distance, size);
				EXPECT_EQ(oa::memmove(actualBase, actualBase + distance, size),
					actualBase);
				ASSERT_EQ(actual, expected)
					<< "left size=" << size << " offset=" << offset
					<< " distance=" << distance;
			}
		}
	}
}

TEST(CoreMemory, FillZeroAndEqualEveryTail) {
	constexpr oa::Usize MaxSize = 1024;
	constexpr oa::Usize Guard = 64;
	constexpr oa::U8 Sentinel = 0xC7;
	std::array<oa::U8, MaxSize + Guard * 3> a{};
	std::array<oa::U8, MaxSize + Guard * 3> b{};
	std::array<oa::U8, MaxSize + Guard * 3> expected{};

	for (oa::Usize size = 0; size <= MaxSize; ++size) {
		for (oa::Usize offset : {oa::Usize{0}, oa::Usize{1}, oa::Usize{17}, oa::Usize{63}}) {
			a.fill(Sentinel);
			expected.fill(Sentinel);
			std::memset(expected.data() + Guard + offset, 0x6D, size);
			EXPECT_EQ(oa::memset(a.data() + Guard + offset, 0x6D, size),
				a.data() + Guard + offset);
			ASSERT_EQ(a, expected) << "fill size=" << size << " offset=" << offset;

			a.fill(Sentinel);
			expected.fill(Sentinel);
			std::memset(expected.data() + Guard + offset, 0, size);
			EXPECT_EQ(oa::memzero(a.data() + Guard + offset, size),
				a.data() + Guard + offset);
			ASSERT_EQ(a, expected) << "zero size=" << size << " offset=" << offset;

			for (oa::Usize index = 0; index < a.size(); ++index) {
				a[index] = static_cast<oa::U8>((index * 29U + 11U) & 0xFFU);
			}
			b = a;
			EXPECT_TRUE(oa::memEqual(a.data() + Guard + offset,
				b.data() + Guard + offset, size));
			if (size > 0) {
				const oa::Usize positions[] = {0, size / 2, size - 1};
				for (oa::Usize position : positions) {
					b[Guard + offset + position] ^= 0xFF;
					EXPECT_FALSE(oa::memEqual(a.data() + Guard + offset,
						b.data() + Guard + offset, size))
						<< "equal size=" << size << " offset=" << offset
						<< " mismatch=" << position;
					b[Guard + offset + position] ^= 0xFF;
				}
			}
		}
	}
}

TEST(CoreMemory, AlignedAlloc) {
	void* ptr = oa::alignedAlloc(1024, 64);
	EXPECT_NE(ptr, nullptr);
	EXPECT_EQ(reinterpret_cast<oa::Usize>(ptr) % 64, 0u);
	oa::alignedFree(ptr);
}

// ─── benchmark (optional, prints timing) ───────────────────────────────────────

TEST(CoreMemory, BenchSmall) {
	oa::U8 src[64], dst[64];
	benchmark("Memcpy 64B", 1000000, [&]() { oa::memcpy(dst, src, 64); });
}

TEST(CoreMemory, BenchLarge) {
	oa::Usize size = 16 * 1024 * 1024;
	void* src = oa::alignedAlloc(size, 64);
	void* dst = oa::alignedAlloc(size, 64);
	benchmark("Memcpy 16MB", 50, [&]() { oa::memcpy(dst, src, size); });
	oa::alignedFree(src);
	oa::alignedFree(dst);
}
