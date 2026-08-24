// memory: oa::memcpy, oa::memzero, oa::memEqual, aligned alloc.

#include "../../oaTest.h"
#include <oa/core/memory.h>
#include <array>
#include <cstring>

TEST(CoreMemory, CopySmall) {
	oa::U8 src[16] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};
	oa::U8 dst[16] = {};
	oa::memcpy(dst, src, 16);
	EXPECT_EQ(std::memcmp(src, dst, 16), 0);
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
