// Memory: OaMemcpy, OaMemzero, OaMemEqual, aligned alloc.

#include "../../OaTest.h"
#include <Oa/Core/Memory.h>
#include <array>
#include <cstring>

TEST(CoreMemory, CopySmall) {
	OaU8 src[16] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};
	OaU8 dst[16] = {};
	OaMemcpy(dst, src, 16);
	EXPECT_EQ(std::memcmp(src, dst, 16), 0);
}

TEST(CoreMemory, CopyMedium) {
	void* src = OaAlignedAlloc(512, 64);
	void* dst = OaAlignedAlloc(512, 64);
	std::memset(src, 0xAB, 512);
	OaMemcpy(dst, src, 512);
	EXPECT_EQ(std::memcmp(src, dst, 512), 0);
	OaAlignedFree(src);
	OaAlignedFree(dst);
}

TEST(CoreMemory, CopyLarge) {
	OaUsize size = 4 * 1024 * 1024;  // 4MB
	void* src = OaAlignedAlloc(size, 64);
	void* dst = OaAlignedAlloc(size, 64);
	std::memset(src, 0xCD, size);
	OaMemcpy(dst, src, size);
	EXPECT_EQ(std::memcmp(src, dst, size), 0);
	OaAlignedFree(src);
	OaAlignedFree(dst);
}

TEST(CoreMemory, CopyEverySmallSizeAndAlignment) {
	constexpr OaUsize MaxSize = 1024;
	constexpr OaUsize Guard = 64;
	constexpr OaU8 Sentinel = 0xA5;
	std::array<OaU8, MaxSize + Guard * 3> src{};
	std::array<OaU8, MaxSize + Guard * 3> actual{};
	std::array<OaU8, MaxSize + Guard * 3> expected{};

	for (OaUsize index = 0; index < src.size(); ++index) {
		src[index] = static_cast<OaU8>((index * 131U + 17U) & 0xFFU);
	}
	for (OaUsize size = 0; size <= MaxSize; ++size) {
		for (OaUsize srcOffset = 0; srcOffset < 64; srcOffset += 7) {
			for (OaUsize dstOffset = 0; dstOffset < 64; dstOffset += 5) {
				actual.fill(Sentinel);
				expected.fill(Sentinel);
				std::memcpy(expected.data() + Guard + dstOffset,
					src.data() + Guard + srcOffset, size);
				EXPECT_EQ(OaMemcpy(actual.data() + Guard + dstOffset,
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
	constexpr OaUsize MaxSize = 2048;
	constexpr OaUsize Guard = 64;
	constexpr OaU8 Sentinel = 0x5A;
	std::array<OaU8, MaxSize + Guard * 3> src{};
	std::array<OaU8, MaxSize + Guard * 3> actual{};
	std::array<OaU8, MaxSize + Guard * 3> expected{};

	for (OaUsize index = 0; index < src.size(); ++index) {
		src[index] = static_cast<OaU8>((index * 67U + 29U) & 0xFFU);
	}
	for (OaUsize size = 0; size <= MaxSize; ++size) {
		for (OaUsize srcOffset : {OaUsize{0}, OaUsize{1}, OaUsize{31}, OaUsize{63}}) {
			for (OaUsize dstOffset : {OaUsize{0}, OaUsize{1}, OaUsize{17}, OaUsize{63}}) {
				actual.fill(Sentinel);
				expected.fill(Sentinel);
				std::memcpy(expected.data() + Guard + dstOffset,
					src.data() + Guard + srcOffset, size);
				EXPECT_EQ(OaMemcpyStream(actual.data() + Guard + dstOffset,
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
	void* buf = OaAlignedAlloc(256, 64);
	std::memset(buf, 0xFF, 256);
	OaMemzero(buf, 256);
	OaU8* bytes = static_cast<OaU8*>(buf);
	for (int i = 0; i < 256; ++i) EXPECT_EQ(bytes[i], 0);
	OaAlignedFree(buf);
}

TEST(CoreMemory, MemEqual) {
	OaU8 a[32] = {};
	OaU8 b[32] = {};
	std::memset(a, 0x42, 32);
	std::memset(b, 0x42, 32);
	EXPECT_TRUE(OaMemEqual(a, b, 32));
	b[15] = 0x99;
	EXPECT_FALSE(OaMemEqual(a, b, 32));
}

TEST(CoreMemory, FillZeroAndEqualEveryTail) {
	constexpr OaUsize MaxSize = 1024;
	constexpr OaUsize Guard = 64;
	constexpr OaU8 Sentinel = 0xC7;
	std::array<OaU8, MaxSize + Guard * 3> a{};
	std::array<OaU8, MaxSize + Guard * 3> b{};
	std::array<OaU8, MaxSize + Guard * 3> expected{};

	for (OaUsize size = 0; size <= MaxSize; ++size) {
		for (OaUsize offset : {OaUsize{0}, OaUsize{1}, OaUsize{17}, OaUsize{63}}) {
			a.fill(Sentinel);
			expected.fill(Sentinel);
			std::memset(expected.data() + Guard + offset, 0x6D, size);
			EXPECT_EQ(OaMemset(a.data() + Guard + offset, 0x6D, size),
				a.data() + Guard + offset);
			ASSERT_EQ(a, expected) << "fill size=" << size << " offset=" << offset;

			a.fill(Sentinel);
			expected.fill(Sentinel);
			std::memset(expected.data() + Guard + offset, 0, size);
			EXPECT_EQ(OaMemzero(a.data() + Guard + offset, size),
				a.data() + Guard + offset);
			ASSERT_EQ(a, expected) << "zero size=" << size << " offset=" << offset;

			for (OaUsize index = 0; index < a.size(); ++index) {
				a[index] = static_cast<OaU8>((index * 29U + 11U) & 0xFFU);
			}
			b = a;
			EXPECT_TRUE(OaMemEqual(a.data() + Guard + offset,
				b.data() + Guard + offset, size));
			if (size > 0) {
				const OaUsize positions[] = {0, size / 2, size - 1};
				for (OaUsize position : positions) {
					b[Guard + offset + position] ^= 0xFF;
					EXPECT_FALSE(OaMemEqual(a.data() + Guard + offset,
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
	void* ptr = OaAlignedAlloc(1024, 64);
	EXPECT_NE(ptr, nullptr);
	EXPECT_EQ(reinterpret_cast<OaUsize>(ptr) % 64, 0u);
	OaAlignedFree(ptr);
}

// ─── Benchmark (optional, prints timing) ───────────────────────────────────────

TEST(CoreMemory, BenchSmall) {
	OaU8 src[64], dst[64];
	OaBenchmark("Memcpy 64B", 1000000, [&]() { OaMemcpy(dst, src, 64); });
}

TEST(CoreMemory, BenchLarge) {
	OaUsize size = 16 * 1024 * 1024;
	void* src = OaAlignedAlloc(size, 64);
	void* dst = OaAlignedAlloc(size, 64);
	OaBenchmark("Memcpy 16MB", 50, [&]() { OaMemcpy(dst, src, size); });
	OaAlignedFree(src);
	OaAlignedFree(dst);
}
