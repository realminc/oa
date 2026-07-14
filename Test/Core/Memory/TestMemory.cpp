// Memory: OaMemcpy, OaMemzero, OaMemEqual, aligned alloc.

#include "../../OaTest.h"
#include <Oa/Core/Memory.h>
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
