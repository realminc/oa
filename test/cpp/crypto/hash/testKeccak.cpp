// CPU Keccak/SHAKE/KMAC tests.
//
// Test vectors from:
//   - Keccak-f[1600]: NIST reference (zero-state permutation)
//   - SHAKE-128/256: NIST FIPS 202 examples (ShortMsgKAT)
//   - KMAC-256: NIST SP 800-185 sample vectors
//
// No vulkan required — pure CPU tests.

#include <gtest/gtest.h>
#include <oa/crypto/keccak.h>

#include <cstring>

static oa::String toHex(const oa::Byte* inData, oa::Usize inLen) {
	static const char kHex[] = "0123456789abcdef";
	oa::String result;
	result.reserve(inLen * 2);
	for (oa::Usize i = 0; i < inLen; ++i) {
		result += kHex[inData[i] >> 4];
		result += kHex[inData[i] & 0x0F];
	}
	return result;
}

static oa::Vector<oa::Byte> fromHex(const oa::String& inHex) {
	oa::Vector<oa::Byte> result;
	result.reserve(inHex.size() / 2);
	for (oa::Usize i = 0; i + 1 < inHex.size(); i += 2) {
		auto nibble = [](char c) -> oa::Byte {
			if (c >= '0' && c <= '9') { return static_cast<oa::Byte>(c - '0'); }
			if (c >= 'a' && c <= 'f') { return static_cast<oa::Byte>(10 + c - 'a'); }
			if (c >= 'A' && c <= 'F') { return static_cast<oa::Byte>(10 + c - 'A'); }
			return static_cast<oa::Byte>(0);
		};
		result.pushBack(static_cast<oa::Byte>((nibble(inHex[i]) << 4) | nibble(inHex[i + 1])));
	}
	return result;
}

// Keccak-f[1600] — zero state test vector.
// The result of applying Keccak-f[1600] to all-zero state is a well-known constant.
// Reference: https://keccak.team/archives.html (KeccakF-1600-IntermediateValues.txt)
TEST(Keccak, F1600ZeroState) {
	oa::U64 state[25] = {};
	oa::keccakF1600(state);

	// After one permutation of all-zero state, lane[0] should be:
	// 0xF1258F7940E1DDE7 (from Keccak reference implementation)
	EXPECT_EQ(state[0], 0xF1258F7940E1DDE7ULL);
	EXPECT_EQ(state[1], 0x84D5CCF933C0478AULL);
	EXPECT_EQ(state[2], 0xD598261EA65AA9EEULL);
	EXPECT_EQ(state[3], 0xBD1547306F80494DULL);
	EXPECT_EQ(state[4], 0x8B284E056253D057ULL);
}

// Keccak-f[1600] — double permutation should be deterministic.
TEST(Keccak, F1600Deterministic) {
	oa::U64 s1[25] = {};
	oa::U64 s2[25] = {};

	oa::keccakF1600(s1);
	oa::keccakF1600(s1);

	oa::keccakF1600(s2);
	oa::keccakF1600(s2);

	EXPECT_EQ(std::memcmp(s1, s2, 200), 0);
}

// SHAKE-256 empty input, 32-byte output.
// NIST FIPS 202 test vector: SHAKE256("", 256 bits)
// expected: 46b9dd2b0ba88d13233b3feb743eeb243fcd52ea62b81b82b50c27646ed5762f
TEST(Shake256, EmptyInput32) {
	oa::Byte digest[32];
	oa::shake256(nullptr, 0, digest, 32);
	oa::String hex = toHex(digest, 32);
	EXPECT_EQ(hex, "46b9dd2b0ba88d13233b3feb743eeb243fcd52ea62b81b82b50c27646ed5762f");
}

// SHAKE-256 empty input, 64-byte output (XOF — squeeze more than one block).
TEST(Shake256, EmptyInput64) {
	oa::Byte digest[64];
	oa::shake256(nullptr, 0, digest, 64);
	oa::String hex = toHex(digest, 64);
	// first 32 bytes must match the 32-byte test above
	EXPECT_EQ(hex.substr(0, 64), "46b9dd2b0ba88d13233b3feb743eeb243fcd52ea62b81b82b50c27646ed5762f");
}

// SHAKE-128 empty input, 32-byte output.
// NIST FIPS 202: SHAKE128("", 256 bits)
// expected: 7f9c2ba4e88f827d616045507605853ed73b8093f6efbc88eb1a6eacfa66ef26
TEST(Shake128, EmptyInput32) {
	oa::Byte digest[32];
	oa::shake128(nullptr, 0, digest, 32);
	oa::String hex = toHex(digest, 32);
	EXPECT_EQ(hex, "7f9c2ba4e88f827d616045507605853ed73b8093f6efbc88eb1a6eacfa66ef26");
}

// SHAKE-256 with known 1-byte input.
// input: 0x61 ("a"), 32-byte output
TEST(Shake256, SingleByte) {
	oa::Byte input = 0x61;
	oa::Byte digest[32];
	oa::shake256(&input, 1, digest, 32);
	oa::String hex = toHex(digest, 32);
	// Cross-validated against Python: hashlib.shake_256(b"a").hexdigest(32)
	EXPECT_EQ(hex, "867e2cb04f5a04dcbd592501a5e8fe9ceaafca50255626ca736c138042530ba4");
}

// SHAKE-256 incremental API should match one-shot.
TEST(Shake256, IncrementalMatchesOneShot) {
	const oa::Byte msg[] = "The quick brown fox jumps over the lazy dog";
	oa::Usize len = sizeof(msg) - 1;

	oa::Byte oneShot[64];
	oa::shake256(msg, len, oneShot, 64);

	oa::ShakeContext ctx;
	oa::shake256Init(ctx);
	// Feed in 3 chunks
	oa::shakeAbsorb(ctx, msg, 10);
	oa::shakeAbsorb(ctx, msg + 10, 20);
	oa::shakeAbsorb(ctx, msg + 30, len - 30);
	oa::Byte incremental[64];
	oa::shakeSqueeze(ctx, incremental, 64);

	EXPECT_EQ(std::memcmp(oneShot, incremental, 64), 0);
}

// SHAKE-256 incremental: squeeze in multiple calls.
TEST(Shake256, IncrementalMultipleSqueeze) {
	const oa::Byte msg[] = "hello";
	oa::Usize len = 5;

	oa::Byte oneShot[96];
	oa::shake256(msg, len, oneShot, 96);

	oa::ShakeContext ctx;
	oa::shake256Init(ctx);
	oa::shakeAbsorb(ctx, msg, len);
	oa::Byte part1[32];
	oa::Byte part2[32];
	oa::Byte part3[32];
	oa::shakeSqueeze(ctx, part1, 32);
	oa::shakeSqueeze(ctx, part2, 32);
	oa::shakeSqueeze(ctx, part3, 32);

	oa::Byte combined[96];
	std::memcpy(combined, part1, 32);
	std::memcpy(combined + 32, part2, 32);
	std::memcpy(combined + 64, part3, 32);

	EXPECT_EQ(std::memcmp(oneShot, combined, 96), 0);
}

// SHAKE-128 incremental API should match one-shot.
TEST(Shake128, IncrementalMatchesOneShot) {
	const oa::Byte msg[] = "test message for shake-128 incremental";
	oa::Usize len = sizeof(msg) - 1;

	oa::Byte oneShot[48];
	oa::shake128(msg, len, oneShot, 48);

	oa::ShakeContext ctx;
	oa::shake128Init(ctx);
	oa::shakeAbsorb(ctx, msg, len);
	oa::Byte incremental[48];
	oa::shakeSqueeze(ctx, incremental, 48);

	EXPECT_EQ(std::memcmp(oneShot, incremental, 48), 0);
}

// SHAKE-256 long input (> rate = 136 bytes) — forces multi-block absorb.
TEST(Shake256, LongInput) {
	oa::Byte input[512];
	for (oa::U32 i = 0; i < 512; ++i) {
		input[i] = static_cast<oa::Byte>(i & 0xFF);
	}

	oa::Byte digest[32];
	oa::shake256(input, 512, digest, 32);

	// verify incremental matches
	oa::ShakeContext ctx;
	oa::shake256Init(ctx);
	oa::shakeAbsorb(ctx, input, 512);
	oa::Byte incremental[32];
	oa::shakeSqueeze(ctx, incremental, 32);
	EXPECT_EQ(std::memcmp(digest, incremental, 32), 0);
}

// KMAC-256 test vector from NIST SP 800-185 (Sample #4).
// key  = 40 41 42 43 44 45 46 47 48 49 4A 4B 4C 4D 4E 4F
//        50 51 52 53 54 55 56 57 58 59 5A 5B 5C 5D 5E 5F
// Data = 00 01 02 03
// Custom = "My Tagged Application"
// output (first 64 bytes, L=512 bits)
TEST(Kmac256, NistSample4) {
	auto key = fromHex(
		"404142434445464748494a4b4c4d4e4f"
		"505152535455565758595a5b5c5d5e5f");
	auto data = fromHex("00010203");
	const oa::Byte custom[] = "My Tagged Application";
	oa::Usize customLen = sizeof(custom) - 1;

	oa::Byte mac[64];
	ASSERT_TRUE(oa::kmac256(
		key.data(), key.size(),
		data.data(), data.size(),
		custom, customLen,
		mac, 64).isOk());

	oa::String hex = toHex(mac, 64);
	EXPECT_EQ(hex,
		"20c570c31346f703c9ac36c61c03cb64"
		"c3970d0cfc787e9b79599d273a68d2f7"
		"f69d4cc3de9d104a351689f27cf6f595"
		"1f0103f33f4f24871024d9c27773a8dd");
}

// KMAC-256 empty key / empty custom — should still produce deterministic output.
TEST(Kmac256, EmptyKeyEmptyCustom) {
	const oa::Byte msg[] = "hello";
	oa::Byte mac1[32];
	oa::Byte mac2[32];

	ASSERT_TRUE(oa::kmac256(nullptr, 0, msg, 5, nullptr, 0, mac1, 32).isOk());
	ASSERT_TRUE(oa::kmac256(nullptr, 0, msg, 5, nullptr, 0, mac2, 32).isOk());

	EXPECT_EQ(std::memcmp(mac1, mac2, 32), 0);
}

// KMAC-256 — different keys produce different MACs.
TEST(Kmac256, DifferentKeys) {
	const oa::Byte msg[] = "same message";
	const oa::Byte key1[] = {0x01, 0x02, 0x03, 0x04};
	const oa::Byte key2[] = {0x05, 0x06, 0x07, 0x08};

	oa::Byte mac1[32];
	oa::Byte mac2[32];
	ASSERT_TRUE(oa::kmac256(key1, 4, msg, 12, nullptr, 0, mac1, 32).isOk());
	ASSERT_TRUE(oa::kmac256(key2, 4, msg, 12, nullptr, 0, mac2, 32).isOk());

	EXPECT_NE(std::memcmp(mac1, mac2, 32), 0);
}

// KMAC-256 — different customization strings produce different MACs.
TEST(Kmac256, DifferentCustom) {
	const oa::Byte msg[] = "same message";
	const oa::Byte key[] = {0x01, 0x02, 0x03, 0x04};
	const oa::Byte custom1[] = "domain-a";
	const oa::Byte custom2[] = "domain-b";

	oa::Byte mac1[32];
	oa::Byte mac2[32];
	ASSERT_TRUE(oa::kmac256(key, 4, msg, 12, custom1, 8, mac1, 32).isOk());
	ASSERT_TRUE(oa::kmac256(key, 4, msg, 12, custom2, 8, mac2, 32).isOk());

	EXPECT_NE(std::memcmp(mac1, mac2, 32), 0);
}

TEST(Kmac256, LargeKeyAndCustomizationAreBoundedSafely) {
	oa::Vector<oa::Byte> key(4096, 0xA5);
	oa::Vector<oa::Byte> custom(4096, 0x5A);
	const oa::Byte msg[] = {'l', 'a', 'r', 'g', 'e'};
	oa::Byte mac1[64];
	oa::Byte mac2[64];
	ASSERT_TRUE(oa::kmac256(key.data(), key.size(), msg, sizeof(msg),
		custom.data(), custom.size(), mac1, sizeof(mac1)).isOk());
	ASSERT_TRUE(oa::kmac256(key.data(), key.size(), msg, sizeof(msg),
		custom.data(), custom.size(), mac2, sizeof(mac2)).isOk());
	EXPECT_EQ(std::memcmp(mac1, mac2, sizeof(mac1)), 0);
}

TEST(Kmac256, RejectsNullNonEmptyInputs) {
	oa::Byte out[32];
	EXPECT_TRUE(oa::kmac256(nullptr, 1, nullptr, 0, nullptr, 0, out, sizeof(out)).isError());
	EXPECT_TRUE(oa::kmac256(nullptr, 0, nullptr, 1, nullptr, 0, out, sizeof(out)).isError());
	EXPECT_TRUE(oa::kmac256(nullptr, 0, nullptr, 0, nullptr, 1, out, sizeof(out)).isError());
	EXPECT_TRUE(oa::kmac256(nullptr, 0, nullptr, 0, nullptr, 0, nullptr, 1).isError());
}

// Performance: SHAKE-256 throughput (1 MB).
TEST(Shake256, Throughput1MB) {
	const oa::Usize size = oa::Usize{1024} * 1024;
	oa::Vector<oa::Byte> data(size, 0xAB);
	oa::Byte digest[32];

	auto start = std::chrono::high_resolution_clock::now();
	const oa::I32 iterations = 100;
	for (oa::I32 i = 0; i < iterations; ++i) {
		oa::shake256(data.data(), size, digest, 32);
	}
	auto end = std::chrono::high_resolution_clock::now();

	double ms = std::chrono::duration<double, std::milli>(end - start).count();
	double mbPerSec = (static_cast<double>(size) * iterations / (1024.0 * 1024.0)) / (ms / 1000.0);

	// Just log — don't fail on speed
	testing::Test::RecordProperty("shake256_mb_per_sec", std::to_string(mbPerSec));
	printf("  SHAKE-256 throughput: %.1f MB/s (1MB x %d, %.1f ms total)\n",
		mbPerSec, iterations, ms);
}
