// TestFnHashExtended — oa::FnHash scale, a direct Keccak KAT on the GPU, and
// deferred cross-kernel chaining (Shake → Merkle in one execute).

#include "../../oaTest.h"

#include <oa/crypto/fnHash.h>
#include <oa/crypto/hash.h>
#include <oa/crypto/keccak.h>
#include <oa/core/fnMatrix.h>
#include <oa/runtime/executionSession.h>
#include <oa/runtime/engine.h>

#include <cstring>
#include <vector>

namespace {

void syncCtx() {
	auto& ctx = oa::ExecutionSession::getActive();
	ASSERT_TRUE(testSubmitAndWait(ctx).isOk());
}

std::vector<oa::Byte> lcg(oa::U32 inSeed, oa::Usize inCount) {
	std::vector<oa::Byte> v(inCount);
	oa::U32 s = inSeed | 1U;
	for (auto& b : v) {
		s = s * 1664525U + 1013904223U;
		b = static_cast<oa::Byte>(s >> 24);
	}
	return v;
}

oa::Matrix uploadBytes(const std::vector<oa::Byte>& inData, oa::I64 inRows, oa::I64 inCols) {
	return oa::FnMatrix::fromBytes(
		oa::Span<const oa::U8>(inData.data(), inData.size()),
		oa::MatrixShape{inRows, inCols}, oa::ScalarType::UInt8);
}

std::vector<oa::Byte> readBack(const oa::Matrix& inM, oa::Usize inBytes) {
	std::vector<oa::Byte> h(inBytes);
	EXPECT_TRUE(oa::FnMatrix::copyToHost(inM, h.data(), inBytes).isOk());
	return h;
}

// ─── Large batch SHAKE-256 (256 messages) vs CPU ───────────────────────────

TEST_F(VkEngineTestFixture, Shake256LargeBatchMatchesCpu) {
	const oa::U32 kRows = 256;
	const oa::U32 msgLen = 137;   // > SHAKE-256 rate → 2 blocks
	auto msgs = lcg(0x1234U, kRows * msgLen);
	oa::Matrix in = uploadBytes(msgs, kRows, msgLen);
	oa::Matrix out = oa::FnHash::shake256(in, 32);
	syncCtx();
	auto host = readBack(out, kRows * 32);
	for (oa::U32 r = 0; r < kRows; ++r) {
		oa::Byte ref[32];
		oa::shake256(msgs.data() + r * msgLen, msgLen, ref, 32);
		ASSERT_EQ(0, std::memcmp(ref, host.data() + r * 32, 32)) << "row " << r;
	}
}

// ─── Large Merkle tree (1024 leaves, 10 reduction levels) vs CPU ───────────

TEST_F(VkEngineTestFixture, MerkleRootLargeMatchesCpu) {
	const oa::U32 n = 1024;
	auto leafBytes = lcg(0x9E3779U, n * 32);
	oa::Matrix in = uploadBytes(leafBytes, n, 32);
	oa::Matrix root = oa::FnHash::merkleRoot(in);
	syncCtx();
	auto host = readBack(root, 32);

	oa::Vec<oa::Hash> leaves;
	leaves.reserve(n);
	for (oa::U32 i = 0; i < n; ++i) {
		oa::Hash h;
		std::memcpy(h.bytes.data(), leafBytes.data() + i * 32, 32);
		leaves.pushBack(h);
	}
	oa::Hash cpu = oa::merkleRoot(leaves);
	EXPECT_EQ(0, std::memcmp(cpu.bytes.data(), host.data(), 32));
}

// ─── Direct Keccak-f[1600] KAT: permutation of the zero state ──────────────
// Keccak team reference: lane 0 of Keccak-f[1600](0) = 0xF1258F7940E1DDE7.

TEST_F(VkEngineTestFixture, KeccakF1600ZeroStateKat) {
	std::vector<oa::Byte> zero(200, 0);
	oa::Matrix in = uploadBytes(zero, 1, 200);
	oa::Matrix out = oa::FnHash::keccakF1600(in);
	syncCtx();
	auto host = readBack(out, 200);
	oa::U64 lane0 = 0;
	std::memcpy(&lane0, host.data(), 8);
	EXPECT_EQ(lane0, 0xF1258F7940E1DDE7ULL);

	// And it must equal the CPU primitive on the same input.
	oa::U64 s[25] = {};
	oa::keccakF1600(s);
	EXPECT_EQ(0, std::memcmp(s, host.data(), 200));
}

// ─── Deferred cross-kernel chaining: Shake256 → MerkleRoot in one execute ──
// Exercises auto-context dependency tracking across two crypto kernels.

TEST_F(VkEngineTestFixture, ShakeThenMerkleChained) {
	const oa::U32 n = 8;
	const oa::U32 msgLen = 64;
	auto msgs = lcg(0xC0DEU, n * msgLen);
	oa::Matrix in = uploadBytes(msgs, n, msgLen);

	oa::Matrix leaves = oa::FnHash::shake256(in, 32);   // [n, 32]
	oa::Matrix root = oa::FnHash::merkleRoot(leaves);    // [1, 32]
	syncCtx();
	auto host = readBack(root, 32);

	oa::Vec<oa::Hash> cpuLeaves;
	cpuLeaves.reserve(n);
	for (oa::U32 i = 0; i < n; ++i) {
		oa::Hash h;
		oa::shake256(msgs.data() + i * msgLen, msgLen, h.bytes.data(), 32);
		cpuLeaves.pushBack(h);
	}
	oa::Hash cpuRoot = oa::merkleRoot(cpuLeaves);
	EXPECT_EQ(0, std::memcmp(cpuRoot.bytes.data(), host.data(), 32));
}

} // namespace
