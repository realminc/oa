// TestFnHashExtended — OaFnHash scale, a direct Keccak KAT on the GPU, and
// deferred cross-kernel chaining (Shake → Merkle in one Execute).

#include "../../OaTest.h"

#include <Oa/Crypto/FnHash.h>
#include <Oa/Crypto/Hash.h>
#include <Oa/Crypto/Keccak.h>
#include <Oa/Core/FnMatrix.h>
#include <Oa/Runtime/Context.h>
#include <Oa/Runtime/Engine.h>

#include <cstring>
#include <vector>

namespace {

void SyncCtx() {
	auto& ctx = OaContext::GetDefault();
	ASSERT_TRUE(ctx.Execute().IsOk());
	ASSERT_TRUE(ctx.Sync().IsOk());
}

std::vector<OaByte> Lcg(OaU32 InSeed, OaUsize InCount) {
	std::vector<OaByte> v(InCount);
	OaU32 s = InSeed | 1U;
	for (auto& b : v) {
		s = s * 1664525U + 1013904223U;
		b = static_cast<OaByte>(s >> 24);
	}
	return v;
}

OaMatrix UploadBytes(const std::vector<OaByte>& InData, OaI64 InRows, OaI64 InCols) {
	return OaFnMatrix::FromBytes(
		OaSpan<const OaU8>(InData.data(), InData.size()),
		OaMatrixShape{InRows, InCols}, OaScalarType::UInt8);
}

std::vector<OaByte> ReadBack(const OaMatrix& InM, OaUsize InBytes) {
	std::vector<OaByte> h(InBytes);
	EXPECT_TRUE(OaFnMatrix::CopyToHost(InM, h.data(), InBytes).IsOk());
	return h;
}

// ─── Large batch SHAKE-256 (256 messages) vs CPU ───────────────────────────

TEST_F(OaVkEngineTestFixture, Shake256LargeBatchMatchesCpu) {
	const OaU32 kRows = 256;
	const OaU32 msgLen = 137;   // > SHAKE-256 rate → 2 blocks
	auto msgs = Lcg(0x1234U, kRows * msgLen);
	OaMatrix in = UploadBytes(msgs, kRows, msgLen);
	OaMatrix out = OaFnHash::Shake256(in, 32);
	SyncCtx();
	auto host = ReadBack(out, kRows * 32);
	for (OaU32 r = 0; r < kRows; ++r) {
		OaByte ref[32];
		OaShake256(msgs.data() + r * msgLen, msgLen, ref, 32);
		ASSERT_EQ(0, std::memcmp(ref, host.data() + r * 32, 32)) << "row " << r;
	}
}

// ─── Large Merkle tree (1024 leaves, 10 reduction levels) vs CPU ───────────

TEST_F(OaVkEngineTestFixture, MerkleRootLargeMatchesCpu) {
	const OaU32 n = 1024;
	auto leafBytes = Lcg(0x9E3779U, n * 32);
	OaMatrix in = UploadBytes(leafBytes, n, 32);
	OaMatrix root = OaFnHash::MerkleRoot(in);
	SyncCtx();
	auto host = ReadBack(root, 32);

	OaVec<OaHash> leaves;
	leaves.reserve(n);
	for (OaU32 i = 0; i < n; ++i) {
		OaHash h;
		std::memcpy(h.Bytes.data(), leafBytes.data() + i * 32, 32);
		leaves.push_back(h);
	}
	OaHash cpu = OaMerkleRoot(leaves);
	EXPECT_EQ(0, std::memcmp(cpu.Bytes.data(), host.data(), 32));
}

// ─── Direct Keccak-f[1600] KAT: permutation of the zero state ──────────────
// Keccak team reference: lane 0 of Keccak-f[1600](0) = 0xF1258F7940E1DDE7.

TEST_F(OaVkEngineTestFixture, KeccakF1600ZeroStateKat) {
	std::vector<OaByte> zero(200, 0);
	OaMatrix in = UploadBytes(zero, 1, 200);
	OaMatrix out = OaFnHash::KeccakF1600(in);
	SyncCtx();
	auto host = ReadBack(out, 200);
	OaU64 lane0 = 0;
	std::memcpy(&lane0, host.data(), 8);
	EXPECT_EQ(lane0, 0xF1258F7940E1DDE7ULL);

	// And it must equal the CPU primitive on the same input.
	OaU64 s[25] = {};
	OaKeccakF1600(s);
	EXPECT_EQ(0, std::memcmp(s, host.data(), 200));
}

// ─── Deferred cross-kernel chaining: Shake256 → MerkleRoot in one Execute ──
// Exercises auto-context dependency tracking across two crypto kernels.

TEST_F(OaVkEngineTestFixture, ShakeThenMerkleChained) {
	const OaU32 n = 8;
	const OaU32 msgLen = 64;
	auto msgs = Lcg(0xC0DEU, n * msgLen);
	OaMatrix in = UploadBytes(msgs, n, msgLen);

	OaMatrix leaves = OaFnHash::Shake256(in, 32);   // [n, 32]
	OaMatrix root = OaFnHash::MerkleRoot(leaves);    // [1, 32]
	SyncCtx();
	auto host = ReadBack(root, 32);

	OaVec<OaHash> cpuLeaves;
	cpuLeaves.reserve(n);
	for (OaU32 i = 0; i < n; ++i) {
		OaHash h;
		OaShake256(msgs.data() + i * msgLen, msgLen, h.Bytes.data(), 32);
		cpuLeaves.push_back(h);
	}
	OaHash cpuRoot = OaMerkleRoot(cpuLeaves);
	EXPECT_EQ(0, std::memcmp(cpuRoot.Bytes.data(), host.data(), 32));
}

} // namespace
