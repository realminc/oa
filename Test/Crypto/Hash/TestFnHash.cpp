// TestFnHash — OaFnHash GPU batch hashing vs the CPU primitives.
//
// The CPU primitives (OaShake256/OaShake128/OaKeccakF1600/OaMerkleRoot) are
// NIST-KAT-anchored in TestKeccak, so byte-for-byte GPU==CPU here transitively
// validates the Vulkan Keccak/SHAKE/Merkle kernels against FIPS 202 vectors.
// Every op is a deferred auto-context dispatch: record → Execute → Sync →
// CopyToHost → compare.

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

// Deterministic pseudo-random bytes (LCG) — reproducible test data.
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

// ─── SHAKE-256 batch vs CPU ────────────────────────────────────────────────
// Message lengths span < rate, == rate (2 blocks), and > rate to exercise the
// multi-block absorb loop and the pad10*1 tail placement.

TEST_F(OaVkEngineTestFixture, Shake256BatchMatchesCpu) {
	const OaU32 kRows = 4;
	for (OaU32 msgLen : {OaU32{1}, OaU32{100}, OaU32{136}, OaU32{200}, OaU32{400}}) {
		auto msgs = Lcg(0xC0FFEEU ^ msgLen, kRows * msgLen);
		OaMatrix in = UploadBytes(msgs, kRows, msgLen);
		OaMatrix out = OaFnHash::Shake256(in, 32);
		SyncCtx();
		auto host = ReadBack(out, kRows * 32);
		for (OaU32 r = 0; r < kRows; ++r) {
			OaByte ref[32];
			OaShake256(msgs.data() + r * msgLen, msgLen, ref, 32);
			EXPECT_EQ(0, std::memcmp(ref, host.data() + r * 32, 32))
				<< "SHAKE256 mismatch: msgLen=" << msgLen << " row=" << r;
		}
	}
}

// ─── SHAKE-128 batch vs CPU (rate 168) ─────────────────────────────────────

TEST_F(OaVkEngineTestFixture, Shake128BatchMatchesCpu) {
	const OaU32 kRows = 4;
	for (OaU32 msgLen : {OaU32{1}, OaU32{168}, OaU32{300}}) {
		auto msgs = Lcg(0xBADA55U ^ msgLen, kRows * msgLen);
		OaMatrix in = UploadBytes(msgs, kRows, msgLen);
		OaMatrix out = OaFnHash::Shake128(in, 16);
		SyncCtx();
		auto host = ReadBack(out, kRows * 16);
		for (OaU32 r = 0; r < kRows; ++r) {
			OaByte ref[16];
			OaShake128(msgs.data() + r * msgLen, msgLen, ref, 16);
			EXPECT_EQ(0, std::memcmp(ref, host.data() + r * 16, 16))
				<< "SHAKE128 mismatch: msgLen=" << msgLen << " row=" << r;
		}
	}
}

// ─── SHAKE-256 extended output (multi-squeeze) vs CPU ──────────────────────

TEST_F(OaVkEngineTestFixture, Shake256LongOutputMatchesCpu) {
	const OaU32 kRows = 3;
	const OaU32 msgLen = 50;
	const OaU32 outLen = 64;
	auto msgs = Lcg(0x5EED1U, kRows * msgLen);
	OaMatrix in = UploadBytes(msgs, kRows, msgLen);
	OaMatrix out = OaFnHash::Shake256(in, outLen);
	SyncCtx();
	auto host = ReadBack(out, kRows * outLen);
	for (OaU32 r = 0; r < kRows; ++r) {
		OaByte ref[64];
		OaShake256(msgs.data() + r * msgLen, msgLen, ref, outLen);
		EXPECT_EQ(0, std::memcmp(ref, host.data() + r * outLen, outLen))
			<< "SHAKE256 long-output mismatch: row=" << r;
	}
}

// ─── Keccak-f[1600] batch permutation vs CPU ───────────────────────────────

TEST_F(OaVkEngineTestFixture, KeccakF1600BatchMatchesCpu) {
	const OaU32 kRows = 5;
	auto states = Lcg(0xBEEF01U, kRows * 200);
	OaMatrix in = UploadBytes(states, kRows, 200);
	OaMatrix out = OaFnHash::KeccakF1600(in);
	SyncCtx();
	auto host = ReadBack(out, kRows * 200);
	for (OaU32 r = 0; r < kRows; ++r) {
		OaU64 s[25];
		std::memcpy(s, states.data() + r * 200, 200);
		OaKeccakF1600(s);
		EXPECT_EQ(0, std::memcmp(s, host.data() + r * 200, 200))
			<< "Keccak-f[1600] mismatch: row=" << r;
	}
}

// ─── Merkle root (power-of-two) vs CPU known root ──────────────────────────

TEST_F(OaVkEngineTestFixture, MerkleRootMatchesCpu) {
	for (OaU32 n : {OaU32{2}, OaU32{4}, OaU32{8}, OaU32{16}, OaU32{64}}) {
		auto leafBytes = Lcg(0xACE00U ^ n, n * 32);
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
		EXPECT_EQ(0, std::memcmp(cpu.Bytes.data(), host.data(), 32))
			<< "Merkle root mismatch: n=" << n;
	}
}

// ─── Merkle root rejects non-power-of-two (honest constraint) ──────────────

TEST_F(OaVkEngineTestFixture, MerkleRootRejectsNonPowerOfTwo) {
	auto leafBytes = Lcg(0x333U, 3 * 32);
	OaMatrix in = UploadBytes(leafBytes, 3, 32);
	OaMatrix root = OaFnHash::MerkleRoot(in);
	EXPECT_EQ(root.NumElements(), 0) << "non-pow2 leaf count must return empty";
}

// ─── Single-leaf Merkle root is the leaf itself ────────────────────────────

TEST_F(OaVkEngineTestFixture, MerkleRootSingleLeaf) {
	auto leafBytes = Lcg(0x111U, 32);
	OaMatrix in = UploadBytes(leafBytes, 1, 32);
	OaMatrix root = OaFnHash::MerkleRoot(in);
	SyncCtx();
	auto host = ReadBack(root, 32);
	EXPECT_EQ(0, std::memcmp(leafBytes.data(), host.data(), 32));
}

TEST_F(OaVkEngineTestFixture, RejectsNonByteMatrices) {
	OaMatrix shakeInput = OaFnMatrix::Full({2, 32}, 1.0F, OaScalarType::Float32);
	OaMatrix stateInput = OaFnMatrix::Full({2, 200}, 0.0F, OaScalarType::Float32);
	OaMatrix leafInput = OaFnMatrix::Full({2, 32}, 0.0F, OaScalarType::Float32);
	EXPECT_TRUE(OaFnHash::Shake256(shakeInput, 32).IsEmpty());
	EXPECT_TRUE(OaFnHash::KeccakF1600(stateInput).IsEmpty());
	EXPECT_TRUE(OaFnHash::MerkleRoot(leafInput).IsEmpty());
}

} // namespace
