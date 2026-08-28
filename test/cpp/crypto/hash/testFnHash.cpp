// TestFnHash — oa::FnHash GPU batch hashing vs the CPU primitives.
//
// The CPU primitives (oa::shake256/oa::shake128/oa::keccakF1600/oa::merkleRoot) are
// NIST-KAT-anchored in TestKeccak, so byte-for-byte GPU==CPU here transitively
// validates the vulkan Keccak/SHAKE/Merkle kernels against FIPS 202 vectors.
// Every op is a deferred auto-context dispatch: record → execute → Sync →
// CopyToHost → compare.

#include "../../oaTest.h"

#include <oa/crypto/fnHash.h>
#include <oa/crypto/hash.h>
#include <oa/crypto/keccak.h>
#include <oa/core/fnMatrix.h>
#include <oa/core/op.h>
#include <oa/runtime/executionSession.h>
#include <oa/runtime/engine.h>
#include <oa/runtime/semanticGraph.h>

#include <cstring>
#include <vector>

namespace {

void syncCtx() {
	auto& ctx = oa::ExecutionSession::getActive();
	ASSERT_TRUE(testSubmitAndWait(ctx).isOk());
}

// Deterministic pseudo-random bytes (LCG) — reproducible test data.
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

// ─── SHAKE-256 batch vs CPU ────────────────────────────────────────────────
// message lengths span < rate, == rate (2 blocks), and > rate to exercise the
// multi-block absorb loop and the pad10*1 tail placement.

TEST_F(VkEngineTestFixture, Shake256BatchMatchesCpu) {
	const oa::U32 kRows = 4;
	for (oa::U32 msgLen : {oa::U32{1}, oa::U32{100}, oa::U32{136}, oa::U32{200}, oa::U32{400}}) {
		auto msgs = lcg(0xC0FFEEU ^ msgLen, kRows * msgLen);
		oa::Matrix in = uploadBytes(msgs, kRows, msgLen);
		oa::Matrix out = oa::FnHash::shake256(in, 32);
		syncCtx();
		auto host = readBack(out, kRows * 32);
		for (oa::U32 r = 0; r < kRows; ++r) {
			oa::Byte ref[32];
			oa::shake256(msgs.data() + r * msgLen, msgLen, ref, 32);
			EXPECT_EQ(0, std::memcmp(ref, host.data() + r * 32, 32))
				<< "SHAKE256 mismatch: msgLen=" << msgLen << " row=" << r;
		}
	}
}

// ─── SHAKE-128 batch vs CPU (rate 168) ─────────────────────────────────────

TEST_F(VkEngineTestFixture, Shake128BatchMatchesCpu) {
	const oa::U32 kRows = 4;
	for (oa::U32 msgLen : {oa::U32{1}, oa::U32{168}, oa::U32{300}}) {
		auto msgs = lcg(0xBADA55U ^ msgLen, kRows * msgLen);
		oa::Matrix in = uploadBytes(msgs, kRows, msgLen);
		oa::Matrix out = oa::FnHash::shake128(in, 16);
		syncCtx();
		auto host = readBack(out, kRows * 16);
		for (oa::U32 r = 0; r < kRows; ++r) {
			oa::Byte ref[16];
			oa::shake128(msgs.data() + r * msgLen, msgLen, ref, 16);
			EXPECT_EQ(0, std::memcmp(ref, host.data() + r * 16, 16))
				<< "SHAKE128 mismatch: msgLen=" << msgLen << " row=" << r;
		}
	}
}

// ─── SHAKE-256 extended output (multi-squeeze) vs CPU ──────────────────────

TEST_F(VkEngineTestFixture, Shake256LongOutputMatchesCpu) {
	const oa::U32 kRows = 3;
	const oa::U32 msgLen = 50;
	const oa::U32 outLen = 64;
	auto msgs = lcg(0x5EED1U, kRows * msgLen);
	oa::Matrix in = uploadBytes(msgs, kRows, msgLen);
	oa::Matrix out = oa::FnHash::shake256(in, outLen);
	syncCtx();
	auto host = readBack(out, kRows * outLen);
	for (oa::U32 r = 0; r < kRows; ++r) {
		oa::Byte ref[64];
		oa::shake256(msgs.data() + r * msgLen, msgLen, ref, outLen);
		EXPECT_EQ(0, std::memcmp(ref, host.data() + r * outLen, outLen))
			<< "SHAKE256 long-output mismatch: row=" << r;
	}
}

// ─── Keccak-f[1600] batch permutation vs CPU ───────────────────────────────

TEST_F(VkEngineTestFixture, KeccakF1600BatchMatchesCpu) {
	const oa::U32 kRows = 5;
	auto states = lcg(0xBEEF01U, kRows * 200);
	oa::Matrix in = uploadBytes(states, kRows, 200);
	oa::Matrix out = oa::FnHash::keccakF1600(in);
	syncCtx();
	auto host = readBack(out, kRows * 200);
	for (oa::U32 r = 0; r < kRows; ++r) {
		oa::U64 s[25];
		std::memcpy(s, states.data() + r * 200, 200);
		oa::keccakF1600(s);
		EXPECT_EQ(0, std::memcmp(s, host.data() + r * 200, 200))
			<< "Keccak-f[1600] mismatch: row=" << r;
	}
}

// ─── Merkle root (power-of-two) vs CPU known root ──────────────────────────

TEST_F(VkEngineTestFixture, MerkleRootMatchesCpu) {
	for (oa::U32 n : {oa::U32{2}, oa::U32{4}, oa::U32{8}, oa::U32{16}, oa::U32{64}}) {
		auto leafBytes = lcg(0xACE00U ^ n, n * 32);
		oa::Matrix in = uploadBytes(leafBytes, n, 32);
		oa::Matrix root = oa::FnHash::merkleRoot(in);
		syncCtx();
		auto host = readBack(root, 32);

		oa::Vector<oa::Hash> leaves;
		leaves.reserve(n);
		for (oa::U32 i = 0; i < n; ++i) {
			oa::Hash h;
			std::memcpy(h.bytes.data(), leafBytes.data() + i * 32, 32);
			leaves.pushBack(h);
		}
		oa::Hash cpu = oa::merkleRoot(leaves);
		EXPECT_EQ(0, std::memcmp(cpu.bytes.data(), host.data(), 32))
			<< "Merkle root mismatch: n=" << n;
	}
}

// ─── Merkle root rejects non-power-of-two (honest constraint) ──────────────

TEST_F(VkEngineTestFixture, MerkleRootRejectsNonPowerOfTwo) {
	auto leafBytes = lcg(0x333U, 3 * 32);
	oa::Matrix in = uploadBytes(leafBytes, 3, 32);
	oa::Matrix root = oa::FnHash::merkleRoot(in);
	EXPECT_EQ(root.numElements(), 0) << "non-pow2 leaf count must return empty";
}

// ─── Single-leaf Merkle root is the leaf itself ────────────────────────────

TEST_F(VkEngineTestFixture, MerkleRootSingleLeaf) {
	auto leafBytes = lcg(0x111U, 32);
	oa::Matrix in = uploadBytes(leafBytes, 1, 32);
	oa::Matrix root = oa::FnHash::merkleRoot(in);
	syncCtx();
	auto host = readBack(root, 32);
	EXPECT_EQ(0, std::memcmp(leafBytes.data(), host.data(), 32));
}

TEST_F(VkEngineTestFixture, RejectsNonByteMatrices) {
	oa::Matrix shakeInput = oa::FnMatrix::full({2, 32}, 1.0F, oa::ScalarType::Float32);
	oa::Matrix stateInput = oa::FnMatrix::full({2, 200}, 0.0F, oa::ScalarType::Float32);
	oa::Matrix leafInput = oa::FnMatrix::full({2, 32}, 0.0F, oa::ScalarType::Float32);
	EXPECT_TRUE(oa::FnHash::shake256(shakeInput, 32).isEmpty());
	EXPECT_TRUE(oa::FnHash::keccakF1600(stateInput).isEmpty());
	EXPECT_TRUE(oa::FnHash::merkleRoot(leafInput).isEmpty());
}

TEST_F(VkEngineTestFixture, SchemaContractsOwnSemanticAndLoweringProvenance) {
	auto& ctx = oa::ExecutionSession::getActive();
	ctx.clear();

	const auto messages = uploadBytes(lcg(0x515151U, 2 * 32), 2, 32);
	const auto states = uploadBytes(lcg(0x616161U, 2 * 200), 2, 200);
	const auto leaves = uploadBytes(lcg(0x717171U, 4 * 32), 4, 32);
	ctx.clear();

	const auto shake = oa::FnHash::shake256(messages, 48);
	const auto keccak = oa::FnHash::keccakF1600(states);
	const auto root = oa::FnHash::merkleRoot(leaves);
	(void)shake;
	(void)keccak;
	(void)root;

	ASSERT_NE(ctx.semanticGraph(), nullptr);
	ASSERT_NE(ctx.graph(), nullptr);
	const auto& operations = ctx.semanticGraph()->operations();
	ASSERT_EQ(operations.size(), 3U);
	EXPECT_EQ(operations[0].name, oa::detail::opRegistry::FnHash::shake256.name);
	EXPECT_EQ(operations[0].contractHash,
		oa::detail::opRegistry::FnHash::shake256.hash);
	ASSERT_EQ(operations[0].attributes.size(), 1U);
	EXPECT_EQ(operations[0].attributes[0].unsignedInteger, 48U);
	EXPECT_EQ(operations[1].name,
		oa::detail::opRegistry::FnHash::keccakF1600.name);
	EXPECT_EQ(operations[1].contractHash,
		oa::detail::opRegistry::FnHash::keccakF1600.hash);
	EXPECT_EQ(operations[2].name,
		oa::detail::opRegistry::FnHash::merkleRoot.name);
	EXPECT_EQ(operations[2].contractHash,
		oa::detail::opRegistry::FnHash::merkleRoot.hash);

	const auto& nodes = ctx.graph()->nodes();
	ASSERT_EQ(nodes.size(), 4U);
	EXPECT_EQ(nodes[0].operation, oa::detail::opRegistry::FnHash::shake256.name);
	EXPECT_EQ(nodes[1].operation,
		oa::detail::opRegistry::FnHash::keccakF1600.name);
	EXPECT_EQ(nodes[2].operation,
		oa::detail::opRegistry::FnHash::merkleRoot.name);
	EXPECT_EQ(nodes[3].operation,
		oa::detail::opRegistry::FnHash::merkleRoot.name);
	for (const auto& node : nodes) {
		ASSERT_EQ(node.semanticOps.size(), 1U);
		EXPECT_EQ(node.opContractHash,
			operations[node.semanticOps[0]].contractHash);
	}
	const auto analysis = oa::analyzeSemanticLowering(
		*ctx.semanticGraph(), *ctx.graph());
	ASSERT_TRUE(analysis.isOk()) << analysis.getStatus().getMessage();
	EXPECT_EQ(analysis.getValue().directOpCount(), 2U);
	EXPECT_EQ(analysis.getValue().decomposedOpCount(), 1U);
	EXPECT_EQ(analysis.getValue().schemaOwnedNodeCount(), 4U);
	EXPECT_EQ(analysis.getValue().compatibilityNodeCount(), 0U);
	ctx.clear();
}

} // namespace
