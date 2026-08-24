// fnHash.cpp — hand-written oa::FnHash GPU implementations (auto-context).
//
// oa::FnHash records batch hashing to oa::ExecutionSession::getActive() and runs on the
// vulkan crypto kernels (Keccak / SHAKE / Merkle). It mirrors oa::FnAudio and
// oa::FnMatrix: plain `oa::Matrix op(const oa::Matrix&, params…)` with no engine,
// runtime, or graph parameter — the body records deferred GPU work.
//
// The CPU primitives (oa::shake256 / oa::shake128 / oa::Hasher in crypto/keccak.h,
// oa::merkleRoot in crypto/hash.h) remain the single-shot API and the numerical
// reference: these GPU ops are validated byte-for-byte against them in
// Test/Crypto/Hash/TestFnHash.
//
// layout convention: byte tensors (oa::ScalarType::UInt8). SHAKE input is
// [N, msgLen]; output is [N, ceil(outLen/8)*8]. KeccakF1600 state is
// [N, 200] (25 lanes × u64). Merkle leaves are [N, 32]. The crypto shaders
// address the heap as raw uint64, so on little-endian devices the GPU digest
// is bit-identical to the CPU sponge.

#include <oa/crypto/fnHash.h>

#include <oa/core/bufferAccess.h>
#include <oa/core/fnMatrix.h>
#include <oa/core/log.h>
#include <oa/core/matrix.h>
#include <oa/core/op.h>
#include <oa/core/types.h>
#include <oa/runtime/executionSession.h>

#include <limits>

namespace oa {

namespace FnHash {

static oa::U32 divCeil(oa::U32 inA, oa::U32 inB) {
	return (inA / inB) + static_cast<oa::U32>((inA % inB) != 0);
}

static bool isByteMatrix(const oa::Matrix& inA, const char* inName) {
	if (inA.getDtype() == oa::ScalarType::UInt8) return true;
	OaLogError(oa::LogComponent::Crypto,
		"oa::FnHash::%s: expected UInt8 input", inName);
	return false;
}

// Batch SHAKE. Padding is applied by the shader while reading the original
// [N, msgLen] byte matrix. This keeps a deferred GPU producer on the GPU: the
// old wrapper synchronously copied every input to the host, padded it, then
// uploaded it again before recording the actual hash dispatch.
static oa::Matrix shakeBatch(
	const oa::Matrix& inA, oa::U32 inRateBytes, oa::U32 inOutputLength,
	const char* inKernel, const char* inName,
	const oa::OpContract& inContract)
{
	if (inA.rank() != 2) {
		OaLogError(oa::LogComponent::Crypto,
			"oa::FnHash::%s: expected a [N, msgLen] byte matrix", inName);
		return {};
	}
	if (!isByteMatrix(inA, inName)) return {};
	const oa::I64 n = inA.size(0);
	const oa::I64 msgLen = inA.size(1);
	if (n <= 0 || msgLen < 0) {
		OaLogError(oa::LogComponent::Crypto,
			"oa::FnHash::%s: empty input", inName);
		return {};
	}
	if (static_cast<oa::U64>(n) > std::numeric_limits<oa::U32>::max() ||
		static_cast<oa::U64>(msgLen) > std::numeric_limits<oa::U32>::max()) {
		OaLogError(oa::LogComponent::Crypto,
			"oa::FnHash::%s: dimensions exceed the shader ABI", inName);
		return {};
	}

	const oa::U32 rate = inRateBytes;
	// pad10*1 always adds at least one byte, so the block count is msgLen/rate+1
	// whether or not msgLen is already a multiple of the rate.
	const oa::U32 numBlocks = static_cast<oa::U32>(msgLen) / rate + 1;
	oa::Matrix in = inA;

	const oa::U32 squeezeU64 = divCeil(inOutputLength, 8U);
	oa::Matrix out = oa::FnMatrix::empty(
		oa::MatrixShape{n, static_cast<oa::I64>(squeezeU64) * 8},
		oa::ScalarType::UInt8);

	struct {
		oa::U32 count;
		oa::U32 msgLen;
		oa::U32 numBlocks;
		oa::U32 squeezeU64;
	} push{.count = static_cast<oa::U32>(n),
	       .msgLen = static_cast<oa::U32>(msgLen),
	       .numBlocks = numBlocks,
	       .squeezeU64 = squeezeU64};
	auto& ctx = oa::ExecutionSession::getActive();
	const auto semantic = ctx.recordOp(
		inContract, {&inA}, {&out},
		{oa::OpAttribute::fromUnsignedInteger(
			"outputLength", inOutputLength)});
	if (not semantic.isOk()) return {};
	oa::BufferAccess access[] = {oa::BufferAccess::Read, oa::BufferAccess::Write};
	ctx.add(
		inKernel, {&in, &out}, access, &push, sizeof(push),
		divCeil(static_cast<oa::U32>(n), 256U), 1, 1,
		inContract.name, 0, inContract.hash, 0, 0, semantic.getValue());
	return out;
}

// SHAKE-256 batch XOF → [N, ceil(outLen/8)*8] bytes (32-byte default).
oa::Matrix shake256(const oa::Matrix& inA, oa::U32 inOutputLength) {
	const oa::U32 outputLength = inOutputLength ? inOutputLength : 32U;
	return shakeBatch(inA, 136, outputLength,
		"Shake/Shake256", "Shake256",
		oa::detail::opRegistry::FnHash::shake256);
}

// SHAKE-128 batch XOF → [N, ceil(outLen/8)*8] bytes (16-byte default).
oa::Matrix shake128(const oa::Matrix& inA, oa::U32 inOutputLength) {
	const oa::U32 outputLength = inOutputLength ? inOutputLength : 16U;
	return shakeBatch(inA, 168, outputLength,
		"Shake/Shake128", "Shake128",
		oa::detail::opRegistry::FnHash::shake128);
}

// Batch Keccak-f[1600] permutation. input/output = [N, 200] bytes (25 lanes ×
// u64 per state). out-of-place: the kernel reads in_idx, writes out_idx.
oa::Matrix keccakF1600(const oa::Matrix& inA) {
	if (inA.rank() != 2 || inA.size(1) != 200) {
		OaLogError(oa::LogComponent::Crypto,
			"oa::FnHash::keccakF1600: expected [N, 200] state bytes");
		return {};
	}
	if (!isByteMatrix(inA, "KeccakF1600")) return {};
	const oa::I64 n = inA.size(0);
	if (n <= 0) return {};
	if (static_cast<oa::U64>(n) > std::numeric_limits<oa::U32>::max()) return {};

	oa::Matrix a = inA;
	oa::Matrix out = oa::FnMatrix::empty(inA.getShape(), oa::ScalarType::UInt8);
	auto& ctx = oa::ExecutionSession::getActive();
	const auto semantic = ctx.recordOp(
		oa::detail::opRegistry::FnHash::keccakF1600, {&inA}, {&out});
	if (not semantic.isOk()) return {};
	struct { oa::U32 count; } push{.count = static_cast<oa::U32>(n)};
	oa::BufferAccess access[] = {oa::BufferAccess::Read, oa::BufferAccess::Write};
	ctx.add(
		"KeccakF1600", {&a, &out}, access, &push, sizeof(push),
		divCeil(static_cast<oa::U32>(n), 256U), 1, 1,
		oa::detail::opRegistry::FnHash::keccakF1600.name, 0,
		oa::detail::opRegistry::FnHash::keccakF1600.hash, 0, 0,
		semantic.getValue());
	return out;
}

// GPU Merkle root via iterative SHAKE-256 pair reduction (MerkleReduce). Each
// dispatch halves the level. Requires a power-of-two leaf count so every level
// is even and the result is bit-identical to the CPU oa::merkleRoot; arbitrary
// counts (with odd-level duplication) stay on the CPU path. input = [N, 32]
// leaf hashes; output = [1, 32] root.
oa::Matrix merkleRoot(const oa::Matrix& inA) {
	if (inA.rank() != 2 || inA.size(1) != 32) {
		OaLogError(oa::LogComponent::Crypto,
			"oa::FnHash::merkleRoot: expected [N, 32] leaf hashes");
		return {};
	}
	if (!isByteMatrix(inA, "MerkleRoot")) return {};
	oa::I64 nodes = inA.size(0);
	if (nodes <= 0) return {};
	if (static_cast<oa::U64>(nodes) > std::numeric_limits<oa::U32>::max()) return {};
	if ((nodes & (nodes - 1)) != 0) {
		OaLogError(oa::LogComponent::Crypto,
			"oa::FnHash::merkleRoot: leaf count must be a power of two (got %lld); "
			"use oa::merkleRoot for arbitrary counts",
			static_cast<long long>(nodes));
		return {};
	}

	auto& ctx = oa::ExecutionSession::getActive();
	oa::Matrix out = oa::FnMatrix::empty(
		oa::MatrixShape{1, 32}, oa::ScalarType::UInt8);
	const auto semantic = ctx.recordOp(
		oa::detail::opRegistry::FnHash::merkleRoot, {&inA}, {&out});
	if (not semantic.isOk()) return {};
	if (nodes == 1) {
		struct { oa::U32 Count; } push{32U};
		oa::BufferAccess access[] = {
			oa::BufferAccess::Read,
			oa::BufferAccess::Write,
		};
		ctx.add(
			"Copy", {&inA, &out}, access, &push, sizeof(push), 1, 1, 1,
			oa::detail::opRegistry::FnHash::merkleRoot.name, 0,
			oa::detail::opRegistry::FnHash::merkleRoot.hash, 0, 0,
			semantic.getValue());
		return out;
	}

	oa::Matrix cur = inA;
	while (nodes > 1) {
		const oa::I64 half = nodes / 2;
		oa::Matrix next = half == 1
			? out
			: oa::FnMatrix::empty(
				oa::MatrixShape{half, 32}, oa::ScalarType::UInt8);
		struct {
			oa::U32 count;
			oa::U32 hashBytes;
		} push{.count = static_cast<oa::U32>(half), .hashBytes = 32U};
		oa::BufferAccess access[] = {oa::BufferAccess::Read, oa::BufferAccess::Write};
		ctx.add(
			"Merkle/MerkleReduce", {&cur, &next}, access, &push, sizeof(push),
			divCeil(static_cast<oa::U32>(half), 256U), 1, 1,
			oa::detail::opRegistry::FnHash::merkleRoot.name, 0,
			oa::detail::opRegistry::FnHash::merkleRoot.hash, 0, 0,
			semantic.getValue());
		cur = next;
		nodes = half;
	}
	return out;
}

} // namespace FnHash

} // namespace oa
