// FnHash.cpp — hand-written OaFnHash GPU implementations (auto-context).
//
// OaFnHash records batch hashing to OaContext::GetDefault() and runs on the
// Vulkan crypto kernels (Keccak / SHAKE / Merkle). It mirrors OaFnAudio and
// OaFnMatrix: plain `OaMatrix Op(const OaMatrix&, params…)` with no engine,
// runtime, or graph parameter — the body records deferred GPU work.
//
// The CPU primitives (OaShake256 / OaShake128 / OaHasher in Crypto/Keccak.h,
// OaMerkleRoot in Crypto/Hash.h) remain the single-shot API and the numerical
// reference: these GPU ops are validated byte-for-byte against them in
// Test/Crypto/Hash/TestFnHash.
//
// Layout convention: byte tensors (OaScalarType::UInt8). SHAKE input is
// [N, MsgLen]; output is [N, ceil(OutLen/8)*8]. KeccakF1600 state is
// [N, 200] (25 lanes × u64). Merkle leaves are [N, 32]. The crypto shaders
// address the heap as raw uint64, so on little-endian devices the GPU digest
// is bit-identical to the CPU sponge.

#include <Oa/Crypto/FnHash.h>

#include <Oa/Core/BufferAccess.h>
#include <Oa/Core/FnMatrix.h>
#include <Oa/Core/Log.h>
#include <Oa/Core/Matrix.h>
#include <Oa/Core/Types.h>
#include <Oa/Runtime/Context.h>

#include <limits>

namespace OaFnHash {

static OaU32 DivCeil(OaU32 InA, OaU32 InB) {
	return (InA / InB) + static_cast<OaU32>((InA % InB) != 0);
}

static bool IsByteMatrix(const OaMatrix& InA, const char* InName) {
	if (InA.GetDtype() == OaScalarType::UInt8) return true;
	OA_LOG_ERROR(OaLogComponent::Crypto,
		"OaFnHash::%s: expected UInt8 input", InName);
	return false;
}

// Batch SHAKE. Padding is applied by the shader while reading the original
// [N, MsgLen] byte matrix. This keeps a deferred GPU producer on the GPU: the
// old wrapper synchronously copied every input to the host, padded it, then
// uploaded it again before recording the actual hash dispatch.
static OaMatrix ShakeBatch(
	const OaMatrix& InA, OaU32 InRateBytes, OaU32 InOutputLength,
	const char* InKernel, const char* InName)
{
	if (InA.Rank() != 2) {
		OA_LOG_ERROR(OaLogComponent::Crypto,
			"OaFnHash::%s: expected a [N, MsgLen] byte matrix", InName);
		return {};
	}
	if (!IsByteMatrix(InA, InName)) return {};
	const OaI64 n = InA.Size(0);
	const OaI64 msgLen = InA.Size(1);
	if (n <= 0 || msgLen < 0) {
		OA_LOG_ERROR(OaLogComponent::Crypto,
			"OaFnHash::%s: empty input", InName);
		return {};
	}
	if (static_cast<OaU64>(n) > std::numeric_limits<OaU32>::max() ||
		static_cast<OaU64>(msgLen) > std::numeric_limits<OaU32>::max()) {
		OA_LOG_ERROR(OaLogComponent::Crypto,
			"OaFnHash::%s: dimensions exceed the shader ABI", InName);
		return {};
	}

	const OaU32 rate = InRateBytes;
	// pad10*1 always adds at least one byte, so the block count is msgLen/rate+1
	// whether or not msgLen is already a multiple of the rate.
	const OaU32 numBlocks = static_cast<OaU32>(msgLen) / rate + 1;
	OaMatrix in = InA;

	const OaU32 squeezeU64 = DivCeil(InOutputLength, 8U);
	OaMatrix out = OaFnMatrix::Empty(
		OaMatrixShape{n, static_cast<OaI64>(squeezeU64) * 8},
		OaScalarType::UInt8);

	struct {
		OaU32 Count;
		OaU32 MsgLen;
		OaU32 NumBlocks;
		OaU32 SqueezeU64;
	} push{.Count = static_cast<OaU32>(n),
	       .MsgLen = static_cast<OaU32>(msgLen),
	       .NumBlocks = numBlocks,
	       .SqueezeU64 = squeezeU64};
	OaBufferAccess access[] = {OaBufferAccess::Read, OaBufferAccess::Write};
	OaContext::GetDefault().Add(
		InKernel, {&in, &out}, access, &push, sizeof(push),
		DivCeil(static_cast<OaU32>(n), 256U), 1, 1);
	return out;
}

// SHAKE-256 batch XOF → [N, ceil(OutLen/8)*8] bytes (32-byte default).
OaMatrix Shake256(const OaMatrix& InA, OaU32 InOutputLength) {
	return ShakeBatch(InA, 136, InOutputLength ? InOutputLength : 32,
	                  "Shake/Shake256", "Shake256");
}

// SHAKE-128 batch XOF → [N, ceil(OutLen/8)*8] bytes (16-byte default).
OaMatrix Shake128(const OaMatrix& InA, OaU32 InOutputLength) {
	return ShakeBatch(InA, 168, InOutputLength ? InOutputLength : 16,
	                  "Shake/Shake128", "Shake128");
}

// Batch Keccak-f[1600] permutation. Input/output = [N, 200] bytes (25 lanes ×
// u64 per state). Out-of-place: the kernel reads in_idx, writes out_idx.
OaMatrix KeccakF1600(const OaMatrix& InA) {
	if (InA.Rank() != 2 || InA.Size(1) != 200) {
		OA_LOG_ERROR(OaLogComponent::Crypto,
			"OaFnHash::KeccakF1600: expected [N, 200] state bytes");
		return {};
	}
	if (!IsByteMatrix(InA, "KeccakF1600")) return {};
	const OaI64 n = InA.Size(0);
	if (n <= 0) return {};
	if (static_cast<OaU64>(n) > std::numeric_limits<OaU32>::max()) return {};

	OaMatrix a = InA;
	OaMatrix out = OaFnMatrix::Empty(InA.GetShape(), OaScalarType::UInt8);
	struct { OaU32 Count; } push{.Count = static_cast<OaU32>(n)};
	OaBufferAccess access[] = {OaBufferAccess::Read, OaBufferAccess::Write};
	OaContext::GetDefault().Add(
		"KeccakF1600", {&a, &out}, access, &push, sizeof(push),
		DivCeil(static_cast<OaU32>(n), 256U), 1, 1);
	return out;
}

// GPU Merkle root via iterative SHAKE-256 pair reduction (MerkleReduce). Each
// dispatch halves the level. Requires a power-of-two leaf count so every level
// is even and the result is bit-identical to the CPU OaMerkleRoot; arbitrary
// counts (with odd-level duplication) stay on the CPU path. Input = [N, 32]
// leaf hashes; output = [1, 32] root.
OaMatrix MerkleRoot(const OaMatrix& InA) {
	if (InA.Rank() != 2 || InA.Size(1) != 32) {
		OA_LOG_ERROR(OaLogComponent::Crypto,
			"OaFnHash::MerkleRoot: expected [N, 32] leaf hashes");
		return {};
	}
	if (!IsByteMatrix(InA, "MerkleRoot")) return {};
	OaI64 nodes = InA.Size(0);
	if (nodes <= 0) return {};
	if (static_cast<OaU64>(nodes) > std::numeric_limits<OaU32>::max()) return {};
	if ((nodes & (nodes - 1)) != 0) {
		OA_LOG_ERROR(OaLogComponent::Crypto,
			"OaFnHash::MerkleRoot: leaf count must be a power of two (got %lld); "
			"use OaMerkleRoot for arbitrary counts",
			static_cast<long long>(nodes));
		return {};
	}

	OaMatrix cur = InA;
	while (nodes > 1) {
		const OaI64 half = nodes / 2;
		OaMatrix next = OaFnMatrix::Empty(
			OaMatrixShape{half, 32}, OaScalarType::UInt8);
		struct {
			OaU32 Count;
			OaU32 HashBytes;
		} push{.Count = static_cast<OaU32>(half), .HashBytes = 32U};
		OaBufferAccess access[] = {OaBufferAccess::Read, OaBufferAccess::Write};
		OaContext::GetDefault().Add(
			"Merkle/MerkleReduce", {&cur, &next}, access, &push, sizeof(push),
			DivCeil(static_cast<OaU32>(half), 256U), 1, 1);
		cur = next;
		nodes = half;
	}
	return cur;
}

} // namespace OaFnHash
