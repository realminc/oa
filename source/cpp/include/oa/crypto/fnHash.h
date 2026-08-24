// oa::FnHash — Crypto module GPU hash operations (batch, active engine).
//
// Follows the oa::FnMatrix / oa::FnAudio stateless-operation formula
//   oa::Matrix op(const oa::Matrix&, params…)
// whose bodies record deferred GPU work through the active engine's private
// recorder. No engine, runtime, or graph parameter appears in the operation
// signature. Host-reading boundaries finish recorded work internally; callers
// that need an explicit completion use oa::Engine::submit() and
// oa::Engine::wait(event).
//
// These operate on byte tensors (oa::ScalarType::UInt8) and run the vulkan
// Keccak/SHAKE/Merkle kernels. The CPU primitives in crypto/keccak.h and
// crypto/hash.h remain the single-shot API and the numerical reference these
// ops are validated against.

#pragma once

#include <oa/core/fnMatrix.h>

namespace oa {

namespace FnHash {
	// SHAKE-256 batch XOF. input [N, msgLen] bytes; output [N, ceil(outLen/8)*8]
	// bytes (32-byte default). Row i = SHAKE256(message i).
	[[nodiscard]] Matrix shake256(const Matrix& inA, oa::U32 inOutputLength = 32);

	// SHAKE-128 batch XOF. input [N, msgLen] bytes; output [N, ceil(outLen/8)*8]
	// bytes (16-byte default). Row i = SHAKE128(message i).
	[[nodiscard]] Matrix shake128(const Matrix& inA, oa::U32 inOutputLength = 16);

	// Batch Keccak-f[1600] permutation. input/output [N, 200] bytes (25 lanes ×
	// u64 per state). out-of-place.
	[[nodiscard]] Matrix keccakF1600(const Matrix& inA);

	// Merkle root by GPU SHAKE-256 pair reduction. input [N, 32] leaf hashes,
	// N a power of two; output [1, 32] root. Bit-identical to oa::merkleRoot for
	// power-of-two leaf counts. Use oa::merkleRoot (CPU) for arbitrary counts, and
	// oa::verifyMerkleProof (CPU) for inclusion proofs.
	[[nodiscard]] Matrix merkleRoot(const Matrix& inA);

} // namespace FnHash

} // namespace oa
