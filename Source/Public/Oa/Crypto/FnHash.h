// OaFnHash — Crypto module GPU hash operations (batch, auto-context).
//
// Follows the OaFnMatrix / OaFnAudio formula: plain free functions
//   OaMatrix Op(const OaMatrix&, params…)
// whose bodies record deferred GPU work to OaContext::GetDefault(). No engine,
// runtime, or graph parameter appears anywhere. Flush with
// OaContext::GetDefault().Execute()/Sync() (or an OaContext scope).
//
// These operate on byte tensors (OaScalarType::UInt8) and run the Vulkan
// Keccak/SHAKE/Merkle kernels. The CPU primitives in Crypto/Keccak.h
// (OaShake256, OaShake128, OaKeccakF1600, OaKmac256, incremental OaHasher) and
// Crypto/Hash.h (OaMerkleRoot, OaVerifyMerkleProof) remain the single-shot API
// and the numerical reference these ops are validated against.
//
// Implementations: hand-written in Hash/FnHash.cpp (body = manual_context).

#pragma once

#include <Oa/Core/FnMatrix.h>

namespace OaFnHash {

// SHAKE-256 batch XOF. Input [N, MsgLen] bytes; output [N, ceil(OutLen/8)*8]
// bytes (32-byte default). Row i = SHAKE256(message i).
[[nodiscard]] OaMatrix Shake256(const OaMatrix& InA, OaU32 InOutputLength = 32);

// SHAKE-128 batch XOF. Input [N, MsgLen] bytes; output [N, ceil(OutLen/8)*8]
// bytes (16-byte default). Row i = SHAKE128(message i).
[[nodiscard]] OaMatrix Shake128(const OaMatrix& InA, OaU32 InOutputLength = 16);

// Batch Keccak-f[1600] permutation. Input/output [N, 200] bytes (25 lanes ×
// u64 per state). Out-of-place.
[[nodiscard]] OaMatrix KeccakF1600(const OaMatrix& InA);

// Merkle root by GPU SHAKE-256 pair reduction. Input [N, 32] leaf hashes,
// N a power of two; output [1, 32] root. Bit-identical to OaMerkleRoot for
// power-of-two leaf counts. Use OaMerkleRoot (CPU) for arbitrary counts, and
// OaVerifyMerkleProof (CPU) for inclusion proofs.
[[nodiscard]] OaMatrix MerkleRoot(const OaMatrix& InA);

} // namespace OaFnHash
