// CPU Keccak-f[1600], SHAKE-128/256, KMAC-256.
//
// Pure C++ implementation — no external dependencies.
// Bit-exact with the GPU Keccak shader (KeccakF1600.slang).
// State: 25 x oa::U64 lanes = 200 bytes, little-endian.
//
// SHAKE XOF: absorb arbitrary input, squeeze arbitrary output.
// KMAC-256: keyed MAC per NIST SP 800-185.

#pragma once

#include <oa/core/types.h>
#include <oa/core/status.h>

namespace oa {

// Keccak-f[1600] permutation (24 rounds, in-place).
// State is 25 x oa::U64 (200 bytes), little-endian lane ordering.
void keccakF1600(oa::U64* inOutState);

// One-shot SHAKE-128: hash inLen bytes into outDigest of outLen bytes.
void shake128(
	const oa::Byte* inData, oa::Usize inLen,
	oa::Byte* outDigest, oa::Usize inOutLen
);

// One-shot SHAKE-256: hash inLen bytes into outDigest of outLen bytes.
void shake256(
	const oa::Byte* inData, oa::Usize inLen,
	oa::Byte* outDigest, oa::Usize inOutLen
);

// Incremental sponge context — absorb in chunks, squeeze in chunks.
class ShakeContext {
public:
	oa::U64 state[25];
	oa::Byte buf[168]; // max rate = SHAKE-128 rate
	oa::U32 bufLen;
	oa::U32 rate; // 168 for SHAKE-128, 136 for SHAKE-256
	oa::Bool squeezing;
};

// initialize SHAKE-128 context.
void shake128Init(ShakeContext& inOutCtx);

// initialize SHAKE-256 context.
void shake256Init(ShakeContext& inOutCtx);

// Absorb inLen bytes. Can be called multiple times before squeeze.
void shakeAbsorb(ShakeContext& inOutCtx, const oa::Byte* inData, oa::Usize inLen);

// squeeze outLen bytes. Can be called multiple times after absorb.
// first call finalizes the sponge (applies padding).
void shakeSqueeze(ShakeContext& inOutCtx, oa::Byte* outData, oa::Usize inOutLen);

// KMAC-256 (NIST SP 800-185) — keyed hash.
// Produces outLen bytes of MAC in outMac.
[[nodiscard]] oa::Status kmac256(
	const oa::Byte* inKey, oa::Usize inKeyLen,
	const oa::Byte* inData, oa::Usize inDataLen,
	const oa::Byte* inCustom, oa::Usize inCustomLen,
	oa::Byte* outMac, oa::Usize inOutLen
);

} // namespace oa
