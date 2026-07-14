// CPU Keccak-f[1600], SHAKE-128/256, KMAC-256.
//
// Pure C++ implementation — no external dependencies.
// Bit-exact with the GPU Keccak shader (KeccakF1600.slang).
// State: 25 x OaU64 lanes = 200 bytes, little-endian.
//
// SHAKE XOF: absorb arbitrary input, squeeze arbitrary output.
// KMAC-256: keyed MAC per NIST SP 800-185.

#pragma once

#include <Oa/Core/Types.h>
#include <Oa/Core/Status.h>

// Keccak-f[1600] permutation (24 rounds, in-place).
// State is 25 x OaU64 (200 bytes), little-endian lane ordering.
void OaKeccakF1600(OaU64* InOutState);

// One-shot SHAKE-128: hash InLen bytes into OutDigest of OutLen bytes.
void OaShake128(
	const OaByte* InData, OaUsize InLen,
	OaByte* OutDigest, OaUsize InOutLen
);

// One-shot SHAKE-256: hash InLen bytes into OutDigest of OutLen bytes.
void OaShake256(
	const OaByte* InData, OaUsize InLen,
	OaByte* OutDigest, OaUsize InOutLen
);

// Incremental sponge context — absorb in chunks, squeeze in chunks.
class OaShakeCtx {
public:
	OaU64 State[25];
	OaByte Buf[168]; // max rate = SHAKE-128 rate
	OaU32 BufLen;
	OaU32 Rate; // 168 for SHAKE-128, 136 for SHAKE-256
	OaBool Squeezing;
};

// Initialize SHAKE-128 context.
void OaShake128Init(OaShakeCtx& InOutCtx);

// Initialize SHAKE-256 context.
void OaShake256Init(OaShakeCtx& InOutCtx);

// Absorb InLen bytes. Can be called multiple times before squeeze.
void OaShakeAbsorb(OaShakeCtx& InOutCtx, const OaByte* InData, OaUsize InLen);

// Squeeze OutLen bytes. Can be called multiple times after absorb.
// First call finalizes the sponge (applies padding).
void OaShakeSqueeze(OaShakeCtx& InOutCtx, OaByte* OutData, OaUsize InOutLen);

// KMAC-256 (NIST SP 800-185) — keyed hash.
// Produces OutLen bytes of MAC in OutMac.
[[nodiscard]] OaStatus OaKmac256(
	const OaByte* InKey, OaUsize InKeyLen,
	const OaByte* InData, OaUsize InDataLen,
	const OaByte* InCustom, OaUsize InCustomLen,
	OaByte* OutMac, OaUsize InOutLen
);
