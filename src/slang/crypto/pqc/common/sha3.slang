// Checked SHA3-224/256/384/512 helpers over the verified Keccak-f[1600] permutation.
//
// Reuses keccak.slang without copying the permutation body.  Domain
// separation follows FIPS 202 §6.1: SHA3 uses padding byte 0x06 (distinct
// from SHAKE's 0x1f used in shake256.slang).
//
// FIPS 202 rates: 144/136/104/72 bytes; digests: 28/32/48/64 bytes.
// Byte-oriented messages only. Physical packed storage must contain complete words.
// Rejected ranges leave state and digest untouched; these helpers are not admitted
// public operations, GPU execution evidence or a constant-time qualification.
//
// These are single-call (non-incremental) device helper functions, not
// standalone entry-point shaders.  Entry-point shaders import this module.
//
// Donor reference: FIPS 202 §6.1; no OA donor shader body is reused.
// Adaptation class: new implementation using existing permutation import.

import keccak;
import storage;

static const uint SHA3_224_RATE_BYTES = 144u;
static const uint SHA3_256_RATE_BYTES = 136u;
static const uint SHA3_384_RATE_BYTES = 104u;
static const uint SHA3_512_RATE_BYTES = 72u;
static const uint SHA3_DOMAIN = 0x06u;

// Word fast path is retained for aligned messages. Unaligned transport uses the
// canonical packed-byte accessor, never an unaligned ByteAddressBuffer Load.
uint sha3_load_word(RWByteAddressBuffer buf, uint address) {
	if ((address & 3u) == 0u) return buf.Load(address);
	return load_u8(buf, address)
		| (load_u8(buf, address + 1u) << 8u)
		| (load_u8(buf, address + 2u) << 16u)
		| (load_u8(buf, address + 3u) << 24u);
}

// ---------------------------------------------------------------------------
// Absorb `msg_len` bytes from `buf` at `msg_base` into a sponge state using
// `rate_bytes` rate.  Pads with SHA3 domain byte 0x06 and multi-rate padding.
// The state is partially initialized by the caller (zeroed for fresh use).
// ---------------------------------------------------------------------------
bool sha3_absorb(
	inout uint2 state[25],
	RWByteAddressBuffer buf,
	uint msg_base,
	uint msg_len,
	uint rate_bytes
) {
	// Validate before touching the caller's state or reading message bytes.
	if (rate_bytes != SHA3_224_RATE_BYTES && rate_bytes != SHA3_256_RATE_BYTES
		&& rate_bytes != SHA3_384_RATE_BYTES && rate_bytes != SHA3_512_RATE_BYTES) return false;
	uint extent;
	buf.GetDimensions(extent);
	if ((extent & 3u) != 0u || msg_base > extent || msg_len > extent - msg_base) return false;
	uint rate_lanes = rate_bytes / 8u;
	uint remaining = msg_len;
	uint offset = 0u;

	// Process full blocks.
	while (remaining >= rate_bytes) {
		for (uint lane = 0u; lane < rate_lanes; ++lane) {
			uint addr = msg_base + offset + lane * 8u;
			uint lo = sha3_load_word(buf, addr);
			uint hi = sha3_load_word(buf, addr + 4u);
			state[lane].x ^= lo;
			state[lane].y ^= hi;
		}
		keccak_f1600_u32(state);
		offset += rate_bytes;
		remaining -= rate_bytes;
	}

	// Absorb remaining bytes with padding.
	// Build the final padded block byte-by-byte into a uint2[rate_lanes] array.
	uint2 padded[18]; // maximum rate: SHA3-224
	for (uint lane = 0u; lane < 18u; ++lane) padded[lane] = uint2(0u, 0u);

	// Copy remaining message bytes.
	for (uint b = 0u; b < remaining; ++b) {
		uint addr = msg_base + offset + b;
		uint byte = load_u8(buf, addr);
		uint lane = b / 8u;
		uint boff = b % 8u;
		if (boff < 4u) padded[lane].x |= (byte << (boff * 8u));
		else padded[lane].y |= (byte << ((boff - 4u) * 8u));
	}
	// Domain separation byte at position `remaining`.
	{
		uint lane = remaining / 8u;
		uint boff = remaining % 8u;
		if (boff < 4u) padded[lane].x |= (SHA3_DOMAIN << (boff * 8u));
		else padded[lane].y |= (SHA3_DOMAIN << ((boff - 4u) * 8u));
	}
	// Multi-rate padding: set bit 7 of last byte of the rate block.
	{
		uint last_byte = rate_bytes - 1u;
		uint lane = last_byte / 8u;
		uint boff = last_byte % 8u;
		if (boff < 4u) padded[lane].x |= (0x80u << (boff * 8u));
		else padded[lane].y |= (0x80u << ((boff - 4u) * 8u));
	}
	// XOR into state and apply final permutation.
	for (uint lane = 0u; lane < rate_lanes; ++lane) {
		state[lane].x ^= padded[lane].x;
		state[lane].y ^= padded[lane].y;
	}
	keccak_f1600_u32(state);
	return true;
}

// ---------------------------------------------------------------------------
// SHA3-256: hash `msg_len` bytes at `msg_base`, write 32 bytes to `out[0..3]`
// (as 4 uint2s = 8 uint32s = 32 bytes, little-endian).
// ---------------------------------------------------------------------------
bool sha3_256(
	RWByteAddressBuffer buf,
	uint msg_base,
	uint msg_len,
	inout uint2 digest[4]
) {
	uint2 state[25];
	for (uint i = 0u; i < 25u; ++i) state[i] = uint2(0u, 0u);
	if (!sha3_absorb(state, buf, msg_base, msg_len, SHA3_256_RATE_BYTES)) return false;
	for (uint lane = 0u; lane < 4u; ++lane) digest[lane] = state[lane];
	return true;
}

// ---------------------------------------------------------------------------
// SHA3-512: hash `msg_len` bytes at `msg_base`, write 64 bytes to `out[0..7]`
// (as 8 uint2s = 16 uint32s = 64 bytes, little-endian).
// ---------------------------------------------------------------------------
bool sha3_512(
	RWByteAddressBuffer buf,
	uint msg_base,
	uint msg_len,
	inout uint2 digest[8]
) {
	uint2 state[25];
	for (uint i = 0u; i < 25u; ++i) state[i] = uint2(0u, 0u);
	if (!sha3_absorb(state, buf, msg_base, msg_len, SHA3_512_RATE_BYTES)) return false;
	for (uint lane = 0u; lane < 8u; ++lane) digest[lane] = state[lane];
	return true;
}

// SHA3-224's final half-lane is exactly one word; no padded digest byte is output.
bool sha3_224(
	RWByteAddressBuffer buf,
	uint msg_base,
	uint msg_len,
	inout uint digest[7]
) {
	uint2 state[25];
	for (uint i = 0u; i < 25u; ++i) state[i] = uint2(0u, 0u);
	if (!sha3_absorb(state, buf, msg_base, msg_len, SHA3_224_RATE_BYTES)) return false;
	for (uint word = 0u; word < 7u; ++word) {
		digest[word] = (word & 1u) == 0u ? state[word / 2u].x : state[word / 2u].y;
	}
	return true;
}

bool sha3_384(
	RWByteAddressBuffer buf,
	uint msg_base,
	uint msg_len,
	inout uint2 digest[6]
) {
	uint2 state[25];
	for (uint i = 0u; i < 25u; ++i) state[i] = uint2(0u, 0u);
	if (!sha3_absorb(state, buf, msg_base, msg_len, SHA3_384_RATE_BYTES)) return false;
	for (uint lane = 0u; lane < 6u; ++lane) digest[lane] = state[lane];
	return true;
}

// ---------------------------------------------------------------------------
// Write a uint2 array as bytes into a local byte array (for downstream use).
// Callers that need a flat [u8; 32] or [u8; 64] can use this helper.
// ---------------------------------------------------------------------------
void digest_to_bytes_32(uint2 digest[4], out uint bytes[32]) {
	for (uint lane = 0u; lane < 4u; ++lane) {
		bytes[lane * 8u + 0u] = (digest[lane].x) & 0xFFu;
		bytes[lane * 8u + 1u] = (digest[lane].x >>  8u) & 0xFFu;
		bytes[lane * 8u + 2u] = (digest[lane].x >> 16u) & 0xFFu;
		bytes[lane * 8u + 3u] = (digest[lane].x >> 24u) & 0xFFu;
		bytes[lane * 8u + 4u] = (digest[lane].y      ) & 0xFFu;
		bytes[lane * 8u + 5u] = (digest[lane].y >>  8u) & 0xFFu;
		bytes[lane * 8u + 6u] = (digest[lane].y >> 16u) & 0xFFu;
		bytes[lane * 8u + 7u] = (digest[lane].y >> 24u) & 0xFFu;
	}
}
