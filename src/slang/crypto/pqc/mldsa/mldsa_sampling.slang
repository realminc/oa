// ML-DSA polynomial sampling routines (FIPS 204).
//
// Implements the two sampling algorithms needed for verification:
//   ExpandA        — expand seed rho into matrix A (k×l) via SHAKE-128 XOF
//   SampleInBall   — expand challenge hash c_tilde into sparse challenge poly c
//
// Secret-polynomial sampling (RejBoundedPoly and ExpandMask) is not needed
// for verification and remains part of the key generation/signing work.
//
// All routines are pure device functions; no buffer bindings are assumed.
// Callers pass in pre-loaded seed bytes and receive coefficients in [0, Q).
//
// Donor reference: FIPS 204 Algorithms 29, 30, 32; no OA donor shader body
// is reused.  Adaptation class: new implementation.

import mldsa_field;
import keccak;

// ---------------------------------------------------------------------------
// Inline SHAKE-128 XOF for polynomial expansion (ExpandA).
//
// Generates a uniform polynomial over [0, Q) from a 34-byte seed
// (rho || col_index_byte || row_index_byte) using rejection sampling.
// FIPS 204 Algorithm 32 (RejNTTPoly).
//
// Continue squeezing until all 256 coefficients are accepted. ExpandA
// outputs NTT-domain evaluations directly; callers must not transform again.
// ---------------------------------------------------------------------------
static const uint SHAKE128_RATE_BYTES = 168u;
static const uint SHAKE128_RATE_LANES = 21u;   // 168/8
static const uint SHAKE_DOMAIN_BYTE   = 0x1Fu;

// Absorb a fixed-length seed (seed_len < 168 bytes, fits in one block)
// and return the initial squeezed state after one permutation.
void shake128_absorb_short(
	out uint2 xof_state[25],
	uint seed_bytes[168],
	uint seed_len
) {
	for (uint i = 0u; i < 25u; ++i) xof_state[i] = uint2(0u, 0u);

	// Absorb seed_len bytes with SHAKE-128 domain padding in one block.
	for (uint b = 0u; b < seed_len; ++b) {
		uint byte  = seed_bytes[b];
		uint lane  = b / 8u;
		uint boff  = b % 8u;
		if (boff < 4u) xof_state[lane].x ^= (byte << (boff * 8u));
		else           xof_state[lane].y ^= (byte << ((boff - 4u) * 8u));
	}
	// Domain byte at position seed_len.
	{
		uint lane = seed_len / 8u;
		uint boff = seed_len % 8u;
		if (boff < 4u) xof_state[lane].x ^= (SHAKE_DOMAIN_BYTE << (boff * 8u));
		else           xof_state[lane].y ^= (SHAKE_DOMAIN_BYTE << ((boff - 4u) * 8u));
	}
	// Multi-rate padding: set bit 7 of last byte of rate block.
	xof_state[20u].y ^= 0x80000000u;   // byte 167 = lane 20 byte 7 = .y bit 31
	keccak_f1600_u32(xof_state);
}

// Extract one byte from an XOF state at linear byte position pos (within one block).
uint xof_byte(uint2 xof_state[25], uint pos) {
	uint lane  = pos / 8u;
	uint boff  = pos % 8u;
	uint word  = (boff < 4u) ? xof_state[lane].x : xof_state[lane].y;
	uint shift = (boff < 4u) ? (boff * 8u) : ((boff - 4u) * 8u);
	return (word >> shift) & 0xFFu;
}

// FIPS 204 Algorithm 29. Challenge seeds are 32/48/64 bytes for 44/65/87.
// Rejection sampling has no silently truncating iteration cap. A subsequent
// dispatch admission must separately account for watchdog/execution limits.
static const uint MLDSA65_TAU = 49u;
// Caller supplies the FIPS parameter seed length (8/12/16 words) and tau (39/49/60).
void mldsa_sample_in_ball(uint c_tilde[16], uint seed_words, uint tau, out uint c_poly[256]) {
	for (uint i = 0u; i < 256u; ++i) c_poly[i] = 0u;
	uint2 state[25];
	for (uint i = 0u; i < 25u; ++i) state[i] = uint2(0u, 0u);
	for (uint w = 0u; w < seed_words; ++w) {
		if ((w & 1u) == 0u) state[w / 2u].x = c_tilde[w];
		else state[w / 2u].y = c_tilde[w];
	}
	state[seed_words / 2u].x ^= 0x1Fu; // domain byte after the challenge seed
	state[16u].y ^= 0x80000000u;
	keccak_f1600_u32(state);
	uint sign_lo = state[0].x, sign_hi = state[0].y;
	uint pos = 8u;
	for (uint i = 256u - tau; i < 256u; ++i) {
		uint j;
		do {
			if (pos == 136u) {
				keccak_f1600_u32(state);
				pos = 0u;
			}
			j = xof_byte(state, pos++);
		} while (j > i);
		c_poly[i] = c_poly[j];
		uint bit = i + tau - 256u;
		uint sign = bit < 32u ? ((sign_lo >> bit) & 1u)
							 : ((sign_hi >> (bit - 32u)) & 1u);
		c_poly[j] = sign != 0u ? MLDSA_Q - 1u : 1u;
	}
}

void mldsa65_sample_in_ball(uint c_tilde[12], out uint c_poly[256]) {
	uint seed[16];
	for (uint w = 0u; w < 16u; ++w) seed[w] = w < 12u ? c_tilde[w] : 0u;
	mldsa_sample_in_ball(seed, 12u, MLDSA65_TAU, c_poly);
}

// ---------------------------------------------------------------------------
// ExpandA — expand seed rho into the k×l matrix A (FIPS 204 Algorithm 34).
//
// For each (row, col), sample a polynomial from SHAKE-128(rho || col || row).
// Returns one polynomial at a time; callers loop over k*l pairs.
//
// ML-DSA-65: k=6, l=5, so 30 NTT-domain polynomials total.
// ---------------------------------------------------------------------------
void mldsa_expand_a_poly(
	uint rho[8],       // seed: 8 uint32s = 32 bytes
	uint row,          // 0..k-1
	uint col,          // 0..l-1
	out uint poly[256] // output polynomial in [0, Q)
) {
	// Build 34-byte seed: rho (32 bytes) || col_byte || row_byte.
	uint seed_bytes[168];
	for (uint i = 0u; i < 168u; ++i) seed_bytes[i] = 0u;
	// Unpack rho uint32s to bytes.
	for (uint w = 0u; w < 8u; ++w) {
		seed_bytes[w * 4u + 0u] = (rho[w]      ) & 0xFFu;
		seed_bytes[w * 4u + 1u] = (rho[w] >>  8u) & 0xFFu;
		seed_bytes[w * 4u + 2u] = (rho[w] >> 16u) & 0xFFu;
		seed_bytes[w * 4u + 3u] = (rho[w] >> 24u) & 0xFFu;
	}
	seed_bytes[32u] = col & 0xFFu;
	seed_bytes[33u] = row & 0xFFu;

	uint2 xof_state[25];
	shake128_absorb_short(xof_state, seed_bytes, 34u);

	// Rejection sample: draw 3-byte samples, accept if < Q.
	uint count    = 0u;
	uint pos      = 0u;
	uint2 block[25];
	for (uint i = 0u; i < 25u; ++i) block[i] = xof_state[i];

	while (count < 256u) {
		if (pos + 3u > SHAKE128_RATE_BYTES) {
			keccak_f1600_u32(block);
			pos = 0u;
		}
		uint b0 = xof_byte(block, pos    );
		uint b1 = xof_byte(block, pos + 1u);
		uint b2 = xof_byte(block, pos + 2u) & 0x7Fu; // top bit masked
		pos += 3u;
		uint coeff = b0 | (b1 << 8u) | (b2 << 16u);
		if (coeff < MLDSA_Q) {
			poly[count] = coeff;
			count++;
		}
	}
}
