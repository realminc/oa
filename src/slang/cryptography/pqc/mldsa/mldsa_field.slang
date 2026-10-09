// ML-DSA field arithmetic over q = 8380417 (FIPS 204).
//
// All arithmetic is 32-bit-only; shaderInt64 is not required.
// Ordinary residues and Montgomery residues are never interchangeable.
// mul32_wide implements exact 32x32 products using 16-bit limbs; reductions
// below use R=2^32 and canonicalize before further modular arithmetic.
//
// Authority: FIPS 204 modulus, Algorithms 35-40 and Appendix A. OA C++
// public ML-DSA delegates to liboqs; no qualified arithmetic shader donor exists.
//
// Adaptation class: new implementation — constants derived from the FIPS 204
// modulus; no donor shader body is reused.

/// FIPS 204 modulus: 2^23 − 2^13 + 1 = 8380417.
static const uint MLDSA_Q = 8380417u;
/// d = 13 (dropped bits in public-key rounding).
static const uint MLDSA_D = 13u;
/// 2^d = 8192.
static const uint MLDSA_D_FACTOR = 8192u;

// Montgomery parameters (R = 2^32, R^2 mod Q, -Q^{-1} mod 2^32):
//   R^2 mod Q = (2^32 mod Q)^2 mod Q.
//   2^32 mod 8380417 = 4294967296 mod 8380417 = 4193792.
//   4193792^2 mod 8380417 = 17587891339264 mod 8380417 = 2365951.
static const uint MLDSA_MONT_R2  = 2365951u;
//   -Q^{-1} mod 2^32: modinv(8380417, 2^32) = 58728449, so -(58728449) mod 2^32
//   = 4236238847.
static const uint MLDSA_QINV_NEG = 4236238847u;

// ---------------------------------------------------------------------------
// 32×32→64-bit multiply helpers using 16-bit halves (no shaderInt64 needed).
// Computes a*b as a uint2 (lo, hi).
// ---------------------------------------------------------------------------
uint2 mul32_wide(uint a, uint b) {
	uint a_lo = a & 0xFFFFu;
	uint a_hi = a >> 16u;
	uint b_lo = b & 0xFFFFu;
	uint b_hi = b >> 16u;

	uint lo_lo = a_lo * b_lo;                  // 0..32 bits
	uint lo_hi = a_lo * b_hi;                  // 16..48 bits
	uint hi_lo = a_hi * b_lo;                  // 16..48 bits
	uint hi_hi = a_hi * b_hi;                  // 32..64 bits

	uint mid   = (lo_lo >> 16u) + (lo_hi & 0xFFFFu) + (hi_lo & 0xFFFFu);
	uint carry = mid >> 16u;

	uint result_lo = (lo_lo & 0xFFFFu) | ((mid & 0xFFFFu) << 16u);
	uint result_hi = hi_hi + (lo_hi >> 16u) + (hi_lo >> 16u) + carry;
	return uint2(result_lo, result_hi);
}

// Return high 32 bits of a*b.
uint mulhi32(uint a, uint b) {
	return mul32_wide(a, b).y;
}

// ---------------------------------------------------------------------------
// Montgomery reduction: given t = a*b (as wide uint2, lo=t.x hi=t.y),
// Precondition: 0 <= t < Q*R. Return t*R^-1 mod Q, result in [0, 2Q).
// ---------------------------------------------------------------------------
uint mldsa_mont_reduce(uint2 t) {
	// m = t_lo * (-Q^{-1}) mod 2^32
	uint m       = t.x * MLDSA_QINV_NEG;
	// add m*Q to t; the low word cancels to zero, take the high word.
	uint2 mq     = mul32_wide(m, MLDSA_Q);
	// low-word carry
	uint c       = (uint)(((t.x + mq.x) < t.x) ? 1u : 0u);
	uint result  = t.y + mq.y + c;
	// result is in [0, 2Q).
	return result;
}

// ---------------------------------------------------------------------------
// Montgomery product: a*b*R^-1 mod Q, canonical result in [0, Q).
// Both inputs must be in [0, Q); their representation is caller-owned.
// ---------------------------------------------------------------------------
uint mldsa_mont_mul(uint a, uint b) {
	uint result = mldsa_mont_reduce(mul32_wide(a, b));
	return (result >= MLDSA_Q) ? result - MLDSA_Q : result;
}

// ---------------------------------------------------------------------------
// Convert integer in [0, Q) to canonical Montgomery form: a*R mod Q.
// ---------------------------------------------------------------------------
uint mldsa_to_mont(uint a) {
	return mldsa_mont_mul(a, MLDSA_MONT_R2);
}

// ---------------------------------------------------------------------------
// Convert Montgomery-form value to [0, Q): a*R^{-1} mod Q.
// ---------------------------------------------------------------------------
uint mldsa_from_mont(uint a) {
	uint t = mldsa_mont_reduce(uint2(a, 0u));
	return (t >= MLDSA_Q) ? t - MLDSA_Q : t;
}

// Ordinary modular product of two canonical residues; result in [0, Q).
uint mldsa_mul(uint a, uint b) {
	return mldsa_mont_mul(mldsa_to_mont(a), b);
}

// ---------------------------------------------------------------------------
// Reduce a in [0, 2Q) to [0, Q).
// ---------------------------------------------------------------------------
uint mldsa_reduce(uint a) {
	uint t = a - MLDSA_Q;
	return (a >= MLDSA_Q) ? t : a;
}

// ---------------------------------------------------------------------------
// Modular add/sub/negate in [0, Q).
// ---------------------------------------------------------------------------
uint mldsa_add(uint a, uint b) {
	return mldsa_reduce(a + b);
}

uint mldsa_sub(uint a, uint b) {
	return mldsa_reduce(a + MLDSA_Q - b);
}

uint mldsa_neg(uint a) {
	return (a == 0u) ? 0u : MLDSA_Q - a;
}

// ---------------------------------------------------------------------------
// Centered representative in (−Q/2, Q/2].
// Input must be in [0, Q); returns a signed int.
// ---------------------------------------------------------------------------
int mldsa_centered(uint a) {
	int s = (int)a;
	if (a > (MLDSA_Q >> 1u)) s -= (int)MLDSA_Q;
	return s;
}

// ---------------------------------------------------------------------------
// Power2Round(a, d): split a mod Q into (a1, a0) such that
//   a mod Q = a1 * 2^d + a0, a0 in (-2^(d-1), 2^(d-1)].
// Returns a1 in high bits, a0 (centered, as int) requires separate return.
// Per FIPS 204 Algorithm 35.
// ---------------------------------------------------------------------------
uint mldsa_power2round_hi(uint a) {
	// a1 = floor((a + 2^(d-1) - 1) / 2^d); a is canonical.
	return (a + (MLDSA_D_FACTOR >> 1u) - 1u) / MLDSA_D_FACTOR;
}

int mldsa_power2round_lo(uint a) {
	uint a1 = mldsa_power2round_hi(a);
	return (int)a - (int)(a1 * MLDSA_D_FACTOR);
}

// ---------------------------------------------------------------------------
// Decompose / HighBits / LowBits, FIPS 204 Algorithms 36-38.
// Inputs: a in [0,Q), gamma2 is (Q-1)/88 or (Q-1)/32.
// Returns (high, low), with the exceptional q-1 wrap preserved in BOTH parts.
// ---------------------------------------------------------------------------
int2 mldsa_decompose(uint a, uint gamma2) {
	uint r = a % (2u * gamma2);
	int low = (int)r;
	if (r > gamma2) low -= (int)(2u * gamma2);
	int adjusted = (int)a - low;
	if (adjusted == (int)(MLDSA_Q - 1u)) return int2(0, low - 1);
	return int2(adjusted / (int)(2u * gamma2), low);
}

uint mldsa_highbits(uint a, uint gamma2) {
	return (uint)mldsa_decompose(a, gamma2).x;
}

int mldsa_lowbits(uint a, uint gamma2) {
	return mldsa_decompose(a, gamma2).y;
}

// MakeHint(z,r), FIPS 204 Algorithm 39. Inputs are canonical residues.
uint mldsa_make_hint(uint z, uint r, uint gamma2) {
	return (mldsa_highbits(r, gamma2) != mldsa_highbits(mldsa_add(r, z), gamma2)) ? 1u : 0u;
}

// UseHint(h,r), FIPS 204 Algorithm 40. hint is 0 or 1.
uint mldsa_use_hint(uint hint, uint a, uint gamma2) {
	uint m = (MLDSA_Q - 1u) / (2u * gamma2);
	int2 decomposed = mldsa_decompose(a, gamma2);
	uint high = (uint)decomposed.x;
	if (hint == 0u) return high;
	if (decomposed.y > 0) return (high + 1u) % m;
	return (high + m - 1u) % m;
}
