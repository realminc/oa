// Philox.slang — Philox4x32-10 Counter-Based PRNG
//
// Pure library module (no entry points). Import this for GPU random number generation.
// Philox is a counter-based PRNG from D.E. Shaw Research.
// 
// Usage:
//   import Philox;
//   uint4 bits = PhiloxRng(counter, key);
//   float r = PhiloxToFloat01(bits.x);
//
// References:
//   - Salmon et al. "Parallel Random Numbers: As Easy as 1, 2, 3" (SC11)
//   - cuRAND documentation (Philox4x32-10 is the default device RNG)

module Philox;

// Philox4x32-10 constants
static const uint PHILOX_M4x32_0 = 0xD2511F53u;
static const uint PHILOX_M4x32_1 = 0xCD9E8D57u;
static const uint PHILOX_W32_0   = 0x9E3779B9u;  // Weyl constant (golden ratio)
static const uint PHILOX_W32_1   = 0xBB67AE85u;

// 64-bit multiply: returns (hi, lo) of a * b
public void MulHiLo(uint a, uint b, out uint hi, out uint lo) {
	uint64_t prod = uint64_t(a) * uint64_t(b);
	hi = uint(prod >> 32u);
	lo = uint(prod & 0xFFFFFFFFu);
}

// Single Philox round
public void PhiloxRound(inout uint4 state, inout uint2 key) {
	uint hi0, lo0, hi1, lo1;
	MulHiLo(PHILOX_M4x32_0, state.x, hi0, lo0);
	MulHiLo(PHILOX_M4x32_1, state.z, hi1, lo1);
	
	state.x = hi1 ^ state.y ^ key.x;
	state.y = lo1;
	state.z = hi0 ^ state.w ^ key.y;
	state.w = lo0;
	
	// Key expansion (Weyl sequence)
	key.x += PHILOX_W32_0;
	key.y += PHILOX_W32_1;
}

// Philox4x32-10: 10 rounds of the bijection
// Input: counter (4 x uint32), key (2 x uint32)
// Output: 4 x uint32 (128 bits of randomness)
public uint4 PhiloxRng(uint4 counter, uint2 key) {
	uint4 state = counter;
	uint2 k = key;
	
	[unroll] for (uint i = 0u; i < 10u; i++) {
		PhiloxRound(state, k);
	}
	
	return state;
}

// Convert uint32 to float in [0, 1)
// Uses 23 bits of mantissa precision (IEEE 754 single precision)
public float PhiloxToFloat01(uint bits) {
	// Map [0, 2^32) → [0, 1) by dividing by 2^32
	// Use 0x1.0p-32f = 1.0 / 4294967296.0
	return float(bits) * 2.3283064365386963e-10f;
}

// Box-Muller transform: convert two uniform [0,1) to two standard normals
// Input: u1, u2 in [0, 1)
// Output: (z0, z1) ~ N(0, 1)
public void BoxMuller(float u1, float u2, out float z0, out float z1) {
	float r = sqrt(-2.0f * log(u1 + 1e-10f));  // Add epsilon to avoid log(0)
	float theta = 6.28318530718f * u2;  // 2*pi
	z0 = r * cos(theta);
	z1 = r * sin(theta);
}

// Generate two standard normal variates from one Philox call
// Uses first two uint32 outputs for Box-Muller
public void PhiloxToNormal(uint4 bits, out float z0, out float z1) {
	float u1 = PhiloxToFloat01(bits.x);
	float u2 = PhiloxToFloat01(bits.y);
	BoxMuller(u1, u2, z0, z1);
}
