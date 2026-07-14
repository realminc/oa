// oa::modular — Integer modular arithmetic for Slang shaders
//
// Import: `import Modular;` (slangc -I to Core/Math or installed lib/Oa/Core/shaders/Math)
// Used by ML-DSA NTT and polynomial operations.
// Pure math — no bindings, no push constants, no entry points.
//
// Ring: Z_q[X] / (X^256 + 1), q = 8380417 = 2^23 - 2^13 + 1

module Modular;

public static const int Q = 8380417;
public static const int QINV = 58728449;

// Conditional add of q: if a < 0, a += q.
public int Caddq(int a) {
	a += (a >> 31) & Q;
	return a;
}

// Freeze to [0, q-1]: subtract q, then conditional add.
public int Freeze(int a) {
	a = a - Q;
	a += (a >> 31) & Q;
	return a;
}

// Upper 32 bits of unsigned 32x32 multiply, via 16-bit decomposition.
// Workaround for Vulkan/SPIR-V lacking native 64-bit int.
public uint Umulhi(uint a, uint b) {
	uint a0 = a & 0xFFFFu;
	uint a1 = a >> 16;
	uint b0 = b & 0xFFFFu;
	uint b1 = b >> 16;
	uint p00 = a0 * b0;
	uint p01 = a0 * b1;
	uint p10 = a1 * b0;
	uint p11 = a1 * b1;
	uint mid = (p00 >> 16) + (p01 & 0xFFFFu) + (p10 & 0xFFFFu);
	return p11 + (p01 >> 16) + (p10 >> 16) + (mid >> 16);
}

// Upper 32 bits of signed 32x32 multiply.
public int Imulhi(int a, int b) {
	uint h = Umulhi(uint(a), uint(b));
	if (a < 0) h -= uint(b);
	if (b < 0) h -= uint(a);
	return int(h);
}

// Montgomery reduction using pure 32-bit math.
// Input: 64-bit value as (lo, hi) where value = hi * 2^32 + lo.
// Returns (value - t*Q) >> 32 where t = lo * QINV mod 2^32.
public int MontgomeryReduce32(uint a_lo, int a_hi) {
	int t = int(a_lo) * QINV;
	return a_hi - Imulhi(t, Q);
}
