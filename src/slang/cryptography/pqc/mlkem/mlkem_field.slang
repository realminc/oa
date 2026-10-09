// ML-KEM ordinary canonical residues, q=3329, FIPS 203 Section 4.2.1.
// Adaptation: new standards-derived implementation. OA C++ has no live
// ML-KEM arithmetic donor, as recorded by the OA C++ donor audit.
// These are private public-data laboratory formulas, not admitted secret ops.
// Products are at most 3328^2=11075584; no shaderInt64 or Montgomery scale.

static const uint MLKEM_Q = 3329u;

// All inputs and outputs are in [0,q).
uint mlkem_add(uint a, uint b) {
	uint sum = a + b;
	return sum >= MLKEM_Q ? sum - MLKEM_Q : sum;
}

uint mlkem_sub(uint a, uint b) {
	uint difference = a + MLKEM_Q - b;
	return difference >= MLKEM_Q ? difference - MLKEM_Q : difference;
}

uint mlkem_neg(uint a) {
	return a == 0u ? 0u : MLKEM_Q - a;
}

uint mlkem_mul(uint a, uint b) {
	return (a * b) % MLKEM_Q;
}

// Centered representative; q is odd, so the endpoints have no tie.
int mlkem_centered(uint a) {
	return a > MLKEM_Q / 2u ? (int)a - (int)MLKEM_Q : (int)a;
}
