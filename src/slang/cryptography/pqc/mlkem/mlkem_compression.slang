// FIPS 203 Equations 4.7-4.8: integer-only nearest rounding, ties upward.
// Adaptation: new standards-derived implementation; no OA C++ ML-KEM donor.
// Public-data laboratory only; no machine-code constant-time claim.
import mlkem_field;

// Preconditions: 1 <= bits <= 11, coefficient in [0,q).
// Largest numerator is 3328*2048+1664=6817408, within uint32.
uint mlkem_compress(uint coefficient, uint bits) {
	uint scale = 1u << bits;
	return ((coefficient * scale + MLKEM_Q / 2u) / MLKEM_Q) & (scale - 1u);
}

// Preconditions: 1 <= bits <= 11, encoded in [0,2^bits).
uint mlkem_decompress(uint encoded, uint bits) {
	return (encoded * MLKEM_Q + (1u << (bits - 1u))) >> bits;
}
