// FIPS 203 Algorithms 5/6. No live qualified OA ML-KEM donor exists;
// adaptation: standards-derived private public-data laboratory implementation.
// Preconditions: 1 <= bits <= 12; source coefficients are canonical modulo
// 2^bits (bits < 12) or q (bits == 12). Bytes occupy low 8 bits of uints.
// Arrays are distinct; output byte range is exactly 32*bits. The remaining
// capacity is caller-owned. No packed-buffer writes or shared-word races here.
import mlkem_field;

void mlkem_byte_encode(uint coefficients[256], uint bits, inout uint bytes[384]) {
	uint reservoir = 0u, available = 0u, position = 0u;
	for (uint i = 0u; i < 256u; ++i) {
		reservoir |= coefficients[i] << available;
		available += bits;
		while (available >= 8u) {
			bytes[position++] = reservoir & 255u;
			reservoir >>= 8u;
			available -= 8u;
		}
	}
}

// Decode always implements FIPS reduction. The additional return value reports
// canonicality BEFORE reduction so an eventual encapsulation-key boundary can
// reject encodings which fail ByteEncode12(ByteDecode12(key)) == key.
bool mlkem_byte_decode(uint bytes[384], uint bits, out uint coefficients[256]) {
	uint reservoir = 0u, available = 0u, position = 0u;
	uint mask = (1u << bits) - 1u;
	bool canonical = true;
	for (uint i = 0u; i < 256u; ++i) {
		while (available < bits) {
			reservoir |= bytes[position++] << available;
			available += 8u;
		}
		uint value = reservoir & mask;
		reservoir >>= bits;
		available -= bits;
		if (bits == 12u) {
			canonical = canonical && value < MLKEM_Q;
			if (value >= MLKEM_Q) value -= MLKEM_Q;
		}
		coefficients[i] = value;
	}
	return canonical;
}
