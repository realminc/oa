// FIPS 203 Algorithms 13-15. No live qualified OA C++ K-PKE donor exists.
// Standards-derived private deterministic laboratory, NOT a standalone scheme
// or an admitted secret execution path. Caller owns distinct arrays and validates
// k in {2,3,4}; eta1 = (k==2 ? 3 : 2), du/dv = (k==4 ? 11/5 : 10/4).
// Byte arrays contain byte-valued uints. Only logical prefixes are written.
import keccak;
import mlkem_field;
import mlkem_sampling;
import mlkem_packing;
import mlkem_ntt;
import mlkem_compression;

// G(d || k) = SHA3-512, 33-byte input with FIPS 202 suffix 0x06.
void mlkem_expand_key_seed(uint d[8], uint k, out uint rho[8], out uint sigma[8]) {
	uint2 state[25];
	for (uint i = 0u; i < 25u; ++i) state[i] = uint2(0u, 0u);
	for (uint i = 0u; i < 8u; ++i) {
		if ((i & 1u) == 0u) state[i / 2u].x = d[i];
		else state[i / 2u].y = d[i];
	}
	state[4].x = k | 0x600u;
	state[8].y = 0x80000000u; // SHA3-512 rate = 72 bytes.
	keccak_f1600_u32(state);
	for (uint i = 0u; i < 8u; ++i) {
		rho[i] = (i & 1u) == 0u ? state[i / 2u].x : state[i / 2u].y;
		sigma[i] = (i & 1u) == 0u ? state[4u + i / 2u].x : state[4u + i / 2u].y;
	}
}

// Accumulate directly in quadratic NTT representation, without a product copy.
void mlkem_accumulate(uint left[256], uint vector[1024], uint offset,
	inout uint result[256]) {
	for (uint pair = 0u; pair < 128u; ++pair) {
		uint gamma = MLKEM_ZETA[64u + pair / 2u];
		if ((pair & 1u) != 0u) gamma = mlkem_neg(gamma);
		uint j = 2u * pair;
		uint2 p = mlkem_base_multiply(uint2(left[j], left[j + 1u]),
			uint2(vector[offset + j], vector[offset + j + 1u]), gamma);
		result[j] = mlkem_add(result[j], p.x);
		result[j + 1u] = mlkem_add(result[j + 1u], p.y);
	}
}

void mlkem_kpke_keygen(uint d[8], uint k, inout uint ek[1568], inout uint dk[1536]) {
	uint rho[8], sigma[8], secret[1024], poly[256], sum[256], bytes[384];
	mlkem_expand_key_seed(d, k, rho, sigma);
	uint eta = k == 2u ? 3u : 2u;
	for (uint i = 0u; i < k; ++i) {
		mlkem_sample_noise(sigma, i, eta, poly);
		mlkem_ntt(poly);
		for (uint j = 0u; j < 256u; ++j) secret[i * 256u + j] = poly[j];
		mlkem_byte_encode(poly, 12u, bytes);
		for (uint j = 0u; j < 384u; ++j) dk[i * 384u + j] = bytes[j];
	}
	for (uint i = 0u; i < k; ++i) {
		mlkem_sample_noise(sigma, k + i, eta, sum);
		mlkem_ntt(sum);
		for (uint j = 0u; j < k; ++j) {
			mlkem_sample_ntt(rho, j, i, poly);
			mlkem_accumulate(poly, secret, j * 256u, sum);
		}
		mlkem_byte_encode(sum, 12u, bytes);
		for (uint j = 0u; j < 384u; ++j) ek[i * 384u + j] = bytes[j];
	}
	for (uint i = 0u; i < 32u; ++i) ek[384u * k + i] = (rho[i / 4u] >> ((i % 4u) * 8u)) & 255u;
}

// Returns pre-reduction public-key canonicality for the future ML-KEM boundary.
// Like K-PKE itself, computes with reduced values; not a public admission check.
bool mlkem_kpke_encrypt(uint ek[1568], uint message[32], uint r[8], uint k,
	inout uint ciphertext[1568]) {
	uint rho[8], y[1024], poly[256], sum[256], bytes[384];
	uint eta = k == 2u ? 3u : 2u, du = k == 4u ? 11u : 10u, dv = k == 4u ? 5u : 4u;
	for (uint i = 0u; i < 8u; ++i) rho[i] = 0u;
	for (uint i = 0u; i < 32u; ++i) rho[i / 4u] |= ek[384u * k + i] << ((i % 4u) * 8u);
	for (uint i = 0u; i < k; ++i) {
		mlkem_sample_noise(r, i, eta, poly);
		mlkem_ntt(poly);
		for (uint j = 0u; j < 256u; ++j) y[i * 256u + j] = poly[j];
	}
	for (uint i = 0u; i < k; ++i) {
		for (uint n = 0u; n < 256u; ++n) sum[n] = 0u;
		for (uint j = 0u; j < k; ++j) {
			mlkem_sample_ntt(rho, i, j, poly); // A^T[i,j] = A[j,i].
			mlkem_accumulate(poly, y, j * 256u, sum);
		}
		mlkem_intt(sum);
		mlkem_sample_noise(r, k + i, 2u, poly);
		for (uint n = 0u; n < 256u; ++n) sum[n] = mlkem_compress(mlkem_add(sum[n], poly[n]), du);
		mlkem_byte_encode(sum, du, bytes);
		for (uint n = 0u; n < 32u * du; ++n) ciphertext[i * 32u * du + n] = bytes[n];
	}
	bool canonical = true;
	for (uint n = 0u; n < 256u; ++n) sum[n] = 0u;
	for (uint i = 0u; i < k; ++i) {
		for (uint n = 0u; n < 384u; ++n) bytes[n] = ek[i * 384u + n];
		bool valid = mlkem_byte_decode(bytes, 12u, poly);
		canonical = canonical && valid;
		mlkem_accumulate(poly, y, i * 256u, sum);
	}
	mlkem_intt(sum);
	mlkem_sample_noise(r, 2u * k, 2u, poly);
	for (uint n = 0u; n < 256u; ++n) {
		uint bit = (message[n / 8u] >> (n % 8u)) & 1u;
		sum[n] = mlkem_compress(mlkem_add(mlkem_add(sum[n], poly[n]), mlkem_decompress(bit, 1u)), dv);
	}
	mlkem_byte_encode(sum, dv, bytes);
	for (uint n = 0u; n < 32u * dv; ++n) ciphertext[k * 32u * du + n] = bytes[n];
	return canonical;
}

void mlkem_kpke_decrypt(uint dk[1536], uint ciphertext[1568], uint k, out uint message[32]) {
	uint u[1024], poly[256], sum[256], bytes[384];
	uint du = k == 4u ? 11u : 10u, dv = k == 4u ? 5u : 4u;
	for (uint i = 0u; i < k; ++i) {
		for (uint n = 0u; n < 32u * du; ++n) bytes[n] = ciphertext[i * 32u * du + n];
		mlkem_byte_decode(bytes, du, poly);
		for (uint n = 0u; n < 256u; ++n) poly[n] = mlkem_decompress(poly[n], du);
		mlkem_ntt(poly);
		for (uint n = 0u; n < 256u; ++n) u[i * 256u + n] = poly[n];
	}
	for (uint n = 0u; n < 256u; ++n) sum[n] = 0u;
	for (uint i = 0u; i < k; ++i) {
		for (uint n = 0u; n < 384u; ++n) bytes[n] = dk[i * 384u + n];
		mlkem_byte_decode(bytes, 12u, poly);
		mlkem_accumulate(poly, u, i * 256u, sum);
	}
	mlkem_intt(sum);
	for (uint n = 0u; n < 32u * dv; ++n) bytes[n] = ciphertext[k * 32u * du + n];
	mlkem_byte_decode(bytes, dv, poly);
	for (uint n = 0u; n < 256u; ++n) poly[n] = mlkem_compress(mlkem_sub(mlkem_decompress(poly[n], dv), sum[n]), 1u);
	mlkem_byte_encode(poly, 1u, bytes);
	for (uint n = 0u; n < 32u; ++n) message[n] = bytes[n];
}
