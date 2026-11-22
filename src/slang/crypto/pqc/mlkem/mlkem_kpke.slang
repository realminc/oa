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

// Storage adaptation of the existing FIPS 203 formulas, following the proven
// ML-DSA cache interface. This cache does not cover all locals or driver spills.
// Every operation writes its complete k*256 prefix before reading that prefix.
static const uint MLKEM_VECTOR_WORDS = 1024u;

interface IMlkemVector {
	uint read(uint index);
	[mutating] void write(uint index, uint value);
}

struct MlkemLocalVector : IMlkemVector {
	uint words[MLKEM_VECTOR_WORDS];
	uint read(uint index) { return words[index]; }
	[mutating] void write(uint index, uint value) { words[index] = value; }
}

struct MlkemBufferVector : IMlkemVector {
	RWByteAddressBuffer buffer;
	uint base;
	uint read(uint index) { return buffer.Load(base + index * 4u); }
	[mutating] void write(uint index, uint value) { buffer.Store(base + index * 4u, value); }
}

bool mlkem_vector_range_valid(RWByteAddressBuffer workspace, uint base, uint k) {
	uint length;
	workspace.GetDimensions(length);
	return k >= 2u && k <= 4u && (base & 3u) == 0u
		&& base <= length && MLKEM_VECTOR_WORDS * 4u <= length - base;
}

// G(d || k) = SHA3-512, 33-byte input with FIPS 202 suffix 0x06.
void mlkem_expand_key_seed(uint d[8], uint k, out uint rho[8], out uint sigma[8]) {
	uint2 state[25];
	[loop]
	for (uint i = 0u; i < 25u; ++i) state[i] = uint2(0u, 0u);
	[loop]
	for (uint i = 0u; i < 8u; ++i) {
		if ((i & 1u) == 0u) state[i / 2u].x = d[i];
		else state[i / 2u].y = d[i];
	}
	state[4].x = k | 0x600u;
	state[8].y = 0x80000000u; // SHA3-512 rate = 72 bytes.
	keccak_f1600_u32(state);
	[loop]
	for (uint i = 0u; i < 8u; ++i) {
		rho[i] = (i & 1u) == 0u ? state[i / 2u].x : state[i / 2u].y;
		sigma[i] = (i & 1u) == 0u ? state[4u + i / 2u].x : state[4u + i / 2u].y;
	}
}

// Accumulate directly in quadratic NTT representation, without a product copy.
void mlkem_accumulate<T : IMlkemVector>(
	uint left[256],
	inout T vector,
	uint offset,
	inout uint result[256]
) {
	[loop]
	for (uint pair = 0u; pair < 128u; ++pair) {
		uint gamma = MLKEM_ZETA[64u + pair / 2u];
		if ((pair & 1u) != 0u) gamma = mlkem_neg(gamma);
		uint j = 2u * pair;
		uint2 p = mlkem_base_multiply(uint2(left[j], left[j + 1u]),
			uint2(vector.read(offset + j), vector.read(offset + j + 1u)), gamma);
		result[j] = mlkem_add(result[j], p.x);
		result[j + 1u] = mlkem_add(result[j + 1u], p.y);
	}
}

void mlkem_kpke_keygen_cached<T : IMlkemVector>(
	uint d[8],
	uint k,
	inout T secret,
	inout uint ek[1568],
	inout uint dk[1536]
) {
	uint rho[8], sigma[8], poly[256], sum[256], bytes[384];
	mlkem_expand_key_seed(d, k, rho, sigma);
	uint eta = k == 2u ? 3u : 2u;
	[loop]
	for (uint i = 0u; i < k; ++i) {
		mlkem_sample_noise(sigma, i, eta, poly);
		mlkem_ntt(poly);
		[loop]
		for (uint j = 0u; j < 256u; ++j) secret.write(i * 256u + j, poly[j]);
		mlkem_byte_encode(poly, 12u, bytes);
		[loop]
		for (uint j = 0u; j < 384u; ++j) dk[i * 384u + j] = bytes[j];
	}
	[loop]
	for (uint i = 0u; i < k; ++i) {
		mlkem_sample_noise(sigma, k + i, eta, sum);
		mlkem_ntt(sum);
		[loop]
		for (uint j = 0u; j < k; ++j) {
			mlkem_sample_ntt(rho, j, i, poly);
			mlkem_accumulate(poly, secret, j * 256u, sum);
		}
		mlkem_byte_encode(sum, 12u, bytes);
		[loop]
		for (uint j = 0u; j < 384u; ++j) ek[i * 384u + j] = bytes[j];
	}
	[loop]
	for (uint i = 0u; i < 32u; ++i) ek[384u * k + i] = (rho[i / 4u] >> ((i % 4u) * 8u)) & 255u;
}

// Returns pre-reduction public-key canonicality for the future ML-KEM boundary.
// Like K-PKE itself, computes with reduced values; not a public admission check.
bool mlkem_kpke_encrypt_cached<T : IMlkemVector>(
	uint ek[1568],
	uint message[32],
	uint r[8],
	uint k,
	inout T y,
	inout uint ciphertext[1568]
) {
	uint rho[8], poly[256], sum[256], bytes[384];
	uint eta = k == 2u ? 3u : 2u, du = k == 4u ? 11u : 10u, dv = k == 4u ? 5u : 4u;
	[loop]
	for (uint i = 0u; i < 8u; ++i) rho[i] = 0u;
	[loop]
	for (uint i = 0u; i < 32u; ++i) rho[i / 4u] |= ek[384u * k + i] << ((i % 4u) * 8u);
	[loop]
	for (uint i = 0u; i < k; ++i) {
		mlkem_sample_noise(r, i, eta, poly);
		mlkem_ntt(poly);
		[loop]
		for (uint j = 0u; j < 256u; ++j) y.write(i * 256u + j, poly[j]);
	}
	[loop]
	for (uint i = 0u; i < k; ++i) {
		[loop]
		for (uint n = 0u; n < 256u; ++n) sum[n] = 0u;
		[loop]
		for (uint j = 0u; j < k; ++j) {
			mlkem_sample_ntt(rho, i, j, poly); // A^T[i,j] = A[j,i].
			mlkem_accumulate(poly, y, j * 256u, sum);
		}
		mlkem_intt(sum);
		mlkem_sample_noise(r, k + i, 2u, poly);
		[loop]
		for (uint n = 0u; n < 256u; ++n) sum[n] = mlkem_compress(mlkem_add(sum[n], poly[n]), du);
		mlkem_byte_encode(sum, du, bytes);
		[loop]
		for (uint n = 0u; n < 32u * du; ++n) ciphertext[i * 32u * du + n] = bytes[n];
	}
	bool canonical = true;
	[loop]
	for (uint n = 0u; n < 256u; ++n) sum[n] = 0u;
	[loop]
	for (uint i = 0u; i < k; ++i) {
		[loop]
		for (uint n = 0u; n < 384u; ++n) bytes[n] = ek[i * 384u + n];
		bool valid = mlkem_byte_decode(bytes, 12u, poly);
		canonical = canonical && valid;
		mlkem_accumulate(poly, y, i * 256u, sum);
	}
	mlkem_intt(sum);
	mlkem_sample_noise(r, 2u * k, 2u, poly);
	[loop]
	for (uint n = 0u; n < 256u; ++n) {
		uint bit = (message[n / 8u] >> (n % 8u)) & 1u;
		sum[n] = mlkem_compress(mlkem_add(mlkem_add(sum[n], poly[n]), mlkem_decompress(bit, 1u)), dv);
	}
	mlkem_byte_encode(sum, dv, bytes);
	[loop]
	for (uint n = 0u; n < 32u * dv; ++n) ciphertext[k * 32u * du + n] = bytes[n];
	return canonical;
}

void mlkem_kpke_decrypt_cached<T : IMlkemVector>(
	uint dk[1536],
	uint ciphertext[1568],
	uint k,
	inout T u,
	out uint message[32]
) {
	uint poly[256], sum[256], bytes[384];
	uint du = k == 4u ? 11u : 10u, dv = k == 4u ? 5u : 4u;
	[loop]
	for (uint i = 0u; i < k; ++i) {
		[loop]
		for (uint n = 0u; n < 32u * du; ++n) bytes[n] = ciphertext[i * 32u * du + n];
		mlkem_byte_decode(bytes, du, poly);
		[loop]
		for (uint n = 0u; n < 256u; ++n) poly[n] = mlkem_decompress(poly[n], du);
		mlkem_ntt(poly);
		[loop]
		for (uint n = 0u; n < 256u; ++n) u.write(i * 256u + n, poly[n]);
	}
	[loop]
	for (uint n = 0u; n < 256u; ++n) sum[n] = 0u;
	[loop]
	for (uint i = 0u; i < k; ++i) {
		[loop]
		for (uint n = 0u; n < 384u; ++n) bytes[n] = dk[i * 384u + n];
		mlkem_byte_decode(bytes, 12u, poly);
		mlkem_accumulate(poly, u, i * 256u, sum);
	}
	mlkem_intt(sum);
	[loop]
	for (uint n = 0u; n < 32u * dv; ++n) bytes[n] = ciphertext[k * 32u * du + n];
	mlkem_byte_decode(bytes, dv, poly);
	[loop]
	for (uint n = 0u; n < 256u; ++n) poly[n] = mlkem_compress(mlkem_sub(mlkem_decompress(poly[n], dv), sum[n]), 1u);
	mlkem_byte_encode(poly, 1u, bytes);
	[loop]
	for (uint n = 0u; n < 32u; ++n) message[n] = bytes[n];
}

// The existing laboratory and dispatch consumers retain their exact interfaces.
void mlkem_kpke_keygen(uint d[8], uint k, inout uint ek[1568], inout uint dk[1536]) {
	MlkemLocalVector cache;
	mlkem_kpke_keygen_cached(d, k, cache, ek, dk);
}

bool mlkem_kpke_encrypt(uint ek[1568], uint message[32], uint r[8], uint k,
	inout uint ciphertext[1568]) {
	MlkemLocalVector cache;
	return mlkem_kpke_encrypt_cached(ek, message, r, k, cache, ciphertext);
}

void mlkem_kpke_decrypt(uint dk[1536], uint ciphertext[1568], uint k, out uint message[32]) {
	MlkemLocalVector cache;
	mlkem_kpke_decrypt_cached(dk, ciphertext, k, cache, message);
}

// Preflight the whole cache before writing output or touching workspace.
// Invalid ranges/parameters preserve caller storage; no iteration cap is added.
bool mlkem_kpke_keygen_buffer(
	uint d[8],
	uint k,
	RWByteAddressBuffer workspace,
	uint base,
	inout uint ek[1568],
	inout uint dk[1536]
) {
	if (!mlkem_vector_range_valid(workspace, base, k)) return false;
	MlkemBufferVector cache;
	cache.buffer = workspace;
	cache.base = base;
	mlkem_kpke_keygen_cached(d, k, cache, ek, dk);
	return true;
}

bool mlkem_kpke_encrypt_buffer(
	uint ek[1568],
	uint message[32],
	uint r[8],
	uint k,
	RWByteAddressBuffer workspace,
	uint base,
	inout uint ciphertext[1568]
) {
	if (!mlkem_vector_range_valid(workspace, base, k)) return false;
	MlkemBufferVector cache;
	cache.buffer = workspace;
	cache.base = base;
	return mlkem_kpke_encrypt_cached(ek, message, r, k, cache, ciphertext);
}

bool mlkem_kpke_decrypt_buffer(
	uint dk[1536],
	uint ciphertext[1568],
	uint k,
	RWByteAddressBuffer workspace,
	uint base,
	inout uint message[32]
) {
	if (!mlkem_vector_range_valid(workspace, base, k)) return false;
	MlkemBufferVector cache;
	cache.buffer = workspace;
	cache.base = base;
	mlkem_kpke_decrypt_cached(dk, ciphertext, k, cache, message);
	return true;
}
