// FIPS 203 Algorithms 16-18, deterministic internal algorithms only.
// No live qualified OA C++ ML-KEM donor exists; reuse OA's uint2 Keccak and
// private K-PKE. Test laboratory: no entropy, secret-device, erasure, timing,
// GPU admission or public input-validation claim. Caller checks k={2,3,4},
// byte-valued arrays, logical sizes and distinct input/output storage.
import keccak;
import mlkem_kpke;

struct MlkemSponge {
	uint2 state[25];
	uint position;
	uint rate;
};

void mlkem_sponge_init(uint rate, out MlkemSponge sponge) {
	for (uint i = 0u; i < 25u; ++i) sponge.state[i] = uint2(0u, 0u);
	sponge.position = 0u;
	sponge.rate = rate;
}

void mlkem_sponge_byte(inout MlkemSponge sponge, uint value) {
	uint lane = sponge.position / 8u, shift = (sponge.position % 4u) * 8u;
	if ((sponge.position & 7u) < 4u) sponge.state[lane].x ^= (value & 255u) << shift;
	else sponge.state[lane].y ^= (value & 255u) << shift;
	if (++sponge.position == sponge.rate) {
		keccak_f1600_u32(sponge.state);
		sponge.position = 0u;
	}
}

// Destructive finish, output <=64 bytes (all callers fit one rate block).
void mlkem_sponge_finish(inout MlkemSponge sponge, uint suffix, out uint digest[64]) {
	uint lane = sponge.position / 8u, shift = (sponge.position % 4u) * 8u;
	if ((sponge.position & 7u) < 4u) sponge.state[lane].x ^= suffix << shift;
	else sponge.state[lane].y ^= suffix << shift;
	sponge.state[sponge.rate / 8u - 1u].y ^= 0x80000000u;
	keccak_f1600_u32(sponge.state);
	for (uint i = 0u; i < 64u; ++i) {
		uint word = (i & 7u) < 4u ? sponge.state[i / 8u].x : sponge.state[i / 8u].y;
		digest[i] = (word >> ((i & 3u) * 8u)) & 255u;
	}
}

void mlkem_hash_key(uint ek[1568], uint k, out uint digest[64]) {
	MlkemSponge sponge;
	mlkem_sponge_init(136u, sponge);
	for (uint i = 0u; i < 384u * k + 32u; ++i) mlkem_sponge_byte(sponge, ek[i]);
	mlkem_sponge_finish(sponge, 6u, digest); // H = SHA3-256; first32 used.
}

void mlkem_derive(uint message[32], uint hash[32], out uint key[32], out uint r[8]) {
	MlkemSponge sponge;
	uint digest[64];
	mlkem_sponge_init(72u, sponge);
	for (uint i = 0u; i < 32u; ++i) mlkem_sponge_byte(sponge, message[i]);
	for (uint i = 0u; i < 32u; ++i) mlkem_sponge_byte(sponge, hash[i]);
	mlkem_sponge_finish(sponge, 6u, digest); // G = SHA3-512.
	for (uint i = 0u; i < 32u; ++i) key[i] = digest[i];
	for (uint i = 0u; i < 8u; ++i) r[i] = digest[32u + 4u*i] | (digest[33u + 4u*i] << 8u)
		| (digest[34u + 4u*i] << 16u) | (digest[35u + 4u*i] << 24u);
}

void mlkem_keygen_internal(uint d[8], uint z[32], uint k,
	inout uint ek[1568], inout uint dk[3168]) {
	uint pkeSecret[1536], digest[64];
	mlkem_kpke_keygen(d, k, ek, pkeSecret);
	mlkem_hash_key(ek, k, digest);
	for (uint i = 0u; i < 384u*k; ++i) dk[i] = pkeSecret[i];
	for (uint i = 0u; i < 384u*k + 32u; ++i) dk[384u*k + i] = ek[i];
	for (uint i = 0u; i < 32u; ++i) {
		dk[768u*k + 32u + i] = digest[i];
		dk[768u*k + 64u + i] = z[i];
	}
}

// Internal formula; returned flag is public-key canonicality, NOT rejection.
bool mlkem_encaps_internal(uint ek[1568], uint message[32], uint k,
	out uint key[32], inout uint ciphertext[1568]) {
	uint digest[64], hash[32], r[8];
	mlkem_hash_key(ek, k, digest);
	for (uint i = 0u; i < 32u; ++i) hash[i] = digest[i];
	mlkem_derive(message, hash, key, r);
	return mlkem_kpke_encrypt(ek, message, r, k, ciphertext);
}

// No rejection flag leaves this function. Fixed-work compare and masked select
// are source-level properties only; machine-code timing and erasure are open.
void mlkem_decaps_internal(uint dk[3168], uint ciphertext[1568], uint k, out uint key[32]) {
	uint pkeSecret[1536], ek[1568], hash[32], message[32], r[8];
	uint candidate[32], rejection[64], check[1568];
	uint size = 32u * (k == 4u ? 11u*k + 5u : 10u*k + 4u);
	for (uint i = 0u; i < 384u*k; ++i) pkeSecret[i] = dk[i];
	for (uint i = 0u; i < 384u*k + 32u; ++i) ek[i] = dk[384u*k + i];
	for (uint i = 0u; i < 32u; ++i) hash[i] = dk[768u*k + 32u + i];
	mlkem_kpke_decrypt(pkeSecret, ciphertext, k, message);
	mlkem_derive(message, hash, candidate, r);
	MlkemSponge sponge;
	mlkem_sponge_init(136u, sponge);
	for (uint i = 0u; i < 32u; ++i) mlkem_sponge_byte(sponge, dk[768u*k + 64u + i]);
	for (uint i = 0u; i < size; ++i) mlkem_sponge_byte(sponge, ciphertext[i]);
	mlkem_sponge_finish(sponge, 31u, rejection); // J = SHAKE256,32bytes.
	mlkem_kpke_encrypt(ek, message, r, k, check);
	uint difference = 0u;
	for (uint i = 0u; i < size; ++i) difference |= ciphertext[i] ^ check[i];
	uint mask = 0u - ((difference | (0u - difference)) >> 31u);
	for (uint i = 0u; i < 32u; ++i) key[i] = (candidate[i] & ~mask) | (rejection[i] & mask);
}

// FIPS 203 sections 7.2/7.3 input checks; private fixed-capacity laboratory.
// Length is the logical byte count, not allocation capacity. Reject before
// indexing when parameter/length is invalid. Array elements must be bytes.
bool mlkem_parameter_valid(uint k) { return k >= 2u && k <= 4u; }

bool mlkem_ciphertext_length_valid(uint k, uint length) {
	if (!mlkem_parameter_valid(k)) return false;
	return length == 32u * (k == 4u ? 11u*k + 5u : 10u*k + 4u);
}

bool mlkem_encapsulation_key_valid(uint ek[1568], uint k, uint length) {
	if (!mlkem_parameter_valid(k) || length != 384u*k + 32u) return false;
	uint invalid = 0u;
	for (uint i = 0u; i < length; ++i) invalid |= ek[i] >> 8u;
	for (uint i = 0u; i < 384u*k; i += 3u) {
		uint a = ek[i] | ((ek[i + 1u] & 15u) << 8u);
		uint b = (ek[i + 1u] >> 4u) | (ek[i + 2u] << 4u);
		invalid |= uint(a >= 3329u) | uint(b >= 3329u);
	}
	return invalid == 0u;
}

bool mlkem_decapsulation_key_valid(uint dk[3168], uint k, uint length) {
	if (!mlkem_parameter_valid(k) || length != 768u*k + 96u) return false;
	uint invalid = 0u, ek[1568], digest[64];
	for (uint i = 0u; i < length; ++i) invalid |= dk[i] >> 8u;
	for (uint i = 0u; i < 384u*k + 32u; ++i) ek[i] = dk[384u*k + i];
	mlkem_hash_key(ek, k, digest);
	for (uint i = 0u; i < 32u; ++i) invalid |= digest[i] ^ dk[768u*k + 32u + i];
	return invalid == 0u;
}
