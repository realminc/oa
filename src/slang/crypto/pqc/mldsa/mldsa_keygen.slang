// ML-DSA-44/65/87 KeyGen_internal, FIPS 204 Algorithms 6, 24 and 31.
// New standards-derived implementation: OA C++ delegates to liboqs; its
// historical device keygen was a status stub, not an algorithm donor.
// Public deterministic test seeds only until secret execution is admitted.
// No production entropy, device-secret or constant-time claim.
import shake;
import mldsa_field;
import mldsa_ntt;
import mldsa_sampling;

// RejBoundedPoly: consume low then high nibble, continue until 256 accepted.
// rho_prime contains 64 bytes, nonce is encoded as two little-endian bytes.
void mldsa_expand_secret_poly(
	uint rho_prime[64],
	uint nonce,
	uint eta,
	out uint poly[256]
) {
	PqcShake256 sponge;
	pqc_shake256_init(sponge);
	for (uint i = 0u; i < 64u; ++i) pqc_shake256_absorb_byte(sponge, rho_prime[i]);
	pqc_shake256_absorb_byte(sponge, nonce & 255u);
	pqc_shake256_absorb_byte(sponge, nonce >> 8u);
	pqc_shake256_finalize(sponge);
	uint count = 0u;
	while (count < 256u) {
		uint value = pqc_shake256_squeeze_byte(sponge);
		for (uint half = 0u; half < 2u && count < 256u; ++half) {
			uint nibble = (value >> (half * 4u)) & 15u;
			if (eta == 2u ? nibble < 15u : nibble < 9u) {
				uint reduced = eta == 2u ? nibble % 5u : nibble;
				int coefficient = (int)eta - (int)reduced;
				poly[count++] = coefficient < 0 ? (uint)(coefficient + (int)MLDSA_Q)
					: (uint)coefficient;
			}
		}
	}
}

// Byte-per-element representation is private laboratory transport, not a
// public key-storage ABI. Only active bytes are written, preserving tail poison.
void mldsa_keygen_pack_poly(
	uint values[256],
	uint bits,
	uint base,
	inout uint encoded[4896]
) {
	uint reservoir = 0u, occupied = 0u, cursor = base;
	for (uint i = 0u; i < 256u; ++i) {
		reservoir |= values[i] << occupied;
		occupied += bits;
		while (occupied >= 8u) {
			encoded[cursor++] = reservoir & 255u;
			reservoir >>= 8u;
			occupied -= 8u;
		}
	}
}

// Fails before any write for unknown parameter sets. Output capacities cover
// Table 2's largest encodings (pk 2592, sk 4896). Caller owns disjoint arrays.
bool mldsa_keygen_internal(
	uint parameter_set,
	uint seed[32],
	inout uint public_key[2592],
	inout uint secret_key[4896]
) {
	if (parameter_set != 44u && parameter_set != 65u && parameter_set != 87u) return false;
	uint rows = parameter_set == 44u ? 4u : (parameter_set == 65u ? 6u : 8u);
	uint cols = parameter_set == 44u ? 4u : (parameter_set == 65u ? 5u : 7u);
	uint eta = parameter_set == 65u ? 4u : 2u;
	uint eta_bits = eta == 2u ? 3u : 4u;
	uint eta_bytes = eta_bits * 32u;
	PqcShake256 sponge;
	pqc_shake256_init(sponge);
	for (uint i = 0u; i < 32u; ++i) pqc_shake256_absorb_byte(sponge, seed[i]);
	pqc_shake256_absorb_byte(sponge, rows);
	pqc_shake256_absorb_byte(sponge, cols);
	pqc_shake256_finalize(sponge);
	uint rho[8], rho_prime[64];
	for (uint i = 0u; i < 8u; ++i) rho[i] = 0u;
	for (uint i = 0u; i < 32u; ++i) {
		uint value = pqc_shake256_squeeze_byte(sponge);
		public_key[i] = value;
		secret_key[i] = value;
		rho[i / 4u] |= value << ((i % 4u) * 8u);
	}
	for (uint i = 0u; i < 64u; ++i) rho_prime[i] = pqc_shake256_squeeze_byte(sponge);
	for (uint i = 0u; i < 32u; ++i) secret_key[32u + i] = pqc_shake256_squeeze_byte(sponge);

	uint s1[7][256], s2[256], packed[256];
	for (uint col = 0u; col < cols; ++col) {
		mldsa_expand_secret_poly(rho_prime, col, eta, s1[col]);
		for (uint i = 0u; i < 256u; ++i) packed[i] = (uint)((int)eta - mldsa_centered(s1[col][i]));
		mldsa_keygen_pack_poly(packed, eta_bits, 128u + col * eta_bytes, secret_key);
		mldsa_ntt(s1[col]);
	}
	for (uint row = 0u; row < rows; ++row) {
		mldsa_expand_secret_poly(rho_prime, cols + row, eta, s2);
		for (uint i = 0u; i < 256u; ++i) packed[i] = (uint)((int)eta - mldsa_centered(s2[i]));
		mldsa_keygen_pack_poly(packed, eta_bits, 128u + (cols + row) * eta_bytes, secret_key);
		uint t[256];
		for (uint i = 0u; i < 256u; ++i) t[i] = 0u;
		for (uint col = 0u; col < cols; ++col) {
			uint a[256];
			mldsa_expand_a_poly(rho, row, col, a);
			for (uint i = 0u; i < 256u; ++i) t[i] = mldsa_add(t[i], mldsa_mul(a[i], s1[col][i]));
		}
		mldsa_intt(t);
		for (uint i = 0u; i < 256u; ++i) {
			t[i] = mldsa_add(t[i], s2[i]);
			packed[i] = (uint)(4096 - mldsa_power2round_lo(t[i]));
		}
		mldsa_keygen_pack_poly(packed, 13u, 128u + (cols + rows) * eta_bytes + row * 416u, secret_key);
		// Public encoding streams one whole byte at a time; no shared-word writes.
		uint reservoir = 0u, occupied = 0u, cursor = 32u + row * 320u;
		for (uint i = 0u; i < 256u; ++i) {
			reservoir |= mldsa_power2round_hi(t[i]) << occupied;
			occupied += 10u;
			while (occupied >= 8u) {
				public_key[cursor++] = reservoir & 255u;
				reservoir >>= 8u;
				occupied -= 8u;
			}
		}
	}
	pqc_shake256_init(sponge);
	for (uint i = 0u; i < 32u + rows * 320u; ++i) pqc_shake256_absorb_byte(sponge, public_key[i]);
	pqc_shake256_finalize(sponge);
	for (uint i = 0u; i < 64u; ++i) secret_key[64u + i] = pqc_shake256_squeeze_byte(sponge);
	return true;
}
