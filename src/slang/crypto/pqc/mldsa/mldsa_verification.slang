// ML-DSA-44/65/87 pure/prehashed verification, FIPS 204 Algorithms 3, 5 and 8.
// New standards implementation: the historical OA verifier was a status stub.
// Public-data helper shared by the quarantined dispatch and test laboratory.
// This formula does not itself admit a GPU capability or secret operations.
import shake;
import mldsa_prehash_layout;
import mldsa_prehash_digest;
import mldsa_field;
import mldsa_ntt;
import mldsa_packing;
import mldsa_sampling;

static const uint MLDSA65_SIG_BYTES = 3309u;
static const uint MLDSA65_PK_BYTES = 1952u;
static const uint MLDSA65_BETA = 196u;

// algorithm=0 selects pure framing; nonzero selects the checked prehash identity.
// hash_message hashes the original message; otherwise the input is its exact digest.
// This internal composition preserves the proven decoding and polynomial body.
bool mldsa_verify_framed(
	uint parameter_set,
	RWByteAddressBuffer public_keys, uint pk_base, uint pk_length,
	RWByteAddressBuffer signatures, uint sig_base, uint sig_length,
	RWByteAddressBuffer messages, uint message_base, uint message_length,
	RWByteAddressBuffer contexts, uint context_base, uint context_length,
	uint algorithm, bool hash_message
) {
	// FIPS 204 final Table 1. Reject unknown sets before deriving any size or indexing.
	if (parameter_set != 44u && parameter_set != 65u && parameter_set != 87u) return false;
	uint rows = parameter_set == 44u ? 4u : (parameter_set == 65u ? 6u : 8u);
	uint cols = parameter_set == 44u ? 4u : (parameter_set == 65u ? 5u : 7u);
	uint challenge_words = parameter_set == 44u ? 8u : (parameter_set == 65u ? 12u : 16u);
	uint gamma1 = parameter_set == 44u ? 131072u : 524288u;
	uint gamma2 = parameter_set == 44u ? 95232u : 261888u;
	uint z_bits = parameter_set == 44u ? 18u : 20u;
	uint beta = parameter_set == 44u ? 78u : (parameter_set == 65u ? 196u : 120u);
	uint omega = parameter_set == 44u ? 80u : (parameter_set == 65u ? 55u : 75u);
	uint tau = parameter_set == 44u ? 39u : (parameter_set == 65u ? 49u : 60u);
	uint pk_bytes = 32u + rows * 320u;
	uint sig_bytes = challenge_words * 4u + cols * z_bits * 32u + omega + rows;
	if (pk_length != pk_bytes || sig_length != sig_bytes || context_length > 255u) return false;
	if (algorithm != 0u && (mldsa_prehash_bytes(algorithm) == 0u
		|| (!hash_message && message_length != mldsa_prehash_bytes(algorithm)))) return false;
	if (!pqc_buffer_range(public_keys, pk_base, pk_length)
		|| !pqc_buffer_range(signatures, sig_base, sig_length)
		|| !pqc_buffer_range(messages, message_base, message_length)
		|| !pqc_buffer_range(contexts, context_base, context_length)) return false;

	uint rho[8], t1[8][256], c_tilde[16], z[7][256], h[8][256];
	for (uint w = 0u; w < 8u; ++w) rho[w] = read_le_word(public_keys, pk_base + w * 4u);
	for (uint row = 0u; row < rows; ++row)
		mldsa_unpack_poly_unsigned(public_keys, pk_base + 32u + row * 320u, 10u, t1[row]);
	for (uint w = 0u; w < 16u; ++w)
		c_tilde[w] = w < challenge_words ? read_le_word(signatures, sig_base + w * 4u) : 0u;
	for (uint col = 0u; col < cols; ++col) {
		mldsa_unpack_poly_signed(signatures, sig_base + challenge_words * 4u + col * z_bits * 32u,
			z_bits, gamma1, z[col]);
		if (!mldsa_check_norm(z[col], gamma1 - beta)) return false;
	}
	uint h_base = sig_base + challenge_words * 4u + cols * z_bits * 32u;
	uint idx = 0u;
	for (uint row = 0u; row < rows; ++row)
		if (!mldsa_unpack_hint_row(signatures, h_base, omega, row, idx, h[row])) return false;
	if (!mldsa_hint_tail_valid(signatures, h_base, idx, omega)) return false;

	PqcShake256 sponge;
	pqc_shake256_init(sponge);
	pqc_shake256_absorb_buffer(sponge, public_keys, pk_base, pk_length);
	uint2 tr[8];
	pqc_shake256_finish(sponge, tr);

	pqc_shake256_init(sponge);
	pqc_shake256_absorb_digest(sponge, tr);
	if (algorithm == 0u) {
		// External pure M' = 0x00 || len(ctx) || ctx || message.
		pqc_shake256_absorb_byte(sponge, 0u);
		pqc_shake256_absorb_byte(sponge, context_length);
		pqc_shake256_absorb_buffer(sponge, contexts, context_base, context_length);
		pqc_shake256_absorb_buffer(sponge, messages, message_base, message_length);
	} else if (hash_message) {
		uint digest[16];
		if (!mldsa_prehash_digest(messages, message_base, message_length, algorithm, digest)) return false;
		mldsa_absorb_prehash_header(sponge, contexts, context_base, context_length, algorithm);
		for (uint i = 0u; i < mldsa_prehash_bytes(algorithm); ++i)
			pqc_shake256_absorb_byte(sponge, (digest[i / 4u] >> ((i & 3u) * 8u)) & 255u);
	} else {
		mldsa_absorb_prehash_framing(sponge, messages, message_base, message_length,
			contexts, context_base, context_length, algorithm);
	}
	uint2 mu[8];
	pqc_shake256_finish(sponge, mu);

	uint c[256];
	mldsa_sample_in_ball(c_tilde, challenge_words, tau, c);
	mldsa_ntt(c);
	for (uint l = 0u; l < cols; ++l) mldsa_ntt(z[l]);
	for (uint k = 0u; k < rows; ++k) {
		for (uint i = 0u; i < 256u; ++i) t1[k][i] = mldsa_mul(t1[k][i], MLDSA_D_FACTOR);
		mldsa_ntt(t1[k]);
	}

	pqc_shake256_init(sponge);
	pqc_shake256_absorb_digest(sponge, mu);
	for (uint k = 0u; k < rows; ++k) {
		uint w[256];
		for (uint i = 0u; i < 256u; ++i) w[i] = 0u;
		for (uint l = 0u; l < cols; ++l) {
			uint a[256];
			// ExpandA already returns NTT-domain coefficients.
			mldsa_expand_a_poly(rho, k, l, a);
			for (uint i = 0u; i < 256u; ++i) w[i] = mldsa_add(w[i], mldsa_mul(a[i], z[l][i]));
		}
		for (uint i = 0u; i < 256u; ++i) w[i] = mldsa_sub(w[i], mldsa_mul(c[i], t1[k][i]));
		mldsa_intt(w);
		// FIPS w1Encode: six-bit coefficients for 44, four-bit for 65/87.
		// Stream packed bytes directly into the hash; do not retain an encoded row.
		uint bits = parameter_set == 44u ? 6u : 4u;
		uint reservoir = 0u, occupied = 0u;
		for (uint i = 0u; i < 256u; ++i) {
			uint high = mldsa_use_hint(h[k][i], w[i], gamma2);
			reservoir |= high << occupied;
			occupied += bits;
			if (occupied >= 8u) {
				pqc_shake256_absorb_byte(sponge, reservoir & 255u);
				reservoir >>= 8u;
				occupied -= 8u;
			}
		}

	}
	uint2 expected[8];
	pqc_shake256_finish(sponge, expected);
	uint difference = 0u;
	for (uint i = 0u; i < challenge_words; ++i) {
		uint word = ((i & 1u) == 0u) ? expected[i / 2u].x : expected[i / 2u].y;
		difference |= c_tilde[i] ^ word;
	}
	return difference == 0u;
}

// Existing external pure contract and descriptor ABI remain unchanged.
bool mldsa_verify_external(
	uint parameter_set,
	RWByteAddressBuffer public_keys, uint pk_base, uint pk_length,
	RWByteAddressBuffer signatures, uint sig_base, uint sig_length,
	RWByteAddressBuffer messages, uint message_base, uint message_length,
	RWByteAddressBuffer contexts, uint context_base, uint context_length
) {
	return mldsa_verify_framed(parameter_set, public_keys, pk_base, pk_length,
		signatures, sig_base, sig_length, messages, message_base, message_length,
		contexts, context_base, context_length, 0u, false);
}

// Precomputed public digest only. Unknown selectors, including zero, are false;
// no fallback to pure mode or arbitrary caller-supplied DER identifier exists.
bool mldsa_verify_prehashed(
	uint parameter_set,
	RWByteAddressBuffer public_keys, uint pk_base, uint pk_length,
	RWByteAddressBuffer signatures, uint sig_base, uint sig_length,
	RWByteAddressBuffer digests, uint digest_base, uint digest_length,
	RWByteAddressBuffer contexts, uint context_base, uint context_length,
	uint algorithm
) {
	if (mldsa_prehash_bytes(algorithm) == 0u) return false;
	return mldsa_verify_framed(parameter_set, public_keys, pk_base, pk_length,
		signatures, sig_base, sig_length, digests, digest_base, digest_length,
		contexts, context_base, context_length, algorithm, false);
}

// Original-message HashML-DSA: hash locally, preserving the shared verifier body.
// The digest is never materialized in an intermediate device allocation.
bool mldsa_verify_hash_message(
	uint parameter_set,
	RWByteAddressBuffer public_keys, uint pk_base, uint pk_length,
	RWByteAddressBuffer signatures, uint sig_base, uint sig_length,
	RWByteAddressBuffer messages, uint message_base, uint message_length,
	RWByteAddressBuffer contexts, uint context_base, uint context_length,
	uint algorithm
) {
	if (mldsa_prehash_bytes(algorithm) == 0u) return false;
	return mldsa_verify_framed(parameter_set, public_keys, pk_base, pk_length,
		signatures, sig_base, sig_length, messages, message_base, message_length,
		contexts, context_base, context_length, algorithm, true);
}

// Preserve the existing private 65 call sites and schema ABI.
bool mldsa65_verify_external(
	RWByteAddressBuffer public_keys, uint pk_base, uint pk_length,
	RWByteAddressBuffer signatures, uint sig_base, uint sig_length,
	RWByteAddressBuffer messages, uint message_base, uint message_length,
	RWByteAddressBuffer contexts, uint context_base, uint context_length
) {
	return mldsa_verify_external(65u, public_keys, pk_base, pk_length,
		signatures, sig_base, sig_length, messages, message_base, message_length,
		contexts, context_base, context_length);
}
