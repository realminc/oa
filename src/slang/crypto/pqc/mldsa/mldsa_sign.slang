// ML-DSA-44/65/87 signing core, FIPS 204 Algorithms 7 and 34.
// New standards-derived implementation: OA C++ source/cpp/lib/oa/crypto/sign.cpp
// delegates to liboqs; the donor audit records no device implementation.
// Private implementation. Public-fixture proofs do not establish secret-device
// admission, constant-time behavior, or performance qualification.
import shake;
import mldsa_field;
import mldsa_ntt;
import mldsa_sampling;
import mldsa_packing;
import mldsa_sign_workspace_layout;

// Storage policy only: the same signing formulas use a local cache
// or an explicitly owned device workspace. These are cached NTT(s1/s2/t0),
// not a claim that every secret local or compiler spill has been removed.
interface IMldsaSignCache {
	uint read(uint index);
	[mutating] void write(uint index, uint value);
}

struct MldsaLocalSignCache : IMldsaSignCache {
	uint words[MLDSA_SIGN_CACHE_WORDS];
	uint read(uint index) { return words[index]; }
	[mutating] void write(uint index, uint value) { words[index] = value; }
}

struct MldsaBufferSignCache : IMldsaSignCache {
	RWByteAddressBuffer buffer;
	uint base;
	uint read(uint index) { return buffer.Load(base + index * 4u); }
	[mutating] void write(uint index, uint value) { buffer.Store(base + index * 4u, value); }
}

// ExpandMask consumes the complete 576/640-byte XOF stream. Nonce must fit
// FIPS IntegerToBytes(nonce, 2); callers reject exhaustion rather than repeat it.
void mldsa_expand_mask_poly(uint seed[64], uint nonce, uint bits, uint gamma1,
	out uint poly[256]) {
	PqcShake256 sponge;
	pqc_shake256_init(sponge);
	for (uint i = 0u; i < 64u; ++i) pqc_shake256_absorb_byte(sponge, seed[i]);
	pqc_shake256_absorb_byte(sponge, nonce & 255u);
	pqc_shake256_absorb_byte(sponge, nonce >> 8u);
	pqc_shake256_finalize(sponge);
	uint reservoir = 0u, occupied = 0u;
	for (uint i = 0u; i < 256u; ++i) {
		while (occupied < bits) {
			reservoir |= pqc_shake256_squeeze_byte(sponge) << occupied;
			occupied += 8u;
		}
		int value = (int)gamma1 - (int)(reservoir & ((1u << bits) - 1u));
		poly[i] = value < 0 ? (uint)(value + (int)MLDSA_Q) : (uint)value;
		reservoir >>= bits;
		occupied -= bits;
	}
}

// Decode one bounded polynomial from byte-per-uint laboratory transport.
// Every active input byte and the complete fixed-size key are preflighted below.
bool mldsa_sign_unpack_poly(uint key[4896], uint base, uint bits, uint bound,
	out uint poly[256]) {
	uint reservoir = 0u, occupied = 0u, cursor = base;
	for (uint i = 0u; i < 256u; ++i) {
		while (occupied < bits) {
			reservoir |= key[cursor++] << occupied;
			occupied += 8u;
		}
		uint encoded = reservoir & ((1u << bits) - 1u);
		// Small eta encodings have unused bit patterns. t0 uses all 13-bit patterns.
		if (bits < 13u && encoded > 2u * bound) return false;
		int value = (int)bound - (int)encoded;
		poly[i] = value < 0 ? (uint)(value + (int)MLDSA_Q) : (uint)value;
		reservoir >>= bits;
		occupied -= bits;
	}
	return true;
}

// mu is H(tr || M', 64), not a raw message. rnd is 32 explicitly supplied bytes;
// deterministic signing supplies zero, hedged signing requires an admitted RNG.
// Output remains untouched on failure. attempts is private laboratory evidence,
// never a public secret-bearing diagnostic. No arbitrary rejection limit exists.
bool mldsa_sign_mu_cached<T : IMldsaSignCache>(
	uint parameter_set,
	uint key[4896],
	uint mu[64],
	uint rnd[32],
	inout T cache,
	inout uint signature[4627],
	out uint attempts
) {
	attempts = 0u;
	if (parameter_set != 44u && parameter_set != 65u && parameter_set != 87u) return false;
	uint rows = parameter_set == 44u ? 4u : (parameter_set == 65u ? 6u : 8u);
	uint cols = parameter_set == 44u ? 4u : (parameter_set == 65u ? 5u : 7u);
	uint eta = parameter_set == 65u ? 4u : 2u;
	uint eta_bits = eta == 2u ? 3u : 4u;
	uint gamma1 = parameter_set == 44u ? 131072u : 524288u;
	uint gamma2 = parameter_set == 44u ? 95232u : 261888u;
	uint z_bits = parameter_set == 44u ? 18u : 20u;
	uint beta = parameter_set == 44u ? 78u : (parameter_set == 65u ? 196u : 120u);
	uint omega = parameter_set == 44u ? 80u : (parameter_set == 65u ? 55u : 75u);
	uint tau = parameter_set == 44u ? 39u : (parameter_set == 65u ? 49u : 60u);
	uint challenge_bytes = parameter_set == 44u ? 32u : (parameter_set == 65u ? 48u : 64u);
	uint key_bytes = 128u + (cols + rows) * eta_bits * 32u + rows * 416u;
	for (uint i = 0u; i < key_bytes; ++i) if (key[i] > 255u) return false;
	for (uint i = 0u; i < 64u; ++i) if (mu[i] > 255u) return false;
	for (uint i = 0u; i < 32u; ++i) if (rnd[i] > 255u) return false;
	uint rho[8], rho_prime[64];
	for (uint i = 0u; i < 8u; ++i) {
		rho[i] = 0u;
		for (uint j = 0u; j < 4u; ++j) rho[i] |= key[i * 4u + j] << (j * 8u);
	}
	uint poly[256];
	for (uint col = 0u; col < cols; ++col) {
		if (!mldsa_sign_unpack_poly(key, 128u + col * eta_bits * 32u,
			eta_bits, eta, poly)) return false;
		mldsa_ntt(poly);
		for (uint i = 0u; i < 256u; ++i) cache.write(col * 256u + i, poly[i]);
	}
	for (uint row = 0u; row < rows; ++row) {
		if (!mldsa_sign_unpack_poly(key, 128u + (cols + row) * eta_bits * 32u,
			eta_bits, eta, poly)) return false;
		mldsa_ntt(poly);
		for (uint i = 0u; i < 256u; ++i) cache.write((7u + row) * 256u + i, poly[i]);
		if (!mldsa_sign_unpack_poly(key, 128u + (cols + rows) * eta_bits * 32u + row * 416u,
			13u, 4096u, poly)) return false;
		mldsa_ntt(poly);
		for (uint i = 0u; i < 256u; ++i) cache.write((15u + row) * 256u + i, poly[i]);
	}
	PqcShake256 sponge;
	pqc_shake256_init(sponge);
	for (uint i = 0u; i < 32u; ++i) pqc_shake256_absorb_byte(sponge, key[32u + i]);
	for (uint i = 0u; i < 32u; ++i) pqc_shake256_absorb_byte(sponge, rnd[i]);
	for (uint i = 0u; i < 64u; ++i) pqc_shake256_absorb_byte(sponge, mu[i]);
	pqc_shake256_finalize(sponge);
	for (uint i = 0u; i < 64u; ++i) rho_prime[i] = pqc_shake256_squeeze_byte(sponge);
	uint nonce = 0u;
	while (nonce <= 65536u - cols) {
		++attempts;
		uint y[7][256], z[7][256], w[8][256];
		for (uint col = 0u; col < cols; ++col) {
			mldsa_expand_mask_poly(rho_prime, nonce + col, z_bits, gamma1, y[col]);
			for (uint i = 0u; i < 256u; ++i) z[col][i] = y[col][i];
			mldsa_ntt(z[col]);
		}
		nonce += cols;
		pqc_shake256_init(sponge);
		for (uint i = 0u; i < 64u; ++i) pqc_shake256_absorb_byte(sponge, mu[i]);
		for (uint row = 0u; row < rows; ++row) {
			for (uint i = 0u; i < 256u; ++i) w[row][i] = 0u;
			for (uint col = 0u; col < cols; ++col) {
				uint a[256];
				mldsa_expand_a_poly(rho, row, col, a);
				for (uint i = 0u; i < 256u; ++i)
					w[row][i] = mldsa_add(w[row][i], mldsa_mul(a[i], z[col][i]));
			}
			mldsa_intt(w[row]);
			uint bits = parameter_set == 44u ? 6u : 4u;
			uint reservoir = 0u, occupied = 0u;
			for (uint i = 0u; i < 256u; ++i) {
				reservoir |= mldsa_highbits(w[row][i], gamma2) << occupied;
				occupied += bits;
				if (occupied >= 8u) {
					pqc_shake256_absorb_byte(sponge, reservoir & 255u);
					reservoir >>= 8u;
					occupied -= 8u;
				}
			}
		}
		pqc_shake256_finalize(sponge);
		uint challenge[16], c[256];
		for (uint i = 0u; i < 16u; ++i) challenge[i] = 0u;
		for (uint i = 0u; i < challenge_bytes; ++i)
			challenge[i / 4u] |= pqc_shake256_squeeze_byte(sponge) << ((i % 4u) * 8u);
		mldsa_sample_in_ball(challenge, challenge_bytes / 4u, tau, c);
		mldsa_ntt(c);
		bool reject = false;
		for (uint col = 0u; col < cols; ++col) {
			for (uint i = 0u; i < 256u; ++i) z[col][i] = mldsa_mul(c[i], cache.read(col * 256u + i));
			mldsa_intt(z[col]);
			for (uint i = 0u; i < 256u; ++i) z[col][i] = mldsa_add(y[col][i], z[col][i]);
			if (!mldsa_check_norm(z[col], gamma1 - beta)) reject = true;
		}
		if (reject) continue;
		uint hint_positions[80]; // max omega=80; separate row endpoints below
		uint hint_ends[8], hint_count = 0u;
		for (uint row = 0u; row < rows; ++row) {
			uint cs2[256], ct0[256];
			for (uint i = 0u; i < 256u; ++i) {
				cs2[i] = mldsa_mul(c[i], cache.read((7u + row) * 256u + i));
				ct0[i] = mldsa_mul(c[i], cache.read((15u + row) * 256u + i));
			}
			mldsa_intt(cs2);
			mldsa_intt(ct0);
			if (!mldsa_check_norm(ct0, gamma2)) reject = true;
			for (uint i = 0u; i < 256u; ++i) {
				uint r = mldsa_sub(w[row][i], cs2[i]);
				int low = mldsa_lowbits(r, gamma2);
				if ((uint)(low < 0 ? -low : low) >= gamma2 - beta) reject = true;
				uint hint = mldsa_make_hint(mldsa_neg(ct0[i]), mldsa_add(r, ct0[i]), gamma2);
				if (hint != 0u) {
					if (hint_count < omega) hint_positions[hint_count] = i;
					++hint_count;
				}
			}
			hint_ends[row] = hint_count;
		}
		if (reject || hint_count > omega) continue;
		for (uint i = 0u; i < challenge_bytes; ++i)
			signature[i] = (challenge[i / 4u] >> ((i % 4u) * 8u)) & 255u;
		for (uint col = 0u; col < cols; ++col) {
			uint reservoir = 0u, occupied = 0u, cursor = challenge_bytes + col * z_bits * 32u;
			for (uint i = 0u; i < 256u; ++i) {
				reservoir |= (uint)((int)gamma1 - mldsa_centered(z[col][i])) << occupied;
				occupied += z_bits;
				while (occupied >= 8u) {
					signature[cursor++] = reservoir & 255u;
					reservoir >>= 8u;
					occupied -= 8u;
				}
			}
		}
		uint base = challenge_bytes + cols * z_bits * 32u;
		for (uint i = 0u; i < omega; ++i) signature[base + i] = i < hint_count ? hint_positions[i] : 0u;
		for (uint row = 0u; row < rows; ++row) signature[base + omega + row] = hint_ends[row];
		return true;
	}
	// Exhaustion of the specified two-byte nonce space is an explicit failure,
	// never nonce wrap, success with a partial signature, or an unproved small cap.
	return false;
}

// Fixture callers retain their local cache and contract.
bool mldsa_sign_mu(
	uint parameter_set,
	uint key[4896],
	uint mu[64],
	uint rnd[32],
	inout uint signature[4627],
	out uint attempts
) {
	MldsaLocalSignCache cache;
	return mldsa_sign_mu_cached(parameter_set, key, mu, rnd, cache, signature, attempts);
}

// The caller owns and erases this complete workspace after exact completion.
// Range/alignment checks precede all workspace access, even on invalid keys.
// Active cache rows are written before their first read; unused rows are never
// read. The whole allocation must be erased, including unused rows and padding.
bool mldsa_sign_mu_buffer(
	uint parameter_set,
	uint key[4896],
	uint mu[64],
	uint rnd[32],
	RWByteAddressBuffer workspace,
	uint workspace_base,
	inout uint signature[4627],
	out uint attempts
) {
	attempts = 0u;
	if ((workspace_base & 3u) != 0u
		|| !pqc_buffer_range(workspace, workspace_base, MLDSA_SIGN_CACHE_WORDS * 4u)) return false;
	MldsaBufferSignCache cache;
	cache.buffer = workspace;
	cache.base = workspace_base;
	return mldsa_sign_mu_cached(parameter_set, key, mu, rnd, cache, signature, attempts);
}

// FIPS 204 Algorithms 2/7 message hashing. The external pure interface owns
// 0x00 || context_length || context framing; the internal interface receives M'.
// Prehash OID/digest framing is a separate contract and is not implied here.
bool mldsa_sign_message_cached<T : IMldsaSignCache>(uint parameter_set, uint key[4896], uint rnd[32],
	RWByteAddressBuffer messages, uint message_base, uint message_length,
	RWByteAddressBuffer contexts, uint context_base, uint context_length,
	bool external_pure, inout T cache, inout uint signature[4627], out uint attempts) {
	attempts = 0u;
	if (parameter_set != 44u && parameter_set != 65u && parameter_set != 87u) return false;
	if (context_length > 255u || (!external_pure && context_length != 0u)) return false;
	if (!pqc_buffer_range(messages, message_base, message_length)
		|| !pqc_buffer_range(contexts, context_base, context_length)) return false;
	for (uint i = 64u; i < 128u; ++i) if (key[i] > 255u) return false;
	PqcShake256 sponge;
	pqc_shake256_init(sponge);
	for (uint i = 64u; i < 128u; ++i) pqc_shake256_absorb_byte(sponge, key[i]);
	if (external_pure) {
		pqc_shake256_absorb_byte(sponge, 0u);
		pqc_shake256_absorb_byte(sponge, context_length);
		pqc_shake256_absorb_buffer(sponge, contexts, context_base, context_length);
	}
	pqc_shake256_absorb_buffer(sponge, messages, message_base, message_length);
	pqc_shake256_finalize(sponge);
	uint mu[64];
	for (uint i = 0u; i < 64u; ++i) mu[i] = pqc_shake256_squeeze_byte(sponge);
	return mldsa_sign_mu_cached(parameter_set, key, mu, rnd, cache, signature, attempts);
}

// Existing fixture callers retain their local cache.
bool mldsa_sign_message(uint parameter_set, uint key[4896], uint rnd[32],
	RWByteAddressBuffer messages, uint message_base, uint message_length,
	RWByteAddressBuffer contexts, uint context_base, uint context_length,
	bool external_pure, inout uint signature[4627], out uint attempts) {
	MldsaLocalSignCache cache;
	return mldsa_sign_message_cached(parameter_set, key, rnd,
		messages, message_base, message_length, contexts, context_base, context_length,
		external_pure, cache, signature, attempts);
}

// Workspace preflight precedes framing and every secret-derived cache access.
bool mldsa_sign_message_buffer(uint parameter_set, uint key[4896], uint rnd[32],
	RWByteAddressBuffer messages, uint message_base, uint message_length,
	RWByteAddressBuffer contexts, uint context_base, uint context_length,
	bool external_pure, RWByteAddressBuffer workspace, uint workspace_base,
	inout uint signature[4627], out uint attempts) {
	attempts = 0u;
	if ((workspace_base & 3u) != 0u
		|| !pqc_buffer_range(workspace, workspace_base, MLDSA_SIGN_CACHE_WORDS * 4u)) return false;
	MldsaBufferSignCache cache;
	cache.buffer = workspace;
	cache.base = workspace_base;
	return mldsa_sign_message_cached(parameter_set, key, rnd,
		messages, message_base, message_length, contexts, context_base, context_length,
		external_pure, cache, signature, attempts);
}
