// Workgroup-parallel FIPS 204 signing. Same architecture as
// mldsa_verification_par: one 128-lane workgroup per sign operation, all
// polynomial and Keccak state in groupshared memory, scalar-only helper
// parameters, lane 0 serializing sponge/pack steps inside uniform barriers.
// The NTT key cache stays in the caller-bound workspace buffer; no invocation
// carries a coefficient array in registers.
import mldsa_verification_par;
import mldsa_field;
import mldsa_packing;
import mldsa_prehash_layout;
import shake;

// g_poly region map for signing. Slots 30 and 31 carry the per-coefficient
// hint flags and position list so the auxiliary footprint stays ~33KB.
static const uint SG_Y = 0u;             // 7 mask polynomials
static const uint SG_Z = 7u * 256u;      // 7 z = y + intt(c*s1) columns
static const uint SG_A = 14u * 256u;     // 7 ExpandA staging; slots 0/1 hold cs2/ct0
static const uint SG_W = 21u * 256u;     // 8 pre-highbits w rows
static const uint SG_C = 29u * 256u;     // 1 challenge polynomial
static const uint SG_HFLAG = 30u * 256u; // hint flags for the current row
static const uint SG_HPOS = 31u * 256u;  // canonical hint positions (omega max)

groupshared uint g_mu[16];
groupshared uint g_hends[8];
groupshared uint g_hcount;
groupshared uint g_reject;

[noinline]
void wg_sp_absorb_mu() {
	for (uint i = 0u; i < 64u; ++i)
		wg_sp_absorb_byte((g_mu[i / 4u] >> ((i & 3u) * 8u)) & 255u);
}

[noinline]
void wg_squeeze_mu() {
	for (uint w = 0u; w < 16u; ++w)
		g_mu[w] = wg_sp_squeeze_byte() | (wg_sp_squeeze_byte() << 8u)
			| (wg_sp_squeeze_byte() << 16u) | (wg_sp_squeeze_byte() << 24u);
}

// Byte-reservoir unpack with the serial donor's encoded-bound check. One lane
// owns a whole polynomial while sibling lanes unpack the others.
[noinline]
void wg_sign_unpack_poly(RWByteAddressBuffer key, uint base, uint bits, uint bound, uint dest) {
	uint reservoir = 0u, occupied = 0u, cursor = base;
	for (uint i = 0u; i < 256u; ++i) {
		while (occupied < bits) {
			uint a = cursor++;
			reservoir |= ((key.Load(a & ~3u) >> ((a & 3u) * 8u)) & 255u) << occupied;
			occupied += 8u;
		}
		uint encoded = reservoir & ((1u << bits) - 1u);
		if (encoded > 2u * bound) g_flag = 1u;
		int value = (int)bound - (int)encoded;
		g_poly[dest + i] = value < 0 ? (uint)(value + (int)MLDSA_Q) : (uint)value;
		reservoir >>= bits;
		occupied -= bits;
	}
}

// FIPS 204 ExpandMask on a per-lane SHAKE-256 slot: rho_prime || nonce16le,
// then a bit-reservoir decode producing gamma1 - encoded coefficients.
[noinline]
void wg_expand_mask(uint slot, uint nonce, uint bits, uint gamma1, uint dest) {
	for (uint i = 0u; i < 25u; ++i) g_state[slot][i] = uint2(0u, 0u);
	for (uint b = 0u; b < 66u; ++b) {
		uint byte;
		if (b < 64u) byte = (g_rho_prime[b / 4u] >> ((b & 3u) * 8u)) & 255u;
		else byte = b == 64u ? (nonce & 255u) : ((nonce >> 8u) & 255u);
		uint w = b / 8u, boff = b % 8u;
		if (boff < 4u) g_state[slot][w].x ^= (byte << (boff * 8u));
		else g_state[slot][w].y ^= (byte << ((boff - 4u) * 8u));
	}
	{
		uint w = 66u / 8u, boff = 66u % 8u;
		if (boff < 4u) g_state[slot][w].x ^= (0x1Fu << (boff * 8u));
		else g_state[slot][w].y ^= (0x1Fu << ((boff - 4u) * 8u));
	}
	g_state[slot][16].y ^= 0x80000000u;
	wg_keccak(slot);
	uint reservoir = 0u, occupied = 0u, pos = 0u;
	for (uint i = 0u; i < 256u; ++i) {
		while (occupied < bits) {
			if (pos == 136u) {
				wg_keccak(slot);
				pos = 0u;
			}
			reservoir |= wg_xof_byte(slot, pos++) << occupied;
			occupied += 8u;
		}
		int value = (int)gamma1 - (int)(reservoir & ((1u << bits) - 1u));
		g_poly[dest + i] = value < 0 ? (uint)(value + (int)MLDSA_Q) : (uint)value;
		reservoir >>= bits;
		occupied -= bits;
	}
}

// Norm check identical to wg_check_norm but marks a per-attempt rejection
// instead of the fatal flag.
[noinline]
void wg_norm_reject(uint dest, uint bound) {
	for (uint i = 0u; i < 256u; ++i) {
		int c = mldsa_centered(g_poly[dest + i]);
		uint abs_c = c < 0 ? (uint)(-c) : (uint)c;
		if (abs_c >= bound) g_reject = 1u;
	}
}

// One rejection-sampling attempt behind a call boundary so the driver's
// loop passes see a small loop body. Scalar parameters only; all shared
// state flows through groupshared storage.
[noinline]
bool wg_sign_attempt(
	uint nonce, uint lane,
	uint rows, uint cols,
	uint gamma1, uint gamma2,
	uint z_bits, uint w1_bits,
	uint beta, uint omega, uint tau,
	uint challenge_words,
	RWByteAddressBuffer ws, uint ws_base,
	RWByteAddressBuffer sig
) {
	uint challenge_bytes = challenge_words * 4u;
	uint hint_base = challenge_bytes + cols * z_bits * 32u;
	if (lane == 0u) { g_reject = 0u; g_hcount = 0u; }
	if (lane < cols)
		wg_expand_mask(lane, nonce + lane, z_bits, gamma1, SG_Y + lane * 256u);
	wg_sync();
	for (uint i = lane; i < cols * 256u; i += WG_LANES)
		g_poly[SG_Z + i] = g_poly[SG_Y + i];
	wg_sync();
	for (uint c = 0u; c < cols; ++c) wg_ntt(SG_Z + c * 256u, lane);

	if (lane == 0u) {
		wg_sp_init();
		wg_sp_absorb_mu();
	}
	for (uint row = 0u; row < rows; ++row) {
		if (lane < cols)
		wg_expand_a(lane, row, lane, SG_A + lane * 256u);
		wg_sync();
		for (uint i = lane; i < 256u; i += WG_LANES) {
		uint acc = 0u;
		for (uint l = 0u; l < cols; ++l)
			acc = mldsa_add(acc, mldsa_mul(g_poly[SG_A + l * 256u + i],
			g_poly[SG_Z + l * 256u + i]));
		g_poly[SG_W + row * 256u + i] = acc;
		}
		wg_sync();
		wg_intt(SG_W + row * 256u, lane);
		if (lane == 0u) {
		uint reservoir = 0u, occupied = 0u;
		for (uint i = 0u; i < 256u; ++i) {
			reservoir |= mldsa_highbits(g_poly[SG_W + row * 256u + i], gamma2) << occupied;
			occupied += w1_bits;
			if (occupied >= 8u) {
			wg_sp_absorb_byte(reservoir & 255u);
			reservoir >>= 8u;
			occupied -= 8u;
			}
		}
		}
		wg_sync();
	}

	if (lane == 0u) {
		wg_sp_finalize();
		for (uint i = 0u; i < 16u; ++i) g_ctilde[i] = 0u;
		for (uint i = 0u; i < challenge_bytes; ++i)
		g_ctilde[i / 4u] |= wg_sp_squeeze_byte() << ((i % 4u) * 8u);
		wg_sample_ball(SLOT_SPONGE, challenge_words, tau, SG_C);
	}
	wg_sync();
	wg_ntt(SG_C, lane);

	// z = y + intt(c*s1); norm bound checked per column on its own lane.
	for (uint c = 0u; c < cols; ++c) {
		for (uint i = lane; i < 256u; i += WG_LANES)
		g_poly[SG_Z + c * 256u + i] = mldsa_mul(g_poly[SG_C + i],
			ws.Load(ws_base + (c * 256u + i) * 4u));
		wg_sync();
		wg_intt(SG_Z + c * 256u, lane);
		for (uint i = lane; i < 256u; i += WG_LANES)
		g_poly[SG_Z + c * 256u + i] = mldsa_add(g_poly[SG_Y + c * 256u + i],
			g_poly[SG_Z + c * 256u + i]);
		wg_sync();
	}
	if (lane < cols) wg_norm_reject(SG_Z + lane * 256u, gamma1 - beta);
	wg_sync();
	if (g_reject != 0u) return false;

	// r = w - cs2 row, lowbits bound, and hint bits in one lane pass;
	// lane 0 then scans flags into canonical positions and row ends.
	for (uint row = 0u; row < rows; ++row) {
		for (uint i = lane; i < 256u; i += WG_LANES) {
		g_poly[SG_A + i] = mldsa_mul(g_poly[SG_C + i],
			ws.Load(ws_base + ((7u + row) * 256u + i) * 4u));
		g_poly[SG_A + 256u + i] = mldsa_mul(g_poly[SG_C + i],
			ws.Load(ws_base + ((15u + row) * 256u + i) * 4u));
		}
		wg_sync();
		wg_intt(SG_A, lane);
		wg_intt(SG_A + 256u, lane);
		for (uint i = lane; i < 256u; i += WG_LANES) {
		int c0 = mldsa_centered(g_poly[SG_A + 256u + i]);
		if ((uint)(c0 < 0 ? -c0 : c0) >= gamma2) g_reject = 1u;
		uint r = mldsa_sub(g_poly[SG_W + row * 256u + i], g_poly[SG_A + i]);
		int low = mldsa_lowbits(r, gamma2);
		if ((uint)(low < 0 ? -low : low) >= gamma2 - beta) g_reject = 1u;
		g_poly[SG_HFLAG + i] = mldsa_make_hint(
			mldsa_neg(g_poly[SG_A + 256u + i]),
			mldsa_add(r, g_poly[SG_A + 256u + i]), gamma2);
		}
		wg_sync();
		if (lane == 0u) {
		for (uint i = 0u; i < 256u; ++i) {
			if (g_poly[SG_HFLAG + i] != 0u) {
			if (g_hcount < omega) g_poly[SG_HPOS + g_hcount] = i;
			++g_hcount;
			}
		}
		g_hends[row] = g_hcount;
		}
		wg_sync();
	}
	if (g_reject != 0u || g_hcount > omega) return false;

	// Success: challenge words, z columns and the hint block are packed
	// straight into the signature buffer in canonical order.
	if (lane < challenge_words) sig.Store(lane * 4u, g_ctilde[lane]);
	if (lane < cols)
		wg_pack_output(sig, challenge_bytes + lane * z_bits * 32u,
		SG_Z + lane * 256u, z_bits, 0u, gamma1);
	if (lane == 0u) {
		uint total = omega + rows;
		for (uint w = 0u; w * 4u < total; ++w) {
		uint word = 0u;
		for (uint j = 0u; j < 4u; ++j) {
			uint idx = w * 4u + j;
			uint byte = 0u;
			if (idx < total) {
			if (idx < omega)
				byte = idx < g_hcount ? g_poly[SG_HPOS + idx] : 0u;
			else byte = g_hends[idx - omega];
			}
			word |= byte << (j * 8u);
		}
		sig.Store(hint_base + w * 4u, word);
		}
	}
		return true;
}

// Signing core. g_mu must already hold the staged 64-byte message
// representative; each entry computes it from its own framing first.
bool mldsa_sign_core_par(
	uint parameter_set,
	RWByteAddressBuffer key,
	RWByteAddressBuffer rnd,
	RWByteAddressBuffer ws, uint ws_base,
	RWByteAddressBuffer sig,
	uint lane
) {
	uint rows = parameter_set == 44u ? 4u : (parameter_set == 65u ? 6u : 8u);
	uint cols = parameter_set == 44u ? 4u : (parameter_set == 65u ? 5u : 7u);
	uint eta = parameter_set == 65u ? 4u : 2u;
	uint eta_bits = eta == 2u ? 3u : 4u;
	uint eta_bytes = eta_bits * 32u;
	uint gamma1 = parameter_set == 44u ? 131072u : 524288u;
	uint gamma2 = parameter_set == 44u ? 95232u : 261888u;
	uint z_bits = parameter_set == 44u ? 18u : 20u;
	uint w1_bits = parameter_set == 44u ? 6u : 4u;
	uint beta = parameter_set == 44u ? 78u : (parameter_set == 65u ? 196u : 120u);
	uint omega = parameter_set == 44u ? 80u : (parameter_set == 65u ? 55u : 75u);
	uint tau = parameter_set == 44u ? 39u : (parameter_set == 65u ? 49u : 60u);
	uint challenge_words = parameter_set == 44u ? 8u : (parameter_set == 65u ? 12u : 16u);
	uint challenge_bytes = challenge_words * 4u;
	uint hint_base = challenge_bytes + cols * z_bits * 32u;

	// rho staging plus NTT key cache: one lane unpacks each s1/s2/t0
	// polynomial at its fixed cache slot (s1@0, s2@7, t0@15 polynomials), then
	// all transforms run as 128-lane butterflies and land in the workspace.
	if (lane < 8u) g_rho[lane] = key.Load(lane * 4u);
	uint count = cols + 2u * rows;
	if (lane < count) {
		uint base, bits, bound, slot;
		if (lane < cols) {
			base = 128u + lane * eta_bytes;
			bits = eta_bits; bound = eta; slot = lane;
		} else if (lane < cols + rows) {
			base = 128u + lane * eta_bytes;
			bits = eta_bits; bound = eta; slot = 7u + (lane - cols);
		} else {
			uint row = lane - cols - rows;
			base = 128u + (cols + rows) * eta_bytes + row * 416u;
			bits = 13u; bound = 4096u; slot = 15u + row;
		}
		wg_sign_unpack_poly(key, base, bits, bound, slot * 256u);
	}
	if (wg_failed(lane)) return false;
	for (uint c = 0u; c < cols; ++c) wg_ntt(c * 256u, lane);
	for (uint r = 0u; r < rows; ++r) {
		wg_ntt((7u + r) * 256u, lane);
		wg_ntt((15u + r) * 256u, lane);
	}
	for (uint i = lane; i < cols * 256u; i += WG_LANES)
		ws.Store(ws_base + i * 4u, g_poly[i]);
	for (uint i = lane; i < rows * 256u; i += WG_LANES) {
		ws.Store(ws_base + (7u * 256u + i) * 4u, g_poly[7u * 256u + i]);
		ws.Store(ws_base + (15u * 256u + i) * 4u, g_poly[15u * 256u + i]);
	}

	// rho_prime = H(K || rnd || mu); g_mu arrives staged by the caller.
	for (uint i = lane; i < 16u; i += WG_LANES) g_rho_prime[i] = 0u;
	if (lane == 0u) {
		wg_sp_init();
		wg_sp_absorb_buffer(key, 32u, 32u);
		wg_sp_absorb_buffer(rnd, 0u, 32u);
		wg_sp_absorb_mu();
		wg_sp_finalize();
		for (uint i = 0u; i < 64u; ++i)
			g_rho_prime[i / 4u] |= wg_sp_squeeze_byte() << ((i % 4u) * 8u);
	}
	wg_sync();

	// Rejection loop. nonce is a uniform local; every continue/return decision
	// is gated on shared flags read after a barrier, so control flow stays
	// uniform across the workgroup.
	uint nonce = 0u;
	while (nonce <= 65536u - cols) {
		if (wg_sign_attempt(nonce, lane, rows, cols, gamma1, gamma2, z_bits, w1_bits,
			beta, omega, tau, challenge_words, ws, ws_base, sig))
			return true;
		nonce += cols;
	}
	return false;
}


// Entry points per dispatch wrapper. Bases are zero because each private
// kernel binds whole buffers; validation identical on every lane precedes the
// first barrier so early exits stay uniform.
bool mldsa_sign_mu_par(
	uint parameter_set,
	RWByteAddressBuffer key,
	RWByteAddressBuffer rnd,
	RWByteAddressBuffer mus, uint mu_base,
	RWByteAddressBuffer ws, uint ws_base,
	RWByteAddressBuffer sig,
	uint lane
) {
	if (parameter_set != 44u && parameter_set != 65u && parameter_set != 87u) return false;
	if (!pqc_buffer_range(mus, mu_base, 64u)) return false;
	if (lane == 0u) {
		g_flag = 0u;
		for (uint w = 0u; w < 16u; ++w) g_mu[w] = mus.Load(mu_base + w * 4u);
	}
	wg_sync();
	return mldsa_sign_core_par(parameter_set, key, rnd, ws, ws_base, sig, lane);
}

