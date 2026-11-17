// Workgroup-parallel FIPS 203 ML-KEM, Algorithms 6-18 on shared storage.
// One workgroup of 128 lanes owns one keygen, encapsulation or decapsulation.
// Polynomial and Keccak stream state lives in groupshared so no invocation
// carries a coefficient array in registers; XOF rejection streams run one
// stream per lane on indexed state slots, and NTT, packing, pointwise and
// compression phases run with barrier-separated parallelism. Every helper
// takes scalar arguments and reads or writes module shared storage: SPIR-V
// function calls carrying array pointers defeat the current-generation
// driver compiler, so no parameter carries an array or sponge structure.
// The public-data contract matches mlkem_kem.slang / mlkem_kpke.slang exactly.
import keccak;
import mlkem_field;
import mlkem_ntt;
import mlkem_compression;

static const uint WG_LANES = 128u;

// Keccak state slots: 0..15 per-lane XOF streams (matrix and noise), 16 the
// serial KEM sponge for H, G and J.
static const uint SLOT_SPONGE = 16u;
static const uint SLOTS = 17u;

// Flat groupshared polynomial workspace, offsets in coefficients.
static const uint P_S = 0u;            // 4 polynomials: secret vector s / y / u
static const uint P_E = 4u * 256u;     // 4 polynomials: noise e / decoded t
static const uint P_A = 8u * 256u;     // 4 polynomials: one A row staging
static const uint P_ACC = 12u * 256u;  // 1 accumulator polynomial
static const uint P_TMP = 13u * 256u;  // 1 scratch polynomial
static const uint P_TOTAL = 14u * 256u;

groupshared uint g_poly[P_TOTAL];
groupshared uint2 g_state[SLOTS][25];
groupshared uint g_sp_pos;
groupshared uint g_rho[8];
groupshared uint g_sigma[8];
groupshared uint g_r[8];
groupshared uint g_key[8];
groupshared uint g_hash[8];
groupshared uint g_msg[8];
groupshared uint g_d64[16];
groupshared uint g_bytes[1568];
groupshared uint g_flag;

void wg_sync() {
	GroupMemoryBarrierWithGroupSync();
}

// ---------------------------------------------------------------------------
// Keccak-f[1600] on a shared state slot; donor body unchanged.
// ---------------------------------------------------------------------------

void wg_keccak(uint slot) {
	for (uint round = 0u; round < 24u; ++round) {
		uint2 columns[5];
		for (uint i = 0u; i < 5u; ++i) {
			columns[i] = g_state[slot][i] ^ g_state[slot][5 + i] ^ g_state[slot][10 + i]
				^ g_state[slot][15 + i] ^ g_state[slot][20 + i];
		}
		uint2 differences[5];
		for (uint i = 0u; i < 5u; ++i) {
			differences[i] = columns[(i + 4u) % 5u] ^ rotl64(columns[(i + 1u) % 5u], 1u);
		}
		for (uint i = 0u; i < 25u; ++i) g_state[slot][i] ^= differences[i % 5u];

		uint2 rotated[25];
		for (uint i = 0u; i < 25u; ++i) {
			uint x = i % 5u;
			uint y = i / 5u;
			rotated[5u * ((2u * x + 3u * y) % 5u) + y] = rotl64(g_state[slot][i], ROT[i]);
		}
		for (uint y = 0u; y < 5u; ++y) {
			uint base = y * 5u;
			g_state[slot][base] = rotated[base] ^ (~rotated[base + 1u] & rotated[base + 2u]);
			g_state[slot][base + 1u] = rotated[base + 1u] ^ (~rotated[base + 2u] & rotated[base + 3u]);
			g_state[slot][base + 2u] = rotated[base + 2u] ^ (~rotated[base + 3u] & rotated[base + 4u]);
			g_state[slot][base + 3u] = rotated[base + 3u] ^ (~rotated[base + 4u] & rotated[base]);
			g_state[slot][base + 4u] = rotated[base + 4u] ^ (~rotated[base] & rotated[base + 1u]);
		}
		g_state[slot][0] ^= uint2(RC_LO[round], RC_HI[round]);
	}
}

uint wg_xof_byte(uint slot, uint pos) {
	uint lane = pos / 8u;
	uint boff = pos % 8u;
	uint word = (boff < 4u) ? g_state[slot][lane].x : g_state[slot][lane].y;
	uint shift = (boff < 4u) ? (boff * 8u) : ((boff - 4u) * 8u);
	return (word >> shift) & 0xFFu;
}

uint2 wg_base_multiply(uint2 a, uint2 b, uint gamma) {
	uint even = mlkem_add(mlkem_mul(a.x, b.x), mlkem_mul(mlkem_mul(a.y, b.y), gamma));
	uint odd = mlkem_add(mlkem_mul(a.x, b.y), mlkem_mul(a.y, b.x));
	return uint2(even, odd);
}

// ---------------------------------------------------------------------------
// Per-lane XOF streams: one stream per lane on its own state slot, so up to
// sixteen rejection streams run concurrently. Seed layout matches
// mlkem_seed_xof: 32-byte seed then the suffix bytes, rate boundary pad.
// ---------------------------------------------------------------------------

void wg_stream_seed(uint slot, uint word_index, uint value) {
	if ((word_index & 1u) == 0u) g_state[slot][word_index / 2u].x = value;
	else g_state[slot][word_index / 2u].y = value;
}

// Matrix stream: XOF(rho || firstIndex || secondIndex), rate 168.
void wg_matrix_one(uint slot, uint firstIndex, uint secondIndex, uint dest_base) {
	for (uint i = 0u; i < 25u; ++i) g_state[slot][i] = uint2(0u, 0u);
	for (uint w = 0u; w < 8u; ++w) wg_stream_seed(slot, w, g_rho[w]);
	g_state[slot][4].x = (firstIndex & 255u) | ((secondIndex & 255u) << 8u) | (31u << 16u);
	g_state[slot][20].y ^= 0x80000000u;
	wg_keccak(slot);

	uint count = 0u, pos = 0u;
	while (count < 256u) {
		if (pos + 3u > 168u) {
			wg_keccak(slot);
			pos = 0u;
		}
		uint a = wg_xof_byte(slot, pos);
		uint b = wg_xof_byte(slot, pos + 1u);
		uint c = wg_xof_byte(slot, pos + 2u);
		pos += 3u;
		uint first = a | ((b & 15u) << 8u);
		uint second = (b >> 4u) | (c << 4u);
		if (first < MLKEM_Q && count < 256u) g_poly[dest_base + count++] = first;
		if (second < MLKEM_Q && count < 256u) g_poly[dest_base + count++] = second;
	}
}

// Noise stream: PRF_eta(seed || nonce) = SHAKE-256, rate 136, CBD sampling.
// The seed source is g_sigma, which the caller stages with r or sigma.
void wg_noise_one(uint slot, uint nonce, uint dest_base, uint eta) {
	for (uint i = 0u; i < 25u; ++i) g_state[slot][i] = uint2(0u, 0u);
	for (uint w = 0u; w < 8u; ++w) wg_stream_seed(slot, w, g_sigma[w]);
	g_state[slot][4].x = (nonce & 255u) | (31u << 8u);
	g_state[slot][16].y ^= 0x80000000u;
	wg_keccak(slot);

	uint mask = (1u << eta) - 1u;
	uint pos = 0u, reservoir = 0u, available = 0u;
	for (uint i = 0u; i < 256u; ++i) {
		while (available < 2u * eta) {
			if (pos == 136u) {
				wg_keccak(slot);
				pos = 0u;
			}
			reservoir |= wg_xof_byte(slot, pos++) << available;
			available += 8u;
		}
		uint positive = countbits(reservoir & mask);
		uint negative = countbits((reservoir >> eta) & mask);
		g_poly[dest_base + i] = mlkem_sub(positive, negative);
		reservoir >>= 2u * eta;
		available -= 2u * eta;
	}
}

// ---------------------------------------------------------------------------
// Workgroup-parallel NTT / INTT on a shared polynomial. Butterfly lane in a
// stage of span length has block = lane / length and j = block*2*length +
// lane % length; the zeta indices are the serial donor's counters flattened:
// stage `length` consumes ZETA[128/length + block] (forward) or
// ZETA[2*(128/length) - 1 - block] (inverse).
// ---------------------------------------------------------------------------

void wg_ntt(uint base, uint lane) {
	for (uint length = 128u; length >= 2u; length >>= 1u) {
		uint block = lane / length;
		uint j = block * 2u * length + (lane % length);
		uint zeta = MLKEM_ZETA[128u / length + block];
		uint product = mlkem_mul(zeta, g_poly[base + j + length]);
		uint value = g_poly[base + j];
		g_poly[base + j] = mlkem_add(value, product);
		g_poly[base + j + length] = mlkem_sub(value, product);
		wg_sync();
	}
}

void wg_intt(uint base, uint lane) {
	for (uint length = 2u; length <= 128u; length <<= 1u) {
		uint block = lane / length;
		uint j = block * 2u * length + (lane % length);
		uint zeta = MLKEM_ZETA[2u * (128u / length) - 1u - block];
		uint a = g_poly[base + j], b = g_poly[base + j + length];
		g_poly[base + j] = mlkem_add(a, b);
		g_poly[base + j + length] = mlkem_mul(zeta, mlkem_sub(b, a));
		wg_sync();
	}
	for (uint j = lane; j < 256u; j += WG_LANES)
		g_poly[base + j] = mlkem_mul(g_poly[base + j], 3303u);
	wg_sync();
}

// Pairwise NTT-domain accumulate over 128 quadratic residues, Algorithm 11.
// Lane owns pair lane.
void wg_accumulate(uint acc_base, uint left_base, uint vec_base, uint lane) {
	uint gamma = MLKEM_ZETA[64u + lane / 2u];
	if ((lane & 1u) != 0u) gamma = mlkem_neg(gamma);
	uint j = 2u * lane;
	uint2 p = wg_base_multiply(
		uint2(g_poly[left_base + j], g_poly[left_base + j + 1u]),
		uint2(g_poly[vec_base + j], g_poly[vec_base + j + 1u]), gamma);
	g_poly[acc_base + j] = mlkem_add(g_poly[acc_base + j], p.x);
	g_poly[acc_base + j + 1u] = mlkem_add(g_poly[acc_base + j + 1u], p.y);
}

void wg_zero_poly(uint base, uint lane) {
	for (uint i = lane; i < 256u; i += WG_LANES) g_poly[base + i] = 0u;
	wg_sync();
}

void wg_copy_poly(uint dst_base, uint src_base, uint lane) {
	for (uint i = lane; i < 256u; i += WG_LANES)
		g_poly[dst_base + i] = g_poly[src_base + i];
	wg_sync();
}

void wg_add_poly(uint dst_base, uint src_base, uint lane) {
	for (uint i = lane; i < 256u; i += WG_LANES)
		g_poly[dst_base + i] = mlkem_add(g_poly[dst_base + i], g_poly[src_base + i]);
	wg_sync();
}

void wg_compress_add_noise(uint acc_base, uint noise_base, uint bits, uint lane) {
	for (uint i = lane; i < 256u; i += WG_LANES)
		g_poly[acc_base + i] = mlkem_compress(
			mlkem_add(g_poly[acc_base + i], g_poly[noise_base + i]), bits);
	wg_sync();
}

void wg_decompress_poly(uint base, uint bits, uint lane) {
	for (uint i = lane; i < 256u; i += WG_LANES)
		g_poly[base + i] = mlkem_decompress(g_poly[base + i], bits);
	wg_sync();
}

void wg_compress_poly(uint base, uint bits, uint lane) {
	for (uint i = lane; i < 256u; i += WG_LANES)
		g_poly[base + i] = mlkem_compress(g_poly[base + i], bits);
	wg_sync();
}

// ---------------------------------------------------------------------------
// Bit packing. Byte p of an encode covers bit window [8p, 8p+8) of the
// coefficient stream; coefficient c of a decode covers [c*bits,(c+1)*bits).
// Each lane extracts its own window — no serial reservoir.
// ---------------------------------------------------------------------------

// Encode a shared poly into g_bytes byte-valued staging at dst_base.
void wg_encode_shared(uint poly_base, uint bits, uint dst_base, uint lane) {
	uint total = 32u * bits;
	for (uint p = lane; p < total; p += WG_LANES) {
		uint value = 0u;
		for (uint i = 0u; i < 8u; ++i) {
			uint c = (p * 8u + i) / bits;
			value |= ((g_poly[poly_base + c] >> ((p * 8u + i) - c * bits)) & 1u) << i;
		}
		g_bytes[dst_base + p] = value;
	}
	wg_sync();
}

// Decode from g_bytes byte-valued staging into a shared poly. For bits==12
// canonicality failures accumulate on g_flag before reduction.
void wg_decode_shared(uint src_base, uint bits, uint poly_base, uint lane) {
	uint mask = (1u << bits) - 1u;
	for (uint c = lane; c < 256u; c += WG_LANES) {
		uint bitpos = c * bits;
		uint value = 0u;
		for (uint i = 0u; i < bits; ++i) {
			uint byte = g_bytes[src_base + (bitpos + i) / 8u];
			value |= ((byte >> ((bitpos + i) % 8u)) & 1u) << i;
		}
		if (bits == 12u && value >= MLKEM_Q) {
			g_flag = 1u;
			value -= MLKEM_Q;
		}
		g_poly[poly_base + c] = value & mask;
	}
	wg_sync();
}

// Decode directly from a storage buffer's packed bytes.
void wg_decode_buf(RWByteAddressBuffer buf, uint byte_base, uint bits, uint poly_base, uint lane) {
	uint mask = (1u << bits) - 1u;
	for (uint c = lane; c < 256u; c += WG_LANES) {
		uint bitpos = c * bits;
		uint value = 0u;
		for (uint i = 0u; i < bits; ++i) {
			uint bi = byte_base + (bitpos + i) / 8u;
			uint word = buf.Load(bi & ~3u);
			uint byte = (word >> ((bi & 3u) * 8u)) & 255u;
			value |= ((byte >> ((bitpos + i) % 8u)) & 1u) << i;
		}
		if (bits == 12u && value >= MLKEM_Q) {
			g_flag = 1u;
			value -= MLKEM_Q;
		}
		g_poly[poly_base + c] = value & mask;
	}
	wg_sync();
}

// Stage packed buffer bytes into g_bytes byte-valued staging.
void wg_load_bytes(RWByteAddressBuffer buf, uint src, uint dst_base, uint n, uint lane) {
	for (uint w = lane; w < n / 4u; w += WG_LANES) {
		uint word = buf.Load(src + 4u * w);
		g_bytes[dst_base + 4u * w] = word & 255u;
		g_bytes[dst_base + 4u * w + 1u] = (word >> 8u) & 255u;
		g_bytes[dst_base + 4u * w + 2u] = (word >> 16u) & 255u;
		g_bytes[dst_base + 4u * w + 3u] = (word >> 24u) & 255u;
	}
	wg_sync();
}

// Store g_bytes byte-valued staging as packed words.
void wg_store_bytes(RWByteAddressBuffer buf, uint dst, uint src_base, uint n, uint lane) {
	for (uint w = lane; w < n / 4u; w += WG_LANES) {
		uint word = g_bytes[src_base + 4u * w]
			| (g_bytes[src_base + 4u * w + 1u] << 8u)
			| (g_bytes[src_base + 4u * w + 2u] << 16u)
			| (g_bytes[src_base + 4u * w + 3u] << 24u);
		buf.Store(dst + 4u * w, word);
	}
	wg_sync();
}

// ---------------------------------------------------------------------------
// Serial KEM sponge on slot 16 with word-parallel absorb. All absorbed
// regions are multiples of four bytes starting at word-aligned positions, so
// lanes own distinct state words and XOR stays commutative and exact.
// ---------------------------------------------------------------------------

void wg_sp_init(uint lane) {
	if (lane < 25u) g_state[SLOT_SPONGE][lane] = uint2(0u, 0u);
	if (lane == 0u) g_sp_pos = 0u;
	wg_sync();
}

void wg_sp_absorb_shared(uint rate, uint base, uint n, uint lane) {
	uint pos = g_sp_pos;
	uint done = 0u;
	while (done < n) {
		uint take = min(rate - pos, n - done);
		uint wbase = pos / 4u;
		if (4u * lane < take) {
			uint word = g_bytes[base + done + 4u * lane]
				| (g_bytes[base + done + 4u * lane + 1u] << 8u)
				| (g_bytes[base + done + 4u * lane + 2u] << 16u)
				| (g_bytes[base + done + 4u * lane + 3u] << 24u);
			uint wpos = wbase + lane;
			if ((wpos & 1u) == 0u) g_state[SLOT_SPONGE][wpos / 2u].x ^= word;
			else g_state[SLOT_SPONGE][wpos / 2u].y ^= word;
		}
		pos += take;
		done += take;
		wg_sync();
		if (pos == rate) {
			if (lane == 0u) wg_keccak(SLOT_SPONGE);
			pos = 0u;
			wg_sync();
		}
	}
	if (lane == 0u) g_sp_pos = pos;
	wg_sync();
}

void wg_sp_absorb_buf(uint rate, RWByteAddressBuffer buf, uint base, uint n, uint lane) {
	uint pos = g_sp_pos;
	uint done = 0u;
	while (done < n) {
		uint take = min(rate - pos, n - done);
		uint wbase = pos / 4u;
		if (4u * lane < take) {
			uint word = buf.Load(base + done + 4u * lane);
			uint wpos = wbase + lane;
			if ((wpos & 1u) == 0u) g_state[SLOT_SPONGE][wpos / 2u].x ^= word;
			else g_state[SLOT_SPONGE][wpos / 2u].y ^= word;
		}
		pos += take;
		done += take;
		wg_sync();
		if (pos == rate) {
			if (lane == 0u) wg_keccak(SLOT_SPONGE);
			pos = 0u;
			wg_sync();
		}
	}
	if (lane == 0u) g_sp_pos = pos;
	wg_sync();
}

// Destructive finish: suffix pad, one permutation, squeeze n bytes as u32
// words into g_d64. n <= 64 covers every caller.
void wg_sp_finish(uint rate, uint suffix, uint n, uint lane) {
	uint pos = g_sp_pos;
	uint li = pos / 8u;
	if (lane == 0u) {
		if ((pos % 8u) < 4u) g_state[SLOT_SPONGE][li].x ^= suffix << ((pos % 4u) * 8u);
		else g_state[SLOT_SPONGE][li].y ^= suffix << ((pos % 8u - 4u) * 8u);
		g_state[SLOT_SPONGE][rate / 8u - 1u].y ^= 0x80000000u;
		wg_keccak(SLOT_SPONGE);
	}
	wg_sync();
	for (uint w = lane; w < n / 4u; w += WG_LANES)
		g_d64[w] = (w & 1u) == 0u ? g_state[SLOT_SPONGE][w / 2u].x
			: g_state[SLOT_SPONGE][w / 2u].y;
	wg_sync();
}

// G(d || k) expand seed: single-block SHA3-512 on lane 0. d arrives staged in
// g_rho; outputs land in g_rho and g_sigma.
void wg_expand_seed(uint k, uint lane) {
	if (lane != 0u) return;
	for (uint i = 0u; i < 25u; ++i) g_state[SLOT_SPONGE][i] = uint2(0u, 0u);
	for (uint i = 0u; i < 8u; ++i) wg_stream_seed(SLOT_SPONGE, i, g_rho[i]);
	g_state[SLOT_SPONGE][4].x = k | 0x600u;
	g_state[SLOT_SPONGE][8].y = 0x80000000u;
	wg_keccak(SLOT_SPONGE);
	for (uint i = 0u; i < 8u; ++i) {
		g_rho[i] = (i & 1u) == 0u ? g_state[SLOT_SPONGE][i / 2u].x
			: g_state[SLOT_SPONGE][i / 2u].y;
		g_sigma[i] = (i & 1u) == 0u ? g_state[SLOT_SPONGE][4u + i / 2u].x
			: g_state[SLOT_SPONGE][4u + i / 2u].y;
	}
}

// ---------------------------------------------------------------------------
// Workgroup-parallel K-PKE keygen, encrypt and decrypt. Ciphertext/ek staging
// uses g_bytes[0..1568); private-key s encodes go straight to the private
// buffer so ek staging stays live for H.
// ---------------------------------------------------------------------------

void wg_kpke_keygen(uint k, uint lane) {
	uint eta = k == 2u ? 3u : 2u;
	// Noise streams on lanes 0..2k-1: s_i = nonce i, e_i = nonce k+i.
	if (lane < 2u * k) {
		uint dest = lane < k ? P_S + lane * 256u : P_E + (lane - k) * 256u;
		wg_noise_one(lane, lane, dest, eta);
	}
	wg_sync();
	for (uint p = 0u; p < k; ++p) {
		wg_ntt(P_S + p * 256u, lane);
		wg_ntt(P_E + p * 256u, lane);
	}
	// t_i = e_i + sum_j A[j][i] * s_j.
	for (uint i = 0u; i < k; ++i) {
		if (lane < k) wg_matrix_one(lane, lane, i, P_A + lane * 256u);
		wg_sync();
		for (uint j = lane; j < 256u; j += WG_LANES)
			g_poly[P_ACC + j] = g_poly[P_E + i * 256u + j];
		wg_sync();
		for (uint j = 0u; j < k; ++j) {
			wg_accumulate(P_ACC, P_A + j * 256u, P_S + j * 256u, lane);
			wg_sync();
		}
		wg_encode_shared(P_ACC, 12u, i * 384u, lane);
	}
	// ek tail: rho bytes.
	for (uint i = lane; i < 32u; i += WG_LANES)
		g_bytes[384u * k + i] = (g_rho[i / 4u] >> ((i % 4u) * 8u)) & 255u;
	wg_sync();
}

// Encrypt into g_bytes[0..cl). ek t polys arrive decoded from ek_buf at
// ek_base (public buffer for encaps, private dk segment for decaps). rho is
// read from the same segment's tail. Canonicality accumulates on g_flag.
void wg_kpke_encrypt(uint k, uint lane, RWByteAddressBuffer ek_buf, uint ek_base) {
	uint eta = k == 2u ? 3u : 2u;
	uint du = k == 4u ? 11u : 10u, dv = k == 4u ? 5u : 4u;
	// rho words from ek tail.
	if (lane < 8u) g_rho[lane] = ek_buf.Load(ek_base + 384u * k + 4u * lane);
	wg_sync();
	// Noise streams: y_i nonce i (eta), e_i nonce k+i (2), e2 nonce 2k (2).
	if (lane < 2u * k + 1u) {
		uint nonce = lane;
		uint dest, e;
		if (nonce < k) { dest = P_S + nonce * 256u; e = eta; }
		else if (nonce < 2u * k) { dest = P_E + (nonce - k) * 256u; e = 2u; }
		else { dest = P_TMP; e = 2u; }
		wg_noise_one(nonce, nonce, dest, e);
	}
	wg_sync();
	for (uint p = 0u; p < k; ++p) wg_ntt(P_S + p * 256u, lane);
	// u_i = intt(sum_j A^T[i][j] * y_j) + e_i; compress du; encode c1_i.
	for (uint i = 0u; i < k; ++i) {
		if (lane < k) wg_matrix_one(lane, i, lane, P_A + lane * 256u);
		wg_sync();
		wg_zero_poly(P_ACC, lane);
		for (uint j = 0u; j < k; ++j) {
			wg_accumulate(P_ACC, P_A + j * 256u, P_S + j * 256u, lane);
			wg_sync();
		}
		wg_intt(P_ACC, lane);
		wg_compress_add_noise(P_ACC, P_E + i * 256u, du, lane);
		wg_encode_shared(P_ACC, du, i * 32u * du, lane);
	}
	// v = intt(sum_i t_i * y_i) + e2 + m; decode t_i from ek directly.
	if (lane == 0u) g_flag = 0u;
	wg_sync();
	wg_zero_poly(P_ACC, lane);
	for (uint i = 0u; i < k; ++i) {
		wg_decode_buf(ek_buf, ek_base + i * 384u, 12u, P_A + i * 256u, lane);
		wg_accumulate(P_ACC, P_A + i * 256u, P_S + i * 256u, lane);
		wg_sync();
	}
	wg_intt(P_ACC, lane);
	wg_add_poly(P_ACC, P_TMP, lane);
	for (uint i = lane; i < 256u; i += WG_LANES) {
		uint bit = (g_msg[i / 32u] >> (i % 32u)) & 1u;
		g_poly[P_ACC + i] = mlkem_add(g_poly[P_ACC + i], mlkem_decompress(bit, 1u));
	}
	wg_sync();
	wg_compress_poly(P_ACC, dv, lane);
	wg_encode_shared(P_ACC, dv, k * 32u * du, lane);
}

// Decrypt from g_bytes ct staging into g_msg (8 words = 32 bytes).
void wg_kpke_decrypt(uint k, uint lane, RWByteAddressBuffer private_buf) {
	uint du = k == 4u ? 11u : 10u, dv = k == 4u ? 5u : 4u;
	// u_i: decode c1_i, decompress, NTT.
	for (uint i = 0u; i < k; ++i) {
		wg_decode_shared(i * 32u * du, du, P_S + i * 256u, lane);
		wg_decompress_poly(P_S + i * 256u, du, lane);
		wg_ntt(P_S + i * 256u, lane);
	}
	// s_i: decode from dk prefix in the private buffer.
	wg_zero_poly(P_ACC, lane);
	for (uint i = 0u; i < k; ++i) {
		wg_decode_buf(private_buf, i * 384u, 12u, P_A + i * 256u, lane);
		wg_accumulate(P_ACC, P_A + i * 256u, P_S + i * 256u, lane);
		wg_sync();
	}
	wg_intt(P_ACC, lane);
	// v: decode c2, decompress, subtract, compress to 1 bit.
	wg_decode_shared(k * 32u * du, dv, P_TMP, lane);
	for (uint i = lane; i < 256u; i += WG_LANES)
		g_poly[P_TMP + i] = mlkem_compress(
			mlkem_sub(mlkem_decompress(g_poly[P_TMP + i], dv), g_poly[P_ACC + i]), 1u);
	wg_sync();
	// Message words.
	for (uint w = lane; w < 8u; w += WG_LANES) {
		uint word = 0u;
		for (uint b = 0u; b < 32u; ++b)
			word |= (g_poly[P_TMP + w * 32u + b] & 1u) << b;
		g_msg[w] = word;
	}
	wg_sync();
}

// ---------------------------------------------------------------------------
// FIPS 203 KEM layer. Dispatch bodies keep admission checks; these entries
// assume validated sizes.
// ---------------------------------------------------------------------------

void mlkem_keygen_par(
	uint k,
	uint lane,
	RWByteAddressBuffer seed_buf,
	RWByteAddressBuffer private_buf,
	RWByteAddressBuffer public_buf
) {
	uint el = 384u * k + 32u;
	// d words into g_rho, z words into g_d64[8..16).
	if (lane < 8u) g_rho[lane] = seed_buf.Load(4u * lane);
	else if (lane < 16u) g_d64[lane] = seed_buf.Load(4u * lane);
	wg_sync();
	wg_expand_seed(k, lane);
	wg_sync();
	wg_kpke_keygen(k, lane);
	// ek staging is g_bytes[0..el); store public key.
	wg_store_bytes(public_buf, 0u, 0u, el, lane);
	// s_i encodes for dk: lane w<96 writes word w of each 384-byte segment.
	for (uint i = 0u; i < k; ++i) {
		if (lane < 96u) {
			uint p = lane * 4u;
			uint value = 0u;
			for (uint b = 0u; b < 32u; ++b) {
				uint bitpos = p * 8u + b;
				uint c = bitpos / 12u;
				value |= ((g_poly[P_S + i * 256u + c] >> (bitpos - c * 12u)) & 1u) << b;
			}
			private_buf.Store(i * 384u + 4u * lane, value);
		}
	}
	wg_sync();
	// ek copy into dk segment.
	for (uint w = lane; w < el / 4u; w += WG_LANES) {
		uint word = g_bytes[4u * w]
			| (g_bytes[4u * w + 1u] << 8u)
			| (g_bytes[4u * w + 2u] << 16u)
			| (g_bytes[4u * w + 3u] << 24u);
		private_buf.Store(384u * k + 4u * w, word);
	}
	wg_sync();
	// H(ek) = SHA3-256 over staged ek.
	wg_sp_init(lane);
	wg_sp_absorb_shared(136u, 0u, el, lane);
	wg_sp_finish(136u, 6u, 32u, lane);
	for (uint w = lane; w < 8u; w += WG_LANES)
		private_buf.Store(768u * k + 32u + 4u * w, g_d64[w]);
	// z tail from g_d64[8..16).
	for (uint w = lane; w < 8u; w += WG_LANES)
		private_buf.Store(768u * k + 64u + 4u * w, g_d64[8u + w]);
	wg_sync();
}

void mlkem_encaps_par(
	uint k,
	uint lane,
	RWByteAddressBuffer seed_buf,
	RWByteAddressBuffer public_buf,
	RWByteAddressBuffer shared_buf,
	RWByteAddressBuffer ciphertext_buf,
	RWByteAddressBuffer status_buf
) {
	uint el = 384u * k + 32u;
	uint cl = 32u * (k == 4u ? 11u * k + 5u : 10u * k + 4u);
	// H(ek) = SHA3-256 straight from the public buffer.
	wg_sp_init(lane);
	wg_sp_absorb_buf(136u, public_buf, 0u, el, lane);
	wg_sp_finish(136u, 6u, 32u, lane);
	for (uint w = lane; w < 8u; w += WG_LANES) g_hash[w] = g_d64[w];
	// ek canonicality: canonical decode check on the k t polys.
	if (lane == 0u) g_flag = 0u;
	wg_sync();
	for (uint i = 0u; i < k; ++i)
		wg_decode_buf(public_buf, i * 384u, 12u, P_A + i * 256u, lane);
	bool valid = g_flag == 0u;
	if (lane == 0u) status_buf.Store(0u, uint(valid));
	wg_sync();
	if (!valid) {
		for (uint i = lane; i < 8u; i += WG_LANES) shared_buf.Store(4u * i, 0u);
		for (uint w = lane; w < cl / 4u; w += WG_LANES) ciphertext_buf.Store(4u * w, 0u);
		wg_sync();
		return;
	}
	// message words.
	if (lane < 8u) g_msg[lane] = seed_buf.Load(4u * lane);
	wg_sync();
	// G(m || h) = SHA3-512, staged as byte values.
	for (uint i = lane; i < 8u; i += WG_LANES) {
		uint m = g_msg[i], h = g_hash[i];
		g_bytes[4u * i] = m & 255u;
		g_bytes[4u * i + 1u] = (m >> 8u) & 255u;
		g_bytes[4u * i + 2u] = (m >> 16u) & 255u;
		g_bytes[4u * i + 3u] = (m >> 24u) & 255u;
		g_bytes[32u + 4u * i] = h & 255u;
		g_bytes[32u + 4u * i + 1u] = (h >> 8u) & 255u;
		g_bytes[32u + 4u * i + 2u] = (h >> 16u) & 255u;
		g_bytes[32u + 4u * i + 3u] = (h >> 24u) & 255u;
	}
	wg_sync();
	wg_sp_init(lane);
	wg_sp_absorb_shared(72u, 0u, 64u, lane);
	wg_sp_finish(72u, 6u, 64u, lane);
	for (uint w = lane; w < 8u; w += WG_LANES) {
		g_key[w] = g_d64[w];
		g_sigma[w] = g_d64[8u + w]; // r drives the noise/matrix streams
	}
	wg_sync();
	if (lane == 0u) g_flag = 0u;
	wg_sync();
	wg_kpke_encrypt(k, lane, public_buf, 0u);
	// ciphertext staging -> buffer.
	wg_store_bytes(ciphertext_buf, 0u, 0u, cl, lane);
	// shared key.
	for (uint w = lane; w < 8u; w += WG_LANES) shared_buf.Store(4u * w, g_key[w]);
	wg_sync();
}

void mlkem_decaps_par(
	uint k,
	uint lane,
	RWByteAddressBuffer private_buf,
	RWByteAddressBuffer ciphertext_buf,
	RWByteAddressBuffer shared_buf
) {
	uint cl = 32u * (k == 4u ? 11u * k + 5u : 10u * k + 4u);
	// Stage ciphertext for decrypt u/v decodes and J absorb.
	wg_load_bytes(ciphertext_buf, 0u, 0u, cl, lane);
	wg_kpke_decrypt(k, lane, private_buf);
	// h = dk[768k+32..).
	if (lane < 8u) g_hash[lane] = private_buf.Load(768u * k + 32u + 4u * lane);
	wg_sync();
	// G(m' || h): stage m' then h into g_bytes.
	for (uint i = lane; i < 8u; i += WG_LANES) {
		uint m = g_msg[i], h = g_hash[i];
		g_bytes[4u * i] = m & 255u;
		g_bytes[4u * i + 1u] = (m >> 8u) & 255u;
		g_bytes[4u * i + 2u] = (m >> 16u) & 255u;
		g_bytes[4u * i + 3u] = (m >> 24u) & 255u;
		g_bytes[32u + 4u * i] = h & 255u;
		g_bytes[32u + 4u * i + 1u] = (h >> 8u) & 255u;
		g_bytes[32u + 4u * i + 2u] = (h >> 16u) & 255u;
		g_bytes[32u + 4u * i + 3u] = (h >> 24u) & 255u;
	}
	wg_sync();
	wg_sp_init(lane);
	wg_sp_absorb_shared(72u, 0u, 64u, lane);
	wg_sp_finish(72u, 6u, 64u, lane);
	for (uint w = lane; w < 8u; w += WG_LANES) {
		g_key[w] = g_d64[w];
		g_sigma[w] = g_d64[8u + w];
	}
	wg_sync();
	// J(z || ct) = SHAKE-256, both parts absorbed from buffers.
	wg_sp_init(lane);
	wg_sp_absorb_buf(136u, private_buf, 768u * k + 64u, 32u, lane);
	wg_sp_absorb_buf(136u, ciphertext_buf, 0u, cl, lane);
	wg_sp_finish(136u, 31u, 32u, lane);
	// rejection digest in g_d64[0..8).
	if (lane == 0u) g_flag = 0u;
	wg_sync();
	// Re-encrypt: ek is the dk segment at byte 384k.
	wg_kpke_encrypt(k, lane, private_buf, 384u * k);
	// Compare staged ct' against the received ciphertext.
	if (lane == 0u) g_flag = 0u;
	wg_sync();
	for (uint w = lane; w < cl / 4u; w += WG_LANES) {
		uint a = ciphertext_buf.Load(4u * w);
		uint b = g_bytes[4u * w]
			| (g_bytes[4u * w + 1u] << 8u)
			| (g_bytes[4u * w + 2u] << 16u)
			| (g_bytes[4u * w + 3u] << 24u);
		if (a != b) g_flag = 1u;
	}
	wg_sync();
	uint mask = g_flag != 0u ? 0xFFFFFFFFu : 0u;
	for (uint w = lane; w < 8u; w += WG_LANES)
		shared_buf.Store(4u * w, (g_key[w] & ~mask) | (g_d64[w] & mask));
	wg_sync();
}
