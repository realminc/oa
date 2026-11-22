// Workgroup-parallel ML-DSA-44/65/87 verification, FIPS 204 Algorithms 3, 5, 8.
// One workgroup of 128 lanes verifies one signature. Polynomial and XOF state
// lives in groupshared storage so no invocation carries a coefficient array in
// registers; NTT, packing, pointwise and hint phases run with barrier-separated
// parallelism. Every helper takes scalar arguments and reads or writes module
// shared storage: SPIR-V function calls carrying array pointers defeat the
// current-generation driver compiler, so no parameter carries an array or the
// sponge/XOF structures. The public-data contract matches
// mldsa_verification.slang exactly.
import shake;
import keccak;
import sha2_u32;
import sha2_u32_pair;
import sha3;
import storage;
import mldsa_prehash_layout;
import mldsa_field;
import mldsa_ntt;
import mldsa_packing;
import mldsa_sampling;

static const uint WG_LANES = 128u;

// Flat groupshared polynomial workspace, offsets in coefficients.
static const uint P_T1 = 0u;           // 8 polynomials
static const uint P_Z = 8u * 256u;     // 7 polynomials
static const uint P_H = 15u * 256u;    // 8 polynomials
static const uint P_A = 23u * 256u;    // 7 per-row ExpandA staging slots
static const uint P_W = 30u * 256u;    // 1 row accumulator / w1 hints
static const uint P_C = 31u * 256u;    // 1 challenge polynomial
static const uint P_TOTAL = 32u * 256u;

// Shared Keccak-state slots: 0..6 per-lane ExpandA streams, 7 SampleInBall,
// 8 the serial message sponge, 9 prehash digest scratch. The digest slot must
// stay disjoint from the sponge: HashML-DSA digests M' while the mu sponge is
// already absorbing tr, so reusing it would clobber live sponge state.
static const uint SLOT_BALL = 7u;
static const uint SLOT_SPONGE = 8u;
static const uint SLOT_DIGEST = 9u;

groupshared uint g_poly[P_TOTAL];
groupshared uint2 g_state[10][25];
groupshared uint g_sp_pos;
groupshared uint2 g_digest[8];
groupshared uint g_d32[16];
groupshared uint g_sha32[24];   // u32 SHA-2: state[8] then block[16]
groupshared uint2 g_sha64[24];  // u64 SHA-2: state[8] then block[16]
groupshared uint g_rho[8];
groupshared uint g_ctilde[16];
groupshared uint g_hint_idx;
groupshared uint g_flag;
groupshared uint g_result;
groupshared uint g_rho_prime[16];

// SHAKE-256 squeeze from the serial sponge slot; donor byte order preserved.

uint wg_sp_squeeze_byte() {
	if (g_sp_pos == 136u) {
		wg_keccak(SLOT_SPONGE);
		g_sp_pos = 0u;
	}
	uint position = g_sp_pos++;
	uint word = (position % 8u < 4u) ? g_state[SLOT_SPONGE][position / 8u].x
		: g_state[SLOT_SPONGE][position / 8u].y;
	return (word >> ((position % 4u) * 8u)) & 255u;
}

// Serialize one polynomial's encoded byte stream to the output buffer.
// mode 0: eta encoding (bound - centered); mode 1: power2round low
// (4096 - lo); mode 2: power2round high (t1).

void wg_pack_output(RWByteAddressBuffer buf, uint base, uint src, uint bits, uint mode, uint bound) {
	uint reservoir = 0u, occupied = 0u, cursor = base;
	uint out_word = 0u;
	for (uint i = 0u; i < 256u; ++i) {
		uint c = g_poly[src + i];
		uint v;
		if (mode == 0u) v = (uint)((int)bound - mldsa_centered(c));
		else if (mode == 1u) v = (uint)(4096 - mldsa_power2round_lo(c));
		else v = mldsa_power2round_hi(c);
		reservoir |= v << occupied;
		occupied += bits;
		while (occupied >= 8u) {
			out_word |= (reservoir & 255u) << (((cursor - base) & 3u) * 8u);
			if (((cursor - base) & 3u) == 3u) {
				buf.Store(cursor & ~3u, out_word);
				out_word = 0u;
			}
			++cursor;
			reservoir >>= 8u;
			occupied -= 8u;
		}
	}
}

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

// ---------------------------------------------------------------------------
// SHAKE-256 message sponge on shared slot 8 (lane 0 drives; byte semantics
// identical to pqc_shake256_* in shake.slang).
// ---------------------------------------------------------------------------

void wg_sp_init() {
	for (uint i = 0u; i < 25u; ++i) g_state[SLOT_SPONGE][i] = uint2(0u, 0u);
	g_sp_pos = 0u;
}


void wg_sp_absorb_byte(uint value) {
	uint lane = g_sp_pos / 8u;
	uint shift = (g_sp_pos % 4u) * 8u;
	if ((g_sp_pos % 8u) < 4u) g_state[SLOT_SPONGE][lane].x ^= (value & 255u) << shift;
	else g_state[SLOT_SPONGE][lane].y ^= (value & 255u) << shift;
	++g_sp_pos;
	if (g_sp_pos == 136u) {
		wg_keccak(SLOT_SPONGE);
		g_sp_pos = 0u;
	}
}


// Absorb one aligned 32-bit word; requires g_sp_pos % 4 == 0.
void wg_sp_absorb_word(uint value) {
	uint lane = g_sp_pos / 8u;
	if ((g_sp_pos % 8u) < 4u) g_state[SLOT_SPONGE][lane].x ^= value;
	else g_state[SLOT_SPONGE][lane].y ^= value;
	g_sp_pos += 4u;
	if (g_sp_pos == 136u) {
		wg_keccak(SLOT_SPONGE);
		g_sp_pos = 0u;
	}
}


void wg_sp_absorb_buffer(RWByteAddressBuffer buffer, uint base, uint length) {
	uint i = 0u;
	// Lead bytes until the sponge position is word-aligned, then aligned u32
	// absorbs while the source is aligned, then any tail bytes.
	while (i < length && (g_sp_pos % 4u) != 0u) {
		uint address = base + i;
		wg_sp_absorb_byte(buffer.Load(address & ~3u) >> ((address & 3u) * 8u));
		++i;
	}
	if ((base + i) % 4u == 0u)
		for (; i + 4u <= length; i += 4u)
			wg_sp_absorb_word(buffer.Load(base + i));
	for (; i < length; ++i) {
		uint address = base + i;
		wg_sp_absorb_byte(buffer.Load(address & ~3u) >> ((address & 3u) * 8u));
	}
}


void wg_sp_absorb_digest() {
	for (uint i = 0u; i < 8u; ++i) {
		wg_sp_absorb_word(g_digest[i].x);
		wg_sp_absorb_word(g_digest[i].y);
	}
}


void wg_sp_finalize() {
	uint lane = g_sp_pos / 8u;
	uint shift = (g_sp_pos % 4u) * 8u;
	if ((g_sp_pos % 8u) < 4u) g_state[SLOT_SPONGE][lane].x ^= 31u << shift;
	else g_state[SLOT_SPONGE][lane].y ^= 31u << shift;
	g_state[SLOT_SPONGE][16].y ^= 0x80000000u;
	wg_keccak(SLOT_SPONGE);
	g_sp_pos = 0u;
}

// finish = finalize + first 64 squeezed bytes to g_digest.

void wg_sp_finish() {
	wg_sp_finalize();
	for (uint i = 0u; i < 8u; ++i) g_digest[i] = g_state[SLOT_SPONGE][i];
	g_sp_pos = 64u;
}

// ---------------------------------------------------------------------------
// Parallel unpack phase: scalar-parameter twins of the serial packing helpers,
// writing coefficients directly into the shared workspace.
// ---------------------------------------------------------------------------
uint wg_unsigned_coeff(RWByteAddressBuffer buf, uint buf_base, uint coeff_idx, uint bits) {
	uint bit_offset = coeff_idx * bits;
	uint byte_start = bit_offset / 8u;
	uint bit_shift = bit_offset % 8u;
	uint byte_count = (bit_shift + bits + 7u) / 8u;
	uint window = 0u;
	for (uint j = 0u; j < byte_count; ++j) {
		window |= read_byte(buf, buf_base + byte_start + j) << (j * 8u);
	}
	return (window >> bit_shift) & ((1u << bits) - 1u);
}


void wg_unpack_unsigned(RWByteAddressBuffer buf, uint buf_base, uint bits, uint dest) {
	for (uint i = 0u; i < 256u; ++i) {
		g_poly[dest + i] = wg_unsigned_coeff(buf, buf_base, i, bits);
	}
}


void wg_unpack_signed(RWByteAddressBuffer buf, uint buf_base, uint bits, uint half_range, uint dest) {
	for (uint i = 0u; i < 256u; ++i) {
		uint p = wg_unsigned_coeff(buf, buf_base, i, bits);
		int c = (int)half_range - (int)p;
		g_poly[dest + i] = (c < 0) ? (uint)(c + (int)MLDSA_Q) : (uint)c;
	}
}


void wg_check_norm(uint dest, uint bound) {
	for (uint i = 0u; i < 256u; ++i) {
		int c = mldsa_centered(g_poly[dest + i]);
		uint abs_c = (c < 0) ? (uint)(-c) : (uint)c;
		if (abs_c >= bound) g_flag = 1u;
	}
}

// Canonical hint row decode; identical ordering checks to the serial helper.

void wg_unpack_hint_row(RWByteAddressBuffer buf, uint h_base, uint omega, uint row, uint dest) {
	for (uint i = 0u; i < 256u; ++i) g_poly[dest + i] = 0u;
	uint end = read_byte(buf, h_base + omega + row);
	if (end < g_hint_idx || end > omega) {
		g_flag = 1u;
		return;
	}
	uint start = g_hint_idx;
	while (g_hint_idx < end) {
		uint pos = read_byte(buf, h_base + g_hint_idx);
		if (g_hint_idx > start && pos <= read_byte(buf, h_base + g_hint_idx - 1u)) {
			g_flag = 1u;
			return;
		}
		g_poly[dest + pos] = 1u;
		++g_hint_idx;
	}
}


void wg_hint_tail(RWByteAddressBuffer buf, uint h_base, uint omega) {
	for (uint i = g_hint_idx; i < omega; ++i)
		if (read_byte(buf, h_base + i) != 0u) g_flag = 1u;
}

// ---------------------------------------------------------------------------
// Per-lane XOF streams. wg_expand_a and wg_sample_ball are the serial donor
// formulas transposed onto shared Keccak slots.
// ---------------------------------------------------------------------------
uint wg_xof_byte(uint slot, uint pos) {
	uint lane = pos / 8u;
	uint boff = pos % 8u;
	uint word = (boff < 4u) ? g_state[slot][lane].x : g_state[slot][lane].y;
	uint shift = (boff < 4u) ? (boff * 8u) : ((boff - 4u) * 8u);
	return (word >> shift) & 0xFFu;
}

// SHAKE-128 short absorb of rho || col || row plus the rejection loop,
// FIPS 204 Algorithm 34.

void wg_expand_a(uint slot, uint row, uint col, uint dest) {
	for (uint i = 0u; i < 25u; ++i) g_state[slot][i] = uint2(0u, 0u);
	for (uint b = 0u; b < 34u; ++b) {
		uint byte;
		if (b < 32u) byte = (g_rho[b / 4u] >> ((b & 3u) * 8u)) & 0xFFu;
		else byte = b == 32u ? (col & 0xFFu) : (row & 0xFFu);
		uint lane = b / 8u;
		uint boff = b % 8u;
		if (boff < 4u) g_state[slot][lane].x ^= (byte << (boff * 8u));
		else g_state[slot][lane].y ^= (byte << ((boff - 4u) * 8u));
	}
	{
		uint lane = 34u / 8u;
		uint boff = 34u % 8u;
		if (boff < 4u) g_state[slot][lane].x ^= (0x1Fu << (boff * 8u));
		else g_state[slot][lane].y ^= (0x1Fu << ((boff - 4u) * 8u));
	}
	g_state[slot][20].y ^= 0x80000000u;
	wg_keccak(slot);

	uint count = 0u;
	uint pos = 0u;
	while (count < 256u) {
		if (pos + 3u > 168u) {
			wg_keccak(slot);
			pos = 0u;
		}
		uint b0 = wg_xof_byte(slot, pos);
		uint b1 = wg_xof_byte(slot, pos + 1u);
		uint b2 = wg_xof_byte(slot, pos + 2u) & 0x7Fu;
		pos += 3u;
		uint coeff = b0 | (b1 << 8u) | (b2 << 16u);
		if (coeff < MLDSA_Q) {
			g_poly[dest + count] = coeff;
			++count;
		}
	}
}

// FIPS 204 Algorithm 29 challenge sampling on a shared slot; c_tilde is staged
// in g_ctilde by the parse phase.

void wg_sample_ball(uint slot, uint seed_words, uint tau, uint dest) {
	for (uint i = 0u; i < 256u; ++i) g_poly[dest + i] = 0u;
	for (uint i = 0u; i < 25u; ++i) g_state[slot][i] = uint2(0u, 0u);
	for (uint w = 0u; w < seed_words; ++w) {
		if ((w & 1u) == 0u) g_state[slot][w / 2u].x = g_ctilde[w];
		else g_state[slot][w / 2u].y = g_ctilde[w];
	}
	g_state[slot][seed_words / 2u].x ^= 0x1Fu;
	g_state[slot][16].y ^= 0x80000000u;
	wg_keccak(slot);
	uint sign_lo = g_state[slot][0].x, sign_hi = g_state[slot][0].y;
	uint pos = 8u;
	for (uint i = 256u - tau; i < 256u; ++i) {
		uint j;
		bool repeat = true;
		while (repeat) {
			if (pos == 136u) {
				wg_keccak(slot);
				pos = 0u;
			}
			j = wg_xof_byte(slot, pos++);
			repeat = j > i;
		}
		g_poly[dest + i] = g_poly[dest + j];
		uint bit = i + tau - 256u;
		uint sign = bit < 32u ? ((sign_lo >> bit) & 1u)
							 : ((sign_hi >> (bit - 32u)) & 1u);
		g_poly[dest + j] = sign != 0u ? MLDSA_Q - 1u : 1u;
	}
}

// ---------------------------------------------------------------------------
// Prehash digest twins for the HashML-DSA message route. Identical formulas to
// sha2_u32.slang, sha2_u32_pair.slang, sha3.slang and shake_prehash.slang, on
// shared scratch so no helper carries an array parameter.
// ---------------------------------------------------------------------------

void wg_sha2_compress() {
	uint schedule[16];
	for (uint i = 0u; i < 16u; ++i) schedule[i] = g_sha32[8 + i];
	uint a = g_sha32[0], b = g_sha32[1], c = g_sha32[2], d = g_sha32[3];
	uint e = g_sha32[4], f = g_sha32[5], g = g_sha32[6], h = g_sha32[7];
	for (uint round = 0u; round < 64u; ++round) {
		uint slot = round & 15u;
		if (round >= 16u) {
			uint x = schedule[(round - 15u) & 15u];
			uint y = schedule[(round - 2u) & 15u];
			uint sigma0 = sha2_u32_rotr(x, 7u) ^ sha2_u32_rotr(x, 18u) ^ (x >> 3u);
			uint sigma1 = sha2_u32_rotr(y, 17u) ^ sha2_u32_rotr(y, 19u) ^ (y >> 10u);
			schedule[slot] += sigma0 + schedule[(round - 7u) & 15u] + sigma1;
		}
		uint sum1 = sha2_u32_rotr(e, 6u) ^ sha2_u32_rotr(e, 11u) ^ sha2_u32_rotr(e, 25u);
		uint choose = (e & f) ^ (~e & g);
		uint temp1 = h + sum1 + choose + SHA2_U32_K[round] + schedule[slot];
		uint sum0 = sha2_u32_rotr(a, 2u) ^ sha2_u32_rotr(a, 13u) ^ sha2_u32_rotr(a, 22u);
		uint majority = (a & b) ^ (a & c) ^ (b & c);
		uint temp2 = sum0 + majority;
		h = g; g = f; f = e; e = d + temp1;
		d = c; c = b; b = a; a = temp1 + temp2;
	}
	g_sha32[0] += a; g_sha32[1] += b; g_sha32[2] += c; g_sha32[3] += d;
	g_sha32[4] += e; g_sha32[5] += f; g_sha32[6] += g; g_sha32[7] += h;
}


void wg_sha2_init(uint bits) {
	if (bits == 224u) {
		g_sha32[0] = 0xc1059ed8u; g_sha32[1] = 0x367cd507u;
		g_sha32[2] = 0x3070dd17u; g_sha32[3] = 0xf70e5939u;
		g_sha32[4] = 0xffc00b31u; g_sha32[5] = 0x68581511u;
		g_sha32[6] = 0x64f98fa7u; g_sha32[7] = 0xbefa4fa4u;
	} else {
		g_sha32[0] = 0x6a09e667u; g_sha32[1] = 0xbb67ae85u;
		g_sha32[2] = 0x3c6ef372u; g_sha32[3] = 0xa54ff53au;
		g_sha32[4] = 0x510e527fu; g_sha32[5] = 0x9b05688cu;
		g_sha32[6] = 0x1f83d9abu; g_sha32[7] = 0x5be0cd19u;
	}
}

// SHA-224/SHA-256 into g_d32; matches sha2_u32_digest exactly.


void wg_sha2_digest(RWByteAddressBuffer buf, uint msg_base, uint msg_len, uint bits) {
	wg_sha2_init(bits);
	uint offset = 0u;
	uint remaining = msg_len;
	while (remaining >= 64u) {
		for (uint word = 0u; word < 16u; ++word) {
			uint address = msg_base + offset + word * 4u;
			if ((address & 3u) == 0u) g_sha32[8 + word] = sha2_u32_swap(buf.Load(address));
			else g_sha32[8 + word] = (load_u8(buf, address) << 24u)
				| (load_u8(buf, address + 1u) << 16u)
				| (load_u8(buf, address + 2u) << 8u) | load_u8(buf, address + 3u);
		}
		wg_sha2_compress();
		offset += 64u;
		remaining -= 64u;
	}
	for (uint word = 0u; word < 16u; ++word) g_sha32[8 + word] = 0u;
	for (uint byte = 0u; byte < remaining; ++byte) {
		g_sha32[8 + byte / 4u] |= load_u8(buf, msg_base + offset + byte) << (24u - (byte & 3u) * 8u);
	}
	g_sha32[8 + remaining / 4u] |= 0x80u << (24u - (remaining & 3u) * 8u);
	if (remaining >= 56u) {
		wg_sha2_compress();
		for (uint word = 0u; word < 16u; ++word) g_sha32[8 + word] = 0u;
	}
	uint2 bit_length = sha2_u32_bit_length(msg_len);
	g_sha32[8 + 14] = bit_length.x;
	g_sha32[8 + 15] = bit_length.y;
	wg_sha2_compress();
	for (uint word = 0u; word < bits / 32u; ++word) g_d32[word] = sha2_u32_swap(g_sha32[word]);
}


void wg_sha2p_compress() {
	uint2 schedule[16];
	for (uint i = 0u; i < 16u; ++i) schedule[i] = g_sha64[8 + i];
	uint2 a = g_sha64[0], b = g_sha64[1], c = g_sha64[2], d = g_sha64[3];
	uint2 e = g_sha64[4], f = g_sha64[5], g = g_sha64[6], h = g_sha64[7];
	for (uint round = 0u; round < 80u; ++round) {
		uint slot = round & 15u;
		if (round >= 16u) {
			uint2 x = schedule[(round - 15u) & 15u];
			uint2 y = schedule[(round - 2u) & 15u];
			uint2 sigma0 = sha2_u32_pair_rotr(x, 1u) ^ sha2_u32_pair_rotr(x, 8u) ^ sha2_u32_pair_shr(x, 7u);
			uint2 sigma1 = sha2_u32_pair_rotr(y, 19u) ^ sha2_u32_pair_rotr(y, 61u) ^ sha2_u32_pair_shr(y, 6u);
			schedule[slot] = sha2_u32_pair_add(sha2_u32_pair_add(schedule[slot], sigma0),
				sha2_u32_pair_add(schedule[(round - 7u) & 15u], sigma1));
		}
		uint2 sum1 = sha2_u32_pair_rotr(e, 14u) ^ sha2_u32_pair_rotr(e, 18u) ^ sha2_u32_pair_rotr(e, 41u);
		uint2 choose = (e & f) ^ (~e & g);
		uint2 temp1 = sha2_u32_pair_add(sha2_u32_pair_add(h, sum1), choose);
		temp1 = sha2_u32_pair_add(temp1, sha2_u32_pair_add(SHA2_U32_PAIR_K[round], schedule[slot]));
		uint2 sum0 = sha2_u32_pair_rotr(a, 28u) ^ sha2_u32_pair_rotr(a, 34u) ^ sha2_u32_pair_rotr(a, 39u);
		uint2 majority = (a & b) ^ (a & c) ^ (b & c);
		uint2 temp2 = sha2_u32_pair_add(sum0, majority);
		h = g; g = f; f = e; e = sha2_u32_pair_add(d, temp1);
		d = c; c = b; b = a; a = sha2_u32_pair_add(temp1, temp2);
	}
	g_sha64[0] = sha2_u32_pair_add(g_sha64[0], a); g_sha64[1] = sha2_u32_pair_add(g_sha64[1], b);
	g_sha64[2] = sha2_u32_pair_add(g_sha64[2], c); g_sha64[3] = sha2_u32_pair_add(g_sha64[3], d);
	g_sha64[4] = sha2_u32_pair_add(g_sha64[4], e); g_sha64[5] = sha2_u32_pair_add(g_sha64[5], f);
	g_sha64[6] = sha2_u32_pair_add(g_sha64[6], g); g_sha64[7] = sha2_u32_pair_add(g_sha64[7], h);
}


void wg_sha2p_init(uint bits) {
	if (bits == 224u) {
		g_sha64[0] = uint2(0x19544da2u, 0x8c3d37c8u);
		g_sha64[1] = uint2(0x89dcd4d6u, 0x73e19966u);
		g_sha64[2] = uint2(0x32ff9c82u, 0x1dfab7aeu);
		g_sha64[3] = uint2(0x582f9fcfu, 0x679dd514u);
		g_sha64[4] = uint2(0x7bd44da8u, 0x0f6d2b69u);
		g_sha64[5] = uint2(0x04c48942u, 0x77e36f73u);
		g_sha64[6] = uint2(0x6a1d36c8u, 0x3f9d85a8u);
		g_sha64[7] = uint2(0x91d692a1u, 0x1112e6adu);
	} else if (bits == 256u) {
		g_sha64[0] = uint2(0xfc2bf72cu, 0x22312194u);
		g_sha64[1] = uint2(0xc84c64c2u, 0x9f555fa3u);
		g_sha64[2] = uint2(0x6f53b151u, 0x2393b86bu);
		g_sha64[3] = uint2(0x5940eabdu, 0x96387719u);
		g_sha64[4] = uint2(0xa88effe3u, 0x96283ee2u);
		g_sha64[5] = uint2(0x53863992u, 0xbe5e1e25u);
		g_sha64[6] = uint2(0x2c85b8aau, 0x2b0199fcu);
		g_sha64[7] = uint2(0x81c52ca2u, 0x0eb72ddcu);
	} else if (bits == 384u) {
		g_sha64[0] = uint2(0xc1059ed8u, 0xcbbb9d5du);
		g_sha64[1] = uint2(0x367cd507u, 0x629a292au);
		g_sha64[2] = uint2(0x3070dd17u, 0x9159015au);
		g_sha64[3] = uint2(0xf70e5939u, 0x152fecd8u);
		g_sha64[4] = uint2(0xffc00b31u, 0x67332667u);
		g_sha64[5] = uint2(0x68581511u, 0x8eb44a87u);
		g_sha64[6] = uint2(0x64f98fa7u, 0xdb0c2e0du);
		g_sha64[7] = uint2(0xbefa4fa4u, 0x47b5481du);
	} else {
		g_sha64[0] = uint2(0xf3bcc908u, 0x6a09e667u);
		g_sha64[1] = uint2(0x84caa73bu, 0xbb67ae85u);
		g_sha64[2] = uint2(0xfe94f82bu, 0x3c6ef372u);
		g_sha64[3] = uint2(0x5f1d36f1u, 0xa54ff53au);
		g_sha64[4] = uint2(0xade682d1u, 0x510e527fu);
		g_sha64[5] = uint2(0x2b3e6c1fu, 0x9b05688cu);
		g_sha64[6] = uint2(0xfb41bd6bu, 0x1f83d9abu);
		g_sha64[7] = uint2(0x137e2179u, 0x5be0cd19u);
	}
}

// SHA-384/SHA-512 and the 512/224, 512/256 truncations into g_d32; matches
// sha2_u32_pair_digest exactly.


void wg_sha2p_digest(RWByteAddressBuffer buf, uint msg_base, uint msg_len, uint bits) {
	wg_sha2p_init(bits);
	uint offset = 0u;
	uint remaining = msg_len;
	while (remaining >= 128u) {
		for (uint word = 0u; word < 32u; ++word) {
			uint address = msg_base + offset + word * 4u;
			uint value;
			if ((address & 3u) == 0u) value = sha2_u32_swap(buf.Load(address));
			else value = (load_u8(buf, address) << 24u) | (load_u8(buf, address + 1u) << 16u)
				| (load_u8(buf, address + 2u) << 8u) | load_u8(buf, address + 3u);
			if ((word & 1u) == 0u) g_sha64[8 + word / 2u].y = value;
			else g_sha64[8 + word / 2u].x = value;
		}
		wg_sha2p_compress();
		offset += 128u;
		remaining -= 128u;
	}
	for (uint word = 0u; word < 16u; ++word) g_sha64[8 + word] = uint2(0u, 0u);
	for (uint byte = 0u; byte <= remaining; ++byte) {
		uint value = byte == remaining ? 0x80u : load_u8(buf, msg_base + offset + byte);
		value <<= 24u - (byte & 3u) * 8u;
		if ((byte & 7u) < 4u) g_sha64[8 + byte / 8u].y |= value;
		else g_sha64[8 + byte / 8u].x |= value;
	}
	if (remaining >= 112u) {
		wg_sha2p_compress();
		for (uint word = 0u; word < 16u; ++word) g_sha64[8 + word] = uint2(0u, 0u);
	}
	g_sha64[8 + 15] = uint2(msg_len << 3u, msg_len >> 29u);
	wg_sha2p_compress();
	for (uint word = 0u; word < bits / 32u; ++word) {
		uint value = (word & 1u) == 0u ? g_sha64[word / 2u].y : g_sha64[word / 2u].x;
		g_d32[word] = sha2_u32_swap(value);
	}
}

// SHA-3 absorb into shared slot 8 and squeeze to g_d32; matches sha3_absorb
// plus the fixed-length reads in mldsa_prehash_digest.


void wg_sha3_digest(RWByteAddressBuffer buf, uint msg_base, uint msg_len, uint rate_bytes, uint words) {
	for (uint i = 0u; i < 25u; ++i) g_state[SLOT_DIGEST][i] = uint2(0u, 0u);
	uint rate_lanes = rate_bytes / 8u;
	uint remaining = msg_len;
	uint offset = 0u;
	while (remaining >= rate_bytes) {
		for (uint lane = 0u; lane < rate_lanes; ++lane) {
			uint addr = msg_base + offset + lane * 8u;
			uint lo = sha3_load_word(buf, addr);
			uint hi = sha3_load_word(buf, addr + 4u);
			g_state[SLOT_DIGEST][lane].x ^= lo;
			g_state[SLOT_DIGEST][lane].y ^= hi;
		}
		wg_keccak(SLOT_DIGEST);
		offset += rate_bytes;
		remaining -= rate_bytes;
	}
	uint2 padded[18];
	for (uint lane = 0u; lane < 18u; ++lane) padded[lane] = uint2(0u, 0u);
	for (uint b = 0u; b < remaining; ++b) {
		uint byte = load_u8(buf, msg_base + offset + b);
		uint lane = b / 8u;
		uint boff = b % 8u;
		if (boff < 4u) padded[lane].x |= (byte << (boff * 8u));
		else padded[lane].y |= (byte << ((boff - 4u) * 8u));
	}
	{
		uint lane = remaining / 8u;
		uint boff = remaining % 8u;
		if (boff < 4u) padded[lane].x |= (SHA3_DOMAIN << (boff * 8u));
		else padded[lane].y |= (SHA3_DOMAIN << ((boff - 4u) * 8u));
	}
	{
		uint lane = (rate_bytes - 1u) / 8u;
		uint boff = (rate_bytes - 1u) % 8u;
		padded[lane].y |= 0x80000000u;
	}
	for (uint lane = 0u; lane < rate_lanes; ++lane) g_state[SLOT_DIGEST][lane] ^= padded[lane];
	wg_keccak(SLOT_DIGEST);
	for (uint i = 0u; i < words; ++i) g_d32[i] = (i & 1u) == 0u ? g_state[SLOT_DIGEST][i / 2u].x : g_state[SLOT_DIGEST][i / 2u].y;
}

// SHAKE-128/256 fixed digest into g_d32; matches pqc_shake_prehash exactly.


void wg_shake_ph_digest(RWByteAddressBuffer buf, uint msg_base, uint msg_len, uint bits) {
	uint rate = bits == 128u ? 168u : 136u;
	uint words = bits == 128u ? 8u : 16u;
	for (uint i = 0u; i < 25u; ++i) g_state[SLOT_DIGEST][i] = uint2(0u, 0u);
	uint offset = 0u;
	uint remaining = msg_len;
	while (true) {
		uint amount = min(remaining, rate);
		for (uint byte = 0u; byte < rate; ++byte) {
			uint value = byte < amount ? load_u8(buf, msg_base + offset + byte) : 0u;
			if (amount < rate && byte == amount) value |= 0x1fu;
			if (amount < rate && byte == rate - 1u) value |= 0x80u;
			uint lane = byte / 8u;
			if ((byte & 7u) < 4u) g_state[SLOT_DIGEST][lane].x ^= value << ((byte & 3u) * 8u);
			else g_state[SLOT_DIGEST][lane].y ^= value << ((byte & 3u) * 8u);
		}
		wg_keccak(SLOT_DIGEST);
		if (amount < rate) break;
		offset += amount;
		remaining -= amount;
	}
	for (uint word = 0u; word < words; ++word)
		g_d32[word] = (word & 1u) == 0u ? g_state[SLOT_DIGEST][word / 2u].x : g_state[SLOT_DIGEST][word / 2u].y;
}

// Prehash message digest on lane 0. Dispatch mirrors mldsa_prehash_digest's
// generated OID switch; failures raise the shared flag.

void wg_prehash_digest(RWByteAddressBuffer messages, uint base, uint length, uint algorithm) {
	switch (algorithm) {
		case 1u: { wg_sha2_digest(messages, base, length, 256u); break; }
		case 2u: { wg_sha2p_digest(messages, base, length, 384u); break; }
		case 3u: { wg_sha2p_digest(messages, base, length, 512u); break; }
		case 4u: { wg_sha2_digest(messages, base, length, 224u); break; }
		case 5u: { wg_sha2p_digest(messages, base, length, 224u); break; }
		case 6u: { wg_sha2p_digest(messages, base, length, 256u); break; }
		case 7u: { wg_sha3_digest(messages, base, length, SHA3_224_RATE_BYTES, 7u); break; }
		case 8u: { wg_sha3_digest(messages, base, length, SHA3_256_RATE_BYTES, 8u); break; }
		case 9u: { wg_sha3_digest(messages, base, length, SHA3_384_RATE_BYTES, 12u); break; }
		case 10u: { wg_sha3_digest(messages, base, length, SHA3_512_RATE_BYTES, 16u); break; }
		case 11u: { wg_shake_ph_digest(messages, base, length, 128u); break; }
		case 12u: { wg_shake_ph_digest(messages, base, length, 256u); break; }
		default: g_flag = 1u;
	}
}

// ---------------------------------------------------------------------------
// Workgroup NTT: in-place forward and inverse transforms on shared
// polynomials, FIPS 204 Algorithms 41 and 42 structures.
// ---------------------------------------------------------------------------
void wg_ntt_step(uint base, uint len, uint k, uint lane) {
	uint blk = lane / len;
	uint j = lane - blk * len;
	uint i0 = base + blk * 2u * len + j;
	uint i1 = i0 + len;
	uint zeta = ZETA_MONT[k + 1u + blk];
	uint a = g_poly[i0];
	uint t = mldsa_mont_mul(zeta, g_poly[i1]);
	g_poly[i0] = mldsa_add(a, t);
	g_poly[i1] = mldsa_sub(a, t);
}


void wg_ntt(uint base, uint lane) {
	uint k = 0u;
	for (uint len = 128u; len >= 1u; len >>= 1u) {
		wg_ntt_step(base, len, k, lane);
		wg_sync();
		k += 128u / len;
	}
}

void wg_intt_step(uint base, uint len, uint k, uint lane) {
	uint blk = lane / len;
	uint j = lane - blk * len;
	uint i0 = base + blk * 2u * len + j;
	uint i1 = i0 + len;
	uint zeta = MLDSA_Q - ZETA_MONT[k - 1u - blk];
	uint a = g_poly[i0];
	uint b = g_poly[i1];
	g_poly[i0] = mldsa_add(a, b);
	g_poly[i1] = mldsa_mont_mul(zeta, mldsa_sub(a, b));
}

void wg_intt_scale(uint base, uint lane) {
	for (uint i = lane; i < 256u; i += WG_LANES)
		g_poly[base + i] = mldsa_mont_mul(MLDSA_INV_N_MONT, g_poly[base + i]);
}


void wg_intt(uint base, uint lane) {
	uint k = 256u;
	for (uint len = 1u; len < 256u; len <<= 1u) {
		wg_intt_step(base, len, k, lane);
		wg_sync();
		k -= 128u / len;
	}
	wg_intt_scale(base, lane);
	wg_sync();
}

bool wg_failed(uint lane) {
	wg_sync();
	return g_flag != 0u;
}

// ---------------------------------------------------------------------------
// Parallel verifier body. Every lane executes it; scalar decisions stay uniform
// and lane 0 owns every sequential serialization step while other lanes wait
// at the enclosing barriers.
// ---------------------------------------------------------------------------
// Shared parameter, range, and signature-parse phase for every verifier entry.
// All exits stay uniform: failures return false on every lane before or after
// an enclosing barrier, never between lanes.
bool wg_verify_parse(
	uint parameter_set,
	RWByteAddressBuffer public_keys, uint pk_base, uint pk_length,
	RWByteAddressBuffer signatures, uint sig_base, uint sig_length,
	uint lane
) {
	if (parameter_set != 44u && parameter_set != 65u && parameter_set != 87u) return false;
	uint rows = parameter_set == 44u ? 4u : (parameter_set == 65u ? 6u : 8u);
	uint cols = parameter_set == 44u ? 4u : (parameter_set == 65u ? 5u : 7u);
	uint challenge_words = parameter_set == 44u ? 8u : (parameter_set == 65u ? 12u : 16u);
	uint gamma1 = parameter_set == 44u ? 131072u : 524288u;
	uint z_bits = parameter_set == 44u ? 18u : 20u;
	uint beta = parameter_set == 44u ? 78u : (parameter_set == 65u ? 196u : 120u);
	uint omega = parameter_set == 44u ? 80u : (parameter_set == 65u ? 55u : 75u);
	uint pk_bytes = 32u + rows * 320u;
	uint sig_bytes = challenge_words * 4u + cols * z_bits * 32u + omega + rows;
	if (pk_length != pk_bytes || sig_length != sig_bytes) return false;
	if (!pqc_buffer_range(public_keys, pk_base, pk_length)
		|| !pqc_buffer_range(signatures, sig_base, sig_length)) return false;

	if (lane == 0u) {
		g_flag = 0u;
		g_result = 0u;
		g_hint_idx = 0u;
	}
	wg_sync();

	// Parse phase: one lane per polynomial for the packed public key and z
	// rows; lane 0 owns the serial hint cursor; every lane copies its share of
	// the challenge seed words.
	if (lane < 8u) g_rho[lane] = read_le_word(public_keys, pk_base + lane * 4u);
	if (lane < 16u)
		g_ctilde[lane] = lane < challenge_words
			? read_le_word(signatures, sig_base + lane * 4u) : 0u;
	if (lane < rows) {
		wg_unpack_unsigned(public_keys, pk_base + 32u + lane * 320u, 10u, P_T1 + lane * 256u);
	} else if (lane - rows < cols) {
		uint l = lane - rows;
		wg_unpack_signed(signatures,
			sig_base + challenge_words * 4u + l * z_bits * 32u, z_bits, gamma1,
			P_Z + l * 256u);
		wg_check_norm(P_Z + l * 256u, gamma1 - beta);
	}
	if (lane == 0u) {
		uint h_base = sig_base + challenge_words * 4u + cols * z_bits * 32u;
		for (uint row = 0u; row < rows && g_flag == 0u; ++row)
			wg_unpack_hint_row(signatures, h_base, omega, row, P_H + row * 256u);
		if (g_flag == 0u) wg_hint_tail(signatures, h_base, omega);
	}
	if (wg_failed(lane)) return false;
	return true;
}

// External-mu verification: the caller supplies the finished 64-byte mu so the
// kernel skips the tr/message sponge chain entirely.
bool mldsa_verify_mu_par(
	uint parameter_set,
	RWByteAddressBuffer public_keys, uint pk_base, uint pk_length,
	RWByteAddressBuffer signatures, uint sig_base, uint sig_length,
	RWByteAddressBuffer mus, uint mu_base,
	uint lane
) {
	if (!pqc_buffer_range(mus, mu_base, 64u)) return false;
	if (!wg_verify_parse(parameter_set, public_keys, pk_base, pk_length,
		signatures, sig_base, sig_length, lane)) return false;
	uint challenge_words = parameter_set == 44u ? 8u : (parameter_set == 65u ? 12u : 16u);
	uint tau = parameter_set == 44u ? 39u : (parameter_set == 65u ? 49u : 60u);
	if (lane == 0u) {
		for (uint w = 0u; w < 8u; ++w)
			g_digest[w] = uint2(mus.Load(mu_base + w * 8u), mus.Load(mu_base + w * 8u + 4u));
		wg_sp_init();
		wg_sp_absorb_digest();
	} else if (lane == 1u) {
		wg_sample_ball(SLOT_BALL, challenge_words, tau, P_C);
	}
	if (wg_failed(lane)) return false;
	return mldsa_verify_finish_par(parameter_set, lane);
}

// Shared tail of every verifier entry. The caller leaves the finished mu
// digest in g_digest, has already primed the final sponge on lane 0, and lane 1
// has sampled the challenge polynomial.
bool mldsa_verify_finish_par(uint parameter_set, uint lane) {
	uint rows = parameter_set == 44u ? 4u : (parameter_set == 65u ? 6u : 8u);
	uint cols = parameter_set == 44u ? 4u : (parameter_set == 65u ? 5u : 7u);
	uint challenge_words = parameter_set == 44u ? 8u : (parameter_set == 65u ? 12u : 16u);
	uint gamma2 = parameter_set == 44u ? 95232u : 261888u;

	// NTT phase: c, every z column, then the D-scaled t1 rows — sequential
	// polynomials, 128 parallel butterflies each.
	wg_ntt(P_C, lane);
	for (uint l = 0u; l < cols; ++l) wg_ntt(P_Z + l * 256u, lane);
	for (uint i = lane; i < rows * 256u; i += WG_LANES)
		g_poly[P_T1 + i] = mldsa_mul(g_poly[P_T1 + i], MLDSA_D_FACTOR);
	wg_sync();
	for (uint k = 0u; k < rows; ++k) wg_ntt(P_T1 + k * 256u, lane);

	// Matrix-vector phase: one ExpandA per lane covers a row's columns, the
	// pointwise accumulate and inverse transform run across all 128 lanes, and
	// lane 0 streams the w1 encoding into the primed sponge.
	for (uint k = 0u; k < rows; ++k) {
		if (lane < cols) {
			wg_expand_a(lane, k, lane, P_A + lane * 256u);
		}
		wg_sync();
		for (uint i = lane; i < 256u; i += WG_LANES) {
			uint acc = 0u;
			for (uint l = 0u; l < cols; ++l)
				acc = mldsa_add(acc, mldsa_mul(g_poly[P_A + l * 256u + i], g_poly[P_Z + l * 256u + i]));
			g_poly[P_W + i] = mldsa_sub(acc, mldsa_mul(g_poly[P_C + i], g_poly[P_T1 + k * 256u + i]));
		}
		wg_sync();
		wg_intt(P_W, lane);
		uint bits = parameter_set == 44u ? 6u : 4u;
		for (uint i = lane; i < 256u; i += WG_LANES)
			g_poly[P_W + i] = mldsa_use_hint(g_poly[P_H + k * 256u + i], g_poly[P_W + i], gamma2);
		wg_sync();
		if (lane == 0u) {
			uint reservoir = 0u, occupied = 0u;
			for (uint i = 0u; i < 256u; ++i) {
				reservoir |= g_poly[P_W + i] << occupied;
				occupied += bits;
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
		wg_sp_finish();
		uint difference = 0u;
		for (uint i = 0u; i < challenge_words; ++i) {
			uint word = ((i & 1u) == 0u) ? g_digest[i / 2u].x : g_digest[i / 2u].y;
			difference |= g_ctilde[i] ^ word;
		}
		g_result = difference == 0u ? 1u : 0u;
	}
	wg_sync();
	return g_result != 0u;
}

// FIPS 204 mu = H(tr || M', 64) over public inputs only. The caller packs the
// M' head bytes — the pure 0x00 or prehash 0x01 domain tag, the context
// length, and the optional eleven-byte OID trailer — into push words so the
// kernel keeps a single branch-free absorb path: tr, head, context, payload.
// Lane 0 owns the serial sponge; every other lane only validates.
bool mldsa_mu_par(
	uint parameter_set,
	RWByteAddressBuffer public_key, uint pk_base, uint pk_length,
	RWByteAddressBuffer payload, uint payload_base, uint payload_length,
	RWByteAddressBuffer context, uint context_base, uint context_length,
	uint4 head, uint head_length,
	RWByteAddressBuffer mu, uint mu_base,
	uint lane
) {
	uint pk_len = parameter_set == 44u ? 1312u : parameter_set == 65u ? 1952u : parameter_set == 87u ? 2592u : 0u;
	if (pk_len == 0u || pk_length != pk_len || context_length > 255u || head_length > 16u) return false;
	if (!pqc_buffer_range(public_key, pk_base, pk_length)
		|| !pqc_buffer_range(payload, payload_base, payload_length)
		|| !pqc_buffer_range(context, context_base, context_length)
		|| !pqc_buffer_range(mu, mu_base, 64u)) return false;
	if (lane == 0u) {
		g_flag = 0u;
		for (uint i = 0u; i < 8u; ++i) g_digest[i] = uint2(0u, 0u);
		wg_sp_init();
		wg_sp_absorb_buffer(public_key, pk_base, pk_length);
		wg_sp_finish();
		wg_sp_init();
		wg_sp_absorb_digest();
		uint words[4] = {head.x, head.y, head.z, head.w};
		// The first two head bytes (domain tag, context length) precede the
		// context; the remaining OID trailer follows it, matching the FIPS 204
		// 0x01 || len(ctx) || ctx || OID || alg || PH(M) stream exactly.
		for (uint i = 0u; i < 2u; ++i)
			wg_sp_absorb_byte((words[i / 4u] >> ((i & 3u) * 8u)) & 255u);
		wg_sp_absorb_buffer(context, context_base, context_length);
		for (uint i = 2u; i < head_length; ++i)
			wg_sp_absorb_byte((words[i / 4u] >> ((i & 3u) * 8u)) & 255u);
		wg_sp_absorb_buffer(payload, payload_base, payload_length);
		wg_sp_finish();
		for (uint w = 0u; w < 8u; ++w) {
			mu.Store(mu_base + w * 8u, g_digest[w].x);
			mu.Store(mu_base + w * 8u + 4u, g_digest[w].y);
		}
	}
	return true;
}

// Batched prehash stage for the public verifier: one workgroup digests one
// variable-length message row into the packed [batch, digest_bytes] staging
// buffer. Every output row stays exactly digest_bytes wide.
bool mldsa_ph_batch_par(
	uint algorithm,
	RWByteAddressBuffer messages, uint message_base, uint message_length,
	RWByteAddressBuffer digests, uint digest_base,
	uint lane
) {
	uint need = mldsa_prehash_bytes(algorithm);
	if (need == 0u || !pqc_buffer_range(messages, message_base, message_length)
		|| !pqc_buffer_range(digests, digest_base, need)) return false;
	if (lane == 0u) {
		g_flag = 0u;
		for (uint i = 0u; i < 16u; ++i) g_d32[i] = 0u;
		wg_prehash_digest(messages, message_base, message_length, algorithm);
		if (g_flag == 0u)
			for (uint w = 0u; w < need / 4u; ++w)
				digests.Store(digest_base + w * 4u, g_d32[w]);
	}
	wg_sync();
	return g_flag == 0u;
}
