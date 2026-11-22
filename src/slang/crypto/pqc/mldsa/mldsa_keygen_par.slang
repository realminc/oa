// Workgroup-parallel ML-DSA-44/65/87 key generation, FIPS 204 Algorithms 6,
// 24 and 31. One workgroup of 128 lanes generates one key pair; polynomial and
// XOF state lives in groupshared storage and every helper takes scalar
// arguments only, matching the contract in mldsa_verification_par.slang.
// Public deterministic test seeds only until secret execution is admitted.
import shake;
import mldsa_verification_par;
import mldsa_field;
import mldsa_ntt;
import mldsa_packing;
import mldsa_sampling;

// Keygen regions inside the shared flat polynomial workspace.
static const uint KG_S1 = 0u;            // 7 polynomials, kept in NTT form
static const uint KG_S2 = 7u * 256u;     // 8 polynomials, time domain
static const uint KG_A = 15u * 256u;     // 7 per-row ExpandA staging slots
static const uint KG_T = 22u * 256u;     // 1 row accumulator

// RejBoundedPoly on a shared XOF slot, FIPS 204 Algorithm 35: absorb
// rho_prime[64] || nonce u16le, then accept eta-limited nibbles.
[noinline]
void wg_expand_secret(uint slot, uint eta, uint nonce, uint dest) {
	for (uint i = 0u; i < 25u; ++i) g_state[slot][i] = uint2(0u, 0u);
	for (uint b = 0u; b < 66u; ++b) {
		uint byte;
		if (b < 64u) byte = (g_rho_prime[b / 4u] >> ((b & 3u) * 8u)) & 255u;
		else byte = b == 64u ? (nonce & 255u) : ((nonce >> 8u) & 255u);
		uint lane = b / 8u;
		uint boff = b % 8u;
		if (boff < 4u) g_state[slot][lane].x ^= (byte << (boff * 8u));
		else g_state[slot][lane].y ^= (byte << ((boff - 4u) * 8u));
	}
	{
		uint lane = 66u / 8u;
		uint boff = 66u % 8u;
		if (boff < 4u) g_state[slot][lane].x ^= 0x1Fu << (boff * 8u);
		else g_state[slot][lane].y ^= 0x1Fu << ((boff - 4u) * 8u);
	}
	g_state[slot][16].y ^= 0x80000000u;
	wg_keccak(slot);

	uint count = 0u;
	uint pos = 0u;
	while (count < 256u) {
		if (pos == 136u) {
			wg_keccak(slot);
			pos = 0u;
		}
		uint value = wg_xof_byte(slot, pos++);
		for (uint half = 0u; half < 2u && count < 256u; ++half) {
			uint nibble = (value >> (half * 4u)) & 15u;
			if (eta == 2u ? nibble < 15u : nibble < 9u) {
				uint reduced = eta == 2u ? nibble % 5u : nibble;
				int coefficient = (int)eta - (int)reduced;
				g_poly[dest + count] = coefficient < 0
					? (uint)(coefficient + (int)MLDSA_Q) : (uint)coefficient;
				++count;
			}
		}
	}
}

// Whole-buffer zero pass split across lanes before packing begins.
[noinline]
void wg_clear_buffer(RWByteAddressBuffer buf, uint words, uint lane) {
	for (uint i = lane; i < words; i += WG_LANES) buf.Store(i * 4u, 0u);
}

// FIPS 204 KeyGen_internal on one workgroup. seed is 32 secret-or-fixture
// bytes; public_key and secret_key are the generated FIPS encodings written
// directly to the output buffers.
bool mldsa_keygen_par(
	uint parameter_set,
	RWByteAddressBuffer seed,
	RWByteAddressBuffer public_key,
	RWByteAddressBuffer secret_key,
	uint lane
) {
	if (parameter_set != 44u && parameter_set != 65u && parameter_set != 87u) return false;
	uint rows = parameter_set == 44u ? 4u : (parameter_set == 65u ? 6u : 8u);
	uint cols = parameter_set == 44u ? 4u : (parameter_set == 65u ? 5u : 7u);
	uint eta = parameter_set == 65u ? 4u : 2u;
	uint eta_bits = eta == 2u ? 3u : 4u;
	uint eta_bytes = eta_bits * 32u;
	uint pk_bytes = 32u + rows * 320u;
	uint sk_bytes = 128u + (cols + rows) * eta_bytes + rows * 416u;
	if (!pqc_buffer_range(seed, 0u, 32u)
		|| !pqc_buffer_range(public_key, 0u, pk_bytes)
		|| !pqc_buffer_range(secret_key, 0u, sk_bytes)) return false;

	if (lane == 0u) {
		g_flag = 0u;
		g_result = 0u;
		for (uint i = 0u; i < 8u; ++i) g_rho[i] = 0u;
		for (uint i = 0u; i < 16u; ++i) g_rho_prime[i] = 0u;
	}
	wg_clear_buffer(public_key, pk_bytes / 4u, lane);
	wg_clear_buffer(secret_key, sk_bytes / 4u, lane);
	wg_sync();

	// Seed expansion: lane 0 drives the serial sponge for
	// rho || rho_prime || K while other lanes wait.
	if (lane == 0u) {
		wg_sp_init();
		wg_sp_absorb_buffer(seed, 0u, 32u);
		wg_sp_absorb_byte(rows);
		wg_sp_absorb_byte(cols);
		wg_sp_finalize();
		for (uint i = 0u; i < 32u; ++i) {
			uint value = wg_sp_squeeze_byte();
			uint address = i;
			public_key.Store(address & ~3u,
				public_key.Load(address & ~3u) | (value << ((address & 3u) * 8u)));
			secret_key.Store(address & ~3u,
				secret_key.Load(address & ~3u) | (value << ((address & 3u) * 8u)));
			g_rho[i / 4u] |= value << ((i % 4u) * 8u);
		}
		for (uint i = 0u; i < 64u; ++i)
			g_rho_prime[i / 4u] |= wg_sp_squeeze_byte() << ((i % 4u) * 8u);
		for (uint i = 0u; i < 32u; ++i) {
			uint value = wg_sp_squeeze_byte();
			uint address = 32u + i;
			secret_key.Store(address & ~3u,
				secret_key.Load(address & ~3u) | (value << ((address & 3u) * 8u)));
		}
	}
	wg_sync();

	// One lane per s1 column expands RejBoundedPoly, packs the eta encoding
	// into the secret key, then every column's NTT runs across the workgroup.
	if (lane < cols) {
		wg_expand_secret(lane, eta, lane, KG_S1 + lane * 256u);
		wg_pack_output(secret_key, 128u + lane * eta_bytes, KG_S1 + lane * 256u,
			eta_bits, 0u, eta);
	}
	wg_sync();
	for (uint col = 0u; col < cols; ++col) wg_ntt(KG_S1 + col * 256u, lane);
	if (lane < rows) {
		wg_expand_secret(lane, eta, cols + lane, KG_S2 + lane * 256u);
		wg_pack_output(secret_key, 128u + (cols + lane) * eta_bytes,
			KG_S2 + lane * 256u, eta_bits, 0u, eta);
	}
	wg_sync();

	// Per row: parallel ExpandA per column, accumulate, inverse transform,
	// add s2, then pack t0 into the secret key and t1 into the public key.
	for (uint row = 0u; row < rows; ++row) {
		if (lane < cols) wg_expand_a(lane, row, lane, KG_A + lane * 256u);
		wg_sync();
		for (uint i = lane; i < 256u; i += WG_LANES) {
			uint acc = 0u;
			for (uint l = 0u; l < cols; ++l)
				acc = mldsa_add(acc, mldsa_mul(g_poly[KG_A + l * 256u + i], g_poly[KG_S1 + l * 256u + i]));
			g_poly[KG_T + i] = acc;
		}
		wg_sync();
		wg_intt(KG_T, lane);
		for (uint i = lane; i < 256u; i += WG_LANES)
			g_poly[KG_T + i] = mldsa_add(g_poly[KG_T + i], g_poly[KG_S2 + row * 256u + i]);
		wg_sync();
		if (lane == row) {
			wg_pack_output(secret_key,
				128u + (cols + rows) * eta_bytes + row * 416u, KG_T, 13u, 1u, 0u);
			wg_pack_output(public_key, 32u + row * 320u, KG_T, 10u, 2u, 0u);
		}
		wg_sync();
	}

	// tr = H(pk) into secret_key[64..128); lane 0 owns the serial sponge.
	if (lane == 0u) {
		wg_sp_init();
		wg_sp_absorb_buffer(public_key, 0u, pk_bytes);
		wg_sp_finalize();
		for (uint i = 0u; i < 64u; ++i) {
			uint value = wg_sp_squeeze_byte();
			uint address = 64u + i;
			secret_key.Store(address & ~3u,
				secret_key.Load(address & ~3u) | (value << ((address & 3u) * 8u)));
		}
	}
	wg_sync();
	return true;
}
