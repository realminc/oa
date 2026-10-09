// FIPS 203 Algorithms 7/8 and PRF_eta (Section 4.1). No qualified OA
// ML-KEM donor exists; new composition over OA's proven uint2 Keccak body.
// Public-data laboratory only, not a secret execution/constant-time claim.
import keccak;
import mlkem_field;

// Short seed SHAKE: 32-byte seed followed by one (PRF) or two (matrix) bytes.
// Caller supplies byte-valued suffix and matching rate (136 or 168).
void mlkem_seed_xof(uint seed[8], uint suffix, uint suffixBytes, uint rate,
	out uint2 state[25]) {
	for (uint i = 0u; i < 25u; ++i) state[i] = uint2(0u, 0u);
	for (uint i = 0u; i < 8u; ++i) {
		if ((i & 1u) == 0u) state[i / 2u].x = seed[i];
		else state[i / 2u].y = seed[i];
	}
	state[4].x = suffix | (31u << (suffixBytes * 8u));
	state[rate / 8u - 1u].y ^= 0x80000000u;
	keccak_f1600_u32(state);
}

// Advancing XOF cursor; refill only at a rate boundary, never discard bytes.
uint mlkem_xof_next(inout uint2 state[25], inout uint position, uint rate) {
	if (position == rate) {
		keccak_f1600_u32(state);
		position = 0u;
	}
	uint word = (position % 8u) < 4u ? state[position / 8u].x : state[position / 8u].y;
	uint value = (word >> ((position % 4u) * 8u)) & 255u;
	++position;
	return value;
}

// Both candidates are considered in order. count <= 256; the second candidate
// must not write past the final coefficient. Candidates are 12-bit integers.
void mlkem_accept_pair(uint first, uint second, inout uint count, inout uint poly[256]) {
	if (first < MLKEM_Q && count < 256u) poly[count++] = first;
	if (second < MLKEM_Q && count < 256u) poly[count++] = second;
}

// Outputs NTT-domain coefficients directly; no additional transform is needed.
// Unbounded rejection has no silent truncation. GPU dispatch/watchdog policy
// remains a separate admission gate. Indices are bytes in the requested order.
void mlkem_sample_ntt(uint seed[8], uint firstIndex, uint secondIndex, out uint poly[256]) {
	uint2 state[25];
	mlkem_seed_xof(seed, firstIndex | (secondIndex << 8u), 2u, 168u, state);
	uint count = 0u, position = 0u;
	while (count < 256u) {
		uint a = mlkem_xof_next(state, position, 168u);
		uint b = mlkem_xof_next(state, position, 168u);
		uint c = mlkem_xof_next(state, position, 168u);
		mlkem_accept_pair(a | ((b & 15u) << 8u), (b >> 4u) | (c << 4u), count, poly);
	}
}

// PRF_eta(seed, nonce) = SHAKE256(seed || nonce, 64*eta).
// eta is 2 or 3; nonce is a byte. Streaming avoids a materialized PRF byte
// array. Consume exactly 2*eta bits/coefficient in little-endian order.
void mlkem_sample_noise(uint seed[8], uint nonce, uint eta, out uint poly[256]) {
	uint2 state[25];
	mlkem_seed_xof(seed, nonce, 1u, 136u, state);
	uint position = 0u, reservoir = 0u, available = 0u;
	uint mask = (1u << eta) - 1u;
	for (uint i = 0u; i < 256u; ++i) {
		while (available < 2u * eta) {
			reservoir |= mlkem_xof_next(state, position, 136u) << available;
			available += 8u;
		}
		uint positive = countbits(reservoir & mask);
		uint negative = countbits((reservoir >> eta) & mask);
		poly[i] = mlkem_sub(positive, negative);
		reservoir >>= 2u * eta;
		available -= 2u * eta;
	}
}
