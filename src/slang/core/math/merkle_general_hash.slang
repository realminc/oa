// Hash-family public Merkle pair hashing. SHA-2 compression bodies and Keccak
// are imported unchanged from OA's qualified helpers; IV initialization was
// mechanically extracted for reuse. New composition; no C++ multi-hash donor.
// Poseidon2 field algorithms (6..=9) live in merkle_general so hash-only
// kernels do not pay the field-permutation import.
import sha2_u32;
import sha2_u32_pair;
import keccak;

uint merkle_digest_bytes(uint algorithm) {
	return algorithm == 1u || algorithm == 3u ? 64u : 32u;
}

void merkle_hash_pair(
	inout uint current[16],
	uint sibling[16],
	bool sibling_left,
	uint algorithm
) {
	uint message[32];
	uint words = merkle_digest_bytes(algorithm) / 4u;
	for (uint i = 0u; i < 32u; ++i) message[i] = 0u;
	for (uint i = 0u; i < words; ++i) {
		message[i] = sibling_left ? sibling[i] : current[i];
		message[i + words] = sibling_left ? current[i] : sibling[i];
	}
	if (algorithm == 0u) {
		uint state[8], block[16];
		sha2_u32_initialize(256u, state);
		for (uint i = 0u; i < 16u; ++i) block[i] = sha2_u32_swap(message[i]);
		sha2_u32_compress(state, block);
		for (uint i = 0u; i < 16u; ++i) block[i] = 0u;
		block[0] = 0x80000000u; block[15] = 512u;
		sha2_u32_compress(state, block);
		for (uint i = 0u; i < 8u; ++i) current[i] = sha2_u32_swap(state[i]);
		return;
	}
	if (algorithm == 1u) {
		uint2 state[8], block[16];
		sha2_u32_pair_initialize(512u, state);
		for (uint i = 0u; i < 16u; ++i)
			block[i] = uint2(sha2_u32_swap(message[i * 2u + 1u]), sha2_u32_swap(message[i * 2u]));
		sha2_u32_pair_compress(state, block);
		for (uint i = 0u; i < 16u; ++i) block[i] = uint2(0u, 0u);
		block[0] = uint2(0u, 0x80000000u); block[15] = uint2(1024u, 0u);
		sha2_u32_pair_compress(state, block);
		for (uint i = 0u; i < 8u; ++i) {
			current[i * 2u] = sha2_u32_swap(state[i].y);
			current[i * 2u + 1u] = sha2_u32_swap(state[i].x);
		}
		return;
	}
	uint rate = algorithm == 2u ? 136u : algorithm == 3u ? 72u : algorithm == 4u ? 168u : 136u;
	uint domain = algorithm == 2u || algorithm == 3u ? 0x06u : 0x1fu;
	uint2 state[25];
	for (uint i = 0u; i < 25u; ++i) state[i] = uint2(0u, 0u);
	uint cursor = 0u;
	for (uint i = 0u; i < words * 8u; ++i) {
		uint value = (message[i / 4u] >> ((i % 4u) * 8u)) & 255u;
		uint lane = cursor / 8u;
		if (cursor % 8u < 4u) state[lane].x ^= value << ((cursor % 4u) * 8u);
		else state[lane].y ^= value << ((cursor % 4u) * 8u);
		if (++cursor == rate) { keccak_f1600_u32(state); cursor = 0u; }
	}
	uint lane = cursor / 8u;
	if (cursor % 8u < 4u) state[lane].x ^= domain << ((cursor % 4u) * 8u);
	else state[lane].y ^= domain << ((cursor % 4u) * 8u);
	state[rate / 8u - 1u].y ^= 0x80000000u;
	keccak_f1600_u32(state);
	for (uint i = 0u; i < words; ++i) current[i] = i % 2u == 0u ? state[i / 2u].x : state[i / 2u].y;
}
