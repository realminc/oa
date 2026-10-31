// OA donor: source/cpp/lib/oa/crypto/shader/compute/merkle/merkleReduce.slang.
// Mechanical extraction from hash/merkle_reduce.slang's verified paired-word
// SHAKE-256 body. Permutation, padding and output bytes remain identical.
// The caller proves the sibling's complete 32-byte range. Public-data helper;
// not a secret execution or constant-time contract.
import keccak;

void merkle_hash_pair(
	inout uint current[8], RWByteAddressBuffer siblings,
	uint sibling_base, bool sibling_left
) {
	uint2 state[25];
	for (uint lane = 0u; lane < 25u; ++lane) state[lane] = uint2(0u, 0u);
	for (uint lane = 0u; lane < 4u; ++lane) {
		state[(sibling_left ? 4u : 0u) + lane] = uint2(current[lane * 2u], current[lane * 2u + 1u]);
		state[(sibling_left ? 0u : 4u) + lane] = uint2(
			siblings.Load(sibling_base + lane * 8u), siblings.Load(sibling_base + lane * 8u + 4u));
	}
	state[8].x ^= 0x1fu;
	state[16].y ^= 0x80000000u;
	keccak_f1600_u32(state);
	for (uint lane = 0u; lane < 4u; ++lane) {
		current[lane * 2u] = state[lane].x;
		current[lane * 2u + 1u] = state[lane].y;
	}
}
