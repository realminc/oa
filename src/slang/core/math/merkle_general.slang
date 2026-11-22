// Full-algorithm public Merkle pair hashing. Hash-family algorithms (0..=5)
// delegate to merkle_general_hash; Poseidon2 field algorithms (6..=9) add the
// pinned permutation import only to kernels that can execute them.
import merkle_general_hash;
import merkle_poseidon2;

void merkle_general_pair(
	inout uint current[16],
	uint sibling[16],
	bool sibling_left,
	uint algorithm
) {
	if (algorithm < 6u) {
		merkle_hash_pair(current, sibling, sibling_left, algorithm);
		return;
	}
	uint message[32];
	uint words = merkle_digest_bytes(algorithm) / 4u;
	for (uint i = 0u; i < 32u; ++i) message[i] = 0u;
	for (uint i = 0u; i < words; ++i) {
		message[i] = sibling_left ? sibling[i] : current[i];
		message[i + words] = sibling_left ? current[i] : sibling[i];
	}
	merkle_poseidon2_pair(message, algorithm, current);
}
