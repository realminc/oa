// Merkle proof verification — parallel batch verification.
//
// Verifies multiple Merkle proofs in parallel.
// Each proof consists of a leaf hash, sibling hashes, and path indices.
//
// Algorithm for each proof:
//   current = leaf_hash
//   for each level in proof:
//     sibling = proof[level]
//     if path_bit[level] == 0:
//       current = SHAKE256(current || sibling)
//     else:
//       current = SHAKE256(sibling || current)
//   return current == expected_root
//
// Input layout:
//   leaves: [N * 32] bytes - leaf hashes to verify
//   proofs: [N * proof_depth * 32] bytes - sibling hashes for each level
//   paths: [N * 4] bytes - path indices (bit i = direction at level i)
//   roots: [N * 32] bytes - expected root hashes
// Output:
//   results: [N * 4] bytes - 1 if valid, 0 if invalid
//
// One thread per proof. Dispatch ceil(N / 256) workgroups.

import Keccak;

struct PushConstants {
	uint leaves_idx;
	uint proofs_idx;
	uint paths_idx;
	uint roots_idx;
	uint results_idx;
	uint count;
	uint proof_depth;  // Number of levels in the tree
	uint hash_bytes;   // Hash size (typically 32)
};

[[vk::push_constant]] PushConstants pc;

[[vk::binding(0, 0)]] RWByteAddressBuffer heap[];

static const uint RATE_U64 = 17;  // SHAKE-256 rate = 136 bytes

void shake256_pair(uint64_t left[4], uint64_t right[4], out uint64_t output[4]) {
	// Hash two 32-byte values together
	uint64_t state[25];
	for (uint i = 0; i < 25; i++) state[i] = 0ULL;

	// Absorb left (32 bytes = 4 x uint64)
	for (uint i = 0; i < 4; i++) {
		state[i] ^= left[i];
	}

	// Absorb right (32 bytes = 4 x uint64)
	for (uint i = 0; i < 4; i++) {
		state[4 + i] ^= right[i];
	}

	// Pad (SHAKE-256: 0x1F at byte 64, 0x80 at byte 135)
	state[8] ^= 0x1FULL;
	state[16] ^= 0x8000000000000000ULL;

	// Apply Keccak-f[1600]
	KeccakF1600(state);

	// Squeeze output (32 bytes = 4 x uint64)
	for (uint i = 0; i < 4; i++) {
		output[i] = state[i];
	}
}

[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 gid : SV_GroupID, uint3 tid : SV_GroupThreadID) {
	uint idx = gid.x * 256 + tid.x;
	if (idx >= pc.count) return;

	// Load leaf hash (current hash in the path)
	uint64_t current[4];
	uint leaf_offset = idx * pc.hash_bytes;
	for (uint i = 0; i < 4; i++) {
		current[i] = heap[pc.leaves_idx].Load<uint64_t>(leaf_offset + i * 8);
	}

	// Load path indices (which side is the sibling at each level)
	uint path = heap[pc.paths_idx].Load<uint>(idx * 4);

	// Traverse up the tree
	for (uint level = 0; level < pc.proof_depth; level++) {
		// Load sibling hash for this level
		uint64_t sibling[4];
		uint sibling_offset = (idx * pc.proof_depth + level) * pc.hash_bytes;
		for (uint i = 0; i < 4; i++) {
			sibling[i] = heap[pc.proofs_idx].Load<uint64_t>(sibling_offset + i * 8);
		}

		// Determine order based on path bit
		uint64_t left[4], right[4];
		bool current_is_left = ((path >> level) & 1) == 0;

		if (current_is_left) {
			// Current is left child, sibling is right
			for (uint i = 0; i < 4; i++) {
				left[i] = current[i];
				right[i] = sibling[i];
			}
		} else {
			// Sibling is left child, current is right
			for (uint i = 0; i < 4; i++) {
				left[i] = sibling[i];
				right[i] = current[i];
			}
		}

		// Hash to get parent
		shake256_pair(left, right, current);
	}

	// Load expected root
	uint64_t expected_root[4];
	uint root_offset = idx * pc.hash_bytes;
	for (uint i = 0; i < 4; i++) {
		expected_root[i] = heap[pc.roots_idx].Load<uint64_t>(root_offset + i * 8);
	}

	// Compare computed root with expected root
	bool valid = true;
	for (uint i = 0; i < 4; i++) {
		if (current[i] != expected_root[i]) {
			valid = false;
			break;
		}
	}

	// Store result
	heap[pc.results_idx].Store<uint>(idx * 4, valid ? 1u : 0u);
}