// Merkle tree reduction — parallel binary tree hash computation.
//
// Computes Merkle root from N leaf hashes using SHAKE-256.
// Uses parallel reduction: each level halves the number of nodes.
//
// Algorithm:
//   Level 0: N leaves (input)
//   Level 1: N/2 nodes = SHAKE256(leaf[2i] || leaf[2i+1])
//   Level 2: N/4 nodes = SHAKE256(node[2i] || node[2i+1])
//   ...
//   Level log2(N): 1 root
//
// Strategy: iterative reduction with ping-pong buffers.
// Each workgroup processes 256 pairs -> 256 parent hashes.
// Dispatch ceil(N/2 / 256) workgroups per level.
//
// Input: leaves buffer [N * 32] bytes (32-byte hashes)
// Output: root buffer [32] bytes
// Scratch: temp buffer [N * 32] bytes for intermediate levels

import Keccak;

struct PushConstants {
	uint input_idx;
	uint output_idx;
	uint count;        // Number of pairs to process this level
	uint hash_bytes;   // Hash size (typically 32)
};

[[vk::push_constant]] PushConstants pc;

[[vk::binding(0, 0)]] RWByteAddressBuffer heap[];

static const uint RATE_U64 = 17;  // SHAKE-256 rate = 136 bytes = 17 x uint64

[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 gid : SV_GroupID, uint3 tid : SV_GroupThreadID) {
	uint idx = gid.x * 256 + tid.x;
	if (idx >= pc.count) return;

	// Load left and right child hashes
	uint left_offset = (2 * idx) * pc.hash_bytes;
	uint right_offset = (2 * idx + 1) * pc.hash_bytes;

	// Initialize Keccak state
	uint64_t state[25];
	for (uint i = 0; i < 25; i++) state[i] = 0ULL;

	// Absorb left child (32 bytes = 4 x uint64)
	uint absorbed = 0;
	for (uint i = 0; i < pc.hash_bytes / 8; i++) {
		state[absorbed++] ^= heap[pc.input_idx].Load<uint64_t>(left_offset + i * 8);
	}

	// Absorb right child (32 bytes = 4 x uint64)
	for (uint i = 0; i < pc.hash_bytes / 8; i++) {
		state[absorbed++] ^= heap[pc.input_idx].Load<uint64_t>(right_offset + i * 8);
	}

	// Pad and finalize (SHAKE-256 padding: 0x1F)
	// Total absorbed: 64 bytes = 8 x uint64, rate = 17 x uint64
	// Padding: state[8] ^= 0x1F, state[16] ^= 0x80
	state[8] ^= 0x1FULL;
	state[16] ^= 0x8000000000000000ULL;

	// Apply Keccak-f[1600]
	KeccakF1600(state);

	// Squeeze output (32 bytes = 4 x uint64)
	uint out_offset = idx * pc.hash_bytes;
	for (uint i = 0; i < pc.hash_bytes / 8; i++) {
		heap[pc.output_idx].Store<uint64_t>(out_offset + i * 8, state[i]);
	}
}