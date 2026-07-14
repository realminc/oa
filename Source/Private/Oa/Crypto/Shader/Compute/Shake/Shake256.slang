// SHAKE-256 batch hash — GPU implementation.
//
// Absorbs variable-length messages, squeezes fixed output.
// Built on Keccak-f[1600]. Rate = 136 bytes (1088 bits) for SHAKE-256.
//
// One thread per hash instance. Dispatch ceil(N / 256) workgroups.
// The shader applies SHAKE's 0x1f/0x80 pad10*1 suffix directly while reading
// the original [N, message_len] byte rows, so deferred GPU inputs never need a
// host readback/re-upload boundary.
//
// Input layout (per instance): messages [N, message_len] bytes.
//   squeeze_len: number of uint64 to squeeze (must be <= output buffer capacity)
// Output: [N * squeeze_len] uint64

import Keccak;

struct PushConstants {
	uint messages_idx;
	uint output_idx;
	uint count;
	uint message_len;
	uint num_blocks;
	uint squeeze_u64;
};

[[vk::push_constant]] PushConstants pc;

[[vk::binding(0, 0)]] RWByteAddressBuffer heap[];

static const uint RATE_U64 = 17;
static const uint RATE_BYTES = RATE_U64 * 8;

uint LoadMessageByte(uint InRowBase, uint InByte) {
	uint address = InRowBase + InByte;
	uint word = heap[pc.messages_idx].Load(address & ~3u);
	return (word >> ((address & 3u) * 8u)) & 0xffu;
}

uint64_t LoadPaddedLane(uint InRowBase, uint InPaddedByte) {
	uint64_t lane = 0;
	for (uint byte = 0; byte < 8; ++byte) {
		uint pos = InPaddedByte + byte;
		uint value = pos < pc.message_len ? LoadMessageByte(InRowBase, pos) : 0u;
		if (pos == pc.message_len) value |= 0x1fu;
		if (pos == pc.num_blocks * RATE_BYTES - 1u) value |= 0x80u;
		lane |= uint64_t(value) << (byte * 8u);
	}
	return lane;
}

[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 gid : SV_GroupID, uint3 tid : SV_GroupThreadID) {
	uint idx = gid.x * 256 + tid.x;
	if (idx >= pc.count) return;

	uint64_t state[25];
	for (uint i = 0; i < 25; i++) state[i] = 0ULL;

	uint msg_base = idx * pc.message_len;
	for (uint blk = 0; blk < pc.num_blocks; blk++) {
		for (uint i = 0; i < RATE_U64; i++) {
			state[i] ^= LoadPaddedLane(msg_base, blk * RATE_BYTES + i * 8u);
		}
		KeccakF1600(state);
	}

	uint out_base = idx * pc.squeeze_u64;
	uint remaining = pc.squeeze_u64;
	uint out_off = 0;

	while (remaining > 0) {
		uint to_copy = remaining < RATE_U64 ? remaining : RATE_U64;
		for (uint i = 0; i < to_copy; i++) {
			heap[pc.output_idx].Store<uint64_t>((out_base + out_off + i) * 8, state[i]);
		}
		out_off += to_copy;
		remaining -= to_copy;
		if (remaining > 0) {
			KeccakF1600(state);
		}
	}
}
