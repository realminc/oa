// Global sum reduction. Single workgroup (256 threads), stride load + tree reduce.
// Dispatch with 1 workgroup.
// Subgroup-size-independent: a plain shared-memory tree reduction. The previous
// WaveActiveSum path assumed exactly 32-lane subgroups and dropped partial sums
// on drivers/devices with a different subgroup width (huge relative error).

import Storage;

struct PushConstants {
	uint x_idx;
	uint out_idx;
	uint count;
};

[[vk::push_constant]] PushConstants pc;

[[vk::binding(0, 0)]] RWByteAddressBuffer heap[];

groupshared float g_buf[256];

[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 tid : SV_GroupThreadID) {
	uint lid = tid.x;

	float acc = 0.0;
	for (uint i = lid; i < pc.count; i += 256) {
		acc += OaLoad(heap[pc.x_idx], i);
	}
	g_buf[lid] = acc;
	GroupMemoryBarrierWithGroupSync();

	// Inclusive tree reduction: 128 -> 64 -> ... -> 1.
	[unroll] for (uint s = 128; s > 0; s >>= 1) {
		if (lid < s) {
			g_buf[lid] += g_buf[lid + s];
		}
		GroupMemoryBarrierWithGroupSync();
	}

	if (lid == 0) {
		OaStore(heap[pc.out_idx], 0, g_buf[0]);
	}
}
