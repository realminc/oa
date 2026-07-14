// Global max reduction. Single workgroup (256 threads), stride load + tree reduce.
// Dispatch with 1 workgroup.
// Subgroup-size-independent: a plain shared-memory tree reduction. The previous
// WaveActiveMax path assumed exactly 32-lane subgroups AND kept each lane's tree
// result in a local (no shared write-back), so lane 0 only folded wave-leaders
// 0,1,2,4 and silently dropped leaders 3,5,6,7 — the global max was lost whenever
// it landed in a dropped wave. Mirrors the Sum.slang fix.

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

	float best = -3.402823466e+38f;
	for (uint i = lid; i < pc.count; i += 256) {
		best = max(best, OaLoad(heap[pc.x_idx], i));
	}
	g_buf[lid] = best;
	GroupMemoryBarrierWithGroupSync();

	// Inclusive tree reduction: 128 -> 64 -> ... -> 1.
	[unroll] for (uint s = 128; s > 0; s >>= 1) {
		if (lid < s) {
			g_buf[lid] = max(g_buf[lid], g_buf[lid + s]);
		}
		GroupMemoryBarrierWithGroupSync();
	}

	if (lid == 0) {
		OaStore(heap[pc.out_idx], 0, g_buf[0]);
	}
}
