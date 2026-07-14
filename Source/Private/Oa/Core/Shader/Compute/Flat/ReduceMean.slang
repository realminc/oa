import Storage;

struct PushConstants {
	uint input_idx;
	uint out_idx;
	uint count;
};

[[vk::push_constant]] PushConstants pc;

[[vk::binding(0, 0)]] RWByteAddressBuffer heap[];

groupshared float g_buf[256];

[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 gid : SV_GroupID, uint3 tid : SV_GroupThreadID) {
	uint lane = tid.x;

	float local_sum = 0.0;
	for (uint i = lane; i < pc.count; i += 256) {
		local_sum += OaLoadF32(heap[pc.input_idx], i);
	}
	g_buf[lane] = local_sum;
	GroupMemoryBarrierWithGroupSync();

	// Subgroup-size-independent tree reduction: 128 -> 64 -> ... -> 1.
	[unroll] for (uint s = 128; s > 0; s >>= 1) {
		if (lane < s) {
			g_buf[lane] += g_buf[lane + s];
		}
		GroupMemoryBarrierWithGroupSync();
	}

	if (lane == 0) {
		OaStoreF32(heap[pc.out_idx], 0, g_buf[0] / float(pc.count));
	}
}
