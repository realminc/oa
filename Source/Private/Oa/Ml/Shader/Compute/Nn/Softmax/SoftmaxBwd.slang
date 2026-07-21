// Softmax backward: dx_i = s_i * (d_out_i - sum(d_out * s))
// Inlined reduction to avoid cross-module groupshared bug.

import Storage;

struct PushConstants {
	uint softmax_out_idx;
	uint d_out_idx;
	uint d_x_idx;
	uint outer_size;
	uint dim_size;
	uint inner_size;
};

[[vk::push_constant]] PushConstants pc;

[[vk::binding(0, 0)]] RWByteAddressBuffer heap[];

groupshared float g_buf[256];

[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 gid : SV_GroupID, uint3 tid : SV_GroupThreadID) {
	uint group = gid.x;
	if (group >= pc.outer_size * pc.inner_size) return;

	uint lane = tid.x;
	uint outer = group / pc.inner_size;
	uint inner = group % pc.inner_size;
	uint base = outer * pc.dim_size * pc.inner_size + inner;

	float localDot = 0.0;
	for (uint i = lane; i < pc.dim_size; i += 256) {
		uint index = base + i * pc.inner_size;
		localDot += OaLoad(heap[pc.d_out_idx], index)
			* OaLoad(heap[pc.softmax_out_idx], index);
	}

	g_buf[lane] = localDot;
	GroupMemoryBarrierWithGroupSync();
	for (uint s = 128; s > 0; s >>= 1) {
		if (lane < s) g_buf[lane] += g_buf[lane + s];
		GroupMemoryBarrierWithGroupSync();
	}
	float dot = g_buf[0];

	GroupMemoryBarrierWithGroupSync();

	for (uint i = lane; i < pc.dim_size; i += 256) {
		uint index = base + i * pc.inner_size;
		float s = OaLoad(heap[pc.softmax_out_idx], index);
		float d = OaLoad(heap[pc.d_out_idx], index);
		OaStore(heap[pc.d_x_idx], index, s * (d - dot));
	}
}
