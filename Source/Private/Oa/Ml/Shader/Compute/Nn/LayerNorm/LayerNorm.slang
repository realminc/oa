// LayerNorm: (x - mean) / sqrt(var + eps) * weight + bias
// Inlined reduction to avoid cross-module groupshared bug.

import Storage;

struct PushConstants {
	uint x_idx;
	uint weight_idx;
	uint bias_idx;
	uint out_idx;
	uint rows;
	uint cols;
	float eps;
};

[[vk::push_constant]] PushConstants pc;

[[vk::binding(0, 0)]] RWByteAddressBuffer heap[];

groupshared float g_buf[256];

float LocalReduceSum256(float val, uint lane) {
	g_buf[lane] = val;
	GroupMemoryBarrierWithGroupSync();
	for (uint s = 128; s > 0; s >>= 1) {
		if (lane < s) g_buf[lane] += g_buf[lane + s];
		GroupMemoryBarrierWithGroupSync();
	}
	return g_buf[0];
}

[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 gid : SV_GroupID, uint3 tid : SV_GroupThreadID) {
	uint row = gid.x;
	if (row >= pc.rows) return;

	uint lane = tid.x;
	uint base = row * pc.cols;

	float local_sum = 0.0;
	for (uint i = lane; i < pc.cols; i += 256) {
		local_sum += OaLoad(heap[pc.x_idx], base + i);
	}
	float mean = LocalReduceSum256(local_sum, lane) / float(pc.cols);

	GroupMemoryBarrierWithGroupSync();

	float local_var = 0.0;
	for (uint i = lane; i < pc.cols; i += 256) {
		float diff = OaLoad(heap[pc.x_idx], base + i) - mean;
		local_var += diff * diff;
	}
	float inv_std = rsqrt(LocalReduceSum256(local_var, lane) / float(pc.cols) + pc.eps);

	for (uint i = lane; i < pc.cols; i += 256) {
		float normed = (OaLoad(heap[pc.x_idx], base + i) - mean) * inv_std;
		OaStore(heap[pc.out_idx], base + i, normed * OaLoad(heap[pc.weight_idx], i) + OaLoad(heap[pc.bias_idx], i));
	}
}
