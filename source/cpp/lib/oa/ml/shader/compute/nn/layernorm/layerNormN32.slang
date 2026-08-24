// LayerNorm schedule for rows with at most 32 columns.
// One thread owns one element, retaining x across both reductions and output.

import storage;

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

groupshared float g_sum[32];
groupshared float g_var[32];

float LocalReduceMeanSum32(float val, uint lane) {
	float accum = val;
	g_sum[lane] = accum;
	GroupMemoryBarrierWithGroupSync();
	for (uint s = 16; s > 0; s >>= 1) {
		if (lane < s) {
			accum += g_sum[lane + s];
			g_sum[lane] = accum;
		}
		GroupMemoryBarrierWithGroupSync();
	}
	return g_sum[0];
}

float LocalReduceVarianceSum32(float val, uint lane) {
	float accum = val;
	g_var[lane] = accum;
	GroupMemoryBarrierWithGroupSync();
	for (uint s = 16; s > 0; s >>= 1) {
		if (lane < s) {
			accum += g_var[lane + s];
			g_var[lane] = accum;
		}
		GroupMemoryBarrierWithGroupSync();
	}
	return g_var[0];
}

[shader("compute")]
[numthreads(32, 1, 1)]
void main(uint3 gid : SV_GroupID, uint3 tid : SV_GroupThreadID) {
	uint row = gid.x;
	if (row >= pc.rows) return;

	uint lane = tid.x;
	uint base = row * pc.cols;
	bool active = lane < pc.cols;
	float value = active ? oaLoad(heap[pc.x_idx], base + lane) : 0.0;
	float mean = LocalReduceMeanSum32(value, lane) / float(pc.cols);
	float diff = active ? value - mean : 0.0;
	float inv_std = rsqrt(
		LocalReduceVarianceSum32(diff * diff, lane) / float(pc.cols) + pc.eps);

	if (active) {
		float normed = diff * inv_std;
		oaStore(heap[pc.out_idx], base + lane,
			normed * oaLoad(heap[pc.weight_idx], lane)
			+ oaLoad(heap[pc.bias_idx], lane));
	}
}
