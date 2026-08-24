// Scaled masked softmax schedule for rows with at most 32 columns.
// One thread retains its transformed score and exponent across both reductions.

import storage;

struct PushConstants {
	uint scores_idx;
	uint mask_idx;
	uint out_idx;
	uint rows;
	uint cols;
	float scale;
};

[[vk::push_constant]] PushConstants pc;

[[vk::binding(0, 0)]] RWByteAddressBuffer heap[];

groupshared float g_max[32];
groupshared float g_sum[32];

float LocalReduceMax32(float val, uint lane) {
	float accum = val;
	g_max[lane] = accum;
	GroupMemoryBarrierWithGroupSync();
	for (uint s = 16; s > 0; s >>= 1) {
		if (lane < s) {
			accum = max(accum, g_max[lane + s]);
			g_max[lane] = accum;
		}
		GroupMemoryBarrierWithGroupSync();
	}
	return g_max[0];
}

float LocalReduceSum32(float val, uint lane) {
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

[shader("compute")]
[numthreads(32, 1, 1)]
void main(uint3 gid : SV_GroupID, uint3 tid : SV_GroupThreadID) {
	uint row = gid.x;
	if (row >= pc.rows) return;

	uint lane = tid.x;
	uint base = row * pc.cols;
	bool active = lane < pc.cols;
	float value = -3.402823466e+38f;
	if (active) {
		value = oaLoad(heap[pc.scores_idx], base + lane) * pc.scale;
		value += oaLoadF32(heap[pc.mask_idx], base + lane);
	}
	float rowMax = LocalReduceMax32(value, lane);
	float exponent = active ? exp(value - rowMax) : 0.0;
	float invSum = 1.0 / LocalReduceSum32(exponent, lane);
	if (active) {
		oaStore(heap[pc.out_idx], base + lane, exponent * invSum);
	}
}
