// Narrow-row gated RMSNorm schedule for at most 32 columns.
// One lane owns one value, avoiding the 224 idle lanes and three redundant
// reduction rounds of the general 256-thread schedule.

import storage;
import activations;

struct PushConstants {
	uint x_idx;
	uint weight_idx;
	uint bias_idx;
	uint z_idx;
	uint out_idx;
	uint rows;
	uint cols;
	uint groups;
	float eps;
	uint has_bias;
	uint has_z;
	uint norm_before_gate;
};

[[vk::push_constant]] PushConstants pc;
[[vk::binding(0, 0)]] RWByteAddressBuffer heap[];

groupshared float g_sum[32];

float ReduceSum32(float value, uint lane) {
	g_sum[lane] = value;
	GroupMemoryBarrierWithGroupSync();
	for (uint stride = 16; stride > 0; stride >>= 1) {
		if (lane < stride) g_sum[lane] += g_sum[lane + stride];
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
	uint affineBase = (row % pc.groups) * pc.cols;
	bool active = lane < pc.cols;
	float x = active ? oaLoad(heap[pc.x_idx], base + lane) : 0.0;
	float invRms = rsqrt(ReduceSum32(x * x, lane) / float(pc.cols) + pc.eps);

	if (pc.has_z != 0 && pc.norm_before_gate == 0) {
		if (active) {
			x *= Silu(oaLoad(heap[pc.z_idx], base + lane));
			oaStore(heap[pc.x_idx], base + lane, x);
		}
		GroupMemoryBarrierWithGroupSync();
		invRms = rsqrt(ReduceSum32(x * x, lane) / float(pc.cols) + pc.eps);
	}

	if (active) {
		float value = x * invRms * oaLoad(heap[pc.weight_idx], affineBase + lane);
		if (pc.has_bias != 0) value += oaLoad(heap[pc.bias_idx], affineBase + lane);
		if (pc.has_z != 0 && pc.norm_before_gate != 0) {
			value *= Silu(oaLoad(heap[pc.z_idx], base + lane));
		}
		oaStore(heap[pc.out_idx], base + lane, value);
	}
}
