// SmoothL1Mean forward: SmoothL1(A, B) reduced to a scalar mean in one pass.
// Avoids the intermediate per-element buffer + separate Mean dispatch.
// Input: a, b (flat, count elements). Output: out[0] = mean(SmoothL1(a, b)).

import Storage;

struct PushConstants {
	uint a_idx;
	uint b_idx;
	uint out_idx;
	uint count;
};

[[vk::push_constant]] PushConstants pc;

[[vk::binding(0, 0)]] RWByteAddressBuffer heap[];

groupshared float g_buf[256];

float SmoothL1(float diff) {
	float absDiff = abs(diff);
	if (absDiff < 1.0f) {
		return 0.5f * diff * diff;
	} else {
		return absDiff - 0.5f;
	}
}

float LocalReduceSum256(float val, uint lane) {
	g_buf[lane] = val;
	GroupMemoryBarrierWithGroupSync();
	for (uint s = 128; s > 0; s >>= 1) {
		if (lane < s) {
			g_buf[lane] += g_buf[lane + s];
		}
		GroupMemoryBarrierWithGroupSync();
	}
	return g_buf[0];
}

[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
	uint lane = tid.x;

	float local_sum = 0.0f;
	for (uint idx = lane; idx < pc.count; idx += 256) {
		float a = OaLoad(heap[pc.a_idx], idx);
		float b = OaLoad(heap[pc.b_idx], idx);
		local_sum += SmoothL1(a - b);
	}

	float total = LocalReduceSum256(local_sum, lane);
	if (lane == 0) {
		OaStore(heap[pc.out_idx], 0, total / float(pc.count));
	}
}
