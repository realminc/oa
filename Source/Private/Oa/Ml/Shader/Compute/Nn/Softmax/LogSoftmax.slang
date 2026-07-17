// Stable log-softmax: y_i = x_i - max(x) - log(sum(exp(x - max(x)))).

import Storage;

struct PushConstants {
	uint x_idx;
	uint out_idx;
	uint rows;
	uint cols;
};

[[vk::push_constant]] PushConstants pc;
[[vk::binding(0, 0)]] RWByteAddressBuffer heap[];

groupshared float reduction[256];

[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 group : SV_GroupID, uint3 thread : SV_GroupThreadID) {
	uint row = group.x;
	uint lane = thread.x;
	if (row >= pc.rows) return;
	uint base = row * pc.cols;

	float localMax = -3.402823466e+38f;
	for (uint column = lane; column < pc.cols; column += 256u) {
		localMax = max(localMax, OaLoad(heap[pc.x_idx], base + column));
	}
	reduction[lane] = localMax;
	GroupMemoryBarrierWithGroupSync();
	for (uint width = 128u; width > 0u; width >>= 1u) {
		if (lane < width) reduction[lane] = max(
			reduction[lane], reduction[lane + width]);
		GroupMemoryBarrierWithGroupSync();
	}
	float rowMax = reduction[0];

	float localSum = 0.0f;
	for (uint column = lane; column < pc.cols; column += 256u) {
		localSum += exp(OaLoad(heap[pc.x_idx], base + column) - rowMax);
	}
	reduction[lane] = localSum;
	GroupMemoryBarrierWithGroupSync();
	for (uint width = 128u; width > 0u; width >>= 1u) {
		if (lane < width) reduction[lane] += reduction[lane + width];
		GroupMemoryBarrierWithGroupSync();
	}
	float logNormalizer = rowMax + log(reduction[0]);
	for (uint column = lane; column < pc.cols; column += 256u) {
		OaStore(heap[pc.out_idx], base + column,
			OaLoad(heap[pc.x_idx], base + column) - logNormalizer);
	}
}
