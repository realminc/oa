// Stable log-softmax: y_i = x_i - max(x) - log(sum(exp(x - max(x)))).

import Storage;

struct PushConstants {
	uint x_idx;
	uint out_idx;
	uint outer_size;
	uint dim_size;
	uint inner_size;
};

[[vk::push_constant]] PushConstants pc;
[[vk::binding(0, 0)]] RWByteAddressBuffer heap[];

groupshared float reduction[256];

[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 group : SV_GroupID, uint3 thread : SV_GroupThreadID) {
	uint axisGroup = group.x;
	uint lane = thread.x;
	if (axisGroup >= pc.outer_size * pc.inner_size) return;
	uint outer = axisGroup / pc.inner_size;
	uint inner = axisGroup % pc.inner_size;
	uint base = outer * pc.dim_size * pc.inner_size + inner;

	float localMax = -3.402823466e+38f;
	for (uint column = lane; column < pc.dim_size; column += 256u) {
		localMax = max(localMax, OaLoad(
			heap[pc.x_idx], base + column * pc.inner_size));
	}
	reduction[lane] = localMax;
	GroupMemoryBarrierWithGroupSync();
	for (uint width = 128u; width > 0u; width >>= 1u) {
		if (lane < width) reduction[lane] = max(
			reduction[lane], reduction[lane + width]);
		GroupMemoryBarrierWithGroupSync();
	}
	float rowMax = reduction[0];
	// Every lane must finish reading the max before lane 0 can reuse
	// reduction[0] for the sum phase.
	GroupMemoryBarrierWithGroupSync();

	float localSum = 0.0f;
	for (uint column = lane; column < pc.dim_size; column += 256u) {
		localSum += exp(OaLoad(
			heap[pc.x_idx], base + column * pc.inner_size) - rowMax);
	}
	reduction[lane] = localSum;
	GroupMemoryBarrierWithGroupSync();
	for (uint width = 128u; width > 0u; width >>= 1u) {
		if (lane < width) reduction[lane] += reduction[lane + width];
		GroupMemoryBarrierWithGroupSync();
	}
	float logNormalizer = rowMax + log(reduction[0]);
	for (uint column = lane; column < pc.dim_size; column += 256u) {
		uint index = base + column * pc.inner_size;
		OaStore(heap[pc.out_idx], index,
			OaLoad(heap[pc.x_idx], index) - logNormalizer);
	}
}
