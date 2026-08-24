// AudioReverbSum — normalized sum of four parallel Schroeder combs.

import storage;

struct PushConstants {
	uint a_idx;
	uint b_idx;
	uint c_idx;
	uint d_idx;
	uint out_idx;
	uint count;
	float normalization;
};

[[vk::push_constant]] PushConstants pc;
[[vk::binding(0, 0)]] RWByteAddressBuffer heap[];

[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
	const uint index = tid.x;
	if (index >= pc.count) return;
	const float sum = oaLoad(heap[pc.a_idx], index)
		+ oaLoad(heap[pc.b_idx], index)
		+ oaLoad(heap[pc.c_idx], index)
		+ oaLoad(heap[pc.d_idx], index);
	oaStore(heap[pc.out_idx], index, sum * pc.normalization);
}
