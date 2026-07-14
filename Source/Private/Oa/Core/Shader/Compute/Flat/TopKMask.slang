// Exact dense membership mask from TopK's deterministic Int32 indices.
// Input [T,K], output [T,E]. Each row has exactly K ones when indices are unique.

import Storage;

struct PushConstants {
	uint indices_idx;
	uint mask_idx;
	uint T;
	uint E;
	uint K;
};

[[vk::push_constant]] PushConstants pc;
[[vk::binding(0, 0)]] RWByteAddressBuffer heap[];

[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
	uint linear = tid.x;
	uint count = pc.T * pc.E;
	if (linear >= count) return;
	uint row = linear / pc.E;
	uint expert = linear - row * pc.E;
	float selected = 0.0f;
	for (uint k = 0; k < pc.K; ++k) {
		uint index = heap[pc.indices_idx].Load((row * pc.K + k) * 4);
		if (index == expert) { selected = 1.0f; break; }
	}
	OaStore(heap[pc.mask_idx], linear, selected);
}
