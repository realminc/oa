// Argmax over a flat array. Output is a single uint index.

import Storage;

struct PushConstants {
	uint x_idx;
	uint out_idx;
	uint count;
};

[[vk::push_constant]] PushConstants pc;

[[vk::binding(0, 0)]] RWByteAddressBuffer heap[];

groupshared float sval[256];
groupshared uint sidx[256];

[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 tid : SV_GroupThreadID) {
	uint lid = tid.x;

	float bestVal = -3.402823466e+38f;
	uint bestIdx = 0;
	for (uint i = lid; i < pc.count; i += 256) {
		float v = OaLoad(heap[pc.x_idx], i);
		if (v > bestVal) {
			bestVal = v;
			bestIdx = i;
		}
	}
	sval[lid] = bestVal;
	sidx[lid] = bestIdx;
	GroupMemoryBarrierWithGroupSync();

	for (uint s = 128; s > 0; s >>= 1) {
		if (lid < s && sval[lid + s] > sval[lid]) {
			sval[lid] = sval[lid + s];
			sidx[lid] = sidx[lid + s];
		}
		GroupMemoryBarrierWithGroupSync();
	}

	if (lid == 0) {
		heap[pc.out_idx].Store(0, sidx[0]);
	}
}
