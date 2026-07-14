// Fused transformer attention score normalisation:
//   out = softmax(scores * scale + mask)
// Combines the Scale → Add(mask) → Softmax chain into a single row-wise
// reduction kernel. The mask is added before softmax but is treated as
// constant during the backward pass (no gradient flows into it).

import Storage;

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

groupshared float sdata[256];

[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 gid : SV_GroupID, uint3 tid : SV_GroupThreadID) {
	uint row = gid.x;
	uint lid = tid.x;
	if (row >= pc.rows) return;

	uint base = row * pc.cols;

	float localMax = -3.402823466e+38f;
	for (uint i = lid; i < pc.cols; i += 256) {
		float val = OaLoad(heap[pc.scores_idx], base + i) * pc.scale;
		val += OaLoadF32(heap[pc.mask_idx], base + i);
		localMax = max(localMax, val);
	}
	sdata[lid] = localMax;
	GroupMemoryBarrierWithGroupSync();
	for (uint s = 128; s > 0; s >>= 1) {
		if (lid < s) sdata[lid] = max(sdata[lid], sdata[lid + s]);
		GroupMemoryBarrierWithGroupSync();
	}
	float rowMax = sdata[0];
	GroupMemoryBarrierWithGroupSync();

	float localSum = 0.0;
	for (uint i = lid; i < pc.cols; i += 256) {
		float val = OaLoad(heap[pc.scores_idx], base + i) * pc.scale;
		val += OaLoadF32(heap[pc.mask_idx], base + i);
		localSum += exp(val - rowMax);
	}
	sdata[lid] = localSum;
	GroupMemoryBarrierWithGroupSync();
	for (uint s = 128; s > 0; s >>= 1) {
		if (lid < s) sdata[lid] += sdata[lid + s];
		GroupMemoryBarrierWithGroupSync();
	}
	float invSum = 1.0 / sdata[0];
	GroupMemoryBarrierWithGroupSync();

	for (uint i = lid; i < pc.cols; i += 256) {
		float val = OaLoad(heap[pc.scores_idx], base + i) * pc.scale;
		val += OaLoadF32(heap[pc.mask_idx], base + i);
		OaStore(heap[pc.out_idx], base + i, exp(val - rowMax) * invSum);
	}
}
