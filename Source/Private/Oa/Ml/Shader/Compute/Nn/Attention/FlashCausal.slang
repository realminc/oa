// IO-aware causal scaled dot-product attention.
// One workgroup owns one query row. Scores/probabilities live only in
// workgroup memory; global storage contains O and one FP32 log-sum-exp per row.

import Storage;

static const uint kThreads = 64;
static const uint kMaxSeqLen = 1024;

struct PushConstants {
	uint q_idx;
	uint k_idx;
	uint v_idx;
	uint out_idx;
	uint lse_idx;
	uint batchHeads;
	uint seqLen;
	uint headDim;
	float scale;
};

[[vk::push_constant]] PushConstants pc;
[[vk::binding(0, 0)]] RWByteAddressBuffer heap[];

groupshared float gWeight[kMaxSeqLen];
groupshared float gReduce[kThreads];

[shader("compute")]
[numthreads(kThreads, 1, 1)]
void main(uint3 group : SV_GroupID, uint3 thread : SV_GroupThreadID) {
	uint row = group.x;
	uint lane = thread.x;
	uint totalRows = pc.batchHeads * pc.seqLen;
	if (row >= totalRows) return;
	uint bh = row / pc.seqLen;
	uint query = row - bh * pc.seqLen;
	uint qBase = row * pc.headDim;
	uint kvHeadBase = bh * pc.seqLen * pc.headDim;

	float localMax = -3.402823466e+38f;
	for (uint key = lane; key <= query; key += kThreads) {
		uint kBase = kvHeadBase + key * pc.headDim;
		float score = 0.0f;
		for (uint d = 0; d < pc.headDim; ++d) {
			score += OaLoad(heap[pc.q_idx], qBase + d) * OaLoad(heap[pc.k_idx], kBase + d);
		}
		score *= pc.scale;
		gWeight[key] = score;
		localMax = max(localMax, score);
	}
	gReduce[lane] = localMax;
	GroupMemoryBarrierWithGroupSync();
	for (uint stride = kThreads / 2; stride > 0; stride >>= 1) {
		if (lane < stride) gReduce[lane] = max(gReduce[lane], gReduce[lane + stride]);
		GroupMemoryBarrierWithGroupSync();
	}
	float rowMax = gReduce[0];
	float localSum = 0.0f;
	for (uint key = lane; key <= query; key += kThreads) {
		float weight = exp(gWeight[key] - rowMax);
		gWeight[key] = weight;
		localSum += weight;
	}
	gReduce[lane] = localSum;
	GroupMemoryBarrierWithGroupSync();
	for (uint stride = kThreads / 2; stride > 0; stride >>= 1) {
		if (lane < stride) gReduce[lane] += gReduce[lane + stride];
		GroupMemoryBarrierWithGroupSync();
	}
	float sum = gReduce[0];
	if (lane == 0) OaStoreF32(heap[pc.lse_idx], row, rowMax + log(sum));
	GroupMemoryBarrierWithGroupSync();

	for (uint d = lane; d < pc.headDim; d += kThreads) {
		float acc = 0.0f;
		for (uint key = 0; key <= query; ++key) {
			uint vIndex = kvHeadBase + key * pc.headDim + d;
			acc += (gWeight[key] / sum) * OaLoad(heap[pc.v_idx], vIndex);
		}
		OaStore(heap[pc.out_idx], qBase + d, acc);
	}
}
