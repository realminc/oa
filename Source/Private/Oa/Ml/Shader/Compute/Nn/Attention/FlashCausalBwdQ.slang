// Deterministic dQ for causal FlashAttention. Probabilities are recomputed from
// Q/K and the forward FP32 log-sum-exp; no SxS activation is retained.

import Storage;

static const uint kThreads = 64;
static const uint kMaxSeqLen = 1024;

struct PushConstants {
	uint q_idx; uint k_idx; uint v_idx; uint out_idx;
	uint d_out_idx; uint lse_idx; uint d_q_idx;
	uint batchHeads; uint seqLen; uint headDim; float scale;
};

[[vk::push_constant]] PushConstants pc;
[[vk::binding(0, 0)]] RWByteAddressBuffer heap[];
groupshared float gDs[kMaxSeqLen];
groupshared float gReduce[kThreads];

[shader("compute")]
[numthreads(kThreads, 1, 1)]
void main(uint3 group : SV_GroupID, uint3 thread : SV_GroupThreadID) {
	uint row = group.x;
	uint lane = thread.x;
	if (row >= pc.batchHeads * pc.seqLen) return;
	uint bh = row / pc.seqLen;
	uint query = row - bh * pc.seqLen;
	uint qBase = row * pc.headDim;
	uint kvHeadBase = bh * pc.seqLen * pc.headDim;

	float localDelta = 0.0f;
	for (uint d = lane; d < pc.headDim; d += kThreads) {
		localDelta += OaLoad(heap[pc.d_out_idx], qBase + d) * OaLoad(heap[pc.out_idx], qBase + d);
	}
	gReduce[lane] = localDelta;
	GroupMemoryBarrierWithGroupSync();
	for (uint stride = kThreads / 2; stride > 0; stride >>= 1) {
		if (lane < stride) gReduce[lane] += gReduce[lane + stride];
		GroupMemoryBarrierWithGroupSync();
	}
	float delta = gReduce[0];
	float lse = OaLoadF32(heap[pc.lse_idx], row);
	for (uint key = lane; key <= query; key += kThreads) {
		uint kvBase = kvHeadBase + key * pc.headDim;
		float score = 0.0f;
		float dProb = 0.0f;
		for (uint d = 0; d < pc.headDim; ++d) {
			score += OaLoad(heap[pc.q_idx], qBase + d) * OaLoad(heap[pc.k_idx], kvBase + d);
			dProb += OaLoad(heap[pc.d_out_idx], qBase + d) * OaLoad(heap[pc.v_idx], kvBase + d);
		}
		float probability = exp(score * pc.scale - lse);
		gDs[key] = probability * (dProb - delta) * pc.scale;
	}
	GroupMemoryBarrierWithGroupSync();
	for (uint d = lane; d < pc.headDim; d += kThreads) {
		float acc = 0.0f;
		for (uint key = 0; key <= query; ++key) {
			acc += gDs[key] * OaLoad(heap[pc.k_idx], kvHeadBase + key * pc.headDim + d);
		}
		OaStore(heap[pc.d_q_idx], qBase + d, acc);
	}
}
