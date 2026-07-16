// Deterministic dK/dV for causal FlashAttention. One workgroup owns one key row,
// so every output element has a single writer and no atomic accumulation.

import Storage;

static const uint kThreads = 64;
static const uint kMaxSeqLen = 1024;

struct PushConstants {
	uint q_idx; uint k_idx; uint v_idx; uint out_idx;
	uint d_out_idx; uint lse_idx; uint d_k_idx; uint d_v_idx;
	uint batchHeads; uint seqLen; uint headDim; float scale;
};

[[vk::push_constant]] PushConstants pc;
[[vk::binding(0, 0)]] RWByteAddressBuffer heap[];
groupshared float gProbability[kMaxSeqLen];
groupshared float gDs[kMaxSeqLen];

[shader("compute")]
[numthreads(kThreads, 1, 1)]
void main(uint3 group : SV_GroupID, uint3 thread : SV_GroupThreadID) {
	uint row = group.x;
	uint lane = thread.x;
	if (row >= pc.batchHeads * pc.seqLen) return;
	uint bh = row / pc.seqLen;
	uint key = row - bh * pc.seqLen;
	uint kvBase = row * pc.headDim;
	uint headBase = bh * pc.seqLen * pc.headDim;

	for (uint query = key + lane; query < pc.seqLen; query += kThreads) {
		uint qBase = headBase + query * pc.headDim;
		float score = 0.0f;
		float dProb = 0.0f;
		float delta = 0.0f;
		for (uint d = 0; d < pc.headDim; ++d) {
			float dOut = OaLoad(heap[pc.d_out_idx], qBase + d);
			score += OaLoad(heap[pc.q_idx], qBase + d) * OaLoad(heap[pc.k_idx], kvBase + d);
			dProb += dOut * OaLoad(heap[pc.v_idx], kvBase + d);
			delta += dOut * OaLoad(heap[pc.out_idx], qBase + d);
		}
		float probability = exp(score * pc.scale - OaLoadF32(heap[pc.lse_idx], bh * pc.seqLen + query));
		gProbability[query] = probability;
		gDs[query] = probability * (dProb - delta) * pc.scale;
	}
	GroupMemoryBarrierWithGroupSync();
	for (uint d = lane; d < pc.headDim; d += kThreads) {
		float dK = 0.0f;
		float dV = 0.0f;
		for (uint query = key; query < pc.seqLen; ++query) {
			uint qIndex = headBase + query * pc.headDim + d;
			dK += gDs[query] * OaLoad(heap[pc.q_idx], qIndex);
			dV += gProbability[query] * OaLoad(heap[pc.d_out_idx], qIndex);
		}
		OaStore(heap[pc.d_k_idx], kvBase + d, dK);
		OaStore(heap[pc.d_v_idx], kvBase + d, dV);
	}
}
