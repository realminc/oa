// Backward pass for SoftmaxScaledMasked.
// Computes d_scores = softmax_out * (d_out - sum(d_out * softmax_out)) * scale
// The mask is constant, so it contributes nothing to the gradient w.r.t. scores.

import Storage;

struct PushConstants {
	uint softmax_out_idx;
	uint d_out_idx;
	uint d_scores_idx;
	uint rows;
	uint cols;
	float scale;
};

[[vk::push_constant]] PushConstants pc;

[[vk::binding(0, 0)]] RWByteAddressBuffer heap[];

groupshared float g_buf[256];

[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 gid : SV_GroupID, uint3 tid : SV_GroupThreadID) {
	uint row = gid.x;
	if (row >= pc.rows) return;

	uint lane = tid.x;
	uint base = row * pc.cols;

	float localDot = 0.0;
	for (uint i = lane; i < pc.cols; i += 256) {
		localDot += OaLoad(heap[pc.d_out_idx], base + i) * OaLoad(heap[pc.softmax_out_idx], base + i);
	}

	g_buf[lane] = localDot;
	GroupMemoryBarrierWithGroupSync();
	for (uint s = 128; s > 0; s >>= 1) {
		if (lane < s) g_buf[lane] += g_buf[lane + s];
		GroupMemoryBarrierWithGroupSync();
	}
	float dot = g_buf[0];

	GroupMemoryBarrierWithGroupSync();

	for (uint i = lane; i < pc.cols; i += 256) {
		float s = OaLoad(heap[pc.softmax_out_idx], base + i);
		float d = OaLoad(heap[pc.d_out_idx], base + i);
		OaStore(heap[pc.d_scores_idx], base + i, s * (d - dot) * pc.scale);
	}
}
