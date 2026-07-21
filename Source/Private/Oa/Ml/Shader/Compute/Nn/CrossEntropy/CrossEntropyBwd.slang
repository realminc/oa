// CrossEntropy backward: compute gradient w.r.t. logits
// Bindless: buffer indices (logits, targets, d_logits) prepended by OaVkDispatch::Run
// Output: (softmax - onehot) / batch (correct gradient for mean cross-entropy)

import Storage;

struct PushConstants {
	uint logits_idx;    // Prepended by OaVkDispatch::Run
	uint targets_idx;   // Prepended by OaVkDispatch::Run
	uint d_logits_idx;  // Prepended by OaVkDispatch::Run
	uint batch;         // From C++ code
	uint classes;       // From C++ code
	uint target_dtype;  // 0=UInt8, 1=UInt32 or non-negative Int32
};

[[vk::push_constant]] PushConstants pc;

[[vk::binding(0, 0)]] RWByteAddressBuffer heap[];

groupshared float g_buf[256];
static const uint kPortableDispatchTileWidthX = 65535;

[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 gid : SV_GroupID, uint3 tid : SV_GroupThreadID) {
	uint row = gid.x + gid.y * kPortableDispatchTileWidthX;
	if (row >= pc.batch) return;

	uint lane = tid.x;
	uint base = row * pc.classes;

	// RowMax (inlined)
	float localMax = -3.402823466e+38f;
	for (uint i = lane; i < pc.classes; i += 256) {
		localMax = max(localMax, OaLoad(heap[pc.logits_idx], base + i));
	}
	g_buf[lane] = localMax;
	GroupMemoryBarrierWithGroupSync();
	for (uint s = 128; s > 0; s >>= 1) {
		if (lane < s) g_buf[lane] = max(g_buf[lane], g_buf[lane + s]);
		GroupMemoryBarrierWithGroupSync();
	}
	float row_max = g_buf[0];
	GroupMemoryBarrierWithGroupSync();

	// RowSumExp (inlined)
	float localSum = 0.0;
	for (uint i = lane; i < pc.classes; i += 256) {
		localSum += exp(OaLoad(heap[pc.logits_idx], base + i) - row_max);
	}
	g_buf[lane] = localSum;
	GroupMemoryBarrierWithGroupSync();
	for (uint s = 128; s > 0; s >>= 1) {
		if (lane < s) g_buf[lane] += g_buf[lane + s];
		GroupMemoryBarrierWithGroupSync();
	}
	float row_sum = g_buf[0];
	float inv_sum = 1.0 / row_sum;

	uint target = (pc.target_dtype == 0)
		? (uint)heap[pc.targets_idx].Load<uint8_t>(row)
		: heap[pc.targets_idx].Load<uint>(row * 4);
	if (target >= pc.classes) {
		for (uint i = lane; i < pc.classes; i += 256) {
			OaStore(heap[pc.d_logits_idx], base + i, asfloat(0x7FC00000u));
		}
		return;
	}
	float inv_batch = 1.0 / float(pc.batch);
	
	for (uint i = lane; i < pc.classes; i += 256) {
		float prob = exp(OaLoad(heap[pc.logits_idx], base + i) - row_max) * inv_sum;
		float grad = prob;
		if (i == target) grad -= 1.0;
		// Divide by batch for mean cross-entropy loss
		grad *= inv_batch;
		OaStore(heap[pc.d_logits_idx], base + i, grad);
	}
}
