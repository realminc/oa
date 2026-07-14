// CrossEntropyLossGradBwd: fused per-sample CE loss and gradient w.r.t. logits.
// Output loss is per-sample; C++ records a Mean reduction for the scalar loss.

import Storage;

struct PushConstants {
	uint logits_idx;
	uint targets_idx;
	uint loss_idx;
	uint d_logits_idx;
	uint batch;
	uint classes;
	uint target_dtype;  // 0=UInt8, 1=UInt32
};

[[vk::push_constant]] PushConstants pc;
[[vk::binding(0, 0)]] RWByteAddressBuffer heap[];

groupshared float g_buf[256];

[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 gid : SV_GroupID, uint3 tid : SV_GroupThreadID) {
	uint row = gid.x;
	if (row >= pc.batch) return;

	uint lane = tid.x;
	uint base = row * pc.classes;

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

	if (lane == 0) {
		float log_sum_exp = log(row_sum) + row_max;
		OaStore(heap[pc.loss_idx], row,
			log_sum_exp - OaLoad(heap[pc.logits_idx], base + target));
	}

	float inv_batch = 1.0 / float(pc.batch);
	for (uint i = lane; i < pc.classes; i += 256) {
		float prob = exp(OaLoad(heap[pc.logits_idx], base + i) - row_max) * inv_sum;
		float grad = prob;
		if (i == target) grad -= 1.0;
		OaStore(heap[pc.d_logits_idx], base + i, grad * inv_batch);
	}
}
