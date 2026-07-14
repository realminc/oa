// CrossEntropy forward: compute per-sample cross-entropy loss
// Bindless: buffer indices (logits, targets, loss) prepended by OaVkDispatch::Run

import Storage;

struct PushConstants {
	uint logits_idx;    // Prepended by OaVkDispatch::Run
	uint targets_idx;   // Prepended by OaVkDispatch::Run
	uint loss_idx;      // Prepended by OaVkDispatch::Run
	uint batch;         // From C++ code
	uint classes;       // From C++ code
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

	float log_sum_exp = log(row_sum) + row_max;

	if (lane == 0) {
		uint target = (pc.target_dtype == 0)
			? (uint)heap[pc.targets_idx].Load<uint8_t>(row)
			: heap[pc.targets_idx].Load<uint>(row * 4);
		// Per-sample loss is a dtype-invariant FP32 output (perSample is always Float32,
		// see FnLoss::CrossEntropy) so the Sum/Mean reduction accumulates in fp32 without
		// rounding each ~ln(V) row loss into bf16. Logits are still loaded via branched
		// OaLoad (2-byte in BF16 mode); only the loss write is forced fp32.
		OaStoreF32(heap[pc.loss_idx], row, log_sum_exp - OaLoad(heap[pc.logits_idx], base + target));
	}
}
