// Backward for masked mean cross-entropy. Invalid rows write exact zero.
import Storage;

struct PushConstants {
	uint logits_idx;
	uint targets_idx;
	uint mask_idx;
	uint d_logits_idx;
	uint rows;
	uint classes;
	uint target_dtype;
	uint valid_count;
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
	uint base = row * pc.classes;
	if (OaLoad(heap[pc.mask_idx], row) == 0.0) {
		for (uint i = lane; i < pc.classes; i += 256)
			OaStore(heap[pc.d_logits_idx], base + i, 0.0);
		return;
	}
	float local_max = -3.402823466e+38f;
	for (uint i = lane; i < pc.classes; i += 256)
		local_max = max(local_max, OaLoad(heap[pc.logits_idx], base + i));
	g_buf[lane] = local_max;
	GroupMemoryBarrierWithGroupSync();
	for (uint s = 128; s > 0; s >>= 1) {
		if (lane < s) g_buf[lane] = max(g_buf[lane], g_buf[lane + s]);
		GroupMemoryBarrierWithGroupSync();
	}
	float row_max = g_buf[0];
	float local_sum = 0.0;
	for (uint i = lane; i < pc.classes; i += 256)
		local_sum += exp(OaLoad(heap[pc.logits_idx], base + i) - row_max);
	g_buf[lane] = local_sum;
	GroupMemoryBarrierWithGroupSync();
	for (uint s = 128; s > 0; s >>= 1) {
		if (lane < s) g_buf[lane] += g_buf[lane + s];
		GroupMemoryBarrierWithGroupSync();
	}
	uint target = (pc.target_dtype == 0)
		? (uint)heap[pc.targets_idx].Load<uint8_t>(row)
		: heap[pc.targets_idx].Load<uint>(row * 4);
	float inv_sum = 1.0 / g_buf[0];
	float inv_valid = 1.0 / float(pc.valid_count);
	for (uint i = lane; i < pc.classes; i += 256) {
		float grad = exp(OaLoad(heap[pc.logits_idx], base + i) - row_max) * inv_sum;
		if (i == target) grad -= 1.0;
		OaStore(heap[pc.d_logits_idx], base + i, grad * inv_valid);
	}
}
