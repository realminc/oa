// Cross-entropy per row with a floating validity mask. Invalid rows write 0.
import Storage;

struct PushConstants {
	uint logits_idx;
	uint targets_idx;
	uint mask_idx;
	uint loss_idx;
	uint rows;
	uint classes;
	uint target_dtype;
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
	if (OaLoad(heap[pc.mask_idx], row) == 0.0) {
		if (lane == 0) OaStoreF32(heap[pc.loss_idx], row, 0.0);
		return;
	}
	uint base = row * pc.classes;
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
	if (lane == 0) {
		uint target = (pc.target_dtype == 0)
			? (uint)heap[pc.targets_idx].Load<uint8_t>(row)
			: heap[pc.targets_idx].Load<uint>(row * 4);
		OaStoreF32(heap[pc.loss_idx], row,
			log(g_buf[0]) + row_max - OaLoad(heap[pc.logits_idx], base + target));
	}
}
