// Cross-entropy per row with a floating validity mask. Invalid rows write 0.
import storage;

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
static const uint kPortableDispatchTileWidthX = 65535;

[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 gid : SV_GroupID, uint3 tid : SV_GroupThreadID) {
	uint row = gid.x + gid.y * kPortableDispatchTileWidthX;
	if (row >= pc.rows) return;
	uint lane = tid.x;
	if (oaLoad(heap[pc.mask_idx], row) == 0.0) {
		if (lane == 0) oaStoreF32(heap[pc.loss_idx], row, 0.0);
		return;
	}
	uint base = row * pc.classes;
	float local_max = -3.402823466e+38f;
	for (uint i = lane; i < pc.classes; i += 256)
		local_max = max(local_max, oaLoad(heap[pc.logits_idx], base + i));
	g_buf[lane] = local_max;
	GroupMemoryBarrierWithGroupSync();
	for (uint s = 128; s > 0; s >>= 1) {
		if (lane < s) g_buf[lane] = max(g_buf[lane], g_buf[lane + s]);
		GroupMemoryBarrierWithGroupSync();
	}
	float row_max = g_buf[0];
	float local_sum = 0.0;
	for (uint i = lane; i < pc.classes; i += 256)
		local_sum += exp(oaLoad(heap[pc.logits_idx], base + i) - row_max);
	GroupMemoryBarrierWithGroupSync();
	g_buf[lane] = local_sum;
	GroupMemoryBarrierWithGroupSync();
	for (uint s = 128; s > 0; s >>= 1) {
		if (lane < s) g_buf[lane] += g_buf[lane + s];
		GroupMemoryBarrierWithGroupSync();
	}
	if (lane == 0) {
		uint target = (pc.target_dtype == 0)
			? oaLoadU8(heap[pc.targets_idx], row)
			: heap[pc.targets_idx].Load<uint>(row * 4);
		if (target >= pc.classes) {
			oaStoreF32(heap[pc.loss_idx], row, asfloat(0x7FC00000u));
			return;
		}
		oaStoreF32(heap[pc.loss_idx], row,
			log(g_buf[0]) + row_max - oaLoad(heap[pc.logits_idx], base + target));
	}
}
