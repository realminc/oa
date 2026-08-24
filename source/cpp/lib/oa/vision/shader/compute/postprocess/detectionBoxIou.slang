struct PushConstants {
	uint boxes_a_idx, boxes_b_idx, out_idx;
	uint rows_a, rows_b;
};
[[vk::push_constant]] PushConstants pc;
[[vk::binding(0, 0)]] RWByteAddressBuffer heap[];

float loadF32(uint buffer, uint index) {
	return asfloat(heap[buffer].Load(index * 4u));
}

float iou(uint rowA, uint rowB) {
	uint a = rowA * 4u;
	uint b = rowB * 4u;
	float acx = loadF32(pc.boxes_a_idx, a + 0u);
	float acy = loadF32(pc.boxes_a_idx, a + 1u);
	float aw = max(loadF32(pc.boxes_a_idx, a + 2u), 0.0f);
	float ah = max(loadF32(pc.boxes_a_idx, a + 3u), 0.0f);
	float bcx = loadF32(pc.boxes_b_idx, b + 0u);
	float bcy = loadF32(pc.boxes_b_idx, b + 1u);
	float bw = max(loadF32(pc.boxes_b_idx, b + 2u), 0.0f);
	float bh = max(loadF32(pc.boxes_b_idx, b + 3u), 0.0f);
	float ax0 = acx - aw * 0.5f;
	float ay0 = acy - ah * 0.5f;
	float ax1 = acx + aw * 0.5f;
	float ay1 = acy + ah * 0.5f;
	float bx0 = bcx - bw * 0.5f;
	float by0 = bcy - bh * 0.5f;
	float bx1 = bcx + bw * 0.5f;
	float by1 = bcy + bh * 0.5f;
	float iw = max(min(ax1, bx1) - max(ax0, bx0), 0.0f);
	float ih = max(min(ay1, by1) - max(ay0, by0), 0.0f);
	float intersection = iw * ih;
	float unionArea = aw * ah + bw * bh - intersection;
	return unionArea > 0.0f ? intersection / unionArea : 0.0f;
}

[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
	uint pair = tid.x;
	uint pairCount = pc.rows_a * pc.rows_b;
	if (pair >= pairCount) return;
	uint rowA = pair / pc.rows_b;
	uint rowB = pair - rowA * pc.rows_b;
	heap[pc.out_idx].Store(pair * 4u, asuint(iou(rowA, rowB)));
}
