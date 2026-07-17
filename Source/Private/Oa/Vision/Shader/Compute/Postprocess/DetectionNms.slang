struct PushConstants {
	uint boxes_idx, scores_idx, classes_idx, indices_idx, count_idx;
	uint count, maximum, class_agnostic;
	float iou_threshold, score_threshold;
};
[[vk::push_constant]] PushConstants pc;
[[vk::binding(0, 0)]] RWByteAddressBuffer heap[];

float loadF32(uint buffer, uint index) {
	return asfloat(heap[buffer].Load(index * 4u));
}

float iou(uint rowA, uint rowB) {
	uint a = rowA * 4u;
	uint b = rowB * 4u;
	float acx = loadF32(pc.boxes_idx, a + 0u);
	float acy = loadF32(pc.boxes_idx, a + 1u);
	float aw = max(loadF32(pc.boxes_idx, a + 2u), 0.0f);
	float ah = max(loadF32(pc.boxes_idx, a + 3u), 0.0f);
	float bcx = loadF32(pc.boxes_idx, b + 0u);
	float bcy = loadF32(pc.boxes_idx, b + 1u);
	float bw = max(loadF32(pc.boxes_idx, b + 2u), 0.0f);
	float bh = max(loadF32(pc.boxes_idx, b + 3u), 0.0f);
	float intersection = max(min(acx + aw * 0.5f, bcx + bw * 0.5f)
		- max(acx - aw * 0.5f, bcx - bw * 0.5f), 0.0f)
		* max(min(acy + ah * 0.5f, bcy + bh * 0.5f)
		- max(acy - ah * 0.5f, bcy - bh * 0.5f), 0.0f);
	float unionArea = aw * ah + bw * bh - intersection;
	return unionArea > 0.0f ? intersection / unionArea : 0.0f;
}

[shader("compute")]
[numthreads(1, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
	for (uint i = 0; i < pc.maximum; ++i) {
		heap[pc.indices_idx].Store(i * 4u, 0xffffffffu);
	}
	uint selectedCount = 0u;
	for (uint output = 0; output < pc.maximum; ++output) {
		uint best = 0xffffffffu;
		float bestScore = -3.402823466e+38f;
		for (uint candidate = 0; candidate < pc.count; ++candidate) {
			float score = loadF32(pc.scores_idx, candidate);
			if (score != score || score < pc.score_threshold) continue;
			bool suppressed = false;
			for (uint prior = 0; prior < selectedCount; ++prior) {
				uint selected = heap[pc.indices_idx].Load(prior * 4u);
				if (candidate == selected) { suppressed = true; break; }
				bool sameClass = pc.class_agnostic != 0u
					|| heap[pc.classes_idx].Load(candidate * 4u)
						== heap[pc.classes_idx].Load(selected * 4u);
				if (sameClass && iou(candidate, selected) >= pc.iou_threshold) {
					suppressed = true;
					break;
				}
			}
			if (!suppressed && (score > bestScore
				|| (score == bestScore && candidate < best))) {
				best = candidate;
				bestScore = score;
			}
		}
		if (best == 0xffffffffu) break;
		heap[pc.indices_idx].Store(selectedCount * 4u, best);
		selectedCount++;
	}
	heap[pc.count_idx].Store(0, selectedCount);
}
