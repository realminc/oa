struct PushConstants {
	uint curve_idx, counts_idx, metrics_idx;
	uint predicted, pairs;
};
[[vk::push_constant]] PushConstants pc;
[[vk::binding(0, 0)]] RWByteAddressBuffer heap[];

float loadF32(uint buffer, uint index) {
	return asfloat(heap[buffer].Load(index * 4u));
}

[shader("compute")]
[numthreads(1, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
	uint pair = tid.x;
	if (pair >= pc.pairs) return;
	uint countsBase = pair * 3u;
	uint targetCount = heap[pc.counts_idx].Load((countsBase + 0u) * 4u)
		+ heap[pc.counts_idx].Load((countsBase + 2u) * 4u);
	if (targetCount == 0u) {
		heap[pc.metrics_idx].Store((pair * 4u + 3u) * 4u, 0u);
		return;
	}

	uint curveBase = pair * pc.predicted * 2u;
	float sum = 0.0f;
	for (uint recallPoint = 0; recallPoint <= 100u; ++recallPoint) {
		float requiredRecall = float(recallPoint) * 0.01f;
		float bestPrecision = 0.0f;
		for (uint rank = 0; rank < pc.predicted; ++rank) {
			float precision = loadF32(pc.curve_idx, curveBase + rank * 2u + 0u);
			float recall = loadF32(pc.curve_idx, curveBase + rank * 2u + 1u);
			if (recall + 1.0e-7f >= requiredRecall) {
				bestPrecision = max(bestPrecision, precision);
			}
		}
		sum += bestPrecision;
	}
	heap[pc.metrics_idx].Store(
		(pair * 4u + 3u) * 4u, asuint(sum / 101.0f));
}
