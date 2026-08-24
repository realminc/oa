struct PushConstants {
	uint counts_idx, metrics_idx, map_idx, mean_idx;
	uint thresholds, classes;
};
[[vk::push_constant]] PushConstants pc;
[[vk::binding(0, 0)]] RWByteAddressBuffer heap[];

float loadF32(uint buffer, uint index) {
	return asfloat(heap[buffer].Load(index * 4u));
}

[shader("compute")]
[numthreads(1, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
	float total = 0.0f;
	uint validThresholds = 0u;
	for (uint threshold = 0; threshold < pc.thresholds; ++threshold) {
		float classTotal = 0.0f;
		uint validClasses = 0u;
		for (uint classIndex = 0; classIndex < pc.classes; ++classIndex) {
			uint pair = threshold * pc.classes + classIndex;
			uint countsBase = pair * 3u;
			uint targetCount = heap[pc.counts_idx].Load((countsBase + 0u) * 4u)
				+ heap[pc.counts_idx].Load((countsBase + 2u) * 4u);
			if (targetCount == 0u) continue;
			classTotal += loadF32(pc.metrics_idx, pair * 4u + 3u);
			validClasses++;
		}
		float map = validClasses > 0u ? classTotal / float(validClasses) : 0.0f;
		heap[pc.map_idx].Store(threshold * 4u, asuint(map));
		if (validClasses > 0u) {
			total += map;
			validThresholds++;
		}
	}
	float mean = validThresholds > 0u ? total / float(validThresholds) : 0.0f;
	heap[pc.mean_idx].Store(0u, asuint(mean));
}
