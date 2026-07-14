// Exact gradient of selected probability normalization.
// Every [T,E] output element has one writer; unselected experts receive zero.
import Storage;

struct PushConstants {
	uint gradOut_idx;
	uint probs_idx;
	uint indices_idx;
	uint routeWeights_idx;
	uint gradProbs_idx;
	uint T;
	uint E;
	uint K;
};

[[vk::push_constant]] PushConstants pc;
[[vk::binding(0, 0)]] RWByteAddressBuffer heap[];

[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
	uint idx = tid.x;
	if (idx >= pc.T * pc.E) return;
	uint t = idx / pc.E;
	uint e = idx % pc.E;

	float denom = 0.0;
	float dot = 0.0;
	int selectedSlot = -1;
	for (uint k = 0; k < pc.K; ++k) {
		uint route = t * pc.K + k;
		int expert = asint(heap[pc.indices_idx].Load(route * 4));
		if (expert >= 0 && uint(expert) < pc.E) {
			denom += OaLoad(heap[pc.probs_idx], t * pc.E + uint(expert));
			dot += OaLoad(heap[pc.gradOut_idx], route) *
				OaLoad(heap[pc.routeWeights_idx], route);
			if (uint(expert) == e) selectedSlot = int(k);
		}
	}

	float grad = 0.0;
	if (selectedSlot >= 0 && denom > 0.0) {
		uint route = t * pc.K + uint(selectedSlot);
		grad = (OaLoad(heap[pc.gradOut_idx], route) - dot) / denom;
	}
	OaStore(heap[pc.gradProbs_idx], idx, grad);
}
