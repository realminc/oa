// Normalize only the selected router probabilities.
// Probs [T,E] + exact TopK indices [T,K] -> selected weights [T,K].
// This replaces the sparse path's dense mask/mul/sum/div/gather chain.
import Storage;

struct PushConstants {
	uint probs_idx;
	uint indices_idx;
	uint out_idx;
	uint T;
	uint E;
	uint K;
};

[[vk::push_constant]] PushConstants pc;
[[vk::binding(0, 0)]] RWByteAddressBuffer heap[];

[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
	uint t = tid.x;
	if (t >= pc.T) return;

	float denom = 0.0;
	for (uint k = 0; k < pc.K; ++k) {
		uint route = t * pc.K + k;
		int expert = asint(heap[pc.indices_idx].Load(route * 4));
		if (expert >= 0 && uint(expert) < pc.E)
			denom += OaLoad(heap[pc.probs_idx], t * pc.E + uint(expert));
	}

	float invDenom = denom > 0.0 ? 1.0 / denom : 0.0;
	for (uint k = 0; k < pc.K; ++k) {
		uint route = t * pc.K + k;
		int expert = asint(heap[pc.indices_idx].Load(route * 4));
		float value = 0.0;
		if (expert >= 0 && uint(expert) < pc.E)
			value = OaLoad(heap[pc.probs_idx], t * pc.E + uint(expert)) * invDenom;
		OaStore(heap[pc.out_idx], route, value);
	}
}
