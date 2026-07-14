// LinearDataReluBwd - fused data gradient and ReLU backward.
// Computes: d_input[i,k] = (sum_j d_output[i,j] * weight[j,k]) * (activation[i,k] > 0)

import Math;

struct PushConstants {
	uint d_out_idx;
	uint weight_idx;
	uint activation_idx;
	uint d_input_idx;
	uint M;
	uint N;
	uint K;
};

[[vk::push_constant]] PushConstants pc;

[[vk::binding(0, 0)]] RWByteAddressBuffer heap[];

[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
	uint idx = tid.x;
	if (idx >= pc.M * pc.K) return;

	uint i = idx / pc.K;
	uint k = idx % pc.K;

	float sum = 0.0;
	for (uint j = 0; j < pc.N; j++) {
		sum += OaLoad(heap[pc.d_out_idx], i * pc.N + j) * OaLoad(heap[pc.weight_idx], j * pc.K + k);
	}

	float activation = OaLoad(heap[pc.activation_idx], idx);
	EpilogueStore(heap[pc.d_input_idx], idx, activation > 0.0 ? sum : 0.0);
}
