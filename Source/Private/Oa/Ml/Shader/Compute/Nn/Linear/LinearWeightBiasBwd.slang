// LinearWeightBiasBwd - fused weight and bias gradient for linear layers.
// Computes:
//   d_weight[j,k] = sum_i d_output[i,j] * input[i,k]
//   d_bias[j]     = sum_i d_output[i,j]

import Math;

struct PushConstants {
	uint d_out_idx;
	uint input_idx;
	uint d_weight_idx;
	uint d_bias_idx;
	uint M;
	uint N;
	uint K;
	uint Total;
};

[[vk::push_constant]] PushConstants pc;

[[vk::binding(0, 0)]] RWByteAddressBuffer heap[];

[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
	uint idx = tid.x;
	if (idx >= pc.Total) return;

	uint weightCount = pc.N * pc.K;
	if (idx < weightCount) {
		uint j = idx / pc.K;
		uint k = idx % pc.K;

		float sum = 0.0;
		for (uint i = 0; i < pc.M; i++) {
			sum += OaLoad(heap[pc.d_out_idx], i * pc.N + j) * OaLoad(heap[pc.input_idx], i * pc.K + k);
		}
		EpilogueStore(heap[pc.d_weight_idx], idx, sum);
		return;
	}

	uint jBias = idx - weightCount;
	if (jBias >= pc.N) return;

	float biasSum = 0.0;
	for (uint i = 0; i < pc.M; i++) {
		biasSum += OaLoad(heap[pc.d_out_idx], i * pc.N + jBias);
	}
	EpilogueStore(heap[pc.d_bias_idx], jBias, biasSum);
}
