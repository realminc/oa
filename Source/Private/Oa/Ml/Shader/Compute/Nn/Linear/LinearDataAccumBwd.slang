// LinearDataAccumBwd — Backward pass for linear layer with gradient accumulation
// Computes: d_input += d_output @ weight
// Uses Math.Epilogue for accumulation pattern

import Math;

struct PushConstants {
	uint d_out_idx;
	uint weight_idx;
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

	// Compute: d_input[i,k] += sum_j d_output[i,j] * weight[j,k]
	float sum = 0.0;
	for (uint j = 0; j < pc.N; j++) {
		sum += OaLoad(heap[pc.d_out_idx], i * pc.N + j) * OaLoad(heap[pc.weight_idx], j * pc.K + k);
	}
	
	// Use Math.Epilogue for accumulation (instead of manual load+add+store)
	EpilogueAccumulate(heap[pc.d_input_idx], idx, sum);
}