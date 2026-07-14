// LinearReluBwdData — fused in-layer LinearRelu(x,W,b) backward, data path.
//
// For y = ReLU(x @ W^T + b), this computes d_x in a single dispatch:
//   d_z[i,j] = d_y[i,j] * (act[i,j] > 0)
//   d_x[i,k] = sum_j d_z[i,j] * W[j,k]
//          = sum_j (d_y[i,j] * (act[i,j] > 0)) * W[j,k]
//
// The gate is applied INSIDE the inner sum (so we never materialize d_z),
// which is the inverse fusion direction of LinearDataReluBwd (that one
// applies the gate AFTER the matmul, for a *cross-layer* boundary case).
//
// Shapes (M=batch, N=out_features, K=in_features):
//   d_y:   [M, N]
//   act:   [M, N]   (post-ReLU forward output of THIS layer)
//   W:     [N, K]
//   d_x:   [M, K]
//
// Keep the saved ReLU activation mask consistent with the autograd node.

import Math;

struct PushConstants {
	uint d_out_idx;
	uint activation_idx;
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

	float sum = 0.0;
	for (uint j = 0; j < pc.N; j++) {
		float d_y  = OaLoad(heap[pc.d_out_idx],      i * pc.N + j);
		float act  = OaLoad(heap[pc.activation_idx], i * pc.N + j);
		float gate = act > 0.0 ? 1.0 : 0.0;
		sum += (d_y * gate) * OaLoad(heap[pc.weight_idx], j * pc.K + k);
	}

	EpilogueStore(heap[pc.d_input_idx], idx, sum);
}
