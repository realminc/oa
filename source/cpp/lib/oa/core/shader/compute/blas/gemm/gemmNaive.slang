// C[i,j] = sum_k A[i,k] * B[j,k]  (B is transposed: C = A @ B^T)
// One thread per output element — correct but slow. Good for verification.

import storage;

struct FullPushConstants {
	uint a_idx;
	uint b_idx;
	uint out_idx;
	uint M;  // rows of A / rows of C
	uint N;  // rows of B / cols of C
	uint K;  // cols of A / cols of B
};

[[vk::push_constant]] FullPushConstants pc;

[[vk::binding(0, 0)]] RWByteAddressBuffer heap[];

[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
	uint idx = tid.x;
	if (idx >= pc.M * pc.N) return;

	uint i = idx / pc.N;
	uint j = idx % pc.N;

	float sum = 0.0;
	for (uint k = 0; k < pc.K; k++) {
		sum += oaLoad(heap[pc.a_idx], i * pc.K + k) * oaLoad(heap[pc.b_idx], j * pc.K + k);
	}
	oaStore(heap[pc.out_idx], i * pc.N + j, sum);
}
