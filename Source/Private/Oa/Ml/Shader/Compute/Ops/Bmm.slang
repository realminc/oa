// Batched matrix multiply: A[N,M,K] @ B[N,K,P] = out[N,M,P].
// out[n,i,j] = sum_k A[n,i,k] * B[n,k,j]. One thread per output element.
// The enabler for differentiable forward kinematics (per-frame 3x3 rotation
// composition) and any per-row small-matrix algebra. DTYPE-aware (fp32 or bf16
// storage via OaLoad/OaStore; accumulation is fp32). Was OaLoadF32/OaStoreF32,
// which read 4 bytes/element and corrupted bf16-stored attention → NaN (the
// db51c5c storage fix missed this shader; transformers NaN'd in bf16, GRU didn't).

import Storage;

struct PushConstants {
	uint a_idx;      // bufs[0] — runtime prepends
	uint b_idx;      // bufs[1]
	uint out_idx;    // bufs[2]
	uint batch;      // N
	uint m;          // rows of A / out
	uint k;          // shared dim
	uint p;          // cols of B / out
};

[[vk::push_constant]] PushConstants pc;

[[vk::binding(0, 0)]] RWByteAddressBuffer heap[];

[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
	uint idx = tid.x;
	uint total = pc.batch * pc.m * pc.p;
	if (idx >= total) {
		return;
	}
	uint mp  = pc.m * pc.p;
	uint n   = idx / mp;
	uint rem = idx - n * mp;
	uint i   = rem / pc.p;
	uint j   = rem - i * pc.p;

	uint aBase = n * pc.m * pc.k + i * pc.k;   // A[n,i,0]
	uint bBase = n * pc.k * pc.p + j;          // B[n,0,j]
	float acc = 0.0f;
	for (uint kk = 0; kk < pc.k; ++kk) {
		acc += OaLoad(heap[pc.a_idx], aBase + kk)
		     * OaLoad(heap[pc.b_idx], bBase + kk * pc.p);
	}
	OaStore(heap[pc.out_idx], idx, acc);
}
