// Batched matrix multiply with transposed right-operand storage:
// A[N,M,K] @ B[N,P,K]^T = out[N,M,P]. DTYPE-aware storage, FP32 accumulation.

import storage;

struct PushConstants {
	uint a_idx;
	uint b_idx;
	uint out_idx;
	uint batch;
	uint m;
	uint k;
	uint p;
};

[[vk::push_constant]] PushConstants pc;
[[vk::binding(0, 0)]] RWByteAddressBuffer heap[];

[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
	uint idx = tid.x;
	uint total = pc.batch * pc.m * pc.p;
	if (idx >= total) return;

	uint mp = pc.m * pc.p;
	uint batch = idx / mp;
	uint rem = idx - batch * mp;
	uint row = rem / pc.p;
	uint col = rem - row * pc.p;
	uint aBase = (batch * pc.m + row) * pc.k;
	uint bBase = (batch * pc.p + col) * pc.k;
	float accum = 0.0;
	for (uint kk = 0; kk < pc.k; ++kk) {
		accum += oaLoad(heap[pc.a_idx], aBase + kk)
			* oaLoad(heap[pc.b_idx], bBase + kk);
	}
	oaStore(heap[pc.out_idx], idx, accum);
}
