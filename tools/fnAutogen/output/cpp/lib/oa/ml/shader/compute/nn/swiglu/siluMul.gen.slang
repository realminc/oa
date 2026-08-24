// AUTO-GENERATED from ml/mlFnMatrixActivation.toml — DO NOT EDIT.
// Kernel: SiluMul (SiLU on first half, multiply with second half: out = SiLU(A[:N]) * A[N:])

import storage;

struct PushConstants {
	uint x_idx;
	uint out_idx;
	uint count;
	uint intermediate_size;
};

[[vk::push_constant]] PushConstants pc;

[[vk::binding(0, 0)]] RWByteAddressBuffer heap[];

[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
	uint idx = tid.x;
	if (idx < pc.count) {
		let a = oaLoad(heap[pc.x_idx], idx);
		oaStore(heap[pc.out_idx], idx, SiluMul(a));
	}
}
