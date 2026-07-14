// AUTO-GENERATED from MlFnMatrixActivation.toml — DO NOT EDIT.
// Kernel: Elu (ELU activation: Out = A if A > 0 else Alpha * (exp(A) - 1))

import Storage;

struct PushConstants {
	uint x_idx;
	uint out_idx;
	uint count;
	float Alpha;
};

[[vk::push_constant]] PushConstants pc;

[[vk::binding(0, 0)]] RWByteAddressBuffer heap[];

[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
	uint idx = tid.x;
	if (idx < pc.count) {
		let a = OaLoad(heap[pc.x_idx], idx);
		OaStore(heap[pc.out_idx], idx, a > 0.0f ? a : pc.Alpha * (exp(a) - 1.0f));
	}
}
