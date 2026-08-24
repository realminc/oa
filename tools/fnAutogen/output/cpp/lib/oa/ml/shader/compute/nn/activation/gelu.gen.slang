// AUTO-GENERATED from ml/mlFnMatrixActivation.toml — DO NOT EDIT.
// Kernel: Gelu (GELU activation: out = GELU(A))

import storage;

struct PushConstants {
	uint x_idx;
	uint out_idx;
	uint count;
};

[[vk::push_constant]] PushConstants pc;

[[vk::binding(0, 0)]] RWByteAddressBuffer heap[];

[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
	uint idx = tid.x;
	if (idx < pc.count) {
		let a = oaLoad(heap[pc.x_idx], idx);
		oaStore(heap[pc.out_idx], idx, 0.5f * a * (1.0f + tanh(0.7978845608f * (a + 0.044715f * a * a * a))));
	}
}
