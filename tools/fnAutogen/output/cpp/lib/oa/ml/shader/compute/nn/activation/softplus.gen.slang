// AUTO-GENERATED from ml/mlFnMatrixActivation.toml — DO NOT EDIT.
// Kernel: Softplus (Softplus: out = log(1+exp(A)), computed stably as max(A,0)+log1p(exp(-|A|)) (no f32 overflow))

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
		oaStore(heap[pc.out_idx], idx, max(a, 0.0f) + log(1.0f + exp(-abs(a))));
	}
}
