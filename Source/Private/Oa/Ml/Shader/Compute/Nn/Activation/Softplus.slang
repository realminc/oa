// AUTO-GENERATED from Ml/MlFnMatrixActivation.toml — DO NOT EDIT.
// Kernel: Softplus (Softplus activation: Out = log(1 + exp(A)))

import Storage;

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
		let a = OaLoad(heap[pc.x_idx], idx);
		// Numerically stable softplus: max(a,0)+log1p(e^-|a|). Identical to
		// log(1+e^a) but the exp argument is always <= 0, so no f32 overflow.
		OaStore(heap[pc.out_idx], idx, max(a, 0.0f) + log(1.0f + exp(-abs(a))));
	}
}
