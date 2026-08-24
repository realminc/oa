// AUTO-GENERATED from core/coreFnMatrixElemwise.toml — DO NOT EDIT.
// Kernel: ClampMax (Element-wise clamp max: out = min(A, Max))

import storage;

struct PushConstants {
	uint x_idx;
	uint out_idx;
	uint count;
	float Max;
};

[[vk::push_constant]] PushConstants pc;

[[vk::binding(0, 0)]] RWByteAddressBuffer heap[];

[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
	uint idx = tid.x;
	if (idx < pc.count) {
		let a = oaLoad(heap[pc.x_idx], idx);
		oaStore(heap[pc.out_idx], idx, min(a, pc.Max));
	}
}
