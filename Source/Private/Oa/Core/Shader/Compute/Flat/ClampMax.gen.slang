// AUTO-GENERATED from Core/CoreFnMatrixElemwise.toml — DO NOT EDIT.
// Kernel: ClampMax (Element-wise clamp max: Out = min(A, Max))

import Storage;

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
		let a = OaLoad(heap[pc.x_idx], idx);
		OaStore(heap[pc.out_idx], idx, min(a, pc.Max));
	}
}
