// AUTO-GENERATED from Core/CoreFnMatrixElemwise.toml — DO NOT EDIT.
// Kernel: Sub (Element-wise subtraction: Out = A - B)

import Storage;

struct PushConstants {
	uint a_idx;
	uint b_idx;
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
		let a = OaLoad(heap[pc.a_idx], idx);
		let b = OaLoad(heap[pc.b_idx], idx);
		OaStore(heap[pc.out_idx], idx, a - b);
	}
}
