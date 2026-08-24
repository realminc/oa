// AUTO-GENERATED from core/coreFnMatrixElemwise.toml — DO NOT EDIT.
// Kernel: Div (Element-wise division: out = A / B)

import storage;

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
		let a = oaLoad(heap[pc.a_idx], idx);
		let b = oaLoad(heap[pc.b_idx], idx);
		oaStore(heap[pc.out_idx], idx, a / b);
	}
}
