// AUTO-GENERATED from Core/CoreFnMatrixRng.toml — DO NOT EDIT.
// Kernel: Fill (Fill tensor with constant value: Out = Value)

import Storage;

struct PushConstants {
	uint out_idx;
	uint count;
	float Value;
};

[[vk::push_constant]] PushConstants pc;

[[vk::binding(0, 0)]] RWByteAddressBuffer heap[];

[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
	uint idx = tid.x;
	if (idx < pc.count) {
		OaStore(heap[pc.out_idx], idx, pc.Value);
	}
}
