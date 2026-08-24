// AUTO-GENERATED from core/coreFnMatrixRng.toml — DO NOT EDIT.
// Kernel: Fill (Create a Float32 matrix filled with a constant value)

import storage;

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
		oaStore(heap[pc.out_idx], idx, pc.Value);
	}
}
