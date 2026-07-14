// AUTO-GENERATED from CoreFnMatrixElemwise.toml — DO NOT EDIT.
// Kernel: Pow (Element-wise power: Out = A^Exponent)

import Storage;

struct PushConstants {
	uint x_idx;
	uint out_idx;
	uint count;
	float Exponent;
};

[[vk::push_constant]] PushConstants pc;

[[vk::binding(0, 0)]] RWByteAddressBuffer heap[];

[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
	uint idx = tid.x;
	if (idx < pc.count) {
		let a = OaLoad(heap[pc.x_idx], idx);
		OaStore(heap[pc.out_idx], idx, pow(a, pc.Exponent));
	}
}
