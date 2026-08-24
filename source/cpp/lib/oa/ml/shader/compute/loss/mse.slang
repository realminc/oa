// Mse forward: compute per-element squared error
// Output: per-element (A-B)^2 — caller reduces with Mean for scalar loss.
// Bindless: buffer indices (a, b, out) prepended by oavk::Dispatch::Run

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
	if (idx >= pc.count) return;

	float a = oaLoad(heap[pc.a_idx], idx);
	float b = oaLoad(heap[pc.b_idx], idx);
	float diff = a - b;

	oaStore(heap[pc.out_idx], idx, diff * diff);
}
