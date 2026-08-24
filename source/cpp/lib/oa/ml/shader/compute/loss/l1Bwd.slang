// L1 backward: compute gradient w.r.t. input A
// dL/dA = sign(A-B)/N
// Bindless: buffer indices (a, b, d_a) prepended by oavk::Dispatch::Run

import storage;

struct PushConstants {
	uint a_idx;
	uint b_idx;
	uint d_a_idx;
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
	float invN = 1.0f / float(pc.count);

	float grad = (diff > 0.0f) ? invN : (diff < 0.0f ? -invN : 0.0f);
	oaStore(heap[pc.d_a_idx], idx, grad);
}
