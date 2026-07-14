// Bce backward: compute gradient w.r.t. input A (predictions)
// dL/dA = -(B/A - (1-B)/(1-A)) / N = (A - B) / (A*(1-A)) / N
// Clamps A to [eps, 1-eps] for numerical stability.
// Bindless: buffer indices (a, b, d_a) prepended by OaVkDispatch::Run

import Storage;

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

	float a = OaLoad(heap[pc.a_idx], idx);
	float b = OaLoad(heap[pc.b_idx], idx);

	float eps = 1e-7f;
	float clampedA = clamp(a, eps, 1.0f - eps);
	float invN = 1.0f / float(pc.count);

	float grad = (clampedA - b) / (clampedA * (1.0f - clampedA)) * invN;
	OaStore(heap[pc.d_a_idx], idx, grad);
}
