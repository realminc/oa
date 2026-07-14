// SmoothL1 backward: compute gradient w.r.t. input A
// For |x| < 1: dL/dA = x / N  (where x = A - B, N = count)
// For |x| >= 1: dL/dA = sign(x) / N
// Bindless: buffer indices (a, b, d_a) prepended by OaVkDispatch::Run

import Storage;

struct PushConstants {
	uint a_idx;     // Prepended by OaVkDispatch::Run
	uint b_idx;     // Prepended by OaVkDispatch::Run
	uint d_a_idx;   // Prepended by OaVkDispatch::Run
	uint count;     // From C++ code
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
	float diff = a - b;
	float absDiff = abs(diff);

	float invN = 1.0f / float(pc.count);

	float grad;
	if (absDiff < 1.0f) {
		grad = diff * invN;
	} else {
		grad = (diff >= 0.0f ? 1.0f : -1.0f) * invN;
	}

	OaStore(heap[pc.d_a_idx], idx, grad);
}
