// SmoothL1 forward: compute per-element smooth L1 / Huber loss (beta=1.0)
// For |x| < 1: 0.5 * x^2
// For |x| >= 1: |x| - 0.5
// Output: per-element loss [N] — caller reduces with Mean for scalar loss.
// Bindless: buffer indices (a, b, out) prepended by OaVkDispatch::Run

import Storage;

struct PushConstants {
	uint a_idx;     // Prepended by OaVkDispatch::Run
	uint b_idx;     // Prepended by OaVkDispatch::Run
	uint out_idx;   // Prepended by OaVkDispatch::Run
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

	float loss;
	if (absDiff < 1.0f) {
		loss = 0.5f * diff * diff;
	} else {
		loss = absDiff - 0.5f;
	}

	OaStore(heap[pc.out_idx], idx, loss);
}
