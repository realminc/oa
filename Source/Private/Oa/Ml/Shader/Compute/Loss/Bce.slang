// Bce forward: compute per-element binary cross-entropy
// BCE = -(B * log(A) + (1-B) * log(1-A))
// Clamps A to [eps, 1-eps] for numerical stability.
// Bindless: buffer indices (a, b, out) prepended by OaVkDispatch::Run

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
	if (idx >= pc.count) return;

	float a = OaLoad(heap[pc.a_idx], idx);
	float b = OaLoad(heap[pc.b_idx], idx);

	float eps = 1e-7f;
	float clampedA = clamp(a, eps, 1.0f - eps);

	float loss = -(b * log(clampedA) + (1.0f - b) * log(1.0f - clampedA));
	OaStore(heap[pc.out_idx], idx, loss);
}
