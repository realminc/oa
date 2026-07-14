// Auxiliary-loss-free routing-bias update. One lane owns one expert and scans
// its selection column; E is small, so this avoids host readback and atomics.

import Storage;

struct PushConstants {
	uint mask_idx;
	uint bias_idx;
	uint T;
	uint E;
	uint K;
	float gamma;
};

[[vk::push_constant]] PushConstants pc;
[[vk::binding(0, 0)]] RWByteAddressBuffer heap[];

[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
	uint expert = tid.x;
	if (expert >= pc.E) return;
	float selected = 0.0f;
	for (uint row = 0; row < pc.T; ++row)
		selected += OaLoad(heap[pc.mask_idx], row * pc.E + expert);
	float targetCount = float(pc.T * pc.K) / float(pc.E);
	float step = selected > targetCount ? 1.0f : (selected < targetCount ? -1.0f : 0.0f);
	OaStore(heap[pc.bias_idx], expert, OaLoad(heap[pc.bias_idx], expert) - pc.gamma * step);
}
