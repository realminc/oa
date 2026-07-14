// CategoricalAccuracyCount — last-axis argmax accuracy reduced to one UInt32.
// One workgroup avoids global atomics/temporary partials; lanes stride over rows.

import Storage;

struct PushConstants {
	uint logits_idx;
	uint labels_idx;
	uint output_idx;
	uint rows;
	uint classes;
	uint labelType; // 0=u8, 1=u32, 2=i32
};

[[vk::push_constant]] PushConstants pc;
[[vk::binding(0, 0)]] RWByteAddressBuffer heap[];
groupshared uint partial[256];

[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 tid : SV_GroupThreadID) {
	uint correct = 0;
	for (uint row = tid.x; row < pc.rows; row += 256) {
		uint best = 0;
		float bestValue = OaLoad(heap[pc.logits_idx], row * pc.classes);
		for (uint c = 1; c < pc.classes; ++c) {
			float value = OaLoad(heap[pc.logits_idx], row * pc.classes + c);
			if (value > bestValue) { bestValue = value; best = c; }
		}
		uint label = pc.labelType == 0
			? heap[pc.labels_idx].Load<uint8_t>(row)
			: heap[pc.labels_idx].Load(row * 4);
		correct += best == label ? 1 : 0;
	}
	partial[tid.x] = correct;
	GroupMemoryBarrierWithGroupSync();
	for (uint stride = 128; stride > 0; stride >>= 1) {
		if (tid.x < stride) partial[tid.x] += partial[tid.x + stride];
		GroupMemoryBarrierWithGroupSync();
	}
	if (tid.x == 0) heap[pc.output_idx].Store(0, partial[0]);
}
