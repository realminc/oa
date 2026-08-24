// AudioBiquadBlockSummary — per-block affine summaries for parallel IIR filtering.

import storage;

struct PushConstants {
	uint a_idx;
	uint summary_idx;
	uint channels;
	uint samples;
	uint blocks;
	uint block_size;
	float b0;
	float b1;
	float b2;
	float a1;
	float a2;
};

[[vk::push_constant]] PushConstants pc;
[[vk::binding(0, 0)]] RWByteAddressBuffer heap[];

[shader("compute")]
[numthreads(64, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
	const uint task = tid.x;
	const uint task_count = pc.channels * pc.blocks;
	if (task >= task_count) return;

	const uint channel = task / pc.blocks;
	const uint block = task - channel * pc.blocks;
	const uint first = block * pc.block_size;
	const uint last = min(first + pc.block_size, pc.samples);
	const uint input_base = channel * pc.samples;

	// State transition for transposed direct-form II:
	//   state' = A*state + c*x. Accumulate one affine transform per block.
	float m00 = 1.0f;
	float m01 = 0.0f;
	float m10 = 0.0f;
	float m11 = 1.0f;
	float v0 = 0.0f;
	float v1 = 0.0f;
	for (uint sample = first; sample < last; ++sample) {
		const float x = oaLoad(heap[pc.a_idx], input_base + sample);
		const float y = pc.b0 * x + v0;
		const float next_v0 = pc.b1 * x - pc.a1 * y + v1;
		const float next_v1 = pc.b2 * x - pc.a2 * y;

		const float next_m00 = -pc.a1 * m00 + m10;
		const float next_m01 = -pc.a1 * m01 + m11;
		const float next_m10 = -pc.a2 * m00;
		const float next_m11 = -pc.a2 * m01;
		m00 = next_m00;
		m01 = next_m01;
		m10 = next_m10;
		m11 = next_m11;
		v0 = next_v0;
		v1 = next_v1;
	}

	const uint out = task * 6;
	oaStore(heap[pc.summary_idx], out + 0, m00);
	oaStore(heap[pc.summary_idx], out + 1, m01);
	oaStore(heap[pc.summary_idx], out + 2, m10);
	oaStore(heap[pc.summary_idx], out + 3, m11);
	oaStore(heap[pc.summary_idx], out + 4, v0);
	oaStore(heap[pc.summary_idx], out + 5, v1);
}
