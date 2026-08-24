// AudioBiquadApply — block-parallel transposed direct-form-II filtering.

import storage;

struct PushConstants {
	uint a_idx;
	uint state_idx;
	uint out_idx;
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
	const uint sample_base = channel * pc.samples;
	const uint state_base = task * 2;
	float state0 = oaLoad(heap[pc.state_idx], state_base + 0);
	float state1 = oaLoad(heap[pc.state_idx], state_base + 1);

	for (uint sample = first; sample < last; ++sample) {
		const uint index = sample_base + sample;
		const float x = oaLoad(heap[pc.a_idx], index);
		const float y = pc.b0 * x + state0;
		const float next0 = pc.b1 * x - pc.a1 * y + state1;
		const float next1 = pc.b2 * x - pc.a2 * y;
		oaStore(heap[pc.out_idx], index, y);
		state0 = next0;
		state1 = next1;
	}
}
