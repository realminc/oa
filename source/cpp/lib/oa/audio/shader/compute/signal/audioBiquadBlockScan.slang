// AudioBiquadBlockScan — exact block-boundary state propagation per channel.

import storage;

struct PushConstants {
	uint summary_idx;
	uint state_idx;
	uint channels;
	uint blocks;
};

[[vk::push_constant]] PushConstants pc;
[[vk::binding(0, 0)]] RWByteAddressBuffer heap[];

[shader("compute")]
[numthreads(64, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
	const uint channel = tid.x;
	if (channel >= pc.channels) return;

	float state0 = 0.0f;
	float state1 = 0.0f;
	for (uint block = 0; block < pc.blocks; ++block) {
		const uint task = channel * pc.blocks + block;
		const uint state_out = task * 2;
		oaStore(heap[pc.state_idx], state_out + 0, state0);
		oaStore(heap[pc.state_idx], state_out + 1, state1);

		const uint summary = task * 6;
		const float m00 = oaLoad(heap[pc.summary_idx], summary + 0);
		const float m01 = oaLoad(heap[pc.summary_idx], summary + 1);
		const float m10 = oaLoad(heap[pc.summary_idx], summary + 2);
		const float m11 = oaLoad(heap[pc.summary_idx], summary + 3);
		const float v0 = oaLoad(heap[pc.summary_idx], summary + 4);
		const float v1 = oaLoad(heap[pc.summary_idx], summary + 5);
		const float next0 = m00 * state0 + m01 * state1 + v0;
		const float next1 = m10 * state0 + m11 * state1 + v1;
		state0 = next0;
		state1 = next1;
	}
}
