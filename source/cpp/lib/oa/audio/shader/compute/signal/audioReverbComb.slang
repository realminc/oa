// AudioReverbComb — one zero-state Schroeder feedback comb over planar audio.

import storage;

struct PushConstants {
	uint a_idx;
	uint out_idx;
	uint channels;
	uint in_samples;
	uint out_samples;
	uint delay_samples;
	float feedback;
};

[[vk::push_constant]] PushConstants pc;
[[vk::binding(0, 0)]] RWByteAddressBuffer heap[];

[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
	const uint sequence = tid.x;
	const uint sequence_count = pc.channels * pc.delay_samples;
	if (sequence >= sequence_count) return;

	const uint channel = sequence / pc.delay_samples;
	const uint residue = sequence - channel * pc.delay_samples;
	const uint input_base = channel * pc.in_samples;
	const uint output_base = channel * pc.out_samples;
	float previous = 0.0f;
	for (uint sample = residue; sample < pc.out_samples; sample += pc.delay_samples) {
		float input = 0.0f;
		if (sample >= pc.delay_samples) {
			const uint source = sample - pc.delay_samples;
			if (source < pc.in_samples) {
				input = oaLoad(heap[pc.a_idx], input_base + source);
			}
		}
		const float value = input + pc.feedback * previous;
		oaStore(heap[pc.out_idx], output_base + sample, value);
		previous = value;
	}
}
