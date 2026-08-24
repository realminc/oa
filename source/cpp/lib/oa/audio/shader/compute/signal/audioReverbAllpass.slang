// AudioReverbAllpass — one zero-state Schroeder all-pass diffuser.

import storage;

struct PushConstants {
	uint a_idx;
	uint out_idx;
	uint channels;
	uint samples;
	uint delay_samples;
	float gain;
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
	const uint base = channel * pc.samples;
	float delayed_input = 0.0f;
	float delayed_output = 0.0f;
	for (uint sample = residue; sample < pc.samples; sample += pc.delay_samples) {
		const float input = oaLoad(heap[pc.a_idx], base + sample);
		const float output = -pc.gain * input
			+ delayed_input + pc.gain * delayed_output;
		oaStore(heap[pc.out_idx], base + sample, output);
		delayed_input = input;
		delayed_output = output;
	}
}
