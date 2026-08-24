// AudioReverbMix — dry/wet mix with an explicit finite tail.

import storage;

struct PushConstants {
	uint dry_idx;
	uint wet_idx;
	uint out_idx;
	uint channels;
	uint in_samples;
	uint out_samples;
	float wet;
};

[[vk::push_constant]] PushConstants pc;
[[vk::binding(0, 0)]] RWByteAddressBuffer heap[];

[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
	const uint index = tid.x;
	const uint count = pc.channels * pc.out_samples;
	if (index >= count) return;

	const uint channel = index / pc.out_samples;
	const uint sample = index - channel * pc.out_samples;
	float dry = 0.0f;
	if (sample < pc.in_samples) {
		dry = oaLoad(heap[pc.dry_idx], channel * pc.in_samples + sample);
	}
	const float reverberated = oaLoad(heap[pc.wet_idx], index);
	oaStore(heap[pc.out_idx], index, dry + (reverberated - dry) * pc.wet);
}
