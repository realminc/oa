// AudioSaturate — tanh soft clipping over planar F32 audio.

import storage;

struct PushConstants {
	uint a_idx;
	uint out_idx;
	uint count;
	float drive_db;
	float mix;
};

[[vk::push_constant]] PushConstants pc;
[[vk::binding(0, 0)]] RWByteAddressBuffer heap[];

[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
	uint idx = tid.x;
	if (idx >= pc.count) return;

	const float dry = oaLoad(heap[pc.a_idx], idx);
	const float drive = exp2(pc.drive_db / 6.020599913279624f);
	const float wet = tanh(dry * drive);
	oaStore(heap[pc.out_idx], idx, dry + (wet - dry) * pc.mix);
}
