// AudioFade — linear fade-in/out envelope over planar [Channels, Samples] F32.

import Storage;

struct PushConstants {
	uint a_idx;
	uint out_idx;
	uint count;
	uint samples;
	uint fade_in;
	uint fade_out;
};

[[vk::push_constant]] PushConstants pc;
[[vk::binding(0, 0)]] RWByteAddressBuffer heap[];

[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
	uint idx = tid.x;
	if (idx >= pc.count) return;

	uint sample = idx % pc.samples;
	float envelope = 1.0f;
	if (pc.fade_in > 0 && sample < pc.fade_in) {
		envelope = float(sample) / float(pc.fade_in);
	}
	uint reverse = pc.samples - 1 - sample;
	if (pc.fade_out > 0 && reverse < pc.fade_out) {
		envelope = min(envelope, float(reverse) / float(pc.fade_out));
	}
	OaStore(heap[pc.out_idx], idx, OaLoad(heap[pc.a_idx], idx) * envelope);
}
