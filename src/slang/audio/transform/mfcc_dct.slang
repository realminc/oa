import storage;
import attributes;

static const float PI = 3.14159265358979323846f;
struct PushConstants {
	uint mel_index;
	uint output_index;
	uint channels;
	uint frames;
	uint num_mels;
	uint num_coeffs;
};
[[vk::push_constant]] PushConstants push;
[[vk::binding(0, 0)]] RWByteAddressBuffer storage_buffers[];

[kernel_name("mfcc_dct")]
[domain("audio")]
[variant("generic")]
[dtype("f32")]
[status("experimental")]
[shader("compute")]
[numthreads(32, 1, 1)]
void main(uint3 group_id : SV_GroupID, uint3 group_thread : SV_GroupThreadID) {
	uint coefficient = group_id.x * 32 + group_thread.x;
	uint frame = group_id.y, channel = group_id.z;
	if (coefficient >= push.num_coeffs || frame >= push.frames || channel >= push.channels) return;
	float scale = coefficient == 0
		? sqrt(1.0f / float(push.num_mels))
		: sqrt(2.0f / float(push.num_mels));
	float value = 0.0f;
	for (uint mel = 0; mel < push.num_mels; ++mel) {
		float basis = cos(PI / float(push.num_mels) * (float(mel) + 0.5f) * float(coefficient));
		uint input = (channel * push.num_mels + mel) * push.frames + frame;
		value += load_f32(storage_buffers[push.mel_index], input) * basis * scale;
	}
	uint output = (channel * push.num_coeffs + coefficient) * push.frames + frame;
	store_f32(storage_buffers[push.output_index], output, value);
}
