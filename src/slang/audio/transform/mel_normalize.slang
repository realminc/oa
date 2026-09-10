import storage;
import attributes;

struct PushConstants {
	uint input_index;
	uint output_index;
	uint channels;
	uint values_per_channel;
};
[[vk::push_constant]] PushConstants push;
[[vk::binding(0, 0)]] RWByteAddressBuffer storage_buffers[];

[kernel_name("mel_normalize")]
[domain("audio")]
[variant("generic")]
[dtype("f32")]
[status("experimental")]
[shader("compute")]
[numthreads(1, 1, 1)]
void main(uint3 thread_id : SV_DispatchThreadID) {
	uint channel = thread_id.x;
	if (channel >= push.channels) return;
	uint base = channel * push.values_per_channel;
	float mean = 0.0f;
	for (uint index = 0; index < push.values_per_channel; ++index) {
		mean += load_f32(storage_buffers[push.input_index], base + index);
	}
	mean /= float(push.values_per_channel);
	float variance = 0.0f;
	for (uint index = 0; index < push.values_per_channel; ++index) {
		float centered = load_f32(storage_buffers[push.input_index], base + index) - mean;
		variance += centered * centered;
	}
	float inverse_stddev = rsqrt(variance / float(push.values_per_channel) + 1e-5f);
	for (uint index = 0; index < push.values_per_channel; ++index) {
		float centered = load_f32(storage_buffers[push.input_index], base + index) - mean;
		store_f32(storage_buffers[push.output_index], base + index, centered * inverse_stddev);
	}
}
