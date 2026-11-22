import storage;
import attributes;

struct PushConstants {
	uint input_index;
	uint output_index;
	uint channels;
	uint samples;
};
[[vk::push_constant]] PushConstants push;
[[vk::binding(0, 0)]] RWByteAddressBuffer storage_buffers[];

[kernel_name("to_mono")]
[domain("audio")]
[variant("generic")]
[dtype("f32")]
[status("experimental")]
[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 thread_id : SV_DispatchThreadID) {
	uint sample = thread_id.x;
	if (sample >= push.samples) return;
	float sum = 0.0f;
	for (uint channel = 0; channel < push.channels; ++channel) {
		sum += load_f32(storage_buffers[push.input_index], channel * push.samples + sample);
	}
	store_f32(storage_buffers[push.output_index], sample, sum / float(push.channels));
}
