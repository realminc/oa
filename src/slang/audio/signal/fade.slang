import storage;
import attributes;

struct PushConstants {
	uint input_index;
	uint output_index;
	uint count;
	uint samples;
	uint fade_in;
	uint fade_out;
};
[[vk::push_constant]] PushConstants push;
[[vk::binding(0, 0)]] RWByteAddressBuffer storage_buffers[];

[kernel_name("fade")]
[domain("audio")]
[variant("generic")]
[dtype("f32")]
[status("experimental")]
[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 thread_id : SV_DispatchThreadID) {
	uint index = thread_id.x;
	if (index >= push.count) return;
	uint sample = index % push.samples;
	float envelope = 1.0f;
	if (push.fade_in > 0 && sample < push.fade_in) envelope = float(sample) / float(push.fade_in);
	uint reverse = push.samples - 1 - sample;
	if (push.fade_out > 0 && reverse < push.fade_out) {
		envelope = min(envelope, float(reverse) / float(push.fade_out));
	}
	store_f32(storage_buffers[push.output_index], index,
		load_f32(storage_buffers[push.input_index], index) * envelope);
}
