import storage;
import attributes;

struct PushConstants {
	uint input_index;
	uint output_index;
	uint count;
	uint samples;
	float alpha;
};
[[vk::push_constant]] PushConstants push;
[[vk::binding(0, 0)]] RWByteAddressBuffer storage_buffers[];

[kernel_name("pre_emphasis")]
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
	float value = load_f32(storage_buffers[push.input_index], index);
	if (sample > 0) value -= push.alpha * load_f32(storage_buffers[push.input_index], index - 1);
	store_f32(storage_buffers[push.output_index], index, value);
}
