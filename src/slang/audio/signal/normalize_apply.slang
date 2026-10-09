import storage;
import attributes;

struct PushConstants {
	uint input_index;
	uint level_index;
	uint output_index;
	uint count;
	float target_linear;
};
[[vk::push_constant]] PushConstants push;
[[vk::binding(0, 0)]] RWByteAddressBuffer storage_buffers[];

[kernel_name("normalize_apply")]
[domain("audio")]
[variant("generic")]
[dtype("f32")]
[status("experimental")]
[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 thread_id : SV_DispatchThreadID) {
	uint index = thread_id.x;
	if (index >= push.count) return;
	float level = load_f32(storage_buffers[push.level_index], 0);
	float value = load_f32(storage_buffers[push.input_index], index);
	store_f32(storage_buffers[push.output_index], index, value * push.target_linear / level);
}
