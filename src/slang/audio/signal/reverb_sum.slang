import storage;
import attributes;

struct PushConstants {
	uint a_index;
	uint b_index;
	uint c_index;
	uint d_index;
	uint output_index;
	uint count;
	float normalization;
};
[[vk::push_constant]] PushConstants push;
[[vk::binding(0, 0)]] RWByteAddressBuffer storage_buffers[];

[kernel_name("reverb_sum")]
[domain("audio")]
[variant("generic")]
[dtype("f32")]
[status("experimental")]
[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 thread_id : SV_DispatchThreadID) {
	uint index = thread_id.x;
	if (index >= push.count) return;
	float value = load_f32(storage_buffers[push.a_index], index)
		+ load_f32(storage_buffers[push.b_index], index)
		+ load_f32(storage_buffers[push.c_index], index)
		+ load_f32(storage_buffers[push.d_index], index);
	store_f32(storage_buffers[push.output_index], index, value * push.normalization);
}
