import storage;
import attributes;

struct PushConstants {
	uint input_index;
	uint output_index;
	uint element_count;
};

[[vk::push_constant]] PushConstants push;
[[vk::binding(0, 0)]] RWByteAddressBuffer storage_buffers[];

[kernel_name("cast_f32_i32")]
[domain("matrix")]
[variant("generic")]
[dtype("f32")]
[status("experimental")]
[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 dispatch_thread_id : SV_DispatchThreadID) {
	uint index = dispatch_thread_id.x;
	if (index >= push.element_count) {
		return;
	}
	store_i32(
		storage_buffers[push.output_index],
		index,
		int(load_f32(storage_buffers[push.input_index], index))
	);
}
