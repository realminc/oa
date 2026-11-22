import storage;
import attributes;

struct PushConstants {
	uint input_index, output_index;
	uint batch_size, channels, height, width, horizontal, vertical;
};
[[vk::push_constant]] PushConstants push;
[[vk::binding(0, 0)]] RWByteAddressBuffer storage_buffers[];

[kernel_name("flip")]
[domain("image")]
[variant("nchw_or_chw")]
[dtype("f32")]
[layout("nchw_or_chw")]
[status("experimental")]
[shader("compute")]
[numthreads(16, 16, 1)]
void main(uint3 thread_id : SV_DispatchThreadID) {
	uint x = thread_id.x, y = thread_id.y, plane = thread_id.z;
	if (x >= push.width || y >= push.height || plane >= push.batch_size * push.channels) return;
	uint source_x = push.horizontal != 0u ? push.width - 1u - x : x;
	uint source_y = push.vertical != 0u ? push.height - 1u - y : y;
	uint source = (plane * push.height + source_y) * push.width + source_x;
	uint target = (plane * push.height + y) * push.width + x;
	store_f32(storage_buffers[push.output_index], target,
		load_f32(storage_buffers[push.input_index], source));
}
