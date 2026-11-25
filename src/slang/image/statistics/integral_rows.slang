import attributes;
import storage;

struct PushConstants {
	uint input_index, output_index, batch_size, channels, height, width;
};
[[vk::push_constant]] PushConstants push;
[[vk::binding(0, 0)]] RWByteAddressBuffer storage_buffers[];

[kernel_name("integral_rows")]
[domain("image")]
[variant("prefix_rows")]
[dtype("f32")]
[layout("nchw_or_chw")]
[status("experimental")]
[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 thread_id : SV_DispatchThreadID) {
	uint index = thread_id.x;
	uint plane_count = push.batch_size * push.channels;
	if (index >= plane_count * push.height) return;
	uint plane = index / push.height;
	uint y = index - plane * push.height;
	uint out_width = push.width + 1u;
	uint in_base = plane * push.height * push.width;
	uint out_base = plane * (push.height + 1u) * out_width + (y + 1u) * out_width;
	float sum = 0.0;
	for (uint x = 0; x < push.width; ++x) {
		sum += load_f32(storage_buffers[push.input_index], in_base + y * push.width + x);
		store_f32(storage_buffers[push.output_index], out_base + x + 1u, sum);
	}
}
