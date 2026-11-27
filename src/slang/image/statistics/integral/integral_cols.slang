import attributes;
import storage;

struct PushConstants {
	uint output_index, batch_size, channels, height, width;
};
[[vk::push_constant]] PushConstants push;
[[vk::binding(0, 0)]] RWByteAddressBuffer storage_buffers[];

[kernel_name("integral_cols")]
[domain("image")]
[variant("prefix_cols")]
[dtype("f32")]
[layout("nchw_or_chw")]
[status("experimental")]
[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 thread_id : SV_DispatchThreadID) {
	uint index = thread_id.x;
	uint plane_count = push.batch_size * push.channels;
	uint out_width = push.width + 1u;
	uint out_height = push.height + 1u;
	if (index >= plane_count * out_width) return;
	uint plane = index / out_width;
	uint x = index - plane * out_width;
	uint out_base = plane * out_height * out_width;
	float sum = 0.0;
	for (uint y = 0; y < out_height; ++y) {
		uint offset = out_base + y * out_width + x;
		if (y > 0u) {
			sum += load_f32(storage_buffers[push.output_index], offset);
		}
		store_f32(storage_buffers[push.output_index], offset, sum);
	}
}
