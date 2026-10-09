import attributes;
import storage;

struct PushConstants {
	uint input_index, output_index, batch_size, height, width, swap_red_blue;
};
[[vk::push_constant]] PushConstants push;
[[vk::binding(0, 0)]] RWByteAddressBuffer storage_buffers[];

[kernel_name("convert_color")]
[domain("image")]
[variant("rgb_bgr")]
[dtype("f32")]
[layout("nchw_or_chw")]
[status("experimental")]
[shader("compute")]
[numthreads(16, 16, 1)]
void main(uint3 thread_id : SV_DispatchThreadID) {
	uint x = thread_id.x, y = thread_id.y, plane_index = thread_id.z;
	if (x >= push.width || y >= push.height || plane_index >= push.batch_size * 3u) return;
	uint batch = plane_index / 3u, channel = plane_index % 3u;
	uint source_channel = push.swap_red_blue != 0u && channel != 1u ? 2u - channel : channel;
	uint plane = push.height * push.width, pixel = y * push.width + x;
	uint source = (batch * 3u + source_channel) * plane + pixel;
	uint target = (batch * 3u + channel) * plane + pixel;
	store_f32(storage_buffers[push.output_index], target,
		load_f32(storage_buffers[push.input_index], source));
}
