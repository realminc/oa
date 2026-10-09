import storage;
import attributes;

struct PushConstants {
	uint input_index, transform_index, output_index;
	uint batch_size, channels, height, width;
};
[[vk::push_constant]] PushConstants push;
[[vk::binding(0, 0)]] RWByteAddressBuffer storage_buffers[];

[kernel_name("color_twist")]
[domain("image")]
[variant("rgb_3x4")]
[dtype("f32")]
[layout("nchw_or_chw")]
[status("experimental")]
[shader("compute")]
[numthreads(16, 16, 1)]
void main(uint3 thread_id : SV_DispatchThreadID) {
	uint x = thread_id.x, y = thread_id.y, plane_index = thread_id.z;
	if (x >= push.width || y >= push.height || plane_index >= push.batch_size * push.channels) return;
	uint batch = plane_index / push.channels, channel = plane_index % push.channels;
	uint pixel = y * push.width + x;
	uint plane = push.height * push.width;
	uint base = batch * push.channels * plane + pixel;
	float result = load_f32(storage_buffers[push.input_index], base + channel * plane);
	if (channel < 3u) {
		uint row = channel * 4u;
		result = load_f32(storage_buffers[push.transform_index], row) * load_f32(storage_buffers[push.input_index], base) +
			load_f32(storage_buffers[push.transform_index], row + 1u) * load_f32(storage_buffers[push.input_index], base + plane) +
			load_f32(storage_buffers[push.transform_index], row + 2u) * load_f32(storage_buffers[push.input_index], base + 2u * plane) +
			load_f32(storage_buffers[push.transform_index], row + 3u);
	}
	store_f32(storage_buffers[push.output_index], base + channel * plane, result);
}
