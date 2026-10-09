import attributes;
import storage;

struct PushConstants {
	uint input_index, output_index, batch_size, channels;
	uint input_height, input_width, output_height, output_width;
	float mean_0, mean_1, mean_2, std_0, std_1, std_2;
};
[[vk::push_constant]] PushConstants push;
[[vk::binding(0, 0)]] RWByteAddressBuffer storage_buffers[];

[kernel_name("resize_normalize")]
[domain("image")]
[variant("bilinear_half_pixel")]
[dtype("f32")]
[layout("nchw_or_chw")]
[status("experimental")]
[shader("compute")]
[numthreads(16, 16, 1)]
void main(uint3 thread_id : SV_DispatchThreadID) {
	uint x = thread_id.x, y = thread_id.y, plane_index = thread_id.z;
	if (x >= push.output_width || y >= push.output_height ||
		plane_index >= push.batch_size * push.channels) return;
	float source_x = clamp((float(x) + 0.5f) * float(push.input_width) / float(push.output_width) - 0.5f, 0.0f, float(push.input_width - 1u));
	float source_y = clamp((float(y) + 0.5f) * float(push.input_height) / float(push.output_height) - 0.5f, 0.0f, float(push.input_height - 1u));
	uint x0 = uint(floor(source_x)), y0 = uint(floor(source_y));
	uint x1 = min(x0 + 1u, push.input_width - 1u), y1 = min(y0 + 1u, push.input_height - 1u);
	float fx = source_x - float(x0), fy = source_y - float(y0);
	uint base = plane_index * push.input_height * push.input_width;
	float top = lerp(load_f32(storage_buffers[push.input_index], base + y0 * push.input_width + x0), load_f32(storage_buffers[push.input_index], base + y0 * push.input_width + x1), fx);
	float bottom = lerp(load_f32(storage_buffers[push.input_index], base + y1 * push.input_width + x0), load_f32(storage_buffers[push.input_index], base + y1 * push.input_width + x1), fx);
	float value = lerp(top, bottom, fy);
	uint channel = plane_index % push.channels;
	float mean = channel == 0u ? push.mean_0 : channel == 1u ? push.mean_1 : push.mean_2;
	float stddev = channel == 0u ? push.std_0 : channel == 1u ? push.std_1 : push.std_2;
	uint target = (plane_index * push.output_height + y) * push.output_width + x;
	store_f32(storage_buffers[push.output_index], target, (value - mean) / stddev);
}
