import storage;
import attributes;

struct PushConstants {
	uint input_index;
	uint output_index;
	uint batch_size;
	uint channels;
	uint input_height;
	uint input_width;
	uint output_height;
	uint output_width;
};
[[vk::push_constant]] PushConstants push;
[[vk::binding(0, 0)]] RWByteAddressBuffer storage_buffers[];

[kernel_name("resize_bilinear")]
[domain("image")]
[variant("bilinear_align_corners_false")]
[dtype("f32")]
[layout("nchw_or_chw")]
[status("experimental")]
[shader("compute")]
[numthreads(16, 16, 1)]
void main(uint3 thread_id : SV_DispatchThreadID) {
	uint output_x = thread_id.x;
	uint output_y = thread_id.y;
	uint batch_channel = thread_id.z;
	uint batch_channels = push.batch_size * push.channels;
	if (output_x >= push.output_width || output_y >= push.output_height ||
		batch_channel >= batch_channels) return;

	float input_x_f = (float(output_x) + 0.5f) * float(push.input_width) /
		float(push.output_width) - 0.5f;
	float input_y_f = (float(output_y) + 0.5f) * float(push.input_height) /
		float(push.output_height) - 0.5f;
	input_x_f = clamp(input_x_f, 0.0f, float(push.input_width - 1u));
	input_y_f = clamp(input_y_f, 0.0f, float(push.input_height - 1u));
	uint x0 = min((uint)floor(input_x_f), push.input_width - 1u);
	uint y0 = min((uint)floor(input_y_f), push.input_height - 1u);
	uint x1 = min(x0 + 1u, push.input_width - 1u);
	uint y1 = min(y0 + 1u, push.input_height - 1u);
	float x_weight = input_x_f - float(x0);
	float y_weight = input_y_f - float(y0);

	uint plane = batch_channel * push.input_height * push.input_width;
	float top_left = load_f32(storage_buffers[push.input_index], plane + y0 * push.input_width + x0);
	float top_right = load_f32(storage_buffers[push.input_index], plane + y0 * push.input_width + x1);
	float bottom_left = load_f32(storage_buffers[push.input_index], plane + y1 * push.input_width + x0);
	float bottom_right = load_f32(storage_buffers[push.input_index], plane + y1 * push.input_width + x1);
	float top = lerp(top_left, top_right, x_weight);
	float bottom = lerp(bottom_left, bottom_right, x_weight);
	uint output_index =
		(batch_channel * push.output_height + output_y) * push.output_width + output_x;
	store_f32(storage_buffers[push.output_index], output_index, lerp(top, bottom, y_weight));
}
