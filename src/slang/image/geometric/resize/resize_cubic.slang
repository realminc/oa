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

// Keys bicubic kernel with a = -0.75 (OpenCV INTER_CUBIC).
float cubic_weight(float t) {
	const float a = -0.75f;
	float x = abs(t);
	if (x <= 1.0f) {
		return (a + 2.0f) * x * x * x - (a + 3.0f) * x * x + 1.0f;
	}
	if (x < 2.0f) {
		return a * x * x * x - 5.0f * a * x * x + 8.0f * a * x - 4.0f * a;
	}
	return 0.0f;
}

[kernel_name("resize_cubic")]
[domain("image")]
[variant("nchw_or_chw")]
[dtype("f32")]
[layout("nchw_or_chw")]
[status("experimental")]
[shader("compute")]
[numthreads(16, 16, 1)]
void main(uint3 thread_id : SV_DispatchThreadID) {
	uint output_x = thread_id.x;
	uint output_y = thread_id.y;
	uint batch_channel = thread_id.z;
	if (output_x >= push.output_width || output_y >= push.output_height ||
		batch_channel >= push.batch_size * push.channels) return;
	float input_x_f = (float(output_x) + 0.5f) * float(push.input_width) /
		float(push.output_width) - 0.5f;
	float input_y_f = (float(output_y) + 0.5f) * float(push.input_height) /
		float(push.output_height) - 0.5f;
	int base_x = int(floor(input_x_f));
	int base_y = int(floor(input_y_f));
	uint plane = batch_channel * push.input_height * push.input_width;
	float sum = 0.0f;
	float weight = 0.0f;
	for (int dy = -1; dy <= 2; ++dy) {
		int sy = clamp(base_y + dy, 0, int(push.input_height) - 1);
		float wy = cubic_weight(input_y_f - float(base_y) - float(dy));
		for (int dx = -1; dx <= 2; ++dx) {
			int sx = clamp(base_x + dx, 0, int(push.input_width) - 1);
			float w = wy * cubic_weight(input_x_f - float(base_x) - float(dx));
			sum += w * load_f32(
				storage_buffers[push.input_index],
				plane + uint(sy) * push.input_width + uint(sx));
			weight += w;
		}
	}
	uint output_index =
		(batch_channel * push.output_height + output_y) * push.output_width + output_x;
	store_f32(
		storage_buffers[push.output_index],
		output_index,
		weight != 0.0f ? sum / weight : 0.0f);
}
