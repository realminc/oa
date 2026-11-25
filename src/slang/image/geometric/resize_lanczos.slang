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

float sinc(float x) {
	if (abs(x) < 1e-5f) return 1.0f;
	float px = 3.14159265358979323846f * x;
	return sin(px) / px;
}

// Lanczos-4 kernel (OpenCV INTER_LANCZOS4).
float lanczos_weight(float t) {
	const float a = 4.0f;
	float x = abs(t);
	if (x >= a) return 0.0f;
	return sinc(x) * sinc(x / a);
}

[kernel_name("resize_lanczos")]
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
	float scale_x = float(push.output_width) / float(push.input_width);
	float scale_y = float(push.output_height) / float(push.input_height);
	float input_x_f = (float(output_x) + 0.5f) / scale_x - 0.5f;
	float input_y_f = (float(output_y) + 0.5f) / scale_y - 0.5f;
	// Widen the filter when downscaling so it covers the projected source cell.
	float widen_x = max(1.0f, 1.0f / scale_x);
	float widen_y = max(1.0f, 1.0f / scale_y);
	uint plane = batch_channel * push.input_height * push.input_width;
	float sum = 0.0f;
	float weight = 0.0f;
	int lo_y = int(ceil(input_y_f - 4.0f * widen_y));
	int hi_y = int(floor(input_y_f + 4.0f * widen_y));
	int lo_x = int(ceil(input_x_f - 4.0f * widen_x));
	int hi_x = int(floor(input_x_f + 4.0f * widen_x));
	for (int sy = lo_y; sy <= hi_y; ++sy) {
		int cy = clamp(sy, 0, int(push.input_height) - 1);
		float wy = lanczos_weight((float(sy) - input_y_f) / widen_y);
		for (int sx = lo_x; sx <= hi_x; ++sx) {
			int cx = clamp(sx, 0, int(push.input_width) - 1);
			float w = wy * lanczos_weight((float(sx) - input_x_f) / widen_x);
			sum += w * load_f32(
				storage_buffers[push.input_index],
				plane + uint(cy) * push.input_width + uint(cx));
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
