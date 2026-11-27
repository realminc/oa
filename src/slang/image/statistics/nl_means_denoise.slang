import attributes;
import storage;

struct PushConstants {
	uint input_index, output_index;
	uint batch_size, channels, height, width;
	uint template_radius, search_radius;
	float h_sqr;
};
[[vk::push_constant]] PushConstants push;
[[vk::binding(0, 0)]] RWByteAddressBuffer storage_buffers[];

// fastNlMeansDenoising pass. Each invocation owns one (x, y, plane)
// output pixel and averages samples in the (2 * search_radius + 1)^2
// search window with weights exp(-d / h_sqr), where d is the patch
// SSD over the (2 * template_radius + 1)^2 template normalized by
// its pixel count. Out-of-bounds samples clamp to the border row or
// column; the center weight equals the largest neighbor weight, as
// OpenCV does, so an almost-uniform region keeps weight one instead
// of amplifying its own sample.
[kernel_name("nl_means_denoise")]
[domain("image")]
[variant("nchw_or_chw")]
[dtype("f32")]
[layout("nchw_or_chw")]
[status("experimental")]
[shader("compute")]
[numthreads(16, 16, 1)]
void main(uint3 thread_id : SV_DispatchThreadID) {
	uint x = thread_id.x, y = thread_id.y, plane = thread_id.z;
	if (x >= push.width || y >= push.height) return;
	uint plane_count = push.batch_size * push.channels;
	if (plane >= plane_count) return;
	RWByteAddressBuffer input_buffer = storage_buffers[push.input_index];
	RWByteAddressBuffer output_buffer = storage_buffers[push.output_index];
	uint plane_size = push.height * push.width;
	uint base = plane * plane_size;
	int radius = int(push.template_radius);
	int span = int(push.search_radius);
	int max_x = int(push.width) - 1;
	int max_y = int(push.height) - 1;
	float template_count = float((2 * radius + 1) * (2 * radius + 1));
	float weighted = 0.0f;
	float total = 0.0f;
	float max_weight = 0.0f;
	for (int dy = -span; dy <= span; ++dy) {
		int ny = clamp(int(y) + dy, 0, max_y);
		for (int dx = -span; dx <= span; ++dx) {
			int nx = clamp(int(x) + dx, 0, max_x);
			float ssd = 0.0f;
			for (int ty = -radius; ty <= radius; ++ty) {
				int py = clamp(int(y) + ty, 0, max_y);
				int qy = clamp(ny + ty, 0, max_y);
				for (int tx = -radius; tx <= radius; ++tx) {
					int px = clamp(int(x) + tx, 0, max_x);
					int qx = clamp(nx + tx, 0, max_x);
					float diff = load_f32(input_buffer, base + uint(py) * push.width + uint(px))
						- load_f32(input_buffer, base + uint(qy) * push.width + uint(qx));
					ssd += diff * diff;
				}
			}
			float weight = exp(-(ssd / template_count) / push.h_sqr);
			if (dx != 0 || dy != 0) max_weight = max(max_weight, weight);
			weighted += weight * load_f32(input_buffer, base + uint(ny) * push.width + uint(nx));
			total += weight;
		}
	}
	uint center_index = base + y * push.width + x;
	float center_value = load_f32(input_buffer, center_index);
	float center_weight = max(max_weight, 0.0001f);
	weighted += (center_weight - 1.0f) * center_value;
	total += center_weight - 1.0f;
	store_f32(output_buffer, center_index, weighted / max(total, 0.0001f));
}
