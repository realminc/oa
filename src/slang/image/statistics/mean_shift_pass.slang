import attributes;
import storage;

struct PushConstants {
	uint input_index, seed_index, output_index;
	uint batch_size, channels, height, width;
	uint seed_width, seed_height;
	uint spatial_radius, max_iterations, init_mode;
	float range_radius_sqr;
};
[[vk::push_constant]] PushConstants push;
[[vk::binding(0, 0)]] RWByteAddressBuffer storage_buffers[];

// pyrMeanShiftFiltering pass. Each invocation owns one (x, y, plane)
// pixel and iterates its own mean-shift trajectory entirely in
// registers: from the current position, average every window sample
// whose channel-vector distance to the current color stays below the
// range radius, then adopt that mean position and color until the
// spatial shift drops under one pixel or max_iterations is reached.
// init_mode = 1 seeds the starting color from an upsampled coarser
// pyramid level; the position always starts at the pixel itself.
[kernel_name("mean_shift_pass")]
[domain("image")]
[variant("nchw_or_chw")]
[dtype("f32")]
[layout("nchw_or_chw")]
[status("experimental")]
[shader("compute")]
[numthreads(16, 16, 1)]
void main(uint3 thread_id : SV_DispatchThreadID) {
	uint x = thread_id.x, y = thread_id.y, plane = thread_id.z;
	if (x >= push.width || y >= push.height || plane >= push.batch_size) return;
	RWByteAddressBuffer input_buffer = storage_buffers[push.input_index];
	RWByteAddressBuffer output_buffer = storage_buffers[push.output_index];
	uint plane_size = push.height * push.width;
	uint channel_stride = plane_size;
	uint plane_base = plane * push.channels * plane_size;
	uint center = y * push.width + x;
	float pos_x = float(x), pos_y = float(y);
	float color[4];
	RWByteAddressBuffer seed_buffer = storage_buffers[push.seed_index];
	uint seed_plane_size = push.seed_width * push.seed_height;
	uint seed_base = plane * push.channels * seed_plane_size
		+ min(y / 2u, push.seed_height - 1u) * push.seed_width
		+ min(x / 2u, push.seed_width - 1u);
	for (uint c = 0; c < push.channels && c < 4u; ++c) {
		color[c] = (push.init_mode == 1)
			? load_f32(seed_buffer, seed_base + c * seed_plane_size)
			: load_f32(input_buffer, plane_base + c * channel_stride + center);
	}
	int radius = int(push.spatial_radius);
	int max_x = int(push.width) - 1;
	int max_y = int(push.height) - 1;
	for (uint iteration = 0; iteration < push.max_iterations; ++iteration) {
		int cx = int(pos_x + 0.5f);
		int cy = int(pos_y + 0.5f);
		float sum_x = 0.0f, sum_y = 0.0f, count = 0.0f;
		float sum_color[4];
		for (uint c = 0; c < 4u; ++c) sum_color[c] = 0.0f;
		for (int j = -radius; j <= radius; ++j) {
			int sy = clamp(cy + j, 0, max_y);
			for (int i = -radius; i <= radius; ++i) {
				int sx = clamp(cx + i, 0, max_x);
				float distance = 0.0f;
				float sample[4];
				uint sample_index = plane_base + uint(sy) * push.width + uint(sx);
				for (uint c = 0; c < push.channels && c < 4u; ++c) {
					sample[c] = load_f32(input_buffer, sample_index + c * channel_stride);
					float diff = sample[c] - color[c];
					distance += diff * diff;
				}
				if (distance >= push.range_radius_sqr) continue;
				sum_x += float(sx);
				sum_y += float(sy);
				for (uint c = 0; c < push.channels && c < 4u; ++c)
					sum_color[c] += sample[c];
				count += 1.0f;
			}
		}
		if (count < 0.5f) break;
		float new_x = sum_x / count;
		float new_y = sum_y / count;
		float shift = (new_x - pos_x) * (new_x - pos_x)
			+ (new_y - pos_y) * (new_y - pos_y);
		pos_x = new_x;
		pos_y = new_y;
		for (uint c = 0; c < push.channels && c < 4u; ++c)
			color[c] = sum_color[c] / count;
		if (shift < 1.0f) break;
	}
	for (uint c = 0; c < push.channels && c < 4u; ++c)
		store_f32(output_buffer, plane_base + c * channel_stride + center, color[c]);
}
