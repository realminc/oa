import attributes;
import storage;

struct PushConstants {
	uint left_index, right_index, output_index;
	uint batch_size, channels, height, width;
	uint num_disparities, block_radius;
};
[[vk::push_constant]] PushConstants push;
[[vk::binding(0, 0)]] RWByteAddressBuffer storage_buffers[];

// StereoBM matching-cost pass. Each invocation owns one (x, y,
// plane * num_disparities + d) element of the SAD cost volume and
// accumulates |L(x + i, y + j) - R(x + i - d, y + j)| over the
// (2 * block_radius + 1)^2 matching window. Out-of-bounds samples
// clamp to the border row/column; the winner-take-all pass owns
// the valid-region mask.
[kernel_name("stereo_sad")]
[domain("image")]
[variant("nchw_or_chw")]
[dtype("f32")]
[layout("nchw_or_chw")]
[status("experimental")]
[shader("compute")]
[numthreads(16, 16, 1)]
void main(uint3 thread_id : SV_DispatchThreadID) {
	uint x = thread_id.x, y = thread_id.y, slot = thread_id.z;
	uint plane_count = push.batch_size * push.channels;
	uint disp_count = plane_count * push.num_disparities;
	if (x >= push.width || y >= push.height || slot >= disp_count) return;
	uint plane = slot / push.num_disparities;
	uint disparity = slot % push.num_disparities;
	RWByteAddressBuffer left_buffer = storage_buffers[push.left_index];
	RWByteAddressBuffer right_buffer = storage_buffers[push.right_index];
	RWByteAddressBuffer output_buffer = storage_buffers[push.output_index];
	uint plane_size = push.height * push.width;
	uint image_base = plane * plane_size;
	int radius = int(push.block_radius);
	int max_x = int(push.width) - 1;
	int max_y = int(push.height) - 1;
	float sad = 0.0f;
	for (int j = -radius; j <= radius; ++j) {
		int sy = clamp(int(y) + j, 0, max_y);
		for (int i = -radius; i <= radius; ++i) {
			int left_x = clamp(int(x) + i, 0, max_x);
			int right_x = clamp(int(x) + i - int(disparity), 0, max_x);
			float left_value = load_f32(left_buffer, image_base + uint(sy) * push.width + uint(left_x));
			float right_value = load_f32(right_buffer, image_base + uint(sy) * push.width + uint(right_x));
			sad += abs(left_value - right_value);
		}
	}
	uint out_index = slot * plane_size + y * push.width + x;
	store_f32(output_buffer, out_index, sad);
}
