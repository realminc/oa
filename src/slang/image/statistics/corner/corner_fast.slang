import attributes;
import storage;

struct PushConstants {
	uint input_index, output_index;
	uint batch_size, channels, height, width;
	uint arc_length;
};
[[vk::push_constant]] PushConstants push;
[[vk::binding(0, 0)]] RWByteAddressBuffer storage_buffers[];

// FAST score on the radius-3 Bresenham circle. Score is the largest
// deviation t where arc_length contiguous samples all exceed center + t
// or all fall below center - t.
[kernel_name("corner_fast")]
[domain("image")]
[variant("nchw_or_chw")]
[dtype("f32")]
[status("experimental")]
[shader("compute")]
[numthreads(16, 16, 1)]
void main(uint3 thread_id : SV_DispatchThreadID) {
	uint x = thread_id.x;
	uint y = thread_id.y;
	uint plane = thread_id.z;
	if (x >= push.width || y >= push.height || plane >= push.batch_size * push.channels) return;
	RWByteAddressBuffer input_buffer = storage_buffers[push.input_index];
	uint plane_size = push.height * push.width;
	uint base = plane * plane_size;
	uint index = base + y * push.width + x;
	float score = 0.0;
	if (x >= 3 && y >= 3 && x + 3 < push.width && y + 3 < push.height) {
		static const int circle_x[16] = {0, 1, 2, 3, 3, 3, 2, 1, 0, -1, -2, -3, -3, -3, -2, -1};
		static const int circle_y[16] = {-3, -3, -2, -1, 0, 1, 2, 3, 3, 3, 2, 1, 0, -1, -2, -3};
		float center = load_f32(input_buffer, index);
		float diffs[16];
		for (uint i = 0u; i < 16u; i++) {
			uint sx = uint(int(x) + circle_x[i]);
			uint sy = uint(int(y) + circle_y[i]);
			diffs[i] = load_f32(input_buffer, base + sy * push.width + sx) - center;
		}
		for (uint start = 0u; start < 16u; start++) {
			float bright = 1.0e30;
			float dark = 1.0e30;
			for (uint j = 0u; j < push.arc_length; j++) {
				float d = diffs[(start + j) % 16u];
				bright = min(bright, d);
				dark = min(dark, -d);
			}
			score = max(score, max(bright, dark));
		}
		score = max(score, 0.0);
	}
	store_f32(storage_buffers[push.output_index], index, score);
}
