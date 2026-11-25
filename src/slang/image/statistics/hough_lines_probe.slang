import attributes;
import storage;

struct PushConstants {
	uint input_index, lines_index, segments_index;
	uint plane_count, max_lines;
	uint height, width;
	float edge_threshold, min_line_length;
	uint max_line_gap;
};
[[vk::push_constant]] PushConstants push;
[[vk::binding(0, 0)]] RWByteAddressBuffer storage_buffers[];

[kernel_name("hough_lines_probe")]
[domain("image")]
[variant("nchw_or_chw")]
[dtype("f32")]
[status("experimental")]
[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 thread_id : SV_DispatchThreadID) {
	uint slot = thread_id.x;
	uint total = push.plane_count * push.max_lines;
	if (slot >= total) return;
	uint plane = slot / push.max_lines;
	uint line = slot % push.max_lines;
	RWByteAddressBuffer lines = storage_buffers[push.lines_index];
	uint in_row = plane * push.max_lines * 3u + line * 3u;
	float rho = load_f32(lines, in_row);
	float theta_deg = load_f32(lines, in_row + 1u);
	float votes = load_f32(lines, in_row + 2u);
	RWByteAddressBuffer segments = storage_buffers[push.segments_index];
	uint out_base = slot * 4u;
	if (votes <= 0.0) {
		for (uint k = 0; k < 4u; ++k) store_f32(segments, out_base + k, 0.0);
		return;
	}
	float theta = theta_deg * 3.14159265 / 180.0;
	float nx = cos(theta), ny = sin(theta);
	float dx = ny, dy = -nx;
	float fx = rho * nx, fy = rho * ny;
	int span = int(ceil(sqrt(float(push.width * push.width + push.height * push.height))));
	RWByteAddressBuffer input = storage_buffers[push.input_index];
	uint plane_base = plane * push.height * push.width;
	float best_len = 0.0;
	float bx1 = 0.0, by1 = 0.0, bx2 = 0.0, by2 = 0.0;
	bool run_active = false;
	int run_start = 0, last_edge = 0;
	float sx = 0.0, sy = 0.0, ex = 0.0, ey = 0.0;
	uint gap = 0u;
	for (int t = -span; t <= span; ++t) {
		int px = int(floor(fx + float(t) * dx + 0.5));
		int py = int(floor(fy + float(t) * dy + 0.5));
		bool edge = false;
		if (px >= 0 && py >= 0 && px < int(push.width) && py < int(push.height)) {
			float v = load_f32(input, plane_base + uint(py) * push.width + uint(px));
			edge = v > push.edge_threshold;
		}
		if (edge) {
			if (!run_active) {
				run_active = true;
				run_start = t;
				sx = float(px);
				sy = float(py);
			}
			last_edge = t;
			ex = float(px);
			ey = float(py);
			gap = 0u;
		} else if (run_active) {
			gap += 1u;
			if (gap > push.max_line_gap) {
				run_active = false;
				float len = float(last_edge - run_start) + 1.0;
				if (len >= push.min_line_length && len > best_len) {
					best_len = len;
					bx1 = sx; by1 = sy; bx2 = ex; by2 = ey;
				}
			}
		}
	}
	if (run_active) {
		float len = float(last_edge - run_start) + 1.0;
		if (len >= push.min_line_length && len > best_len) {
			best_len = len;
			bx1 = sx; by1 = sy; bx2 = ex; by2 = ey;
		}
	}
	store_f32(segments, out_base, bx1);
	store_f32(segments, out_base + 1u, by1);
	store_f32(segments, out_base + 2u, bx2);
	store_f32(segments, out_base + 3u, by2);
}
