import attributes;
import storage;

struct PushConstants {
	uint previous_index, next_index, points_index, state_index;
	uint status_index, error_index;
	uint batch_size, channels, count;
	uint height, width, window, iterations;
	uint initialize, finalize;
	float scale, min_eigen;
};
[[vk::push_constant]] PushConstants push;
[[vk::binding(0, 0)]] RWByteAddressBuffer storage_buffers[];

float sample_at(RWByteAddressBuffer buffer, uint base, uint width, uint height, float y, float x) {
	if (x < 0.0f || y < 0.0f || x > float(width - 1u) || y > float(height - 1u)) return 0.0f;
	int x0 = int(floor(x));
	int y0 = int(floor(y));
	int x1 = min(x0 + 1, int(width) - 1);
	int y1 = min(y0 + 1, int(height) - 1);
	float fx = x - float(x0);
	float fy = y - float(y0);
	return lerp(
		lerp(load_f32(buffer, base + uint(y0) * width + uint(x0)),
			load_f32(buffer, base + uint(y0) * width + uint(x1)), fx),
		lerp(load_f32(buffer, base + uint(y1) * width + uint(x0)),
			load_f32(buffer, base + uint(y1) * width + uint(x1)), fx),
		fy);
}

// Sparse Lucas-Kanade refinement at one pyramid level. One invocation per
// (plane, tracked point); the shader carries the whole iteration loop so a
// dispatch per level is recorded from coarse to fine. `state` holds the
// current next-frame estimate in level-0 pixel coordinates.
[kernel_name("flow_lk_refine")]
[domain("image")]
[variant("nchw_or_chw")]
[dtype("f32")]
[layout("nchw_or_chw")]
[status("experimental")]
[shader("compute")]
[numthreads(64, 1, 1)]
void main(uint3 thread_id : SV_DispatchThreadID) {
	uint index = thread_id.x;
	uint plane = index / push.count;
	uint point_index = index - plane * push.count;
	if (plane >= push.batch_size * push.channels) return;
	uint plane_size = push.height * push.width;
	uint previous_base = plane * plane_size;
	uint next_base = plane * plane_size;
	RWByteAddressBuffer previous_buffer = storage_buffers[push.previous_index];
	RWByteAddressBuffer next_buffer = storage_buffers[push.next_index];
	RWByteAddressBuffer points_buffer = storage_buffers[push.points_index];
	RWByteAddressBuffer state_buffer = storage_buffers[push.state_index];
	uint state_base = (plane * push.count + point_index) * 2u;
	float px = load_f32(points_buffer, point_index * 2u) * push.scale;
	float py = load_f32(points_buffer, point_index * 2u + 1u) * push.scale;
	float ex = push.initialize != 0u
		? px
		: load_f32(state_buffer, state_base) * push.scale;
	float ey = push.initialize != 0u
		? py
		: load_f32(state_buffer, state_base + 1u) * push.scale;
	uint w = push.window;
	uint side = 2u * w + 1u;
	uint win_count = side * side;
	float margin = float(w) + 1.0f;
	bool inside = px >= margin && py >= margin
		&& px <= float(push.width) - margin && py <= float(push.height) - margin;
	// Spatial gradient normal matrix over the window at the source point.
	float g00 = 0.0f, g01 = 0.0f, g11 = 0.0f;
	if (inside) {
		for (int oy = -int(w); oy <= int(w); ++oy) {
			for (int ox = -int(w); ox <= int(w); ++ox) {
				float y = py + float(oy);
				float x = px + float(ox);
				float ix = (sample_at(previous_buffer, previous_base, push.width, push.height, y, x + 1.0f)
					- sample_at(previous_buffer, previous_base, push.width, push.height, y, x - 1.0f)) * 0.5f;
				float iy = (sample_at(previous_buffer, previous_base, push.width, push.height, y + 1.0f, x)
					- sample_at(previous_buffer, previous_base, push.width, push.height, y - 1.0f, x)) * 0.5f;
				g00 += ix * ix;
				g01 += ix * iy;
				g11 += iy * iy;
			}
		}
	}
	float trace = g00 + g11;
	float disc = max(trace * trace - 4.0f * (g00 * g11 - g01 * g01), 0.0f);
	float min_eigen = 0.5f * (trace - sqrt(disc));
	bool well_posed = inside && min_eigen >= push.min_eigen;
	float dx = 0.0f, dy = 0.0f;
	if (well_posed) {
		float det = g00 * g11 - g01 * g01;
		for (uint iteration = 0u; iteration < push.iterations; ++iteration) {
			float bx = 0.0f, by = 0.0f;
			for (int oy = -int(w); oy <= int(w); ++oy) {
				for (int ox = -int(w); ox <= int(w); ++ox) {
					float y = py + float(oy);
					float x = px + float(ox);
					float ix = (sample_at(previous_buffer, previous_base, push.width, push.height, y, x + 1.0f)
						- sample_at(previous_buffer, previous_base, push.width, push.height, y, x - 1.0f)) * 0.5f;
					float iy = (sample_at(previous_buffer, previous_base, push.width, push.height, y + 1.0f, x)
						- sample_at(previous_buffer, previous_base, push.width, push.height, y - 1.0f, x)) * 0.5f;
					float difference = sample_at(previous_buffer, previous_base, push.width, push.height, y, x)
						- sample_at(next_buffer, next_base, push.width, push.height, ey + dy + float(oy), ex + dx + float(ox));
					bx += ix * difference;
					by += iy * difference;
				}
			}
			float nx = (g11 * bx - g01 * by) / det;
			float ny = (g00 * by - g01 * bx) / det;
			dx += nx;
			dy += ny;
			if (abs(nx) + abs(ny) < 0.01f) break;
		}
	}
	ex += dx;
	ey += dy;
	store_f32(state_buffer, state_base, ex / push.scale);
	store_f32(state_buffer, state_base + 1u, ey / push.scale);
	if (push.finalize == 0u) return;
	float status = 0.0f;
	float error = 0.0f;
	if (well_posed && ex >= margin && ey >= margin
		&& ex <= float(push.width) - margin && ey <= float(push.height) - margin) {
		status = 1.0f;
		float sum = 0.0f;
		for (int oy = -int(w); oy <= int(w); ++oy) {
			for (int ox = -int(w); ox <= int(w); ++ox) {
				float difference = sample_at(previous_buffer, previous_base, push.width, push.height, py + float(oy), px + float(ox))
					- sample_at(next_buffer, next_base, push.width, push.height, ey + float(oy), ex + float(ox));
				sum += difference * difference;
			}
		}
		error = sqrt(sum / float(win_count));
	}
	RWByteAddressBuffer status_buffer = storage_buffers[push.status_index];
	RWByteAddressBuffer error_buffer = storage_buffers[push.error_index];
	uint out_index = plane * push.count + point_index;
	store_f32(status_buffer, out_index, status);
	store_f32(error_buffer, out_index, error);
}
