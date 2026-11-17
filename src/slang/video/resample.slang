// Per-plane u8 resampling for native-YUV geometry operations.
//
// One kernel serves crop, resize, flip, rotate90, pad, letterbox and stack:
// the host computes the destination->source mapping per plane, so luma,
// planar chroma (element stride 1) and interleaved NV12 chroma pairs
// (element stride 2) all run through the same code. A swapped mapping flag
// transposes for rotate90/rotate270. Pixels outside the valid destination
// window receive the plane's fill code.

import attributes;
import storage;

struct PushConstants {
	uint src_index;
	uint dst_index;
	uint src_width;
	uint src_height;
	uint src_elem_stride;
	uint dst_elem_stride;
	uint src_component;
	uint dst_component;
	uint elem_count;
	uint dst_width;
	uint dst_height;
	uint dst_row_stride;
	uint dst_off_x;
	uint dst_off_y;
	uint swap_xy;
	uint interp_mode;
	uint fill_value;
	float src_off_x;
	float src_off_y;
	float src_scale_x;
	float src_scale_y;
	float valid_min_x;
	float valid_min_y;
	float valid_max_x;
	float valid_max_y;
};
[[vk::push_constant]] PushConstants push;

[[vk::binding(0, 0)]] RWByteAddressBuffer storage_buffers[];

uint load_element(RWByteAddressBuffer plane, uint x, uint y, uint component) {
	return load_u8(plane, (y * push.src_width + x) * push.src_elem_stride + component);
}

float sample_element(float x, float y, uint component) {
	uint width = push.src_width;
	uint height = push.src_height;
	if (push.interp_mode != 0u) {
		int xi = clamp(int(floor(x + 0.5f)), 0, int(width) - 1);
		int yi = clamp(int(floor(y + 0.5f)), 0, int(height) - 1);
		return float(load_element(
			storage_buffers[push.src_index],
			uint(xi),
			uint(yi),
			component
		));
	}
	float xc = clamp(x, 0.0f, float(width) - 1.0f);
	float yc = clamp(y, 0.0f, float(height) - 1.0f);
	uint x0 = min(uint(floor(xc)), width - 1u);
	uint y0 = min(uint(floor(yc)), height - 1u);
	uint x1 = min(x0 + 1u, width - 1u);
	uint y1 = min(y0 + 1u, height - 1u);
	float fx = xc - float(x0);
	float fy = yc - float(y0);
	RWByteAddressBuffer plane = storage_buffers[push.src_index];
	float top_left = float(load_element(plane, x0, y0, component));
	float top_right = float(load_element(plane, x1, y0, component));
	float bottom_left = float(load_element(plane, x0, y1, component));
	float bottom_right = float(load_element(plane, x1, y1, component));
	float top = lerp(top_left, top_right, fx);
	float bottom = lerp(bottom_left, bottom_right, fx);
	return lerp(top, bottom, fy);
}

[kernel_name("resample")]
[domain("video")]
[variant("plane_u8")]
[dtype("u8")]
[status("experimental")]
[shader("compute")]
[numthreads(16, 16, 1)]
void main(uint3 thread_id : SV_DispatchThreadID) {
	uint out_x = thread_id.x;
	uint out_y = thread_id.y;
	if (out_x >= push.dst_width || out_y >= push.dst_height) return;

	float px = float(out_x) + 0.5f;
	float py = float(out_y) + 0.5f;
	bool inside = px >= push.valid_min_x && px < push.valid_max_x && py >= push.valid_min_y
		&& py < push.valid_max_y;

	float sx;
	float sy;
	if (push.swap_xy != 0u) {
		sx = py * push.src_scale_y + push.src_off_y;
		sy = px * push.src_scale_x + push.src_off_x;
	} else {
		sx = px * push.src_scale_x + push.src_off_x;
		sy = py * push.src_scale_y + push.src_off_y;
	}

	uint dst_base = (out_y + push.dst_off_y) * push.dst_row_stride + out_x + push.dst_off_x;
	for (uint component = 0u; component < push.elem_count; component += 1u) {
		uint code = inside
			? uint(round(clamp(sample_element(sx, sy, push.src_component + component), 0.0f, 255.0f)))
			: push.fill_value;
		store_u8(
			storage_buffers[push.dst_index],
			dst_base * push.dst_elem_stride + push.dst_component + component,
			code
		);
	}
}
