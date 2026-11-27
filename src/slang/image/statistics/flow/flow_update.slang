import attributes;
import storage;

struct PushConstants {
	uint poly1_index, poly2_index, flow_in_index, flow_out_index;
	uint batch_size, channels, height, width;
	uint in_height, in_width, win, init_mode;
};
[[vk::push_constant]] PushConstants push;
[[vk::binding(0, 0)]] RWByteAddressBuffer storage_buffers[];

float sample_coeff(RWByteAddressBuffer buffer, uint channel_base, uint width, uint height, float y, float x) {
	x = clamp(x, 0.0f, float(width) - 1.0f);
	y = clamp(y, 0.0f, float(height) - 1.0f);
	int x0 = int(floor(x));
	int y0 = int(floor(y));
	int x1 = min(x0 + 1, int(width) - 1);
	int y1 = min(y0 + 1, int(height) - 1);
	float fx = x - float(x0);
	float fy = y - float(y0);
	return lerp(
		lerp(load_f32(buffer, channel_base + uint(y0) * width + uint(x0)),
			load_f32(buffer, channel_base + uint(y0) * width + uint(x1)), fx),
		lerp(load_f32(buffer, channel_base + uint(y1) * width + uint(x0)),
			load_f32(buffer, channel_base + uint(y1) * width + uint(x1)), fx),
		fy);
}

// One Jacobi iteration of the dense Farneback update at one pyramid level.
// For each pixel the kernel accumulates the normal-equation terms
// G = A^T A and h = A^T r over a (2*win+1)^2 neighbourhood, where each
// window element contributes its averaged quadratic form
// A = (A1(y) + A2(y + d_y)) / 2 and its flow residual
// r = A d_y + (b1(y) - b2(y + d_y)) / 2, then solves the 2x2 system for
// the new displacement. `init_mode` selects the incoming field:
// 0 = zero (coarsest level, first iteration), 1 = upsample the coarser
// parent level by 2x, 2 = read the same-level previous iterate.
[kernel_name("flow_update")]
[domain("image")]
[variant("nchw_or_chw")]
[dtype("f32")]
[layout("nchw_or_chw")]
[status("experimental")]
[shader("compute")]
[numthreads(16, 16, 1)]
void main(uint3 thread_id : SV_DispatchThreadID) {
	uint x = thread_id.x, y = thread_id.y, plane = thread_id.z;
	if (x >= push.width || y >= push.height || plane >= push.batch_size * push.channels) return;
	RWByteAddressBuffer poly1_buffer = storage_buffers[push.poly1_index];
	RWByteAddressBuffer poly2_buffer = storage_buffers[push.poly2_index];
	RWByteAddressBuffer flow_in_buffer = storage_buffers[push.flow_in_index];
	RWByteAddressBuffer flow_out_buffer = storage_buffers[push.flow_out_index];
	uint plane_size = push.height * push.width;
	uint poly_base = plane * 5u * plane_size;
	uint in_plane_size = push.in_height * push.in_width;
	uint in_base = plane * 2u * in_plane_size;
	uint w = push.win;
	int x0 = max(int(x) - int(w), 0);
	int x1 = min(int(x) + int(w), int(push.width) - 1);
	int y0 = max(int(y) - int(w), 0);
	int y1 = min(int(y) + int(w), int(push.height) - 1);
	float s00 = 0.0f, s01 = 0.0f, s11 = 0.0f, sx = 0.0f, sy = 0.0f;
	for (int wy = y0; wy <= y1; ++wy) {
		for (int wx = x0; wx <= x1; ++wx) {
			// Incoming displacement at this window element.
			float du = 0.0f, dv = 0.0f;
			if (push.init_mode == 1u) {
				float px = float(wx) * 0.5f;
				float py = float(wy) * 0.5f;
				du = 2.0f * sample_coeff(flow_in_buffer, in_base, push.in_width, push.in_height, py, px);
				dv = 2.0f * sample_coeff(flow_in_buffer, in_base + in_plane_size, push.in_width, push.in_height, py, px);
			} else if (push.init_mode == 2u) {
				uint in_index = in_base + uint(wy) * push.width + uint(wx);
				du = load_f32(flow_in_buffer, in_index);
				dv = load_f32(flow_in_buffer, in_index + in_plane_size);
			}
			uint p_index = poly_base + uint(wy) * push.width + uint(wx);
			// Averaged quadratic form A = (A1(y) + A2(y + d_y)) / 2.
			float sx2 = float(wx) + du;
			float sy2 = float(wy) + dv;
			float a00 = 0.5f * (load_f32(poly1_buffer, p_index)
				+ sample_coeff(poly2_buffer, poly_base, push.width, push.height, sy2, sx2));
			float a01 = 0.5f * (load_f32(poly1_buffer, p_index + plane_size)
				+ sample_coeff(poly2_buffer, poly_base + plane_size, push.width, push.height, sy2, sx2));
			float a11 = 0.5f * (load_f32(poly1_buffer, p_index + 2u * plane_size)
				+ sample_coeff(poly2_buffer, poly_base + 2u * plane_size, push.width, push.height, sy2, sx2));
			// Flow residual r = A d_y + (b1(y) - b2(y + d_y)) / 2.
			float dbx = 0.5f * (load_f32(poly1_buffer, p_index + 3u * plane_size)
				- sample_coeff(poly2_buffer, poly_base + 3u * plane_size, push.width, push.height, sy2, sx2));
			float dby = 0.5f * (load_f32(poly1_buffer, p_index + 4u * plane_size)
				- sample_coeff(poly2_buffer, poly_base + 4u * plane_size, push.width, push.height, sy2, sx2));
			float rx = a00 * du + a01 * dv + dbx;
			float ry = a01 * du + a11 * dv + dby;
			// Normal-equation terms G = A^T A, h = A^T r for symmetric A.
			s00 += a00 * a00 + a01 * a01;
			s01 += (a00 + a11) * a01;
			s11 += a01 * a01 + a11 * a11;
			sx += a00 * rx + a01 * ry;
			sy += a01 * rx + a11 * ry;
		}
	}
	// Solve the 2x2 normal system. The diagonal regularizer mirrors
	// OpenCV's degenerate-neighbourhood damping but scales with the
	// accumulated curvature trace so it stays meaningful on unit-range
	// imagery where an absolute 1e-3 would swamp the determinant.
	float eps = 1e-3f * 0.5f * (s00 + s11) + 1e-9f;
	float det = (s00 + eps) * (s11 + eps) - s01 * s01;
	float inv_det = 1.0f / det;
	float out_x = (s11 * sx - s01 * sy) * inv_det;
	float out_y = (s00 * sy - s01 * sx) * inv_det;
	uint out_base = plane * 2u * plane_size;
	uint out_index = out_base + y * push.width + x;
	store_f32(flow_out_buffer, out_index, out_x);
	store_f32(flow_out_buffer, out_index + plane_size, out_y);
}
