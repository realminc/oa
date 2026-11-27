import attributes;
import storage;

struct PushConstants {
	uint input_index, output_index;
	uint batch_size, channels, height, width;
	uint poly_n;
	float poly_sigma;
};
[[vk::push_constant]] PushConstants push;
[[vk::binding(0, 0)]] RWByteAddressBuffer storage_buffers[];

int reflect101(int coordinate, int extent) {
	if (extent <= 1) return 0;
	int period = 2 * (extent - 1);
	int folded = abs(coordinate) % period;
	return folded < extent ? folded : period - folded;
}

// Horizontal pass of Farneback polynomial expansion. For each pixel it
// accumulates the horizontal moments h0/h1/h2 of the vertical moment map
// and emits the five separable least-squares coefficients of
//   I(x, y) ~= a x^2 + 2 d xy + e y^2 + b0 x + b1 y + c
// as [a, d, e, b0, b1] in the output's five channels. The symmetric
// gaussian window makes the odd-order weight sums vanish, so the 3x3
// normal equations factor into scalar moments S0, S2, S4.
[kernel_name("flow_expand_h")]
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
	RWByteAddressBuffer input_buffer = storage_buffers[push.input_index];
	RWByteAddressBuffer output_buffer = storage_buffers[push.output_index];
	uint plane_size = push.height * push.width;
	uint input_base = plane * 3u * plane_size;
	uint output_base = plane * 5u * plane_size;
	int n = int(push.poly_n);
	float variance = 2.0f * push.poly_sigma * push.poly_sigma;
	float h0[3], h1[3], h2[3];
	for (uint channel = 0u; channel < 3u; ++channel) {
		h0[channel] = 0.0f;
		h1[channel] = 0.0f;
		h2[channel] = 0.0f;
	}
	for (int dx = -n; dx <= n; ++dx) {
		int sx = reflect101(int(x) + dx, int(push.width));
		float weight = exp(-float(dx * dx) / variance);
		for (uint channel = 0u; channel < 3u; ++channel) {
			float value = load_f32(input_buffer,
				input_base + channel * plane_size + y * push.width + uint(sx));
			h0[channel] += weight * value;
			h1[channel] += weight * float(dx) * value;
			h2[channel] += weight * float(dx * dx) * value;
		}
	}
	// 1D weight moments (identical on both axes for the symmetric window).
	float s0 = 0.0f, s2 = 0.0f, s4 = 0.0f;
	for (int k = -n; k <= n; ++k) {
		float weight = exp(-float(k * k) / variance);
		s0 += weight;
		s2 += weight * float(k * k);
		s4 += weight * float(k * k * k * k);
	}
	float determinant = s0 * s4 - s2 * s2;
	float inv_d = determinant != 0.0f ? 1.0f / determinant : 0.0f;
	float inv_s0 = s0 != 0.0f ? 1.0f / s0 : 0.0f;
	float inv_s2 = s2 != 0.0f ? 1.0f / s2 : 0.0f;
	// a = quadratic coefficient in x of the y-mean m0.
	float a = (s0 * h2[0] - s2 * h0[0]) * inv_d * inv_s0;
	// e = quadratic coefficient in y, evaluated from the m2 moment map.
	float e = (s0 * h0[2] - s2 * h0[0]) * inv_d * inv_s0;
	// d = joint xy coefficient; stored halved per the 2A01 convention.
	float d = h1[1] * inv_s2 * inv_s2 * 0.5f;
	// b0/b1 = linear coefficients.
	float b0 = h1[0] * inv_s2 * inv_s0;
	float b1 = h0[1] * inv_s0 * inv_s2;
	uint index = output_base + y * push.width + x;
	store_f32(output_buffer, index, a);
	store_f32(output_buffer, index + plane_size, d);
	store_f32(output_buffer, index + 2u * plane_size, e);
	store_f32(output_buffer, index + 3u * plane_size, b0);
	store_f32(output_buffer, index + 4u * plane_size, b1);
}
