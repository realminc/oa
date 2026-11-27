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

// Vertical pass of Farneback polynomial expansion: per-pixel weighted
// moments m0, m1, m2 of the source column over a (2n+1)-tap gaussian
// window. The horizontal pass combines them into the five polynomial
// coefficients consumed by the update stage.
[kernel_name("flow_expand_v")]
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
	uint input_base = plane * plane_size;
	uint output_base = plane * 3u * plane_size;
	int n = int(push.poly_n);
	float variance = 2.0f * push.poly_sigma * push.poly_sigma;
	float m0 = 0.0f, m1 = 0.0f, m2 = 0.0f;
	for (int dy = -n; dy <= n; ++dy) {
		int sy = reflect101(int(y) + dy, int(push.height));
		float weight = exp(-float(dy * dy) / variance);
		float value = load_f32(input_buffer, input_base + uint(sy) * push.width + x);
		m0 += weight * value;
		m1 += weight * float(dy) * value;
		m2 += weight * float(dy * dy) * value;
	}
	uint index = output_base + y * push.width + x;
	store_f32(output_buffer, index, m0);
	store_f32(output_buffer, index + plane_size, m1);
	store_f32(output_buffer, index + 2u * plane_size, m2);
}
