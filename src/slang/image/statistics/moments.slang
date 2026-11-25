import attributes;
import storage;

struct PushConstants {
	uint input_index, output_index;
	uint batch_size, channels, height, width;
};
[[vk::push_constant]] PushConstants push;
[[vk::binding(0, 0)]] RWByteAddressBuffer storage_buffers[];

[kernel_name("moments")]
[domain("image")]
[variant("serial_plane")]
[dtype("f32")]
[status("experimental")]
[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 thread_id : SV_DispatchThreadID) {
	uint plane = thread_id.x;
	if (plane >= push.batch_size * push.channels) return;
	RWByteAddressBuffer input_buffer = storage_buffers[push.input_index];
	uint base = plane * push.height * push.width;
	float m00 = 0.0, m10 = 0.0, m01 = 0.0, m20 = 0.0, m11 = 0.0;
	float m02 = 0.0, m30 = 0.0, m21 = 0.0, m12 = 0.0, m03 = 0.0;
	for (uint y = 0u; y < push.height; ++y) {
		for (uint x = 0u; x < push.width; ++x) {
			float value = load_f32(input_buffer, base + y * push.width + x);
			float fx = float(x), fy = float(y);
			float x2 = fx * fx, y2 = fy * fy;
			m00 += value;
			m10 += fx * value;
			m01 += fy * value;
			m20 += x2 * value;
			m11 += fx * fy * value;
			m02 += y2 * value;
			m30 += x2 * fx * value;
			m21 += x2 * fy * value;
			m12 += fx * y2 * value;
			m03 += y2 * fy * value;
		}
	}
	RWByteAddressBuffer output_buffer = storage_buffers[push.output_index];
	uint out_base = plane * 10u;
	store_f32(output_buffer, out_base + 0u, m00);
	store_f32(output_buffer, out_base + 1u, m10);
	store_f32(output_buffer, out_base + 2u, m01);
	store_f32(output_buffer, out_base + 3u, m20);
	store_f32(output_buffer, out_base + 4u, m11);
	store_f32(output_buffer, out_base + 5u, m02);
	store_f32(output_buffer, out_base + 6u, m30);
	store_f32(output_buffer, out_base + 7u, m21);
	store_f32(output_buffer, out_base + 8u, m12);
	store_f32(output_buffer, out_base + 9u, m03);
}
