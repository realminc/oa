import attributes;
import storage;

struct PushConstants {
	uint input_index, output_index, batch_size, channels;
	uint src_height, src_width, dst_height, dst_width;
};
[[vk::push_constant]] PushConstants push;
[[vk::binding(0, 0)]] RWByteAddressBuffer storage_buffers[];

static const float KERNEL[5] = { 1.0, 4.0, 6.0, 4.0, 1.0 };

int reflect101(int coordinate, int extent) {
	if (extent <= 1) return 0;
	int period = 2 * (extent - 1);
	int folded = abs(coordinate) % period;
	return folded < extent ? folded : period - folded;
}

[kernel_name("pyr_up")]
[domain("image")]
[variant("nchw_or_chw")]
[dtype("f32")]
[layout("nchw_or_chw")]
[status("experimental")]
[shader("compute")]
[numthreads(16, 16, 1)]
void main(uint3 thread_id : SV_DispatchThreadID) {
	uint x = thread_id.x, y = thread_id.y, plane = thread_id.z;
	if (x >= push.dst_width || y >= push.dst_height
		|| plane >= push.batch_size * push.channels) return;
	float sum = 0.0;
	for (int dy = -2; dy <= 2; ++dy) {
		int uy = int(y) + dy;
		if ((uy & 1) != 0) continue;
		int sy = reflect101(uy / 2, int(push.src_height));
		for (int dx = -2; dx <= 2; ++dx) {
			int ux = int(x) + dx;
			if ((ux & 1) != 0) continue;
			int sx = reflect101(ux / 2, int(push.src_width));
			float value = load_f32(storage_buffers[push.input_index],
				(plane * push.src_height + uint(sy)) * push.src_width + uint(sx));
			sum += KERNEL[dy + 2] * KERNEL[dx + 2] * value;
		}
	}
	store_f32(storage_buffers[push.output_index],
		(plane * push.dst_height + y) * push.dst_width + x, sum * (4.0 / 256.0));
}
