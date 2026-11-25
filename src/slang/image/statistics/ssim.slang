import attributes;
import storage;

struct PushConstants {
	uint first_index, second_index, output_index;
	uint batch_size, channels, height, width;
};
[[vk::push_constant]] PushConstants push;
[[vk::binding(0, 0)]] RWByteAddressBuffer storage_buffers[];

[kernel_name("ssim")]
[domain("image")]
[variant("serial_plane")]
[dtype("f32")]
[status("experimental")]
[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 thread_id : SV_DispatchThreadID) {
	uint plane = thread_id.x;
	if (plane >= push.batch_size * push.channels) return;
	RWByteAddressBuffer first = storage_buffers[push.first_index];
	RWByteAddressBuffer second = storage_buffers[push.second_index];
	uint base = plane * push.height * push.width;
	uint count = push.height * push.width;
	float sum_a = 0.0, sum_b = 0.0, sum_aa = 0.0, sum_bb = 0.0, sum_ab = 0.0;
	for (uint element = 0u; element < count; ++element) {
		float a = load_f32(first, base + element);
		float b = load_f32(second, base + element);
		sum_a += a;
		sum_b += b;
		sum_aa += a * a;
		sum_bb += b * b;
		sum_ab += a * b;
	}
	float n = float(count);
	float mean_a = sum_a / n;
	float mean_b = sum_b / n;
	float var_a = sum_aa / n - mean_a * mean_a;
	float var_b = sum_bb / n - mean_b * mean_b;
	float cov = sum_ab / n - mean_a * mean_b;
	float c1 = 0.0001;
	float c2 = 0.0009;
	float value = (2.0 * mean_a * mean_b + c1) * (2.0 * cov + c2)
		/ ((mean_a * mean_a + mean_b * mean_b + c1) * (var_a + var_b + c2));
	store_f32(storage_buffers[push.output_index], plane, value);
}
