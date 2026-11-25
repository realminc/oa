import attributes;
import storage;

struct PushConstants {
	uint input_index, template_index, output_index;
	uint batch_size, channels, height, width, template_height, template_width;
	uint out_height, out_width, method;
};
[[vk::push_constant]] PushConstants push;
[[vk::binding(0, 0)]] RWByteAddressBuffer storage_buffers[];

[kernel_name("match_template")]
[domain("image")]
[variant("nchw_or_chw")]
[dtype("f32")]
[status("experimental")]
[shader("compute")]
[numthreads(16, 16, 1)]
void main(uint3 thread_id : SV_DispatchThreadID) {
	uint x = thread_id.x, y = thread_id.y, plane = thread_id.z;
	if (x >= push.out_width || y >= push.out_height
		|| plane >= push.batch_size * push.channels) return;
	RWByteAddressBuffer input_buffer = storage_buffers[push.input_index];
	RWByteAddressBuffer template_buffer = storage_buffers[push.template_index];
	uint input_base = plane * push.height * push.width;
	uint template_base = plane % push.channels * push.template_height * push.template_width;
	float sum_i = 0.0, sum_i2 = 0.0, sum_t = 0.0, sum_t2 = 0.0, sum_it = 0.0;
	for (uint ty = 0u; ty < push.template_height; ++ty) {
		for (uint tx = 0u; tx < push.template_width; ++tx) {
			float image_value = load_f32(
				input_buffer, input_base + (y + ty) * push.width + x + tx);
			float template_value = load_f32(
				template_buffer, template_base + ty * push.template_width + tx);
			sum_i += image_value;
			sum_i2 += image_value * image_value;
			sum_t += template_value;
			sum_t2 += template_value * template_value;
			sum_it += image_value * template_value;
		}
	}
	float n = float(push.template_width * push.template_height);
	float denom = sqrt(max(sum_i2 * sum_t2, 0.0));
	float score = 0.0;
	switch (push.method) {
	case 0u:
		score = sum_i2 + sum_t2 - 2.0 * sum_it;
		break;
	case 1u:
		score = denom > 0.0 ? (sum_i2 + sum_t2 - 2.0 * sum_it) / denom : 0.0;
		break;
	case 2u:
		score = sum_it;
		break;
	case 3u:
		score = denom > 0.0 ? sum_it / denom : 0.0;
		break;
	case 4u:
		score = sum_it - sum_i * sum_t / n;
		break;
	case 5u:
		{
			float var_i = sum_i2 - sum_i * sum_i / n;
			float var_t = sum_t2 - sum_t * sum_t / n;
			float dd = sqrt(max(var_i * var_t, 0.0));
			score = dd > 0.0 ? (sum_it - sum_i * sum_t / n) / dd : 0.0;
		}
		break;
	default:
		break;
	}
	uint element = (plane * push.out_height + y) * push.out_width + x;
	store_f32(storage_buffers[push.output_index], element, score);
}
