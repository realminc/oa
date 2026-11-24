import attributes;
import storage;

struct PushConstants {
	uint first_index, second_index, output_index, row_count, bin_count, method;
};
[[vk::push_constant]] PushConstants push;
[[vk::binding(0, 0)]] RWByteAddressBuffer storage_buffers[];

[kernel_name("compare_hist")]
[domain("image")]
[variant("row_reduce")]
[dtype("u32")]
[status("experimental")]
[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 thread_id : SV_DispatchThreadID) {
	uint row = thread_id.x;
	if (row >= push.row_count) return;
	uint base = row * push.bin_count;
	float score = 0.0;
	if (push.method == 0u) {
		float mean_first = 0.0, mean_second = 0.0;
		for (uint i = 0u; i < push.bin_count; ++i) {
			mean_first += float(load_u32(storage_buffers[push.first_index], base + i));
			mean_second += float(load_u32(storage_buffers[push.second_index], base + i));
		}
		mean_first /= float(push.bin_count);
		mean_second /= float(push.bin_count);
		float numerator = 0.0, first_squares = 0.0, second_squares = 0.0;
		for (uint i = 0u; i < push.bin_count; ++i) {
			float a = float(load_u32(storage_buffers[push.first_index], base + i)) - mean_first;
			float b = float(load_u32(storage_buffers[push.second_index], base + i)) - mean_second;
			numerator += a * b;
			first_squares += a * a;
			second_squares += b * b;
		}
		float denominator = sqrt(first_squares * second_squares);
		score = denominator > 0.0 ? numerator / denominator : 0.0;
	} else if (push.method == 1u) {
		for (uint i = 0u; i < push.bin_count; ++i) {
			float a = float(load_u32(storage_buffers[push.first_index], base + i));
			float b = float(load_u32(storage_buffers[push.second_index], base + i));
			if (a > 0.0) score += (a - b) * (a - b) / a;
		}
	} else if (push.method == 2u) {
		float overlap = 0.0, first_total = 0.0;
		for (uint i = 0u; i < push.bin_count; ++i) {
			float a = float(load_u32(storage_buffers[push.first_index], base + i));
			float b = float(load_u32(storage_buffers[push.second_index], base + i));
			overlap += min(a, b);
			first_total += a;
		}
		score = overlap / max(first_total, 1.0);
	} else {
		float coefficient = 0.0, first_total = 0.0, second_total = 0.0;
		for (uint i = 0u; i < push.bin_count; ++i) {
			float a = float(load_u32(storage_buffers[push.first_index], base + i));
			float b = float(load_u32(storage_buffers[push.second_index], base + i));
			coefficient += sqrt(a * b);
			first_total += a;
			second_total += b;
		}
		float denominator = sqrt(first_total * second_total);
		score = denominator > 0.0 ? sqrt(max(1.0 - coefficient / denominator, 0.0)) : 0.0;
	}
	store_f32(storage_buffers[push.output_index], row, score);
}
