import attributes;
import storage;

struct PushConstants {
	uint values_index;
	uint indices_index;
	uint random_index;
	uint output_index;
	uint rows;
	uint candidate_count;
	float temperature;
	float top_p;
};

[[vk::push_constant]] PushConstants push;
[[vk::binding(0, 0)]] RWByteAddressBuffer storage_buffers[];

[kernel_name("sample_logits_sorted")]
[domain("matrix")]
[variant("top_k_top_p")]
[dtype("f32")]
[status("experimental")]
[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 dispatch_thread_id : SV_DispatchThreadID) {
	uint row = dispatch_thread_id.x;
	if (row >= push.rows) {
		return;
	}
	uint base = row * push.candidate_count;
	float maximum = load_f32(storage_buffers[push.values_index], base);
	float total = 0.0;
	for (uint candidate = 0; candidate < push.candidate_count; ++candidate) {
		total += exp((load_f32(storage_buffers[push.values_index], base + candidate) - maximum) / push.temperature);
	}
	float nucleus_target = total * push.top_p;
	float nucleus_mass = 0.0;
	for (uint candidate = 0; candidate < push.candidate_count; ++candidate) {
		nucleus_mass += exp((load_f32(storage_buffers[push.values_index], base + candidate) - maximum) / push.temperature);
		if (nucleus_mass >= nucleus_target) {
			break;
		}
	}
	float threshold = load_f32(storage_buffers[push.random_index], row) * nucleus_mass;
	float cumulative = 0.0;
	uint selected = 0;
	for (uint candidate = 0; candidate < push.candidate_count; ++candidate) {
		cumulative += exp((load_f32(storage_buffers[push.values_index], base + candidate) - maximum) / push.temperature);
		selected = candidate;
		if (cumulative >= threshold) {
			break;
		}
	}
	int token = load_i32(storage_buffers[push.indices_index], base + selected);
	store_i32(storage_buffers[push.output_index], row, token);
}
