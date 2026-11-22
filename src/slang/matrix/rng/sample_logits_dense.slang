import attributes;
import storage;

struct PushConstants {
	uint logits_index;
	uint random_index;
	uint output_index;
	uint rows;
	uint vocabulary;
	float temperature;
};

[[vk::push_constant]] PushConstants push;
[[vk::binding(0, 0)]] RWByteAddressBuffer storage_buffers[];

[kernel_name("sample_logits_dense")]
[domain("matrix")]
[variant("dense")]
[dtype("f32")]
[status("experimental")]
[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 dispatch_thread_id : SV_DispatchThreadID) {
	uint row = dispatch_thread_id.x;
	if (row >= push.rows) {
		return;
	}
	uint base = row * push.vocabulary;
	float maximum = load_f32(storage_buffers[push.logits_index], base);
	for (uint column = 1; column < push.vocabulary; ++column) {
		maximum = max(maximum, load_f32(storage_buffers[push.logits_index], base + column));
	}
	float total = 0.0;
	for (uint column = 0; column < push.vocabulary; ++column) {
		total += exp((load_f32(storage_buffers[push.logits_index], base + column) - maximum) / push.temperature);
	}
	float threshold = load_f32(storage_buffers[push.random_index], row) * total;
	float cumulative = 0.0;
	uint selected = 0;
	for (uint column = 0; column < push.vocabulary; ++column) {
		cumulative += exp((load_f32(storage_buffers[push.logits_index], base + column) - maximum) / push.temperature);
		selected = column;
		if (cumulative >= threshold) {
			break;
		}
	}
	store_i32(storage_buffers[push.output_index], row, int(selected));
}
