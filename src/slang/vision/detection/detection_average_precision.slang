import storage;
import attributes;

struct PushConstants {
	uint curve_index;
	uint counts_index;
	uint metrics_index;
	uint predicted_count;
	uint pair_count;
};
[[vk::push_constant]] PushConstants push;
[[vk::binding(0, 0)]] RWByteAddressBuffer storage_buffers[];

[kernel_name("detection_average_precision")]
[domain("vision")]
[variant("interpolated_101_point")]
[dtype("f32")]
[status("experimental")]
[shader("compute")]
[numthreads(1, 1, 1)]
void main(uint3 thread_id : SV_DispatchThreadID) {
	uint pair = thread_id.x;
	if (pair >= push.pair_count) return;
	uint counts_base = pair * 3u;
	uint target_count = load_u32(storage_buffers[push.counts_index], counts_base) +
		load_u32(storage_buffers[push.counts_index], counts_base + 2u);
	if (target_count == 0u) {
		store_f32(storage_buffers[push.metrics_index], pair * 4u + 3u, 0.0f);
		return;
	}
	uint curve_base = pair * push.predicted_count * 2u;
	float sum = 0.0f;
	for (uint recall_point = 0; recall_point <= 100u; ++recall_point) {
		float required_recall = float(recall_point) * 0.01f;
		float best_precision = 0.0f;
		for (uint rank = 0; rank < push.predicted_count; ++rank) {
			float precision = load_f32(storage_buffers[push.curve_index], curve_base + rank * 2u);
			float recall = load_f32(storage_buffers[push.curve_index], curve_base + rank * 2u + 1u);
			if (recall + 1.0e-7f >= required_recall) best_precision = max(best_precision, precision);
		}
		sum += best_precision;
	}
	store_f32(storage_buffers[push.metrics_index], pair * 4u + 3u, sum / 101.0f);
}
