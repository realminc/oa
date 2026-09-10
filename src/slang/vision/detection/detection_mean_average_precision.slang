import storage;
import attributes;

struct PushConstants {
	uint counts_index;
	uint metrics_index;
	uint map_index;
	uint mean_index;
	uint threshold_count;
	uint class_count;
};
[[vk::push_constant]] PushConstants push;
[[vk::binding(0, 0)]] RWByteAddressBuffer storage_buffers[];

[kernel_name("detection_mean_average_precision")]
[domain("vision")]
[variant("target_present_classes")]
[dtype("f32")]
[status("experimental")]
[shader("compute")]
[numthreads(1, 1, 1)]
void main(uint3 thread_id : SV_DispatchThreadID) {
	float total = 0.0f;
	uint valid_thresholds = 0u;
	for (uint threshold = 0; threshold < push.threshold_count; ++threshold) {
		float class_total = 0.0f;
		uint valid_classes = 0u;
		for (uint class_index = 0; class_index < push.class_count; ++class_index) {
			uint pair = threshold * push.class_count + class_index;
			uint counts_base = pair * 3u;
			uint target_count = load_u32(storage_buffers[push.counts_index], counts_base) +
				load_u32(storage_buffers[push.counts_index], counts_base + 2u);
			if (target_count == 0u) continue;
			class_total += load_f32(storage_buffers[push.metrics_index], pair * 4u + 3u);
			valid_classes++;
		}
		float mean_average_precision = valid_classes > 0u ? class_total / float(valid_classes) : 0.0f;
		store_f32(storage_buffers[push.map_index], threshold, mean_average_precision);
		if (valid_classes > 0u) {
			total += mean_average_precision;
			valid_thresholds++;
		}
	}
	store_f32(storage_buffers[push.mean_index], 0u,
		valid_thresholds > 0u ? total / float(valid_thresholds) : 0.0f);
}
