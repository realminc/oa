// Mechanical adaptation of OA's last-axis categorical-accuracy counter.

import storage;
import attributes;

struct PushConstants {
	uint logits_index;
	uint labels_index;
	uint output_index;
	uint rows;
	uint classes;
	uint label_dtype;
};

[[vk::push_constant]] PushConstants push;
[[vk::binding(0, 0)]] RWByteAddressBuffer storage_buffers[];

groupshared uint shared_counts[256];

[kernel_name("categorical_accuracy_count")]
[domain("matrix")]
[variant("generic")]
[dtype("f32")]
[status("experimental")]
[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 group_thread_id : SV_GroupThreadID) {
	uint lane = group_thread_id.x;
	uint correct = 0;
	for (uint row = lane; row < push.rows; row += 256) {
		uint best = 0;
		float best_value = load_f32(storage_buffers[push.logits_index], row * push.classes);
		for (uint class_index = 1; class_index < push.classes; ++class_index) {
			float value = load_f32(
				storage_buffers[push.logits_index],
				row * push.classes + class_index
			);
			if (value > best_value) {
				best_value = value;
				best = class_index;
			}
		}
		uint label = push.label_dtype == 0
			? load_u8(storage_buffers[push.labels_index], row)
			: load_u32(storage_buffers[push.labels_index], row);
		correct += best == label ? 1 : 0;
	}
	shared_counts[lane] = correct;
	GroupMemoryBarrierWithGroupSync();
	for (uint stride = 128; stride > 0; stride >>= 1) {
		if (lane < stride) {
			shared_counts[lane] += shared_counts[lane + stride];
		}
		GroupMemoryBarrierWithGroupSync();
	}
	if (lane == 0) {
		store_u32(storage_buffers[push.output_index], 0, shared_counts[0]);
	}
}
