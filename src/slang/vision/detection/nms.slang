import storage;
import attributes;

struct PushConstants {
	uint boxes_index;
	uint scores_index;
	uint classes_index;
	uint indices_index;
	uint count_index;
	uint box_count;
	uint maximum;
	uint class_agnostic;
	float iou_threshold;
	float score_threshold;
};
[[vk::push_constant]] PushConstants push;
[[vk::binding(0, 0)]] RWByteAddressBuffer storage_buffers[];

float overlap(uint row_a, uint row_b) {
	uint a = row_a * 4u;
	uint b = row_b * 4u;
	float a_center_x = load_f32(storage_buffers[push.boxes_index], a);
	float a_center_y = load_f32(storage_buffers[push.boxes_index], a + 1u);
	float a_width = load_f32(storage_buffers[push.boxes_index], a + 2u);
	float a_height = load_f32(storage_buffers[push.boxes_index], a + 3u);
	float b_center_x = load_f32(storage_buffers[push.boxes_index], b);
	float b_center_y = load_f32(storage_buffers[push.boxes_index], b + 1u);
	float b_width = load_f32(storage_buffers[push.boxes_index], b + 2u);
	float b_height = load_f32(storage_buffers[push.boxes_index], b + 3u);
	if (!isfinite(a_center_x) || !isfinite(a_center_y) || !isfinite(a_width) ||
		!isfinite(a_height) || !isfinite(b_center_x) || !isfinite(b_center_y) ||
		!isfinite(b_width) || !isfinite(b_height)) return 0.0f;
	a_width = max(a_width, 0.0f);
	a_height = max(a_height, 0.0f);
	b_width = max(b_width, 0.0f);
	b_height = max(b_height, 0.0f);
	float intersection = max(
		min(a_center_x + a_width * 0.5f, b_center_x + b_width * 0.5f) -
		max(a_center_x - a_width * 0.5f, b_center_x - b_width * 0.5f),
		0.0f) * max(
		min(a_center_y + a_height * 0.5f, b_center_y + b_height * 0.5f) -
		max(a_center_y - a_height * 0.5f, b_center_y - b_height * 0.5f),
		0.0f);
	float union_area = a_width * a_height + b_width * b_height - intersection;
	return isfinite(union_area) && union_area > 0.0f ? intersection / union_area : 0.0f;
}

[kernel_name("nms")]
[domain("vision")]
[variant("deterministic_serial")]
[dtype("f32")]
[layout("center_xywh")]
[status("experimental")]
[shader("compute")]
[numthreads(1, 1, 1)]
void main(uint3 thread_id : SV_DispatchThreadID) {
	for (uint index = 0; index < push.maximum; ++index) {
		store_i32(storage_buffers[push.indices_index], index, -1);
	}
	uint selected_count = 0u;
	for (uint output = 0; output < push.maximum; ++output) {
		uint best = 0xffffffffu;
		float best_score = -3.402823466e+38f;
		for (uint candidate = 0; candidate < push.box_count; ++candidate) {
			float score = load_f32(storage_buffers[push.scores_index], candidate);
			if (!isfinite(score) || score < push.score_threshold) continue;
			bool suppressed = false;
			for (uint prior = 0; prior < selected_count; ++prior) {
				uint selected = uint(load_i32(storage_buffers[push.indices_index], prior));
				if (candidate == selected) {
					suppressed = true;
					break;
				}
				bool same_class = push.class_agnostic != 0u ||
					load_i32(storage_buffers[push.classes_index], candidate) ==
					load_i32(storage_buffers[push.classes_index], selected);
				if (same_class && overlap(candidate, selected) >= push.iou_threshold) {
					suppressed = true;
					break;
				}
			}
			if (!suppressed && (score > best_score || (score == best_score && candidate < best))) {
				best = candidate;
				best_score = score;
			}
		}
		if (best == 0xffffffffu) break;
		store_i32(storage_buffers[push.indices_index], selected_count, int(best));
		selected_count++;
	}
	store_u32(storage_buffers[push.count_index], 0u, selected_count);
}
