import storage;
import attributes;

struct PushConstants {
	uint predicted_boxes_index;
	uint predicted_scores_index;
	uint predicted_classes_index;
	uint predicted_images_index;
	uint target_boxes_index;
	uint target_classes_index;
	uint target_images_index;
	uint thresholds_index;
	uint state_index;
	uint curve_index;
	uint counts_index;
	uint metrics_index;
	uint predicted_count;
	uint target_count;
	uint threshold_count;
	uint class_count;
	uint state_stride;
	float score_threshold;
};
[[vk::push_constant]] PushConstants push;
[[vk::binding(0, 0)]] RWByteAddressBuffer storage_buffers[];

float box_iou(uint predicted_row, uint target_row) {
	uint a = predicted_row * 4u;
	uint b = target_row * 4u;
	float a_center_x = load_f32(storage_buffers[push.predicted_boxes_index], a);
	float a_center_y = load_f32(storage_buffers[push.predicted_boxes_index], a + 1u);
	float a_width = load_f32(storage_buffers[push.predicted_boxes_index], a + 2u);
	float a_height = load_f32(storage_buffers[push.predicted_boxes_index], a + 3u);
	float b_center_x = load_f32(storage_buffers[push.target_boxes_index], b);
	float b_center_y = load_f32(storage_buffers[push.target_boxes_index], b + 1u);
	float b_width = load_f32(storage_buffers[push.target_boxes_index], b + 2u);
	float b_height = load_f32(storage_buffers[push.target_boxes_index], b + 3u);
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

[kernel_name("detection_metric_curves")]
[domain("vision")]
[variant("greedy_score_rank")]
[dtype("f32")]
[status("experimental")]
[shader("compute")]
[numthreads(1, 1, 1)]
void main(uint3 thread_id : SV_DispatchThreadID) {
	uint pair = thread_id.x;
	uint pair_count = push.threshold_count * push.class_count;
	if (pair >= pair_count) return;
	uint threshold_index = pair / push.class_count;
	uint class_index = pair - threshold_index * push.class_count;
	float iou_threshold = load_f32(storage_buffers[push.thresholds_index], threshold_index);
	uint state_base = pair * push.state_stride;
	uint curve_base = pair * push.predicted_count * 2u;
	for (uint index = 0; index < push.state_stride; ++index) {
		store_u32(storage_buffers[push.state_index], state_base + index, 0u);
	}
	for (uint index = 0; index < push.predicted_count * 2u; ++index) {
		store_f32(storage_buffers[push.curve_index], curve_base + index, 0.0f);
	}

	uint target_count_for_class = 0u;
	for (uint target = 0; target < push.target_count; ++target) {
		if (load_i32(storage_buffers[push.target_classes_index], target) == int(class_index)) {
			target_count_for_class++;
		}
	}
	uint true_positive = 0u;
	uint false_positive = 0u;
	uint evaluated_true_positive = 0u;
	uint evaluated_false_positive = 0u;
	uint rank_count = 0u;
	for (uint rank = 0; rank < push.predicted_count; ++rank) {
		uint best_prediction = 0xffffffffu;
		float best_score = -3.402823466e+38f;
		for (uint candidate = 0; candidate < push.predicted_count; ++candidate) {
			if (load_u32(storage_buffers[push.state_index], state_base + candidate) != 0u) continue;
			if (load_i32(storage_buffers[push.predicted_classes_index], candidate) != int(class_index)) continue;
			float score = load_f32(storage_buffers[push.predicted_scores_index], candidate);
			if (!isfinite(score)) continue;
			if (score > best_score || (score == best_score && candidate < best_prediction)) {
				best_prediction = candidate;
				best_score = score;
			}
		}
		if (best_prediction == 0xffffffffu) break;
		store_u32(storage_buffers[push.state_index], state_base + best_prediction, 1u);

		uint best_target = 0xffffffffu;
		float best_iou = -1.0f;
		int predicted_image = load_i32(storage_buffers[push.predicted_images_index], best_prediction);
		for (uint target = 0; target < push.target_count; ++target) {
			if (load_u32(storage_buffers[push.state_index], state_base + push.predicted_count + target) != 0u) continue;
			if (load_i32(storage_buffers[push.target_classes_index], target) != int(class_index) ||
				load_i32(storage_buffers[push.target_images_index], target) != predicted_image) continue;
			float candidate_iou = box_iou(best_prediction, target);
			if (candidate_iou > best_iou || (candidate_iou == best_iou && target < best_target)) {
				best_target = target;
				best_iou = candidate_iou;
			}
		}
		bool matched = best_target != 0xffffffffu && isfinite(iou_threshold) &&
			iou_threshold >= 0.0f && iou_threshold <= 1.0f && best_iou >= iou_threshold;
		if (matched) {
			store_u32(storage_buffers[push.state_index], state_base + push.predicted_count + best_target, 1u);
			true_positive++;
		} else {
			false_positive++;
		}
		if (best_score >= push.score_threshold) {
			if (matched) evaluated_true_positive++;
			else evaluated_false_positive++;
		}
		float precision = float(true_positive) / float(true_positive + false_positive);
		float recall = target_count_for_class > 0u ? float(true_positive) / float(target_count_for_class) : 0.0f;
		store_f32(storage_buffers[push.curve_index], curve_base + rank_count * 2u, precision);
		store_f32(storage_buffers[push.curve_index], curve_base + rank_count * 2u + 1u, recall);
		rank_count++;
	}
	uint false_negative = target_count_for_class > evaluated_true_positive ?
		target_count_for_class - evaluated_true_positive : 0u;
	uint counts_base = pair * 3u;
	store_u32(storage_buffers[push.counts_index], counts_base, evaluated_true_positive);
	store_u32(storage_buffers[push.counts_index], counts_base + 1u, evaluated_false_positive);
	store_u32(storage_buffers[push.counts_index], counts_base + 2u, false_negative);
	float precision = evaluated_true_positive + evaluated_false_positive > 0u ?
		float(evaluated_true_positive) / float(evaluated_true_positive + evaluated_false_positive) : 0.0f;
	float recall = target_count_for_class > 0u ?
		float(evaluated_true_positive) / float(target_count_for_class) : 0.0f;
	float f1 = precision + recall > 0.0f ? 2.0f * precision * recall / (precision + recall) : 0.0f;
	uint metrics_base = pair * 4u;
	store_f32(storage_buffers[push.metrics_index], metrics_base, precision);
	store_f32(storage_buffers[push.metrics_index], metrics_base + 1u, recall);
	store_f32(storage_buffers[push.metrics_index], metrics_base + 2u, f1);
	store_f32(storage_buffers[push.metrics_index], metrics_base + 3u, 0.0f);
}
