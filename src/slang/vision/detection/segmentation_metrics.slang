import storage;
import attributes;

struct PushConstants {
	uint confusion_index;
	uint per_class_index;
	uint mean_iou_index;
	uint pixel_accuracy_index;
	uint class_count;
};
[[vk::push_constant]] PushConstants push;
[[vk::binding(0, 0)]] RWByteAddressBuffer storage_buffers[];

[kernel_name("segmentation_metrics")]
[domain("vision")]
[variant("per_class_and_means")]
[dtype("f32")]
[status("experimental")]
[shader("compute")]
[numthreads(1, 1, 1)]
void main(uint3 thread_id : SV_DispatchThreadID) {
	uint total = 0u;
	uint correct = 0u;
	float iou_total = 0.0f;
	uint valid_iou_classes = 0u;
	for (uint class_index = 0; class_index < push.class_count; ++class_index) {
		uint true_positive = load_u32(storage_buffers[push.confusion_index],
			class_index * push.class_count + class_index);
		uint target_total = 0u;
		uint predicted_total = 0u;
		for (uint other = 0; other < push.class_count; ++other) {
			target_total += load_u32(storage_buffers[push.confusion_index],
				class_index * push.class_count + other);
			predicted_total += load_u32(storage_buffers[push.confusion_index],
				other * push.class_count + class_index);
		}
		uint false_positive = predicted_total - true_positive;
		uint false_negative = target_total - true_positive;
		uint union_count = true_positive + false_positive + false_negative;
		float precision = predicted_total > 0u ? float(true_positive) / float(predicted_total) : 0.0f;
		float recall = target_total > 0u ? float(true_positive) / float(target_total) : 0.0f;
		float f1 = precision + recall > 0.0f ? 2.0f * precision * recall / (precision + recall) : 0.0f;
		float iou = union_count > 0u ? float(true_positive) / float(union_count) : 0.0f;
		uint output_base = class_index * 4u;
		store_f32(storage_buffers[push.per_class_index], output_base, precision);
		store_f32(storage_buffers[push.per_class_index], output_base + 1u, recall);
		store_f32(storage_buffers[push.per_class_index], output_base + 2u, f1);
		store_f32(storage_buffers[push.per_class_index], output_base + 3u, iou);
		if (union_count > 0u) {
			iou_total += iou;
			valid_iou_classes++;
		}
		total += target_total;
		correct += true_positive;
	}
	store_f32(storage_buffers[push.mean_iou_index], 0u,
		valid_iou_classes > 0u ? iou_total / float(valid_iou_classes) : 0.0f);
	store_f32(storage_buffers[push.pixel_accuracy_index], 0u,
		total > 0u ? float(correct) / float(total) : 0.0f);
}
