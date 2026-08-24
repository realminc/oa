struct PushConstants {
	uint confusion_idx, per_class_idx, mean_iou_idx, pixel_accuracy_idx;
	uint classes;
};
[[vk::push_constant]] PushConstants pc;
[[vk::binding(0, 0)]] RWByteAddressBuffer heap[];

[shader("compute")]
[numthreads(1, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
	uint total = 0u;
	uint correct = 0u;
	float iouTotal = 0.0f;
	uint validIouClasses = 0u;
	for (uint classIndex = 0; classIndex < pc.classes; ++classIndex) {
		uint truePositive = heap[pc.confusion_idx].Load(
			(classIndex * pc.classes + classIndex) * 4u);
		uint targetTotal = 0u;
		uint predictedTotal = 0u;
		for (uint other = 0; other < pc.classes; ++other) {
			targetTotal += heap[pc.confusion_idx].Load(
				(classIndex * pc.classes + other) * 4u);
			predictedTotal += heap[pc.confusion_idx].Load(
				(other * pc.classes + classIndex) * 4u);
		}
		uint falsePositive = predictedTotal - truePositive;
		uint falseNegative = targetTotal - truePositive;
		uint unionCount = truePositive + falsePositive + falseNegative;
		float precision = predictedTotal > 0u
			? float(truePositive) / float(predictedTotal) : 0.0f;
		float recall = targetTotal > 0u
			? float(truePositive) / float(targetTotal) : 0.0f;
		float f1 = precision + recall > 0.0f
			? 2.0f * precision * recall / (precision + recall) : 0.0f;
		float iou = unionCount > 0u
			? float(truePositive) / float(unionCount) : 0.0f;
		uint outBase = classIndex * 4u;
		heap[pc.per_class_idx].Store((outBase + 0u) * 4u, asuint(precision));
		heap[pc.per_class_idx].Store((outBase + 1u) * 4u, asuint(recall));
		heap[pc.per_class_idx].Store((outBase + 2u) * 4u, asuint(f1));
		heap[pc.per_class_idx].Store((outBase + 3u) * 4u, asuint(iou));
		if (unionCount > 0u) {
			iouTotal += iou;
			validIouClasses++;
		}
		total += targetTotal;
		correct += truePositive;
	}
	float meanIou = validIouClasses > 0u
		? iouTotal / float(validIouClasses) : 0.0f;
	float accuracy = total > 0u ? float(correct) / float(total) : 0.0f;
	heap[pc.mean_iou_idx].Store(0u, asuint(meanIou));
	heap[pc.pixel_accuracy_idx].Store(0u, asuint(accuracy));
}
