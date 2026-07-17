struct PushConstants {
	uint predicted_boxes_idx, predicted_scores_idx;
	uint predicted_classes_idx, predicted_images_idx;
	uint target_boxes_idx, target_classes_idx, target_images_idx;
	uint thresholds_idx, state_idx, curve_idx, counts_idx, metrics_idx;
	uint predicted, targets, thresholds, classes, state_stride;
	float score_threshold;
};
[[vk::push_constant]] PushConstants pc;
[[vk::binding(0, 0)]] RWByteAddressBuffer heap[];

float loadF32(uint buffer, uint index) {
	return asfloat(heap[buffer].Load(index * 4u));
}

int loadI32(uint buffer, uint index) {
	return asint(heap[buffer].Load(index * 4u));
}

float boxIou(uint predictedRow, uint targetRow) {
	uint a = predictedRow * 4u;
	uint b = targetRow * 4u;
	float acx = loadF32(pc.predicted_boxes_idx, a + 0u);
	float acy = loadF32(pc.predicted_boxes_idx, a + 1u);
	float aw = max(loadF32(pc.predicted_boxes_idx, a + 2u), 0.0f);
	float ah = max(loadF32(pc.predicted_boxes_idx, a + 3u), 0.0f);
	float bcx = loadF32(pc.target_boxes_idx, b + 0u);
	float bcy = loadF32(pc.target_boxes_idx, b + 1u);
	float bw = max(loadF32(pc.target_boxes_idx, b + 2u), 0.0f);
	float bh = max(loadF32(pc.target_boxes_idx, b + 3u), 0.0f);
	float intersection = max(min(acx + aw * 0.5f, bcx + bw * 0.5f)
		- max(acx - aw * 0.5f, bcx - bw * 0.5f), 0.0f)
		* max(min(acy + ah * 0.5f, bcy + bh * 0.5f)
		- max(acy - ah * 0.5f, bcy - bh * 0.5f), 0.0f);
	float unionArea = aw * ah + bw * bh - intersection;
	return unionArea > 0.0f ? intersection / unionArea : 0.0f;
}

[shader("compute")]
[numthreads(1, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
	uint pair = tid.x;
	uint pairCount = pc.thresholds * pc.classes;
	if (pair >= pairCount) return;
	uint thresholdIndex = pair / pc.classes;
	uint classIndex = pair - thresholdIndex * pc.classes;
	float iouThreshold = loadF32(pc.thresholds_idx, thresholdIndex);
	uint stateBase = pair * pc.state_stride;
	uint curveBase = pair * pc.predicted * 2u;

	for (uint i = 0; i < pc.state_stride; ++i) {
		heap[pc.state_idx].Store((stateBase + i) * 4u, 0u);
	}
	for (uint i = 0; i < pc.predicted * 2u; ++i) {
		heap[pc.curve_idx].Store((curveBase + i) * 4u, 0u);
	}

	uint targetCount = 0u;
	for (uint target = 0; target < pc.targets; ++target) {
		if (loadI32(pc.target_classes_idx, target) == int(classIndex)) {
			targetCount++;
		}
	}

	uint truePositive = 0u;
	uint falsePositive = 0u;
	uint evalTruePositive = 0u;
	uint evalFalsePositive = 0u;
	uint rankCount = 0u;
	for (uint rank = 0; rank < pc.predicted; ++rank) {
		uint bestPrediction = 0xffffffffu;
		float bestScore = -3.402823466e+38f;
		for (uint candidate = 0; candidate < pc.predicted; ++candidate) {
			if (heap[pc.state_idx].Load((stateBase + candidate) * 4u) != 0u) continue;
			if (loadI32(pc.predicted_classes_idx, candidate) != int(classIndex)) continue;
			float score = loadF32(pc.predicted_scores_idx, candidate);
			if (score != score) continue;
			if (score > bestScore || (score == bestScore && candidate < bestPrediction)) {
				bestPrediction = candidate;
				bestScore = score;
			}
		}
		if (bestPrediction == 0xffffffffu) break;
		heap[pc.state_idx].Store((stateBase + bestPrediction) * 4u, 1u);

		uint bestTarget = 0xffffffffu;
		float bestIou = -1.0f;
		int predictedImage = loadI32(pc.predicted_images_idx, bestPrediction);
		for (uint target = 0; target < pc.targets; ++target) {
			if (heap[pc.state_idx].Load(
				(stateBase + pc.predicted + target) * 4u) != 0u) continue;
			if (loadI32(pc.target_classes_idx, target) != int(classIndex)
				|| loadI32(pc.target_images_idx, target) != predictedImage) continue;
			float overlap = boxIou(bestPrediction, target);
			if (overlap > bestIou || (overlap == bestIou && target < bestTarget)) {
				bestTarget = target;
				bestIou = overlap;
			}
		}

		bool matched = bestTarget != 0xffffffffu && iouThreshold == iouThreshold
			&& iouThreshold >= 0.0f && iouThreshold <= 1.0f
			&& bestIou >= iouThreshold;
		if (matched) {
			heap[pc.state_idx].Store(
				(stateBase + pc.predicted + bestTarget) * 4u, 1u);
			truePositive++;
		} else {
			falsePositive++;
		}
		if (bestScore >= pc.score_threshold) {
			if (matched) evalTruePositive++;
			else evalFalsePositive++;
		}

		float precision = float(truePositive) / float(truePositive + falsePositive);
		float recall = targetCount > 0u
			? float(truePositive) / float(targetCount) : 0.0f;
		heap[pc.curve_idx].Store(
			(curveBase + rankCount * 2u + 0u) * 4u, asuint(precision));
		heap[pc.curve_idx].Store(
			(curveBase + rankCount * 2u + 1u) * 4u, asuint(recall));
		rankCount++;
	}

	uint falseNegative = targetCount > evalTruePositive
		? targetCount - evalTruePositive : 0u;
	uint countsBase = pair * 3u;
	heap[pc.counts_idx].Store((countsBase + 0u) * 4u, evalTruePositive);
	heap[pc.counts_idx].Store((countsBase + 1u) * 4u, evalFalsePositive);
	heap[pc.counts_idx].Store((countsBase + 2u) * 4u, falseNegative);
	float precision = evalTruePositive + evalFalsePositive > 0u
		? float(evalTruePositive) / float(evalTruePositive + evalFalsePositive) : 0.0f;
	float recall = targetCount > 0u
		? float(evalTruePositive) / float(targetCount) : 0.0f;
	float f1 = precision + recall > 0.0f
		? 2.0f * precision * recall / (precision + recall) : 0.0f;
	uint metricsBase = pair * 4u;
	heap[pc.metrics_idx].Store((metricsBase + 0u) * 4u, asuint(precision));
	heap[pc.metrics_idx].Store((metricsBase + 1u) * 4u, asuint(recall));
	heap[pc.metrics_idx].Store((metricsBase + 2u) * 4u, asuint(f1));
	heap[pc.metrics_idx].Store((metricsBase + 3u) * 4u, 0u);
}
