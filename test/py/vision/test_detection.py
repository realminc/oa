"""Python binding tests for oa.vision detection surface."""

import math
import unittest

import oa


def _box_iou_ref(a: list[float], b: list[float]) -> float:
	"""Center-coordinate IoU reference matching the Rust oracle."""
	if any(not math.isfinite(v) for v in a + b):
		return 0.0
	aw, ah = max(a[2], 0.0), max(a[3], 0.0)
	bw, bh = max(b[2], 0.0), max(b[3], 0.0)
	ax0, ay0, ax1, ay1 = a[0] - aw * 0.5, a[1] - ah * 0.5, a[0] + aw * 0.5, a[1] + ah * 0.5
	bx0, by0, bx1, by1 = b[0] - bw * 0.5, b[1] - bh * 0.5, b[0] + bw * 0.5, b[1] + bh * 0.5
	iw = max(min(ax1, bx1) - max(ax0, bx0), 0.0)
	ih = max(min(ay1, by1) - max(ay0, by0), 0.0)
	inter = iw * ih
	union = aw * ah + bw * bh - inter
	return inter / union if math.isfinite(union) and union > 0.0 else 0.0


def _pairwise_iou_ref(boxes_a: list[float], boxes_b: list[float]) -> list[float]:
	a_rows = [boxes_a[i * 4 : i * 4 + 4] for i in range(len(boxes_a) // 4)]
	b_rows = [boxes_b[i * 4 : i * 4 + 4] for i in range(len(boxes_b) // 4)]
	return [_box_iou_ref(a, b) for a in a_rows for b in b_rows]


class VisionBoxIouTest(unittest.TestCase):
	@classmethod
	def setUpClass(cls) -> None:
		cls.engine = oa.Engine()

	def test_box_iou_matches_pairwise_oracle(self) -> None:
		boxes_a = [0.5, 0.5, 1.0, 1.0, 0.0, 0.0, 2.0, 2.0]
		boxes_b = [0.5, 0.5, 1.0, 1.0, 1.0, 0.5, 1.0, 1.0]
		left = oa.Matrix.from_f32(self.engine, [2, 4], boxes_a)
		right = oa.Matrix.from_f32(self.engine, [2, 4], boxes_b)
		output = oa.vision.box_iou(left, right)
		self.assertEqual(output.shape, [2, 2])
		self.assertEqual(output.dtype, "f32")
		expected = _pairwise_iou_ref(boxes_a, boxes_b)
		for actual, exp in zip(output.read_f32(), expected):
			self.assertAlmostEqual(actual, exp, places=5)

	def test_box_iou_result_type(self) -> None:
		m = oa.Matrix.from_f32(self.engine, [1, 4], [0.5, 0.5, 1.0, 1.0])
		result = oa.vision.box_iou(m, m)
		self.assertIsInstance(result, oa.Matrix)


class VisionNmsTest(unittest.TestCase):
	@classmethod
	def setUpClass(cls) -> None:
		cls.engine = oa.Engine()

	def test_nms_class_aware_score_ranked(self) -> None:
		boxes = oa.Matrix.from_f32(
			self.engine,
			[5, 4],
			[
				0.50, 0.50, 0.40, 0.40,
				0.51, 0.50, 0.40, 0.40,
				0.50, 0.50, 0.40, 0.40,
				0.10, 0.10, 0.10, 0.10,
				0.90, 0.90, 0.10, 0.10,
			],
		)
		scores = oa.Matrix.from_f32(self.engine, [5], [0.90, 0.80, 0.85, 0.70, 0.70])
		classes = oa.Matrix.from_i32(self.engine, [5], [0, 0, 1, 0, 0])
		result = oa.vision.nms(boxes, scores, classes, iou_threshold=0.5, max_detections=5)
		self.assertIsInstance(result, oa.vision.NmsResult)
		self.assertIsInstance(result.indices, oa.Matrix)
		self.assertIsInstance(result.count, oa.Matrix)
		self.assertEqual(result.indices.shape, [5])
		self.assertEqual(result.indices.dtype, "i32")
		self.assertEqual(result.count.dtype, "u32")
		self.assertEqual(result.count.read_u32(), [4])
		self.assertEqual(result.indices.read_i32(), [0, 2, 3, 4, -1])

	def test_nms_class_agnostic(self) -> None:
		boxes = oa.Matrix.from_f32(
			self.engine,
			[5, 4],
			[
				0.50, 0.50, 0.40, 0.40,
				0.51, 0.50, 0.40, 0.40,
				0.50, 0.50, 0.40, 0.40,
				0.10, 0.10, 0.10, 0.10,
				0.90, 0.90, 0.10, 0.10,
			],
		)
		scores = oa.Matrix.from_f32(self.engine, [5], [0.90, 0.80, 0.85, 0.70, 0.70])
		classes = oa.Matrix.from_i32(self.engine, [5], [0, 0, 1, 0, 0])
		result = oa.vision.nms(
			boxes, scores, classes, iou_threshold=0.5, max_detections=3, class_agnostic=True
		)
		self.assertEqual(result.count.read_u32(), [3])
		self.assertEqual(result.indices.read_i32(), [0, 3, 4])


class VisionConfusionAndMaskTest(unittest.TestCase):
	@classmethod
	def setUpClass(cls) -> None:
		cls.engine = oa.Engine()

	def test_confusion_matrix_matches_oracle(self) -> None:
		predicted = oa.Matrix.from_i32(self.engine, [7], [0, 1, 2, 1, -1, 3, 2])
		target = oa.Matrix.from_i32(self.engine, [7], [0, 2, 2, 1, 0, 0, 7])
		confusion = oa.vision.confusion_matrix(predicted, target, 3)
		self.assertEqual(confusion.shape, [3, 3])
		self.assertEqual(confusion.read_u32(), [1, 0, 0, 0, 1, 0, 0, 1, 1])

	def test_binary_mask_counts_matches_oracle(self) -> None:
		predicted = oa.Matrix.from_u8(self.engine, [7], [1, 1, 0, 0, 3, 0, 1])
		target = oa.Matrix.from_u8(self.engine, [7], [1, 0, 1, 0, 1, 0, 0])
		counts = oa.vision.binary_mask_counts(predicted, target)
		self.assertEqual(counts.read_u32(), [2, 2, 1, 2])


class VisionEvaluateTest(unittest.TestCase):
	@classmethod
	def setUpClass(cls) -> None:
		cls.engine = oa.Engine()

	def test_evaluate_returns_detection_metrics_result(self) -> None:
		predicted_boxes = oa.Matrix.from_f32(
			self.engine,
			[4, 4],
			[
				0.20, 0.20, 0.20, 0.20,
				0.20, 0.20, 0.20, 0.20,
				0.50, 0.50, 0.20, 0.20,
				0.80, 0.80, 0.20, 0.20,
			],
		)
		predicted_scores = oa.Matrix.from_f32(self.engine, [4], [0.90, 0.80, 0.70, 0.60])
		predicted_classes = oa.Matrix.from_i32(self.engine, [4], [0, 0, 1, 0])
		predicted_images = oa.Matrix.from_i32(self.engine, [4], [0, 0, 0, 1])
		target_boxes = oa.Matrix.from_f32(
			self.engine,
			[3, 4],
			[0.20, 0.20, 0.20, 0.20, 0.80, 0.80, 0.20, 0.20, 0.50, 0.50, 0.20, 0.20],
		)
		target_classes = oa.Matrix.from_i32(self.engine, [3], [0, 0, 1])
		target_images = oa.Matrix.from_i32(self.engine, [3], [0, 1, 0])
		thresholds = oa.Matrix.from_f32(self.engine, [2], [0.50, 0.75])
		result = oa.vision.evaluate(
			predicted_boxes,
			predicted_scores,
			predicted_classes,
			predicted_images,
			target_boxes,
			target_classes,
			target_images,
			thresholds,
			2,
			0.75,
		)
		self.assertIsInstance(result, oa.vision.DetectionMetricsResult)
		self.assertIsInstance(result.counts, oa.Matrix)
		self.assertIsInstance(result.per_class, oa.Matrix)
		self.assertIsInstance(result.mean_average_precision_by_threshold, oa.Matrix)
		self.assertIsInstance(result.mean_average_precision, oa.Matrix)
		self.assertEqual(result.counts.shape, [2, 2, 3])
		self.assertEqual(result.counts.read_u32(), [1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 1])
		self.assertAlmostEqual(result.mean_average_precision.read_f32()[0], 0.917_491_7, places=4)

	def test_evaluate_segmentation_returns_segmentation_metrics_result(self) -> None:
		predicted = oa.Matrix.from_i32(self.engine, [2, 3], [0, 1, 2, 1, 8, 0])
		target = oa.Matrix.from_i32(self.engine, [2, 3], [0, 2, 2, 1, 0, -1])
		result = oa.vision.evaluate_segmentation(predicted, target, 3)
		self.assertIsInstance(result, oa.vision.SegmentationMetricsResult)
		self.assertIsInstance(result.confusion, oa.Matrix)
		self.assertIsInstance(result.per_class, oa.Matrix)
		self.assertIsInstance(result.mean_iou, oa.Matrix)
		self.assertIsInstance(result.pixel_accuracy, oa.Matrix)
		self.assertEqual(result.confusion.read_u32(), [1, 0, 0, 0, 1, 0, 0, 1, 1])
		self.assertAlmostEqual(result.mean_iou.read_f32()[0], 2.0 / 3.0, places=5)
		self.assertAlmostEqual(result.pixel_accuracy.read_f32()[0], 0.75, places=5)


if __name__ == "__main__":
	unittest.main()
