//! Stateless object-detection operations.

use crate::{
	DType, Error, Matrix, OpAttribute, OperationContract, Result,
	runtime::{BufferBinding, ComputeDispatch, KernelId, PushConstant, SemanticDispatch},
};

/// Deterministic non-maximum-suppression policy.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct NmsConfig {
	/// Suppress a same-class box when its IoU is at least this value.
	pub iou_threshold: f32,
	/// Ignore candidates below this score.
	pub score_threshold: f32,
	/// Maximum number of selected source rows.
	pub max_detections: i32,
	/// Suppress overlaps across class labels when true.
	pub class_agnostic: bool,
}

impl Default for NmsConfig {
	fn default() -> Self {
		Self {
			iou_threshold: 0.45,
			score_threshold: 0.0,
			max_detections: 100,
			class_agnostic: false,
		}
	}
}

/// Device-resident result of [`nms`].
pub struct NmsResult {
	/// I32 selected source-row indices, padded with `-1` after [`Self::count`].
	pub indices: Matrix,
	/// U32 scalar containing the selected prefix length.
	pub count: Matrix,
}

/// Device-resident object-detection evaluation outputs.
pub struct DetectionMetricsResult {
	/// U32 `[T, C, 3]`: true positives, false positives, false negatives.
	pub counts: Matrix,
	/// FP32 `[T, C, 4]`: precision, recall, F1, interpolated AP.
	pub per_class: Matrix,
	/// FP32 `[T]`: mean AP across classes containing at least one target.
	pub mean_average_precision_by_threshold: Matrix,
	/// FP32 `[1]`: mean over thresholds containing at least one target class.
	pub mean_average_precision: Matrix,
}

/// Device-resident semantic-segmentation evaluation outputs.
pub struct SegmentationMetricsResult {
	/// U32 `[C, C]`, with target rows and predicted columns.
	pub confusion: Matrix,
	/// FP32 `[C, 4]`: precision, recall, F1/Dice, IoU.
	pub per_class: Matrix,
	/// FP32 `[1]`: mean IoU across classes with nonzero union.
	pub mean_iou: Matrix,
	/// FP32 `[1]`: correct divided by admitted labels.
	pub pixel_accuracy: Matrix,
}

/// Compute every pairwise intersection-over-union for center-coordinate boxes.
///
/// `boxes_a` and `boxes_b` must be nonempty FP32 matrices shaped `[N, 4]` and
/// `[M, 4]`. Each row is `[center_x, center_y, width, height]`; negative widths
/// and heights are clamped to zero, matching donor OA. A pair with no positive
/// finite union produces zero. The returned matrix has shape `[N, M]`.
///
/// The call records asynchronously and returns its output immediately. Reading
/// the result is an explicit submission and observation point.
///
/// # Errors
///
/// Returns an error when an input has the wrong rank, extent, dtype, or engine,
/// a shader-ABI size overflows, allocation fails, or runtime recording fails.
pub fn box_iou(boxes_a: &Matrix, boxes_b: &Matrix) -> Result<Matrix> {
	let rows_a = box_rows(boxes_a, "boxes_a")?;
	let rows_b = box_rows(boxes_b, "boxes_b")?;
	let engine = boxes_a.engine_handle();
	if !engine.same_as(boxes_b.engine_handle()) {
		return Err(Error::invalid_argument(
			"vision::box_iou inputs must belong to the same engine",
		));
	}
	let pair_count = rows_a
		.checked_mul(rows_b)
		.ok_or_else(|| Error::out_of_range("vision::box_iou pair count exceeds u32"))?;
	let output_elements = usize::try_from(pair_count)
		.map_err(|_| Error::out_of_range("vision::box_iou pair count exceeds usize"))?;
	let output = Matrix::allocate(
		engine,
		vec![rows_a as usize, rows_b as usize],
		output_elements,
		DType::F32,
	)?;
	let kernel = KernelId::VisionBoxIouF32;
	let buffers = [
		BufferBinding::read(boxes_a.storage()),
		BufferBinding::read(boxes_b.storage()),
		BufferBinding::write(output.storage()),
	];
	let push_constants = [PushConstant::U32(rows_a), PushConstant::U32(rows_b)];
	let inputs = [boxes_a, boxes_b];
	let outputs = [&output];
	engine.record_semantic(
		ComputeDispatch {
			kernel,
			buffers: &buffers,
			push_constants: &push_constants,
			workgroups: kernel.linear_workgroups(pair_count),
		},
		SemanticDispatch {
			contract: crate::core::operation::vision::BOX_IOU,
			inputs: &inputs,
			outputs: &outputs,
			attributes: &[],
		},
	)?;
	Ok(output)
}

/// Select score-ranked boxes with deterministic class-aware suppression.
///
/// Equal scores select the smaller source index. NaN or infinite scores are
/// ignored. All outputs remain device-resident and the operation never sorts or
/// compacts on the host.
///
/// # Errors
///
/// Returns an error unless boxes are nonempty FP32 `[N, 4]`, scores are FP32
/// `[N]`, classes are I32 `[N]`, all inputs share an engine, thresholds are
/// finite, `iou_threshold` is in `[0, 1]`, and `max_detections` is positive.
pub fn nms(
	boxes: &Matrix,
	scores: &Matrix,
	classes: &Matrix,
	config: NmsConfig,
) -> Result<NmsResult> {
	let count = box_rows(boxes, "boxes")?;
	vector(scores, "scores", count as usize, DType::F32)?;
	vector(classes, "classes", count as usize, DType::I32)?;
	validate_same_engine(boxes, &[scores, classes], "vision::nms")?;
	if !config.iou_threshold.is_finite() || !(0.0..=1.0).contains(&config.iou_threshold) {
		return Err(Error::invalid_argument(
			"vision::nms iou_threshold must be finite and in [0, 1]",
		));
	}
	if !config.score_threshold.is_finite() {
		return Err(Error::invalid_argument(
			"vision::nms score_threshold must be finite",
		));
	}
	let maximum = u32::try_from(config.max_detections)
		.map_err(|_| Error::invalid_argument("vision::nms max_detections must be positive"))?
		.min(count);
	if maximum == 0 {
		return Err(Error::invalid_argument(
			"vision::nms max_detections must be positive",
		));
	}
	let engine = boxes.engine_handle();
	let indices = Matrix::allocate(engine, vec![maximum as usize], maximum as usize, DType::I32)?;
	let selected_count = Matrix::allocate(engine, vec![1], 1, DType::U32)?;
	let buffers = [
		BufferBinding::read(boxes.storage()),
		BufferBinding::read(scores.storage()),
		BufferBinding::read(classes.storage()),
		BufferBinding::write(indices.storage()),
		BufferBinding::write(selected_count.storage()),
	];
	let push_constants = [
		PushConstant::U32(count),
		PushConstant::U32(maximum),
		PushConstant::U32(u32::from(config.class_agnostic)),
		PushConstant::F32(config.iou_threshold),
		PushConstant::F32(config.score_threshold),
	];
	let inputs = [boxes, scores, classes];
	let outputs = [&indices, &selected_count];
	let attributes = [
		OpAttribute::Float {
			name: "iou_threshold".into(),
			value: f64::from(config.iou_threshold),
		},
		OpAttribute::Float {
			name: "score_threshold".into(),
			value: f64::from(config.score_threshold),
		},
		OpAttribute::SignedInteger {
			name: "max_detections".into(),
			value: i64::from(config.max_detections),
		},
		OpAttribute::Boolean {
			name: "class_agnostic".into(),
			value: config.class_agnostic,
		},
	];
	engine.record_semantic(
		ComputeDispatch {
			kernel: KernelId::VisionNmsF32,
			buffers: &buffers,
			push_constants: &push_constants,
			workgroups: [1, 1, 1],
		},
		SemanticDispatch {
			contract: crate::core::operation::vision::NMS,
			inputs: &inputs,
			outputs: &outputs,
			attributes: &attributes,
		},
	)?;
	Ok(NmsResult {
		indices,
		count: selected_count,
	})
}

/// Accumulate an integer classification confusion matrix.
///
/// Rows are targets and columns are predictions. Either label being outside
/// `[0, class_count)` ignores that element.
///
/// # Errors
///
/// Returns an error unless inputs are equal nonempty I32 vectors on one engine,
/// `class_count` is positive, and all sizes fit the shader ABI.
pub fn confusion_matrix(predicted: &Matrix, target: &Matrix, class_count: i32) -> Result<Matrix> {
	let element_count = equal_nonempty(predicted, target, DType::I32, "vision::confusion_matrix")?;
	if predicted.shape().len() != 1 {
		return Err(Error::invalid_argument(format!(
			"vision::confusion_matrix inputs must be vectors; received {:?}",
			predicted.shape()
		)));
	}
	let classes = positive_class_count(class_count, "vision::confusion_matrix")?;
	let output_elements = classes
		.checked_mul(classes)
		.ok_or_else(|| Error::out_of_range("vision::confusion_matrix output exceeds u32"))?;
	record_confusion(
		predicted,
		target,
		classes,
		element_count,
		KernelId::VisionConfusionMatrixClearU32,
		KernelId::VisionConfusionMatrixI32,
		crate::core::operation::vision::CONFUSION_MATRIX,
		vec![classes as usize, classes as usize],
		output_elements as usize,
	)
}

/// Count `[true_positive, false_positive, false_negative, true_negative]` for
/// two equally-shaped U8 masks. Every nonzero byte is true.
///
/// # Errors
///
/// Returns an error unless inputs are nonempty, equal-shaped U8 matrices on one
/// engine and their element count fits the shader ABI.
pub fn binary_mask_counts(predicted: &Matrix, target: &Matrix) -> Result<Matrix> {
	let element_count = equal_nonempty(predicted, target, DType::U8, "vision::binary_mask_counts")?;
	let output = Matrix::allocate(predicted.engine_handle(), vec![4], 4, DType::U32)?;
	let clear_buffers = [BufferBinding::write(output.storage())];
	let clear_push = [PushConstant::U32(4)];
	let count_buffers = [
		BufferBinding::read(predicted.storage()),
		BufferBinding::read(target.storage()),
		BufferBinding::read_write(output.storage()),
	];
	let count_push = [PushConstant::U32(element_count)];
	let dispatches = [
		ComputeDispatch {
			kernel: KernelId::VisionBinaryMaskCountsClearU32,
			buffers: &clear_buffers,
			push_constants: &clear_push,
			workgroups: KernelId::VisionBinaryMaskCountsClearU32.linear_workgroups(4),
		},
		ComputeDispatch {
			kernel: KernelId::VisionBinaryMaskCountsU8,
			buffers: &count_buffers,
			push_constants: &count_push,
			workgroups: KernelId::VisionBinaryMaskCountsU8.linear_workgroups(element_count),
		},
	];
	let inputs = [predicted, target];
	let outputs = [&output];
	predicted.engine_handle().record_split_semantic(
		&dispatches,
		SemanticDispatch {
			contract: crate::core::operation::vision::BINARY_MASK_COUNTS,
			inputs: &inputs,
			outputs: &outputs,
			attributes: &[],
		},
	)?;
	Ok(output)
}

/// Evaluate object-detection precision, recall, F1, AP, and mAP.
///
/// Matching is class-aware, image-aware, greedy by descending score, and uses
/// 101-point interpolated average precision. Classes without targets do not
/// contribute to mAP. Scratch and outputs remain GPU-resident.
///
/// # Errors
///
/// Returns an error when any shape, dtype, ownership, class-count,
/// score-threshold, or 32-bit scratch-size contract is invalid.
#[allow(clippy::too_many_arguments)]
pub fn evaluate(
	predicted_boxes: &Matrix,
	predicted_scores: &Matrix,
	predicted_classes: &Matrix,
	predicted_image_ids: &Matrix,
	target_boxes: &Matrix,
	target_classes: &Matrix,
	target_image_ids: &Matrix,
	iou_thresholds: &Matrix,
	class_count: i32,
	score_threshold: f32,
) -> Result<DetectionMetricsResult> {
	let predicted = box_rows(predicted_boxes, "predicted_boxes")?;
	let targets = box_rows(target_boxes, "target_boxes")?;
	vector(
		predicted_scores,
		"predicted_scores",
		predicted as usize,
		DType::F32,
	)?;
	vector(
		predicted_classes,
		"predicted_classes",
		predicted as usize,
		DType::I32,
	)?;
	vector(
		predicted_image_ids,
		"predicted_image_ids",
		predicted as usize,
		DType::I32,
	)?;
	vector(
		target_classes,
		"target_classes",
		targets as usize,
		DType::I32,
	)?;
	vector(
		target_image_ids,
		"target_image_ids",
		targets as usize,
		DType::I32,
	)?;
	let threshold_count = nonempty_vector(iou_thresholds, "iou_thresholds", DType::F32)?;
	let classes = positive_class_count(class_count, "vision::evaluate")?;
	if !score_threshold.is_finite() {
		return Err(Error::invalid_argument(
			"vision::evaluate score_threshold must be finite",
		));
	}
	validate_same_engine(
		predicted_boxes,
		&[
			predicted_scores,
			predicted_classes,
			predicted_image_ids,
			target_boxes,
			target_classes,
			target_image_ids,
			iou_thresholds,
		],
		"vision::evaluate",
	)?;
	let pair_count = threshold_count
		.checked_mul(classes)
		.ok_or_else(|| Error::out_of_range("vision::evaluate threshold-class pairs exceed u32"))?;
	let state_stride = predicted
		.checked_add(targets)
		.ok_or_else(|| Error::out_of_range("vision::evaluate state stride exceeds u32"))?;
	let state_elements = pair_count
		.checked_mul(state_stride)
		.ok_or_else(|| Error::out_of_range("vision::evaluate state scratch exceeds u32"))?;
	let curve_elements = pair_count
		.checked_mul(predicted)
		.and_then(|value| value.checked_mul(2))
		.ok_or_else(|| Error::out_of_range("vision::evaluate curve scratch exceeds u32"))?;
	let counts_elements = pair_count
		.checked_mul(3)
		.ok_or_else(|| Error::out_of_range("vision::evaluate counts output exceeds u32"))?;
	let metrics_elements = pair_count
		.checked_mul(4)
		.ok_or_else(|| Error::out_of_range("vision::evaluate metrics output exceeds u32"))?;
	let engine = predicted_boxes.engine_handle();
	let state = allocate(
		engine,
		&[pair_count, state_stride],
		state_elements,
		DType::U32,
	)?;
	let curve = allocate(
		engine,
		&[pair_count, predicted, 2],
		curve_elements,
		DType::F32,
	)?;
	let counts = allocate(
		engine,
		&[threshold_count, classes, 3],
		counts_elements,
		DType::U32,
	)?;
	let per_class = allocate(
		engine,
		&[threshold_count, classes, 4],
		metrics_elements,
		DType::F32,
	)?;
	let map_by_threshold = allocate(engine, &[threshold_count], threshold_count, DType::F32)?;
	let mean_average_precision = allocate(engine, &[1], 1, DType::F32)?;

	let curves_buffers = [
		BufferBinding::read(predicted_boxes.storage()),
		BufferBinding::read(predicted_scores.storage()),
		BufferBinding::read(predicted_classes.storage()),
		BufferBinding::read(predicted_image_ids.storage()),
		BufferBinding::read(target_boxes.storage()),
		BufferBinding::read(target_classes.storage()),
		BufferBinding::read(target_image_ids.storage()),
		BufferBinding::read(iou_thresholds.storage()),
		BufferBinding::write(state.storage()),
		BufferBinding::write(curve.storage()),
		BufferBinding::write(counts.storage()),
		BufferBinding::write(per_class.storage()),
	];
	let curves_push = [
		PushConstant::U32(predicted),
		PushConstant::U32(targets),
		PushConstant::U32(threshold_count),
		PushConstant::U32(classes),
		PushConstant::U32(state_stride),
		PushConstant::F32(score_threshold),
	];
	let ap_buffers = [
		BufferBinding::read(curve.storage()),
		BufferBinding::read(counts.storage()),
		BufferBinding::read_write(per_class.storage()),
	];
	let ap_push = [PushConstant::U32(predicted), PushConstant::U32(pair_count)];
	let mean_buffers = [
		BufferBinding::read(counts.storage()),
		BufferBinding::read(per_class.storage()),
		BufferBinding::write(map_by_threshold.storage()),
		BufferBinding::write(mean_average_precision.storage()),
	];
	let mean_push = [
		PushConstant::U32(threshold_count),
		PushConstant::U32(classes),
	];
	let dispatches = [
		ComputeDispatch {
			kernel: KernelId::VisionDetectionMetricCurvesF32,
			buffers: &curves_buffers,
			push_constants: &curves_push,
			workgroups: [pair_count, 1, 1],
		},
		ComputeDispatch {
			kernel: KernelId::VisionDetectionAveragePrecisionF32,
			buffers: &ap_buffers,
			push_constants: &ap_push,
			workgroups: [pair_count, 1, 1],
		},
		ComputeDispatch {
			kernel: KernelId::VisionDetectionMeanAveragePrecisionF32,
			buffers: &mean_buffers,
			push_constants: &mean_push,
			workgroups: [1, 1, 1],
		},
	];
	let inputs = [
		predicted_boxes,
		predicted_scores,
		predicted_classes,
		predicted_image_ids,
		target_boxes,
		target_classes,
		target_image_ids,
		iou_thresholds,
	];
	let outputs = [
		&counts,
		&per_class,
		&map_by_threshold,
		&mean_average_precision,
	];
	let attributes = [
		OpAttribute::SignedInteger {
			name: "class_count".into(),
			value: i64::from(class_count),
		},
		OpAttribute::Float {
			name: "score_threshold".into(),
			value: f64::from(score_threshold),
		},
	];
	engine.record_split_semantic(
		&dispatches,
		SemanticDispatch {
			contract: crate::core::operation::vision::EVALUATE,
			inputs: &inputs,
			outputs: &outputs,
			attributes: &attributes,
		},
	)?;
	Ok(DetectionMetricsResult {
		counts,
		per_class,
		mean_average_precision_by_threshold: map_by_threshold,
		mean_average_precision,
	})
}

/// Evaluate multiclass semantic-segmentation labels.
///
/// Labels outside `[0, class_count)` are ignored. The operation records a
/// confusion accumulator followed by a metrics kernel as one semantic Vision
/// operation.
///
/// # Errors
///
/// Returns an error unless inputs are equal-shaped nonempty I32 matrices on one
/// engine, `class_count` is positive, and all sizes fit the shader ABI.
pub fn evaluate_segmentation(
	predicted: &Matrix,
	target: &Matrix,
	class_count: i32,
) -> Result<SegmentationMetricsResult> {
	let element_count = equal_nonempty(
		predicted,
		target,
		DType::I32,
		"vision::evaluate_segmentation",
	)?;
	let classes = positive_class_count(class_count, "vision::evaluate_segmentation")?;
	let confusion_elements = classes
		.checked_mul(classes)
		.ok_or_else(|| Error::out_of_range("vision::evaluate_segmentation confusion exceeds u32"))?;
	let per_class_elements = classes
		.checked_mul(4)
		.ok_or_else(|| Error::out_of_range("vision::evaluate_segmentation metrics exceed u32"))?;
	let engine = predicted.engine_handle();
	let confusion = allocate(engine, &[classes, classes], confusion_elements, DType::U32)?;
	let per_class = allocate(engine, &[classes, 4], per_class_elements, DType::F32)?;
	let mean_iou = allocate(engine, &[1], 1, DType::F32)?;
	let pixel_accuracy = allocate(engine, &[1], 1, DType::F32)?;
	let confusion_buffers = [
		BufferBinding::read(predicted.storage()),
		BufferBinding::read(target.storage()),
		BufferBinding::read_write(confusion.storage()),
	];
	let confusion_push = [PushConstant::U32(element_count), PushConstant::U32(classes)];
	let clear_buffers = [BufferBinding::write(confusion.storage())];
	let clear_push = [PushConstant::U32(confusion_elements)];
	let metrics_buffers = [
		BufferBinding::read(confusion.storage()),
		BufferBinding::write(per_class.storage()),
		BufferBinding::write(mean_iou.storage()),
		BufferBinding::write(pixel_accuracy.storage()),
	];
	let metrics_push = [PushConstant::U32(classes)];
	let dispatches = [
		ComputeDispatch {
			kernel: KernelId::VisionSegmentationConfusionClearU32,
			buffers: &clear_buffers,
			push_constants: &clear_push,
			workgroups: KernelId::VisionSegmentationConfusionClearU32
				.linear_workgroups(confusion_elements),
		},
		ComputeDispatch {
			kernel: KernelId::VisionSegmentationConfusionI32,
			buffers: &confusion_buffers,
			push_constants: &confusion_push,
			workgroups: KernelId::VisionSegmentationConfusionI32.linear_workgroups(element_count),
		},
		ComputeDispatch {
			kernel: KernelId::VisionSegmentationMetricsF32,
			buffers: &metrics_buffers,
			push_constants: &metrics_push,
			workgroups: [1, 1, 1],
		},
	];
	let inputs = [predicted, target];
	let outputs = [&confusion, &per_class, &mean_iou, &pixel_accuracy];
	let attributes = [OpAttribute::SignedInteger {
		name: "class_count".into(),
		value: i64::from(class_count),
	}];
	engine.record_split_semantic(
		&dispatches,
		SemanticDispatch {
			contract: crate::core::operation::vision::EVALUATE_SEGMENTATION,
			inputs: &inputs,
			outputs: &outputs,
			attributes: &attributes,
		},
	)?;
	Ok(SegmentationMetricsResult {
		confusion,
		per_class,
		mean_iou,
		pixel_accuracy,
	})
}

#[allow(clippy::too_many_arguments)]
fn record_confusion(
	predicted: &Matrix,
	target: &Matrix,
	class_count: u32,
	element_count: u32,
	clear_kernel: KernelId,
	count_kernel: KernelId,
	contract: OperationContract,
	shape: Vec<usize>,
	output_elements: usize,
) -> Result<Matrix> {
	let output = Matrix::allocate(
		predicted.engine_handle(),
		shape,
		output_elements,
		DType::U32,
	)?;
	let clear_count = u32::try_from(output_elements)
		.map_err(|_| Error::out_of_range("vision confusion output exceeds u32"))?;
	let clear_buffers = [BufferBinding::write(output.storage())];
	let clear_push = [PushConstant::U32(clear_count)];
	let count_buffers = [
		BufferBinding::read(predicted.storage()),
		BufferBinding::read(target.storage()),
		BufferBinding::read_write(output.storage()),
	];
	let count_push = [
		PushConstant::U32(element_count),
		PushConstant::U32(class_count),
	];
	let dispatches = [
		ComputeDispatch {
			kernel: clear_kernel,
			buffers: &clear_buffers,
			push_constants: &clear_push,
			workgroups: clear_kernel.linear_workgroups(clear_count),
		},
		ComputeDispatch {
			kernel: count_kernel,
			buffers: &count_buffers,
			push_constants: &count_push,
			workgroups: count_kernel.linear_workgroups(element_count),
		},
	];
	let inputs = [predicted, target];
	let outputs = [&output];
	let attributes = [OpAttribute::SignedInteger {
		name: "class_count".into(),
		value: i64::from(class_count),
	}];
	predicted.engine_handle().record_split_semantic(
		&dispatches,
		SemanticDispatch {
			contract,
			inputs: &inputs,
			outputs: &outputs,
			attributes: &attributes,
		},
	)?;
	Ok(output)
}

fn box_rows(boxes: &Matrix, label: &str) -> Result<u32> {
	if boxes.dtype() != DType::F32 {
		return Err(Error::invalid_argument(format!(
			"vision box input {label} must use f32 storage; received {}",
			boxes.dtype().token()
		)));
	}
	let [rows, columns] = boxes.shape() else {
		return Err(Error::invalid_argument(format!(
			"vision box input {label} must have shape [N, 4]; received {:?}",
			boxes.shape()
		)));
	};
	if *rows == 0 || *columns != 4 {
		return Err(Error::invalid_argument(format!(
			"vision box input {label} must have nonempty shape [N, 4]; received {:?}",
			boxes.shape()
		)));
	}
	u32::try_from(*rows)
		.map_err(|_| Error::out_of_range(format!("vision box input {label} row count exceeds u32")))
}

fn vector(matrix: &Matrix, label: &str, length: usize, dtype: DType) -> Result<()> {
	if matrix.shape() != [length] || matrix.dtype() != dtype {
		return Err(Error::invalid_argument(format!(
			"vision input {label} must have shape [{length}] and dtype {}; received {:?} {}",
			dtype.token(),
			matrix.shape(),
			matrix.dtype().token()
		)));
	}
	Ok(())
}

fn nonempty_vector(matrix: &Matrix, label: &str, dtype: DType) -> Result<u32> {
	let [length] = matrix.shape() else {
		return Err(Error::invalid_argument(format!(
			"vision input {label} must be a nonempty {} vector; received {:?}",
			dtype.token(),
			matrix.shape()
		)));
	};
	if *length == 0 || matrix.dtype() != dtype {
		return Err(Error::invalid_argument(format!(
			"vision input {label} must be a nonempty {} vector; received {:?} {}",
			dtype.token(),
			matrix.shape(),
			matrix.dtype().token()
		)));
	}
	u32::try_from(*length)
		.map_err(|_| Error::out_of_range(format!("vision input {label} length exceeds u32")))
}

fn equal_nonempty(left: &Matrix, right: &Matrix, dtype: DType, operation: &str) -> Result<u32> {
	if left.shape() != right.shape() || left.dtype() != dtype || right.dtype() != dtype {
		return Err(Error::invalid_argument(format!(
			"{operation} requires equal-shaped {} inputs; received {:?} {} and {:?} {}",
			dtype.token(),
			left.shape(),
			left.dtype().token(),
			right.shape(),
			right.dtype().token()
		)));
	}
	if left.num_elements() == 0 {
		return Err(Error::invalid_argument(format!(
			"{operation} inputs must be nonempty"
		)));
	}
	validate_same_engine(left, &[right], operation)?;
	u32::try_from(left.num_elements())
		.map_err(|_| Error::out_of_range(format!("{operation} element count exceeds u32")))
}

fn validate_same_engine(reference: &Matrix, others: &[&Matrix], operation: &str) -> Result<()> {
	if others
		.iter()
		.any(|matrix| !reference.engine_handle().same_as(matrix.engine_handle()))
	{
		return Err(Error::invalid_argument(format!(
			"{operation} inputs must belong to the same engine"
		)));
	}
	Ok(())
}

fn positive_class_count(class_count: i32, operation: &str) -> Result<u32> {
	u32::try_from(class_count)
		.ok()
		.filter(|count| *count != 0)
		.ok_or_else(|| Error::invalid_argument(format!("{operation} class_count must be positive")))
}

fn allocate(
	engine: &crate::runtime::EngineHandle,
	shape: &[u32],
	element_count: u32,
	dtype: DType,
) -> Result<Matrix> {
	Matrix::allocate(
		engine,
		shape.iter().map(|value| *value as usize).collect(),
		element_count as usize,
		dtype,
	)
}
