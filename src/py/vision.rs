use pyo3::prelude::*;

use crate::{error::python_error, matrix::PythonMatrix};

// ── Result wrappers ──────────────────────────────────────────────────────────

#[pyclass(name = "NmsResult", unsendable)]
pub(crate) struct PythonNmsResult {
	pub indices: PythonMatrix,
	pub count: PythonMatrix,
}

#[pymethods]
impl PythonNmsResult {
	#[getter]
	fn indices(&self) -> PythonMatrix {
		PythonMatrix::wrap(self.indices.inner.clone())
	}

	#[getter]
	fn count(&self) -> PythonMatrix {
		PythonMatrix::wrap(self.count.inner.clone())
	}
}

#[pyclass(name = "DetectionMetricsResult", unsendable)]
pub(crate) struct PythonDetectionMetricsResult {
	pub counts: PythonMatrix,
	pub per_class: PythonMatrix,
	pub mean_average_precision_by_threshold: PythonMatrix,
	pub mean_average_precision: PythonMatrix,
}

#[pymethods]
impl PythonDetectionMetricsResult {
	#[getter]
	fn counts(&self) -> PythonMatrix {
		PythonMatrix::wrap(self.counts.inner.clone())
	}

	#[getter]
	fn per_class(&self) -> PythonMatrix {
		PythonMatrix::wrap(self.per_class.inner.clone())
	}

	#[getter]
	fn mean_average_precision_by_threshold(&self) -> PythonMatrix {
		PythonMatrix::wrap(self.mean_average_precision_by_threshold.inner.clone())
	}

	#[getter]
	fn mean_average_precision(&self) -> PythonMatrix {
		PythonMatrix::wrap(self.mean_average_precision.inner.clone())
	}
}

#[pyclass(name = "SegmentationMetricsResult", unsendable)]
pub(crate) struct PythonSegmentationMetricsResult {
	pub confusion: PythonMatrix,
	pub per_class: PythonMatrix,
	pub mean_iou: PythonMatrix,
	pub pixel_accuracy: PythonMatrix,
}

#[pymethods]
impl PythonSegmentationMetricsResult {
	#[getter]
	fn confusion(&self) -> PythonMatrix {
		PythonMatrix::wrap(self.confusion.inner.clone())
	}

	#[getter]
	fn per_class(&self) -> PythonMatrix {
		PythonMatrix::wrap(self.per_class.inner.clone())
	}

	#[getter]
	fn mean_iou(&self) -> PythonMatrix {
		PythonMatrix::wrap(self.mean_iou.inner.clone())
	}

	#[getter]
	fn pixel_accuracy(&self) -> PythonMatrix {
		PythonMatrix::wrap(self.pixel_accuracy.inner.clone())
	}
}

// ── Stateless operations ─────────────────────────────────────────────────────

#[pyfunction]
pub(crate) fn vision_box_iou(
	boxes1: &PythonMatrix,
	boxes2: &PythonMatrix,
) -> PyResult<PythonMatrix> {
	oa::vision::box_iou(&boxes1.inner, &boxes2.inner)
		.map(PythonMatrix::wrap)
		.map_err(python_error)
}

#[pyfunction]
pub(crate) fn vision_confusion_matrix(
	predictions: &PythonMatrix,
	targets: &PythonMatrix,
	num_classes: i32,
) -> PyResult<PythonMatrix> {
	oa::vision::confusion_matrix(&predictions.inner, &targets.inner, num_classes)
		.map(PythonMatrix::wrap)
		.map_err(python_error)
}

#[pyfunction]
pub(crate) fn vision_binary_mask_counts(
	predicted: &PythonMatrix,
	target: &PythonMatrix,
) -> PyResult<PythonMatrix> {
	oa::vision::binary_mask_counts(&predicted.inner, &target.inner)
		.map(PythonMatrix::wrap)
		.map_err(python_error)
}

#[pyfunction]
#[pyo3(signature = (boxes, scores, classes, iou_threshold=0.45, score_threshold=0.0, max_detections=100, class_agnostic=false))]
pub(crate) fn vision_nms(
	boxes: &PythonMatrix,
	scores: &PythonMatrix,
	classes: &PythonMatrix,
	iou_threshold: f32,
	score_threshold: f32,
	max_detections: i32,
	class_agnostic: bool,
) -> PyResult<PythonNmsResult> {
	let config = oa::vision::NmsConfig {
		iou_threshold,
		score_threshold,
		max_detections,
		class_agnostic,
	};
	oa::vision::nms(&boxes.inner, &scores.inner, &classes.inner, config)
		.map(|r| PythonNmsResult {
			indices: PythonMatrix::wrap(r.indices),
			count: PythonMatrix::wrap(r.count),
		})
		.map_err(python_error)
}

#[pyfunction]
#[pyo3(signature = (predicted_boxes, predicted_scores, predicted_classes, predicted_images, target_boxes, target_classes, target_images, thresholds, num_classes, score_threshold))]
#[allow(clippy::too_many_arguments)]
pub(crate) fn vision_evaluate(
	predicted_boxes: &PythonMatrix,
	predicted_scores: &PythonMatrix,
	predicted_classes: &PythonMatrix,
	predicted_images: &PythonMatrix,
	target_boxes: &PythonMatrix,
	target_classes: &PythonMatrix,
	target_images: &PythonMatrix,
	thresholds: &PythonMatrix,
	num_classes: i32,
	score_threshold: f32,
) -> PyResult<PythonDetectionMetricsResult> {
	oa::vision::evaluate(
		&predicted_boxes.inner,
		&predicted_scores.inner,
		&predicted_classes.inner,
		&predicted_images.inner,
		&target_boxes.inner,
		&target_classes.inner,
		&target_images.inner,
		&thresholds.inner,
		num_classes,
		score_threshold,
	)
	.map(|r| PythonDetectionMetricsResult {
		counts: PythonMatrix::wrap(r.counts),
		per_class: PythonMatrix::wrap(r.per_class),
		mean_average_precision_by_threshold: PythonMatrix::wrap(r.mean_average_precision_by_threshold),
		mean_average_precision: PythonMatrix::wrap(r.mean_average_precision),
	})
	.map_err(python_error)
}

#[pyfunction]
pub(crate) fn vision_evaluate_segmentation(
	predicted: &PythonMatrix,
	target: &PythonMatrix,
	num_classes: i32,
) -> PyResult<PythonSegmentationMetricsResult> {
	oa::vision::evaluate_segmentation(&predicted.inner, &target.inner, num_classes)
		.map(|r| PythonSegmentationMetricsResult {
			confusion: PythonMatrix::wrap(r.confusion),
			per_class: PythonMatrix::wrap(r.per_class),
			mean_iou: PythonMatrix::wrap(r.mean_iou),
			pixel_accuracy: PythonMatrix::wrap(r.pixel_accuracy),
		})
		.map_err(python_error)
}

pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
	module.add_class::<PythonNmsResult>()?;
	module.add_class::<PythonDetectionMetricsResult>()?;
	module.add_class::<PythonSegmentationMetricsResult>()?;
	module.add_function(wrap_pyfunction!(vision_box_iou, module)?)?;
	module.add_function(wrap_pyfunction!(vision_confusion_matrix, module)?)?;
	module.add_function(wrap_pyfunction!(vision_binary_mask_counts, module)?)?;
	module.add_function(wrap_pyfunction!(vision_nms, module)?)?;
	module.add_function(wrap_pyfunction!(vision_evaluate, module)?)?;
	module.add_function(wrap_pyfunction!(vision_evaluate_segmentation, module)?)?;
	Ok(())
}
