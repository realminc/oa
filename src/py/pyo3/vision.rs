use pyo3::prelude::*;

use crate::matrix::PythonMatrix;

#[path = "vision/vision.gen.rs"]
mod generated;

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

pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
	module.add_class::<PythonNmsResult>()?;
	module.add_class::<PythonDetectionMetricsResult>()?;
	module.add_class::<PythonSegmentationMetricsResult>()?;
	generated::register(module)?;
	Ok(())
}
