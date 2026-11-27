use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use crate::error::python_error;
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

#[pyclass(name = "HomographyResult", unsendable)]
pub(crate) struct PythonHomographyResult {
	pub matrix: PythonMatrix,
	pub mask: PythonMatrix,
}

#[pymethods]
impl PythonHomographyResult {
	#[getter]
	fn matrix(&self) -> PythonMatrix {
		PythonMatrix::wrap(self.matrix.inner.clone())
	}

	#[getter]
	fn mask(&self) -> PythonMatrix {
		PythonMatrix::wrap(self.mask.inner.clone())
	}
}

#[pyclass(name = "PnpResult", unsendable)]
pub(crate) struct PythonPnpResult {
	pub rotation_vector: PythonMatrix,
	pub translation_vector: PythonMatrix,
}

#[pymethods]
impl PythonPnpResult {
	#[getter]
	fn rotation_vector(&self) -> PythonMatrix {
		PythonMatrix::wrap(self.rotation_vector.inner.clone())
	}

	#[getter]
	fn translation_vector(&self) -> PythonMatrix {
		PythonMatrix::wrap(self.translation_vector.inner.clone())
	}
}

#[pyclass(name = "PnpRansacResult", unsendable)]
pub(crate) struct PythonPnpRansacResult {
	pub rotation_vector: PythonMatrix,
	pub translation_vector: PythonMatrix,
	pub mask: PythonMatrix,
}

#[pymethods]
impl PythonPnpRansacResult {
	#[getter]
	fn rotation_vector(&self) -> PythonMatrix {
		PythonMatrix::wrap(self.rotation_vector.inner.clone())
	}

	#[getter]
	fn translation_vector(&self) -> PythonMatrix {
		PythonMatrix::wrap(self.translation_vector.inner.clone())
	}

	#[getter]
	fn mask(&self) -> PythonMatrix {
		PythonMatrix::wrap(self.mask.inner.clone())
	}
}

#[pyclass(name = "EpipolarResult", unsendable)]
pub(crate) struct PythonEpipolarResult {
	pub matrix: PythonMatrix,
	pub mask: PythonMatrix,
}

#[pymethods]
impl PythonEpipolarResult {
	#[getter]
	fn matrix(&self) -> PythonMatrix {
		PythonMatrix::wrap(self.matrix.inner.clone())
	}

	#[getter]
	fn mask(&self) -> PythonMatrix {
		PythonMatrix::wrap(self.mask.inner.clone())
	}
}

#[pyclass(name = "RecoverPoseResult", unsendable)]
pub(crate) struct PythonRecoverPoseResult {
	pub rotation: PythonMatrix,
	pub translation: PythonMatrix,
	pub mask: PythonMatrix,
}

#[pymethods]
impl PythonRecoverPoseResult {
	#[getter]
	fn rotation(&self) -> PythonMatrix {
		PythonMatrix::wrap(self.rotation.inner.clone())
	}

	#[getter]
	fn translation(&self) -> PythonMatrix {
		PythonMatrix::wrap(self.translation.inner.clone())
	}

	#[getter]
	fn mask(&self) -> PythonMatrix {
		PythonMatrix::wrap(self.mask.inner.clone())
	}
}

#[pyclass(name = "CalibrateCameraResult", unsendable)]
pub(crate) struct PythonCalibrateCameraResult {
	pub camera: PythonMatrix,
	pub distortion: PythonMatrix,
	pub rotations: PythonMatrix,
	pub translations: PythonMatrix,
	pub rms_error: PythonMatrix,
}

#[pymethods]
impl PythonCalibrateCameraResult {
	#[getter]
	fn camera(&self) -> PythonMatrix {
		PythonMatrix::wrap(self.camera.inner.clone())
	}

	#[getter]
	fn distortion(&self) -> PythonMatrix {
		PythonMatrix::wrap(self.distortion.inner.clone())
	}

	#[getter]
	fn rotations(&self) -> PythonMatrix {
		PythonMatrix::wrap(self.rotations.inner.clone())
	}

	#[getter]
	fn translations(&self) -> PythonMatrix {
		PythonMatrix::wrap(self.translations.inner.clone())
	}

	#[getter]
	fn rms_error(&self) -> PythonMatrix {
		PythonMatrix::wrap(self.rms_error.inner.clone())
	}
}

fn estimation_method(token: &str) -> PyResult<oa::vision::EstimationMethod> {
	match token {
		"least_squares" => Ok(oa::vision::EstimationMethod::LeastSquares),
		"ransac" => Ok(oa::vision::EstimationMethod::Ransac),
		_ => Err(PyValueError::new_err(format!(
			"unknown estimation method: {token}"
		))),
	}
}

/// Estimate a 3x3 homography from `[N, 2]` point correspondences.
#[pyfunction]
#[pyo3(signature = (source, destination, method="ransac", ransac_threshold=3.0, max_iterations=2000, seed=0), text_signature = "(source, destination, method='ransac', ransac_threshold=3.0, max_iterations=2000, seed=0)")]
fn vision_find_homography(
	source: &PythonMatrix,
	destination: &PythonMatrix,
	method: &str,
	ransac_threshold: f32,
	max_iterations: i32,
	seed: u64,
) -> PyResult<PythonHomographyResult> {
	oa::vision::find_homography(
		&source.inner,
		&destination.inner,
		estimation_method(method)?,
		ransac_threshold,
		max_iterations,
		seed,
	)
	.map(|result| PythonHomographyResult {
		matrix: PythonMatrix::wrap(result.matrix),
		mask: PythonMatrix::wrap(result.mask),
	})
	.map_err(python_error)
}

/// Solve camera pose from `[N, 3]` object and `[N, 2]` image points.
#[pyfunction]
#[pyo3(signature = (object, image, camera), text_signature = "(object, image, camera)")]
fn vision_solve_pnp(
	object: &PythonMatrix,
	image: &PythonMatrix,
	camera: &PythonMatrix,
) -> PyResult<PythonPnpResult> {
	oa::vision::solve_pnp(&object.inner, &image.inner, &camera.inner)
		.map(|result| PythonPnpResult {
			rotation_vector: PythonMatrix::wrap(result.rotation_vector),
			translation_vector: PythonMatrix::wrap(result.translation_vector),
		})
		.map_err(python_error)
}

/// Robust RANSAC camera pose from 3D-2D correspondences.
#[pyfunction]
#[pyo3(signature = (object, image, camera, reprojection_threshold=8.0, max_iterations=100, seed=0), text_signature = "(object, image, camera, reprojection_threshold=8.0, max_iterations=100, seed=0)")]
fn vision_solve_pnp_ransac(
	object: &PythonMatrix,
	image: &PythonMatrix,
	camera: &PythonMatrix,
	reprojection_threshold: f32,
	max_iterations: i32,
	seed: u64,
) -> PyResult<PythonPnpRansacResult> {
	oa::vision::solve_pnp_ransac(
		&object.inner,
		&image.inner,
		&camera.inner,
		reprojection_threshold,
		max_iterations,
		seed,
	)
	.map(|result| PythonPnpRansacResult {
		rotation_vector: PythonMatrix::wrap(result.rotation_vector),
		translation_vector: PythonMatrix::wrap(result.translation_vector),
		mask: PythonMatrix::wrap(result.mask),
	})
	.map_err(python_error)
}

/// Convert between a `[3, 1]` Rodrigues vector and a `[3, 3]` rotation.
#[pyfunction]
#[pyo3(signature = (input), text_signature = "(input)")]
fn vision_rodrigues(input: &PythonMatrix) -> PyResult<PythonMatrix> {
	oa::vision::rodrigues(&input.inner)
		.map(PythonMatrix::wrap)
		.map_err(python_error)
}

/// Project `[N, 3]` object points through pose, intrinsics, and distortion.
#[pyfunction]
#[pyo3(signature = (object, rvec, tvec, camera, distortion), text_signature = "(object, rvec, tvec, camera, distortion)")]
fn vision_project_points(
	object: &PythonMatrix,
	rvec: &PythonMatrix,
	tvec: &PythonMatrix,
	camera: &PythonMatrix,
	distortion: &PythonMatrix,
) -> PyResult<PythonMatrix> {
	oa::vision::project_points(
		&object.inner,
		&rvec.inner,
		&tvec.inner,
		&camera.inner,
		&distortion.inner,
	)
	.map(PythonMatrix::wrap)
	.map_err(python_error)
}

/// Estimate the 3x3 fundamental matrix from `[N, 2]` correspondences.
#[pyfunction]
#[pyo3(signature = (points1, points2, method="ransac", ransac_threshold=1.0, max_iterations=1000, seed=0), text_signature = "(points1, points2, method='ransac', ransac_threshold=1.0, max_iterations=1000, seed=0)")]
fn vision_find_fundamental(
	points1: &PythonMatrix,
	points2: &PythonMatrix,
	method: &str,
	ransac_threshold: f32,
	max_iterations: i32,
	seed: u64,
) -> PyResult<PythonEpipolarResult> {
	oa::vision::find_fundamental(
		&points1.inner,
		&points2.inner,
		estimation_method(method)?,
		ransac_threshold,
		max_iterations,
		seed,
	)
	.map(|result| PythonEpipolarResult {
		matrix: PythonMatrix::wrap(result.matrix),
		mask: PythonMatrix::wrap(result.mask),
	})
	.map_err(python_error)
}

/// Estimate the 3x3 essential matrix under a `[3, 3]` camera matrix.
#[pyfunction]
#[pyo3(signature = (points1, points2, camera, method="ransac", ransac_threshold=1.0, max_iterations=1000, seed=0), text_signature = "(points1, points2, camera, method='ransac', ransac_threshold=1.0, max_iterations=1000, seed=0)")]
fn vision_find_essential(
	points1: &PythonMatrix,
	points2: &PythonMatrix,
	camera: &PythonMatrix,
	method: &str,
	ransac_threshold: f32,
	max_iterations: i32,
	seed: u64,
) -> PyResult<PythonEpipolarResult> {
	oa::vision::find_essential(
		&points1.inner,
		&points2.inner,
		&camera.inner,
		estimation_method(method)?,
		ransac_threshold,
		max_iterations,
		seed,
	)
	.map(|result| PythonEpipolarResult {
		matrix: PythonMatrix::wrap(result.matrix),
		mask: PythonMatrix::wrap(result.mask),
	})
	.map_err(python_error)
}

/// Recover relative camera pose `(R, t)` from an essential matrix.
#[pyfunction]
#[pyo3(signature = (essential, points1, points2, camera), text_signature = "(essential, points1, points2, camera)")]
fn vision_recover_pose(
	essential: &PythonMatrix,
	points1: &PythonMatrix,
	points2: &PythonMatrix,
	camera: &PythonMatrix,
) -> PyResult<PythonRecoverPoseResult> {
	oa::vision::recover_pose(
		&essential.inner,
		&points1.inner,
		&points2.inner,
		&camera.inner,
	)
	.map(|result| PythonRecoverPoseResult {
		rotation: PythonMatrix::wrap(result.rotation),
		translation: PythonMatrix::wrap(result.translation),
		mask: PythonMatrix::wrap(result.mask),
	})
	.map_err(python_error)
}

/// Triangulate correspondences under two `[3, 4]` projection matrices.
#[pyfunction]
#[pyo3(signature = (projection1, projection2, points1, points2), text_signature = "(projection1, projection2, points1, points2)")]
fn vision_triangulate_points(
	projection1: &PythonMatrix,
	projection2: &PythonMatrix,
	points1: &PythonMatrix,
	points2: &PythonMatrix,
) -> PyResult<PythonMatrix> {
	oa::vision::triangulate_points(
		&projection1.inner,
		&projection2.inner,
		&points1.inner,
		&points2.inner,
	)
	.map(PythonMatrix::wrap)
	.map_err(python_error)
}

/// Build a `[1, 2, height, width]` undistort+rectify map for `remap`.
#[pyfunction]
#[pyo3(signature = (camera, distortion, width, height, rectification=None, projection=None), text_signature = "(camera, distortion, width, height, rectification=None, projection=None)")]
fn vision_undistort_rectify_map(
	camera: &PythonMatrix,
	distortion: &PythonMatrix,
	width: i32,
	height: i32,
	rectification: Option<&PythonMatrix>,
	projection: Option<&PythonMatrix>,
) -> PyResult<PythonMatrix> {
	oa::vision::undistort_rectify_map(
		&camera.inner,
		&distortion.inner,
		rectification.map(|m| &m.inner),
		projection.map(|m| &m.inner),
		width,
		height,
	)
	.map(PythonMatrix::wrap)
	.map_err(python_error)
}

/// Undistort `[N, 2]` pixel points with optional rectification/projection.
#[pyfunction]
#[pyo3(signature = (points, camera, distortion, rectification=None, projection=None), text_signature = "(points, camera, distortion, rectification=None, projection=None)")]
fn vision_undistort_points(
	points: &PythonMatrix,
	camera: &PythonMatrix,
	distortion: &PythonMatrix,
	rectification: Option<&PythonMatrix>,
	projection: Option<&PythonMatrix>,
) -> PyResult<PythonMatrix> {
	oa::vision::undistort_points(
		&points.inner,
		&camera.inner,
		&distortion.inner,
		rectification.map(|m| &m.inner),
		projection.map(|m| &m.inner),
	)
	.map(PythonMatrix::wrap)
	.map_err(python_error)
}

/// Calibrate a camera from `[V, N, 3]` planar object and `[V, N, 2]` image grids.
#[pyfunction]
#[pyo3(signature = (object, image, refine_iterations=20), text_signature = "(object, image, refine_iterations=20)")]
fn vision_calibrate_camera(
	object: &PythonMatrix,
	image: &PythonMatrix,
	refine_iterations: i32,
) -> PyResult<PythonCalibrateCameraResult> {
	oa::vision::calibrate_camera(&object.inner, &image.inner, refine_iterations)
		.map(|result| PythonCalibrateCameraResult {
			camera: PythonMatrix::wrap(result.camera),
			distortion: PythonMatrix::wrap(result.distortion),
			rotations: PythonMatrix::wrap(result.rotations),
			translations: PythonMatrix::wrap(result.translations),
			rms_error: PythonMatrix::wrap(result.rms_error),
		})
		.map_err(python_error)
}

#[pyclass(name = "KalmanFilter", unsendable)]
pub(crate) struct PythonKalmanFilter {
	inner: oa::vision::KalmanFilter,
}

#[pymethods]
impl PythonKalmanFilter {
	/// Create a filter over `dynam`-dimensional state, `measure`-dimensional
	/// measurements, and a `control`-dimensional input (0 disables).
	#[new]
	#[pyo3(signature = (dynam_params, measure_params, control_params=0))]
	fn new(dynam_params: usize, measure_params: usize, control_params: usize) -> PyResult<Self> {
		oa::vision::KalmanFilter::new(dynam_params, measure_params, control_params)
			.map(|inner| Self { inner })
			.map_err(python_error)
	}

	fn set_transition_matrix(&mut self, matrix: &PythonMatrix) -> PyResult<()> {
		self
			.inner
			.set_transition_matrix(&matrix.inner)
			.map_err(python_error)
	}

	fn set_control_matrix(&mut self, matrix: &PythonMatrix) -> PyResult<()> {
		self
			.inner
			.set_control_matrix(&matrix.inner)
			.map_err(python_error)
	}

	fn set_measurement_matrix(&mut self, matrix: &PythonMatrix) -> PyResult<()> {
		self
			.inner
			.set_measurement_matrix(&matrix.inner)
			.map_err(python_error)
	}

	fn set_process_noise_cov(&mut self, matrix: &PythonMatrix) -> PyResult<()> {
		self
			.inner
			.set_process_noise_cov(&matrix.inner)
			.map_err(python_error)
	}

	fn set_measurement_noise_cov(&mut self, matrix: &PythonMatrix) -> PyResult<()> {
		self
			.inner
			.set_measurement_noise_cov(&matrix.inner)
			.map_err(python_error)
	}

	fn set_state_post(&mut self, state: &PythonMatrix) -> PyResult<()> {
		self
			.inner
			.set_state_post(&state.inner)
			.map_err(python_error)
	}

	fn set_error_cov_post(&mut self, matrix: &PythonMatrix) -> PyResult<()> {
		self
			.inner
			.set_error_cov_post(&matrix.inner)
			.map_err(python_error)
	}

	/// Predicted state `[dynam, 1]` (state_pre).
	fn state_pre(&self) -> PyResult<PythonMatrix> {
		self
			.inner
			.state_pre()
			.map(PythonMatrix::wrap)
			.map_err(python_error)
	}

	/// Corrected state `[dynam, 1]` (state_post).
	fn state_post(&self) -> PyResult<PythonMatrix> {
		self
			.inner
			.state_post()
			.map(PythonMatrix::wrap)
			.map_err(python_error)
	}

	/// Advance the model one step; returns the predicted state `[dynam, 1]`.
	#[pyo3(signature = (control=None))]
	fn predict(&mut self, control: Option<&PythonMatrix>) -> PyResult<PythonMatrix> {
		self
			.inner
			.predict(control.map(|matrix| &matrix.inner))
			.map(PythonMatrix::wrap)
			.map_err(python_error)
	}

	/// Fuse a measurement; returns the corrected state `[dynam, 1]`.
	fn correct(&mut self, measurement: &PythonMatrix) -> PyResult<PythonMatrix> {
		self
			.inner
			.correct(&measurement.inner)
			.map(PythonMatrix::wrap)
			.map_err(python_error)
	}
}

pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
	module.add_class::<PythonNmsResult>()?;
	module.add_class::<PythonDetectionMetricsResult>()?;
	module.add_class::<PythonSegmentationMetricsResult>()?;
	module.add_class::<PythonHomographyResult>()?;
	module.add_class::<PythonPnpResult>()?;
	module.add_class::<PythonPnpRansacResult>()?;
	module.add_class::<PythonEpipolarResult>()?;
	module.add_class::<PythonRecoverPoseResult>()?;
	module.add_class::<PythonCalibrateCameraResult>()?;
	module.add_class::<PythonKalmanFilter>()?;
	macro_rules! add_functions {
		($($function:ident),+ $(,)?) => { $(module.add_function(wrap_pyfunction!($function, module)?)?;)+ };
	}
	generated::register(module)?;
	add_functions!(
		vision_find_homography,
		vision_solve_pnp,
		vision_solve_pnp_ransac,
		vision_rodrigues,
		vision_project_points,
		vision_find_fundamental,
		vision_find_essential,
		vision_recover_pose,
		vision_triangulate_points,
		vision_undistort_rectify_map,
		vision_undistort_points,
		vision_calibrate_camera,
	);
	Ok(())
}
