//! Constant-model Kalman predictor/corrector, following `cv::KalmanFilter`.
//!
//! The filter carries `dynam`-dimensional state, `measure`-dimensional
//! measurements, and an optional `control`-dimensional input. All state and
//! covariance math runs in `f64` on the host; callers pass and receive F32
//! [`Matrix`] values exactly as the other host estimators in this module.
//!
//! `predict` advances `state_post` through the transition (plus optional
//! control) into `state_pre`; `correct` fuses a measurement into
//! `state_post` via the Kalman gain `P_pre * H' * (H * P_pre * H' + R)^-1`.

use crate::{Error, Result, linalg::solve_dynamic, matrix::Matrix};

/// Discrete linear Kalman filter over fixed-size state, measurement, and
/// control vectors.
///
/// Construct with [`KalmanFilter::new`], assign the model matrices with the
/// `set_*` methods, seed [`KalmanFilter::set_state_post`] and
/// [`KalmanFilter::set_error_cov_post`], then alternate `predict`/`correct`
/// calls.
pub struct KalmanFilter {
	dynam_params: usize,
	measure_params: usize,
	control_params: usize,
	state_pre: Vec<f64>,
	state_post: Vec<f64>,
	transition_matrix: Vec<f64>,
	control_matrix: Vec<f64>,
	measurement_matrix: Vec<f64>,
	process_noise_cov: Vec<f64>,
	measurement_noise_cov: Vec<f64>,
	error_cov_pre: Vec<f64>,
	error_cov_post: Vec<f64>,
}

impl KalmanFilter {
	/// Creates a filter with zero-filled model matrices and covariances,
	/// matching `cv::KalmanFilter::KalmanFilter`.
	///
	/// # Errors
	///
	/// Returns an error unless `dynam_params` and `measure_params` are
	/// nonzero; `control_params` may be zero to disable the control input.
	pub fn new(dynam_params: usize, measure_params: usize, control_params: usize) -> Result<Self> {
		if dynam_params == 0 || measure_params == 0 {
			return Err(Error::invalid_argument(
				"vision::KalmanFilter requires nonzero state and measurement dimensions",
			));
		}
		Ok(Self {
			dynam_params,
			measure_params,
			control_params,
			state_pre: vec![0.0; dynam_params],
			state_post: vec![0.0; dynam_params],
			transition_matrix: vec![0.0; dynam_params * dynam_params],
			control_matrix: vec![0.0; dynam_params * control_params],
			measurement_matrix: vec![0.0; measure_params * dynam_params],
			process_noise_cov: vec![0.0; dynam_params * dynam_params],
			measurement_noise_cov: vec![0.0; measure_params * measure_params],
			error_cov_pre: vec![0.0; dynam_params * dynam_params],
			error_cov_post: vec![0.0; dynam_params * dynam_params],
		})
	}

	/// Assigns the `dynam x dynam` state-transition matrix `A`.
	///
	/// # Errors
	///
	/// Returns an error unless `matrix` is `[dynam, dynam]` F32.
	pub fn set_transition_matrix(&mut self, matrix: &Matrix) -> Result<()> {
		self.transition_matrix = read_rect(
			matrix,
			self.dynam_params,
			self.dynam_params,
			"vision::KalmanFilter::set_transition_matrix",
		)?;
		Ok(())
	}

	/// Assigns the `dynam x control` control-input matrix `B`.
	///
	/// # Errors
	///
	/// Returns an error unless `matrix` is `[dynam, control]` F32.
	pub fn set_control_matrix(&mut self, matrix: &Matrix) -> Result<()> {
		self.control_matrix = read_rect(
			matrix,
			self.dynam_params,
			self.control_params,
			"vision::KalmanFilter::set_control_matrix",
		)?;
		Ok(())
	}

	/// Assigns the `measure x dynam` measurement matrix `H`.
	///
	/// # Errors
	///
	/// Returns an error unless `matrix` is `[measure, dynam]` F32.
	pub fn set_measurement_matrix(&mut self, matrix: &Matrix) -> Result<()> {
		self.measurement_matrix = read_rect(
			matrix,
			self.measure_params,
			self.dynam_params,
			"vision::KalmanFilter::set_measurement_matrix",
		)?;
		Ok(())
	}

	/// Assigns the `dynam x dynam` process-noise covariance `Q`.
	///
	/// # Errors
	///
	/// Returns an error unless `matrix` is `[dynam, dynam]` F32.
	pub fn set_process_noise_cov(&mut self, matrix: &Matrix) -> Result<()> {
		self.process_noise_cov = read_rect(
			matrix,
			self.dynam_params,
			self.dynam_params,
			"vision::KalmanFilter::set_process_noise_cov",
		)?;
		Ok(())
	}

	/// Assigns the `measure x measure` measurement-noise covariance `R`.
	///
	/// # Errors
	///
	/// Returns an error unless `matrix` is `[measure, measure]` F32.
	pub fn set_measurement_noise_cov(&mut self, matrix: &Matrix) -> Result<()> {
		self.measurement_noise_cov = read_rect(
			matrix,
			self.measure_params,
			self.measure_params,
			"vision::KalmanFilter::set_measurement_noise_cov",
		)?;
		Ok(())
	}

	/// Seeds the corrected state estimate consumed by the next `predict`
	/// call.
	///
	/// # Errors
	///
	/// Returns an error unless `state` holds exactly `dynam` F32 values.
	pub fn set_state_post(&mut self, state: &Matrix) -> Result<()> {
		self.state_post = read_vector(
			state,
			self.dynam_params,
			"vision::KalmanFilter::set_state_post",
		)?;
		Ok(())
	}

	/// Seeds the corrected `dynam x dynam` error covariance `P`.
	///
	/// # Errors
	///
	/// Returns an error unless `matrix` is `[dynam, dynam]` F32.
	pub fn set_error_cov_post(&mut self, matrix: &Matrix) -> Result<()> {
		self.error_cov_post = read_rect(
			matrix,
			self.dynam_params,
			self.dynam_params,
			"vision::KalmanFilter::set_error_cov_post",
		)?;
		Ok(())
	}

	/// Current predicted state `[dynam, 1]` (`state_pre`).
	///
	/// # Errors
	///
	/// Returns an error when the F32 output cannot be materialized.
	pub fn state_pre(&self) -> Result<Matrix> {
		vector_matrix(&self.state_pre)
	}

	/// Current corrected state `[dynam, 1]` (`state_post`).
	///
	/// # Errors
	///
	/// Returns an error when the F32 output cannot be materialized.
	pub fn state_post(&self) -> Result<Matrix> {
		vector_matrix(&self.state_post)
	}

	/// Advances the model one step: `state_pre = A * state_post + B * u`,
	/// `error_cov_pre = A * P * A' + Q`. `control` must be `[control, 1]`
	/// when the filter was built with a nonzero control dimension. Like
	/// `cv::KalmanFilter::predict`, only the `*_pre` fields advance.
	///
	/// Returns `state_pre` as `[dynam, 1]`.
	///
	/// # Errors
	///
	/// Returns an error for a wrongly sized or missing control input.
	pub fn predict(&mut self, control: Option<&Matrix>) -> Result<Matrix> {
		let n = self.dynam_params;
		let control_vec = match control {
			Some(matrix) => Some(read_vector(
				matrix,
				self.control_params,
				"vision::KalmanFilter::predict",
			)?),
			None if self.control_params == 0 => None,
			None => {
				return Err(Error::invalid_argument(
					"vision::KalmanFilter::predict requires a control input",
				));
			}
		};
		let mut next = mat_vec(&self.transition_matrix, &self.state_post, n, n);
		if let Some(u) = &control_vec {
			let driven = mat_vec(&self.control_matrix, u, n, self.control_params);
			for (value, add) in next.iter_mut().zip(driven) {
				*value += add;
			}
		}
		let ap = mat_mul(&self.transition_matrix, &self.error_cov_post, n, n, n);
		let at = transpose(&self.transition_matrix, n, n);
		let mut cov = mat_mul(&ap, &at, n, n, n);
		for (value, noise) in cov.iter_mut().zip(&self.process_noise_cov) {
			*value += noise;
		}
		self.state_pre = next;
		self.error_cov_pre = cov;
		self.state_pre()
	}

	/// Fuses a `measure`-dimensional measurement: gain
	/// `K = P_pre * H' * (H * P_pre * H' + R)^-1`, then
	/// `state_post = state_pre + K * (z - H * state_pre)` and
	/// `error_cov_post = (I - K * H) * P_pre`.
	///
	/// Returns `state_post` as `[dynam, 1]`.
	///
	/// # Errors
	///
	/// Returns an error for a wrongly sized measurement or a singular
	/// innovation covariance `H * P_pre * H' + R`.
	pub fn correct(&mut self, measurement: &Matrix) -> Result<Matrix> {
		let n = self.dynam_params;
		let m = self.measure_params;
		let z = read_vector(measurement, m, "vision::KalmanFilter::correct")?;
		let hp = mat_mul(&self.measurement_matrix, &self.error_cov_pre, m, n, n);
		let ht = transpose(&self.measurement_matrix, m, n);
		let mut innovation = mat_mul(&hp, &ht, m, n, m);
		for (value, noise) in innovation.iter_mut().zip(&self.measurement_noise_cov) {
			*value += noise;
		}
		let inv = invert(&innovation, m).ok_or_else(|| {
			Error::invalid_argument("vision::KalmanFilter::correct innovation covariance is singular")
		})?;
		// gain = P_pre * H' * S^-1 = (H * P_pre)' * S^-1
		let gain = mat_mul(&transpose(&hp, m, n), &inv, n, m, m);
		let hx = mat_vec(&self.measurement_matrix, &self.state_pre, m, n);
		let residual: Vec<f64> = z.iter().zip(hx).map(|(a, b)| a - b).collect();
		let update = mat_vec(&gain, &residual, n, m);
		let mut post = self.state_pre.clone();
		for (value, add) in post.iter_mut().zip(update) {
			*value += add;
		}
		self.state_post = post;
		let kh = mat_mul(&gain, &self.measurement_matrix, n, m, n);
		let factor: Vec<f64> = identity(n).iter().zip(&kh).map(|(a, b)| a - b).collect();
		self.error_cov_post = mat_mul(&factor, &self.error_cov_pre, n, n, n);
		self.state_post()
	}
}

// ---- dynamic dense helpers -------------------------------------------------

fn mat_vec(a: &[f64], v: &[f64], rows: usize, cols: usize) -> Vec<f64> {
	let mut out = vec![0.0f64; rows];
	for (row, value) in out.iter_mut().enumerate() {
		for col in 0..cols {
			*value += a[row * cols + col] * v[col];
		}
	}
	out
}

fn mat_mul(a: &[f64], b: &[f64], rows: usize, inner: usize, cols: usize) -> Vec<f64> {
	let mut out = vec![0.0f64; rows * cols];
	for row in 0..rows {
		for col in 0..cols {
			let mut sum = 0.0f64;
			for k in 0..inner {
				sum += a[row * inner + k] * b[k * cols + col];
			}
			out[row * cols + col] = sum;
		}
	}
	out
}

fn transpose(a: &[f64], rows: usize, cols: usize) -> Vec<f64> {
	let mut out = vec![0.0f64; rows * cols];
	for row in 0..rows {
		for col in 0..cols {
			out[col * rows + row] = a[row * cols + col];
		}
	}
	out
}

fn identity(n: usize) -> Vec<f64> {
	let mut out = vec![0.0f64; n * n];
	for i in 0..n {
		out[i * n + i] = 1.0;
	}
	out
}

fn invert(a: &[f64], n: usize) -> Option<Vec<f64>> {
	let rows: Vec<Vec<f64>> = a.chunks(n).map(Vec::from).collect();
	let mut inverse = Vec::with_capacity(n * n);
	for column in 0..n {
		let mut unit = vec![0.0f64; n];
		unit[column] = 1.0;
		inverse.extend(solve_dynamic(&mut rows.clone(), &mut unit, n)?);
	}
	// columns solved independently; transpose back to row-major
	Some(transpose(&inverse, n, n))
}

fn vector_matrix(v: &[f64]) -> Result<Matrix> {
	Matrix::new_from_f32(
		[v.len(), 1],
		&v.iter().map(|value| *value as f32).collect::<Vec<_>>(),
	)
}

fn read_vector(matrix: &Matrix, expected: usize, operation: &str) -> Result<Vec<f64>> {
	let values = matrix.read_f32()?;
	if values.len() != expected {
		return Err(Error::invalid_argument(format!(
			"{operation} requires {expected} values"
		)));
	}
	Ok(values.iter().map(|value| f64::from(*value)).collect())
}

fn read_rect(matrix: &Matrix, rows: usize, cols: usize, operation: &str) -> Result<Vec<f64>> {
	if matrix.shape() != [rows, cols] {
		return Err(Error::invalid_argument(format!(
			"{operation} requires a [{rows}, {cols}] matrix"
		)));
	}
	read_vector(matrix, rows * cols, operation)
}
