//! Host-side geometric estimation: homography, epipolar geometry, camera
//! pose, lens distortion, and calibration.
//!
//! Mirrors the `calib3d` estimator family: `findHomography`, `solvePnP`
//! and `solvePnPRansac`, `findFundamentalMat`/`findEssentialMat`,
//! `recoverPose`, `triangulatePoints`, `Rodrigues`, `initUndistortRectifyMap`,
//! `undistortPoints`, `calibrateCamera`, and `projectPoints`.
//!
//! ## Host/GPU split contract
//!
//! All estimation math runs on the host in f64 and returns plain F32
//! [`Matrix`] values, so no op here records device work and results compose
//! inside captured plans. Raster work stays device-side: a homography feeds
//! [`crate::image::warp_perspective`] and an undistortion map feeds
//! [`crate::image::remap`]. Robust sampling is seeded through the donor PCG32
//! sequence, so a nonzero `seed` reproduces the same consensus set on every
//! call.

use crate::linalg::{damped_least_squares, jacobi_eigen, smallest_right_singular};
use crate::matrix::Matrix;
use crate::{Error, Result};

/// Robust estimator selection shared by the correspondence ops.
///
/// `LeastSquares` consumes every point through the direct linear solver.
/// `Ransac` draws seeded minimal samples, keeps the largest consensus set
/// under the reprojection threshold, then refits on its inliers.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum EstimationMethod {
	LeastSquares,
	Ransac,
}

/// `find_homography` output: the 3x3 transform and the per-point inlier mask
/// (`1.0` inlier, `0.0` outlier; all ones under `LeastSquares`).
pub struct HomographyResult {
	pub matrix: Matrix,
	pub mask: Matrix,
}

/// `solve_pnp` output: rotation as a Rodrigues vector `[3, 1]` and the camera
/// translation `[3, 1]`.
pub struct PnpResult {
	pub rotation_vector: Matrix,
	pub translation_vector: Matrix,
}

/// `solve_pnp_ransac` output: pose plus the per-point inlier mask.
pub struct PnpRansacResult {
	pub rotation_vector: Matrix,
	pub translation_vector: Matrix,
	pub mask: Matrix,
}

/// `find_fundamental`/`find_essential` output: the 3x3 matrix and inlier mask.
pub struct EpipolarResult {
	pub matrix: Matrix,
	pub mask: Matrix,
}

/// `recover_pose` output: camera-2 relative rotation `[3, 3]`, unit
/// translation `[3, 1]`, and the per-point cheirality mask.
pub struct RecoverPoseResult {
	pub rotation: Matrix,
	pub translation: Matrix,
	pub mask: Matrix,
}

/// `calibrate_camera` output: intrinsic matrix `[3, 3]`, distortion
/// coefficients `[5, 1]` ordered `(k1, k2, p1, p2, k3)`, per-view rotation
/// vectors `[V, 3]`, per-view translations `[V, 3]`, and scalar RMS
/// reprojection error.
pub struct CalibrateCameraResult {
	pub camera: Matrix,
	pub distortion: Matrix,
	pub rotations: Matrix,
	pub translations: Matrix,
	pub rms_error: Matrix,
}

// ---- public operations -----------------------------------------------------

/// Estimate the 3x3 homography mapping `source` `[x, y]` points to
/// `destination` points. `LeastSquares` runs normalized DLT over all points;
/// `Ransac` draws `max_iterations` seeded 4-point samples, keeps the largest
/// consensus set under `ransac_threshold` symmetric transfer error (pixels),
/// and refits on its inliers. Both inputs are `[N, 2]` F32 with `N >= 4`.
///
/// # Errors
///
/// Returns an error for mismatched/degenerate inputs, a nonpositive
/// threshold or iteration bound, or when no nonsingular model exists.
pub fn find_homography(
	source: &Matrix,
	destination: &Matrix,
	method: EstimationMethod,
	ransac_threshold: f32,
	max_iterations: i32,
	seed: u64,
) -> Result<HomographyResult> {
	let source = read_points2(source, 4, "find_homography")?;
	let destination = read_points2(destination, 4, "find_homography")?;
	if source.len() != destination.len() {
		return Err(Error::invalid_argument(
			"find_homography source and destination must share point count",
		));
	}
	match method {
		EstimationMethod::LeastSquares => {
			let h = dlt_homography(&source, &destination).ok_or_else(|| degenerate("find_homography"))?;
			Ok(HomographyResult {
				matrix: mat3_f32(h)?,
				mask: ones_mask(source.len())?,
			})
		}
		EstimationMethod::Ransac => {
			if !ransac_threshold.is_finite() || ransac_threshold <= 0.0 {
				return Err(Error::invalid_argument(
					"find_homography ransac_threshold must be positive and finite",
				));
			}
			if max_iterations < 1 {
				return Err(Error::invalid_argument(
					"find_homography max_iterations must be positive",
				));
			}
			let n = source.len();
			let threshold = ransac_threshold as f64;
			let inliers = ransac_consensus(4, n, max_iterations as usize, seed, |sample| {
				let src: Vec<[f64; 2]> = sample.iter().map(|i| source[*i]).collect();
				let dst: Vec<[f64; 2]> = sample.iter().map(|i| destination[*i]).collect();
				dlt_homography(&src, &dst).map(|h| {
					inlier_indices(n, |i| {
						homography_error(&h, &source[i], &destination[i]) <= threshold
					})
				})
			})
			.ok_or_else(|| degenerate("find_homography"))?;
			let src: Vec<[f64; 2]> = inliers.iter().map(|i| source[*i]).collect();
			let dst: Vec<[f64; 2]> = inliers.iter().map(|i| destination[*i]).collect();
			let h = dlt_homography(&src, &dst).ok_or_else(|| degenerate("find_homography"))?;
			Ok(HomographyResult {
				matrix: mat3_f32(h)?,
				mask: index_mask(n, &inliers)?,
			})
		}
	}
}

/// Solve camera pose from `object` `[x, y, z]` world points and `image`
/// `[u, v]` projections under a `camera` `[3, 3]` intrinsic matrix. Runs a
/// normalized DLT pose estimate refined by damped Gauss-Newton on the
/// reprojection residuals, matching `solvePnP` `SOLVEPNP_ITERATIVE`.
/// `object` is `[N, 3]` and `image` `[N, 2]` F32 with `N >= 6`.
///
/// # Errors
///
/// Returns an error for degenerate inputs, a singular camera matrix, or
/// failure to initialize a nonsingular pose.
pub fn solve_pnp(object: &Matrix, image: &Matrix, camera: &Matrix) -> Result<PnpResult> {
	let object = read_points3(object, 4, "solve_pnp")?;
	let image = read_points2(image, 4, "solve_pnp")?;
	if object.len() != image.len() {
		return Err(Error::invalid_argument(
			"solve_pnp object and image must share point count",
		));
	}
	let k = read_mat3(camera, "solve_pnp")?;
	let pose = dlt_pose(&object, &image, &k).ok_or_else(|| degenerate("solve_pnp"))?;
	let pose = refine_pose(&object, &image, &k, pose).ok_or_else(|| degenerate("solve_pnp"))?;
	Ok(PnpResult {
		rotation_vector: vector_f32(&rodrigues_vector(&pose.0))?,
		translation_vector: vector_f32(&[pose.1[0], pose.1[1], pose.1[2]])?,
	})
}

/// Robust `solve_pnp`: draws `iterations` seeded 6-point samples, keeps the
/// largest consensus set under `reprojection_threshold` (pixels), refits by
/// damped Gauss-Newton over the inliers, and returns the inlier mask.
/// `SOLVEPNP_ITERATIVE`-style scoring matches `solvePnPRansac`.
///
/// # Errors
///
/// Returns an error for degenerate inputs, nonpositive bounds, or when no
/// sample yields a nonsingular pose.
pub fn solve_pnp_ransac(
	object: &Matrix,
	image: &Matrix,
	camera: &Matrix,
	reprojection_threshold: f32,
	max_iterations: i32,
	seed: u64,
) -> Result<PnpRansacResult> {
	let object = read_points3(object, 6, "solve_pnp_ransac")?;
	let image = read_points2(image, 6, "solve_pnp_ransac")?;
	if object.len() != image.len() {
		return Err(Error::invalid_argument(
			"solve_pnp_ransac object and image must share point count",
		));
	}
	if !reprojection_threshold.is_finite() || reprojection_threshold <= 0.0 {
		return Err(Error::invalid_argument(
			"solve_pnp_ransac reprojection_threshold must be positive and finite",
		));
	}
	if max_iterations < 1 {
		return Err(Error::invalid_argument(
			"solve_pnp_ransac max_iterations must be positive",
		));
	}
	let k = read_mat3(camera, "solve_pnp_ransac")?;
	let n = object.len();
	let threshold = reprojection_threshold as f64;
	let inliers = ransac_consensus(6, n, max_iterations as usize, seed, |sample| {
		let obj: Vec<[f64; 3]> = sample.iter().map(|i| object[*i]).collect();
		let img: Vec<[f64; 2]> = sample.iter().map(|i| image[*i]).collect();
		dlt_pose(&obj, &img, &k)
			.and_then(|pose| refine_pose(&obj, &img, &k, pose))
			.map(|pose| {
				inlier_indices(n, |i| {
					reprojection_error(&pose, &k, &object[i], &image[i]) <= threshold
				})
			})
	})
	.ok_or_else(|| degenerate("solve_pnp_ransac"))?;
	let obj: Vec<[f64; 3]> = inliers.iter().map(|i| object[*i]).collect();
	let img: Vec<[f64; 2]> = inliers.iter().map(|i| image[*i]).collect();
	let pose = dlt_pose(&obj, &img, &k)
		.and_then(|pose| refine_pose(&obj, &img, &k, pose))
		.ok_or_else(|| degenerate("solve_pnp_ransac"))?;
	Ok(PnpRansacResult {
		rotation_vector: vector_f32(&rodrigues_vector(&pose.0))?,
		translation_vector: vector_f32(&[pose.1[0], pose.1[1], pose.1[2]])?,
		mask: index_mask(n, &inliers)?,
	})
}

/// Convert between a `[3, 1]` Rodrigues rotation vector and a `[3, 3]`
/// rotation matrix, matching `Rodrigues` in both directions.
///
/// # Errors
///
/// Returns an error unless the input is a `[3, 1]`/`[1, 3]` or `[3, 3]` F32
/// matrix.
pub fn rodrigues(input: &Matrix) -> Result<Matrix> {
	let shape = input.shape();
	let values = input.read_f32()?;
	if shape == [3, 1].as_slice() || shape == [1, 3].as_slice() {
		let r = [values[0] as f64, values[1] as f64, values[2] as f64];
		let rotation = rodrigues_matrix(&r);
		mat3_f32(rotation)
	} else if shape == [3, 3].as_slice() {
		let m = read_mat3(input, "rodrigues")?;
		vector_f32(&rodrigues_vector(&m))
	} else {
		Err(Error::invalid_argument(
			"rodrigues requires a [3, 1]/[1, 3] rotation vector or [3, 3] rotation matrix",
		))
	}
}

/// Project `object` `[x, y, z]` points through `rvec`, `tvec`, `camera`, and
/// `distortion` coefficients `(k1, k2, p1, p2[, k3[, k4..]])`, matching
/// `projectPoints` with the radial+tangential model. `object` is `[N, 3]`,
/// `distortion` a `[k]`/`[k, 1]`/`[1, k]` vector (empty admitted), and the
/// output is `[N, 2]` F32.
///
/// # Errors
///
/// Returns an error for malformed inputs, a non-finite projection, or a
/// zero/negative optical-center denominator.
pub fn project_points(
	object: &Matrix,
	rvec: &Matrix,
	tvec: &Matrix,
	camera: &Matrix,
	distortion: &Matrix,
) -> Result<Matrix> {
	let object = read_points3(object, 1, "project_points")?;
	let r = read_vector(rvec, 3, "project_points")?;
	let t = read_vector(tvec, 3, "project_points")?;
	let k = read_mat3(camera, "project_points")?;
	let distortion = read_vector_free(distortion, "project_points")?;
	let rotation = rodrigues_matrix(&[r[0], r[1], r[2]]);
	let mut projected = Vec::with_capacity(object.len() * 2);
	for point in &object {
		let [x, y, z] = camera_project(&rotation, &[t[0], t[1], t[2]], point);
		if z.abs() < 1e-12 {
			return Err(Error::invalid_argument(
				"project_points produced a point at the camera plane",
			));
		}
		let (xn, yn) = distort_normalized(x / z, y / z, &distortion);
		projected.push((k[0][0] * xn + k[0][1] * yn + k[0][2]) as f32);
		projected.push((k[1][0] * xn + k[1][1] * yn + k[1][2]) as f32);
	}
	Matrix::new_from_f32([object.len(), 2], &projected)
}

/// Build the forward/undistort rectification coordinate map matching
/// `initUndistortRectifyMap` for a `width`x`height` output, returned as the
/// `[1, 2, height, width]` F32 map consumed by [`crate::image::remap`].
/// `rectification` is an optional `[3, 3]` rectifying rotation and
/// `projection` an optional `[3, 3]` new camera matrix (defaulting to the
/// input camera).
///
/// # Errors
///
/// Returns an error for malformed matrices or nonpositive dimensions.
pub fn undistort_rectify_map(
	camera: &Matrix,
	distortion: &Matrix,
	rectification: Option<&Matrix>,
	projection: Option<&Matrix>,
	width: i32,
	height: i32,
) -> Result<Matrix> {
	if width < 1 || height < 1 {
		return Err(Error::invalid_argument(
			"undistort_rectify_map requires positive width and height",
		));
	}
	let k = read_mat3(camera, "undistort_rectify_map")?;
	let distortion = read_vector_free(distortion, "undistort_rectify_map")?;
	let rotation = match rectification {
		Some(matrix) => read_mat3(matrix, "undistort_rectify_map")?,
		None => IDENTITY3,
	};
	let projection = match projection {
		Some(matrix) => read_mat3(matrix, "undistort_rectify_map")?,
		None => k,
	};
	let pinv = invert3(projection)
		.ok_or_else(|| Error::invalid_argument("undistort_rectify_map projection is singular"))?;
	let (w, h) = (width as usize, height as usize);
	let mut map = vec![0.0f32; 2 * w * h];
	for v in 0..h {
		for u in 0..w {
			// Output pixel -> normalized ideal coordinates through R^-1.
			let pin = [
				pinv[0][0] * u as f64 + pinv[0][1] * v as f64 + pinv[0][2],
				pinv[1][0] * u as f64 + pinv[1][1] * v as f64 + pinv[1][2],
				pinv[2][0] * u as f64 + pinv[2][1] * v as f64 + pinv[2][2],
			];
			let ideal = [
				rotation[0][0] * pin[0] + rotation[0][1] * pin[1] + rotation[0][2] * pin[2],
				rotation[1][0] * pin[0] + rotation[1][1] * pin[1] + rotation[1][2] * pin[2],
				rotation[2][0] * pin[0] + rotation[2][1] * pin[1] + rotation[2][2] * pin[2],
			];
			if ideal[2].abs() < 1e-12 {
				return Err(Error::invalid_argument(
					"undistort_rectify_map rectification produced a point at the plane",
				));
			}
			let (xd, yd) = distort_normalized(ideal[0] / ideal[2], ideal[1] / ideal[2], &distortion);
			map[v * w + u] = (k[0][0] * xd + k[0][1] * yd + k[0][2]) as f32;
			map[w * h + v * w + u] = (k[1][0] * xd + k[1][1] * yd + k[1][2]) as f32;
		}
	}
	Matrix::new_from_f32([1, 2, h, w], &map)
}

/// Undistort `[u, v]` `points` by iterative inversion of the distortion
/// model, matching `undistortPoints`. Without `projection` the result is
/// normalized camera coordinates; with `projection` (and optional
/// `rectification`) it is reprojected pixel coordinates.
///
/// # Errors
///
/// Returns an error for malformed matrices or a singular camera/rectification.
pub fn undistort_points(
	points: &Matrix,
	camera: &Matrix,
	distortion: &Matrix,
	rectification: Option<&Matrix>,
	projection: Option<&Matrix>,
) -> Result<Matrix> {
	let points = read_points2(points, 1, "undistort_points")?;
	let k = read_mat3(camera, "undistort_points")?;
	let distortion = read_vector_free(distortion, "undistort_points")?;
	let kinv =
		invert3(k).ok_or_else(|| Error::invalid_argument("undistort_points camera is singular"))?;
	let rotation = match rectification {
		Some(matrix) => read_mat3(matrix, "undistort_points")?,
		None => IDENTITY3,
	};
	let mut output = Vec::with_capacity(points.len() * 2);
	for &[u, v] in &points {
		let mut x = kinv[0][0] * u + kinv[0][1] * v + kinv[0][2];
		let mut y = kinv[1][0] * u + kinv[1][1] * v + kinv[1][2];
		let w_hom = kinv[2][0] * u + kinv[2][1] * v + kinv[2][2];
		if w_hom.abs() < 1e-12 {
			return Err(Error::invalid_argument(
				"undistort_points camera produced a point at infinity",
			));
		}
		x /= w_hom;
		y /= w_hom;
		// Fixed-point inversion of the forward distortion, OpenCV's iteration:
		// the distorted source stays fixed while the iterate's r^2 feeds the
		// radial divisor.
		let (x0, y0) = (x, y);
		let (k1, k2t, p1, p2, k3) = distortion_terms(&distortion);
		for _ in 0..8 {
			let r2 = x * x + y * y;
			let icd = 1.0 / (1.0 + k1 * r2 + k2t * r2 * r2 + k3 * r2 * r2 * r2);
			let delta_x = 2.0 * p1 * x * y + p2 * (r2 + 2.0 * x * x);
			let delta_y = p1 * (r2 + 2.0 * y * y) + 2.0 * p2 * x * y;
			x = (x0 - delta_x) * icd;
			y = (y0 - delta_y) * icd;
		}
		let rx = rotation[0][0] * x + rotation[0][1] * y + rotation[0][2];
		let ry = rotation[1][0] * x + rotation[1][1] * y + rotation[1][2];
		let rz = rotation[2][0] * x + rotation[2][1] * y + rotation[2][2];
		let (mut ox, mut oy) = (rx, ry);
		if let Some(projection_matrix) = projection {
			let p = read_mat3(projection_matrix, "undistort_points")?;
			if rz.abs() < 1e-12 {
				return Err(Error::invalid_argument(
					"undistort_points rectification produced a point at the plane",
				));
			}
			ox = p[0][0] * rx / rz + p[0][1] * ry / rz + p[0][2];
			oy = p[1][0] * rx / rz + p[1][1] * ry / rz + p[1][2];
		}
		output.push(ox as f32);
		output.push(oy as f32);
	}
	Matrix::new_from_f32([points.len(), 2], &output)
}

fn degenerate(operation: &str) -> Error {
	Error::invalid_argument(format!("{operation} point set is degenerate"))
}

// ---- matrix readers ---------------------------------------------------------

const IDENTITY3: [[f64; 3]; 3] = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];

fn read_points2(matrix: &Matrix, minimum: usize, operation: &str) -> Result<Vec<[f64; 2]>> {
	let shape = matrix.shape();
	if shape.len() != 2 || shape[1] != 2 || shape[0] < minimum {
		return Err(Error::invalid_argument(format!(
			"{operation} requires an [N, 2] matrix with N >= {minimum}"
		)));
	}
	let values = matrix.read_f32()?;
	Ok(
		values
			.as_chunks::<2>()
			.0
			.iter()
			.map(|pair| [pair[0] as f64, pair[1] as f64])
			.collect(),
	)
}

fn read_points3(matrix: &Matrix, minimum: usize, operation: &str) -> Result<Vec<[f64; 3]>> {
	let shape = matrix.shape();
	if shape.len() != 2 || shape[1] != 3 || shape[0] < minimum {
		return Err(Error::invalid_argument(format!(
			"{operation} requires an [N, 3] matrix with N >= {minimum}"
		)));
	}
	let values = matrix.read_f32()?;
	Ok(
		values
			.as_chunks::<3>()
			.0
			.iter()
			.map(|tri| [tri[0] as f64, tri[1] as f64, tri[2] as f64])
			.collect(),
	)
}

fn read_vector(matrix: &Matrix, expected: usize, operation: &str) -> Result<Vec<f64>> {
	let values = matrix.read_f32()?;
	if values.len() != expected {
		return Err(Error::invalid_argument(format!(
			"{operation} requires a length-{expected} vector"
		)));
	}
	Ok(values.iter().map(|value| *value as f64).collect())
}

fn read_vector_free(matrix: &Matrix, operation: &str) -> Result<Vec<f64>> {
	let shape = matrix.shape();
	if shape.len() > 2 || (shape.len() == 2 && shape[0] != 1 && shape[1] != 1) {
		return Err(Error::invalid_argument(format!(
			"{operation} requires a vector-shaped distortion matrix"
		)));
	}
	let values = matrix.read_f32()?;
	Ok(values.iter().map(|value| *value as f64).collect())
}

fn read_mat3(matrix: &Matrix, operation: &str) -> Result<[[f64; 3]; 3]> {
	if matrix.shape() != [3, 3].as_slice() {
		return Err(Error::invalid_argument(format!(
			"{operation} requires a [3, 3] matrix"
		)));
	}
	let values = matrix.read_f32()?;
	let mut out = [[0.0f64; 3]; 3];
	for row in 0..3 {
		for column in 0..3 {
			out[row][column] = values[row * 3 + column] as f64;
		}
	}
	Ok(out)
}

fn mat3_f32(m: [[f64; 3]; 3]) -> Result<Matrix> {
	let mut values = [0.0f32; 9];
	for row in 0..3 {
		for column in 0..3 {
			values[row * 3 + column] = m[row][column] as f32;
		}
	}
	Matrix::new_from_f32([3, 3], &values)
}

fn vector_f32(values: &[f64]) -> Result<Matrix> {
	let flat: Vec<f32> = values.iter().map(|value| *value as f32).collect();
	Matrix::new_from_f32([values.len(), 1], &flat)
}

fn ones_mask(n: usize) -> Result<Matrix> {
	Matrix::new_from_f32([n, 1], &vec![1.0f32; n])
}

fn index_mask(n: usize, indices: &[usize]) -> Result<Matrix> {
	let mut mask = vec![0.0f32; n];
	for &index in indices {
		mask[index] = 1.0;
	}
	Matrix::new_from_f32([n, 1], &mask)
}

fn inlier_indices(n: usize, predicate: impl Fn(usize) -> bool) -> Vec<usize> {
	(0..n).filter(|&index| predicate(index)).collect()
}

// ---- numeric helpers ---------------------------------------------------------

fn invert3(m: [[f64; 3]; 3]) -> Option<[[f64; 3]; 3]> {
	let det = m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1])
		- m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
		+ m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0]);
	if det.abs() < 1e-14 {
		return None;
	}
	let inv = 1.0 / det;
	Some([
		[
			(m[1][1] * m[2][2] - m[1][2] * m[2][1]) * inv,
			(m[0][2] * m[2][1] - m[0][1] * m[2][2]) * inv,
			(m[0][1] * m[1][2] - m[0][2] * m[1][1]) * inv,
		],
		[
			(m[1][2] * m[2][0] - m[1][0] * m[2][2]) * inv,
			(m[0][0] * m[2][2] - m[0][2] * m[2][0]) * inv,
			(m[0][2] * m[1][0] - m[0][0] * m[1][2]) * inv,
		],
		[
			(m[1][0] * m[2][1] - m[1][1] * m[2][0]) * inv,
			(m[0][1] * m[2][0] - m[0][0] * m[2][1]) * inv,
			(m[0][0] * m[1][1] - m[0][1] * m[1][0]) * inv,
		],
	])
}

fn cross(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
	[
		a[1] * b[2] - a[2] * b[1],
		a[2] * b[0] - a[0] * b[2],
		a[0] * b[1] - a[1] * b[0],
	]
}

fn norm(a: [f64; 3]) -> f64 {
	(a[0] * a[0] + a[1] * a[1] + a[2] * a[2]).sqrt()
}

fn normalize_vec(a: [f64; 3]) -> [f64; 3] {
	let n = norm(a);
	if n < 1e-14 {
		return a;
	}
	[a[0] / n, a[1] / n, a[2] / n]
}

fn dot(a: [f64; 3], b: [f64; 3]) -> f64 {
	a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

fn mul_mat3(a: &[[f64; 3]; 3], b: &[[f64; 3]; 3]) -> [[f64; 3]; 3] {
	let mut out = [[0.0f64; 3]; 3];
	for i in 0..3 {
		for j in 0..3 {
			out[i][j] = a[i][0] * b[0][j] + a[i][1] * b[1][j] + a[i][2] * b[2][j];
		}
	}
	out
}

fn transpose3(m: [[f64; 3]; 3]) -> [[f64; 3]; 3] {
	[
		[m[0][0], m[1][0], m[2][0]],
		[m[0][1], m[1][1], m[2][1]],
		[m[0][2], m[1][2], m[2][2]],
	]
}

/// 3x3 SVD through the eigenbasis of `m^T m` / `m m^T`, ordered by descending
/// singular value with determinant-fixed orthonormal factors.
#[allow(clippy::type_complexity)]
fn svd3(m: [[f64; 3]; 3]) -> Option<([[f64; 3]; 3], [f64; 3], [[f64; 3]; 3])> {
	let mt = transpose3(m);
	let mtm = mul_mat3(&mt, &m);
	let mmt = mul_mat3(&m, &mt);
	let (vvals, vvec) = jacobi_eigen(&mtm, 3)?;
	let (uvals, uvec) = jacobi_eigen(&mmt, 3)?;
	let mut s = [0.0f64; 3];
	let mut v = [[0.0f64; 3]; 3];
	let mut u = [[0.0f64; 3]; 3];
	for column in 0..3 {
		s[column] = vvals[2 - column].max(0.0).sqrt();
		for row in 0..3 {
			v[row][column] = vvec[row][2 - column];
			u[row][column] = uvec[row][2 - column];
		}
	}
	let _ = uvals;
	// Rebuild U = M V / s for nonzero singular values; orthonormalize the rest.
	for column in 0..3 {
		if s[column] > 1e-10 {
			for row in 0..3 {
				u[row][column] =
					(m[row][0] * v[0][column] + m[row][1] * v[1][column] + m[row][2] * v[2][column])
						/ s[column];
			}
		}
	}
	// Fix degenerate components by orthonormal completion.
	for column in 0..3 {
		let column_vec = [u[0][column], u[1][column], u[2][column]];
		if norm(column_vec) < 0.5 {
			let others: Vec<[f64; 3]> = (0..3)
				.filter(|&c| c != column)
				.map(|c| [u[0][c], u[1][c], u[2][c]])
				.collect();
			let candidate = normalize_vec(cross(others[0], others[1]));
			for (row, ur) in u.iter_mut().enumerate() {
				ur[column] = candidate[row];
			}
		}
	}
	// Determinant signs: U and V must be rotations for pose candidates.
	if det3(&v) < 0.0 {
		for r in v.iter_mut() {
			r[2] = -r[2];
		}
	}
	if det3(&u) < 0.0 {
		for r in u.iter_mut() {
			r[2] = -r[2];
		}
	}
	Some((u, s, transpose3(v)))
}

fn det3(m: &[[f64; 3]; 3]) -> f64 {
	m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1])
		- m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
		+ m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0])
}

// ---- homography -----------------------------------------------------------

/// Hartley-normalized DLT homography from at least four correspondences.
fn dlt_homography(source: &[[f64; 2]], destination: &[[f64; 2]]) -> Option<[[f64; 3]; 3]> {
	let (ts, ns) = normalize2d(source)?;
	let (td, nd) = normalize2d(destination)?;
	let n = ns.len();
	let mut a = vec![0.0f64; 2 * n * 9];
	for (index, (&[x, y], &[u, v])) in ns.iter().zip(nd.iter()).enumerate() {
		let row = 2 * index * 9;
		a[row..row + 9].copy_from_slice(&[x, y, 1.0, 0.0, 0.0, 0.0, -u * x, -u * y, -u]);
		a[row + 9..row + 18].copy_from_slice(&[0.0, 0.0, 0.0, x, y, 1.0, -v * x, -v * y, -v]);
	}
	let vectors = smallest_right_singular::<9>(&a, 2 * n)?;
	let mut h = [[0.0f64; 3]; 3];
	for row in 0..3 {
		for column in 0..3 {
			h[row][column] = vectors[row * 3 + column][0];
		}
	}
	// Denormalize: H = Td^-1 * Hn * Ts.
	let td_inv = invert3(td)?;
	let ts_mul = mul_mat3(&h, &ts);
	let full = mul_mat3(&td_inv, &ts_mul);
	let scale = full[2][2];
	if scale.abs() < 1e-14 {
		return None;
	}
	let mut out = [[0.0f64; 3]; 3];
	for row in 0..3 {
		for column in 0..3 {
			out[row][column] = full[row][column] / scale;
		}
	}
	Some(out)
}

/// Similarity normalization used by DLT estimators: centroid + rms scale.
#[allow(clippy::type_complexity)]
fn normalize2d(points: &[[f64; 2]]) -> Option<([[f64; 3]; 3], Vec<[f64; 2]>)> {
	let n = points.len() as f64;
	if n < 1.0 {
		return None;
	}
	let (mut cx, mut cy) = (0.0, 0.0);
	for &[x, y] in points {
		cx += x;
		cy += y;
	}
	cx /= n;
	cy /= n;
	let mut scale = 0.0;
	for &[x, y] in points {
		scale += ((x - cx) * (x - cx) + (y - cy) * (y - cy)).sqrt();
	}
	scale /= n;
	if scale < 1e-14 {
		scale = 1.0;
	}
	let s = 1.0 / scale;
	let transform = [[s, 0.0, -s * cx], [0.0, s, -s * cy], [0.0, 0.0, 1.0]];
	let normalized = points
		.iter()
		.map(|&[x, y]| [(x - cx) * s, (y - cy) * s])
		.collect();
	Some((transform, normalized))
}

/// Symmetric transfer error (pixels) for a homography correspondence.
fn homography_error(h: &[[f64; 3]; 3], source: &[f64; 2], destination: &[f64; 2]) -> f64 {
	let forward = match project_h(h, source) {
		Some([x, y]) => (x - destination[0]).powi(2) + (y - destination[1]).powi(2),
		None => return f64::MAX,
	};
	let backward = match invert3(*h).and_then(|hinv| project_h(&hinv, destination)) {
		Some([x, y]) => (x - source[0]).powi(2) + (y - source[1]).powi(2),
		None => return f64::MAX,
	};
	forward + backward
}

/// Apply a homogeneous transform to a `[x, y]` point.
fn project_h(h: &[[f64; 3]; 3], p: &[f64; 2]) -> Option<[f64; 2]> {
	let x = h[0][0] * p[0] + h[0][1] * p[1] + h[0][2];
	let y = h[1][0] * p[0] + h[1][1] * p[1] + h[1][2];
	let w = h[2][0] * p[0] + h[2][1] * p[1] + h[2][2];
	if w.abs() < 1e-14 {
		return None;
	}
	Some([x / w, y / w])
}

// ---- RANSAC ------------------------------------------------------------------

// Donor: oa/core/std/random.h, Random. Retain its PCG32 sequence=1 and
// two-draw nextU64 order so seeded RANSAC selection matches the donor stream.
struct Pcg32 {
	state: u64,
}

impl Pcg32 {
	fn new(seed: u64) -> Self {
		let mut rng = Self { state: 0 };
		rng.next_u32();
		rng.state = rng.state.wrapping_add(seed);
		rng.next_u32();
		rng
	}

	fn next_u32(&mut self) -> u32 {
		let old = self.state;
		self.state = old.wrapping_mul(6364136223846793005).wrapping_add(3);
		let shifted = (((old >> 18) ^ old) >> 27) as u32;
		shifted.rotate_right((old >> 59) as u32)
	}

	fn next_u64(&mut self) -> u64 {
		(u64::from(self.next_u32()) << 32) | u64::from(self.next_u32())
	}
}

/// Seeded minimal-sample consensus: draw `iterations` `sample_size` subsets,
/// evaluate each through `estimate` (returning the inlier index list for the
/// sampled model), and return the largest consensus set.
fn ransac_consensus(
	sample_size: usize,
	n: usize,
	iterations: usize,
	seed: u64,
	mut estimate: impl FnMut(&[usize]) -> Option<Vec<usize>>,
) -> Option<Vec<usize>> {
	if n < sample_size {
		return None;
	}
	let mut rng = Pcg32::new(seed);
	let mut best: Vec<usize> = Vec::new();
	for _ in 0..iterations {
		let mut sample: Vec<usize> = (0..n).collect();
		for count in (2..=n).rev() {
			if count <= n - sample_size {
				break;
			}
			let range = count as u64;
			let reject = range.wrapping_neg() % range;
			let value = loop {
				let value = rng.next_u64();
				if value >= reject {
					break value;
				}
			};
			sample.swap(count - 1, (value % range) as usize);
		}
		let sample = &sample[n - sample_size..];
		if let Some(inliers) = estimate(sample).filter(|i| i.len() > best.len()) {
			best = inliers;
		}
	}
	if best.is_empty() { None } else { Some(best) }
}

// ---- pose / projection ------------------------------------------------------

/// Camera-frame coordinates of a world point under rotation+translation.
fn camera_project(rotation: &[[f64; 3]; 3], t: &[f64; 3], point: &[f64; 3]) -> [f64; 3] {
	[
		rotation[0][0] * point[0] + rotation[0][1] * point[1] + rotation[0][2] * point[2] + t[0],
		rotation[1][0] * point[0] + rotation[1][1] * point[1] + rotation[1][2] * point[2] + t[1],
		rotation[2][0] * point[0] + rotation[2][1] * point[1] + rotation[2][2] * point[2] + t[2],
	]
}

/// Reprojection pixel error for a pose candidate.
fn reprojection_error(
	pose: &([[f64; 3]; 3], [f64; 3]),
	camera: &[[f64; 3]; 3],
	object: &[f64; 3],
	image: &[f64; 2],
) -> f64 {
	let [x, y, z] = camera_project(&pose.0, &pose.1, object);
	if z.abs() < 1e-12 {
		return f64::MAX;
	}
	let u = (camera[0][0] * x + camera[0][1] * y + camera[0][2] * z) / z;
	let v = (camera[1][0] * x + camera[1][1] * y + camera[1][2] * z) / z;
	let du = u - image[0];
	let dv = v - image[1];
	du * du + dv * dv
}

/// Rodrigues rotation vector -> rotation matrix.
fn rodrigues_matrix(r: &[f64; 3]) -> [[f64; 3]; 3] {
	let theta = norm(*r);
	if theta < 1e-12 {
		return IDENTITY3;
	}
	let k = [r[0] / theta, r[1] / theta, r[2] / theta];
	let (c, s) = (theta.cos(), theta.sin());
	let outer = |i: usize, j: usize| k[i] * k[j] * (1.0 - c);
	let skew = [
		[0.0, -k[2] * s, k[1] * s],
		[k[2] * s, 0.0, -k[0] * s],
		[-k[1] * s, k[0] * s, 0.0],
	];
	let mut out = [[0.0f64; 3]; 3];
	for i in 0..3 {
		for j in 0..3 {
			out[i][j] = (if i == j { c } else { 0.0 }) + outer(i, j) + skew[i][j];
		}
	}
	out
}

/// Rotation matrix -> Rodrigues rotation vector.
fn rodrigues_vector(rotation: &[[f64; 3]; 3]) -> [f64; 3] {
	let trace = rotation[0][0] + rotation[1][1] + rotation[2][2];
	let cosine = ((trace - 1.0) / 2.0).clamp(-1.0, 1.0);
	let theta = cosine.acos();
	if theta < 1e-8 {
		return [0.0, 0.0, 0.0];
	}
	let denominator = 2.0 * theta.sin();
	if denominator.abs() < 1e-8 {
		// Near-pi rotation: recover the axis from the diagonal terms.
		let mut axis = [
			((rotation[0][0] + 1.0) / 2.0).max(0.0).sqrt(),
			((rotation[1][1] + 1.0) / 2.0).max(0.0).sqrt(),
			((rotation[2][2] + 1.0) / 2.0).max(0.0).sqrt(),
		];
		if rotation[0][1] < 0.0 {
			axis[1] = -axis[1];
		}
		if rotation[0][2] < 0.0 {
			axis[2] = -axis[2];
		}
		return normalize_vec(axis).map(|v| v * theta);
	}
	let scale = theta / denominator;
	[
		(rotation[2][1] - rotation[1][2]) * scale,
		(rotation[0][2] - rotation[2][0]) * scale,
		(rotation[1][0] - rotation[0][1]) * scale,
	]
}

/// DLT camera pose: estimate R|t through a 12-parameter homogeneous solve on
/// normalized image coordinates, then orthonormalize the rotation block.
fn dlt_pose(
	object: &[[f64; 3]],
	image: &[[f64; 2]],
	camera: &[[f64; 3]; 3],
) -> Option<([[f64; 3]; 3], [f64; 3])> {
	let kinv = invert3(*camera)?;
	let normalized: Vec<[f64; 2]> = image
		.iter()
		.map(|&[u, v]| {
			let x = kinv[0][0] * u + kinv[0][1] * v + kinv[0][2];
			let y = kinv[1][0] * u + kinv[1][1] * v + kinv[1][2];
			let w = kinv[2][0] * u + kinv[2][1] * v + kinv[2][2];
			[x / w, y / w]
		})
		.collect();
	let n = object.len();
	let mut a = vec![0.0f64; 2 * n * 12];
	for (index, (&point, &[xn, yn])) in object.iter().zip(normalized.iter()).enumerate() {
		let [x, y, z] = point;
		let row = 2 * index * 12;
		a[row..row + 12].copy_from_slice(&[
			x,
			y,
			z,
			1.0,
			0.0,
			0.0,
			0.0,
			0.0,
			-xn * x,
			-xn * y,
			-xn * z,
			-xn,
		]);
		a[row + 12..row + 24].copy_from_slice(&[
			0.0,
			0.0,
			0.0,
			0.0,
			x,
			y,
			z,
			1.0,
			-yn * x,
			-yn * y,
			-yn * z,
			-yn,
		]);
	}
	let vectors = smallest_right_singular::<12>(&a, 2 * n)?;
	let mut rotation = [[0.0f64; 3]; 3];
	let mut translation = [0.0f64; 3];
	for row in 0..3 {
		for column in 0..3 {
			rotation[row][column] = vectors[row * 4 + column][0];
		}
		translation[row] = vectors[row * 4 + 3][0];
	}
	// Orthonormalize: R -> nearest rotation through eigen polar factor.
	let (mut u, _, vt) = svd3(rotation)?;
	let mut proper = mul_mat3(&u, &vt);
	if det3(&proper) < 0.0 {
		for r in u.iter_mut() {
			r[2] = -r[2];
		}
		proper = mul_mat3(&u, &vt);
	}
	let scale = norm([proper[0][0], proper[0][1], proper[0][2]]);
	let _ = scale;
	// Rescale translation to the orthonormalized rotation scale.
	let column_norms: Vec<f64> = (0..3)
		.map(|c| norm([rotation[0][c], rotation[1][c], rotation[2][c]]))
		.collect();
	let mean = (column_norms[0] + column_norms[1] + column_norms[2]) / 3.0;
	if mean < 1e-14 {
		return None;
	}
	let factor = 1.0 / mean;
	for t in translation.iter_mut() {
		*t *= factor;
	}
	// Ensure positive depth: flip the pose when the mean z is negative.
	let mut depth = 0.0;
	for &point in object {
		let p = camera_project(&proper, &translation, &point);
		depth += p[2];
	}
	if depth < 0.0 {
		for (r, t) in proper.iter_mut().zip(translation.iter_mut()) {
			for c in r.iter_mut() {
				*c = -*c;
			}
			*t = -*t;
		}
	}
	Some((proper, translation))
}

/// Damped Gauss-Newton pose refinement over the 6-dof rvec+tvec parameters.
fn refine_pose(
	object: &[[f64; 3]],
	image: &[[f64; 2]],
	camera: &[[f64; 3]; 3],
	pose: ([[f64; 3]; 3], [f64; 3]),
) -> Option<([[f64; 3]; 3], [f64; 3])> {
	let mut params = {
		let r = rodrigues_vector(&pose.0);
		[r[0], r[1], r[2], pose.1[0], pose.1[1], pose.1[2]]
	};
	let residuals = |params: &[f64]| -> Vec<f64> {
		let rotation = rodrigues_matrix(&[params[0], params[1], params[2]]);
		let t = [params[3], params[4], params[5]];
		let mut out = Vec::with_capacity(object.len() * 2);
		for (point, &target) in object.iter().zip(image.iter()) {
			let [x, y, z] = camera_project(&rotation, &[t[0], t[1], t[2]], point);
			let (u, v) = if z.abs() < 1e-9 {
				(f64::MAX * 1e-6, f64::MAX * 1e-6)
			} else {
				(
					(camera[0][0] * x + camera[0][1] * y + camera[0][2] * z) / z,
					(camera[1][0] * x + camera[1][1] * y + camera[1][2] * z) / z,
				)
			};
			out.push(u - target[0]);
			out.push(v - target[1]);
		}
		out
	};
	let n = params.len();
	let mut cost: f64 = residuals(&params).iter().map(|r| r * r).sum();
	for _ in 0..20 {
		let r = residuals(&params);
		let mut jacobian = vec![0.0f64; r.len() * n];
		let eps = 1e-6;
		for column in 0..n {
			let mut shifted = params;
			shifted[column] += eps;
			let shifted_residuals = residuals(&shifted);
			for (row, value) in shifted_residuals.iter().enumerate() {
				jacobian[row * n + column] = (value - r[row]) / eps;
			}
		}
		let step = damped_least_squares(&jacobian, &r, r.len(), n, 1e-3)?;
		let mut improved = false;
		for scale in [1.0, 0.5, 0.25] {
			let mut candidate = params;
			for i in 0..n {
				candidate[i] += step[i] * scale;
			}
			let candidate_cost: f64 = residuals(&candidate).iter().map(|r| r * r).sum();
			if candidate_cost < cost {
				params = candidate;
				cost = candidate_cost;
				improved = true;
				break;
			}
		}
		if !improved {
			break;
		}
	}
	Some((
		rodrigues_matrix(&[params[0], params[1], params[2]]),
		[params[3], params[4], params[5]],
	))
}

// ---- distortion ------------------------------------------------------------

/// Expand distortion coefficients to `(k1, k2, p1, p2, k3)` order.
fn distortion_terms(distortion: &[f64]) -> (f64, f64, f64, f64, f64) {
	let term = |index: usize| distortion.get(index).copied().unwrap_or(0.0);
	(term(0), term(1), term(2), term(3), term(4))
}

/// Forward distortion of normalized coordinates `(x, y)` using the
/// `(k1, k2, p1, p2, k3[, k4, k5, k6])` OpenCV rational model when extra
/// coefficients are present, else the radial+tangential base model.
fn distort_normalized(x: f64, y: f64, distortion: &[f64]) -> (f64, f64) {
	let (k1, k2, p1, p2, k3) = distortion_terms(distortion);
	let r2 = x * x + y * y;
	let radial = if distortion.len() >= 8 {
		let (k4, k5, k6) = (
			distortion.get(5).copied().unwrap_or(0.0),
			distortion.get(6).copied().unwrap_or(0.0),
			distortion.get(7).copied().unwrap_or(0.0),
		);
		(1.0 + k1 * r2 + k2 * r2 * r2 + k3 * r2 * r2 * r2)
			/ (1.0 + k4 * r2 + k5 * r2 * r2 + k6 * r2 * r2 * r2)
	} else {
		1.0 + k1 * r2 + k2 * r2 * r2 + k3 * r2 * r2 * r2
	};
	let dx = 2.0 * p1 * x * y + p2 * (r2 + 2.0 * x * x);
	let dy = p1 * (r2 + 2.0 * y * y) + 2.0 * p2 * x * y;
	(x * radial + dx, y * radial + dy)
}

// ---- epipolar geometry -----------------------------------------------------

/// Estimate the 3x3 fundamental matrix between `points1` and `points2`
/// (`[N, 2]` F32, `N >= 8`), matching `findFundamentalMat`. `LeastSquares`
/// runs the normalized eight-point solve; `Ransac` draws `max_iterations`
/// seeded 8-point minimal samples scored by symmetric epipolar distance
/// under `ransac_threshold` (pixels) and refits on the inlier set. The
/// estimate is enforced to rank 2 before return.
///
/// # Errors
///
/// Returns an error for mismatched/degenerate inputs, nonpositive bounds,
/// or when no rank-2 model exists.
pub fn find_fundamental(
	points1: &Matrix,
	points2: &Matrix,
	method: EstimationMethod,
	ransac_threshold: f32,
	max_iterations: i32,
	seed: u64,
) -> Result<EpipolarResult> {
	let points1 = read_points2(points1, 8, "find_fundamental")?;
	let points2 = read_points2(points2, 8, "find_fundamental")?;
	if points1.len() != points2.len() {
		return Err(Error::invalid_argument(
			"find_fundamental inputs must share point count",
		));
	}
	match method {
		EstimationMethod::LeastSquares => {
			let f = eight_point(&points1, &points2).ok_or_else(|| degenerate("find_fundamental"))?;
			Ok(EpipolarResult {
				matrix: mat3_f32(f)?,
				mask: ones_mask(points1.len())?,
			})
		}
		EstimationMethod::Ransac => {
			if !ransac_threshold.is_finite() || ransac_threshold <= 0.0 {
				return Err(Error::invalid_argument(
					"find_fundamental ransac_threshold must be positive and finite",
				));
			}
			if max_iterations < 1 {
				return Err(Error::invalid_argument(
					"find_fundamental max_iterations must be positive",
				));
			}
			let n = points1.len();
			let threshold = ransac_threshold as f64;
			let inliers = ransac_consensus(8, n, max_iterations as usize, seed, |sample| {
				let p1: Vec<[f64; 2]> = sample.iter().map(|i| points1[*i]).collect();
				let p2: Vec<[f64; 2]> = sample.iter().map(|i| points2[*i]).collect();
				eight_point(&p1, &p2).map(|f| {
					inlier_indices(n, |i| {
						epipolar_error(&f, &points1[i], &points2[i]) <= threshold
					})
				})
			})
			.ok_or_else(|| degenerate("find_fundamental"))?;
			let p1: Vec<[f64; 2]> = inliers.iter().map(|i| points1[*i]).collect();
			let p2: Vec<[f64; 2]> = inliers.iter().map(|i| points2[*i]).collect();
			let f = eight_point(&p1, &p2).ok_or_else(|| degenerate("find_fundamental"))?;
			Ok(EpipolarResult {
				matrix: mat3_f32(f)?,
				mask: index_mask(n, &inliers)?,
			})
		}
	}
}

/// Estimate the 3x3 essential matrix between `points1` and `points2`
/// (`[N, 2]` F32, `N >= 8`) under the `camera` intrinsic matrix, matching
/// `findEssentialMat`. Correspondences are normalized through `K^-1`, then
/// the same normalized eight-point solve applies; the singular values of the
/// result are collapsed to `(s, s, 0)` per the essential constraint.
///
/// # Errors
///
/// Returns an error for mismatched/degenerate inputs, a singular camera, or
/// nonpositive RANSAC bounds.
pub fn find_essential(
	points1: &Matrix,
	points2: &Matrix,
	camera: &Matrix,
	method: EstimationMethod,
	ransac_threshold: f32,
	max_iterations: i32,
	seed: u64,
) -> Result<EpipolarResult> {
	let k = read_mat3(camera, "find_essential")?;
	let kinv =
		invert3(k).ok_or_else(|| Error::invalid_argument("find_essential camera is singular"))?;
	let normalize = |matrix: &Matrix, op: &str| -> Result<Vec<[f64; 2]>> {
		let points = read_points2(matrix, 8, op)?;
		Ok(
			points
				.iter()
				.map(|&[u, v]| {
					let x = kinv[0][0] * u + kinv[0][1] * v + kinv[0][2];
					let y = kinv[1][0] * u + kinv[1][1] * v + kinv[1][2];
					let w = kinv[2][0] * u + kinv[2][1] * v + kinv[2][2];
					[x / w, y / w]
				})
				.collect(),
		)
	};
	let points1 = normalize(points1, "find_essential")?;
	let points2 = normalize(points2, "find_essential")?;
	if points1.len() != points2.len() {
		return Err(Error::invalid_argument(
			"find_essential inputs must share point count",
		));
	}
	// Essential threshold is in normalized-image units.
	match method {
		EstimationMethod::LeastSquares => {
			let e = eight_point(&points1, &points2)
				.and_then(essential_enforce)
				.ok_or_else(|| degenerate("find_essential"))?;
			Ok(EpipolarResult {
				matrix: mat3_f32(e)?,
				mask: ones_mask(points1.len())?,
			})
		}
		EstimationMethod::Ransac => {
			if !ransac_threshold.is_finite() || ransac_threshold <= 0.0 {
				return Err(Error::invalid_argument(
					"find_essential ransac_threshold must be positive and finite",
				));
			}
			if max_iterations < 1 {
				return Err(Error::invalid_argument(
					"find_essential max_iterations must be positive",
				));
			}
			let n = points1.len();
			let threshold = ransac_threshold as f64;
			let inliers = ransac_consensus(8, n, max_iterations as usize, seed, |sample| {
				let p1: Vec<[f64; 2]> = sample.iter().map(|i| points1[*i]).collect();
				let p2: Vec<[f64; 2]> = sample.iter().map(|i| points2[*i]).collect();
				eight_point(&p1, &p2).and_then(essential_enforce).map(|e| {
					inlier_indices(n, |i| {
						epipolar_error(&e, &points1[i], &points2[i]) <= threshold
					})
				})
			})
			.ok_or_else(|| degenerate("find_essential"))?;
			let p1: Vec<[f64; 2]> = inliers.iter().map(|i| points1[*i]).collect();
			let p2: Vec<[f64; 2]> = inliers.iter().map(|i| points2[*i]).collect();
			let e = eight_point(&p1, &p2)
				.and_then(essential_enforce)
				.ok_or_else(|| degenerate("find_essential"))?;
			Ok(EpipolarResult {
				matrix: mat3_f32(e)?,
				mask: index_mask(n, &inliers)?,
			})
		}
	}
}

/// Recover the relative camera pose `(R, t)` between views from an
/// `essential` `[3, 3]` matrix, matching `recoverPose`. `points1`/`points2`
/// are the pixel correspondences and `camera` supplies intrinsics; all four
/// `(±t, R | R^W)` candidates are triangulated and the cheirality-majority
/// candidate wins, with the mask recording in-front points.
///
/// # Errors
///
/// Returns an error for malformed inputs or when no candidate yields
/// positive depth.
#[allow(clippy::type_complexity)]
pub fn recover_pose(
	essential: &Matrix,
	points1: &Matrix,
	points2: &Matrix,
	camera: &Matrix,
) -> Result<RecoverPoseResult> {
	let e = read_mat3(essential, "recover_pose")?;
	let k = read_mat3(camera, "recover_pose")?;
	let kinv =
		invert3(k).ok_or_else(|| Error::invalid_argument("recover_pose camera is singular"))?;
	let points1 = read_points2(points1, 1, "recover_pose")?;
	let points2 = read_points2(points2, 1, "recover_pose")?;
	if points1.len() != points2.len() {
		return Err(Error::invalid_argument(
			"recover_pose inputs must share point count",
		));
	}
	let n = points1.len();
	let normalized = |points: &[[f64; 2]]| -> Vec<[f64; 2]> {
		points
			.iter()
			.map(|&[u, v]| {
				let x = kinv[0][0] * u + kinv[0][1] * v + kinv[0][2];
				let y = kinv[1][0] * u + kinv[1][1] * v + kinv[1][2];
				let w = kinv[2][0] * u + kinv[2][1] * v + kinv[2][2];
				[x / w, y / w]
			})
			.collect()
	};
	let pn1 = normalized(&points1);
	let pn2 = normalized(&points2);
	let (u, _, vt) = svd3(e).ok_or_else(|| degenerate("recover_pose"))?;
	let w = [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]];
	let wt = transpose3(w);
	let mut u_adj = u;
	if det3(&u_adj) < 0.0 {
		for r in u_adj.iter_mut() {
			r[2] = -r[2];
		}
	}
	let mut v_adj = transpose3(vt);
	if det3(&v_adj) < 0.0 {
		for r in v_adj.iter_mut() {
			r[2] = -r[2];
		}
	}
	let r1 = mul_mat3(&mul_mat3(&u_adj, &w), &transpose3(v_adj));
	let r2 = mul_mat3(&mul_mat3(&u_adj, &wt), &transpose3(v_adj));
	let t_pos = [u_adj[0][2], u_adj[1][2], u_adj[2][2]];
	let t_neg = [-t_pos[0], -t_pos[1], -t_pos[2]];
	let p1: [[f64; 4]; 3] = [
		[1.0, 0.0, 0.0, 0.0],
		[0.0, 1.0, 0.0, 0.0],
		[0.0, 0.0, 1.0, 0.0],
	];
	let mut best: Option<([[f64; 3]; 3], [f64; 3], Vec<usize>)> = None;
	for (rotation, t) in [(r1, t_pos), (r1, t_neg), (r2, t_pos), (r2, t_neg)] {
		let p2: [[f64; 4]; 3] = [
			[rotation[0][0], rotation[0][1], rotation[0][2], t[0]],
			[rotation[1][0], rotation[1][1], rotation[1][2], t[1]],
			[rotation[2][0], rotation[2][1], rotation[2][2], t[2]],
		];
		let mut in_front = Vec::new();
		for index in 0..n {
			if let Some([x, y, z, w]) = triangulate_point(&p1, &p2, &pn1[index], &pn2[index]) {
				if w.abs() < 1e-14 {
					continue;
				}
				let point = [x / w, y / w, z / w];
				let depth2 =
					rotation[2][0] * point[0] + rotation[2][1] * point[1] + rotation[2][2] * point[2] + t[2];
				if point[2] > 0.0 && depth2 > 0.0 {
					in_front.push(index);
				}
			}
		}
		if best
			.as_ref()
			.is_none_or(|(_, _, best)| in_front.len() > best.len())
		{
			best = Some((rotation, t, in_front));
		}
	}
	let (rotation, translation, in_front) = best.ok_or_else(|| degenerate("recover_pose"))?;
	if in_front.is_empty() {
		return Err(degenerate("recover_pose"));
	}
	Ok(RecoverPoseResult {
		rotation: mat3_f32(rotation)?,
		translation: vector_f32(&translation)?,
		mask: index_mask(n, &in_front)?,
	})
}

/// Triangulate `points1`/`points2` correspondences under the `projection1` /
/// `projection2` `[3, 4]` camera matrices, returning homogeneous `[N, 4]`
/// points matching `triangulatePoints`.
///
/// # Errors
///
/// Returns an error for malformed inputs or mismatched point counts.
pub fn triangulate_points(
	projection1: &Matrix,
	projection2: &Matrix,
	points1: &Matrix,
	points2: &Matrix,
) -> Result<Matrix> {
	let p1 = read_mat34(projection1, "triangulate_points")?;
	let p2 = read_mat34(projection2, "triangulate_points")?;
	let points1 = read_points2(points1, 1, "triangulate_points")?;
	let points2 = read_points2(points2, 1, "triangulate_points")?;
	if points1.len() != points2.len() {
		return Err(Error::invalid_argument(
			"triangulate_points inputs must share point count",
		));
	}
	let mut output = Vec::with_capacity(points1.len() * 4);
	for index in 0..points1.len() {
		let point = triangulate_point(&p1, &p2, &points1[index], &points2[index])
			.ok_or_else(|| degenerate("triangulate_points"))?;
		output.extend(point.iter().map(|v| *v as f32));
	}
	Matrix::new_from_f32([points1.len(), 4], &output)
}

fn read_mat34(matrix: &Matrix, operation: &str) -> Result<[[f64; 4]; 3]> {
	if matrix.shape() != [3, 4].as_slice() {
		return Err(Error::invalid_argument(format!(
			"{operation} requires a [3, 4] projection matrix"
		)));
	}
	let values = matrix.read_f32()?;
	let mut out = [[0.0f64; 4]; 3];
	for row in 0..3 {
		for column in 0..4 {
			out[row][column] = values[row * 4 + column] as f64;
		}
	}
	Ok(out)
}

/// Homogeneous DLT triangulation of one correspondence.
fn triangulate_point(
	p1: &[[f64; 4]; 3],
	p2: &[[f64; 4]; 3],
	point1: &[f64; 2],
	point2: &[f64; 2],
) -> Option<[f64; 4]> {
	let mut a = [0.0f64; 4 * 4];
	for column in 0..4 {
		a[column] = point1[0] * p1[2][column] - p1[0][column];
		a[4 + column] = point1[1] * p1[2][column] - p1[1][column];
		a[8 + column] = point2[0] * p2[2][column] - p2[0][column];
		a[12 + column] = point2[1] * p2[2][column] - p2[1][column];
	}
	let vectors = smallest_right_singular::<4>(&a, 4)?;
	Some([vectors[0][0], vectors[1][0], vectors[2][0], vectors[3][0]])
}

/// Normalized eight-point fundamental solve with rank-2 enforcement.
fn eight_point(points1: &[[f64; 2]], points2: &[[f64; 2]]) -> Option<[[f64; 3]; 3]> {
	let (t1, n1) = normalize2d(points1)?;
	let (t2, n2) = normalize2d(points2)?;
	let n = n1.len();
	let mut a = vec![0.0f64; n * 9];
	for (index, (&[x, y], &[u, v])) in n1.iter().zip(n2.iter()).enumerate() {
		let row = index * 9;
		a[row..row + 9].copy_from_slice(&[u * x, u * y, u, v * x, v * y, v, x, y, 1.0]);
	}
	let vectors = smallest_right_singular::<9>(&a, n)?;
	let mut f = [[0.0f64; 3]; 3];
	for row in 0..3 {
		for column in 0..3 {
			f[row][column] = vectors[row * 3 + column][0];
		}
	}
	// Enforce rank 2.
	let (u, s, vt) = svd3(f)?;
	let mut f_rank2 = [[0.0f64; 3]; 3];
	for i in 0..3 {
		for j in 0..3 {
			f_rank2[i][j] = s[0] * u[i][0] * vt[0][j] + s[1] * u[i][1] * vt[1][j];
		}
	}
	// Denormalize: F = T2^T * Fn * T1.
	let t2t = transpose3(t2);
	Some(mul_mat3(&t2t, &mul_mat3(&f_rank2, &t1)))
}

/// Collapse a rank-2 matrix to the essential constraint `(s, s, 0)`.
fn essential_enforce(e: [[f64; 3]; 3]) -> Option<[[f64; 3]; 3]> {
	let (u, s, vt) = svd3(e)?;
	let mean = (s[0] + s[1]) / 2.0;
	let mut out = [[0.0f64; 3]; 3];
	for i in 0..3 {
		for j in 0..3 {
			out[i][j] = mean * (u[i][0] * vt[0][j] + u[i][1] * vt[1][j]);
		}
	}
	Some(out)
}

/// Symmetric epipolar distance for `p2^T F p1`.
fn epipolar_error(f: &[[f64; 3]; 3], p1: &[f64; 2], p2: &[f64; 2]) -> f64 {
	let x = [p1[0], p1[1], 1.0];
	let xp = [p2[0], p2[1], 1.0];
	let fx = [
		dot([f[0][0], f[0][1], f[0][2]], x),
		dot([f[1][0], f[1][1], f[1][2]], x),
		dot([f[2][0], f[2][1], f[2][2]], x),
	];
	let ftx = [
		dot([f[0][0], f[1][0], f[2][0]], xp),
		dot([f[0][1], f[1][1], f[2][1]], xp),
		dot([f[0][2], f[1][2], f[2][2]], xp),
	];
	let residual = dot(xp, fx);
	let denom1 = fx[0] * fx[0] + fx[1] * fx[1];
	let denom2 = ftx[0] * ftx[0] + ftx[1] * ftx[1];
	if denom1 < 1e-24 || denom2 < 1e-24 {
		return f64::MAX;
	}
	residual * residual * (1.0 / denom1 + 1.0 / denom2)
}

// ---- calibration -----------------------------------------------------------

/// Calibrate a monocular camera from `views >= 3` planar grids, matching
/// `calibrateCamera` on Zhang's method. `object` is `[V, N, 3]` object
/// coordinates (the plane must satisfy `z = 0`) and `image` `[V, N, 2]`
/// pixel observations with `N >= 4`. Intrinsics solve through the B-matrix
/// constraints; extrinsics recover per-view; `refine_iterations > 0` then
/// runs damped bundle adjustment over `(fx, fy, cx, cy, k1, k2)` plus
/// per-view pose. Returns camera, `(k1, k2, p1, p2, k3)` distortion,
/// per-view rvecs `[V, 3]`, tvecs `[V, 3]`, and RMS reprojection error.
///
/// # Errors
///
/// Returns an error for shape violations, degenerate grids, or a singular
/// B-matrix solve.
pub fn calibrate_camera(
	object: &Matrix,
	image: &Matrix,
	refine_iterations: i32,
) -> Result<CalibrateCameraResult> {
	let oshape = object.shape();
	let ishape = image.shape();
	if oshape.len() != 3 || oshape[2] != 3 {
		return Err(Error::invalid_argument(
			"calibrate_camera requires object [V, N, 3]",
		));
	}
	if ishape.len() != 3 || ishape[2] != 2 || ishape[0] != oshape[0] || ishape[1] != oshape[1] {
		return Err(Error::invalid_argument(
			"calibrate_camera requires image [V, N, 2] matching object",
		));
	}
	let (views, n) = (oshape[0], oshape[1]);
	if views < 3 {
		return Err(Error::invalid_argument(
			"calibrate_camera requires V >= 3 views",
		));
	}
	if n < 4 {
		return Err(Error::invalid_argument(
			"calibrate_camera requires N >= 4 points per view",
		));
	}
	if refine_iterations < 0 {
		return Err(Error::invalid_argument(
			"calibrate_camera refine_iterations must be nonnegative",
		));
	}
	let ovals = object.read_f32()?;
	let ivals = image.read_f32()?;
	let object_views: Vec<Vec<[f64; 3]>> = (0..views)
		.map(|v| {
			let base = v * n * 3;
			ovals[base..base + n * 3]
				.as_chunks::<3>()
				.0
				.iter()
				.map(|t| [t[0] as f64, t[1] as f64, t[2] as f64])
				.collect()
		})
		.collect();
	let image_views: Vec<Vec<[f64; 2]>> = (0..views)
		.map(|v| {
			let base = v * n * 2;
			ivals[base..base + n * 2]
				.as_chunks::<2>()
				.0
				.iter()
				.map(|t| [t[0] as f64, t[1] as f64])
				.collect()
		})
		.collect();
	for (v, view) in object_views.iter().enumerate() {
		if view.iter().any(|p| p[2].abs() > 1e-6) {
			return Err(Error::invalid_argument(format!(
				"calibrate_camera view {v} is not planar (z != 0)"
			)));
		}
	}
	// Per-view homographies from the planar grid to pixels.
	let homographies: Vec<[[f64; 3]; 3]> = object_views
		.iter()
		.zip(image_views.iter())
		.map(|(obj, img)| {
			let src: Vec<[f64; 2]> = obj.iter().map(|p| [p[0], p[1]]).collect();
			dlt_homography(&src, img)
		})
		.collect::<Option<Vec<_>>>()
		.ok_or_else(|| degenerate("calibrate_camera"))?;
	// B-matrix constraints: rows v12(b) and v11(b) - v22(b), solved as the
	// null vector of the 2V x 6 stack.
	let mut equations = vec![0.0f64; 2 * views * 6];
	for (v, h) in homographies.iter().enumerate() {
		let v_row = |p: usize, q: usize| -> [f64; 6] {
			let hp = |i: usize| h[i][p];
			let hq = |i: usize| h[i][q];
			[
				hp(0) * hq(0),
				hp(0) * hq(1) + hp(1) * hq(0),
				hp(1) * hq(1),
				hp(2) * hq(0) + hp(0) * hq(2),
				hp(2) * hq(1) + hp(1) * hq(2),
				hp(2) * hq(2),
			]
		};
		let v12 = v_row(0, 1);
		let v11 = v_row(0, 0);
		let v22 = v_row(1, 1);
		for i in 0..6 {
			equations[2 * v * 6 + i] = v12[i];
			equations[(2 * v + 1) * 6 + i] = v11[i] - v22[i];
		}
	}
	let b = smallest_right_singular::<6>(&equations, 2 * views)
		.ok_or_else(|| degenerate("calibrate_camera"))?;
	let b = [b[0][0], b[1][0], b[2][0], b[3][0], b[4][0], b[5][0]];
	// The null vector is sign-ambiguous; the B matrix must be positive.
	let mut camera = zhang_intrinsics(&b)
		.or_else(|| zhang_intrinsics(&b.map(|v| -v)))
		.ok_or_else(|| degenerate("calibrate_camera"))?;
	// Per-view extrinsics from K^-1 H.
	let kinv = invert3(camera).ok_or_else(|| degenerate("calibrate_camera"))?;
	let mut poses: Vec<([f64; 3], [f64; 3])> = Vec::with_capacity(views);
	for h in &homographies {
		let kh = mul_mat3(&kinv, h);
		let l1 = norm([kh[0][0], kh[1][0], kh[2][0]]);
		let l2 = norm([kh[0][1], kh[1][1], kh[2][1]]);
		if l1 < 1e-14 || l2 < 1e-14 {
			return Err(degenerate("calibrate_camera"));
		}
		let scale = 2.0 / (l1 + l2);
		let r1 = [kh[0][0] * scale, kh[1][0] * scale, kh[2][0] * scale];
		let r2 = [kh[0][1] * scale, kh[1][1] * scale, kh[2][1] * scale];
		let r3 = cross(r1, r2);
		let mut rotation = [
			[r1[0], r2[0], r3[0]],
			[r1[1], r2[1], r3[1]],
			[r1[2], r2[2], r3[2]],
		];
		// Orthonormalize through the polar factor.
		if let Some((u, _, vt)) = svd3(rotation) {
			rotation = mul_mat3(&u, &vt);
		}
		if det3(&rotation) < 0.0 {
			for r in rotation.iter_mut() {
				for c in r.iter_mut() {
					*c = -*c;
				}
			}
		}
		let t = [kh[0][2] * scale, kh[1][2] * scale, kh[2][2] * scale];
		poses.push((rodrigues_vector(&rotation), t));
	}
	let mut distortion = [0.0f64; 2];
	if refine_iterations > 0 {
		calibrate_refine(
			&mut camera,
			&mut distortion,
			&mut poses,
			&object_views,
			&image_views,
			refine_iterations as usize,
		);
	}
	// RMS reprojection error over every view/point.
	let rotation_mats: Vec<[[f64; 3]; 3]> = poses.iter().map(|(r, _)| rodrigues_matrix(r)).collect();
	let mut sse = 0.0;
	let mut count = 0usize;
	for (v, view) in object_views.iter().enumerate() {
		for (point, &target) in view.iter().zip(image_views[v].iter()) {
			let [x, y, z] = camera_project(&rotation_mats[v], &poses[v].1, point);
			if z.abs() < 1e-12 {
				continue;
			}
			let (xn, yn) = distort_normalized(x / z, y / z, &distortion);
			let u = camera[0][0] * xn + camera[0][1] * yn + camera[0][2];
			let w = camera[1][0] * xn + camera[1][1] * yn + camera[1][2];
			sse += (u - target[0]).powi(2) + (w - target[1]).powi(2);
			count += 1;
		}
	}
	let rms = if count > 0 {
		(sse / count as f64).sqrt()
	} else {
		0.0
	};
	let mut rvecs = Vec::with_capacity(views * 3);
	let mut tvecs = Vec::with_capacity(views * 3);
	for (r, t) in &poses {
		rvecs.extend(r.iter().map(|v| *v as f32));
		tvecs.extend(t.iter().map(|v| *v as f32));
	}
	Ok(CalibrateCameraResult {
		camera: mat3_f32(camera)?,
		distortion: vector_f32(&[distortion[0], distortion[1], 0.0, 0.0, 0.0])?,
		rotations: Matrix::new_from_f32([views, 3], &rvecs)?,
		translations: Matrix::new_from_f32([views, 3], &tvecs)?,
		rms_error: vector_f32(&[rms])?,
	})
}

/// Zhang closed-form intrinsics from the B-vector.
fn zhang_intrinsics(b: &[f64; 6]) -> Option<[[f64; 3]; 3]> {
	let [b0, b1, b2, b3, b4, b5] = *b;
	let denom = b0 * b2 - b1 * b1;
	if denom.abs() < 1e-24 {
		return None;
	}
	let v0 = (b1 * b3 - b0 * b4) / denom;
	let lambda = b5 - (b3 * b3 + v0 * (b1 * b3 - b0 * b4)) / b0;
	if lambda <= 0.0 || b0.abs() < 1e-24 {
		return None;
	}
	let alpha = (lambda / b0).sqrt();
	let inner = b0 * b2 - b1 * b1;
	if inner <= 0.0 {
		return None;
	}
	let beta = (lambda * b0 / inner).sqrt();
	let gamma = -b1 * alpha * alpha * beta / lambda;
	let u0 = gamma * v0 / beta - b3 * alpha * alpha / lambda;
	if !(alpha.is_finite() && beta.is_finite() && u0.is_finite() && v0.is_finite()) {
		return None;
	}
	Some([[alpha, gamma, u0], [0.0, beta, v0], [0.0, 0.0, 1.0]])
}

/// Damped bundle adjustment over intrinsics + (k1, k2) + per-view pose.
#[allow(clippy::type_complexity)]
fn calibrate_refine(
	camera: &mut [[f64; 3]; 3],
	distortion: &mut [f64; 2],
	poses: &mut Vec<([f64; 3], [f64; 3])>,
	object_views: &[Vec<[f64; 3]>],
	image_views: &[Vec<[f64; 2]>],
	iterations: usize,
) {
	let views = poses.len();
	// Parameter layout: [fx, skew, cx, fy, cy, k1, k2] + per-view [rvec(3), t(3)].
	let pack7 = |camera: &[[f64; 3]; 3], distortion: &[f64; 2]| -> Vec<f64> {
		vec![
			camera[0][0],
			camera[0][1],
			camera[0][2],
			camera[1][1],
			camera[1][2],
			distortion[0],
			distortion[1],
		]
	};
	let unpack = |p: &[f64]| -> ([[f64; 3]; 3], [f64; 2], Vec<([f64; 3], [f64; 3])>) {
		let cam = [[p[0], p[1], p[2]], [0.0, p[3], p[4]], [0.0, 0.0, 1.0]];
		let mut ps = Vec::with_capacity(views);
		for v in 0..views {
			let base = 7 + v * 6;
			ps.push((
				[p[base], p[base + 1], p[base + 2]],
				[p[base + 3], p[base + 4], p[base + 5]],
			));
		}
		(cam, [p[5], p[6]], ps)
	};
	let mut params = {
		let mut p = pack7(camera, distortion);
		for (r, t) in poses.iter() {
			p.extend_from_slice(r);
			p.extend_from_slice(t);
		}
		p
	};
	let residuals = |p: &[f64]| -> Vec<f64> {
		let (cam, dist, ps) = unpack(p);
		let mut out = Vec::new();
		for (v, view) in object_views.iter().enumerate() {
			let rotation = rodrigues_matrix(&ps[v].0);
			for (point, &target) in view.iter().zip(image_views[v].iter()) {
				let [x, y, z] = camera_project(&rotation, &ps[v].1, point);
				let (u, w) = if z.abs() < 1e-9 {
					(f64::MAX * 1e-6, f64::MAX * 1e-6)
				} else {
					let (xn, yn) = distort_normalized(x / z, y / z, &[dist[0], dist[1]]);
					(
						cam[0][0] * xn + cam[0][1] * yn + cam[0][2],
						cam[1][1] * yn + cam[1][2],
					)
				};
				out.push(u - target[0]);
				out.push(w - target[1]);
			}
		}
		out
	};
	let n = params.len();
	let mut cost: f64 = residuals(&params).iter().map(|r| r * r).sum();
	for _ in 0..iterations {
		let r = residuals(&params);
		let mut jacobian = vec![0.0f64; r.len() * n];
		for column in 0..n {
			let mut shifted = params.clone();
			let eps = (params[column].abs() * 1e-6).max(1e-6);
			shifted[column] += eps;
			let shifted_residuals = residuals(&shifted);
			for (row, value) in shifted_residuals.iter().enumerate() {
				jacobian[row * n + column] = (value - r[row]) / eps;
			}
		}
		let Some(step) = damped_least_squares(&jacobian, &r, r.len(), n, 1e-3) else {
			break;
		};
		let mut improved = false;
		for scale in [1.0, 0.5, 0.25] {
			let mut candidate = params.clone();
			for i in 0..n {
				candidate[i] += step[i] * scale;
			}
			let candidate_cost: f64 = residuals(&candidate).iter().map(|r| r * r).sum();
			if candidate_cost < cost {
				params = candidate;
				cost = candidate_cost;
				improved = true;
				break;
			}
		}
		if !improved {
			break;
		}
	}
	let (cam, dist, ps) = unpack(&params);
	*camera = cam;
	*distortion = dist;
	*poses = ps;
}
