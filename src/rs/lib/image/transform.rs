//! Host-side geometric transform-matrix helpers.
//!
//! These produce the small 2x3 affine and 3x3 perspective matrices consumed
//! by [`crate::image::warp_affine`] and [`crate::image::warp_perspective`],
//! mirroring `getRotationMatrix2D`, `getPerspectiveTransform`,
//! `invertAffineTransform`, and `estimateAffine2D`. All math runs on the host
//! in f64 and returns plain F32 matrices, so the helpers compose with
//! captured plans without recording device work.

use crate::linalg::solve;
use crate::matrix::Matrix;
use crate::{Error, Result};

/// Build the 2x3 rotation+scale matrix mapping about `center`, matching
/// `getRotationMatrix2D`: `alpha = scale * cos(angle)`, `beta =
/// scale * sin(angle)` with `angle` in degrees counterclockwise.
///
/// # Errors
///
/// Returns an error when any parameter is non-finite.
pub fn rotation_matrix_2d(center_x: f32, center_y: f32, angle: f32, scale: f32) -> Result<Matrix> {
	for (name, value) in [
		("center_x", center_x),
		("center_y", center_y),
		("angle", angle),
		("scale", scale),
	] {
		if !value.is_finite() {
			return Err(Error::invalid_argument(format!(
				"rotation_matrix_2d {name} must be finite"
			)));
		}
	}
	let radians = (angle as f64).to_radians();
	let alpha = scale as f64 * radians.cos();
	let beta = scale as f64 * radians.sin();
	let (cx, cy) = (center_x as f64, center_y as f64);
	let values = [
		alpha,
		beta,
		(1.0 - alpha) * cx - beta * cy,
		-beta,
		alpha,
		beta * cx + (1.0 - alpha) * cy,
	];
	Matrix::new_from_f32([2, 3], &values.map(|value| value as f32))
}

/// Invert a 2x3 affine transform matrix, matching `invertAffineTransform`.
/// The input must be a `[2, 3]` F32 matrix with a nonsingular linear block.
///
/// # Errors
///
/// Returns an error unless `matrix` is a `[2, 3]` F32 matrix whose 2x2
/// linear block is nonsingular.
pub fn invert_affine(matrix: &Matrix) -> Result<Matrix> {
	let [a, b, c, d, e, f] = read_affine(matrix, "invert_affine")?;
	let determinant = a * e - b * d;
	if determinant == 0.0 {
		return Err(Error::invalid_argument(
			"invert_affine requires a nonsingular 2x2 linear block",
		));
	}
	let inverse = 1.0 / determinant;
	let (ia, ib, id, ie) = (e * inverse, -b * inverse, -d * inverse, a * inverse);
	let values = [ia, ib, -(ia * c + ib * f), id, ie, -(id * c + ie * f)];
	Matrix::new_from_f32([2, 3], &values.map(|value| value as f32))
}

/// Solve the 3x3 perspective transform mapping four `[x, y]` source points
/// to four destination points, matching `getPerspectiveTransform`. Both
/// inputs must be `[4, 2]` F32 matrices.
///
/// # Errors
///
/// Returns an error unless both inputs are `[4, 2]` F32 matrices yielding a
/// nonsingular 8x8 system (no three collinear source points).
pub fn perspective_transform(source: &Matrix, destination: &Matrix) -> Result<Matrix> {
	let source = read_points(source, 4, "perspective_transform")?;
	let destination = read_points(destination, 4, "perspective_transform")?;
	let mut a = [[0.0f64; 8]; 8];
	let mut b = [0.0f64; 8];
	for pair in 0..4 {
		let [x, y] = source[pair];
		let [u, v] = destination[pair];
		a[2 * pair] = [x, y, 1.0, 0.0, 0.0, 0.0, -u * x, -u * y];
		b[2 * pair] = u;
		a[2 * pair + 1] = [0.0, 0.0, 0.0, x, y, 1.0, -v * x, -v * y];
		b[2 * pair + 1] = v;
	}
	let h = solve(&mut a, &mut b, 8)
		.ok_or_else(|| Error::invalid_argument("perspective_transform point set is degenerate"))?;
	let values: Vec<f32> = h.iter().map(|value| *value as f32).chain([1.0]).collect();
	Matrix::new_from_f32([3, 3], &values)
}

/// Least-squares estimate the 2x3 affine transform mapping `source` points
/// to `destination` points, matching `estimateAffine2D` without refinement.
/// Both inputs must be `[N, 2]` F32 matrices with `N >= 3` equal rows.
///
/// # Errors
///
/// Returns an error unless both inputs share an `[N, 2]` F32 shape with
/// `N >= 3` and the normal equations are nonsingular.
pub fn estimate_affine_2d(source: &Matrix, destination: &Matrix) -> Result<Matrix> {
	if source.shape() != destination.shape() {
		return Err(Error::invalid_argument(
			"estimate_affine_2d source and destination shapes must match",
		));
	}
	let source = read_points(source, 3, "estimate_affine_2d")?;
	let destination = read_points(destination, 3, "estimate_affine_2d")?;
	let mut normal = [[0.0f64; 6]; 6];
	let mut rhs = [0.0f64; 6];
	for pair in 0..source.len() {
		let [x, y] = source[pair];
		for (row, target) in [
			([x, y, 1.0, 0.0, 0.0, 0.0], destination[pair][0]),
			([0.0, 0.0, 0.0, x, y, 1.0], destination[pair][1]),
		] {
			for i in 0..6 {
				rhs[i] += row[i] * target;
				for j in 0..6 {
					normal[i][j] += row[i] * row[j];
				}
			}
		}
	}
	let parameters = solve(&mut normal, &mut rhs, 6)
		.ok_or_else(|| Error::invalid_argument("estimate_affine_2d point set is degenerate"))?;
	let values: Vec<f32> = parameters.iter().map(|value| *value as f32).collect();
	Matrix::new_from_f32([2, 3], &values)
}

fn read_affine(matrix: &Matrix, operation: &str) -> Result<[f64; 6]> {
	if matrix.shape() != [2, 3] {
		return Err(Error::invalid_argument(format!(
			"{operation} requires a [2, 3] matrix"
		)));
	}
	let values = matrix.read_f32()?;
	let mut affine = [0.0f64; 6];
	for (index, value) in affine.iter_mut().enumerate() {
		*value = values[index] as f64;
	}
	Ok(affine)
}

fn read_points(matrix: &Matrix, minimum: usize, operation: &str) -> Result<Vec<[f64; 2]>> {
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
