//! Crate-private host linear algebra for small dense problems.
//!
//! Geometric-estimation helpers (image transforms, vision geometry) need
//! Gaussian elimination, normal-equation least squares, and symmetric
//! eigendecomposition over matrices too small for device dispatch. All
//! routines run in f64 on the host; they are not a public API surface.

#![allow(dead_code)]

/// Solve an `n`-by-`n` dense system `a * x = b` by Gaussian elimination with
/// partial pivoting. Returns `None` on a singular pivot.
#[allow(clippy::needless_range_loop)]
pub(crate) fn solve<const N: usize>(
	a: &mut [[f64; N]; N],
	b: &mut [f64; N],
	n: usize,
) -> Option<Vec<f64>> {
	for column in 0..n {
		let pivot = (column..n)
			.max_by(|left, right| a[*left][column].abs().total_cmp(&a[*right][column].abs()))?;
		if a[pivot][column].abs() <= 1e-12 {
			return None;
		}
		a.swap(column, pivot);
		b.swap(column, pivot);
		for row in (column + 1)..n {
			let factor = a[row][column] / a[column][column];
			for entry in column..n {
				a[row][entry] -= factor * a[column][entry];
			}
			b[row] -= factor * b[column];
		}
	}
	let mut x = vec![0.0f64; n];
	for row in (0..n).rev() {
		let mut sum = b[row];
		for entry in (row + 1)..n {
			sum -= a[row][entry] * x[entry];
		}
		x[row] = sum / a[row][row];
	}
	Some(x)
}

/// Solve a tall least-squares system `a * x ~= b` through the normal
/// equations. `a` is row-major `rows`-by-`N` and `b` has `rows` entries.
/// Returns `None` when the normal matrix is singular.
#[allow(clippy::needless_range_loop)]
pub(crate) fn least_squares<const N: usize>(a: &[f64], b: &[f64], rows: usize) -> Option<Vec<f64>> {
	let mut normal = [[0.0f64; N]; N];
	let mut rhs = [0.0f64; N];
	for row in 0..rows {
		let base = row * N;
		for i in 0..N {
			rhs[i] += a[base + i] * b[row];
			for j in 0..N {
				normal[i][j] += a[base + i] * a[base + j];
			}
		}
	}
	solve(&mut normal, &mut rhs, N)
}

/// Jacobi eigendecomposition of a symmetric `n`-by-`n` matrix packed in
/// `a`. Returns eigenvalues ascending with matching eigenvector columns in
/// `vectors[row][column]`. Returns `None` when the sweep limit is hit.
#[allow(clippy::needless_range_loop)]
pub(crate) fn jacobi_eigen<const N: usize>(
	a: &[[f64; N]],
	n: usize,
) -> Option<(Vec<f64>, Vec<Vec<f64>>)> {
	let mut m = vec![vec![0.0f64; n]; n];
	for row in 0..n {
		m[row][..n].copy_from_slice(&a[row][..n]);
	}
	let mut vectors = vec![vec![0.0f64; n]; n];
	for index in 0..n {
		vectors[index][index] = 1.0;
	}
	for _ in 0..(64 * n * n) {
		let mut pivot = (0, 1);
		let mut magnitude = 0.0f64;
		for row in 0..n {
			for column in (row + 1)..n {
				if m[row][column].abs() > magnitude {
					magnitude = m[row][column].abs();
					pivot = (row, column);
				}
			}
		}
		if magnitude <= 1e-14 {
			break;
		}
		let (p, q) = pivot;
		let (app, aqq, apq) = (m[p][p], m[q][q], m[p][q]);
		let theta = (aqq - app) / (2.0 * apq);
		let t = theta.signum() / (theta.abs() + (theta * theta + 1.0).sqrt());
		let cosine = 1.0 / (t * t + 1.0).sqrt();
		let sine = t * cosine;
		for k in 0..n {
			let (mkp, mkq) = (m[k][p], m[k][q]);
			m[k][p] = cosine * mkp - sine * mkq;
			m[k][q] = sine * mkp + cosine * mkq;
		}
		for k in 0..n {
			let (mpk, mqk) = (m[p][k], m[q][k]);
			m[p][k] = cosine * mpk - sine * mqk;
			m[q][k] = sine * mpk + cosine * mqk;
		}
		for k in 0..n {
			let (vkp, vkq) = (vectors[k][p], vectors[k][q]);
			vectors[k][p] = cosine * vkp - sine * vkq;
			vectors[k][q] = sine * vkp + cosine * vkq;
		}
	}
	let mut order: Vec<usize> = (0..n).collect();
	order.sort_by(|left, right| m[*left][*left].total_cmp(&m[*right][*right]));
	let values = order.iter().map(|index| m[*index][*index]).collect();
	let mut sorted = vec![vec![0.0f64; n]; n];
	for (column, index) in order.iter().enumerate() {
		for row in 0..n {
			sorted[row][column] = vectors[row][*index];
		}
	}
	Some((values, sorted))
}

/// Right singular vectors of `a` (row-major `rows`-by-`N`) as the eigenbasis
/// of `a^T a`, smallest eigenvalue first. Use column `0` for the null-space
/// solution of an overdetermined homogeneous system.
#[allow(clippy::needless_range_loop)]
pub(crate) fn smallest_right_singular<const N: usize>(
	a: &[f64],
	rows: usize,
) -> Option<Vec<Vec<f64>>> {
	let mut normal = [[0.0f64; N]; N];
	for row in 0..rows {
		let base = row * N;
		for i in 0..N {
			for j in 0..N {
				normal[i][j] += a[base + i] * a[base + j];
			}
		}
	}
	jacobi_eigen(&normal, N).map(|(_, vectors)| vectors)
}

/// Solve `A x = b` for a row-major `rows`-by-`n` system using dynamic storage.
/// Returns `None` on a singular pivot.
#[allow(clippy::needless_range_loop)]
pub(crate) fn solve_dynamic(a: &mut [Vec<f64>], b: &mut [f64], n: usize) -> Option<Vec<f64>> {
	for column in 0..n {
		let pivot = (column..n)
			.max_by(|left, right| a[*left][column].abs().total_cmp(&a[*right][column].abs()))?;
		if a[pivot][column].abs() <= 1e-12 {
			return None;
		}
		a.swap(column, pivot);
		b.swap(column, pivot);
		for row in (column + 1)..n {
			let factor = a[row][column] / a[column][column];
			for entry in column..n {
				a[row][entry] -= factor * a[column][entry];
			}
			b[row] -= factor * b[column];
		}
	}
	let mut x = vec![0.0f64; n];
	for row in (0..n).rev() {
		let mut sum = b[row];
		for entry in (row + 1)..n {
			sum -= a[row][entry] * x[entry];
		}
		x[row] = sum / a[row][row];
	}
	Some(x)
}

/// Levenberg-style damped normal solve: `(J^T J + lambda * diag) x = J^T r`.
/// `j` is row-major `rows`-by-`n`. Returns the step vector.
#[allow(clippy::needless_range_loop)]
pub(crate) fn damped_least_squares(
	j: &[f64],
	r: &[f64],
	rows: usize,
	n: usize,
	lambda: f64,
) -> Option<Vec<f64>> {
	let mut normal = vec![vec![0.0f64; n]; n];
	let mut rhs = vec![0.0f64; n];
	for row in 0..rows {
		let base = row * n;
		for i in 0..n {
			rhs[i] += j[base + i] * r[row];
			for k in 0..n {
				normal[i][k] += j[base + i] * j[base + k];
			}
		}
	}
	for i in 0..n {
		normal[i][i] += lambda * normal[i][i].abs().max(1e-6);
	}
	solve_dynamic(&mut normal, &mut rhs, n)
}
