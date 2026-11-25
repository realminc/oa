//! Host transform-matrix helper contract tests.

use oa::Result;
use oa::image;
use oa::matrix::Matrix;

fn apply_affine(matrix: &[f32], x: f64, y: f64) -> (f64, f64) {
	(
		matrix[0] as f64 * x + matrix[1] as f64 * y + matrix[2] as f64,
		matrix[3] as f64 * x + matrix[4] as f64 * y + matrix[5] as f64,
	)
}

#[test]
#[ignore = "requires a runtime device"]
fn rotation_matrix_2d_rotates_and_scales_about_center() -> Result<()> {
	// 90 degrees counterclockwise about the origin at unit scale.
	let rotation = image::rotation_matrix_2d(0.0, 0.0, 90.0, 1.0)?.read_f32()?;
	assert_eq!(rotation.len(), 6);
	assert!((rotation[0] as f64).abs() < 1e-6);
	assert!((rotation[1] as f64 - 1.0).abs() < 1e-6);
	assert!((rotation[3] as f64 + 1.0).abs() < 1e-6);
	assert!((rotation[4] as f64).abs() < 1e-6);
	// Center is a fixed point under rotation about itself.
	let centered = image::rotation_matrix_2d(2.0, 3.0, 30.0, 2.0)?.read_f32()?;
	let (x, y) = apply_affine(&centered, 2.0, 3.0);
	assert!((x - 2.0).abs() < 1e-5);
	assert!((y - 3.0).abs() < 1e-5);
	Ok(())
}

#[test]
#[ignore = "requires a runtime device"]
fn invert_affine_recovers_the_input_transform() -> Result<()> {
	let forward = image::rotation_matrix_2d(1.0, -2.0, 25.0, 1.5)?.read_f32()?;
	let inverse = image::invert_affine(&Matrix::new_from_f32([2, 3], &forward)?)?.read_f32()?;
	// Composing forward then inverse returns each sample point.
	for (x, y) in [(0.0, 0.0), (3.5, -1.25), (-4.0, 2.0)] {
		let (ux, uy) = apply_affine(&forward, x, y);
		let (rx, ry) = apply_affine(&inverse, ux, uy);
		assert!((rx - x).abs() < 1e-4, "x {rx} != {x}");
		assert!((ry - y).abs() < 1e-4, "y {ry} != {y}");
	}
	// Singular linear block rejects.
	let singular = Matrix::new_from_f32([2, 3], &[1.0, 2.0, 0.0, 2.0, 4.0, 0.0])?;
	assert!(image::invert_affine(&singular).is_err());
	Ok(())
}

#[test]
#[ignore = "requires a runtime device"]
fn perspective_transform_maps_all_four_points() -> Result<()> {
	let source = Matrix::new_from_f32([4, 2], &[0.0, 0.0, 8.0, 0.0, 8.0, 4.0, 0.0, 4.0])?;
	let destination = Matrix::new_from_f32([4, 2], &[1.0, 1.0, 7.0, 0.5, 7.5, 5.0, 0.5, 4.5])?;
	let homography = image::perspective_transform(&source, &destination)?.read_f32()?;
	assert_eq!(homography.len(), 9);
	assert!((homography[8] as f64 - 1.0).abs() < 1e-6);
	let source_points = source.read_f32()?;
	let destination_points = destination.read_f32()?;
	for pair in 0..4 {
		let (x, y) = (
			source_points[2 * pair] as f64,
			source_points[2 * pair + 1] as f64,
		);
		let h: Vec<f64> = homography.iter().map(|v| *v as f64).collect();
		let w = h[6] * x + h[7] * y + h[8];
		let u = (h[0] * x + h[1] * y + h[2]) / w;
		let v = (h[3] * x + h[4] * y + h[5]) / w;
		assert!((u - destination_points[2 * pair] as f64).abs() < 1e-4);
		assert!((v - destination_points[2 * pair + 1] as f64).abs() < 1e-4);
	}
	// Collinear source points reject.
	let degenerate = Matrix::new_from_f32([4, 2], &[0.0, 0.0, 1.0, 1.0, 2.0, 2.0, 3.0, 3.0])?;
	assert!(image::perspective_transform(&degenerate, &destination).is_err());
	Ok(())
}

#[test]
#[ignore = "requires a runtime device"]
fn estimate_affine_2d_recovers_a_known_transform() -> Result<()> {
	let forward = image::rotation_matrix_2d(-1.0, 0.5, -15.0, 1.25)?.read_f32()?;
	let mut source = Vec::new();
	let mut destination = Vec::new();
	for index in 0..8 {
		let x = (index % 4) as f64 * 2.0 - 3.0;
		let y = (index / 4) as f64 * 3.0 + 1.0 + index as f64 * 0.5;
		source.extend_from_slice(&[x as f32, y as f32]);
		let (u, v) = apply_affine(&forward, x, y);
		destination.extend_from_slice(&[u as f32, v as f32]);
	}
	let estimated = image::estimate_affine_2d(
		&Matrix::new_from_f32([8, 2], &source)?,
		&Matrix::new_from_f32([8, 2], &destination)?,
	)?
	.read_f32()?;
	for index in 0..6 {
		assert!(
			(estimated[index] as f64 - forward[index] as f64).abs() < 1e-4,
			"index {index}: {estimated:?} vs {forward:?}"
		);
	}
	// Fewer than three points rejects.
	let two = Matrix::new_from_f32([2, 2], &[0.0, 0.0, 1.0, 1.0])?;
	assert!(image::estimate_affine_2d(&two, &two).is_err());
	Ok(())
}
