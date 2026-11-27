//! Geometric-estimation contract tests: homography, pose, epipolar geometry,
//! distortion, and calibration. All kernels are host-side; the device gate
//! covers `Matrix` construction on the default runtime.

use oa::Result;
use oa::matrix::Matrix;
use oa::vision::{
	EstimationMethod, calibrate_camera, find_essential, find_fundamental, find_homography,
	project_points, recover_pose, rodrigues, solve_pnp, solve_pnp_ransac, triangulate_points,
	undistort_points, undistort_rectify_map,
};

fn points2(values: &[[f32; 2]]) -> Result<Matrix> {
	let flat: Vec<f32> = values.iter().flatten().copied().collect();
	Matrix::new_from_f32([values.len(), 2], &flat)
}

fn points3(values: &[[f32; 3]]) -> Result<Matrix> {
	let flat: Vec<f32> = values.iter().flatten().copied().collect();
	Matrix::new_from_f32([values.len(), 3], &flat)
}

fn mat3(row_major: [f64; 9]) -> Result<Matrix> {
	let flat: Vec<f32> = row_major.iter().map(|v| *v as f32).collect();
	Matrix::new_from_f32([3, 3], &flat)
}

fn apply_homography(h: &[f32], x: f64, y: f64) -> (f64, f64) {
	let w = h[6] as f64 * x + h[7] as f64 * y + h[8] as f64;
	(
		(h[0] as f64 * x + h[1] as f64 * y + h[2] as f64) / w,
		(h[3] as f64 * x + h[4] as f64 * y + h[5] as f64) / w,
	)
}

fn camera() -> Result<Matrix> {
	mat3([600.0, 0.0, 320.0, 0.0, 600.0, 240.0, 0.0, 0.0, 1.0])
}

fn rodrigues_host(r: [f64; 3]) -> [[f64; 3]; 3] {
	let theta = (r[0] * r[0] + r[1] * r[1] + r[2] * r[2]).sqrt();
	if theta < 1e-12 {
		return [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
	}
	let k = [r[0] / theta, r[1] / theta, r[2] / theta];
	let (c, s) = (theta.cos(), theta.sin());
	[
		[
			c + k[0] * k[0] * (1.0 - c),
			k[0] * k[1] * (1.0 - c) - k[2] * s,
			k[0] * k[2] * (1.0 - c) + k[1] * s,
		],
		[
			k[1] * k[0] * (1.0 - c) + k[2] * s,
			c + k[1] * k[1] * (1.0 - c),
			k[1] * k[2] * (1.0 - c) - k[0] * s,
		],
		[
			k[2] * k[0] * (1.0 - c) - k[1] * s,
			k[2] * k[1] * (1.0 - c) + k[0] * s,
			c + k[2] * k[2] * (1.0 - c),
		],
	]
}

fn project_host(k: [[f64; 3]; 3], r: &[[f64; 3]; 3], t: [f64; 3], p: [f64; 3]) -> [f64; 2] {
	let x = r[0][0] * p[0] + r[0][1] * p[1] + r[0][2] * p[2] + t[0];
	let y = r[1][0] * p[0] + r[1][1] * p[1] + r[1][2] * p[2] + t[1];
	let z = r[2][0] * p[0] + r[2][1] * p[1] + r[2][2] * p[2] + t[2];
	[
		(k[0][0] * x + k[0][1] * y + k[0][2] * z) / z,
		(k[1][0] * x + k[1][1] * y + k[1][2] * z) / z,
	]
}

fn scene_points() -> Vec<[f64; 3]> {
	[
		[-0.5, -0.4, 4.0],
		[0.3, -0.6, 5.0],
		[0.8, 0.2, 6.0],
		[-0.2, 0.5, 4.5],
		[0.6, 0.7, 7.0],
		[-0.7, 0.1, 5.5],
		[0.1, -0.3, 6.5],
		[0.4, 0.4, 4.2],
		[-0.4, -0.7, 6.8],
		[0.9, -0.1, 5.2],
	]
	.to_vec()
}

#[test]
#[ignore = "requires a runtime device"]
fn find_homography_recovers_a_known_transform() -> Result<()> {
	// Scaled rotation with translation and mild perspective.
	let truth = [
		0.9, -0.2, 12.0, //
		0.15, 0.95, -7.5, //
		0.0003, -0.0002, 1.0,
	];
	let source: Vec<[f32; 2]> = [
		[10.0, 20.0],
		[80.0, 15.0],
		[45.0, 90.0],
		[120.0, 70.0],
		[25.0, 130.0],
		[95.0, 110.0],
	]
	.to_vec();
	let truth32: Vec<f32> = truth.iter().map(|v| *v as f32).collect();
	let destination: Vec<[f32; 2]> = source
		.iter()
		.map(|&[x, y]| {
			let (u, v) = apply_homography(&truth32, x as f64, y as f64);
			[u as f32, v as f32]
		})
		.collect();
	let result = find_homography(
		&points2(&source)?,
		&points2(&destination)?,
		EstimationMethod::LeastSquares,
		0.0,
		0,
		0,
	)?;
	let h = result.matrix.read_f32()?;
	let mask = result.mask.read_f32()?;
	assert_eq!(mask.len(), source.len());
	assert!(mask.iter().all(|v| *v == 1.0));
	for (x, y) in source.iter().map(|&[x, y]| (x as f64, y as f64)) {
		let (expected_u, expected_v) = apply_homography(&truth32, x, y);
		let (u, v) = apply_homography(&h, x, y);
		assert!((u - expected_u).abs() < 1e-4, "u {u} != {expected_u}");
		assert!((v - expected_v).abs() < 1e-4, "v {v} != {expected_v}");
	}
	Ok(())
}

#[test]
#[ignore = "requires a runtime device"]
fn find_homography_ransac_marks_outliers_deterministically() -> Result<()> {
	let source: Vec<[f32; 2]> = (0..20)
		.map(|i| [(i * 17 % 100) as f32, (i * 31 % 90) as f32])
		.collect();
	let mut destination: Vec<[f32; 2]> = source.iter().map(|&[x, y]| [x + 5.0, y - 3.0]).collect();
	// Corrupt three correspondences as gross outliers.
	destination[4] = [500.0, 500.0];
	destination[11] = [-200.0, 300.0];
	destination[16] = [640.0, -50.0];
	let src = points2(&source)?;
	let dst = points2(&destination)?;
	let run = |seed: u64| -> Result<(Vec<f32>, Vec<f32>)> {
		let result = find_homography(&src, &dst, EstimationMethod::Ransac, 3.0, 500, seed)?;
		Ok((result.matrix.read_f32()?, result.mask.read_f32()?))
	};
	let (h, mask) = run(7)?;
	for &outlier in &[4usize, 11, 16] {
		assert_eq!(mask[outlier], 0.0, "outlier {outlier} marked inlier");
	}
	assert_eq!(mask.iter().filter(|v| **v == 1.0).count(), 17);
	// Translation-only truth: residual transfer error stays inside 3px.
	for (index, &[x, y]) in source.iter().enumerate() {
		if mask[index] == 0.0 {
			continue;
		}
		let (u, v) = apply_homography(&h, x as f64, y as f64);
		assert!((u - (x as f64 + 5.0)).abs() < 0.5);
		assert!((v - (y as f64 - 3.0)).abs() < 0.5);
	}
	// Same seed reproduces the same consensus set; a different seed still
	// isolates the same gross outliers.
	let (_, mask_repeat) = run(7)?;
	assert_eq!(mask, mask_repeat);
	let (_, mask_other) = run(19)?;
	assert_eq!(mask_other.iter().filter(|v| **v == 1.0).count(), 17);
	Ok(())
}

#[test]
#[ignore = "requires a runtime device"]
fn solve_pnp_recovers_a_synthetic_pose() -> Result<()> {
	let k = [[600.0, 0.0, 320.0], [0.0, 600.0, 240.0], [0.0, 0.0, 1.0]];
	let r_truth = [0.05f64, -0.08, 0.12];
	let t_truth = [0.3f64, -0.15, 0.8];
	let rotation = rodrigues_host(r_truth);
	let object = scene_points();
	let image: Vec<[f32; 2]> = object
		.iter()
		.map(|&p| {
			let [u, v] = project_host(k, &rotation, t_truth, p);
			[u as f32, v as f32]
		})
		.collect();
	let result = solve_pnp(
		&points3(
			&object
				.iter()
				.map(|p| [p[0] as f32, p[1] as f32, p[2] as f32])
				.collect::<Vec<_>>(),
		)?,
		&points2(&image)?,
		&camera()?,
	)?;
	let rvec = result.rotation_vector.read_f32()?;
	let tvec = result.translation_vector.read_f32()?;
	for i in 0..3 {
		assert!(
			(rvec[i] as f64 - r_truth[i]).abs() < 1e-3,
			"rvec[{i}] {} != {}",
			rvec[i],
			r_truth[i]
		);
		assert!(
			(tvec[i] as f64 - t_truth[i]).abs() < 1e-3,
			"tvec[{i}] {} != {}",
			tvec[i],
			t_truth[i]
		);
	}
	Ok(())
}

#[test]
#[ignore = "requires a runtime device"]
fn solve_pnp_ransac_rejects_outliers() -> Result<()> {
	let k = [[600.0, 0.0, 320.0], [0.0, 600.0, 240.0], [0.0, 0.0, 1.0]];
	let rotation = rodrigues_host([0.05, -0.08, 0.12]);
	let t_truth = [0.3f64, -0.15, 0.8];
	let object = scene_points();
	let mut image: Vec<[f32; 2]> = object
		.iter()
		.map(|&p| {
			let [u, v] = project_host(k, &rotation, t_truth, p);
			[u as f32, v as f32]
		})
		.collect();
	image[2] = [900.0, 900.0];
	image[7] = [10.0, 800.0];
	let result = solve_pnp_ransac(
		&points3(
			&object
				.iter()
				.map(|p| [p[0] as f32, p[1] as f32, p[2] as f32])
				.collect::<Vec<_>>(),
		)?,
		&points2(&image)?,
		&camera()?,
		4.0,
		200,
		11,
	)?;
	let mask = result.mask.read_f32()?;
	assert_eq!(mask[2], 0.0);
	assert_eq!(mask[7], 0.0);
	let tvec = result.translation_vector.read_f32()?;
	for i in 0..3 {
		assert!((tvec[i] as f64 - t_truth[i]).abs() < 1e-2);
	}
	Ok(())
}

#[test]
#[ignore = "requires a runtime device"]
fn rodrigues_round_trips_in_both_directions() -> Result<()> {
	let r = [0.3f64, -0.2, 0.1];
	let vector = Matrix::new_from_f32([3, 1], &[r[0] as f32, r[1] as f32, r[2] as f32])?;
	let matrix = rodrigues(&vector)?.read_f32()?;
	// Compare against the host oracle.
	let expected = rodrigues_host(r);
	for i in 0..9 {
		assert!(
			(matrix[i] as f64 - expected[i / 3][i % 3]).abs() < 1e-6,
			"matrix[{i}] {} != {}",
			matrix[i],
			expected[i / 3][i % 3]
		);
	}
	let back = rodrigues(&Matrix::new_from_f32([3, 3], &matrix)?)?.read_f32()?;
	for i in 0..3 {
		assert!((back[i] as f64 - r[i]).abs() < 1e-5);
	}
	Ok(())
}

#[test]
#[ignore = "requires a runtime device"]
fn project_points_matches_analytic_projection() -> Result<()> {
	let object = points3(&[[0.1, 0.2, 3.0], [-0.4, 0.5, 4.0]])?;
	let rvec = Matrix::new_from_f32([3, 1], &[0.0, 0.0, 0.0])?;
	let tvec = Matrix::new_from_f32([3, 1], &[0.0, 0.0, 0.0])?;
	let distortion = Matrix::new_from_f32([0], &[])?;
	let projected = project_points(&object, &rvec, &tvec, &camera()?, &distortion)?.read_f32()?;
	// Identity pose: u = fx * x/z + cx, v = fy * y/z + cy.
	let expected = [
		(600.0 * 0.1 / 3.0 + 320.0, 600.0 * 0.2 / 3.0 + 240.0),
		(600.0 * -0.4 / 4.0 + 320.0, 600.0 * 0.5 / 4.0 + 240.0),
	];
	for (index, (eu, ev)) in expected.iter().enumerate() {
		assert!((projected[2 * index] as f64 - eu).abs() < 1e-3);
		assert!((projected[2 * index + 1] as f64 - ev).abs() < 1e-3);
	}
	Ok(())
}

#[test]
#[ignore = "requires a runtime device"]
fn fundamental_satisfies_the_epipolar_constraint() -> Result<()> {
	let k = [[600.0, 0.0, 320.0], [0.0, 600.0, 240.0], [0.0, 0.0, 1.0]];
	let rotation = rodrigues_host([0.02, 0.1, -0.04]);
	let t = [1.0f64, 0.2, 0.05];
	let object = scene_points();
	let identity = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
	let view1: Vec<[f32; 2]> = object
		.iter()
		.map(|&p| {
			let [u, v] = project_host(k, &identity, [0.0, 0.0, 0.0], p);
			[u as f32, v as f32]
		})
		.collect();
	let view2: Vec<[f32; 2]> = object
		.iter()
		.map(|&p| {
			let [u, v] = project_host(k, &rotation, t, p);
			[u as f32, v as f32]
		})
		.collect();
	let result = find_fundamental(
		&points2(&view1)?,
		&points2(&view2)?,
		EstimationMethod::LeastSquares,
		0.0,
		0,
		0,
	)?;
	let f = result.matrix.read_f32()?;
	for index in 0..object.len() {
		let x = [view1[index][0] as f64, view1[index][1] as f64, 1.0];
		let xp = [view2[index][0] as f64, view2[index][1] as f64, 1.0];
		// |x'^T F x| must vanish on true correspondences.
		let fx = [
			f[0] as f64 * x[0] + f[1] as f64 * x[1] + f[2] as f64 * x[2],
			f[3] as f64 * x[0] + f[4] as f64 * x[1] + f[5] as f64 * x[2],
			f[6] as f64 * x[0] + f[7] as f64 * x[1] + f[8] as f64 * x[2],
		];
		let residual = xp[0] * fx[0] + xp[1] * fx[1] + xp[2] * fx[2];
		let scale = (fx[0] * fx[0] + fx[1] * fx[1]).sqrt().max(1e-12);
		assert!(
			residual.abs() / scale < 0.05,
			"epipolar residual {residual} / {scale}"
		);
	}
	Ok(())
}

#[test]
#[ignore = "requires a runtime device"]
fn essential_and_recover_pose_reconstruct_view_two() -> Result<()> {
	let k = [[600.0, 0.0, 320.0], [0.0, 600.0, 240.0], [0.0, 0.0, 1.0]];
	let rotation = rodrigues_host([0.02, 0.1, -0.04]);
	let t_dir = {
		let t = [1.0f64, 0.2, 0.05];
		let n = (t[0] * t[0] + t[1] * t[1] + t[2] * t[2]).sqrt();
		[t[0] / n, t[1] / n, t[2] / n]
	};
	let object = scene_points();
	let identity = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
	let view1: Vec<[f32; 2]> = object
		.iter()
		.map(|&p| {
			let [u, v] = project_host(k, &identity, [0.0, 0.0, 0.0], p);
			[u as f32, v as f32]
		})
		.collect();
	let view2: Vec<[f32; 2]> = object
		.iter()
		.map(|&p| {
			let [u, v] = project_host(k, &rotation, t_dir, p);
			[u as f32, v as f32]
		})
		.collect();
	let camera = camera()?;
	let essential = find_essential(
		&points2(&view1)?,
		&points2(&view2)?,
		&camera,
		EstimationMethod::LeastSquares,
		0.0,
		0,
		0,
	)?;
	let pose = recover_pose(
		&essential.matrix,
		&points2(&view1)?,
		&points2(&view2)?,
		&camera,
	)?;
	let r = pose.rotation.read_f32()?;
	for i in 0..9 {
		assert!(
			(r[i] as f64 - rotation[i / 3][i % 3]).abs() < 1e-2,
			"R[{i}] {} != {}",
			r[i],
			rotation[i / 3][i % 3]
		);
	}
	// Translation is unit-length, direction comparable up to sign.
	let t = pose.translation.read_f32()?;
	let dot: f64 = (0..3).map(|i| t[i] as f64 * t_dir[i]).sum();
	assert!(dot.abs() > 0.999, "|t . t_true| {dot}");
	let mask = pose.mask.read_f32()?;
	assert_eq!(mask.iter().filter(|v| **v == 1.0).count(), object.len());
	Ok(())
}

#[test]
#[ignore = "requires a runtime device"]
fn triangulate_points_recovers_scene_depth() -> Result<()> {
	let k = [[600.0, 0.0, 320.0], [0.0, 600.0, 240.0], [0.0, 0.0, 1.0]];
	let rotation = rodrigues_host([0.02, 0.1, -0.04]);
	let t = [0.5f64, 0.1, 0.0];
	let object = scene_points();
	let identity = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
	let view1: Vec<[f32; 2]> = object
		.iter()
		.map(|&p| {
			let [u, v] = project_host(k, &identity, [0.0, 0.0, 0.0], p);
			[u as f32, v as f32]
		})
		.collect();
	let view2: Vec<[f32; 2]> = object
		.iter()
		.map(|&p| {
			let [u, v] = project_host(k, &rotation, t, p);
			[u as f32, v as f32]
		})
		.collect();
	// P1 = K [I | 0], P2 = K [R | t] as [3, 4] matrices.
	let mut p1 = Vec::with_capacity(12);
	for krow in k.iter() {
		for &value in krow.iter().chain([0.0].iter()) {
			p1.push(value as f32);
		}
	}
	// P2 = K * [R | t].
	let mut p2v = Vec::with_capacity(12);
	// Column-major view of [R | t] for the dot products.
	let rt_columns: [[f64; 3]; 4] = [
		[rotation[0][0], rotation[1][0], rotation[2][0]],
		[rotation[0][1], rotation[1][1], rotation[2][1]],
		[rotation[0][2], rotation[1][2], rotation[2][2]],
		t,
	];
	for krow in k.iter() {
		for column in rt_columns.iter() {
			let sum: f64 = krow.iter().zip(column.iter()).map(|(kv, rv)| kv * rv).sum();
			p2v.push(sum as f32);
		}
	}
	let projection1 = Matrix::new_from_f32([3, 4], &p1)?;
	let projection2 = Matrix::new_from_f32([3, 4], &p2v)?;
	let hom = triangulate_points(
		&projection1,
		&projection2,
		&points2(&view1)?,
		&points2(&view2)?,
	)?
	.read_f32()?;
	for (index, point) in object.iter().enumerate() {
		let w = hom[4 * index + 3] as f64;
		assert!(w.abs() > 1e-6);
		for axis in 0..3 {
			let estimate = hom[4 * index + axis] as f64 / w;
			assert!(
				(estimate - point[axis]).abs() < 1e-2,
				"point {index} axis {axis}: {estimate} != {}",
				point[axis]
			);
		}
	}
	Ok(())
}

#[test]
#[ignore = "requires a runtime device"]
fn undistort_map_is_identity_under_zero_distortion() -> Result<()> {
	let distortion = Matrix::new_from_f32([0], &[])?;
	let map = undistort_rectify_map(&camera()?, &distortion, None, None, 8, 6)?.read_f32()?;
	// Identity camera under zero distortion maps each output pixel to itself.
	for v in 0..6usize {
		for u in 0..8usize {
			let x = map[v * 8 + u];
			let y = map[48 + v * 8 + u];
			assert!((x as f64 - u as f64).abs() < 1e-4);
			assert!((y as f64 - v as f64).abs() < 1e-4);
		}
	}
	Ok(())
}

#[test]
#[ignore = "requires a runtime device"]
fn undistort_points_inverts_forward_distortion() -> Result<()> {
	let k = [[600.0, 0.0, 320.0], [0.0, 600.0, 240.0], [0.0, 0.0, 1.0]];
	let distortion = Matrix::new_from_f32([5, 1], &[-0.2, 0.05, 0.001, -0.0005, 0.01])?;
	let camera = camera()?;
	// Distort normalized points analytically, then undistort back.
	let normalized = [[0.2f64, 0.1], [-0.3, 0.4], [0.0, -0.5]];
	let pixels: Vec<[f32; 2]> = normalized
		.iter()
		.map(|&[x, y]| {
			let r2 = x * x + y * y;
			let radial = 1.0 - 0.2 * r2 + 0.05 * r2 * r2 + 0.01 * r2 * r2 * r2;
			let dx = 2.0 * 0.001 * x * y - 0.0005 * (r2 + 2.0 * x * x);
			let dy = 0.001 * (r2 + 2.0 * y * y) + 2.0 * -0.0005 * x * y;
			let xd = x * radial + dx;
			let yd = y * radial + dy;
			[
				(k[0][0] * xd + k[0][2]) as f32,
				(k[1][1] * yd + k[1][2]) as f32,
			]
		})
		.collect();
	let out = undistort_points(&points2(&pixels)?, &camera, &distortion, None, None)?.read_f32()?;
	for (index, &[x, y]) in normalized.iter().enumerate() {
		assert!(
			(out[2 * index] as f64 - x).abs() < 1e-3,
			"x {index}: {} != {x}",
			out[2 * index]
		);
		assert!(
			(out[2 * index + 1] as f64 - y).abs() < 1e-3,
			"y {index}: {} != {y}",
			out[2 * index + 1]
		);
	}
	Ok(())
}

#[test]
#[ignore = "requires a runtime device"]
fn calibrate_camera_recovers_intrinsics() -> Result<()> {
	let k_true = [[620.0, 0.0, 310.0], [0.0, 615.0, 245.0], [0.0, 0.0, 1.0]];
	// 4x3 planar grid on z = 0, seen from four distinct poses.
	let mut grid = Vec::new();
	for y in 0..3 {
		for x in 0..4 {
			grid.push([x as f64 * 0.05, y as f64 * 0.05, 0.0]);
		}
	}
	let poses = [
		([0.08, -0.05, 0.10], [0.02, -0.01, 0.5]),
		([-0.12, 0.07, -0.06], [-0.03, 0.02, 0.55]),
		([0.15, 0.02, 0.04], [0.01, 0.03, 0.45]),
		([-0.06, -0.10, 0.12], [-0.01, -0.02, 0.6]),
	];
	let views = poses.len();
	let n = grid.len();
	let mut object = Vec::with_capacity(views * n * 3);
	let mut image = Vec::with_capacity(views * n * 2);
	for (rvec, tvec) in &poses {
		let rotation = rodrigues_host(*rvec);
		for point in &grid {
			object.extend(point.iter().map(|v| *v as f32));
			let [u, v] = project_host(k_true, &rotation, *tvec, *point);
			image.push(u as f32);
			image.push(v as f32);
		}
	}
	let result = calibrate_camera(
		&Matrix::new_from_f32([views, n, 3], &object)?,
		&Matrix::new_from_f32([views, n, 2], &image)?,
		25,
	)?;
	let k = result.camera.read_f32()?;
	assert!((k[0] as f64 - k_true[0][0]).abs() < 2.0, "fx {}", k[0]);
	assert!((k[4] as f64 - k_true[1][1]).abs() < 2.0, "fy {}", k[4]);
	assert!((k[2] as f64 - k_true[0][2]).abs() < 2.0, "cx {}", k[2]);
	assert!((k[5] as f64 - k_true[1][2]).abs() < 2.0, "cy {}", k[5]);
	let rms = result.rms_error.read_f32()?;
	assert!(rms[0] < 0.5, "rms {}", rms[0]);
	Ok(())
}

#[test]
#[ignore = "requires a runtime device"]
fn geometry_ops_reject_degenerate_inputs() -> Result<()> {
	let too_few = points2(&[[1.0, 2.0], [3.0, 4.0]])?;
	assert!(
		find_homography(
			&too_few,
			&too_few,
			EstimationMethod::LeastSquares,
			0.0,
			0,
			0
		)
		.is_err()
	);
	let enough = points2(&[
		[0.0, 0.0],
		[1.0, 0.0],
		[0.0, 1.0],
		[1.0, 1.0],
		[2.0, 0.5],
		[0.5, 2.0],
	])?;
	assert!(find_homography(&enough, &enough, EstimationMethod::Ransac, -1.0, 100, 0).is_err());
	assert!(find_homography(&enough, &enough, EstimationMethod::Ransac, 1.0, 0, 0).is_err());
	let mismatched = points2(&[
		[0.0, 0.0],
		[1.0, 0.0],
		[0.0, 1.0],
		[1.0, 1.0],
		[2.0, 0.5],
		[0.5, 2.0],
		[3.0, 3.0],
	])?;
	assert!(
		find_homography(
			&enough,
			&mismatched,
			EstimationMethod::LeastSquares,
			0.0,
			0,
			0
		)
		.is_err()
	);
	Ok(())
}

// ---- tracking: KalmanFilter -------------------------------------------------

#[test]
#[ignore = "requires a runtime device"]
fn kalman_filter_tracks_constant_velocity() -> Result<()> {
	// Constant-velocity model: state [x, vx]', measurement [x].
	let mut filter = oa::vision::KalmanFilter::new(2, 1, 0)?;
	filter.set_transition_matrix(&Matrix::new_from_f32([2, 2], &[1.0, 1.0, 0.0, 1.0])?)?;
	filter.set_measurement_matrix(&Matrix::new_from_f32([1, 2], &[1.0, 0.0])?)?;
	filter.set_process_noise_cov(&Matrix::new_from_f32([2, 2], &[0.01, 0.0, 0.0, 0.01])?)?;
	filter.set_measurement_noise_cov(&Matrix::new_from_f32([1, 1], &[0.5])?)?;
	filter.set_error_cov_post(&Matrix::new_from_f32([2, 2], &[1.0, 0.0, 0.0, 1.0])?)?;
	// True trajectory x_t = 3 + 2t, noisy measurements.
	for step in 0..10u32 {
		filter.predict(None)?;
		let truth = 3.0 + 2.0 * step as f32;
		let noise = (step as f32 * 0.7).sin() * 0.3;
		filter.correct(&Matrix::new_from_f32([1, 1], &[truth + noise])?)?;
	}
	filter.predict(None)?;
	let state = filter.correct(&Matrix::new_from_f32([1, 1], &[3.0 + 20.0])?)?;
	let values = state.read_f32()?;
	assert!(
		(values[0] - 23.0).abs() < 1.0,
		"kalman position estimate {} should approach 23",
		values[0]
	);
	assert!(
		(values[1] - 2.0).abs() < 1.0,
		"kalman velocity estimate {} should approach 2",
		values[1]
	);
	Ok(())
}

#[test]
#[ignore = "requires a runtime device"]
fn kalman_filter_rejects_bad_shapes() -> Result<()> {
	let mut filter = oa::vision::KalmanFilter::new(2, 1, 0)?;
	assert!(oa::vision::KalmanFilter::new(0, 1, 0).is_err());
	assert!(
		filter
			.set_transition_matrix(&Matrix::new_from_f32([3, 3], &[0.0; 9])?)
			.is_err()
	);
	assert!(
		filter
			.correct(&Matrix::new_from_f32([2, 1], &[1.0, 2.0])?)
			.is_err()
	);
	Ok(())
}
