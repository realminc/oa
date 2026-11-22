use oa::vlm::{
	AffineDecomposition, DMat4, DQuat, DVec3, Mat3, Mat4, Quat, RotationOrder, Vec2, Vec3, Vec4,
	Viewport, cartesian_to_spherical, spherical_to_cartesian,
};

fn close(a: f32, b: f32) -> bool {
	(a - b).abs() <= 1.0e-5_f32.max(1.0e-5 * a.abs().max(b.abs()))
}

fn vec3_close(a: Vec3, b: Vec3) -> bool {
	close(a.x, b.x) && close(a.y, b.y) && close(a.z, b.z)
}

macro_rules! matrix_product_contract {
	($name:ident, $matrix:ident, $scalar:ty, $size:expr) => {
		#[test]
		fn $name() {
			// An independent row/column dot-product oracle preserves IEEE
			// accumulation order. NaN payload propagation is not an API promise.
			let check = |left: $matrix<$scalar>, right: $matrix<$scalar>| {
				let actual = left * right;
				for row in 0..$size {
					for column in 0..$size {
						let mut expected: $scalar = 0.0;
						for inner in 0..$size {
							expected += left.m[row][inner] * right.m[inner][column];
						}
						let value = actual.m[row][column];
						if expected.is_nan() {
							assert!(value.is_nan());
						} else {
							assert_eq!(value.to_bits(), expected.to_bits());
						}
					}
				}
				let mut assigned = left;
				assigned *= right;
				for (value, expected) in assigned.m.iter().flatten().zip(actual.m.iter().flatten()) {
					assert!(value.is_nan() && expected.is_nan() || value.to_bits() == expected.to_bits());
				}
			};
			let mut seed = 0x1234_5678_u32;
			let mut next = || {
				seed = seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
				(seed >> 16) as $scalar / 2048.0 - 16.0
			};
			for _ in 0..128 {
				let left = $matrix {
					m: std::array::from_fn(|_| std::array::from_fn(|_| next())),
				};
				let right = $matrix {
					m: std::array::from_fn(|_| std::array::from_fn(|_| next())),
				};
				check(left, right);
			}
			for special in [
				0.0,
				-0.0,
				<$scalar>::from_bits(1),
				<$scalar>::MIN_POSITIVE,
				<$scalar>::MAX,
				-<$scalar>::MAX,
				<$scalar>::INFINITY,
				<$scalar>::NEG_INFINITY,
				<$scalar>::NAN,
			] {
				for index in 0..($size * $size) {
					let mut left = $matrix::identity();
					left.m[index / $size][index % $size] = special;
					check(
						left,
						$matrix {
							m: [[-0.5; $size]; $size],
						},
					);
					check(
						$matrix {
							m: [[2.0; $size]; $size],
						},
						left,
					);
				}
			}
		}
	};
}

matrix_product_contract!(mat3_product_f32_preserves_accumulation, Mat3, f32, 3);
matrix_product_contract!(mat3_product_f64_preserves_accumulation, Mat3, f64, 3);
matrix_product_contract!(mat4_product_f32_preserves_accumulation, Mat4, f32, 4);
matrix_product_contract!(mat4_product_f64_preserves_accumulation, Mat4, f64, 4);

macro_rules! normalization_contract {
	($test:ident, $value:ident, $scalar:ty, $norm:ident, $length:expr, {$($field:ident: $component:expr),+}) => {
		#[test]
		fn $test() {
			for scale in [
				<$scalar>::from_bits(1), <$scalar>::MIN_POSITIVE / 8.0,
				0.125, 1.0, -1.0, <$scalar>::MAX / 32.0,
			] {
				let input = $value { $($field: ($component as $scalar) * scale),+ };
				let unit = input.try_normalized().expect("finite nonzero normalization");
				let sign = if scale < 0.0 { -1.0 } else { 1.0 };
				$(assert!((unit.$field - sign * ($component as $scalar) / ($length as $scalar)).abs() <= 16.0 * <$scalar>::EPSILON);)+
				assert!((unit.$norm() - 1.0).abs() <= 16.0 * <$scalar>::EPSILON);
				let expected = scale.abs() * ($length as $scalar);
				assert!((input.$norm() - expected).abs() <= (expected * 16.0 * <$scalar>::EPSILON).max(<$scalar>::from_bits(1)));
			}
			let zero = $value { $($field: 0.0 as $scalar),+ };
			assert!(zero.try_normalized().is_none());
			assert_eq!(zero.$norm(), 0.0);
			for bad in [<$scalar>::NAN, <$scalar>::INFINITY, <$scalar>::NEG_INFINITY] {
				let input = $value { $($field: bad),+ };
				assert!(input.try_normalized().is_none());
				assert!(!input.$norm().is_finite());
			}
		}
	};
}

normalization_contract!(vec2_normalization_extremes_f32, Vec2, f32, length, 5.0, {x: 3.0, y: 4.0});
normalization_contract!(vec2_normalization_extremes_f64, Vec2, f64, length, 5.0, {x: 3.0, y: 4.0});
normalization_contract!(vec3_normalization_extremes_f32, Vec3, f32, length, 13.0, {x: 3.0, y: 4.0, z: 12.0});
normalization_contract!(vec3_normalization_extremes_f64, Vec3, f64, length, 13.0, {x: 3.0, y: 4.0, z: 12.0});
normalization_contract!(vec4_normalization_extremes_f32, Vec4, f32, length, 13.0, {x: 3.0, y: 4.0, z: 12.0, w: 0.0});
normalization_contract!(vec4_normalization_extremes_f64, Vec4, f64, length, 13.0, {x: 3.0, y: 4.0, z: 12.0, w: 0.0});
normalization_contract!(quaternion_normalization_extremes_f32, Quat, f32, norm, 13.0, {x: 3.0, y: 4.0, z: 12.0, w: 0.0});
normalization_contract!(quaternion_normalization_extremes_f64, Quat, f64, norm, 13.0, {x: 3.0, y: 4.0, z: 12.0, w: 0.0});

#[test]
fn vector_projection_preserves_scaled_and_invalid_paths() {
	let input = Vec3 {
		x: 3.0_f64,
		y: 4.0,
		z: 0.0,
	};
	for magnitude in [f64::from_bits(1), f64::MAX / 2.0, 1.0] {
		let onto = Vec3 {
			x: magnitude,
			y: 0.0,
			z: 0.0,
		};
		assert_eq!(
			input.try_project_onto(onto, 0.0),
			Some(Vec3 {
				x: 3.0,
				y: 0.0,
				z: 0.0
			})
		);
	}
	assert!(input.try_project_onto(Vec3::default(), 0.0).is_none());
	assert!(input.try_project_onto(input, f64::NAN).is_none());
	assert!(input.try_project_onto(input, -1.0).is_none());
	assert!(
		input
			.try_project_onto(
				Vec3 {
					x: f64::INFINITY,
					y: 0.0,
					z: 0.0
				},
				0.0
			)
			.is_none()
	);
}

#[test]
fn values_are_packed_and_have_stable_defaults() {
	assert_eq!(size_of::<Vec2>(), 2 * size_of::<f32>());
	assert_eq!(size_of::<Vec3>(), 3 * size_of::<f32>());
	assert_eq!(size_of::<Vec4>(), 4 * size_of::<f32>());
	assert_eq!(size_of::<Quat>(), 4 * size_of::<f32>());
	assert_eq!(size_of::<Mat3>(), 9 * size_of::<f32>());
	assert_eq!(size_of::<Mat4>(), 16 * size_of::<f32>());
	assert_eq!(Quat::<f32>::default(), Quat::<f32>::identity());
	assert_eq!(Viewport::<f32>::default().min_depth, 0.0);
	assert_eq!(Viewport::<f32>::default().max_depth, 1.0);
}

#[test]
fn vector_geometry_is_right_handed_and_robust() {
	let x = Vec3 {
		x: 1.0,
		y: 0.0,
		z: 0.0,
	};
	let y = Vec3 {
		x: 0.0,
		y: 1.0,
		z: 0.0,
	};
	assert_eq!(
		x.cross(y),
		Vec3 {
			x: 0.0,
			y: 0.0,
			z: 1.0
		}
	);
	assert_eq!(x.dot(y), 0.0);
	assert!(Vec3::<f32>::default().try_normalized().is_none());
	assert!(
		Vec3 {
			x: f32::NAN,
			y: 0.0,
			z: 0.0
		}
		.try_normalized()
		.is_none()
	);

	let tiny = Vec3 {
		x: f32::MIN_POSITIVE / 8.0,
		y: 0.0,
		z: 0.0,
	};
	assert_eq!(tiny.try_normalized(), Some(x));
	let huge = Vec3 {
		x: f32::MAX / 4.0,
		y: f32::MAX / 4.0,
		z: 0.0,
	};
	let unit = huge.try_normalized().expect("finite extreme vector");
	assert!(close(unit.length(), 1.0));
}

#[test]
fn quaternion_rotation_matches_row_vector_matrix_rotation() {
	let rotation = Quat::try_from_axis_angle(
		Vec3 {
			x: 0.0,
			y: 0.0,
			z: 1.0,
		},
		std::f32::consts::FRAC_PI_2,
	)
	.expect("valid axis angle");
	let vector = Vec3 {
		x: 1.0,
		y: 0.0,
		z: 0.0,
	};
	let quaternion_result = rotation.try_rotate(vector).expect("unit quaternion");
	let matrix = Mat4::try_from_quaternion(rotation).expect("unit quaternion");
	let matrix_result = matrix.transform_direction(vector);
	assert!(vec3_close(
		quaternion_result,
		Vec3 {
			x: 0.0,
			y: 1.0,
			z: 0.0
		}
	));
	assert!(vec3_close(matrix_result, quaternion_result));
}

#[test]
fn all_euler_orders_round_trip_by_rotation() {
	let angles = Vec3 {
		x: 0.31,
		y: -0.47,
		z: 0.83,
	};
	for order in [
		RotationOrder::Xyz,
		RotationOrder::Xzy,
		RotationOrder::Yxz,
		RotationOrder::Yzx,
		RotationOrder::Zxy,
		RotationOrder::Zyx,
	] {
		let rotation = Quat::try_from_euler_radians(angles, order).expect("finite angles");
		let recovered = rotation
			.try_to_euler_radians(order)
			.expect("valid quaternion");
		let round_trip =
			Quat::try_from_euler_radians(recovered, order).expect("recovered finite angles");
		assert!(rotation.approximately_equal(round_trip, 1.0e-5, 1.0e-5));
	}
}

#[test]
fn shortest_path_interpolation_handles_opposite_quaternion_signs() {
	let rotation = Quat::try_from_axis_angle(
		Vec3 {
			x: 0.0,
			y: 1.0,
			z: 0.0,
		},
		0.7,
	)
	.expect("valid rotation");
	let interpolated = rotation
		.try_slerp(-rotation, 0.5)
		.expect("valid interpolation");
	assert!(rotation.approximately_equal(interpolated, 1.0e-5, 1.0e-5));
}

#[test]
fn row_vector_composition_applies_left_matrix_first() {
	let translation = Mat4::translation(Vec3 {
		x: 2.0,
		y: 0.0,
		z: 0.0,
	});
	let scaling = Mat4::scaling(Vec3 {
		x: 3.0,
		y: 3.0,
		z: 3.0,
	});
	let point = Vec3 {
		x: 1.0,
		y: 0.0,
		z: 0.0,
	};
	assert_eq!(
		(translation * scaling).transform_point(point),
		Vec3 {
			x: 9.0,
			y: 0.0,
			z: 0.0
		}
	);
}

#[test]
fn matrix_inverse_rejects_singular_and_recovers_affine_transform() {
	let rotation = Quat::try_from_euler_degrees(
		Vec3 {
			x: 13.0,
			y: -27.0,
			z: 41.0,
		},
		RotationOrder::Yzx,
	)
	.expect("finite angles");
	let transform = Mat4::try_compose_trs(
		Vec3 {
			x: 4.0,
			y: -2.0,
			z: 7.0,
		},
		rotation,
		Vec3 {
			x: 2.0,
			y: 3.0,
			z: 4.0,
		},
	)
	.expect("valid TRS");
	let inverse = transform
		.try_inverse(f32::EPSILON * 32.0)
		.expect("invertible transform");
	assert!((transform * inverse).is_identity(1.0e-4));
	assert!(
		Mat4::scaling(Vec3 {
			x: 1.0,
			y: 0.0,
			z: 1.0
		})
		.try_inverse(f32::EPSILON * 32.0)
		.is_none()
	);
}

#[test]
fn affine_decomposition_preserves_signed_scale_shear_and_reflection() {
	let input = AffineDecomposition {
		translation: Vec3 {
			x: 5.0,
			y: -3.0,
			z: 2.0,
		},
		rotation: Quat::try_from_euler_degrees(
			Vec3 {
				x: 11.0,
				y: 23.0,
				z: -17.0,
			},
			RotationOrder::Zxy,
		)
		.expect("finite angles"),
		scale: Vec3 {
			x: -2.0,
			y: 3.0,
			z: 4.0,
		},
		shear: Vec3 {
			x: 0.2,
			y: -0.1,
			z: 0.3,
		},
		reflected: true,
	};
	let matrix = Mat4::compose_affine(input).expect("finite decomposition");
	let output = matrix.try_decompose_affine(1.0e-5).expect("affine matrix");
	let recomposed = Mat4::compose_affine(output).expect("finite decomposition");
	assert!(recomposed.approximately_equal(matrix, 1.0e-4, 1.0e-4));
	assert!(output.reflected);
	assert!(matrix.try_decompose_trs(1.0e-5).is_none());
}

#[test]
fn look_at_uses_camera_forward_negative_z() {
	let eye = Vec3 {
		x: 0.0,
		y: 0.0,
		z: 5.0,
	};
	let view = Mat4::try_look_at(
		eye,
		Vec3::default(),
		Vec3 {
			x: 0.0,
			y: 1.0,
			z: 0.0,
		},
		1.0e-5,
	)
	.expect("valid camera basis");
	assert!(vec3_close(
		view.transform_point(Vec3::default()),
		Vec3 {
			x: 0.0,
			y: 0.0,
			z: -5.0,
		}
	));
	assert!(
		Mat4::try_look_at(
			eye,
			eye,
			Vec3 {
				x: 0.0,
				y: 1.0,
				z: 0.0
			},
			1.0e-5
		)
		.is_none()
	);
}

#[test]
fn projections_map_vulkan_depth_range() {
	let perspective = Mat4::try_perspective(90.0, 1.0, 1.0, 10.0).expect("valid projection");
	let near = perspective
		.try_project_point(
			Vec3 {
				x: 0.0,
				y: 0.0,
				z: -1.0,
			},
			1.0e-5,
		)
		.expect("project near");
	let far = perspective
		.try_project_point(
			Vec3 {
				x: 0.0,
				y: 0.0,
				z: -10.0,
			},
			1.0e-5,
		)
		.expect("project far");
	assert!(close(near.z, 0.0));
	assert!(close(far.z, 1.0));

	let reversed = Mat4::try_perspective_reverse_z(90.0, 1.0, 1.0, 10.0).expect("valid projection");
	assert!(close(
		reversed
			.try_project_point(
				Vec3 {
					x: 0.0,
					y: 0.0,
					z: -1.0
				},
				1.0e-5
			)
			.expect("project near")
			.z,
		1.0
	));
	assert!(close(
		reversed
			.try_project_point(
				Vec3 {
					x: 0.0,
					y: 0.0,
					z: -10.0
				},
				1.0e-5
			)
			.expect("project far")
			.z,
		0.0
	));
}

#[test]
fn projection_variants_preserve_lens_and_depth_contracts() {
	let off_center = Mat4::try_perspective_off_center(-1.0, 2.0, -2.0, 1.0, 1.0, 9.0)
		.expect("valid off-center perspective");
	assert!(close(
		off_center
			.try_project_point(
				Vec3 {
					x: 0.0,
					y: 0.0,
					z: -1.0,
				},
				1.0e-5,
			)
			.expect("project near")
			.z,
		0.0,
	));
	let shifted = Mat4::try_perspective_shifted(70.0, 1.5, 0.1, 100.0, Vec2 { x: 0.2, y: -0.3 })
		.expect("valid shifted perspective");
	assert!(close(shifted.m[2][0], -0.2));
	assert!(close(shifted.m[2][1], 0.3));

	let orthographic =
		Mat4::try_orthographic(8.0, 6.0, 1.0, 13.0, 2.0).expect("valid orthographic projection");
	assert!(close(
		orthographic
			.try_project_point(
				Vec3 {
					x: 0.0,
					y: 0.0,
					z: -5.0,
				},
				1.0e-5,
			)
			.expect("project point")
			.z,
		1.0 / 3.0,
	));
	let infinite =
		Mat4::try_perspective_reverse_z_infinite(90.0, 1.0, 0.5).expect("valid infinite projection");
	assert!(close(
		infinite
			.try_project_point(
				Vec3 {
					x: 0.0,
					y: 0.0,
					z: -0.5,
				},
				1.0e-5,
			)
			.expect("project near")
			.z,
		1.0,
	));
}

#[test]
fn normal_matrix_uses_inverse_transpose() {
	let transform = Mat4::scaling(Vec3 {
		x: 2.0,
		y: 3.0,
		z: 4.0,
	});
	let normal_matrix = transform
		.try_normal_matrix(f32::EPSILON * 32.0)
		.expect("invertible affine matrix");
	let normal = normal_matrix
		.try_transform_normal(
			Vec3 {
				x: 1.0,
				y: 1.0,
				z: 0.0,
			},
			f32::EPSILON * 32.0,
		)
		.expect("non-zero normal");
	let expected = Vec3 {
		x: 0.5,
		y: 1.0 / 3.0,
		z: 0.0,
	}
	.try_normalized()
	.expect("non-zero expected normal");
	assert!(vec3_close(normal, expected));
}

#[test]
fn rotation_extraction_rejects_scale_shear_and_reflection() {
	assert!(Quat::try_from_rotation_matrix(Mat3::identity(), 1.0e-5).is_some());
	let scaled = Mat3 {
		m: [[2.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
	};
	let sheared = Mat3 {
		m: [[1.0, 0.2, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
	};
	let reflected = Mat3 {
		m: [[-1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
	};
	assert!(Quat::try_from_rotation_matrix(scaled, 1.0e-5).is_none());
	assert!(Quat::try_from_rotation_matrix(sheared, 1.0e-5).is_none());
	assert!(Quat::try_from_rotation_matrix(reflected, 1.0e-5).is_none());
}

#[test]
fn spherical_cartesian_conversion_round_trips() {
	let spherical = Vec3 {
		x: 0.7,
		y: -0.3,
		z: 12.0,
	};
	let cartesian = spherical_to_cartesian(spherical.x, spherical.y, spherical.z);
	let recovered = cartesian_to_spherical(cartesian);
	assert!(vec3_close(spherical, recovered));
}

#[test]
fn top_origin_viewport_round_trips_odd_extents() {
	let viewport = Viewport {
		x: 7.0,
		y: 11.0,
		width: 641.0,
		height: 479.0,
		min_depth: 0.2,
		max_depth: 0.9,
	};
	let point = Vec3 {
		x: -0.37,
		y: 0.61,
		z: 0.43,
	};
	let projected = viewport
		.try_project(point, Mat4::identity(), 1.0e-5)
		.expect("valid viewport");
	let recovered = viewport
		.try_unproject(projected, Mat4::identity(), f32::EPSILON * 32.0)
		.expect("invertible transform");
	assert!(vec3_close(point, recovered));
	assert!(projected.y < viewport.y + viewport.height * 0.5);
}

#[test]
fn double_precision_handles_large_world_inverse() {
	let transform = DMat4::try_compose_trs(
		DVec3 {
			x: 1.0e12,
			y: -2.0e12,
			z: 3.0e12,
		},
		DQuat::try_from_axis_angle(
			DVec3 {
				x: 0.3,
				y: 0.5,
				z: 0.7,
			},
			0.9,
		)
		.expect("valid rotation"),
		DVec3 {
			x: 2.0,
			y: 3.0,
			z: 4.0,
		},
	)
	.expect("valid TRS");
	let inverse = transform
		.try_inverse(f64::EPSILON * 32.0)
		.expect("invertible transform");
	assert!((transform * inverse).is_identity(1.0e-3));
}
