use oa::{
	anim::{
		AnimInterpolation, AnimWrapMode, Joint, JointDegreesOfFreedom, JointRotationLimits, Keyframe,
	},
	core::{Duration, Transform},
	vlm::{AffineDecomposition, Mat4, Quat, RotationOrder, Vec3},
};

fn expect_vec3_near(actual: Vec3, expected: Vec3, tolerance: f32) {
	assert!(
		(actual.x - expected.x).abs() <= tolerance
			&& (actual.y - expected.y).abs() <= tolerance
			&& (actual.z - expected.z).abs() <= tolerance,
		"expected {expected:?}, got {actual:?}"
	);
}

fn expect_mat4_near(actual: Mat4, expected: Mat4, tolerance: f32) {
	for row in 0..4 {
		for column in 0..4 {
			assert!(
				(actual.m[row][column] - expected.m[row][column]).abs() <= tolerance,
				"mismatch at [{row}][{column}]: expected {expected:?}, got {actual:?}"
			);
		}
	}
}

fn transform_at(position: Vec3, rotation: Quat, scale: Vec3) -> Transform {
	Transform::new(position, rotation, scale, Vec3::default())
}

fn identity_transform_at(position: Vec3) -> Transform {
	Transform::new(
		position,
		Quat::identity(),
		Vec3 {
			x: 1.0,
			y: 1.0,
			z: 1.0,
		},
		Vec3::default(),
	)
}

fn radians(degrees: f32) -> f32 {
	oa::vlm::radians(degrees)
}

// ---- Joint ---------------------------------------------------------------

#[test]
fn joint_member_and_function_set_operate_on_one_locked_value() {
	let mut joint = Joint::default();
	joint.transform_mut().set_position(Vec3 {
		x: 2.0,
		y: 3.0,
		z: 4.0,
	});
	joint
		.set_orientation(
			Quat::try_from_axis_angle(
				Vec3 {
					x: 0.0,
					y: 1.0,
					z: 0.0,
				},
				radians(25.0),
			)
			.expect("axis angle"),
		)
		.expect("set orientation");
	joint
		.set_scale_orientation(
			Quat::try_from_axis_angle(
				Vec3 {
					x: 1.0,
					y: 0.0,
					z: 0.0,
				},
				radians(15.0),
			)
			.expect("axis angle"),
		)
		.expect("set scale orientation");
	joint.set_rotation_order(RotationOrder::Zyx);
	joint
		.set_preferred_angles(Vec3 {
			x: 10.0,
			y: 20.0,
			z: 30.0,
		})
		.expect("preferred angles");
	joint
		.set_stiffness(Vec3 {
			x: 0.1,
			y: 0.2,
			z: 0.3,
		})
		.expect("stiffness");

	assert!(joint.validate().is_ok());
	assert_eq!(joint.rotation_order(), RotationOrder::Zyx);
	expect_vec3_near(
		joint.preferred_angles(),
		Vec3 {
			x: 10.0,
			y: 20.0,
			z: 30.0,
		},
		1.0e-4,
	);
	expect_vec3_near(
		joint.stiffness(),
		Vec3 {
			x: 0.1,
			y: 0.2,
			z: 0.3,
		},
		1.0e-4,
	);
}

#[test]
fn joint_matrix_matches_maya_joint_factor_order() {
	let mut joint = Joint::default();
	joint.transform_mut().set_position(Vec3 {
		x: 7.0,
		y: -3.0,
		z: 2.0,
	});
	joint.transform_mut().set_scale(Vec3 {
		x: 2.0,
		y: 3.0,
		z: 4.0,
	});
	joint.transform_mut().set_shear(Vec3 {
		x: 0.1,
		y: -0.2,
		z: 0.3,
	});
	joint
		.set_scale_orientation(
			Quat::try_from_axis_angle(
				Vec3 {
					x: 1.0,
					y: 0.0,
					z: 0.0,
				},
				radians(13.0),
			)
			.expect("axis angle"),
		)
		.expect("set scale orientation");
	joint.transform_mut().set_rotation(
		Quat::try_from_axis_angle(
			Vec3 {
				x: 0.0,
				y: 1.0,
				z: 0.0,
			},
			radians(29.0),
		)
		.expect("axis angle"),
	);
	joint
		.set_orientation(
			Quat::try_from_axis_angle(
				Vec3 {
					x: 0.0,
					y: 0.0,
					z: 1.0,
				},
				radians(-37.0),
			)
			.expect("axis angle"),
		)
		.expect("set orientation");

	let scale_and_shear = AffineDecomposition {
		scale: joint.transform().scale(),
		shear: joint.transform().shear(),
		..AffineDecomposition::default()
	};
	let parent_scale = Vec3 {
		x: 5.0,
		y: 2.0,
		z: 0.5,
	};
	let expected = Mat4::compose_affine(scale_and_shear).expect("compose")
		* Mat4::try_from_quaternion(joint.scale_orientation()).expect("quat")
		* Mat4::try_from_quaternion(joint.transform().rotation()).expect("quat")
		* Mat4::try_from_quaternion(joint.orientation()).expect("quat")
		* Mat4::scaling(Vec3 {
			x: 0.2,
			y: 0.5,
			z: 2.0,
		}) * Mat4::translation(joint.transform().position());
	expect_mat4_near(
		joint.matrix(parent_scale).expect("matrix"),
		expected,
		2.0e-5,
	);
	expect_mat4_near(
		Mat4::try_from_quaternion(joint.oriented_rotation()).expect("quat"),
		Mat4::try_from_quaternion(joint.transform().rotation()).expect("quat")
			* Mat4::try_from_quaternion(joint.orientation()).expect("quat"),
		2.0e-5,
	);
}

#[test]
fn joint_segment_scale_compensation_can_be_disabled() {
	let mut joint = Joint::default();
	joint.transform_mut().set_scale(Vec3 {
		x: 2.0,
		y: 3.0,
		z: 4.0,
	});
	let parent_scale = Vec3 {
		x: 5.0,
		y: 6.0,
		z: 7.0,
	};
	let compensated = joint.matrix(parent_scale).expect("matrix");
	joint.set_segment_scale_compensate(false);
	let uncompensated = joint.matrix(parent_scale).expect("matrix");

	assert_ne!(compensated, uncompensated);
	expect_mat4_near(
		uncompensated,
		Mat4::scaling(Vec3 {
			x: 2.0,
			y: 3.0,
			z: 4.0,
		}),
		1.0e-4,
	);
}

#[test]
fn joint_orientation_and_limits_match_authoring_semantics() {
	let mut joint = Joint::default();
	joint
		.set_orientation(
			Quat::try_from_euler_degrees(
				Vec3 {
					x: 0.0,
					y: 0.0,
					z: 90.0,
				},
				RotationOrder::Xyz,
			)
			.expect("euler"),
		)
		.expect("set orientation");
	joint.transform_mut().set_rotation(
		Quat::try_from_euler_degrees(
			Vec3 {
				x: 0.0,
				y: 0.0,
				z: 90.0,
			},
			RotationOrder::Xyz,
		)
		.expect("euler"),
	);
	expect_vec3_near(
		joint
			.oriented_rotation()
			.try_rotate(Vec3 {
				x: 1.0,
				y: 0.0,
				z: 0.0,
			})
			.expect("rotate"),
		Vec3 {
			x: -1.0,
			y: 0.0,
			z: 0.0,
		},
		1.0e-4,
	);

	let limits = JointRotationLimits {
		minimum_degrees: Vec3 {
			x: -10.0,
			y: -20.0,
			z: -30.0,
		},
		maximum_degrees: Vec3 {
			x: 10.0,
			y: 20.0,
			z: 30.0,
		},
		minimum_enabled: JointDegreesOfFreedom {
			x: true,
			y: false,
			z: true,
		},
		maximum_enabled: JointDegreesOfFreedom {
			x: true,
			y: true,
			z: false,
		},
	};
	joint.set_rotation_limits(limits).expect("limits");
	expect_vec3_near(
		joint
			.clamp_rotation_degrees(Vec3 {
				x: -50.0,
				y: 50.0,
				z: 50.0,
			})
			.expect("clamp"),
		Vec3 {
			x: -10.0,
			y: 20.0,
			z: 50.0,
		},
		1.0e-4,
	);
}

#[test]
fn joint_rejects_invalid_authoring_state() {
	let mut joint = Joint::default();
	assert!(
		joint
			.set_orientation(Quat {
				x: 0.0,
				y: 0.0,
				z: 0.0,
				w: 0.0,
			})
			.is_err()
	);
	assert!(
		joint
			.set_stiffness(Vec3 {
				x: -1.0,
				y: 0.0,
				z: 0.0,
			})
			.is_err()
	);

	let limits = JointRotationLimits {
		minimum_degrees: Vec3 {
			x: 1.0,
			y: 0.0,
			z: 0.0,
		},
		maximum_degrees: Vec3 {
			x: -1.0,
			y: 0.0,
			z: 0.0,
		},
		minimum_enabled: JointDegreesOfFreedom {
			x: true,
			y: false,
			z: false,
		},
		maximum_enabled: JointDegreesOfFreedom {
			x: true,
			y: false,
			z: false,
		},
	};
	assert!(joint.set_rotation_limits(limits).is_err());
	assert!(
		joint
			.matrix(Vec3 {
				x: 0.0,
				y: 1.0,
				z: 1.0,
			})
			.is_err()
	);
}

// ---- AnimClip / SkelPose ---------------------------------------------------

#[test]
fn anim_creates_pose_and_uniform_clip() {
	let frames = [
		identity_transform_at(Vec3::default()),
		identity_transform_at(Vec3 {
			x: 1.0,
			y: 0.0,
			z: 0.0,
		}),
		identity_transform_at(Vec3 {
			x: 10.0,
			y: 0.0,
			z: 0.0,
		}),
		identity_transform_at(Vec3 {
			x: 11.0,
			y: 0.0,
			z: 0.0,
		}),
	];
	let clip = oa::anim::create_clip(42, 2, 1.0, &frames).expect("create clip");
	assert_eq!(clip.skeleton_id(), 42);
	assert_eq!(clip.joint_count(), 2);
	assert_eq!(clip.frame_count(), 2);
	assert_eq!(clip.duration_seconds(), 1.0);

	let frame = oa::anim::frame(&clip, 1).expect("frame");
	assert_eq!(frame.skeleton_id(), 42);
	expect_vec3_near(
		frame.local_transform(1).expect("joint").position(),
		Vec3 {
			x: 11.0,
			y: 0.0,
			z: 0.0,
		},
		1.0e-4,
	);
	assert!(frame.local_transform(frame.joint_count()).is_none());
}

#[test]
fn anim_samples_translation_rotation_scale_and_wrap_modes() {
	let quarter_turn = Quat::try_from_axis_angle(
		Vec3 {
			x: 0.0,
			y: 0.0,
			z: 1.0,
		},
		radians(90.0),
	)
	.expect("axis angle");
	let mut frames = [
		identity_transform_at(Vec3::default()),
		transform_at(
			Vec3 {
				x: 10.0,
				y: 0.0,
				z: 0.0,
			},
			quarter_turn,
			Vec3 {
				x: 3.0,
				y: 3.0,
				z: 3.0,
			},
		),
	];
	frames[1].set_shear(Vec3 {
		x: 0.2,
		y: 0.4,
		z: 0.6,
	});
	let clip = oa::anim::create_clip(7, 1, 1.0, &frames).expect("create clip");

	let middle = clip.sample(0.5, AnimWrapMode::Clamp).expect("sample");
	let transform = middle.local_transform(0).expect("joint");
	expect_vec3_near(
		transform.position(),
		Vec3 {
			x: 5.0,
			y: 0.0,
			z: 0.0,
		},
		1.0e-4,
	);
	expect_vec3_near(
		transform.scale(),
		Vec3 {
			x: 2.0,
			y: 2.0,
			z: 2.0,
		},
		1.0e-4,
	);
	expect_vec3_near(
		transform.shear(),
		Vec3 {
			x: 0.1,
			y: 0.2,
			z: 0.3,
		},
		1.0e-4,
	);
	expect_vec3_near(
		transform
			.rotation()
			.try_rotate(Vec3 {
				x: 1.0,
				y: 0.0,
				z: 0.0,
			})
			.expect("rotate"),
		Vec3 {
			x: std::f32::consts::FRAC_1_SQRT_2,
			y: std::f32::consts::FRAC_1_SQRT_2,
			z: 0.0,
		},
		1.0e-4,
	);

	let before = clip.sample(-100.0, AnimWrapMode::Clamp).expect("before");
	let after = clip.sample(100.0, AnimWrapMode::Clamp).expect("after");
	let looped = clip.sample(2.5, AnimWrapMode::Loop).expect("looped");
	let negative_loop = clip
		.sample(-0.5, AnimWrapMode::Loop)
		.expect("negative loop");
	expect_vec3_near(
		before.local_transform(0).expect("joint").position(),
		Vec3::default(),
		1.0e-4,
	);
	expect_vec3_near(
		after.local_transform(0).expect("joint").position(),
		Vec3 {
			x: 10.0,
			y: 0.0,
			z: 0.0,
		},
		1.0e-4,
	);
	expect_vec3_near(
		looped.local_transform(0).expect("joint").position(),
		Vec3 {
			x: 5.0,
			y: 0.0,
			z: 0.0,
		},
		1.0e-4,
	);
	expect_vec3_near(
		negative_loop.local_transform(0).expect("joint").position(),
		Vec3 {
			x: 5.0,
			y: 0.0,
			z: 0.0,
		},
		1.0e-4,
	);
}

#[test]
fn anim_rejects_malformed_values() {
	let transform = Transform::default();
	assert!(oa::anim::create_pose(0, &[]).is_err());
	assert!(oa::anim::create_clip(0, 0, 30.0, &[]).is_err());
	assert!(oa::anim::create_clip(0, 2, 30.0, &[transform]).is_err());

	let clip = oa::anim::create_clip(0, 1, 30.0, &[transform]).expect("create clip");
	assert!(clip.sample(f64::INFINITY, AnimWrapMode::Clamp).is_err());
	let extreme = clip.sample(f64::MAX, AnimWrapMode::Clamp).expect("sample");
	assert_eq!(extreme.joint_count(), 1);
}

// ---- AnimCurve -------------------------------------------------------------

fn seconds(value: i64) -> Duration {
	Duration::from_seconds(value).expect("duration seconds")
}

fn milliseconds(value: i64) -> Duration {
	Duration::from_milliseconds(value).expect("duration milliseconds")
}

#[test]
fn curve_linear_scalar_uses_signed_clip_relative_time_and_wrap() {
	let keyframes = [
		Keyframe {
			time: seconds(-1),
			value: -10.0_f32,
			..Default::default()
		},
		Keyframe {
			time: seconds(0),
			value: 0.0,
			..Default::default()
		},
		Keyframe {
			time: seconds(1),
			value: 10.0,
			..Default::default()
		},
	];
	let curve = oa::anim::create_curve(AnimInterpolation::Linear, &keyframes).expect("create curve");
	assert_eq!(curve.keyframe_count(), 3);
	assert_eq!(curve.start_time(), seconds(-1));
	assert_eq!(curve.end_time(), seconds(1));
	assert_eq!(curve.duration(), seconds(2));
	assert!(oa::anim::validate_curve(&curve).is_ok());

	let middle = curve
		.sample(milliseconds(500), AnimWrapMode::Clamp)
		.expect("middle");
	let before = curve
		.sample(seconds(-100), AnimWrapMode::Clamp)
		.expect("before");
	let after = curve
		.sample(seconds(100), AnimWrapMode::Clamp)
		.expect("after");
	let looped = curve
		.sample(milliseconds(1_500), AnimWrapMode::Loop)
		.expect("looped");
	let negative_loop = curve
		.sample(milliseconds(-1_500), AnimWrapMode::Loop)
		.expect("negative loop");
	assert!((middle - 5.0).abs() <= f32::EPSILON * 10.0);
	assert!((before - (-10.0)).abs() <= f32::EPSILON * 10.0);
	assert!((after - 10.0).abs() <= f32::EPSILON * 10.0);
	assert!((looped - (-5.0)).abs() <= f32::EPSILON * 10.0);
	assert!((negative_loop - 5.0).abs() <= f32::EPSILON * 10.0);
}

#[test]
fn curve_step_uses_the_left_key_and_exact_keys_advance() {
	let keyframes = [
		Keyframe {
			time: seconds(0),
			value: 2.0_f32,
			..Default::default()
		},
		Keyframe {
			time: seconds(1),
			value: 7.0,
			..Default::default()
		},
		Keyframe {
			time: seconds(2),
			value: 9.0,
			..Default::default()
		},
	];
	let curve = oa::anim::create_curve(AnimInterpolation::Step, &keyframes).expect("create curve");
	let before_second = curve
		.sample(milliseconds(999), AnimWrapMode::Clamp)
		.expect("before second");
	let at_second = curve
		.sample(seconds(1), AnimWrapMode::Clamp)
		.expect("at second");
	assert!((before_second - 2.0).abs() <= f32::EPSILON * 10.0);
	assert!((at_second - 7.0).abs() <= f32::EPSILON * 10.0);
}

#[test]
fn curve_cubic_spline_uses_per_second_tangents() {
	let keyframes = [
		Keyframe {
			time: seconds(0),
			value: 0.0_f32,
			out_tangent: 2.0,
			..Default::default()
		},
		Keyframe {
			time: seconds(2),
			value: 2.0,
			..Default::default()
		},
	];
	let curve =
		oa::anim::create_curve(AnimInterpolation::CubicSpline, &keyframes).expect("create curve");
	let value = curve
		.sample(seconds(1), AnimWrapMode::Clamp)
		.expect("sample");
	assert!((value - 1.5).abs() <= 1.0e-6, "expected 1.5, got {value}");
}

#[test]
fn curve_supports_vec3_and_normalized_quaternion_curves() {
	let default_key = Keyframe::<Quat>::default();
	assert!((default_key.value.norm_squared() - 1.0).abs() <= 1.0e-6);
	assert!((default_key.in_tangent.norm_squared()).abs() <= 1.0e-6);
	assert!((default_key.out_tangent.norm_squared()).abs() <= 1.0e-6);

	let position_keys = [
		Keyframe {
			time: seconds(0),
			value: Vec3 {
				x: 0.0,
				y: 2.0,
				z: 4.0,
			},
			..Default::default()
		},
		Keyframe {
			time: seconds(1),
			value: Vec3 {
				x: 10.0,
				y: 4.0,
				z: 0.0,
			},
			..Default::default()
		},
	];
	let positions = oa::anim::create_curve(AnimInterpolation::Linear, &position_keys).expect("curve");
	let position = positions
		.sample(milliseconds(250), AnimWrapMode::Clamp)
		.expect("sample");
	expect_vec3_near(
		position,
		Vec3 {
			x: 2.5,
			y: 2.5,
			z: 3.0,
		},
		1.0e-4,
	);

	let quarter_turn = Quat::try_from_axis_angle(
		Vec3 {
			x: 0.0,
			y: 0.0,
			z: 1.0,
		},
		radians(90.0),
	)
	.expect("axis angle");
	let rotation_keys = [
		Keyframe {
			time: seconds(0),
			value: Quat {
				x: 0.0,
				y: 0.0,
				z: 0.0,
				w: 2.0,
			},
			..Default::default()
		},
		Keyframe {
			time: seconds(1),
			value: quarter_turn,
			..Default::default()
		},
	];
	let rotations =
		oa::anim::create_curve(AnimInterpolation::CubicSpline, &rotation_keys).expect("curve");
	assert!((rotations.keyframes()[0].value.norm() - 1.0).abs() <= 1.0e-6);
	let rotation = rotations
		.sample(milliseconds(500), AnimWrapMode::Clamp)
		.expect("sample");
	assert!((rotation.norm() - 1.0).abs() <= 1.0e-6);
	expect_vec3_near(
		rotation
			.try_rotate(Vec3 {
				x: 1.0,
				y: 0.0,
				z: 0.0,
			})
			.expect("rotate"),
		Vec3 {
			x: std::f32::consts::FRAC_1_SQRT_2,
			y: std::f32::consts::FRAC_1_SQRT_2,
			z: 0.0,
		},
		1.0e-4,
	);
}

#[test]
fn curve_bake_uses_explicit_uniform_sample_contract() {
	let keyframes = [
		Keyframe {
			time: seconds(0),
			value: 0.0_f32,
			..Default::default()
		},
		Keyframe {
			time: seconds(1),
			value: 10.0,
			..Default::default()
		},
	];
	let curve = oa::anim::create_curve(AnimInterpolation::Linear, &keyframes).expect("curve");
	let baked = oa::anim::bake(&curve, Duration::ZERO, 2.0, 3, AnimWrapMode::Clamp).expect("bake");
	assert_eq!(baked.len(), 3);
	assert!((baked[0] - 0.0).abs() <= f32::EPSILON * 10.0);
	assert!((baked[1] - 5.0).abs() <= f32::EPSILON * 10.0);
	assert!((baked[2] - 10.0).abs() <= f32::EPSILON * 10.0);

	assert!(oa::anim::bake(&curve, Duration::ZERO, 0.0, 3, AnimWrapMode::Clamp).is_err());
	assert!(oa::anim::bake(&curve, Duration::ZERO, 2.0, 0, AnimWrapMode::Clamp).is_err());
	assert!(
		oa::anim::bake(
			&curve,
			Duration::from_nanoseconds(i64::MAX),
			1.0,
			2,
			AnimWrapMode::Clamp,
		)
		.is_err()
	);
}

#[test]
fn curve_rejects_malformed_keys_and_modes() {
	let duplicate_times = [
		Keyframe {
			time: seconds(1),
			value: 1.0_f32,
			..Default::default()
		},
		Keyframe {
			time: seconds(1),
			value: 2.0,
			..Default::default()
		},
	];
	let non_finite = [Keyframe {
		time: Duration::ZERO,
		value: f32::NAN,
		..Default::default()
	}];
	let excessive_span = [
		Keyframe {
			time: Duration::from_nanoseconds(i64::MIN),
			value: 1.0_f32,
			..Default::default()
		},
		Keyframe {
			time: Duration::from_nanoseconds(i64::MAX),
			value: 2.0,
			..Default::default()
		},
	];
	let zero_quaternion = [Keyframe {
		time: Duration::ZERO,
		value: Quat {
			x: 0.0,
			y: 0.0,
			z: 0.0,
			w: 0.0,
		},
		..Default::default()
	}];
	assert!(oa::anim::create_curve(AnimInterpolation::Linear, &[] as &[Keyframe<f32>]).is_err());
	assert!(oa::anim::create_curve(AnimInterpolation::Linear, &duplicate_times).is_err());
	assert!(oa::anim::create_curve(AnimInterpolation::Linear, &non_finite).is_err());
	assert!(oa::anim::create_curve(AnimInterpolation::Linear, &excessive_span).is_err());
	assert!(oa::anim::create_curve(AnimInterpolation::Linear, &zero_quaternion).is_err());

	let valid = [Keyframe {
		time: Duration::ZERO,
		value: 1.0_f32,
		..Default::default()
	}];
	let curve = oa::anim::create_curve(AnimInterpolation::Linear, &valid).expect("create curve");
	assert!(curve.sample(Duration::ZERO, AnimWrapMode::Clamp).is_ok());
}
