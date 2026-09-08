//! OARS half of the matched OA VLM/GLM benchmark.

use std::{env, hint::black_box, time::Instant};

use anyhow::{Context, bail, ensure};
use oa::vlm::{Float, Mat3, Mat4, Quat, Vec2, Vec3, Vec4, Viewport};

trait BenchFloat: Float {
	const NAME: &'static str;
	fn to_f64(self) -> f64;
}

impl BenchFloat for f32 {
	const NAME: &'static str = "f32";
	fn to_f64(self) -> f64 {
		self as f64
	}
}

impl BenchFloat for f64 {
	const NAME: &'static str = "f64";
	fn to_f64(self) -> f64 {
		self
	}
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Precision {
	F32,
	F64,
	Both,
}

struct Options {
	items: usize,
	warmups: usize,
	samples: usize,
	precision: Precision,
	filter: String,
}

struct Inputs<T: BenchFloat> {
	a2: Vec<Vec2<T>>,
	b2: Vec<Vec2<T>>,
	a: Vec<Vec3<T>>,
	b: Vec<Vec3<T>>,
	a4: Vec<Vec4<T>>,
	b4: Vec<Vec4<T>>,
	normals: Vec<Vec3<T>>,
	qa: Vec<Quat<T>>,
	qb: Vec<Quat<T>>,
	ma3: Vec<Mat3<T>>,
	mb3: Vec<Mat3<T>>,
	ma4: Vec<Mat4<T>>,
	mb4: Vec<Mat4<T>>,
	projection: Mat4<T>,
	viewport: Viewport<T>,
	screen: Vec<Vec3<T>>,
}

const CASES: [(&str, &str); 38] = [
	("vec2_add", "arithmetic"),
	("vec2_dot", "arithmetic"),
	("vec2_normalize", "hardened"),
	("vec3_add", "arithmetic"),
	("vec3_component_mul", "arithmetic"),
	("vec3_dot", "arithmetic"),
	("vec3_length", "hardened"),
	("vec3_distance", "hardened"),
	("vec3_cross", "arithmetic"),
	("vec3_normalize", "hardened"),
	("vec3_reflect", "arithmetic"),
	("vec3_lerp", "arithmetic"),
	("vec3_refract", "arithmetic"),
	("vec3_project_checked", "hardened"),
	("vec3_angle_checked", "hardened"),
	("vec4_add", "arithmetic"),
	("vec4_dot", "arithmetic"),
	("vec4_normalize", "hardened"),
	("quat_mul", "arithmetic"),
	("quat_inverse_checked", "hardened"),
	("quat_normalize", "hardened"),
	("quat_rotate", "hardened"),
	("quat_nlerp", "hardened"),
	("quat_slerp", "hardened"),
	("quat_to_mat4", "hardened"),
	("mat3_vec3", "arithmetic"),
	("mat3_add", "arithmetic"),
	("mat3_transpose", "arithmetic"),
	("mat3_mul", "arithmetic"),
	("mat3_determinant", "arithmetic"),
	("mat3_inverse_checked", "hardened"),
	("mat4_vec4", "arithmetic"),
	("mat4_add", "arithmetic"),
	("mat4_transpose", "arithmetic"),
	("mat4_mul", "arithmetic"),
	("mat4_determinant", "arithmetic"),
	("mat4_inverse_checked", "hardened"),
	("transform_point", "arithmetic"),
];

const EXTRA_CASES: [(&str, &str); 11] = [
	("transform_direction", "arithmetic"),
	("affine_inverse_checked", "hardened"),
	("normal_matrix_checked", "hardened"),
	("compose_trs", "arithmetic"),
	("translation_matrix", "arithmetic"),
	("scale_matrix", "arithmetic"),
	("decompose_trs_checked", "hardened"),
	("look_at_checked", "hardened"),
	("perspective_checked", "hardened"),
	("project_point_checked", "hardened"),
	("viewport_project_checked", "hardened"),
];

fn main() -> anyhow::Result<()> {
	let options = parse_options()?;
	println!(
		"OARS VLM matched benchmark: items={} warmups={} samples={} oracle=finite",
		options.items, options.warmups, options.samples
	);
	if matches!(options.precision, Precision::F32 | Precision::Both) {
		run_suite::<f32>(&options)?;
	}
	if matches!(options.precision, Precision::F64 | Precision::Both) {
		run_suite::<f64>(&options)?;
	}
	println!("BENCHMARK oracle=PASS");
	Ok(())
}

fn run_suite<T: BenchFloat>(options: &Options) -> anyhow::Result<()> {
	let inputs: Inputs<T> = Inputs::new(options.items)?;
	let mut count = 0;
	for &(name, contract) in CASES
		.iter()
		.chain(&EXTRA_CASES)
		.chain(&[("viewport_unproject_checked", "hardened")])
	{
		if !options.filter.is_empty() && !name.contains(&options.filter) {
			continue;
		}
		let oracle = evaluate(name, &inputs)?;
		ensure!(oracle.is_finite(), "non-finite oracle for {name}");
		for _ in 0..options.warmups {
			black_box(evaluate(name, &inputs)?);
		}
		let mut times = Vec::with_capacity(options.samples);
		for _ in 0..options.samples {
			let started = Instant::now();
			black_box(evaluate(name, &inputs)?);
			times.push(started.elapsed().as_secs_f64() * 1.0e9 / options.items as f64);
		}
		times.sort_by(f64::total_cmp);
		println!(
			"CASE precision={} case={} contract={} oars_ns={:.6} oars_min={:.6} oars_max={:.6} checksum={:.9}",
			T::NAME,
			name,
			contract,
			percentile(&times, 0.5),
			times[0],
			times[times.len() - 1],
			oracle,
		);
		count += 1;
	}
	ensure!(count > 0, "filter matched no VLM case");
	println!("SUMMARY precision={} cases={} oracle=PASS", T::NAME, count);
	Ok(())
}

impl<T: BenchFloat> Inputs<T> {
	fn new(items: usize) -> anyhow::Result<Self> {
		let mut a2 = Vec::with_capacity(items);
		let mut b2 = Vec::with_capacity(items);
		let mut a = Vec::with_capacity(items);
		let mut b = Vec::with_capacity(items);
		let mut a4 = Vec::with_capacity(items);
		let mut b4 = Vec::with_capacity(items);
		let mut normals = Vec::with_capacity(items);
		let mut qa = Vec::with_capacity(items);
		let mut qb = Vec::with_capacity(items);
		let mut ma3 = Vec::with_capacity(items);
		let mut mb3 = Vec::with_capacity(items);
		let mut ma4 = Vec::with_capacity(items);
		let mut mb4 = Vec::with_capacity(items);
		for index in 0..items {
			let value = T::from_f64((index % 1021) as f64 / 1021.0);
			let av2 = Vec2 {
				x: value + t::<T>(0.125),
				y: value * t::<T>(0.5) - t::<T>(0.25),
			};
			let bv2 = Vec2 {
				x: t::<T>(0.75) - value * t::<T>(0.25),
				y: value + t::<T>(0.375),
			};
			let av = Vec3 {
				x: value + t::<T>(0.125),
				y: value * t::<T>(0.5) - t::<T>(0.25),
				z: T::ONE - value * t::<T>(0.75),
			};
			let bv = Vec3 {
				x: t::<T>(0.75) - value * t::<T>(0.25),
				y: value + t::<T>(0.375),
				z: value * t::<T>(0.2) - t::<T>(0.4),
			};
			let normal = Vec3 {
				x: value + t::<T>(0.25),
				y: t::<T>(1.25) - value,
				z: value * t::<T>(0.3) + t::<T>(0.1),
			}
			.try_normalized()
			.context("normal construction failed")?;
			let qav = Quat::try_from_axis_angle(
				Vec3 {
					x: T::ONE,
					y: t::<T>(0.5) + value,
					z: t::<T>(0.25),
				},
				t::<T>(0.1) + value * t::<T>(0.7),
			)
			.context("quaternion A construction failed")?;
			let qbv = Quat::try_from_axis_angle(
				Vec3 {
					x: t::<T>(0.25),
					y: T::ONE,
					z: t::<T>(0.75) - value * t::<T>(0.2),
				},
				t::<T>(0.2) + value * t::<T>(0.5),
			)
			.context("quaternion B construction failed")?;
			let mav = Mat4::try_compose_trs(
				Vec3 {
					x: value * t::<T>(3.0),
					y: value * t::<T>(-2.0),
					z: value + t::<T>(0.5),
				},
				qav,
				Vec3 {
					x: t::<T>(0.75) + value,
					y: t::<T>(1.25),
					z: t::<T>(1.5) - value * t::<T>(0.25),
				},
			)
			.context("matrix A construction failed")?;
			let mbv = Mat4::try_compose_trs(
				Vec3 {
					x: t::<T>(-1.0) - value,
					y: value * t::<T>(0.5),
					z: t::<T>(2.0) - value,
				},
				qbv,
				Vec3 {
					x: t::<T>(1.1),
					y: t::<T>(0.8) + value * t::<T>(0.2),
					z: t::<T>(1.3),
				},
			)
			.context("matrix B construction failed")?;
			a2.push(av2);
			b2.push(bv2);
			a.push(av);
			b.push(bv);
			a4.push(Vec4 {
				x: av.x,
				y: av.y,
				z: av.z,
				w: value + t::<T>(0.5),
			});
			b4.push(Vec4 {
				x: bv.x,
				y: bv.y,
				z: bv.z,
				w: t::<T>(1.25) - value,
			});
			normals.push(normal);
			qa.push(qav);
			qb.push(qbv);
			ma3.push(mav.linear_part());
			mb3.push(mbv.linear_part());
			ma4.push(mav);
			mb4.push(mbv);
		}
		let projection = Mat4::try_perspective(
			t::<T>(67.0),
			t::<T>(16.0) / t::<T>(9.0),
			t::<T>(0.125),
			t::<T>(4096.0),
		)
		.context("projection construction failed")?;
		let viewport = Viewport {
			x: t::<T>(13.0),
			y: t::<T>(17.0),
			width: t::<T>(1921.0),
			height: t::<T>(1081.0),
			min_depth: T::ZERO,
			max_depth: T::ONE,
		};
		let screen = a
			.iter()
			.map(|value| {
				viewport.try_project(
					Vec3 {
						x: value.x,
						y: value.y,
						z: -value.z - T::ONE,
					},
					projection,
					T::INVERSE_TOLERANCE,
				)
			})
			.collect::<Option<Vec<_>>>()
			.context("viewport fixture construction failed")?;
		Ok(Self {
			a2,
			b2,
			a,
			b,
			a4,
			b4,
			normals,
			qa,
			qb,
			ma3,
			mb3,
			ma4,
			mb4,
			projection,
			viewport,
			screen,
		})
	}
}

fn evaluate<T: BenchFloat>(name: &str, input: &Inputs<T>) -> anyhow::Result<f64> {
	let items = input.a.len();
	let value = match name {
		"vec2_add" => sum_vec2(items, |i| input.a2[i] + input.b2[i]),
		"vec2_dot" => sum_scalar(items, |i| input.a2[i].dot(input.b2[i])),
		"vec2_normalize" => sum_vec2(items, |i| input.a2[i].try_normalized().unwrap_or_default()),
		"vec3_add" => sum_vec3(items, |i| input.a[i] + input.b[i]),
		"vec3_component_mul" => sum_vec3(items, |i| input.a[i].component_mul(input.b[i])),
		"vec3_dot" => sum_scalar(items, |i| input.a[i].dot(input.b[i])),
		"vec3_length" => sum_scalar(items, |i| input.a[i].length()),
		"vec3_distance" => sum_scalar(items, |i| input.a[i].distance(input.b[i])),
		"vec3_cross" => sum_vec3(items, |i| input.a[i].cross(input.b[i])),
		"vec3_normalize" => sum_vec3(items, |i| input.a[i].try_normalized().unwrap_or_default()),
		"vec3_reflect" => sum_vec3(items, |i| input.a[i].reflect(input.normals[i])),
		"vec3_lerp" => sum_vec3(items, |i| input.a[i].lerp(input.b[i], t::<T>(0.37))),
		"vec3_refract" => sum_vec3(items, |i| {
			input.normals[i].refract(input.normals[i], t::<T>(0.75))
		}),
		"vec3_project_checked" => sum_vec3(items, |i| {
			input.a[i]
				.try_project_onto(input.b[i], T::TOLERANCE)
				.unwrap_or_default()
		}),
		"vec3_angle_checked" => sum_scalar(items, |i| {
			input.a[i]
				.try_angle_between(input.b[i], T::TOLERANCE)
				.unwrap_or(T::ZERO)
		}),
		"vec4_add" => sum_vec4(items, |i| input.a4[i] + input.b4[i]),
		"vec4_dot" => sum_scalar(items, |i| input.a4[i].dot(input.b4[i])),
		"vec4_normalize" => sum_vec4(items, |i| input.a4[i].try_normalized().unwrap_or_default()),
		"quat_mul" => sum_quat(items, |i| input.qa[i] * input.qb[i]),
		"quat_inverse_checked" => sum_quat(items, |i| {
			input.qa[i]
				.try_inverse(T::INVERSE_TOLERANCE)
				.unwrap_or_default()
		}),
		"quat_normalize" => sum_quat(items, |i| input.qa[i].try_normalized().unwrap_or_default()),
		"quat_rotate" => sum_vec3(items, |i| {
			input.qa[i].try_rotate(input.a[i]).unwrap_or_default()
		}),
		"quat_nlerp" => sum_quat(items, |i| {
			input.qa[i]
				.try_nlerp(input.qb[i], t::<T>(0.37))
				.unwrap_or_default()
		}),
		"quat_slerp" => sum_quat(items, |i| {
			input.qa[i]
				.try_slerp(input.qb[i], t::<T>(0.37))
				.unwrap_or_default()
		}),
		"quat_to_mat4" => sum_mat4(items, |i| {
			Mat4::try_from_quaternion(input.qa[i]).unwrap_or_default()
		}),
		"mat3_vec3" => sum_vec3(items, |i| input.a[i] * input.ma3[i]),
		"mat3_add" => sum_mat3(items, |i| input.ma3[i] + input.mb3[i]),
		"mat3_transpose" => sum_mat3(items, |i| input.ma3[i].transpose()),
		"mat3_mul" => sum_mat3(items, |i| input.ma3[i] * input.mb3[i]),
		"mat3_determinant" => sum_scalar(items, |i| input.ma3[i].determinant()),
		"mat3_inverse_checked" => sum_mat3(items, |i| {
			input.ma3[i]
				.try_inverse(T::INVERSE_TOLERANCE)
				.unwrap_or_default()
		}),
		"mat4_vec4" => sum_vec3(items, |i| {
			let result = input.ma4[i].transform(Vec4 {
				x: input.a[i].x,
				y: input.a[i].y,
				z: input.a[i].z,
				w: T::ONE,
			});
			Vec3 {
				x: result.x,
				y: result.y,
				z: result.z,
			}
		}),
		"mat4_add" => sum_mat4(items, |i| input.ma4[i] + input.mb4[i]),
		"mat4_transpose" => sum_mat4(items, |i| input.ma4[i].transpose()),
		"mat4_mul" => sum_mat4(items, |i| input.ma4[i] * input.mb4[i]),
		"mat4_determinant" => sum_scalar(items, |i| input.ma4[i].determinant()),
		"mat4_inverse_checked" => sum_mat4(items, |i| {
			input.ma4[i]
				.try_inverse(T::INVERSE_TOLERANCE)
				.unwrap_or_default()
		}),
		"transform_point" => sum_vec3(items, |i| input.ma4[i].transform_point(input.a[i])),
		"transform_direction" => sum_vec3(items, |i| input.ma4[i].transform_direction(input.a[i])),
		"affine_inverse_checked" => sum_mat4(items, |i| {
			input.ma4[i]
				.try_affine_inverse(T::INVERSE_TOLERANCE)
				.unwrap_or_default()
		}),
		"normal_matrix_checked" => sum_mat3(items, |i| {
			input.ma4[i]
				.try_normal_matrix(T::INVERSE_TOLERANCE)
				.unwrap_or_default()
		}),
		"compose_trs" => sum_mat4(items, |i| {
			Mat4::try_compose_trs(
				input.a[i],
				input.qa[i],
				input.b[i]
					+ Vec3 {
						x: t::<T>(1.5),
						y: t::<T>(1.5),
						z: t::<T>(1.5),
					},
			)
			.unwrap_or_default()
		}),
		"translation_matrix" => sum_mat4(items, |i| Mat4::translation(input.a[i])),
		"scale_matrix" => sum_mat4(items, |i| Mat4::scaling(input.a[i])),
		"decompose_trs_checked" => sum_vec3(items, |i| {
			input.ma4[i]
				.try_decompose_trs(T::INVERSE_TOLERANCE)
				.map(|value| value.translation + value.scale)
				.unwrap_or_default()
		}),
		"look_at_checked" => sum_mat4(items, |i| {
			Mat4::try_look_at(
				input.a[i]
					+ Vec3 {
						x: T::ZERO,
						y: T::ZERO,
						z: t::<T>(3.0),
					},
				input.b[i],
				Vec3 {
					x: T::ZERO,
					y: T::ONE,
					z: T::ZERO,
				},
				T::INVERSE_TOLERANCE,
			)
			.unwrap_or_default()
		}),
		"perspective_checked" => sum_mat4(items, |i| {
			Mat4::try_perspective(
				t::<T>(55.0) + T::from_f64((i % 17) as f64) * t::<T>(0.25),
				t::<T>(16.0) / t::<T>(9.0),
				t::<T>(0.1),
				t::<T>(1000.0),
			)
			.unwrap_or_default()
		}),
		"project_point_checked" => sum_vec3(items, |i| {
			input
				.projection
				.try_project_point(
					Vec3 {
						x: input.a[i].x,
						y: input.a[i].y,
						z: -input.a[i].z - T::ONE,
					},
					T::INVERSE_TOLERANCE,
				)
				.unwrap_or_default()
		}),
		"viewport_project_checked" => sum_vec3(items, |i| {
			input
				.viewport
				.try_project(
					Vec3 {
						x: input.a[i].x,
						y: input.a[i].y,
						z: -input.a[i].z - T::ONE,
					},
					input.projection,
					T::INVERSE_TOLERANCE,
				)
				.unwrap_or_default()
		}),
		"viewport_unproject_checked" => sum_vec3(items, |i| {
			input
				.viewport
				.try_unproject(input.screen[i], input.projection, T::INVERSE_TOLERANCE)
				.unwrap_or_default()
		}),
		_ => bail!("unknown VLM benchmark case: {name}"),
	};
	Ok(value)
}

fn t<T: BenchFloat>(value: f64) -> T {
	T::from_f64(value)
}

fn sum_scalar<T: BenchFloat>(items: usize, operation: impl Fn(usize) -> T) -> f64 {
	let mut total = T::ZERO;
	for index in 0..items {
		total = total + operation(index);
	}
	total.to_f64()
}

fn sum_vec2<T: BenchFloat>(items: usize, operation: impl Fn(usize) -> Vec2<T>) -> f64 {
	sum_scalar(items, |index| {
		let value = operation(index);
		value.x + value.y * t::<T>(0.5)
	})
}

fn sum_vec3<T: BenchFloat>(items: usize, operation: impl Fn(usize) -> Vec3<T>) -> f64 {
	sum_scalar(items, |index| {
		let value = operation(index);
		value.x + value.y * t::<T>(0.5) + value.z * t::<T>(0.25)
	})
}

fn sum_vec4<T: BenchFloat>(items: usize, operation: impl Fn(usize) -> Vec4<T>) -> f64 {
	sum_scalar(items, |index| {
		let value = operation(index);
		value.x + value.y * t::<T>(0.5) + value.z * t::<T>(0.25) + value.w * t::<T>(0.125)
	})
}

fn sum_quat<T: BenchFloat>(items: usize, operation: impl Fn(usize) -> Quat<T>) -> f64 {
	sum_scalar(items, |index| {
		let value = operation(index);
		value.x + value.y * t::<T>(0.5) + value.z * t::<T>(0.25) + value.w * t::<T>(0.125)
	})
}

fn sum_mat3<T: BenchFloat>(items: usize, operation: impl Fn(usize) -> Mat3<T>) -> f64 {
	sum_scalar(items, |index| {
		operation(index)
			.m
			.iter()
			.flatten()
			.copied()
			.fold(T::ZERO, |a, b| a + b)
	})
}

fn sum_mat4<T: BenchFloat>(items: usize, operation: impl Fn(usize) -> Mat4<T>) -> f64 {
	sum_scalar(items, |index| {
		operation(index)
			.m
			.iter()
			.flatten()
			.copied()
			.fold(T::ZERO, |a, b| a + b)
	})
}

fn percentile(values: &[f64], fraction: f64) -> f64 {
	let position = fraction * (values.len() - 1) as f64;
	let low = position.floor() as usize;
	let high = position.ceil() as usize;
	let weight = position - low as f64;
	values[low] * (1.0 - weight) + values[high] * weight
}

fn parse_options() -> anyhow::Result<Options> {
	let mut options = Options {
		items: 1 << 16,
		warmups: 3,
		samples: 11,
		precision: Precision::Both,
		filter: String::new(),
	};
	let mut arguments = env::args().skip(1);
	while let Some(argument) = arguments.next() {
		match argument.as_str() {
			"--items" => options.items = parse_positive("--items", arguments.next())?,
			"--warmups" => options.warmups = parse_positive("--warmups", arguments.next())?,
			"--samples" => options.samples = parse_positive("--samples", arguments.next())?,
			"--filter" => options.filter = arguments.next().context("--filter requires a value")?,
			"--precision" => {
				options.precision = match arguments.next().as_deref() {
					Some("f32") => Precision::F32,
					Some("f64") => Precision::F64,
					Some("both") => Precision::Both,
					_ => bail!("--precision must be f32, f64, or both"),
				};
			}
			"--help" | "-h" => {
				println!(
					"usage: core_vlm_bench [--items N] [--warmups N] [--samples N] [--precision f32|f64|both] [--filter substring]"
				);
				std::process::exit(0);
			}
			_ => bail!("unknown argument: {argument}"),
		}
	}
	ensure!(options.samples >= 3, "--samples must be at least three");
	Ok(options)
}

fn parse_positive(flag: &str, value: Option<String>) -> anyhow::Result<usize> {
	let value = value.with_context(|| format!("{flag} requires a positive integer"))?;
	let parsed = value
		.parse()
		.with_context(|| format!("{flag} requires a positive integer"))?;
	ensure!(parsed > 0, "{flag} requires a positive integer");
	Ok(parsed)
}
