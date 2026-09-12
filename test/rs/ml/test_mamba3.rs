use oa::ml::matrix::{SsmBackward, SsmConfig};

#[derive(Clone)]
struct HostInputs {
	c: Vec<f32>,
	b: Vec<f32>,
	x: Vec<f32>,
	z: Vec<f32>,
	adt: Vec<f32>,
	dt: Vec<f32>,
	trap: Vec<f32>,
	angle: Vec<f32>,
	c_bias: Vec<f32>,
	b_bias: Vec<f32>,
	d: Vec<f32>,
}

impl HostInputs {
	fn deterministic(config: SsmConfig) -> Self {
		let groups = if config.num_groups == 0 {
			config.num_heads
		} else {
			config.num_groups
		};
		let fill = |count: usize, offset: usize| {
			(0..count)
				.map(|index| ((0.7 * (index + offset) as f32 + 1.3).sin()) * 0.5)
				.collect::<Vec<_>>()
		};
		let qk_count = config.batch_size * config.sequence_length * groups * config.state_size;
		let vector_count =
			config.batch_size * config.sequence_length * config.num_heads * config.head_dim;
		let scalar_count = config.batch_size * config.sequence_length * config.num_heads;
		let angle_count = config.batch_size * config.sequence_length * config.num_rope_angles;
		let mut dt = fill(scalar_count, 0);
		let mut adt = fill(scalar_count, 3);
		for index in 0..scalar_count {
			dt[index] = 0.05 + 0.04 * (0.5 + dt[index]);
			adt[index] = -(0.3 + 0.2 * (0.5 + adt[index])) * dt[index];
		}
		Self {
			c: fill(qk_count, 0),
			b: fill(qk_count, 11),
			x: fill(vector_count, 5),
			z: fill(vector_count, 23),
			adt,
			dt,
			trap: fill(scalar_count, 7),
			angle: fill(angle_count, 31),
			c_bias: fill(config.num_heads * config.state_size, 2)
				.into_iter()
				.map(|value| 0.1 * value)
				.collect(),
			b_bias: fill(config.num_heads * config.state_size, 4)
				.into_iter()
				.map(|value| 0.1 * value)
				.collect(),
			d: (0..config.num_heads)
				.map(|head| 0.5 + 0.2 * head as f32)
				.collect(),
		}
	}

	fn upload(&self, engine: &oa::Engine, config: SsmConfig) -> oa::Result<DeviceInputs> {
		let groups = if config.num_groups == 0 {
			config.num_heads
		} else {
			config.num_groups
		};
		Ok(DeviceInputs {
			c: oa::Matrix::from_f32(
				engine,
				[
					config.batch_size,
					config.sequence_length,
					groups,
					config.state_size,
				],
				&self.c,
			)?,
			b: oa::Matrix::from_f32(
				engine,
				[
					config.batch_size,
					config.sequence_length,
					groups,
					config.state_size,
				],
				&self.b,
			)?,
			x: oa::Matrix::from_f32(
				engine,
				[
					config.batch_size,
					config.sequence_length,
					config.num_heads,
					config.head_dim,
				],
				&self.x,
			)?,
			z: oa::Matrix::from_f32(
				engine,
				[
					config.batch_size,
					config.sequence_length,
					config.num_heads,
					config.head_dim,
				],
				&self.z,
			)?,
			adt: oa::Matrix::from_f32(
				engine,
				[config.batch_size, config.sequence_length, config.num_heads],
				&self.adt,
			)?,
			dt: oa::Matrix::from_f32(
				engine,
				[config.batch_size, config.sequence_length, config.num_heads],
				&self.dt,
			)?,
			trap: oa::Matrix::from_f32(
				engine,
				[config.batch_size, config.sequence_length, config.num_heads],
				&self.trap,
			)?,
			angle: oa::Matrix::from_f32(
				engine,
				[
					config.batch_size,
					config.sequence_length,
					config.num_rope_angles,
				],
				&self.angle,
			)?,
			c_bias: oa::Matrix::from_f32(
				engine,
				[config.num_heads, config.state_size],
				&self.c_bias,
			)?,
			b_bias: oa::Matrix::from_f32(
				engine,
				[config.num_heads, config.state_size],
				&self.b_bias,
			)?,
			d: oa::Matrix::from_f32(engine, [config.num_heads], &self.d)?,
		})
	}
}

struct DeviceInputs {
	c: oa::Matrix,
	b: oa::Matrix,
	x: oa::Matrix,
	z: oa::Matrix,
	adt: oa::Matrix,
	dt: oa::Matrix,
	trap: oa::Matrix,
	angle: oa::Matrix,
	c_bias: oa::Matrix,
	b_bias: oa::Matrix,
	d: oa::Matrix,
}

impl DeviceInputs {
	fn forward(&self, config: SsmConfig) -> oa::Result<oa::Matrix> {
		oa::ml::matrix::mamba3_siso(
			&self.c,
			&self.b,
			&self.x,
			&self.z,
			&self.adt,
			&self.dt,
			&self.trap,
			&self.angle,
			&self.c_bias,
			&self.b_bias,
			&self.d,
			config,
		)
	}

	fn backward(&self, output_gradient: &oa::Matrix, config: SsmConfig) -> oa::Result<SsmBackward> {
		oa::ml::matrix::mamba3_siso_backward(
			output_gradient,
			&self.c,
			&self.b,
			&self.x,
			&self.z,
			&self.adt,
			&self.dt,
			&self.trap,
			&self.angle,
			&self.c_bias,
			&self.b_bias,
			&self.d,
			config,
		)
	}
}

fn reference_forward(input: &HostInputs, config: SsmConfig) -> Vec<f32> {
	let groups = if config.num_groups == 0 {
		config.num_heads
	} else {
		config.num_groups
	};
	let heads_per_group = config.num_heads / groups;
	let vector_index = |batch: usize, time: usize, head: usize, feature: usize| {
		((batch * config.sequence_length + time) * config.num_heads + head) * config.head_dim
			+ feature
	};
	let scalar_index = |batch: usize, time: usize, head: usize| {
		(batch * config.sequence_length + time) * config.num_heads + head
	};
	let angle_index = |batch: usize, time: usize, angle: usize| {
		(batch * config.sequence_length + time) * config.num_rope_angles + angle
	};
	let qk_index = |batch: usize, time: usize, group: usize, state: usize| {
		((batch * config.sequence_length + time) * groups + group) * config.state_size + state
	};
	let sigmoid = |value: f32| 1.0 / (1.0 + (-value).exp());
	let mut output = vec![0.0; input.x.len()];
	for batch in 0..config.batch_size {
		for head in 0..config.num_heads {
			let group = head / heads_per_group;
			let mut theta = vec![0.0; config.num_rope_angles];
			let mut state = vec![0.0; config.head_dim * config.state_size];
			for time in 0..config.sequence_length {
				let scalar = scalar_index(batch, time, head);
				let dt = input.dt[scalar];
				let trap = sigmoid(input.trap[scalar]);
				let gamma = dt * trap;
				let shifted = if time + 1 < config.sequence_length {
					let next = scalar_index(batch, time + 1, head);
					input.dt[next] * (1.0 - sigmoid(input.trap[next]))
				} else {
					0.0
				};
				let scale = gamma + shifted;
				let decay = input.adt[scalar].exp();
				let mut c = (0..config.state_size)
					.map(|n| {
						input.c[qk_index(batch, time, group, n)]
							+ input.c_bias[head * config.state_size + n]
					})
					.collect::<Vec<_>>();
				let mut b = (0..config.state_size)
					.map(|n| {
						input.b[qk_index(batch, time, group, n)]
							+ input.b_bias[head * config.state_size + n]
					})
					.collect::<Vec<_>>();
				for (angle, theta) in theta.iter_mut().enumerate() {
					*theta += input.angle[angle_index(batch, time, angle)] * dt;
					let (sin, cos) = theta.sin_cos();
					let even = 2 * angle;
					let odd = even + 1;
					let (c0, c1, b0, b1) = (c[even], c[odd], b[even], b[odd]);
					c[even] = c0 * cos - c1 * sin;
					c[odd] = c0 * sin + c1 * cos;
					b[even] = b0 * cos - b1 * sin;
					b[odd] = b0 * sin + b1 * cos;
				}
				let qk = c.iter().zip(&b).map(|(c, b)| c * b).sum::<f32>();
				for feature in 0..config.head_dim {
					let vector = vector_index(batch, time, head, feature);
					let x = input.x[vector];
					let mut value = 0.0;
					for n in 0..config.state_size {
						let index = feature * config.state_size + n;
						let decayed = decay * state[index];
						value += c[n] * decayed;
						state[index] = decayed + scale * x * b[n];
					}
					if config.has_d {
						value += input.d[head] * x;
					}
					value += gamma * qk * x;
					if config.has_z {
						let z = input.z[vector];
						value *= z * sigmoid(z);
					}
					output[vector] = value;
				}
			}
		}
	}
	output
}

fn assert_close(actual: &[f32], expected: &[f32], tolerance: f32) {
	assert_eq!(actual.len(), expected.len());
	for (index, (actual, expected)) in actual.iter().zip(expected).enumerate() {
		assert!(
			(actual - expected).abs() <= tolerance,
			"element {index}: expected {expected}, found {actual}"
		);
	}
}

fn expand_qk_to_heads(input: &HostInputs, config: SsmConfig) -> (HostInputs, SsmConfig) {
	let groups = config.num_groups;
	assert!(groups > 0);
	let heads_per_group = config.num_heads / groups;
	let mut expanded = input.clone();
	let expanded_count =
		config.batch_size * config.sequence_length * config.num_heads * config.state_size;
	expanded.c = vec![0.0; expanded_count];
	expanded.b = vec![0.0; expanded_count];
	for batch in 0..config.batch_size {
		for time in 0..config.sequence_length {
			for head in 0..config.num_heads {
				let group = head / heads_per_group;
				for state in 0..config.state_size {
					let source = ((batch * config.sequence_length + time) * groups + group)
						* config.state_size
						+ state;
					let destination = ((batch * config.sequence_length + time) * config.num_heads
						+ head) * config.state_size
						+ state;
					expanded.c[destination] = input.c[source];
					expanded.b[destination] = input.b[source];
				}
			}
		}
	}
	let mut expanded_config = config;
	expanded_config.num_groups = 0;
	(expanded, expanded_config)
}

fn reduce_head_gradients_to_groups(values: &[f32], config: SsmConfig) -> Vec<f32> {
	let groups = config.num_groups;
	let heads_per_group = config.num_heads / groups;
	let mut reduced =
		vec![0.0; config.batch_size * config.sequence_length * groups * config.state_size];
	for batch in 0..config.batch_size {
		for time in 0..config.sequence_length {
			for head in 0..config.num_heads {
				let group = head / heads_per_group;
				for state in 0..config.state_size {
					let source = ((batch * config.sequence_length + time) * config.num_heads
						+ head) * config.state_size
						+ state;
					let destination = ((batch * config.sequence_length + time) * groups + group)
						* config.state_size
						+ state;
					reduced[destination] += values[source];
				}
			}
		}
	}
	reduced
}

fn config(batch_size: usize, sequence_length: usize, num_heads: usize) -> SsmConfig {
	SsmConfig {
		batch_size,
		sequence_length,
		num_heads,
		num_groups: 0,
		head_dim: 3,
		state_size: 4,
		num_rope_angles: 2,
		mimo_rank: 1,
		has_z: true,
		has_d: true,
		has_output_norm: false,
	}
}

test_vk!(mamba3_siso_matches_independent_donor_recurrence, engine, {
	let config = config(2, 6, 2);
	let host = HostInputs::deterministic(config);
	let expected = reference_forward(&host, config);
	let actual = host.upload(&engine, config)?.forward(config)?.read_f32()?;
	assert_close(&actual, &expected, 1.0e-4);
	Ok(())
});

test_vk!(mamba3_siso_grouped_qk_matches_expanded_heads, engine, {
	let grouped_config = SsmConfig {
		batch_size: 1,
		sequence_length: 5,
		num_heads: 4,
		num_groups: 2,
		head_dim: 3,
		state_size: 4,
		num_rope_angles: 2,
		mimo_rank: 1,
		has_z: true,
		has_d: true,
		has_output_norm: false,
	};
	let grouped = HostInputs::deterministic(grouped_config);
	let (expanded, expanded_config) = expand_qk_to_heads(&grouped, grouped_config);
	let grouped_device = grouped.upload(&engine, grouped_config)?;
	let expanded_device = expanded.upload(&engine, expanded_config)?;
	assert_close(
		&grouped_device.forward(grouped_config)?.read_f32()?,
		&expanded_device.forward(expanded_config)?.read_f32()?,
		2.0e-4,
	);

	let upstream = oa::Matrix::from_f32(
		&engine,
		[
			grouped_config.batch_size,
			grouped_config.sequence_length,
			grouped_config.num_heads,
			grouped_config.head_dim,
		],
		&grouped.x,
	)?;
	let grouped_gradient = grouped_device.backward(&upstream, grouped_config)?;
	let expanded_gradient = expanded_device.backward(&upstream, expanded_config)?;
	let expanded_c = expanded_gradient.c.read_f32()?;
	let expanded_b = expanded_gradient.b.read_f32()?;
	assert_close(
		&grouped_gradient.c.read_f32()?,
		&reduce_head_gradients_to_groups(&expanded_c, grouped_config),
		3.0e-4,
	);
	assert_close(
		&grouped_gradient.b.read_f32()?,
		&reduce_head_gradients_to_groups(&expanded_b, grouped_config),
		3.0e-4,
	);
	for (actual, expected) in [
		(
			grouped_gradient.x.read_f32()?,
			expanded_gradient.x.read_f32()?,
		),
		(
			grouped_gradient.z.read_f32()?,
			expanded_gradient.z.read_f32()?,
		),
		(
			grouped_gradient.adt.read_f32()?,
			expanded_gradient.adt.read_f32()?,
		),
		(
			grouped_gradient.dt.read_f32()?,
			expanded_gradient.dt.read_f32()?,
		),
		(
			grouped_gradient.trap.read_f32()?,
			expanded_gradient.trap.read_f32()?,
		),
		(
			grouped_gradient.angle.read_f32()?,
			expanded_gradient.angle.read_f32()?,
		),
		(
			grouped_gradient.c_bias.read_f32()?,
			expanded_gradient.c_bias.read_f32()?,
		),
		(
			grouped_gradient.b_bias.read_f32()?,
			expanded_gradient.b_bias.read_f32()?,
		),
		(
			grouped_gradient.d.read_f32()?,
			expanded_gradient.d.read_f32()?,
		),
	] {
		assert_close(&actual, &expected, 3.0e-4);
	}
	Ok(())
});

test_vk!(mamba3_siso_tape_routes_the_complete_x_adjoint, engine, {
	let config = config(1, 3, 2);
	let host = HostInputs::deterministic(config);
	let device = host.upload(&engine, config)?;
	let upstream =
		oa::Matrix::from_f32(&engine, device.x.shape().to_vec(), &vec![1.0; host.x.len()])?;
	let expected = device.backward(&upstream, config)?.x.read_f32()?;

	let rows = config.batch_size * config.sequence_length * config.num_heads;
	let embedding = oa::ml::nn::Embedding::from_matrix(oa::Matrix::from_f32(
		&engine,
		[rows, config.head_dim],
		&host.x,
	)?)?;
	let indices = oa::Matrix::from_slice(&engine, [rows], &(0..rows as u32).collect::<Vec<_>>())?;
	let tape = oa::ml::GradientTape::new();
	let x = embedding.forward(&indices)?.reshape([
		config.batch_size,
		config.sequence_length,
		config.num_heads,
		config.head_dim,
	])?;
	let output = oa::ml::matrix::mamba3_siso(
		&device.c,
		&device.b,
		&x,
		&device.z,
		&device.adt,
		&device.dt,
		&device.trap,
		&device.angle,
		&device.c_bias,
		&device.b_bias,
		&device.d,
		config,
	)?;
	let loss = oa::matrix::sum(&output.reshape([host.x.len()])?, 0)?.reshape([])?;
	tape.backward(&loss)?;
	assert_close(
		&embedding
			.weight()
			.gradient()
			.expect("Mamba-3 x predecessor gradient is missing")
			.read_f32()?,
		&expected,
		3.0e-4,
	);
	Ok(())
});

test_vk!(
	mamba3_siso_short_backward_matches_finite_differences,
	engine,
	{
		let config = SsmConfig {
			batch_size: 1,
			sequence_length: 3,
			num_heads: 1,
			num_groups: 0,
			head_dim: 2,
			state_size: 3,
			num_rope_angles: 1,
			mimo_rank: 1,
			has_z: true,
			has_d: true,
			has_output_norm: false,
		};
		let mut host = HostInputs::deterministic(config);
		let upstream = HostInputs::deterministic(config).x;
		let device = host.upload(&engine, config)?;
		let upstream_matrix = oa::Matrix::from_f32(
			&engine,
			[
				config.batch_size,
				config.sequence_length,
				config.num_heads,
				config.head_dim,
			],
			&upstream,
		)?;
		let analytic = device.backward(&upstream_matrix, config)?;
		let analytic = [
			analytic.c.read_f32()?,
			analytic.b.read_f32()?,
			analytic.x.read_f32()?,
			analytic.z.read_f32()?,
			analytic.adt.read_f32()?,
			analytic.dt.read_f32()?,
			analytic.trap.read_f32()?,
			analytic.angle.read_f32()?,
			analytic.c_bias.read_f32()?,
			analytic.b_bias.read_f32()?,
			analytic.d.read_f32()?,
		];
		let epsilon = 2.0e-3_f32;
		for (family, exact_values) in analytic.iter().enumerate() {
			let length = match family {
				0 => host.c.len(),
				1 => host.b.len(),
				2 => host.x.len(),
				3 => host.z.len(),
				4 => host.adt.len(),
				5 => host.dt.len(),
				6 => host.trap.len(),
				7 => host.angle.len(),
				8 => host.c_bias.len(),
				9 => host.b_bias.len(),
				_ => host.d.len(),
			};
			for index in 0..length {
				let values = match family {
					0 => &mut host.c,
					1 => &mut host.b,
					2 => &mut host.x,
					3 => &mut host.z,
					4 => &mut host.adt,
					5 => &mut host.dt,
					6 => &mut host.trap,
					7 => &mut host.angle,
					8 => &mut host.c_bias,
					9 => &mut host.b_bias,
					_ => &mut host.d,
				};
				let original = values[index];
				values[index] = original + epsilon;
				let plus = reference_forward(&host, config)
					.iter()
					.zip(&upstream)
					.map(|(value, gradient)| value * gradient)
					.sum::<f32>();
				let values = match family {
					0 => &mut host.c,
					1 => &mut host.b,
					2 => &mut host.x,
					3 => &mut host.z,
					4 => &mut host.adt,
					5 => &mut host.dt,
					6 => &mut host.trap,
					7 => &mut host.angle,
					8 => &mut host.c_bias,
					9 => &mut host.b_bias,
					_ => &mut host.d,
				};
				values[index] = original - epsilon;
				let minus = reference_forward(&host, config)
					.iter()
					.zip(&upstream)
					.map(|(value, gradient)| value * gradient)
					.sum::<f32>();
				let values = match family {
					0 => &mut host.c,
					1 => &mut host.b,
					2 => &mut host.x,
					3 => &mut host.z,
					4 => &mut host.adt,
					5 => &mut host.dt,
					6 => &mut host.trap,
					7 => &mut host.angle,
					8 => &mut host.c_bias,
					9 => &mut host.b_bias,
					_ => &mut host.d,
				};
				values[index] = original;
				let numerical = (plus - minus) / (2.0 * epsilon);
				let exact = exact_values[index];
				let tolerance = 2.0e-2 + 3.0e-2 * exact.abs();
				assert!(
					(numerical - exact).abs() <= tolerance,
					"gradient family {family} element {index}: numerical {numerical}, analytic {exact}"
				);
			}
		}
		Ok(())
	}
);
