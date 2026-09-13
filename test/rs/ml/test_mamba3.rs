use oa::ml::matrix::{Mamba3PreprocessConfig, Mamba3PreprocessResult, SsmBackward, SsmConfig};

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

	fn token(&self, time: usize, config: SsmConfig) -> Self {
		let groups = if config.num_groups == 0 {
			config.num_heads
		} else {
			config.num_groups
		};
		let gather = |values: &[f32], trailing: usize| {
			let mut token = Vec::with_capacity(config.batch_size * trailing);
			for batch in 0..config.batch_size {
				let begin = (batch * config.sequence_length + time) * trailing;
				token.extend_from_slice(&values[begin..begin + trailing]);
			}
			token
		};
		Self {
			c: gather(&self.c, groups * config.state_size),
			b: gather(&self.b, groups * config.state_size),
			x: gather(&self.x, config.num_heads * config.head_dim),
			z: gather(&self.z, config.num_heads * config.head_dim),
			adt: gather(&self.adt, config.num_heads),
			dt: gather(&self.dt, config.num_heads),
			trap: gather(&self.trap, config.num_heads),
			angle: gather(&self.angle, config.num_rope_angles),
			c_bias: self.c_bias.clone(),
			b_bias: self.b_bias.clone(),
			d: self.d.clone(),
		}
	}

	fn family_mut(&mut self, family: usize) -> &mut Vec<f32> {
		match family {
			0 => &mut self.c,
			1 => &mut self.b,
			2 => &mut self.x,
			3 => &mut self.z,
			4 => &mut self.adt,
			5 => &mut self.dt,
			6 => &mut self.trap,
			7 => &mut self.angle,
			8 => &mut self.c_bias,
			9 => &mut self.b_bias,
			10 => &mut self.d,
			_ => panic!("unknown SISO gradient family {family}"),
		}
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

fn gradcheck_siso(engine: &oa::Engine, config: SsmConfig, tolerance: f32) -> oa::Result<()> {
	let mut host = HostInputs::deterministic(config);
	let upstream = HostInputs::deterministic(config).x;
	let device = host.upload(engine, config)?;
	let upstream_matrix = oa::Matrix::from_f32(
		engine,
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
		let family_len = host.family_mut(family).len();
		for (index, exact) in exact_values.iter().copied().enumerate().take(family_len) {
			let original = host.family_mut(family)[index];
			host.family_mut(family)[index] = original + epsilon;
			let plus = reference_forward(&host, config)
				.iter()
				.zip(&upstream)
				.map(|(value, gradient)| value * gradient)
				.sum::<f32>();
			host.family_mut(family)[index] = original - epsilon;
			let minus = reference_forward(&host, config)
				.iter()
				.zip(&upstream)
				.map(|(value, gradient)| value * gradient)
				.sum::<f32>();
			host.family_mut(family)[index] = original;
			let numerical = (plus - minus) / (2.0 * epsilon);
			let allowed = tolerance + 5.0e-2 * exact.abs();
			assert!(
				(numerical - exact).abs() <= allowed,
				"gradient family {family} element {index}: numerical {numerical}, analytic {exact}"
			);
		}
	}
	Ok(())
}

#[derive(Clone)]
struct MimoHostInputs {
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
	mimo_x: Vec<f32>,
	mimo_z: Vec<f32>,
	mimo_o: Vec<f32>,
	norm_weight: Vec<f32>,
}

impl MimoHostInputs {
	fn deterministic(config: SsmConfig) -> Self {
		let groups = if config.num_groups == 0 {
			config.num_heads
		} else {
			config.num_groups
		};
		let fill = |count: usize, offset: usize| {
			(0..count)
				.map(|index| 0.35 * (0.61 * (index + offset) as f32 + 0.9).sin())
				.collect::<Vec<_>>()
		};
		let qk_count = config.batch_size
			* config.sequence_length
			* config.mimo_rank
			* groups * config.state_size;
		let vector_count =
			config.batch_size * config.sequence_length * config.num_heads * config.head_dim;
		let scalar_count = config.batch_size * config.sequence_length * config.num_heads;
		let mut dt = fill(scalar_count, 3);
		let mut adt = fill(scalar_count, 9);
		for index in 0..scalar_count {
			dt[index] = 0.06 + 0.03 * (0.5 + dt[index]);
			adt[index] = -(0.4 + 0.2 * (0.5 + adt[index])) * dt[index];
		}
		let bias_count = config.num_heads * config.mimo_rank * config.state_size;
		let projection_count = config.num_heads * config.mimo_rank * config.head_dim;
		Self {
			c: fill(qk_count, 1),
			b: fill(qk_count, 13),
			x: fill(vector_count, 5),
			z: fill(vector_count, 17),
			adt,
			dt,
			trap: fill(scalar_count, 21),
			angle: fill(
				config.batch_size * config.sequence_length * config.num_rope_angles,
				29,
			),
			c_bias: fill(bias_count, 7)
				.into_iter()
				.map(|value| value * 0.1)
				.collect(),
			b_bias: fill(bias_count, 11)
				.into_iter()
				.map(|value| value * 0.1)
				.collect(),
			d: fill(config.num_heads, 4)
				.into_iter()
				.map(|value| 0.4 + value)
				.collect(),
			mimo_x: fill(projection_count, 2)
				.into_iter()
				.map(|value| 0.8 + value)
				.collect(),
			mimo_z: fill(projection_count, 19)
				.into_iter()
				.map(|value| 0.9 + value)
				.collect(),
			mimo_o: fill(projection_count, 23)
				.into_iter()
				.map(|value| 0.7 + value)
				.collect(),
			norm_weight: fill(config.num_heads * config.head_dim, 31)
				.into_iter()
				.map(|value| 1.0 + 0.1 * value)
				.collect(),
		}
	}

	fn upload(&self, engine: &oa::Engine, config: SsmConfig) -> oa::Result<MimoDeviceInputs> {
		let groups = if config.num_groups == 0 {
			config.num_heads
		} else {
			config.num_groups
		};
		let matrix =
			|shape: Vec<usize>, values: &[f32]| oa::Matrix::from_f32(engine, shape, values);
		Ok(MimoDeviceInputs {
			c: matrix(
				vec![
					config.batch_size,
					config.sequence_length,
					config.mimo_rank * groups,
					config.state_size,
				],
				&self.c,
			)?,
			b: matrix(
				vec![
					config.batch_size,
					config.sequence_length,
					config.mimo_rank * groups,
					config.state_size,
				],
				&self.b,
			)?,
			x: matrix(
				vec![
					config.batch_size,
					config.sequence_length,
					config.num_heads,
					config.head_dim,
				],
				&self.x,
			)?,
			z: matrix(
				vec![
					config.batch_size,
					config.sequence_length,
					config.num_heads,
					config.head_dim,
				],
				&self.z,
			)?,
			adt: matrix(
				vec![config.batch_size, config.sequence_length, config.num_heads],
				&self.adt,
			)?,
			dt: matrix(
				vec![config.batch_size, config.sequence_length, config.num_heads],
				&self.dt,
			)?,
			trap: matrix(
				vec![config.batch_size, config.sequence_length, config.num_heads],
				&self.trap,
			)?,
			angle: matrix(
				vec![
					config.batch_size,
					config.sequence_length,
					config.num_rope_angles,
				],
				&self.angle,
			)?,
			c_bias: matrix(
				vec![config.num_heads, config.mimo_rank, config.state_size],
				&self.c_bias,
			)?,
			b_bias: matrix(
				vec![config.num_heads, config.mimo_rank, config.state_size],
				&self.b_bias,
			)?,
			d: matrix(vec![config.num_heads], &self.d)?,
			mimo_x: matrix(
				vec![config.num_heads, config.mimo_rank, config.head_dim],
				&self.mimo_x,
			)?,
			mimo_z: matrix(
				vec![config.num_heads, config.mimo_rank, config.head_dim],
				&self.mimo_z,
			)?,
			mimo_o: matrix(
				vec![config.num_heads, config.mimo_rank, config.head_dim],
				&self.mimo_o,
			)?,
			norm_weight: matrix(vec![config.num_heads, config.head_dim], &self.norm_weight)?,
		})
	}

	fn token(&self, time: usize, config: SsmConfig) -> Self {
		let groups = if config.num_groups == 0 {
			config.num_heads
		} else {
			config.num_groups
		};
		let gather = |values: &[f32], trailing: usize| {
			let mut token = Vec::with_capacity(config.batch_size * trailing);
			for batch in 0..config.batch_size {
				let begin = (batch * config.sequence_length + time) * trailing;
				token.extend_from_slice(&values[begin..begin + trailing]);
			}
			token
		};
		Self {
			c: gather(&self.c, config.mimo_rank * groups * config.state_size),
			b: gather(&self.b, config.mimo_rank * groups * config.state_size),
			x: gather(&self.x, config.num_heads * config.head_dim),
			z: gather(&self.z, config.num_heads * config.head_dim),
			adt: gather(&self.adt, config.num_heads),
			dt: gather(&self.dt, config.num_heads),
			trap: gather(&self.trap, config.num_heads),
			angle: gather(&self.angle, config.num_rope_angles),
			c_bias: self.c_bias.clone(),
			b_bias: self.b_bias.clone(),
			d: self.d.clone(),
			mimo_x: self.mimo_x.clone(),
			mimo_z: self.mimo_z.clone(),
			mimo_o: self.mimo_o.clone(),
			norm_weight: self.norm_weight.clone(),
		}
	}

	fn family_mut(&mut self, family: usize) -> &mut Vec<f32> {
		match family {
			0 => &mut self.c,
			1 => &mut self.b,
			2 => &mut self.x,
			3 => &mut self.z,
			4 => &mut self.adt,
			5 => &mut self.dt,
			6 => &mut self.trap,
			7 => &mut self.angle,
			8 => &mut self.c_bias,
			9 => &mut self.b_bias,
			10 => &mut self.d,
			11 => &mut self.mimo_x,
			12 => &mut self.mimo_z,
			13 => &mut self.mimo_o,
			14 => &mut self.norm_weight,
			_ => panic!("invalid MIMO input family {family}"),
		}
	}
}

struct MimoDeviceInputs {
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
	mimo_x: oa::Matrix,
	mimo_z: oa::Matrix,
	mimo_o: oa::Matrix,
	norm_weight: oa::Matrix,
}

impl MimoDeviceInputs {
	fn forward(&self, config: SsmConfig) -> oa::Result<oa::Matrix> {
		oa::ml::matrix::mamba3_mimo(
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
			&self.mimo_x,
			&self.mimo_z,
			&self.mimo_o,
			&self.norm_weight,
			config,
		)
	}

	fn backward(
		&self,
		output_gradient: &oa::Matrix,
		config: SsmConfig,
	) -> oa::Result<oa::ml::matrix::Mamba3MimoBackward> {
		oa::ml::matrix::mamba3_mimo_backward(
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
			&self.mimo_x,
			&self.mimo_z,
			&self.mimo_o,
			&self.norm_weight,
			config,
		)
	}
}

fn reference_mimo_forward(input: &MimoHostInputs, config: SsmConfig) -> Vec<f32> {
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
	let qk_index = |batch: usize, time: usize, rank: usize, group: usize, state: usize| {
		(((batch * config.sequence_length + time) * (config.mimo_rank * groups)
			+ rank * groups
			+ group) * config.state_size)
			+ state
	};
	let parameter_index = |head: usize, rank: usize, extent: usize, index: usize| {
		(head * config.mimo_rank + rank) * extent + index
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
				for (angle, theta_value) in theta.iter_mut().enumerate() {
					*theta_value += input.angle
						[(batch * config.sequence_length + time) * config.num_rope_angles + angle]
						* dt;
				}
				let mut q = vec![0.0; config.mimo_rank * config.state_size];
				let mut k = vec![0.0; config.mimo_rank * config.state_size];
				for rank in 0..config.mimo_rank {
					for n in 0..config.state_size {
						let index = rank * config.state_size + n;
						let bias = parameter_index(head, rank, config.state_size, n);
						q[index] =
							input.c[qk_index(batch, time, rank, group, n)] + input.c_bias[bias];
						k[index] =
							input.b[qk_index(batch, time, rank, group, n)] + input.b_bias[bias];
					}
					for (angle, theta_value) in theta.iter().copied().enumerate() {
						let even = rank * config.state_size + 2 * angle;
						let odd = even + 1;
						let (sin, cos) = theta_value.sin_cos();
						let (q0, q1, k0, k1) = (q[even], q[odd], k[even], k[odd]);
						q[even] = q0 * cos - q1 * sin;
						q[odd] = q0 * sin + q1 * cos;
						k[even] = k0 * cos - k1 * sin;
						k[odd] = k0 * sin + k1 * cos;
					}
				}
				let mut pre = vec![0.0; config.mimo_rank * config.head_dim];
				for feature in 0..config.head_dim {
					let vector = vector_index(batch, time, head, feature);
					let xv = input.x[vector];
					let mut u = vec![0.0; config.state_size];
					for (n, u_value) in u.iter_mut().enumerate() {
						for rank in 0..config.mimo_rank {
							let projection = parameter_index(head, rank, config.head_dim, feature);
							*u_value +=
								xv * input.mimo_x[projection] * k[rank * config.state_size + n];
						}
					}
					for rank in 0..config.mimo_rank {
						let mut value = 0.0;
						for (n, u_value) in u.iter().copied().enumerate() {
							let state_index = feature * config.state_size + n;
							value += q[rank * config.state_size + n]
								* (decay * state[state_index] + gamma * u_value);
						}
						let projection = parameter_index(head, rank, config.head_dim, feature);
						if config.has_d {
							value += input.d[head] * xv * input.mimo_x[projection];
						}
						pre[rank * config.head_dim + feature] = value;
					}
					for (n, u_value) in u.iter().copied().enumerate() {
						let state_index = feature * config.state_size + n;
						state[state_index] = decay * state[state_index] + scale * u_value;
					}
				}
				for rank in 0..config.mimo_rank {
					let inverse_rms = if config.has_output_norm {
						let sum = (0..config.head_dim)
							.map(|feature| pre[rank * config.head_dim + feature].powi(2))
							.sum::<f32>();
						(sum / config.head_dim as f32 + 1.0e-5).sqrt().recip()
					} else {
						1.0
					};
					for feature in 0..config.head_dim {
						let vector = vector_index(batch, time, head, feature);
						let projection = parameter_index(head, rank, config.head_dim, feature);
						let mut core = pre[rank * config.head_dim + feature];
						if config.has_output_norm {
							core *=
								inverse_rms * input.norm_weight[head * config.head_dim + feature];
						}
						let gate = if config.has_z {
							let value = input.z[vector] * input.mimo_z[projection];
							value * sigmoid(value)
						} else {
							1.0
						};
						output[vector] += core * gate * input.mimo_o[projection];
					}
				}
			}
		}
	}
	output
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

test_vk!(
	mamba3_mimo_matches_independent_shared_state_recurrence,
	engine,
	{
		let config = SsmConfig {
			batch_size: 1,
			sequence_length: 3,
			num_heads: 2,
			num_groups: 1,
			head_dim: 2,
			state_size: 3,
			num_rope_angles: 1,
			mimo_rank: 2,
			has_z: true,
			has_d: true,
			has_output_norm: true,
		};
		let host = MimoHostInputs::deterministic(config);
		let expected = reference_mimo_forward(&host, config);
		let actual = host.upload(&engine, config)?.forward(config)?.read_f32()?;
		assert_close(&actual, &expected, 2.0e-4);
		Ok(())
	}
);

test_vk!(mamba3_mimo_recurrent_steps_match_full_scan, engine, {
	let config = SsmConfig {
		batch_size: 1,
		sequence_length: 5,
		num_heads: 2,
		num_groups: 1,
		head_dim: 3,
		state_size: 4,
		num_rope_angles: 2,
		mimo_rank: 2,
		has_z: true,
		has_d: true,
		has_output_norm: true,
	};
	let host = MimoHostInputs::deterministic(config);
	let expected = reference_mimo_forward(&host, config);
	let mut ssm_state = oa::matrix::full(
		&engine,
		[
			config.batch_size,
			config.num_heads,
			config.head_dim,
			config.state_size,
		],
		0.0,
	)?;
	let mut angle_state = oa::matrix::full(
		&engine,
		[config.batch_size, config.num_heads, config.num_rope_angles],
		0.0,
	)?;
	let mut k_state = oa::matrix::full(
		&engine,
		[
			config.batch_size,
			config.num_heads,
			config.mimo_rank,
			config.state_size,
		],
		0.0,
	)?;
	let mut v_state = oa::matrix::full(
		&engine,
		[
			config.batch_size,
			config.num_heads,
			config.mimo_rank,
			config.head_dim,
		],
		0.0,
	)?;
	let token_width = config.num_heads * config.head_dim;
	let mut actual = vec![0.0; expected.len()];
	for time in 0..config.sequence_length {
		let mut step_config = config;
		step_config.sequence_length = 1;
		let token = host.token(time, config).upload(&engine, step_config)?;
		let output = oa::ml::matrix::mamba3_mimo_step(
			&token.c,
			&token.b,
			&token.x,
			&token.z,
			&token.adt,
			&token.dt,
			&token.trap,
			&token.angle,
			&token.c_bias,
			&token.b_bias,
			&token.d,
			&token.mimo_x,
			&token.mimo_z,
			&token.mimo_o,
			&token.norm_weight,
			&mut ssm_state,
			&mut angle_state,
			&mut k_state,
			&mut v_state,
			step_config,
		)?;
		let values = output.read_f32()?;
		for batch in 0..config.batch_size {
			let source = batch * token_width;
			let destination = (batch * config.sequence_length + time) * token_width;
			actual[destination..destination + token_width]
				.copy_from_slice(&values[source..source + token_width]);
		}
	}
	assert_close(&actual, &expected, 2.0e-4);
	Ok(())
});

test_vk!(
	mamba3_mimo_backward_matches_all_fifteen_finite_differences,
	engine,
	{
		let config = SsmConfig {
			batch_size: 1,
			sequence_length: 2,
			num_heads: 1,
			num_groups: 1,
			head_dim: 2,
			state_size: 3,
			num_rope_angles: 1,
			mimo_rank: 2,
			has_z: true,
			has_d: true,
			has_output_norm: true,
		};
		let mut host = MimoHostInputs::deterministic(config);
		let upstream = (0..host.x.len())
			.map(|index| 0.4 * (0.73 * index as f32 + 0.2).cos())
			.collect::<Vec<_>>();
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
			analytic.mimo_x.read_f32()?,
			analytic.mimo_z.read_f32()?,
			analytic.mimo_o.read_f32()?,
			analytic.norm_weight.read_f32()?,
		];
		let epsilon = 2.0e-3_f32;
		for (family, exact_values) in analytic.iter().enumerate() {
			let family_len = host.family_mut(family).len();
			for (index, exact) in exact_values.iter().copied().enumerate().take(family_len) {
				let original = host.family_mut(family)[index];
				host.family_mut(family)[index] = original + epsilon;
				let plus = reference_mimo_forward(&host, config)
					.iter()
					.zip(&upstream)
					.map(|(value, gradient)| value * gradient)
					.sum::<f32>();
				host.family_mut(family)[index] = original - epsilon;
				let minus = reference_mimo_forward(&host, config)
					.iter()
					.zip(&upstream)
					.map(|(value, gradient)| value * gradient)
					.sum::<f32>();
				host.family_mut(family)[index] = original;
				let numerical = (plus - minus) / (2.0 * epsilon);
				let tolerance = 2.5e-2 + 4.0e-2 * exact.abs();
				assert!(
					(numerical - exact).abs() <= tolerance,
					"MIMO gradient family {family} element {index}: numerical {numerical}, analytic {exact}"
				);
			}
		}
		Ok(())
	}
);

test_vk!(mamba3_siso_recurrent_steps_match_full_scan, engine, {
	let config = SsmConfig {
		batch_size: 2,
		sequence_length: 6,
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
	let host = HostInputs::deterministic(config);
	let expected = host.upload(&engine, config)?.forward(config)?.read_f32()?;
	let mut ssm_state = oa::matrix::full(
		&engine,
		[
			config.batch_size,
			config.num_heads,
			config.head_dim,
			config.state_size,
		],
		0.0,
	)?;
	let mut angle_state = oa::matrix::full(
		&engine,
		[config.batch_size, config.num_heads, config.num_rope_angles],
		0.0,
	)?;
	let mut k_state = oa::matrix::full(
		&engine,
		[config.batch_size, config.num_heads, config.state_size],
		0.0,
	)?;
	let mut v_state = oa::matrix::full(
		&engine,
		[config.batch_size, config.num_heads, config.head_dim],
		0.0,
	)?;
	let mut actual = vec![0.0; expected.len()];
	for time in 0..config.sequence_length {
		let mut step_config = config;
		step_config.sequence_length = 1;
		let token = host.token(time, config).upload(&engine, step_config)?;
		let output = oa::ml::matrix::mamba3_siso_step(
			&token.c,
			&token.b,
			&token.x,
			&token.z,
			&token.adt,
			&token.dt,
			&token.trap,
			&token.angle,
			&token.c_bias,
			&token.b_bias,
			&token.d,
			&mut ssm_state,
			&mut angle_state,
			&mut k_state,
			&mut v_state,
			step_config,
		)?;
		let values = output.read_f32()?;
		let token_width = config.num_heads * config.head_dim;
		for batch in 0..config.batch_size {
			let source = batch * token_width;
			let destination = (batch * config.sequence_length + time) * token_width;
			actual[destination..destination + token_width]
				.copy_from_slice(&values[source..source + token_width]);
		}
	}
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
		gradcheck_siso(&engine, config, 2.0e-2)
	}
);

test_vk!(
	mamba3_siso_generic_backward_matches_finite_differences,
	engine,
	{
		let config = SsmConfig {
			batch_size: 1,
			sequence_length: 17,
			num_heads: 1,
			num_groups: 1,
			head_dim: 2,
			state_size: 3,
			num_rope_angles: 1,
			mimo_rank: 1,
			has_z: true,
			has_d: true,
			has_output_norm: false,
		};
		gradcheck_siso(&engine, config, 3.0e-2)
	}
);

test_vk!(
	mamba3_siso_chunked_backward_matches_finite_differences,
	engine,
	{
		let config = SsmConfig {
			batch_size: 1,
			sequence_length: 64,
			num_heads: 1,
			num_groups: 1,
			head_dim: 2,
			state_size: 3,
			num_rope_angles: 1,
			mimo_rank: 1,
			has_z: true,
			has_d: true,
			has_output_norm: false,
		};
		gradcheck_siso(&engine, config, 3.0e-2)
	}
);

#[derive(Clone)]
struct PreprocessValues {
	x: Vec<f32>,
	z: Vec<f32>,
	bh: Vec<f32>,
	ch: Vec<f32>,
	dt: Vec<f32>,
	adt: Vec<f32>,
	trap: Vec<f32>,
	angle: Vec<f32>,
}

impl PreprocessValues {
	fn deterministic(rows: usize, config: Mamba3PreprocessConfig) -> Self {
		let fill = |count: usize, offset: usize| {
			(0..count)
				.map(|index| ((index + offset) as f32 * 0.37 + 0.21).sin() * 0.4 + 0.5)
				.collect()
		};
		let bc_width = config.num_groups * config.mimo_rank * config.state_size;
		Self {
			x: fill(rows * config.inner_size, 1),
			z: fill(rows * config.inner_size, 7),
			bh: fill(rows * bc_width, 13),
			ch: fill(rows * bc_width, 19),
			dt: fill(rows * config.num_heads, 23),
			adt: fill(rows * config.num_heads, 29),
			trap: fill(rows * config.num_heads, 31),
			angle: fill(rows * config.num_rope_angles, 37),
		}
	}

	fn upload(
		&self,
		engine: &oa::Engine,
		rows: usize,
		config: Mamba3PreprocessConfig,
	) -> oa::Result<Mamba3PreprocessResult> {
		let bc_width = config.num_groups * config.mimo_rank * config.state_size;
		Ok(Mamba3PreprocessResult {
			x: oa::Matrix::from_f32(engine, [rows, config.inner_size], &self.x)?,
			z: oa::Matrix::from_f32(engine, [rows, config.inner_size], &self.z)?,
			bh: oa::Matrix::from_f32(engine, [rows, bc_width], &self.bh)?,
			ch: oa::Matrix::from_f32(engine, [rows, bc_width], &self.ch)?,
			dt: oa::Matrix::from_f32(engine, [rows, config.num_heads], &self.dt)?,
			adt: oa::Matrix::from_f32(engine, [rows, config.num_heads], &self.adt)?,
			trap: oa::Matrix::from_f32(engine, [rows, config.num_heads], &self.trap)?,
			angle: oa::Matrix::from_f32(engine, [rows, config.num_rope_angles], &self.angle)?,
		})
	}
}

fn preprocess_width(config: Mamba3PreprocessConfig) -> usize {
	2 * config.inner_size
		+ 2 * config.num_groups * config.mimo_rank * config.state_size
		+ 3 * config.num_heads
		+ config.num_rope_angles
}

fn host_preprocess(
	projected: &[f32],
	dt_bias: &[f32],
	rows: usize,
	config: Mamba3PreprocessConfig,
) -> PreprocessValues {
	let bc_rows = config.num_groups * config.mimo_rank;
	let bc_width = bc_rows * config.state_size;
	let width = preprocess_width(config);
	let mut result = PreprocessValues {
		x: vec![0.0; rows * config.inner_size],
		z: vec![0.0; rows * config.inner_size],
		bh: vec![0.0; rows * bc_width],
		ch: vec![0.0; rows * bc_width],
		dt: vec![0.0; rows * config.num_heads],
		adt: vec![0.0; rows * config.num_heads],
		trap: vec![0.0; rows * config.num_heads],
		angle: vec![0.0; rows * config.num_rope_angles],
	};
	for row in 0..rows {
		let row_base = row * width;
		result.z[row * config.inner_size..(row + 1) * config.inner_size]
			.copy_from_slice(&projected[row_base..row_base + config.inner_size]);
		result.x[row * config.inner_size..(row + 1) * config.inner_size].copy_from_slice(
			&projected[row_base + config.inner_size..row_base + 2 * config.inner_size],
		);
		for (output, offset) in [
			(&mut result.bh, 2 * config.inner_size),
			(&mut result.ch, 2 * config.inner_size + bc_width),
		] {
			for bc_row in 0..bc_rows {
				let input_base = row_base + offset + bc_row * config.state_size;
				let square_mean = projected[input_base..input_base + config.state_size]
					.iter()
					.map(|value| value * value)
					.sum::<f32>() / config.state_size as f32;
				let inverse_rms = 1.0 / (square_mean + config.epsilon).sqrt();
				let output_base = row * bc_width + bc_row * config.state_size;
				for index in 0..config.state_size {
					output[output_base + index] = projected[input_base + index] * inverse_rms;
				}
			}
		}
		let dt_offset = 2 * config.inner_size + 2 * bc_width;
		for head in 0..config.num_heads {
			let scalar = row * config.num_heads + head;
			let raw_dt = projected[row_base + dt_offset + head] + dt_bias[head];
			let softplus = raw_dt.max(0.0) + (-raw_dt.abs()).exp().ln_1p();
			let dt = softplus.clamp(config.dt_min, config.dt_max);
			let raw_a = projected[row_base + dt_offset + config.num_heads + head];
			let heavy_tail = if raw_a >= 0.0 {
				1.0 + raw_a
			} else {
				1.0 / (1.0 - raw_a)
			};
			result.dt[scalar] = dt;
			result.adt[scalar] = (-heavy_tail).min(-config.a_floor) * dt;
			result.trap[scalar] = projected[row_base + dt_offset + 2 * config.num_heads + head];
		}
		let angle_offset = dt_offset + 3 * config.num_heads;
		result.angle[row * config.num_rope_angles..(row + 1) * config.num_rope_angles]
			.copy_from_slice(
				&projected
					[row_base + angle_offset..row_base + angle_offset + config.num_rope_angles],
			);
	}
	result
}

fn preprocess_loss(values: &PreprocessValues, weights: &PreprocessValues) -> f32 {
	[
		(&values.x, &weights.x),
		(&values.z, &weights.z),
		(&values.bh, &weights.bh),
		(&values.ch, &weights.ch),
		(&values.dt, &weights.dt),
		(&values.adt, &weights.adt),
		(&values.trap, &weights.trap),
		(&values.angle, &weights.angle),
	]
	.into_iter()
	.flat_map(|(values, weights)| values.iter().zip(weights))
	.map(|(value, weight)| value * weight)
	.sum()
}

fn sum_weighted(value: &oa::Matrix, weight: &oa::Matrix) -> oa::Result<oa::Matrix> {
	oa::matrix::sum(
		&oa::matrix::mul(value, weight)?.reshape([value.num_elements()])?,
		0,
	)?
	.reshape([])
}

test_vk!(
	mamba3_preprocess_matches_host_and_finite_differences,
	engine,
	{
		let rows = 2;
		let config = Mamba3PreprocessConfig {
			inner_size: 3,
			state_size: 3,
			num_heads: 2,
			num_rope_angles: 1,
			num_groups: 1,
			mimo_rank: 1,
			epsilon: 1.0e-5,
			dt_min: 0.01,
			dt_max: 2.0,
			a_floor: 0.2,
		};
		let width = preprocess_width(config);
		let mut projected = (0..rows * width)
			.map(|index| ((index as f32 * 0.29) + 0.4).sin() * 0.45)
			.collect::<Vec<_>>();
		let mut dt_bias = vec![-0.07, 0.11];
		let expected = host_preprocess(&projected, &dt_bias, rows, config);
		let projected_matrix = oa::Matrix::from_f32(&engine, [rows, width], &projected)?;
		let dt_bias_matrix = oa::Matrix::from_f32(&engine, [config.num_heads], &dt_bias)?;
		let actual = oa::ml::matrix::mamba3_preprocess(&projected_matrix, &dt_bias_matrix, config)?;
		for (actual, expected) in [
			(actual.x.read_f32()?, expected.x),
			(actual.z.read_f32()?, expected.z),
			(actual.bh.read_f32()?, expected.bh),
			(actual.ch.read_f32()?, expected.ch),
			(actual.dt.read_f32()?, expected.dt),
			(actual.adt.read_f32()?, expected.adt),
			(actual.trap.read_f32()?, expected.trap),
			(actual.angle.read_f32()?, expected.angle),
		] {
			assert_close(&actual, &expected, 1.0e-4);
		}

		let weights = PreprocessValues::deterministic(rows, config);
		let gradients = weights.upload(&engine, rows, config)?;
		let analytic = oa::ml::matrix::mamba3_preprocess_backward(
			&projected_matrix,
			&dt_bias_matrix,
			&gradients,
			config,
		)?;
		let analytic_projected = analytic.projected.read_f32()?;
		let analytic_dt_bias = analytic.dt_bias.read_f32()?;
		let epsilon = 1.0e-3;
		for index in 0..projected.len() {
			let original = projected[index];
			projected[index] = original + epsilon;
			let positive = preprocess_loss(
				&host_preprocess(&projected, &dt_bias, rows, config),
				&weights,
			);
			projected[index] = original - epsilon;
			let negative = preprocess_loss(
				&host_preprocess(&projected, &dt_bias, rows, config),
				&weights,
			);
			projected[index] = original;
			let numerical = (positive - negative) / (2.0 * epsilon);
			assert!((analytic_projected[index] - numerical).abs() <= 2.5e-3);
		}
		for index in 0..dt_bias.len() {
			let original = dt_bias[index];
			dt_bias[index] = original + epsilon;
			let positive = preprocess_loss(
				&host_preprocess(&projected, &dt_bias, rows, config),
				&weights,
			);
			dt_bias[index] = original - epsilon;
			let negative = preprocess_loss(
				&host_preprocess(&projected, &dt_bias, rows, config),
				&weights,
			);
			dt_bias[index] = original;
			let numerical = (positive - negative) / (2.0 * epsilon);
			assert!((analytic_dt_bias[index] - numerical).abs() <= 2.5e-3);
		}
		Ok(())
	}
);

test_vk!(
	mamba3_preprocess_tape_merges_all_output_adjoints_once,
	engine,
	{
		let rows = 2;
		let config = Mamba3PreprocessConfig {
			inner_size: 3,
			state_size: 3,
			num_heads: 2,
			num_rope_angles: 1,
			num_groups: 1,
			mimo_rank: 1,
			epsilon: 1.0e-5,
			dt_min: 0.01,
			dt_max: 2.0,
			a_floor: 0.2,
		};
		let width = preprocess_width(config);
		let projected = (0..rows * width)
			.map(|index| ((index as f32 * 0.29) + 0.4).sin() * 0.45)
			.collect::<Vec<_>>();
		let dt_bias = vec![-0.07, 0.11];
		let weights = PreprocessValues::deterministic(rows, config);
		let weight_matrices = weights.upload(&engine, rows, config)?;
		let expected = oa::ml::matrix::mamba3_preprocess_backward(
			&oa::Matrix::from_f32(&engine, [rows, width], &projected)?,
			&oa::Matrix::from_f32(&engine, [config.num_heads], &dt_bias)?,
			&weight_matrices,
			config,
		)?;
		let expected_projected = expected.projected.read_f32()?;
		let expected_dt_bias = expected.dt_bias.read_f32()?;

		let projected_parameter = oa::ml::nn::Embedding::from_matrix(oa::Matrix::from_f32(
			&engine,
			[rows, width],
			&projected,
		)?)?;
		let dt_bias_parameter = oa::ml::nn::Embedding::from_matrix(oa::Matrix::from_f32(
			&engine,
			[config.num_heads, 1],
			&dt_bias,
		)?)?;
		let row_indices = oa::Matrix::from_slice(&engine, [rows], &[0_u32, 1])?;
		let head_indices = oa::Matrix::from_slice(&engine, [config.num_heads], &[0_u32, 1])?;
		let tape = oa::ml::GradientTape::new();
		let projected_value = projected_parameter.forward(&row_indices)?;
		let dt_bias_value = dt_bias_parameter
			.forward(&head_indices)?
			.reshape([config.num_heads])?;
		let output = oa::ml::matrix::mamba3_preprocess(&projected_value, &dt_bias_value, config)?;
		let mut loss = sum_weighted(&output.x, &weight_matrices.x)?;
		for term in [
			sum_weighted(&output.z, &weight_matrices.z)?,
			sum_weighted(&output.bh, &weight_matrices.bh)?,
			sum_weighted(&output.ch, &weight_matrices.ch)?,
			sum_weighted(&output.dt, &weight_matrices.dt)?,
			sum_weighted(&output.adt, &weight_matrices.adt)?,
			sum_weighted(&output.trap, &weight_matrices.trap)?,
			sum_weighted(&output.angle, &weight_matrices.angle)?,
		] {
			loss = oa::matrix::add(&loss, &term)?;
		}
		tape.backward(&loss)?;
		assert_close(
			&projected_parameter
				.weight()
				.gradient()
				.expect("projected preprocess gradient is missing")
				.read_f32()?,
			&expected_projected,
			2.0e-4,
		);
		assert_close(
			&dt_bias_parameter
				.weight()
				.gradient()
				.expect("dt-bias preprocess gradient is missing")
				.read_f32()?,
			&expected_dt_bias,
			2.0e-4,
		);
		Ok(())
	}
);

test_vk!(mamba3_preprocess_tape_zero_fills_unused_outputs, engine, {
	let rows = 1;
	let config = Mamba3PreprocessConfig {
		inner_size: 2,
		state_size: 3,
		num_heads: 1,
		num_rope_angles: 1,
		num_groups: 1,
		mimo_rank: 1,
		epsilon: 1.0e-5,
		dt_min: 0.01,
		dt_max: 2.0,
		a_floor: 0.2,
	};
	let width = preprocess_width(config);
	let projected_parameter = oa::ml::nn::Embedding::from_matrix(oa::Matrix::from_f32(
		&engine,
		[rows, width],
		&vec![0.25; rows * width],
	)?)?;
	let dt_bias_parameter =
		oa::ml::nn::Embedding::from_matrix(oa::Matrix::from_f32(&engine, [1, 1], &[0.0])?)?;
	let index = oa::Matrix::from_slice(&engine, [1], &[0_u32])?;
	let tape = oa::ml::GradientTape::new();
	let projected = projected_parameter.forward(&index)?;
	let dt_bias = dt_bias_parameter.forward(&index)?.reshape([1])?;
	let output = oa::ml::matrix::mamba3_preprocess(&projected, &dt_bias, config)?;
	let loss = oa::matrix::sum(&output.x.reshape([config.inner_size])?, 0)?.reshape([])?;
	tape.backward(&loss)?;
	let projected_gradient = projected_parameter
		.weight()
		.gradient()
		.expect("projected preprocess gradient is missing")
		.read_f32()?;
	assert_eq!(&projected_gradient[..config.inner_size], &[0.0, 0.0]);
	assert_eq!(
		&projected_gradient[config.inner_size..2 * config.inner_size],
		&[1.0, 1.0]
	);
	assert!(
		projected_gradient[2 * config.inner_size..]
			.iter()
			.all(|value| *value == 0.0)
	);
	assert_eq!(
		dt_bias_parameter
			.weight()
			.gradient()
			.expect("dt-bias preprocess gradient is missing")
			.read_f32()?,
		vec![0.0]
	);
	Ok(())
});

test_vk!(
	mamba3_module_owns_and_differentiates_the_complete_siso_block,
	engine,
	{
		use oa::ml::Module as _;

		let config = oa::ml::nn::Mamba3Config {
			model_width: 4,
			state_size: 3,
			expand: 2,
			head_dim: 4,
			num_groups: 1,
			mimo_rank: 1,
			rope_fraction: 0.67,
			dt_min: 0.001,
			dt_max: 0.1,
			dt_init_floor: 0.0001,
			a_floor: 0.0001,
			output_norm: true,
		};
		let module = oa::ml::nn::Mamba3::with_seed(&engine, config, 37)?;
		assert_eq!(module.inner_size(), 8);
		assert_eq!(module.num_heads(), 2);
		assert_eq!(module.num_rope_angles(), 1);
		assert_eq!(module.num_parameters()?, 172);
		assert_eq!(
			module
				.all_named_parameters()?
				.iter()
				.map(|entry| entry.path())
				.collect::<Vec<_>>(),
			[
				"in_proj",
				"dt_bias",
				"B_bias",
				"C_bias",
				"D",
				"out_proj",
				"norm_weight",
			]
		);
		let input = oa::Matrix::from_f32(
			&engine,
			[2, 3, 4],
			&(0..24)
				.map(|index| ((index as f32 + 0.3) * 0.41).sin() * 0.4)
				.collect::<Vec<_>>(),
		)?;
		let targets = oa::Matrix::from_slice(&engine, [6], &[0_u32, 1, 2, 3, 1, 0])?;
		let tape = oa::ml::GradientTape::new();
		let output = module.forward(&input)?;
		assert_eq!(output.shape(), [2, 3, 4]);
		let loss = oa::ml::loss::cross_entropy(&output.reshape([6, 4])?, &targets)?;
		tape.backward(&loss)?;
		let values = output.read_f32()?;
		assert!(values.iter().all(|value| value.is_finite()));
		for named in module.all_named_parameters()? {
			let gradient = named
				.parameter()
				.gradient()
				.unwrap_or_else(|| panic!("{} gradient is missing", named.path()));
			assert_eq!(gradient.shape(), named.parameter().data().shape());
			assert!(
				gradient.read_f32()?.iter().all(|value| value.is_finite()),
				"{} gradient is non-finite",
				named.path()
			);
		}
		Ok(())
	}
);

test_vk!(
	mamba3_module_owns_differentiates_and_steps_the_complete_mimo_block,
	engine,
	{
		use oa::ml::Module as _;

		let config = oa::ml::nn::Mamba3Config {
			model_width: 4,
			state_size: 3,
			expand: 2,
			head_dim: 2,
			num_groups: 2,
			mimo_rank: 2,
			rope_fraction: 0.67,
			dt_min: 0.001,
			dt_max: 0.1,
			dt_init_floor: 0.0001,
			a_floor: 0.0001,
			output_norm: true,
		};
		let module = oa::ml::nn::Mamba3::with_seed(&engine, config, 91)?;
		assert_eq!(module.inner_size(), 8);
		assert_eq!(module.num_heads(), 4);
		assert_eq!(module.num_rope_angles(), 1);
		assert_eq!(module.num_parameters()?, 356);
		assert_eq!(
			module
				.all_named_parameters()?
				.iter()
				.map(|entry| entry.path())
				.collect::<Vec<_>>(),
			[
				"in_proj",
				"dt_bias",
				"B_bias",
				"C_bias",
				"mimo_x",
				"mimo_z",
				"mimo_o",
				"D",
				"out_proj",
				"norm_weight",
			]
		);
		let batch = 2;
		let length = 3;
		let input_values = (0..batch * length * config.model_width)
			.map(|index| ((index as f32 + 0.4) * 0.37).sin() * 0.3)
			.collect::<Vec<_>>();
		let input =
			oa::Matrix::from_f32(&engine, [batch, length, config.model_width], &input_values)?;
		let targets = oa::Matrix::from_slice(&engine, [batch * length], &[0_u32, 1, 2, 3, 1, 0])?;
		let tape = oa::ml::GradientTape::new();
		let output = module.forward(&input)?;
		let expected = output.read_f32()?;
		let loss = oa::ml::loss::cross_entropy(
			&output.reshape([batch * length, config.model_width])?,
			&targets,
		)?;
		tape.backward(&loss)?;
		drop(tape);
		for named in module.all_named_parameters()? {
			let gradient = named
				.parameter()
				.gradient()
				.unwrap_or_else(|| panic!("{} gradient is missing", named.path()));
			assert_eq!(gradient.shape(), named.parameter().data().shape());
			assert!(
				gradient.read_f32()?.iter().all(|value| value.is_finite()),
				"{} gradient is non-finite",
				named.path()
			);
		}

		let mut state = module.new_state(batch)?;
		let mut actual = vec![0.0; expected.len()];
		for time in 0..length {
			let mut token = Vec::with_capacity(batch * config.model_width);
			for batch_index in 0..batch {
				let begin = (batch_index * length + time) * config.model_width;
				token.extend_from_slice(&input_values[begin..begin + config.model_width]);
			}
			let token = oa::Matrix::from_f32(&engine, [batch, 1, config.model_width], &token)?;
			let values = module.step(&token, &mut state)?.read_f32()?;
			for batch_index in 0..batch {
				let source = batch_index * config.model_width;
				let destination = (batch_index * length + time) * config.model_width;
				actual[destination..destination + config.model_width]
					.copy_from_slice(&values[source..source + config.model_width]);
			}
		}
		assert_close(&actual, &expected, 3.0e-4);
		Ok(())
	}
);

test_vk!(
	mamba3_module_recurrent_state_matches_full_forward,
	engine,
	{
		let config = oa::ml::nn::Mamba3Config {
			model_width: 4,
			state_size: 4,
			expand: 2,
			head_dim: 2,
			num_groups: 2,
			mimo_rank: 1,
			rope_fraction: 1.0,
			dt_min: 0.001,
			dt_max: 0.1,
			dt_init_floor: 0.0001,
			a_floor: 0.0001,
			output_norm: true,
		};
		let module = oa::ml::nn::Mamba3::with_seed(&engine, config, 73)?;
		let batch = 2;
		let length = 5;
		let input_values = (0..batch * length * config.model_width)
			.map(|index| ((index as f32 + 0.7) * 0.31).cos() * 0.3)
			.collect::<Vec<_>>();
		let input =
			oa::Matrix::from_f32(&engine, [batch, length, config.model_width], &input_values)?;
		let expected = module.forward(&input)?.read_f32()?;
		let mut state = module.new_state(batch)?;
		assert_eq!(state.batch_size(), batch);
		let mut actual = vec![0.0; expected.len()];
		for time in 0..length {
			let mut token = Vec::with_capacity(batch * config.model_width);
			for batch_index in 0..batch {
				let begin = (batch_index * length + time) * config.model_width;
				token.extend_from_slice(&input_values[begin..begin + config.model_width]);
			}
			let output = module
				.step(
					&oa::Matrix::from_f32(&engine, [batch, 1, config.model_width], &token)?,
					&mut state,
				)?
				.read_f32()?;
			for batch_index in 0..batch {
				let source = batch_index * config.model_width;
				let destination = (batch_index * length + time) * config.model_width;
				actual[destination..destination + config.model_width]
					.copy_from_slice(&output[source..source + config.model_width]);
			}
		}
		assert_close(&actual, &expected, 2.0e-4);

		let other = oa::ml::nn::Mamba3::with_seed(&engine, config, 74)?;
		let token = oa::Matrix::from_f32(
			&engine,
			[batch, 1, config.model_width],
			&input_values[..batch * config.model_width],
		)?;
		assert_eq!(
			other
				.step(&token, &mut state)
				.err()
				.expect("foreign Mamba state must be rejected")
				.kind(),
			oa::ErrorKind::InvalidArgument
		);
		let tape = oa::ml::GradientTape::new();
		let mut state = module.new_state(batch)?;
		assert_eq!(
			module
				.step(&token, &mut state)
				.err()
				.expect("Mamba step under autograd must be rejected")
				.kind(),
			oa::ErrorKind::FailedPrecondition
		);
		drop(tape);
		Ok(())
	}
);
