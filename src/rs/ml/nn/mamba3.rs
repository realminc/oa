use crate::{DType, Engine, Error, Matrix, Result};

use super::super::{
	Module, ModuleRegistry, Parameter, autograd,
	lowering::{matrix as dispatch, ssm as ssm_dispatch},
	matrix as matrix_ops, random,
};

/// Construction contract for the donor-backed Mamba-3 block.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Mamba3Config {
	/// Input and output channel width.
	pub model_width: usize,
	/// Selective recurrent state width.
	pub state_size: usize,
	/// Integer expansion factor applied to `model_width`.
	pub expand: usize,
	/// Value width per selective-state head.
	pub head_dim: usize,
	/// Number of B/C groups shared across heads.
	pub num_groups: usize,
	/// Shared-state input/output rank; one selects the SISO route.
	pub mimo_rank: usize,
	/// Fraction of the state width assigned to rotary pairs.
	pub rope_fraction: f32,
	/// Inclusive minimum transformed time step.
	pub dt_min: f32,
	/// Inclusive maximum transformed time step.
	pub dt_max: f32,
	/// Minimum initialized transformed time step.
	pub dt_init_floor: f32,
	/// Positive floor used by the heavy-tail A transform.
	pub a_floor: f32,
	/// Normalize each head and apply the z gate after the SSM scan.
	pub output_norm: bool,
}

impl Mamba3Config {
	/// Construct the OA defaults for one model width.
	pub const fn new(model_width: usize) -> Self {
		Self {
			model_width,
			state_size: 128,
			expand: 2,
			head_dim: 64,
			num_groups: 1,
			mimo_rank: 1,
			rope_fraction: 0.5,
			dt_min: 0.001,
			dt_max: 0.1,
			dt_init_floor: 0.0001,
			a_floor: 0.0001,
			output_norm: false,
		}
	}
}

/// Explicit recurrent cache for one [`Mamba3`] module and batch size.
///
/// The cache owns the selective state, cumulative rotary angle, previous key,
/// and previous value buffers. It is deliberately separate from the module so
/// independent decoding streams never share hidden mutable state.
pub struct Mamba3State {
	owner_id: u64,
	batch_size: usize,
	ssm: Matrix,
	angle: Matrix,
	key: Matrix,
	value: Matrix,
}

impl Mamba3State {
	/// Return the fixed batch size accepted by this cache.
	pub const fn batch_size(&self) -> usize {
		self.batch_size
	}
}

/// Parameter-owning Mamba-3 selective state-space block.
pub struct Mamba3 {
	config: Mamba3Config,
	inner_size: usize,
	num_heads: usize,
	num_rope_angles: usize,
	in_projection: Parameter,
	dt_bias: Parameter,
	b_bias: Parameter,
	c_bias: Parameter,
	mimo_x: Option<Parameter>,
	mimo_z: Option<Parameter>,
	mimo_o: Option<Parameter>,
	d_skip: Parameter,
	out_projection: Parameter,
	norm_weight: Option<Parameter>,
	norm_identity: Matrix,
	in_projection_bias: Matrix,
	out_projection_bias: Matrix,
	registry: ModuleRegistry,
}

impl Mamba3 {
	/// Construct a deterministically initialized Mamba-3 block.
	///
	/// The module preserves the donor parameter layout and complete
	/// projection → preprocess → SISO scan → optional gated per-head RMSNorm →
	/// output-projection path. `seed` owns an independent deterministic
	/// initializer stream.
	///
	/// # Errors
	///
	/// Returns an error for invalid or overflowing geometry, invalid numerical
	/// bounds, or failed allocation/registration.
	pub fn with_seed(engine: &Engine, config: Mamba3Config, seed: u64) -> Result<Self> {
		let inner_size = config
			.model_width
			.checked_mul(config.expand)
			.ok_or_else(|| Error::invalid_argument("Mamba3 inner width overflows usize"))?;
		if config.model_width == 0
			|| config.state_size == 0
			|| config.state_size > 128
			|| config.expand == 0
			|| config.head_dim == 0
			|| config.head_dim > 128
			|| inner_size == 0
			|| !inner_size.is_multiple_of(config.head_dim)
			|| config.num_groups == 0
			|| !(1..=8).contains(&config.mimo_rank)
			|| (config.mimo_rank > 1 && config.head_dim.saturating_mul(config.state_size) > 4096)
			|| !config.rope_fraction.is_finite()
			|| !(0.0..=1.0).contains(&config.rope_fraction)
			|| !config.dt_min.is_finite()
			|| !config.dt_max.is_finite()
			|| !config.dt_init_floor.is_finite()
			|| !config.a_floor.is_finite()
			|| config.dt_min <= 0.0
			|| config.dt_max < config.dt_min
			|| config.dt_init_floor <= 0.0
			|| config.a_floor <= 0.0
		{
			return Err(Error::invalid_argument(
				"Mamba3 requires positive divisible dimensions, MIMO rank in 1..=8 with P*N <= 4096, rope_fraction in [0,1], ordered positive dt bounds, and positive dt/A floors",
			));
		}
		let num_heads = inner_size / config.head_dim;
		if num_heads == 0 || !num_heads.is_multiple_of(config.num_groups) {
			return Err(Error::invalid_argument(
				"Mamba3 head count must be nonzero and divisible by num_groups",
			));
		}
		let rope_width = ((config.state_size as f64 * f64::from(config.rope_fraction)).floor()
			as usize) & !1_usize;
		let num_rope_angles = rope_width / 2;
		if num_rope_angles > 64 {
			return Err(Error::invalid_argument(
				"Mamba3 rotary angle count exceeds the admitted SISO kernel limit of 64",
			));
		}
		let grouped_state = config
			.num_groups
			.checked_mul(config.state_size)
			.ok_or_else(|| Error::invalid_argument("Mamba3 grouped state width overflows"))?;
		let grouped_rank_state = grouped_state
			.checked_mul(config.mimo_rank)
			.ok_or_else(|| Error::invalid_argument("Mamba3 grouped MIMO state width overflows"))?;
		let projected_width = inner_size
			.checked_mul(2)
			.and_then(|value| {
				grouped_rank_state
					.checked_mul(2)
					.and_then(|bc| value.checked_add(bc))
			})
			.and_then(|value| {
				num_heads
					.checked_mul(3)
					.and_then(|heads| value.checked_add(heads))
			})
			.and_then(|value| value.checked_add(num_rope_angles))
			.ok_or_else(|| Error::invalid_argument("Mamba3 projected width overflows usize"))?;

		let in_count = projected_width
			.checked_mul(config.model_width)
			.ok_or_else(|| Error::invalid_argument("Mamba3 input projection size overflows"))?;
		let in_limit = (6.0_f32 / (projected_width + config.model_width) as f32).sqrt();
		let in_projection = Parameter::new(
			"in_proj",
			Matrix::from_f32(
				engine,
				[projected_width, config.model_width],
				&random::symmetric_uniform(in_count, in_limit, seed),
			)?,
		)?;
		let dt_uniform = random::unit_uniform(num_heads, seed.wrapping_add(1));
		let log_min = config.dt_min.ln();
		let log_range = config.dt_max.ln() - log_min;
		let dt_bias_values = dt_uniform
			.into_iter()
			.map(|sample| {
				let dt = (log_min + sample * log_range)
					.exp()
					.max(config.dt_init_floor);
				dt + (-(-dt).exp_m1()).ln()
			})
			.collect::<Vec<_>>();
		let dt_bias = Parameter::new(
			"dt_bias",
			Matrix::from_f32(engine, [num_heads], &dt_bias_values)?,
		)?;
		let bias_count = num_heads
			.checked_mul(config.mimo_rank)
			.and_then(|value| value.checked_mul(config.state_size))
			.ok_or_else(|| Error::invalid_argument("Mamba3 B/C bias size overflows"))?;
		let bias_shape = if config.mimo_rank > 1 {
			vec![num_heads, config.mimo_rank, config.state_size]
		} else {
			vec![num_heads, config.state_size]
		};
		let b_bias = Parameter::new(
			"B_bias",
			Matrix::from_f32(engine, bias_shape.clone(), &vec![1.0; bias_count])?,
		)?;
		let c_bias = Parameter::new(
			"C_bias",
			Matrix::from_f32(engine, bias_shape, &vec![1.0; bias_count])?,
		)?;
		let mimo_count = num_heads
			.checked_mul(config.mimo_rank)
			.and_then(|value| value.checked_mul(config.head_dim))
			.ok_or_else(|| Error::invalid_argument("Mamba3 MIMO projection size overflows"))?;
		let mimo_shape = [num_heads, config.mimo_rank, config.head_dim];
		let mimo_x = (config.mimo_rank > 1)
			.then(|| {
				Parameter::new(
					"mimo_x",
					Matrix::from_f32(
						engine,
						mimo_shape,
						&vec![1.0 / config.mimo_rank as f32; mimo_count],
					)?,
				)
			})
			.transpose()?;
		let mimo_z = (config.mimo_rank > 1)
			.then(|| {
				Parameter::new(
					"mimo_z",
					Matrix::from_f32(engine, mimo_shape, &vec![1.0; mimo_count])?,
				)
			})
			.transpose()?;
		let mimo_o = (config.mimo_rank > 1)
			.then(|| {
				Parameter::new(
					"mimo_o",
					Matrix::from_f32(
						engine,
						mimo_shape,
						&vec![1.0 / config.mimo_rank as f32; mimo_count],
					)?,
				)
			})
			.transpose()?;
		let d_skip = Parameter::new(
			"D",
			Matrix::from_f32(engine, [num_heads], &vec![0.0; num_heads])?,
		)?;
		let out_count = config
			.model_width
			.checked_mul(inner_size)
			.ok_or_else(|| Error::invalid_argument("Mamba3 output projection size overflows"))?;
		let out_limit = (6.0_f32 / (config.model_width + inner_size) as f32).sqrt();
		let out_projection = Parameter::new(
			"out_proj",
			Matrix::from_f32(
				engine,
				[config.model_width, inner_size],
				&random::symmetric_uniform(out_count, out_limit, seed.wrapping_add(2)),
			)?,
		)?;
		let norm_weight = config
			.output_norm
			.then(|| {
				Parameter::new(
					"norm_weight",
					Matrix::from_f32(engine, [num_heads, config.head_dim], &vec![1.0; inner_size])?,
				)
			})
			.transpose()?;
		let norm_identity =
			Matrix::from_f32(engine, [num_heads, config.head_dim], &vec![1.0; inner_size])?;
		let in_projection_bias =
			Matrix::from_f32(engine, [projected_width], &vec![0.0; projected_width])?;
		let out_projection_bias =
			Matrix::from_f32(engine, [config.model_width], &vec![0.0; config.model_width])?;

		let mut registry = ModuleRegistry::new();
		for (name, parameter) in [
			("in_proj", &in_projection),
			("dt_bias", &dt_bias),
			("B_bias", &b_bias),
			("C_bias", &c_bias),
		] {
			registry.register_parameter(name, parameter.clone())?;
		}
		for (name, parameter) in [
			("mimo_x", &mimo_x),
			("mimo_z", &mimo_z),
			("mimo_o", &mimo_o),
		] {
			if let Some(parameter) = parameter {
				registry.register_parameter(name, parameter.clone())?;
			}
		}
		registry.register_parameter("D", d_skip.clone())?;
		registry.register_parameter("out_proj", out_projection.clone())?;
		if let Some(parameter) = &norm_weight {
			registry.register_parameter("norm_weight", parameter.clone())?;
		}
		Ok(Self {
			config,
			inner_size,
			num_heads,
			num_rope_angles,
			in_projection,
			dt_bias,
			b_bias,
			c_bias,
			mimo_x,
			mimo_z,
			mimo_o,
			d_skip,
			out_projection,
			norm_weight,
			norm_identity,
			in_projection_bias,
			out_projection_bias,
			registry,
		})
	}

	/// Evaluate one complete Mamba-3 block without submitting or waiting.
	///
	/// # Errors
	///
	/// Returns an error unless `input` is a nonempty same-engine F32
	/// `[batch, sequence, model_width]` value or an owned operation cannot be
	/// recorded.
	pub fn forward(&self, input: &Matrix) -> Result<Matrix> {
		let [batch, sequence_length, model_width] = input.shape() else {
			return Err(Error::invalid_argument(
				"Mamba3 input must have shape [batch, sequence, model_width]",
			));
		};
		if *batch == 0
			|| *sequence_length == 0
			|| *model_width != self.config.model_width
			|| input.dtype() != DType::F32
		{
			return Err(Error::invalid_argument(format!(
				"Mamba3 requires nonempty F32 input [B, L, {}]; found {:?} {}",
				self.config.model_width,
				input.shape(),
				input.dtype().token()
			)));
		}
		let rows = batch
			.checked_mul(*sequence_length)
			.ok_or_else(|| Error::invalid_argument("Mamba3 flattened row count overflows"))?;
		let input_2d = input.reshape([rows, self.config.model_width])?;
		let projected =
			self.parameter_linear(&input_2d, &self.in_projection, &self.in_projection_bias)?;

		for parameter in [&self.dt_bias, &self.b_bias, &self.c_bias, &self.d_skip] {
			autograd::record_parameter_leaf(parameter)?;
		}
		for parameter in [&self.mimo_x, &self.mimo_z, &self.mimo_o]
			.into_iter()
			.flatten()
		{
			autograd::record_parameter_leaf(parameter)?;
		}
		let dt_bias = self.dt_bias.data();
		let preprocess = matrix_ops::mamba3_preprocess(
			&projected,
			&dt_bias,
			crate::ml::matrix::Mamba3PreprocessConfig {
				inner_size: self.inner_size,
				state_size: self.config.state_size,
				num_heads: self.num_heads,
				num_rope_angles: self.num_rope_angles,
				num_groups: self.config.num_groups,
				mimo_rank: self.config.mimo_rank,
				epsilon: 1e-5,
				dt_min: self.config.dt_min,
				dt_max: self.config.dt_max,
				a_floor: self.config.a_floor,
			},
		)?;
		let x = preprocess.x.reshape([
			*batch,
			*sequence_length,
			self.num_heads,
			self.config.head_dim,
		])?;
		let z = preprocess.z.reshape(x.shape().to_vec())?;
		let b = preprocess.bh.reshape([
			*batch,
			*sequence_length,
			self.config.num_groups * self.config.mimo_rank,
			self.config.state_size,
		])?;
		let c = preprocess.ch.reshape(b.shape().to_vec())?;
		let adt = preprocess
			.adt
			.reshape([*batch, *sequence_length, self.num_heads])?;
		let dt = preprocess
			.dt
			.reshape([*batch, *sequence_length, self.num_heads])?;
		let trap = preprocess
			.trap
			.reshape([*batch, *sequence_length, self.num_heads])?;
		let angle = preprocess
			.angle
			.reshape([*batch, *sequence_length, self.num_rope_angles])?;
		let c_bias = self.c_bias.data();
		let b_bias = self.b_bias.data();
		let d_skip = self.d_skip.data();
		let scan_config = crate::ml::matrix::SsmConfig {
			batch_size: *batch,
			sequence_length: *sequence_length,
			num_heads: self.num_heads,
			num_groups: self.config.num_groups,
			head_dim: self.config.head_dim,
			state_size: self.config.state_size,
			num_rope_angles: self.num_rope_angles,
			mimo_rank: self.config.mimo_rank,
			has_z: self.config.mimo_rank > 1 || !self.config.output_norm,
			has_d: true,
			has_output_norm: self.config.mimo_rank > 1 && self.config.output_norm,
		};
		let scan = if let (Some(mimo_x), Some(mimo_z), Some(mimo_o)) =
			(&self.mimo_x, &self.mimo_z, &self.mimo_o)
		{
			if let Some(norm_weight) = &self.norm_weight {
				autograd::record_parameter_leaf(norm_weight)?;
			}
			let norm_weight = self
				.norm_weight
				.as_ref()
				.map_or_else(|| self.norm_identity.clone(), Parameter::data);
			matrix_ops::mamba3_mimo(
				&c,
				&b,
				&x,
				&z,
				&adt,
				&dt,
				&trap,
				&angle,
				&c_bias,
				&b_bias,
				&d_skip,
				&mimo_x.data(),
				&mimo_z.data(),
				&mimo_o.data(),
				&norm_weight,
				scan_config,
			)?
		} else {
			matrix_ops::mamba3_siso(
				&c,
				&b,
				&x,
				&z,
				&adt,
				&dt,
				&trap,
				&angle,
				&c_bias,
				&b_bias,
				&d_skip,
				scan_config,
			)?
		};
		let scan = if self.config.mimo_rank == 1 {
			if let Some(norm_weight) = &self.norm_weight {
				autograd::record_parameter_leaf(norm_weight)?;
				matrix_ops::rms_norm_gated(&scan, &norm_weight.data(), None, &z, 1e-5)?
			} else {
				scan
			}
		} else {
			scan
		};
		let flattened = scan.reshape([rows, self.inner_size])?;
		let output =
			self.parameter_linear(&flattened, &self.out_projection, &self.out_projection_bias)?;
		output.reshape([*batch, *sequence_length, self.config.model_width])
	}

	/// Allocate a zero-initialized recurrent cache for one positive batch size.
	///
	/// # Errors
	///
	/// Returns an error when `batch_size` is zero, state geometry overflows, or
	/// device allocation fails.
	pub fn new_state(&self, batch_size: usize) -> Result<Mamba3State> {
		if batch_size == 0 {
			return Err(Error::invalid_argument(
				"Mamba3 recurrent state requires a positive batch size",
			));
		}
		let owner = self.in_projection.data();
		let allocate = |shape: Vec<usize>| -> Result<Matrix> {
			let count = shape.iter().try_fold(1_usize, |count, extent| {
				count.checked_mul(*extent).ok_or_else(|| {
					Error::invalid_argument("Mamba3 recurrent state size overflows usize")
				})
			})?;
			Matrix::allocate(owner.engine_handle(), shape, count, DType::F32)
		};
		Ok(Mamba3State {
			owner_id: owner.value_id(),
			batch_size,
			ssm: allocate(vec![
				batch_size,
				self.num_heads,
				self.config.head_dim,
				self.config.state_size,
			])?,
			angle: allocate(vec![batch_size, self.num_heads, self.num_rope_angles])?,
			key: if self.config.mimo_rank > 1 {
				allocate(vec![
					batch_size,
					self.num_heads,
					self.config.mimo_rank,
					self.config.state_size,
				])?
			} else {
				allocate(vec![batch_size, self.num_heads, self.config.state_size])?
			},
			value: if self.config.mimo_rank > 1 {
				allocate(vec![
					batch_size,
					self.num_heads,
					self.config.mimo_rank,
					self.config.head_dim,
				])?
			} else {
				allocate(vec![batch_size, self.num_heads, self.config.head_dim])?
			},
		})
	}

	/// Advance one autoregressive token using an explicit recurrent cache.
	///
	/// `input` must be F32 `[batch, 1, model_width]`. This is an inference-only
	/// operation: recurrent-state differentiation is not admitted, and calling it
	/// while a [`crate::ml::GradientTape`] is active returns an error.
	///
	/// # Errors
	///
	/// Returns an error for an active gradient tape, a cache owned by another
	/// module, mismatched batch/shape/dtype/Engine ownership, or failed recording.
	pub fn step(&self, input: &Matrix, state: &mut Mamba3State) -> Result<Matrix> {
		if autograd::recording_active() {
			return Err(Error::failed_precondition(
				"Mamba3 recurrent step is inference-only and cannot run inside a GradientTape",
			));
		}
		let owner = self.in_projection.data();
		if state.owner_id != owner.value_id() {
			return Err(Error::invalid_argument(
				"Mamba3 recurrent state belongs to another module instance",
			));
		}
		if input.shape() != [state.batch_size, 1, self.config.model_width]
			|| input.dtype() != DType::F32
		{
			return Err(Error::invalid_argument(format!(
				"Mamba3 step requires F32 input [{}, 1, {}]; found {:?} {}",
				state.batch_size,
				self.config.model_width,
				input.shape(),
				input.dtype().token()
			)));
		}

		let input_2d = input.reshape([state.batch_size, self.config.model_width])?;
		let projected = dispatch::linear(&input_2d, &owner, &self.in_projection_bias)?;
		let preprocess = ssm_dispatch::mamba3_preprocess(
			&projected,
			&self.dt_bias.data(),
			crate::ml::matrix::Mamba3PreprocessConfig {
				inner_size: self.inner_size,
				state_size: self.config.state_size,
				num_heads: self.num_heads,
				num_rope_angles: self.num_rope_angles,
				num_groups: self.config.num_groups,
				mimo_rank: self.config.mimo_rank,
				epsilon: 1e-5,
				dt_min: self.config.dt_min,
				dt_max: self.config.dt_max,
				a_floor: self.config.a_floor,
			},
		)?;
		let x =
			preprocess
				.x
				.reshape([state.batch_size, 1, self.num_heads, self.config.head_dim])?;
		let z = preprocess.z.reshape(x.shape().to_vec())?;
		let b = preprocess.bh.reshape([
			state.batch_size,
			1,
			self.config.num_groups * self.config.mimo_rank,
			self.config.state_size,
		])?;
		let c = preprocess.ch.reshape(b.shape().to_vec())?;
		let adt = preprocess
			.adt
			.reshape([state.batch_size, 1, self.num_heads])?;
		let dt = preprocess
			.dt
			.reshape([state.batch_size, 1, self.num_heads])?;
		let trap = preprocess
			.trap
			.reshape([state.batch_size, 1, self.num_heads])?;
		let angle = preprocess
			.angle
			.reshape([state.batch_size, 1, self.num_rope_angles])?;
		let scan_config = crate::ml::matrix::SsmConfig {
			batch_size: state.batch_size,
			sequence_length: 1,
			num_heads: self.num_heads,
			num_groups: self.config.num_groups,
			head_dim: self.config.head_dim,
			state_size: self.config.state_size,
			num_rope_angles: self.num_rope_angles,
			mimo_rank: self.config.mimo_rank,
			has_z: self.config.mimo_rank > 1 || !self.config.output_norm,
			has_d: true,
			has_output_norm: self.config.mimo_rank > 1 && self.config.output_norm,
		};
		let scan = if let (Some(mimo_x), Some(mimo_z), Some(mimo_o)) =
			(&self.mimo_x, &self.mimo_z, &self.mimo_o)
		{
			let norm_weight = self
				.norm_weight
				.as_ref()
				.map_or_else(|| self.norm_identity.clone(), Parameter::data);
			ssm_dispatch::mamba3_mimo_step(
				&c,
				&b,
				&x,
				&z,
				&adt,
				&dt,
				&trap,
				&angle,
				&self.c_bias.data(),
				&self.b_bias.data(),
				&self.d_skip.data(),
				&mimo_x.data(),
				&mimo_z.data(),
				&mimo_o.data(),
				&norm_weight,
				&state.ssm,
				&state.angle,
				&state.key,
				&state.value,
				scan_config,
			)?
		} else {
			ssm_dispatch::mamba3_siso_step(
				&c,
				&b,
				&x,
				&z,
				&adt,
				&dt,
				&trap,
				&angle,
				&self.c_bias.data(),
				&self.b_bias.data(),
				&self.d_skip.data(),
				&state.ssm,
				&state.angle,
				&state.key,
				&state.value,
				scan_config,
			)?
		};
		let scan = if self.config.mimo_rank == 1 {
			if let Some(weight) = &self.norm_weight {
				dispatch::rms_norm_gated(&scan, &weight.data(), None, &z, 1e-5)?
			} else {
				scan
			}
		} else {
			scan
		};
		let flattened = scan.reshape([state.batch_size, self.inner_size])?;
		let output = dispatch::linear(
			&flattened,
			&self.out_projection.data(),
			&self.out_projection_bias,
		)?;
		output.reshape([state.batch_size, 1, self.config.model_width])
	}

	fn parameter_linear(
		&self,
		input: &Matrix,
		weight: &Parameter,
		bias: &Matrix,
	) -> Result<Matrix> {
		let (weight_value, version, requires_grad) = weight.snapshot();
		let output = dispatch::linear(input, &weight_value, bias)?;
		if requires_grad {
			autograd::record_linear(input, &output, weight.clone(), weight_value, version, None)?;
		}
		Ok(output)
	}

	/// Return the immutable construction contract.
	pub const fn config(&self) -> Mamba3Config {
		self.config
	}

	/// Return the expanded SSM width.
	pub const fn inner_size(&self) -> usize {
		self.inner_size
	}

	/// Return the selective-state head count.
	pub const fn num_heads(&self) -> usize {
		self.num_heads
	}

	/// Return the rotary angle count.
	pub const fn num_rope_angles(&self) -> usize {
		self.num_rope_angles
	}

	/// Return the exact registered parameters in donor order.
	pub fn parameters(&self) -> Vec<Parameter> {
		let mut parameters = vec![
			self.in_projection.clone(),
			self.dt_bias.clone(),
			self.b_bias.clone(),
			self.c_bias.clone(),
		];
		for parameter in [&self.mimo_x, &self.mimo_z, &self.mimo_o]
			.into_iter()
			.flatten()
		{
			parameters.push(parameter.clone());
		}
		parameters.push(self.d_skip.clone());
		parameters.push(self.out_projection.clone());
		if let Some(weight) = &self.norm_weight {
			parameters.push(weight.clone());
		}
		parameters
	}
}

impl Module for Mamba3 {
	fn forward(&self, input: &Matrix) -> Result<Matrix> {
		Mamba3::forward(self, input)
	}

	fn registry(&self) -> &ModuleRegistry {
		&self.registry
	}
}
