//! Private lowering for selective state-space operations.

use crate::{
	DType, Error, Matrix, OpAttribute, Result,
	runtime::{BufferBinding, ComputeDispatch, KernelId, PushConstant, SemanticDispatch},
};

use super::common::{shader_u32, validate_f32_same_engine};
use crate::ml::matrix::{Mamba3PreprocessConfig, Mamba3PreprocessResult, SsmConfig};

const MAMBA3_CHUNK_HISTORY_LIMIT_BYTES: usize = 256 * 1024 * 1024;

pub(in crate::ml) struct Mamba3PreprocessBackward {
	pub(in crate::ml) projected: Matrix,
	pub(in crate::ml) dt_bias: Matrix,
}

pub(in crate::ml) struct SsmBackward {
	pub(in crate::ml) c: Matrix,
	pub(in crate::ml) b: Matrix,
	pub(in crate::ml) x: Matrix,
	pub(in crate::ml) z: Matrix,
	pub(in crate::ml) adt: Matrix,
	pub(in crate::ml) dt: Matrix,
	pub(in crate::ml) trap: Matrix,
	pub(in crate::ml) angle: Matrix,
	pub(in crate::ml) c_bias: Matrix,
	pub(in crate::ml) b_bias: Matrix,
	pub(in crate::ml) d: Matrix,
}

pub(in crate::ml) struct Mamba3MimoBackward {
	pub(in crate::ml) c: Matrix,
	pub(in crate::ml) b: Matrix,
	pub(in crate::ml) x: Matrix,
	pub(in crate::ml) z: Matrix,
	pub(in crate::ml) adt: Matrix,
	pub(in crate::ml) dt: Matrix,
	pub(in crate::ml) trap: Matrix,
	pub(in crate::ml) angle: Matrix,
	pub(in crate::ml) c_bias: Matrix,
	pub(in crate::ml) b_bias: Matrix,
	pub(in crate::ml) d: Matrix,
	pub(in crate::ml) mimo_x: Matrix,
	pub(in crate::ml) mimo_z: Matrix,
	pub(in crate::ml) mimo_o: Matrix,
	pub(in crate::ml) norm_weight: Matrix,
}

struct Geometry {
	batch: u32,
	length: u32,
	heads: u32,
	groups: u32,
	head_dim: u32,
	state_size: u32,
	angles: u32,
	batch_heads: u32,
}

impl Geometry {
	fn resolve(inputs: &[&Matrix; 11], config: SsmConfig, operation: &'static str) -> Result<Self> {
		let groups = if config.num_groups == 0 {
			config.num_heads
		} else {
			config.num_groups
		};
		if config.batch_size == 0
			|| config.sequence_length == 0
			|| config.num_heads == 0
			|| groups == 0
			|| !config.num_heads.is_multiple_of(groups)
			|| config.head_dim == 0
			|| config.head_dim > 128
			|| config.state_size == 0
			|| config.state_size > 128
			|| config.num_rope_angles > 64
			|| config.num_rope_angles.saturating_mul(2) > config.state_size
			|| config.mimo_rank != 1
			|| config.has_output_norm
		{
			return Err(Error::invalid_argument(format!(
				"{operation} requires B/L/H/G/P/N > 0, H divisible by G, P/N <= 128, A <= 64, 2A <= N, MIMO rank one, and no MIMO output normalization"
			)));
		}
		let expected = [
			vec![
				config.batch_size,
				config.sequence_length,
				groups,
				config.state_size,
			],
			vec![
				config.batch_size,
				config.sequence_length,
				groups,
				config.state_size,
			],
			vec![
				config.batch_size,
				config.sequence_length,
				config.num_heads,
				config.head_dim,
			],
			vec![
				config.batch_size,
				config.sequence_length,
				config.num_heads,
				config.head_dim,
			],
			vec![config.batch_size, config.sequence_length, config.num_heads],
			vec![config.batch_size, config.sequence_length, config.num_heads],
			vec![config.batch_size, config.sequence_length, config.num_heads],
			vec![
				config.batch_size,
				config.sequence_length,
				config.num_rope_angles,
			],
			vec![config.num_heads, config.state_size],
			vec![config.num_heads, config.state_size],
			vec![config.num_heads],
		];
		if inputs
			.iter()
			.zip(expected.iter())
			.any(|(matrix, shape)| matrix.shape() != shape)
		{
			return Err(Error::invalid_argument(format!(
				"{operation} operands do not match the explicit SSM configuration"
			)));
		}
		validate_f32_same_engine(operation, inputs)?;
		for shape in &expected {
			let count = checked_count(shape, operation)?;
			shader_u32(count, "tensor element count", operation)?;
		}
		let batch = shader_u32(config.batch_size, "batch size", operation)?;
		let length = shader_u32(config.sequence_length, "sequence length", operation)?;
		let heads = shader_u32(config.num_heads, "head count", operation)?;
		let groups = shader_u32(groups, "group count", operation)?;
		let head_dim = shader_u32(config.head_dim, "head dimension", operation)?;
		let state_size = shader_u32(config.state_size, "state size", operation)?;
		let angles = shader_u32(config.num_rope_angles, "rope angle count", operation)?;
		let batch_heads = batch.checked_mul(heads).ok_or_else(|| {
			Error::invalid_argument(format!("{operation} batch-head count exceeds u32"))
		})?;
		Ok(Self {
			batch,
			length,
			heads,
			groups,
			head_dim,
			state_size,
			angles,
			batch_heads,
		})
	}

	fn output_shape(&self) -> Vec<usize> {
		vec![
			self.batch as usize,
			self.length as usize,
			self.heads as usize,
			self.head_dim as usize,
		]
	}
}

struct MimoGeometry {
	batch: u32,
	length: u32,
	heads: u32,
	groups: u32,
	head_dim: u32,
	state_size: u32,
	angles: u32,
	rank: u32,
	batch_heads: u32,
}

impl MimoGeometry {
	fn resolve(inputs: &[&Matrix; 15], config: SsmConfig, operation: &'static str) -> Result<Self> {
		let groups = if config.num_groups == 0 {
			config.num_heads
		} else {
			config.num_groups
		};
		if config.batch_size == 0
			|| config.sequence_length == 0
			|| config.num_heads == 0
			|| groups == 0
			|| !config.num_heads.is_multiple_of(groups)
			|| config.head_dim == 0
			|| config.head_dim > 128
			|| config.state_size == 0
			|| config.state_size > 128
			|| config.head_dim.saturating_mul(config.state_size) > 4096
			|| config.num_rope_angles > 64
			|| config.num_rope_angles.saturating_mul(2) > config.state_size
			|| !(1..=8).contains(&config.mimo_rank)
		{
			return Err(Error::invalid_argument(format!(
				"{operation} requires B/L/H/G/P/N > 0, H divisible by G, P/N <= 128, P*N <= 4096, A <= 64, 2A <= N, and MIMO rank in 1..=8"
			)));
		}
		let rg = config.mimo_rank.checked_mul(groups).ok_or_else(|| {
			Error::invalid_argument(format!("{operation} rank-group extent overflows usize"))
		})?;
		let expected = [
			vec![
				config.batch_size,
				config.sequence_length,
				rg,
				config.state_size,
			],
			vec![
				config.batch_size,
				config.sequence_length,
				rg,
				config.state_size,
			],
			vec![
				config.batch_size,
				config.sequence_length,
				config.num_heads,
				config.head_dim,
			],
			vec![
				config.batch_size,
				config.sequence_length,
				config.num_heads,
				config.head_dim,
			],
			vec![config.batch_size, config.sequence_length, config.num_heads],
			vec![config.batch_size, config.sequence_length, config.num_heads],
			vec![config.batch_size, config.sequence_length, config.num_heads],
			vec![
				config.batch_size,
				config.sequence_length,
				config.num_rope_angles,
			],
			vec![config.num_heads, config.mimo_rank, config.state_size],
			vec![config.num_heads, config.mimo_rank, config.state_size],
			vec![config.num_heads],
			vec![config.num_heads, config.mimo_rank, config.head_dim],
			vec![config.num_heads, config.mimo_rank, config.head_dim],
			vec![config.num_heads, config.mimo_rank, config.head_dim],
			vec![config.num_heads, config.head_dim],
		];
		if inputs
			.iter()
			.zip(expected.iter())
			.any(|(matrix, shape)| matrix.shape() != shape)
		{
			return Err(Error::invalid_argument(format!(
				"{operation} operands do not match the explicit MIMO configuration"
			)));
		}
		validate_f32_same_engine(operation, inputs)?;
		for shape in &expected {
			shader_u32(
				checked_count(shape, operation)?,
				"tensor element count",
				operation,
			)?;
		}
		let batch = shader_u32(config.batch_size, "batch size", operation)?;
		let length = shader_u32(config.sequence_length, "sequence length", operation)?;
		let heads = shader_u32(config.num_heads, "head count", operation)?;
		let groups = shader_u32(groups, "group count", operation)?;
		let head_dim = shader_u32(config.head_dim, "head dimension", operation)?;
		let state_size = shader_u32(config.state_size, "state size", operation)?;
		let angles = shader_u32(config.num_rope_angles, "rope angle count", operation)?;
		let rank = shader_u32(config.mimo_rank, "MIMO rank", operation)?;
		let batch_heads = batch.checked_mul(heads).ok_or_else(|| {
			Error::invalid_argument(format!("{operation} batch-head count exceeds u32"))
		})?;
		Ok(Self {
			batch,
			length,
			heads,
			groups,
			head_dim,
			state_size,
			angles,
			rank,
			batch_heads,
		})
	}

	fn output_shape(&self) -> Vec<usize> {
		vec![
			self.batch as usize,
			self.length as usize,
			self.heads as usize,
			self.head_dim as usize,
		]
	}

	fn push_constants(&self, config: SsmConfig) -> [PushConstant; 12] {
		[
			PushConstant::U32(self.batch),
			PushConstant::U32(self.length),
			PushConstant::U32(self.heads),
			PushConstant::U32(self.groups),
			PushConstant::U32(self.head_dim),
			PushConstant::U32(self.state_size),
			PushConstant::U32(self.angles),
			PushConstant::U32(self.rank),
			PushConstant::U32(u32::from(config.has_z)),
			PushConstant::U32(u32::from(config.has_d)),
			PushConstant::U32(u32::from(config.has_output_norm)),
			PushConstant::F32(1.0e-5),
		]
	}
}

fn checked_count(shape: &[usize], operation: &'static str) -> Result<usize> {
	shape.iter().try_fold(1_usize, |count, extent| {
		count
			.checked_mul(*extent)
			.ok_or_else(|| Error::invalid_argument(format!("{operation} tensor size overflows usize")))
	})
}

struct PreprocessGeometry {
	rows: usize,
	rows_u32: u32,
	inner_size: usize,
	inner_size_u32: u32,
	state_size_u32: u32,
	heads: usize,
	heads_u32: u32,
	rope_angles: usize,
	rope_angles_u32: u32,
	bc_rows_u32: u32,
	bc_width: usize,
}

impl PreprocessGeometry {
	fn resolve(
		projected: &Matrix,
		dt_bias: &Matrix,
		config: Mamba3PreprocessConfig,
		operation: &'static str,
	) -> Result<Self> {
		let [rows, projected_width] = projected.shape() else {
			return Err(Error::invalid_argument(format!(
				"{operation} projected input must have shape [rows, width]"
			)));
		};
		if *rows == 0
			|| config.inner_size == 0
			|| config.state_size == 0
			|| config.num_heads == 0
			|| config.num_groups == 0
			|| config.mimo_rank == 0
			|| config.num_rope_angles.saturating_mul(2) > config.state_size
			|| !config.epsilon.is_finite()
			|| config.epsilon <= 0.0
			|| !config.dt_min.is_finite()
			|| !config.dt_max.is_finite()
			|| config.dt_min <= 0.0
			|| config.dt_min > config.dt_max
			|| !config.a_floor.is_finite()
			|| config.a_floor <= 0.0
		{
			return Err(Error::invalid_argument(format!(
				"{operation} requires positive geometry, 2*rope_angles <= state_size, finite epsilon/dt/A bounds, epsilon/dt_min/A floor > 0, and dt_min <= dt_max"
			)));
		}
		let bc_rows = config
			.num_groups
			.checked_mul(config.mimo_rank)
			.ok_or_else(|| Error::invalid_argument(format!("{operation} B/C row count overflows")))?;
		let bc_width = bc_rows
			.checked_mul(config.state_size)
			.ok_or_else(|| Error::invalid_argument(format!("{operation} B/C width overflows")))?;
		let expected_width = config
			.inner_size
			.checked_mul(2)
			.and_then(|value| bc_width.checked_mul(2).and_then(|bc| value.checked_add(bc)))
			.and_then(|value| {
				config
					.num_heads
					.checked_mul(3)
					.and_then(|heads| value.checked_add(heads))
			})
			.and_then(|value| value.checked_add(config.num_rope_angles))
			.ok_or_else(|| {
				Error::invalid_argument(format!("{operation} projected width overflows usize"))
			})?;
		if *projected_width != expected_width || dt_bias.shape() != [config.num_heads] {
			return Err(Error::invalid_argument(format!(
				"{operation} projected width or dt-bias shape does not match the configuration"
			)));
		}
		validate_f32_same_engine(operation, &[projected, dt_bias])?;
		shader_u32(expected_width, "projected row width", operation)?;
		shader_u32(bc_width, "B/C width", operation)?;
		for (label, shape) in [
			("projected element count", projected.shape()),
			("vector output element count", &[*rows, config.inner_size]),
			("B/C output element count", &[*rows, bc_width]),
			("scalar output element count", &[*rows, config.num_heads]),
			(
				"angle output element count",
				&[*rows, config.num_rope_angles],
			),
		] {
			shader_u32(checked_count(shape, operation)?, label, operation)?;
		}
		Ok(Self {
			rows: *rows,
			rows_u32: shader_u32(*rows, "row count", operation)?,
			inner_size: config.inner_size,
			inner_size_u32: shader_u32(config.inner_size, "inner size", operation)?,
			state_size_u32: shader_u32(config.state_size, "state size", operation)?,
			heads: config.num_heads,
			heads_u32: shader_u32(config.num_heads, "head count", operation)?,
			rope_angles: config.num_rope_angles,
			rope_angles_u32: shader_u32(config.num_rope_angles, "rope angle count", operation)?,
			bc_rows_u32: shader_u32(bc_rows, "B/C row count", operation)?,
			bc_width,
		})
	}

	fn output_shapes(&self) -> [Vec<usize>; 4] {
		[
			vec![self.rows, self.inner_size],
			vec![self.rows, self.bc_width],
			vec![self.rows, self.heads],
			vec![self.rows, self.rope_angles],
		]
	}

	fn scalar_push_constants(&self, config: Mamba3PreprocessConfig) -> [PushConstant; 10] {
		[
			PushConstant::U32(self.rows_u32),
			PushConstant::U32(self.inner_size_u32),
			PushConstant::U32(self.state_size_u32),
			PushConstant::U32(self.heads_u32),
			PushConstant::U32(self.rope_angles_u32),
			PushConstant::U32(self.bc_rows_u32),
			PushConstant::F32(config.epsilon),
			PushConstant::F32(config.dt_min),
			PushConstant::F32(config.dt_max),
			PushConstant::F32(config.a_floor),
		]
	}
}

pub(in crate::ml) fn mamba3_preprocess(
	projected: &Matrix,
	dt_bias: &Matrix,
	config: Mamba3PreprocessConfig,
) -> Result<Mamba3PreprocessResult> {
	let contract = crate::core::operation::ml::MAMBA3_PREPROCESS;
	let operation = contract.name();
	let geometry = PreprocessGeometry::resolve(projected, dt_bias, config, operation)?;
	let [vector_shape, bc_shape, scalar_shape, angle_shape] = geometry.output_shapes();
	let engine = projected.engine_handle();
	let allocate = |shape: Vec<usize>| {
		let count = checked_count(&shape, operation)?;
		shader_u32(count, "output element count", operation)?;
		Matrix::allocate(engine, shape, count, DType::F32)
	};
	let result = Mamba3PreprocessResult {
		x: allocate(vector_shape.clone())?,
		z: allocate(vector_shape)?,
		bh: allocate(bc_shape.clone())?,
		ch: allocate(bc_shape)?,
		dt: allocate(scalar_shape.clone())?,
		adt: allocate(scalar_shape.clone())?,
		trap: allocate(scalar_shape)?,
		angle: allocate(angle_shape)?,
	};
	let buffers = [
		BufferBinding::read(projected.storage()),
		BufferBinding::read(dt_bias.storage()),
		BufferBinding::write(result.z.storage()),
		BufferBinding::write(result.x.storage()),
		BufferBinding::write(result.bh.storage()),
		BufferBinding::write(result.ch.storage()),
		BufferBinding::write(result.dt.storage()),
		BufferBinding::write(result.adt.storage()),
		BufferBinding::write(result.trap.storage()),
		BufferBinding::write(result.angle.storage()),
	];
	let push_constants = geometry.scalar_push_constants(config);
	let attributes = [OpAttribute::UnsignedInteger {
		name: "config_identity".into(),
		value: config.identity(operation)?,
	}];
	let inputs = [projected, dt_bias];
	let outputs = result.matrices();
	engine.record_semantic(
		ComputeDispatch {
			kernel: KernelId::MlMamba3PreprocessF32,
			buffers: &buffers,
			push_constants: &push_constants,
			workgroups: [geometry.rows_u32, 1, 1],
		},
		SemanticDispatch {
			contract,
			inputs: &inputs,
			outputs: &outputs,
			attributes: &attributes,
		},
	)?;
	Ok(result)
}

pub(in crate::ml) fn mamba3_preprocess_backward(
	projected: &Matrix,
	dt_bias: &Matrix,
	output_gradients: &Mamba3PreprocessResult,
	config: Mamba3PreprocessConfig,
) -> Result<Mamba3PreprocessBackward> {
	let contract = crate::core::operation::ml::MAMBA3_PREPROCESS_BACKWARD;
	let operation = contract.name();
	let geometry = PreprocessGeometry::resolve(projected, dt_bias, config, operation)?;
	let [vector_shape, bc_shape, scalar_shape, angle_shape] = geometry.output_shapes();
	let expected = [
		vector_shape.as_slice(),
		vector_shape.as_slice(),
		bc_shape.as_slice(),
		bc_shape.as_slice(),
		scalar_shape.as_slice(),
		scalar_shape.as_slice(),
		scalar_shape.as_slice(),
		angle_shape.as_slice(),
	];
	let gradient_matrices = output_gradients.matrices();
	if gradient_matrices
		.iter()
		.zip(expected)
		.any(|(matrix, shape)| matrix.shape() != shape)
	{
		return Err(Error::invalid_argument(format!(
			"{operation} output gradients do not match the configured outputs"
		)));
	}
	let mut all_inputs = Vec::with_capacity(10);
	all_inputs.extend([projected, dt_bias]);
	all_inputs.extend(gradient_matrices);
	validate_f32_same_engine(operation, &all_inputs)?;

	let projected_gradient = Matrix::allocate(
		projected.engine_handle(),
		projected.shape().to_vec(),
		projected.element_count(),
		DType::F32,
	)?;
	let partial_count = geometry
		.rows
		.checked_mul(geometry.heads)
		.ok_or_else(|| Error::invalid_argument(format!("{operation} partial size overflows")))?;
	let dt_bias_partial = Matrix::allocate(
		projected.engine_handle(),
		vec![geometry.rows, geometry.heads],
		partial_count,
		DType::F32,
	)?;
	let dt_bias_gradient = Matrix::allocate(
		projected.engine_handle(),
		vec![geometry.heads],
		geometry.heads,
		DType::F32,
	)?;
	let buffers = [
		BufferBinding::read(projected.storage()),
		BufferBinding::read(dt_bias.storage()),
		BufferBinding::read(output_gradients.z.storage()),
		BufferBinding::read(output_gradients.x.storage()),
		BufferBinding::read(output_gradients.bh.storage()),
		BufferBinding::read(output_gradients.ch.storage()),
		BufferBinding::read(output_gradients.dt.storage()),
		BufferBinding::read(output_gradients.adt.storage()),
		BufferBinding::read(output_gradients.trap.storage()),
		BufferBinding::read(output_gradients.angle.storage()),
		BufferBinding::write(projected_gradient.storage()),
		BufferBinding::write(dt_bias_partial.storage()),
	];
	let push_constants = geometry.scalar_push_constants(config);
	let reduce_buffers = [
		BufferBinding::read(dt_bias_partial.storage()),
		BufferBinding::write(dt_bias_gradient.storage()),
	];
	let reduce_push_constants = [
		PushConstant::U32(geometry.rows_u32),
		PushConstant::U32(geometry.heads_u32),
	];
	let dispatches = [
		ComputeDispatch {
			kernel: KernelId::MlMamba3PreprocessBackwardF32,
			buffers: &buffers,
			push_constants: &push_constants,
			workgroups: [geometry.rows_u32, 1, 1],
		},
		ComputeDispatch {
			kernel: KernelId::MlMamba3PreprocessBackwardReduceF32,
			buffers: &reduce_buffers,
			push_constants: &reduce_push_constants,
			workgroups: [geometry.heads_u32.div_ceil(256), 1, 1],
		},
	];
	let attributes = [OpAttribute::UnsignedInteger {
		name: "config_identity".into(),
		value: config.identity(operation)?,
	}];
	let outputs = [&projected_gradient, &dt_bias_gradient];
	projected.engine_handle().record_split_semantic(
		&dispatches,
		SemanticDispatch {
			contract,
			inputs: &all_inputs,
			outputs: &outputs,
			attributes: &attributes,
		},
	)?;
	Ok(Mamba3PreprocessBackward {
		projected: projected_gradient,
		dt_bias: dt_bias_gradient,
	})
}

#[allow(
	clippy::too_many_arguments,
	reason = "preserves the donor eleven-operand selective-scan boundary"
)]
pub(in crate::ml) fn mamba3_siso(
	c: &Matrix,
	b: &Matrix,
	x: &Matrix,
	z: &Matrix,
	adt: &Matrix,
	dt: &Matrix,
	trap: &Matrix,
	angle: &Matrix,
	c_bias: &Matrix,
	b_bias: &Matrix,
	d: &Matrix,
	config: SsmConfig,
) -> Result<Matrix> {
	let contract = crate::core::operation::ml::MAMBA3_SISO;
	let operation = contract.name();
	let inputs = [c, b, x, z, adt, dt, trap, angle, c_bias, b_bias, d];
	let geometry = Geometry::resolve(&inputs, config, operation)?;
	let output_shape = geometry.output_shape();
	let output = Matrix::allocate(
		x.engine_handle(),
		output_shape.clone(),
		checked_count(&output_shape, operation)?,
		DType::F32,
	)?;
	let buffers = [
		BufferBinding::read(c.storage()),
		BufferBinding::read(b.storage()),
		BufferBinding::read(x.storage()),
		BufferBinding::read(z.storage()),
		BufferBinding::read(adt.storage()),
		BufferBinding::read(dt.storage()),
		BufferBinding::read(trap.storage()),
		BufferBinding::read(angle.storage()),
		BufferBinding::read(c_bias.storage()),
		BufferBinding::read(b_bias.storage()),
		BufferBinding::read(d.storage()),
		BufferBinding::write(output.storage()),
	];
	let push_constants = [
		PushConstant::U32(geometry.batch),
		PushConstant::U32(geometry.length),
		PushConstant::U32(geometry.heads),
		PushConstant::U32(geometry.groups),
		PushConstant::U32(geometry.head_dim),
		PushConstant::U32(geometry.state_size),
		PushConstant::U32(geometry.angles),
		PushConstant::U32(u32::from(config.has_z)),
		PushConstant::U32(u32::from(config.has_d)),
	];
	let attributes = [OpAttribute::UnsignedInteger {
		name: "config_identity".into(),
		value: config.identity(operation)?,
	}];
	let outputs = [&output];
	x.engine_handle().record_semantic(
		ComputeDispatch {
			kernel: KernelId::MlMamba3SisoF32,
			buffers: &buffers,
			push_constants: &push_constants,
			workgroups: [geometry.batch_heads, 1, 1],
		},
		SemanticDispatch {
			contract,
			inputs: &inputs,
			outputs: &outputs,
			attributes: &attributes,
		},
	)?;
	Ok(output)
}

#[allow(
	clippy::too_many_arguments,
	reason = "preserves the donor eleven operands plus four explicit recurrent-state buffers"
)]
pub(in crate::ml) fn mamba3_siso_step(
	c: &Matrix,
	b: &Matrix,
	x: &Matrix,
	z: &Matrix,
	adt: &Matrix,
	dt: &Matrix,
	trap: &Matrix,
	angle: &Matrix,
	c_bias: &Matrix,
	b_bias: &Matrix,
	d: &Matrix,
	ssm_state: &Matrix,
	angle_state: &Matrix,
	k_state: &Matrix,
	v_state: &Matrix,
	config: SsmConfig,
) -> Result<Matrix> {
	let contract = crate::core::operation::ml::MAMBA3_SISO_STEP;
	let operation = contract.name();
	let operands = [c, b, x, z, adt, dt, trap, angle, c_bias, b_bias, d];
	let geometry = Geometry::resolve(&operands, config, operation)?;
	if geometry.length != 1 {
		return Err(Error::invalid_argument(format!(
			"{operation} requires sequence_length == 1"
		)));
	}
	let expected_states = [
		vec![
			geometry.batch as usize,
			geometry.heads as usize,
			geometry.head_dim as usize,
			geometry.state_size as usize,
		],
		vec![
			geometry.batch as usize,
			geometry.heads as usize,
			geometry.angles as usize,
		],
		vec![
			geometry.batch as usize,
			geometry.heads as usize,
			geometry.state_size as usize,
		],
		vec![
			geometry.batch as usize,
			geometry.heads as usize,
			geometry.head_dim as usize,
		],
	];
	let states = [ssm_state, angle_state, k_state, v_state];
	if states
		.iter()
		.zip(expected_states.iter())
		.any(|(state, shape)| state.shape() != shape)
	{
		return Err(Error::invalid_argument(format!(
			"{operation} recurrent-state shapes do not match the explicit configuration"
		)));
	}
	validate_f32_same_engine(operation, &[x, ssm_state, angle_state, k_state, v_state])?;
	for shape in &expected_states {
		shader_u32(
			checked_count(shape, operation)?,
			"state element count",
			operation,
		)?;
	}

	let output_shape = geometry.output_shape();
	let output = Matrix::allocate(
		x.engine_handle(),
		output_shape.clone(),
		checked_count(&output_shape, operation)?,
		DType::F32,
	)?;
	let buffers = [
		BufferBinding::read(c.storage()),
		BufferBinding::read(b.storage()),
		BufferBinding::read(x.storage()),
		BufferBinding::read(z.storage()),
		BufferBinding::read(adt.storage()),
		BufferBinding::read(dt.storage()),
		BufferBinding::read(trap.storage()),
		BufferBinding::read(angle.storage()),
		BufferBinding::read(c_bias.storage()),
		BufferBinding::read(b_bias.storage()),
		BufferBinding::read(d.storage()),
		BufferBinding::write(output.storage()),
		BufferBinding::read_write(ssm_state.storage()),
		BufferBinding::read_write(angle_state.storage()),
		BufferBinding::read_write(k_state.storage()),
		BufferBinding::read_write(v_state.storage()),
	];
	let push_constants = [
		PushConstant::U32(geometry.batch),
		PushConstant::U32(geometry.heads),
		PushConstant::U32(geometry.groups),
		PushConstant::U32(geometry.head_dim),
		PushConstant::U32(geometry.state_size),
		PushConstant::U32(geometry.angles),
		PushConstant::U32(u32::from(config.has_z)),
		PushConstant::U32(u32::from(config.has_d)),
	];
	let attributes = [OpAttribute::UnsignedInteger {
		name: "config_identity".into(),
		value: config.identity(operation)?,
	}];
	let semantic_inputs = [
		ssm_state,
		angle_state,
		k_state,
		v_state,
		c,
		b,
		x,
		z,
		adt,
		dt,
		trap,
		angle,
		c_bias,
		b_bias,
		d,
	];
	let semantic_outputs = [&output, ssm_state, angle_state, k_state, v_state];
	x.engine_handle().record_semantic(
		ComputeDispatch {
			kernel: KernelId::MlMamba3SisoStepF32,
			buffers: &buffers,
			push_constants: &push_constants,
			workgroups: [geometry.batch_heads, 1, 1],
		},
		SemanticDispatch {
			contract,
			inputs: &semantic_inputs,
			outputs: &semantic_outputs,
			attributes: &attributes,
		},
	)?;
	Ok(output)
}

#[allow(
	clippy::too_many_arguments,
	reason = "preserves the donor fifteen-operand MIMO selective-scan boundary"
)]
pub(in crate::ml) fn mamba3_mimo(
	c: &Matrix,
	b: &Matrix,
	x: &Matrix,
	z: &Matrix,
	adt: &Matrix,
	dt: &Matrix,
	trap: &Matrix,
	angle: &Matrix,
	c_bias: &Matrix,
	b_bias: &Matrix,
	d: &Matrix,
	mimo_x: &Matrix,
	mimo_z: &Matrix,
	mimo_o: &Matrix,
	norm_weight: &Matrix,
	config: SsmConfig,
) -> Result<Matrix> {
	let contract = crate::core::operation::ml::MAMBA3_MIMO;
	let operation = contract.name();
	let inputs = [
		c,
		b,
		x,
		z,
		adt,
		dt,
		trap,
		angle,
		c_bias,
		b_bias,
		d,
		mimo_x,
		mimo_z,
		mimo_o,
		norm_weight,
	];
	let geometry = MimoGeometry::resolve(&inputs, config, operation)?;
	let output_shape = geometry.output_shape();
	let output = Matrix::allocate(
		x.engine_handle(),
		output_shape.clone(),
		checked_count(&output_shape, operation)?,
		DType::F32,
	)?;
	let buffers = [
		BufferBinding::read(c.storage()),
		BufferBinding::read(b.storage()),
		BufferBinding::read(x.storage()),
		BufferBinding::read(z.storage()),
		BufferBinding::read(adt.storage()),
		BufferBinding::read(dt.storage()),
		BufferBinding::read(trap.storage()),
		BufferBinding::read(angle.storage()),
		BufferBinding::read(c_bias.storage()),
		BufferBinding::read(b_bias.storage()),
		BufferBinding::read(d.storage()),
		BufferBinding::read(mimo_x.storage()),
		BufferBinding::read(mimo_z.storage()),
		BufferBinding::read(mimo_o.storage()),
		BufferBinding::read(norm_weight.storage()),
		BufferBinding::write(output.storage()),
	];
	let push_constants = geometry.push_constants(config);
	let attributes = [OpAttribute::UnsignedInteger {
		name: "config_identity".into(),
		value: config.identity(operation)?,
	}];
	let outputs = [&output];
	x.engine_handle().record_semantic(
		ComputeDispatch {
			kernel: KernelId::MlMamba3MimoF32,
			buffers: &buffers,
			push_constants: &push_constants,
			workgroups: [geometry.batch_heads, 1, 1],
		},
		SemanticDispatch {
			contract,
			inputs: &inputs,
			outputs: &outputs,
			attributes: &attributes,
		},
	)?;
	Ok(output)
}

#[allow(
	clippy::too_many_arguments,
	reason = "preserves the donor fifteen operands plus four explicit recurrent-state buffers"
)]
pub(in crate::ml) fn mamba3_mimo_step(
	c: &Matrix,
	b: &Matrix,
	x: &Matrix,
	z: &Matrix,
	adt: &Matrix,
	dt: &Matrix,
	trap: &Matrix,
	angle: &Matrix,
	c_bias: &Matrix,
	b_bias: &Matrix,
	d: &Matrix,
	mimo_x: &Matrix,
	mimo_z: &Matrix,
	mimo_o: &Matrix,
	norm_weight: &Matrix,
	ssm_state: &Matrix,
	angle_state: &Matrix,
	k_state: &Matrix,
	v_state: &Matrix,
	config: SsmConfig,
) -> Result<Matrix> {
	let contract = crate::core::operation::ml::MAMBA3_MIMO_STEP;
	let operation = contract.name();
	let operands = [
		c,
		b,
		x,
		z,
		adt,
		dt,
		trap,
		angle,
		c_bias,
		b_bias,
		d,
		mimo_x,
		mimo_z,
		mimo_o,
		norm_weight,
	];
	let geometry = MimoGeometry::resolve(&operands, config, operation)?;
	if geometry.length != 1 {
		return Err(Error::invalid_argument(format!(
			"{operation} requires sequence_length == 1"
		)));
	}
	let expected_states = [
		vec![
			geometry.batch as usize,
			geometry.heads as usize,
			geometry.head_dim as usize,
			geometry.state_size as usize,
		],
		vec![
			geometry.batch as usize,
			geometry.heads as usize,
			geometry.angles as usize,
		],
		vec![
			geometry.batch as usize,
			geometry.heads as usize,
			geometry.rank as usize,
			geometry.state_size as usize,
		],
		vec![
			geometry.batch as usize,
			geometry.heads as usize,
			geometry.rank as usize,
			geometry.head_dim as usize,
		],
	];
	let states = [ssm_state, angle_state, k_state, v_state];
	if states
		.iter()
		.zip(expected_states.iter())
		.any(|(state, shape)| state.shape() != shape)
	{
		return Err(Error::invalid_argument(format!(
			"{operation} recurrent-state shapes do not match the explicit MIMO configuration"
		)));
	}
	validate_f32_same_engine(operation, &[x, ssm_state, angle_state, k_state, v_state])?;
	for shape in &expected_states {
		shader_u32(
			checked_count(shape, operation)?,
			"state element count",
			operation,
		)?;
	}
	let output_shape = geometry.output_shape();
	let output = Matrix::allocate(
		x.engine_handle(),
		output_shape.clone(),
		checked_count(&output_shape, operation)?,
		DType::F32,
	)?;
	let buffers = [
		BufferBinding::read(c.storage()),
		BufferBinding::read(b.storage()),
		BufferBinding::read(x.storage()),
		BufferBinding::read(z.storage()),
		BufferBinding::read(adt.storage()),
		BufferBinding::read(dt.storage()),
		BufferBinding::read(trap.storage()),
		BufferBinding::read(angle.storage()),
		BufferBinding::read(c_bias.storage()),
		BufferBinding::read(b_bias.storage()),
		BufferBinding::read(d.storage()),
		BufferBinding::read(mimo_x.storage()),
		BufferBinding::read(mimo_z.storage()),
		BufferBinding::read(mimo_o.storage()),
		BufferBinding::read(norm_weight.storage()),
		BufferBinding::write(output.storage()),
		BufferBinding::read_write(ssm_state.storage()),
		BufferBinding::read_write(angle_state.storage()),
		BufferBinding::read_write(k_state.storage()),
		BufferBinding::read_write(v_state.storage()),
	];
	let push_constants = [
		PushConstant::U32(geometry.batch),
		PushConstant::U32(geometry.heads),
		PushConstant::U32(geometry.groups),
		PushConstant::U32(geometry.head_dim),
		PushConstant::U32(geometry.state_size),
		PushConstant::U32(geometry.angles),
		PushConstant::U32(geometry.rank),
		PushConstant::U32(u32::from(config.has_z)),
		PushConstant::U32(u32::from(config.has_d)),
		PushConstant::U32(u32::from(config.has_output_norm)),
		PushConstant::F32(1.0e-5),
	];
	let attributes = [OpAttribute::UnsignedInteger {
		name: "config_identity".into(),
		value: config.identity(operation)?,
	}];
	let semantic_inputs = [
		ssm_state,
		angle_state,
		k_state,
		v_state,
		c,
		b,
		x,
		z,
		adt,
		dt,
		trap,
		angle,
		c_bias,
		b_bias,
		d,
		mimo_x,
		mimo_z,
		mimo_o,
		norm_weight,
	];
	let semantic_outputs = [&output, ssm_state, angle_state, k_state, v_state];
	x.engine_handle().record_semantic(
		ComputeDispatch {
			kernel: KernelId::MlMamba3MimoStepF32,
			buffers: &buffers,
			push_constants: &push_constants,
			workgroups: [geometry.batch_heads, 1, 1],
		},
		SemanticDispatch {
			contract,
			inputs: &semantic_inputs,
			outputs: &semantic_outputs,
			attributes: &attributes,
		},
	)?;
	Ok(output)
}

#[allow(
	clippy::too_many_arguments,
	reason = "preserves the donor upstream plus fifteen-operand MIMO adjoint boundary"
)]
pub(in crate::ml) fn mamba3_mimo_backward(
	output_gradient: &Matrix,
	c: &Matrix,
	b: &Matrix,
	x: &Matrix,
	z: &Matrix,
	adt: &Matrix,
	dt: &Matrix,
	trap: &Matrix,
	angle: &Matrix,
	c_bias: &Matrix,
	b_bias: &Matrix,
	d: &Matrix,
	mimo_x: &Matrix,
	mimo_z: &Matrix,
	mimo_o: &Matrix,
	norm_weight: &Matrix,
	config: SsmConfig,
) -> Result<Mamba3MimoBackward> {
	let contract = crate::core::operation::ml::MAMBA3_MIMO_BACKWARD;
	let operation = contract.name();
	let operands = [
		c,
		b,
		x,
		z,
		adt,
		dt,
		trap,
		angle,
		c_bias,
		b_bias,
		d,
		mimo_x,
		mimo_z,
		mimo_o,
		norm_weight,
	];
	let geometry = MimoGeometry::resolve(&operands, config, operation)?;
	if output_gradient.shape() != geometry.output_shape() {
		return Err(Error::invalid_argument(format!(
			"{operation} output gradient does not match the configured output"
		)));
	}
	validate_f32_same_engine(operation, &[output_gradient, x])?;

	let engine = x.engine_handle();
	let alloc = |shape: Vec<usize>| -> Result<Matrix> {
		let count = checked_count(&shape, operation)?;
		shader_u32(count, "temporary element count", operation)?;
		Matrix::allocate(engine, shape, count, DType::F32)
	};
	let batch = geometry.batch as usize;
	let length = geometry.length as usize;
	let heads = geometry.heads as usize;
	let groups = geometry.groups as usize;
	let head_dim = geometry.head_dim as usize;
	let state_size = geometry.state_size as usize;
	let angles = geometry.angles as usize;
	let rank = geometry.rank as usize;
	let scalar_shape = vec![batch, length, heads];
	let vector_shape = vec![batch, length, heads, head_dim];
	let qk_shape = vec![batch, length, heads, rank, state_size];

	let q = alloc(qk_shape.clone())?;
	let k = alloc(qk_shape.clone())?;
	let theta = alloc(vec![batch, heads, length, angles])?;
	let h_previous = alloc(vec![batch * heads, length, head_dim, state_size])?;
	let decay = alloc(scalar_shape.clone())?;
	let gamma = alloc(scalar_shape.clone())?;
	let scale = alloc(scalar_shape.clone())?;

	let d_pre = alloc(vec![batch, length, heads, rank, head_dim])?;
	let d_z = alloc(vector_shape.clone())?;
	let d_mimo_z_batch = alloc(vec![batch, heads, rank, head_dim])?;
	let d_mimo_o_batch = alloc(vec![batch, heads, rank, head_dim])?;
	let d_norm_batch = alloc(vec![batch, heads, head_dim])?;

	let d_q = alloc(qk_shape.clone())?;
	let d_k = alloc(qk_shape.clone())?;
	let d_x = alloc(vector_shape)?;
	let d_adt = alloc(scalar_shape.clone())?;
	let d_dt = alloc(scalar_shape.clone())?;
	let d_trap = alloc(scalar_shape.clone())?;
	let d_d_batch = alloc(vec![batch, heads])?;
	let d_mimo_x_batch = alloc(vec![batch, heads, rank, head_dim])?;
	let d_scale_tmp = alloc(scalar_shape)?;

	let d_c_head = alloc(qk_shape.clone())?;
	let d_b_head = alloc(qk_shape)?;
	let d_angle_head = alloc(vec![batch, heads, length, angles])?;
	let d_c_bias_batch = alloc(vec![batch, heads, rank, state_size])?;
	let d_b_bias_batch = alloc(vec![batch, heads, rank, state_size])?;

	let d_c = alloc(vec![batch, length, rank * groups, state_size])?;
	let d_b = alloc(vec![batch, length, rank * groups, state_size])?;
	let d_angle = alloc(vec![batch, length, angles])?;
	let d_c_bias = alloc(vec![heads, rank, state_size])?;
	let d_b_bias = alloc(vec![heads, rank, state_size])?;
	let d_d = alloc(vec![heads])?;
	let d_mimo_x = alloc(vec![heads, rank, head_dim])?;
	let d_mimo_z = alloc(vec![heads, rank, head_dim])?;
	let d_mimo_o = alloc(vec![heads, rank, head_dim])?;
	let d_norm_weight = alloc(vec![heads, head_dim])?;

	let prepare_buffers = [
		BufferBinding::read(c.storage()),
		BufferBinding::read(b.storage()),
		BufferBinding::read(x.storage()),
		BufferBinding::read(adt.storage()),
		BufferBinding::read(dt.storage()),
		BufferBinding::read(trap.storage()),
		BufferBinding::read(angle.storage()),
		BufferBinding::read(c_bias.storage()),
		BufferBinding::read(b_bias.storage()),
		BufferBinding::read(mimo_x.storage()),
		BufferBinding::write(q.storage()),
		BufferBinding::write(k.storage()),
		BufferBinding::write(theta.storage()),
		BufferBinding::write(h_previous.storage()),
		BufferBinding::write(decay.storage()),
		BufferBinding::write(gamma.storage()),
		BufferBinding::write(scale.storage()),
	];
	let prepare_constants = [
		PushConstant::U32(geometry.batch),
		PushConstant::U32(geometry.length),
		PushConstant::U32(geometry.heads),
		PushConstant::U32(geometry.groups),
		PushConstant::U32(geometry.head_dim),
		PushConstant::U32(geometry.state_size),
		PushConstant::U32(geometry.angles),
		PushConstant::U32(geometry.rank),
	];

	let post_buffers = [
		BufferBinding::read(output_gradient.storage()),
		BufferBinding::read(x.storage()),
		BufferBinding::read(z.storage()),
		BufferBinding::read(d.storage()),
		BufferBinding::read(mimo_x.storage()),
		BufferBinding::read(mimo_z.storage()),
		BufferBinding::read(mimo_o.storage()),
		BufferBinding::read(norm_weight.storage()),
		BufferBinding::read(q.storage()),
		BufferBinding::read(k.storage()),
		BufferBinding::read(h_previous.storage()),
		BufferBinding::read(decay.storage()),
		BufferBinding::read(gamma.storage()),
		BufferBinding::write(d_pre.storage()),
		BufferBinding::write(d_z.storage()),
		BufferBinding::write(d_mimo_z_batch.storage()),
		BufferBinding::write(d_mimo_o_batch.storage()),
		BufferBinding::write(d_norm_batch.storage()),
	];
	let post_constants = [
		PushConstant::U32(geometry.batch),
		PushConstant::U32(geometry.length),
		PushConstant::U32(geometry.heads),
		PushConstant::U32(geometry.head_dim),
		PushConstant::U32(geometry.state_size),
		PushConstant::U32(geometry.rank),
		PushConstant::U32(u32::from(config.has_z)),
		PushConstant::U32(u32::from(config.has_d)),
		PushConstant::U32(u32::from(config.has_output_norm)),
		PushConstant::F32(1.0e-5),
	];

	let core_buffers = [
		BufferBinding::read(d_pre.storage()),
		BufferBinding::read(x.storage()),
		BufferBinding::read(d.storage()),
		BufferBinding::read(mimo_x.storage()),
		BufferBinding::read(q.storage()),
		BufferBinding::read(k.storage()),
		BufferBinding::read(h_previous.storage()),
		BufferBinding::read(decay.storage()),
		BufferBinding::read(gamma.storage()),
		BufferBinding::read(scale.storage()),
		BufferBinding::read(dt.storage()),
		BufferBinding::read(trap.storage()),
		BufferBinding::write(d_q.storage()),
		BufferBinding::write(d_k.storage()),
		BufferBinding::write(d_x.storage()),
		BufferBinding::write(d_adt.storage()),
		BufferBinding::write(d_dt.storage()),
		BufferBinding::write(d_trap.storage()),
		BufferBinding::write(d_d_batch.storage()),
		BufferBinding::write(d_mimo_x_batch.storage()),
		BufferBinding::write(d_scale_tmp.storage()),
	];
	let core_constants = [
		PushConstant::U32(geometry.batch),
		PushConstant::U32(geometry.length),
		PushConstant::U32(geometry.heads),
		PushConstant::U32(geometry.head_dim),
		PushConstant::U32(geometry.state_size),
		PushConstant::U32(geometry.rank),
		PushConstant::U32(u32::from(config.has_d)),
	];

	let rotate_buffers = [
		BufferBinding::read(d_q.storage()),
		BufferBinding::read(d_k.storage()),
		BufferBinding::read(c.storage()),
		BufferBinding::read(b.storage()),
		BufferBinding::read(dt.storage()),
		BufferBinding::read(angle.storage()),
		BufferBinding::read(c_bias.storage()),
		BufferBinding::read(b_bias.storage()),
		BufferBinding::read(theta.storage()),
		BufferBinding::write(d_c_head.storage()),
		BufferBinding::write(d_b_head.storage()),
		BufferBinding::write(d_angle_head.storage()),
		BufferBinding::write(d_c_bias_batch.storage()),
		BufferBinding::write(d_b_bias_batch.storage()),
		BufferBinding::read_write(d_dt.storage()),
	];
	let rotate_constants = [
		PushConstant::U32(geometry.batch),
		PushConstant::U32(geometry.length),
		PushConstant::U32(geometry.heads),
		PushConstant::U32(geometry.groups),
		PushConstant::U32(geometry.state_size),
		PushConstant::U32(geometry.angles),
		PushConstant::U32(geometry.rank),
	];

	let group_count = geometry
		.batch
		.checked_mul(geometry.length)
		.and_then(|value| value.checked_mul(geometry.rank))
		.and_then(|value| value.checked_mul(geometry.groups))
		.and_then(|value| value.checked_mul(geometry.state_size))
		.ok_or_else(|| Error::invalid_argument(format!("{operation} group reduction exceeds u32")))?;
	let angle_count = geometry
		.batch
		.checked_mul(geometry.length)
		.and_then(|value| value.checked_mul(geometry.angles))
		.ok_or_else(|| Error::invalid_argument(format!("{operation} angle reduction exceeds u32")))?;
	let bias_count = geometry
		.heads
		.checked_mul(geometry.rank)
		.and_then(|value| value.checked_mul(geometry.state_size))
		.ok_or_else(|| Error::invalid_argument(format!("{operation} bias reduction exceeds u32")))?;
	let mimo_count = geometry
		.heads
		.checked_mul(geometry.rank)
		.and_then(|value| value.checked_mul(geometry.head_dim))
		.ok_or_else(|| Error::invalid_argument(format!("{operation} MIMO reduction exceeds u32")))?;
	let norm_count = geometry
		.heads
		.checked_mul(geometry.head_dim)
		.ok_or_else(|| Error::invalid_argument(format!("{operation} norm reduction exceeds u32")))?;
	let reduce_count = group_count
		.checked_add(angle_count)
		.and_then(|value| value.checked_add(bias_count))
		.and_then(|value| value.checked_add(geometry.heads))
		.and_then(|value| value.checked_add(mimo_count))
		.and_then(|value| value.checked_add(norm_count))
		.ok_or_else(|| Error::invalid_argument(format!("{operation} reduction size exceeds u32")))?;
	let reduce_buffers = [
		BufferBinding::read(d_c_head.storage()),
		BufferBinding::read(d_b_head.storage()),
		BufferBinding::read(d_angle_head.storage()),
		BufferBinding::read(d_c_bias_batch.storage()),
		BufferBinding::read(d_b_bias_batch.storage()),
		BufferBinding::read(d_d_batch.storage()),
		BufferBinding::read(d_mimo_x_batch.storage()),
		BufferBinding::read(d_mimo_z_batch.storage()),
		BufferBinding::read(d_mimo_o_batch.storage()),
		BufferBinding::read(d_norm_batch.storage()),
		BufferBinding::write(d_c.storage()),
		BufferBinding::write(d_b.storage()),
		BufferBinding::write(d_angle.storage()),
		BufferBinding::write(d_c_bias.storage()),
		BufferBinding::write(d_b_bias.storage()),
		BufferBinding::write(d_d.storage()),
		BufferBinding::write(d_mimo_x.storage()),
		BufferBinding::write(d_mimo_z.storage()),
		BufferBinding::write(d_mimo_o.storage()),
		BufferBinding::write(d_norm_weight.storage()),
	];
	let reduce_constants = [
		PushConstant::U32(geometry.batch),
		PushConstant::U32(geometry.length),
		PushConstant::U32(geometry.heads),
		PushConstant::U32(geometry.groups),
		PushConstant::U32(geometry.head_dim),
		PushConstant::U32(geometry.state_size),
		PushConstant::U32(geometry.angles),
		PushConstant::U32(geometry.rank),
		PushConstant::U32(reduce_count),
	];

	let bh = geometry.batch_heads;
	let dispatches = [
		ComputeDispatch {
			kernel: KernelId::MlMamba3MimoBackwardPrepareF32,
			buffers: &prepare_buffers,
			push_constants: &prepare_constants,
			workgroups: [bh, 1, 1],
		},
		ComputeDispatch {
			kernel: KernelId::MlMamba3MimoBackwardPostF32,
			buffers: &post_buffers,
			push_constants: &post_constants,
			workgroups: [bh, 1, 1],
		},
		ComputeDispatch {
			kernel: KernelId::MlMamba3MimoBackwardCoreF32,
			buffers: &core_buffers,
			push_constants: &core_constants,
			workgroups: [bh, 1, 1],
		},
		ComputeDispatch {
			kernel: KernelId::MlMamba3MimoBackwardRotateF32,
			buffers: &rotate_buffers,
			push_constants: &rotate_constants,
			workgroups: [bh, 1, 1],
		},
		ComputeDispatch {
			kernel: KernelId::MlMamba3MimoBackwardReduceF32,
			buffers: &reduce_buffers,
			push_constants: &reduce_constants,
			workgroups: [reduce_count.div_ceil(256), 1, 1],
		},
	];
	let semantic_inputs = [
		output_gradient,
		c,
		b,
		x,
		z,
		adt,
		dt,
		trap,
		angle,
		c_bias,
		b_bias,
		d,
		mimo_x,
		mimo_z,
		mimo_o,
		norm_weight,
	];
	let semantic_outputs = [
		&d_c,
		&d_b,
		&d_x,
		&d_z,
		&d_adt,
		&d_dt,
		&d_trap,
		&d_angle,
		&d_c_bias,
		&d_b_bias,
		&d_d,
		&d_mimo_x,
		&d_mimo_z,
		&d_mimo_o,
		&d_norm_weight,
	];
	let attributes = [OpAttribute::UnsignedInteger {
		name: "config_identity".into(),
		value: config.identity(operation)?,
	}];
	engine.record_split_semantic(
		&dispatches,
		SemanticDispatch {
			contract,
			inputs: &semantic_inputs,
			outputs: &semantic_outputs,
			attributes: &attributes,
		},
	)?;
	Ok(Mamba3MimoBackward {
		c: d_c,
		b: d_b,
		x: d_x,
		z: d_z,
		adt: d_adt,
		dt: d_dt,
		trap: d_trap,
		angle: d_angle,
		c_bias: d_c_bias,
		b_bias: d_b_bias,
		d: d_d,
		mimo_x: d_mimo_x,
		mimo_z: d_mimo_z,
		mimo_o: d_mimo_o,
		norm_weight: d_norm_weight,
	})
}

#[allow(
	clippy::too_many_arguments,
	reason = "preserves the donor upstream plus eleven-operand adjoint boundary"
)]
pub(in crate::ml) fn mamba3_siso_backward(
	output_gradient: &Matrix,
	c: &Matrix,
	b: &Matrix,
	x: &Matrix,
	z: &Matrix,
	adt: &Matrix,
	dt: &Matrix,
	trap: &Matrix,
	angle: &Matrix,
	c_bias: &Matrix,
	b_bias: &Matrix,
	d: &Matrix,
	config: SsmConfig,
) -> Result<SsmBackward> {
	let contract = crate::core::operation::ml::MAMBA3_SISO_BACKWARD;
	let operation = contract.name();
	let operands = [c, b, x, z, adt, dt, trap, angle, c_bias, b_bias, d];
	let geometry = Geometry::resolve(&operands, config, operation)?;
	if output_gradient.shape() != geometry.output_shape() {
		return Err(Error::invalid_argument(format!(
			"{operation} output gradient does not match the configured output"
		)));
	}
	validate_f32_same_engine(operation, &[output_gradient, x])?;
	let engine = x.engine_handle();
	let short_route = geometry.length <= 16
		&& geometry.head_dim <= 16
		&& geometry.state_size <= 32
		&& geometry.angles <= 8;
	if !short_route {
		let history_bytes = checked_count(
			&[
				geometry.batch as usize,
				geometry.heads as usize,
				geometry.length as usize,
				geometry.head_dim as usize,
				geometry.state_size as usize,
			],
			operation,
		)?
		.checked_mul(std::mem::size_of::<f32>())
		.ok_or_else(|| {
			Error::invalid_argument(format!("{operation} history byte size overflows usize"))
		})?;
		if geometry.length >= 64
			&& geometry.head_dim <= 32
			&& geometry.state_size <= 32
			&& geometry.angles <= 8
			&& history_bytes <= MAMBA3_CHUNK_HISTORY_LIMIT_BYTES
		{
			return mamba3_siso_backward_chunked(
				output_gradient,
				c,
				b,
				x,
				z,
				adt,
				dt,
				trap,
				angle,
				c_bias,
				b_bias,
				d,
				config,
				&geometry,
			);
		}
		return mamba3_siso_backward_generic(
			output_gradient,
			c,
			b,
			x,
			z,
			adt,
			dt,
			trap,
			angle,
			c_bias,
			b_bias,
			d,
			config,
			&geometry,
		);
	}

	let alloc = |shape: Vec<usize>| -> Result<Matrix> {
		let count = checked_count(&shape, operation)?;
		shader_u32(count, "temporary element count", operation)?;
		Matrix::allocate(engine, shape, count, DType::F32)
	};
	let bsz = geometry.batch as usize;
	let len = geometry.length as usize;
	let heads = geometry.heads as usize;
	let groups = geometry.groups as usize;
	let head_dim = geometry.head_dim as usize;
	let state = geometry.state_size as usize;
	let angles = geometry.angles as usize;
	let scalar_shape = vec![bsz, len, heads];
	let vector_shape = vec![bsz, len, heads, head_dim];
	let head_state_shape = vec![bsz, len, heads, state];

	let c_rot = alloc(head_state_shape.clone())?;
	let b_rot = alloc(head_state_shape.clone())?;
	let theta = alloc(vec![bsz * heads, len, angles])?;
	let qk = alloc(scalar_shape.clone())?;
	let decay = alloc(scalar_shape.clone())?;
	let gamma = alloc(scalar_shape.clone())?;
	let scale = alloc(scalar_shape.clone())?;
	let h_previous = alloc(vec![bsz * heads, len, head_dim, state])?;
	let d_pre = alloc(vector_shape.clone())?;
	let d_z = alloc(vector_shape.clone())?;
	let d_c_head = alloc(head_state_shape.clone())?;
	let d_b_head = alloc(head_state_shape)?;
	let d_x = alloc(vector_shape)?;
	let d_adt = alloc(scalar_shape.clone())?;
	let d_gamma = alloc(scalar_shape.clone())?;
	let d_scale = alloc(scalar_shape.clone())?;
	let d_theta = alloc(vec![bsz * heads, len, angles])?;
	let d_c_bias_batch = alloc(vec![bsz, heads, state])?;
	let d_b_bias_batch = alloc(vec![bsz, heads, state])?;
	let d_d_batch = alloc(vec![bsz, heads])?;
	let d_dt = alloc(scalar_shape.clone())?;
	let d_trap = alloc(scalar_shape)?;
	let d_angle_head = alloc(vec![bsz, heads, len, angles])?;
	let d_angle = alloc(vec![bsz, len, angles])?;
	let d_c_bias = alloc(vec![heads, state])?;
	let d_b_bias = alloc(vec![heads, state])?;
	let d_d = alloc(vec![heads])?;
	let d_c = alloc(vec![bsz, len, groups, state])?;
	let d_b = alloc(vec![bsz, len, groups, state])?;

	let prepare_buffers = [
		BufferBinding::read(c.storage()),
		BufferBinding::read(b.storage()),
		BufferBinding::read(dt.storage()),
		BufferBinding::read(trap.storage()),
		BufferBinding::read(adt.storage()),
		BufferBinding::read(angle.storage()),
		BufferBinding::read(c_bias.storage()),
		BufferBinding::read(b_bias.storage()),
		BufferBinding::write(c_rot.storage()),
		BufferBinding::write(b_rot.storage()),
		BufferBinding::write(theta.storage()),
		BufferBinding::write(qk.storage()),
		BufferBinding::write(decay.storage()),
		BufferBinding::write(gamma.storage()),
		BufferBinding::write(scale.storage()),
	];
	let prepare_constants = [
		PushConstant::U32(geometry.batch),
		PushConstant::U32(geometry.length),
		PushConstant::U32(geometry.heads),
		PushConstant::U32(geometry.groups),
		PushConstant::U32(geometry.state_size),
		PushConstant::U32(geometry.angles),
	];
	let state_buffers = [
		BufferBinding::read(b_rot.storage()),
		BufferBinding::read(x.storage()),
		BufferBinding::read(decay.storage()),
		BufferBinding::read(scale.storage()),
		BufferBinding::read(c_rot.storage()),
		BufferBinding::read(z.storage()),
		BufferBinding::read(gamma.storage()),
		BufferBinding::read(d.storage()),
		BufferBinding::read(qk.storage()),
		BufferBinding::read(output_gradient.storage()),
		BufferBinding::write(h_previous.storage()),
		BufferBinding::write(d_pre.storage()),
		BufferBinding::write(d_z.storage()),
	];
	let state_constants = [
		PushConstant::U32(geometry.batch),
		PushConstant::U32(geometry.length),
		PushConstant::U32(geometry.heads),
		PushConstant::U32(geometry.head_dim),
		PushConstant::U32(geometry.state_size),
		PushConstant::U32(u32::from(config.has_z)),
		PushConstant::U32(u32::from(config.has_d)),
	];
	let reverse_buffers = [
		BufferBinding::read(d_pre.storage()),
		BufferBinding::read(c_rot.storage()),
		BufferBinding::read(b_rot.storage()),
		BufferBinding::read(theta.storage()),
		BufferBinding::read(gamma.storage()),
		BufferBinding::read(scale.storage()),
		BufferBinding::read(decay.storage()),
		BufferBinding::read(h_previous.storage()),
		BufferBinding::read(x.storage()),
		BufferBinding::read(d.storage()),
		BufferBinding::read(qk.storage()),
		BufferBinding::write(d_c_head.storage()),
		BufferBinding::write(d_b_head.storage()),
		BufferBinding::write(d_x.storage()),
		BufferBinding::write(d_adt.storage()),
		BufferBinding::write(d_gamma.storage()),
		BufferBinding::write(d_scale.storage()),
		BufferBinding::write(d_theta.storage()),
		BufferBinding::write(d_c_bias_batch.storage()),
		BufferBinding::write(d_b_bias_batch.storage()),
		BufferBinding::write(d_d_batch.storage()),
	];
	let reverse_constants = [
		PushConstant::U32(geometry.batch),
		PushConstant::U32(geometry.length),
		PushConstant::U32(geometry.heads),
		PushConstant::U32(geometry.head_dim),
		PushConstant::U32(geometry.state_size),
		PushConstant::U32(geometry.angles),
		PushConstant::U32(u32::from(config.has_d)),
	];
	let finalize_buffers = [
		BufferBinding::read(dt.storage()),
		BufferBinding::read(trap.storage()),
		BufferBinding::read(angle.storage()),
		BufferBinding::read(d_gamma.storage()),
		BufferBinding::read(d_scale.storage()),
		BufferBinding::read(d_theta.storage()),
		BufferBinding::write(d_dt.storage()),
		BufferBinding::write(d_trap.storage()),
		BufferBinding::write(d_angle_head.storage()),
	];
	let finalize_constants = [
		PushConstant::U32(geometry.batch),
		PushConstant::U32(geometry.length),
		PushConstant::U32(geometry.heads),
		PushConstant::U32(geometry.angles),
	];
	let reduce_count = geometry
		.batch
		.checked_mul(geometry.length)
		.and_then(|value| value.checked_mul(geometry.angles))
		.and_then(|value| {
			geometry
				.heads
				.checked_mul(geometry.state_size)
				.and_then(|hn| hn.checked_mul(2))
				.and_then(|twice_hn| value.checked_add(twice_hn))
		})
		.and_then(|value| value.checked_add(geometry.heads))
		.ok_or_else(|| Error::invalid_argument(format!("{operation} reduction size exceeds u32")))?;
	let reduce_buffers = [
		BufferBinding::read(d_angle_head.storage()),
		BufferBinding::read(d_c_bias_batch.storage()),
		BufferBinding::read(d_b_bias_batch.storage()),
		BufferBinding::read(d_d_batch.storage()),
		BufferBinding::write(d_angle.storage()),
		BufferBinding::write(d_c_bias.storage()),
		BufferBinding::write(d_b_bias.storage()),
		BufferBinding::write(d_d.storage()),
	];
	let reduce_constants = [
		PushConstant::U32(geometry.batch),
		PushConstant::U32(geometry.length),
		PushConstant::U32(geometry.heads),
		PushConstant::U32(geometry.state_size),
		PushConstant::U32(geometry.angles),
		PushConstant::U32(reduce_count),
	];
	let group_count = geometry
		.batch
		.checked_mul(geometry.length)
		.and_then(|value| value.checked_mul(geometry.groups))
		.and_then(|value| value.checked_mul(geometry.state_size))
		.ok_or_else(|| Error::invalid_argument(format!("{operation} group reduction exceeds u32")))?;
	let group_buffers = [
		BufferBinding::read(d_c_head.storage()),
		BufferBinding::read(d_b_head.storage()),
		BufferBinding::write(d_c.storage()),
		BufferBinding::write(d_b.storage()),
	];
	let group_constants = [
		PushConstant::U32(geometry.batch),
		PushConstant::U32(geometry.length),
		PushConstant::U32(geometry.heads),
		PushConstant::U32(geometry.groups),
		PushConstant::U32(geometry.state_size),
		PushConstant::U32(group_count),
	];
	let bh = geometry.batch_heads;
	let dispatches = [
		ComputeDispatch {
			kernel: KernelId::MlMamba3SisoBackwardPrepareF32,
			buffers: &prepare_buffers,
			push_constants: &prepare_constants,
			workgroups: [bh.div_ceil(64), 1, 1],
		},
		ComputeDispatch {
			kernel: KernelId::MlMamba3SisoBackwardStateF32,
			buffers: &state_buffers,
			push_constants: &state_constants,
			workgroups: [bh, 1, 1],
		},
		ComputeDispatch {
			kernel: KernelId::MlMamba3SisoBackwardReverseP16F32,
			buffers: &reverse_buffers,
			push_constants: &reverse_constants,
			workgroups: [bh, 1, 1],
		},
		ComputeDispatch {
			kernel: KernelId::MlMamba3SisoBackwardFinalizeF32,
			buffers: &finalize_buffers,
			push_constants: &finalize_constants,
			workgroups: [bh.div_ceil(64), 1, 1],
		},
		ComputeDispatch {
			kernel: KernelId::MlMamba3SisoBackwardReduceF32,
			buffers: &reduce_buffers,
			push_constants: &reduce_constants,
			workgroups: [reduce_count.div_ceil(256), 1, 1],
		},
		ComputeDispatch {
			kernel: KernelId::MlMamba3SisoBackwardGroupReduceF32,
			buffers: &group_buffers,
			push_constants: &group_constants,
			workgroups: [group_count.div_ceil(256), 1, 1],
		},
	];
	let semantic_inputs = [
		output_gradient,
		c,
		b,
		x,
		z,
		adt,
		dt,
		trap,
		angle,
		c_bias,
		b_bias,
		d,
	];
	let semantic_outputs = [
		&d_c, &d_b, &d_x, &d_z, &d_adt, &d_dt, &d_trap, &d_angle, &d_c_bias, &d_b_bias, &d_d,
	];
	let attributes = [OpAttribute::UnsignedInteger {
		name: "config_identity".into(),
		value: config.identity(operation)?,
	}];
	engine.record_split_semantic(
		&dispatches,
		SemanticDispatch {
			contract,
			inputs: &semantic_inputs,
			outputs: &semantic_outputs,
			attributes: &attributes,
		},
	)?;
	Ok(SsmBackward {
		c: d_c,
		b: d_b,
		x: d_x,
		z: d_z,
		adt: d_adt,
		dt: d_dt,
		trap: d_trap,
		angle: d_angle,
		c_bias: d_c_bias,
		b_bias: d_b_bias,
		d: d_d,
	})
}

#[allow(
	clippy::too_many_arguments,
	reason = "preserves the donor upstream plus eleven-operand adjoint boundary"
)]
fn mamba3_siso_backward_chunked(
	output_gradient: &Matrix,
	c: &Matrix,
	b: &Matrix,
	x: &Matrix,
	z: &Matrix,
	adt: &Matrix,
	dt: &Matrix,
	trap: &Matrix,
	angle: &Matrix,
	c_bias: &Matrix,
	b_bias: &Matrix,
	d: &Matrix,
	config: SsmConfig,
	geometry: &Geometry,
) -> Result<SsmBackward> {
	let contract = crate::core::operation::ml::MAMBA3_SISO_BACKWARD;
	let operation = contract.name();
	let engine = x.engine_handle();
	let allocate = |shape: Vec<usize>| -> Result<Matrix> {
		let count = checked_count(&shape, operation)?;
		shader_u32(count, "chunked SISO tensor element count", operation)?;
		Matrix::allocate(engine, shape, count, DType::F32)
	};
	let batch = geometry.batch as usize;
	let length = geometry.length as usize;
	let heads = geometry.heads as usize;
	let groups = geometry.groups as usize;
	let head_dim = geometry.head_dim as usize;
	let state_size = geometry.state_size as usize;
	let angles = geometry.angles as usize;
	let batch_heads = geometry.batch_heads as usize;
	let chunks = length.div_ceil(16);
	let chunks_u32 = shader_u32(chunks, "chunk count", operation)?;
	let chunk_workgroups = geometry
		.batch_heads
		.checked_mul(chunks_u32)
		.ok_or_else(|| Error::invalid_argument(format!("{operation} chunk count exceeds u32")))?;

	let scalar_shape = vec![batch, length, heads];
	let vector_shape = vec![batch, length, heads, head_dim];
	let head_state_shape = vec![batch, length, heads, state_size];
	let chunk_state_shape = vec![batch_heads, chunks, head_dim, state_size];
	let c_rot = allocate(head_state_shape.clone())?;
	let b_rot = allocate(head_state_shape.clone())?;
	let theta = allocate(vec![batch_heads, length, angles])?;
	let qk = allocate(scalar_shape.clone())?;
	let decay = allocate(scalar_shape.clone())?;
	let gamma = allocate(scalar_shape.clone())?;
	let scale = allocate(scalar_shape.clone())?;
	let summary = allocate(chunk_state_shape.clone())?;
	let product = allocate(vec![batch_heads, chunks])?;
	let entry = allocate(chunk_state_shape.clone())?;
	let d_pre = allocate(vector_shape.clone())?;
	let h_previous = allocate(vec![batch_heads, length, head_dim, state_size])?;
	let d_z = allocate(vector_shape.clone())?;
	let reverse_summary = allocate(chunk_state_shape.clone())?;
	let chunk_exit = allocate(chunk_state_shape)?;
	let d_c_head = allocate(head_state_shape.clone())?;
	let d_b_head = allocate(head_state_shape)?;
	let d_x = allocate(vector_shape)?;
	let d_adt = allocate(scalar_shape.clone())?;
	let d_gamma = allocate(scalar_shape.clone())?;
	let d_scale = allocate(scalar_shape.clone())?;
	let d_theta = allocate(vec![batch_heads, length, angles])?;
	let d_d_token = allocate(scalar_shape.clone())?;
	let d_dt = allocate(scalar_shape.clone())?;
	let d_trap = allocate(scalar_shape)?;
	let d_angle_head = allocate(vec![batch, heads, length, angles])?;
	let d_c = allocate(vec![batch, length, groups, state_size])?;
	let d_b = allocate(vec![batch, length, groups, state_size])?;
	let d_angle = allocate(vec![batch, length, angles])?;
	let d_c_bias = allocate(vec![heads, state_size])?;
	let d_b_bias = allocate(vec![heads, state_size])?;
	let d_d = allocate(vec![heads])?;

	let prepare_buffers = [
		BufferBinding::read(c.storage()),
		BufferBinding::read(b.storage()),
		BufferBinding::read(dt.storage()),
		BufferBinding::read(trap.storage()),
		BufferBinding::read(adt.storage()),
		BufferBinding::read(angle.storage()),
		BufferBinding::read(c_bias.storage()),
		BufferBinding::read(b_bias.storage()),
		BufferBinding::write(c_rot.storage()),
		BufferBinding::write(b_rot.storage()),
		BufferBinding::write(theta.storage()),
		BufferBinding::write(qk.storage()),
		BufferBinding::write(decay.storage()),
		BufferBinding::write(gamma.storage()),
		BufferBinding::write(scale.storage()),
	];
	let prepare_constants = [
		PushConstant::U32(geometry.batch),
		PushConstant::U32(geometry.length),
		PushConstant::U32(geometry.heads),
		PushConstant::U32(geometry.groups),
		PushConstant::U32(geometry.state_size),
		PushConstant::U32(geometry.angles),
	];
	let summary_buffers = [
		BufferBinding::read(b_rot.storage()),
		BufferBinding::read(x.storage()),
		BufferBinding::read(decay.storage()),
		BufferBinding::read(scale.storage()),
		BufferBinding::write(summary.storage()),
		BufferBinding::write(product.storage()),
	];
	let summary_constants = [
		PushConstant::U32(geometry.batch),
		PushConstant::U32(geometry.length),
		PushConstant::U32(geometry.heads),
		PushConstant::U32(geometry.head_dim),
		PushConstant::U32(geometry.state_size),
		PushConstant::U32(chunks_u32),
	];
	let prefix_buffers = [
		BufferBinding::read(summary.storage()),
		BufferBinding::read(product.storage()),
		BufferBinding::write(entry.storage()),
	];
	let prefix_constants = [
		PushConstant::U32(geometry.batch_heads),
		PushConstant::U32(geometry.head_dim),
		PushConstant::U32(geometry.state_size),
		PushConstant::U32(chunks_u32),
	];
	let dpre_buffers = [
		BufferBinding::read(c_rot.storage()),
		BufferBinding::read(b_rot.storage()),
		BufferBinding::read(x.storage()),
		BufferBinding::read(z.storage()),
		BufferBinding::read(decay.storage()),
		BufferBinding::read(gamma.storage()),
		BufferBinding::read(scale.storage()),
		BufferBinding::read(qk.storage()),
		BufferBinding::read(d.storage()),
		BufferBinding::read(output_gradient.storage()),
		BufferBinding::read(entry.storage()),
		BufferBinding::write(h_previous.storage()),
		BufferBinding::write(d_pre.storage()),
		BufferBinding::write(d_z.storage()),
	];
	let dpre_constants = [
		PushConstant::U32(geometry.batch),
		PushConstant::U32(geometry.length),
		PushConstant::U32(geometry.heads),
		PushConstant::U32(geometry.head_dim),
		PushConstant::U32(geometry.state_size),
		PushConstant::U32(chunks_u32),
		PushConstant::U32(u32::from(config.has_z)),
		PushConstant::U32(u32::from(config.has_d)),
	];
	let backward_summary_buffers = [
		BufferBinding::read(c_rot.storage()),
		BufferBinding::read(decay.storage()),
		BufferBinding::read(d_pre.storage()),
		BufferBinding::write(reverse_summary.storage()),
	];
	let backward_summary_constants = summary_constants;
	let backward_prefix_buffers = [
		BufferBinding::read(reverse_summary.storage()),
		BufferBinding::read(product.storage()),
		BufferBinding::write(chunk_exit.storage()),
	];
	let backward_prefix_constants = prefix_constants;
	let backward_buffers = [
		BufferBinding::read(c_rot.storage()),
		BufferBinding::read(b_rot.storage()),
		BufferBinding::read(theta.storage()),
		BufferBinding::read(x.storage()),
		BufferBinding::read(decay.storage()),
		BufferBinding::read(gamma.storage()),
		BufferBinding::read(scale.storage()),
		BufferBinding::read(qk.storage()),
		BufferBinding::read(d.storage()),
		BufferBinding::read(d_pre.storage()),
		BufferBinding::read(h_previous.storage()),
		BufferBinding::read(chunk_exit.storage()),
		BufferBinding::write(d_c_head.storage()),
		BufferBinding::write(d_b_head.storage()),
		BufferBinding::write(d_x.storage()),
		BufferBinding::write(d_adt.storage()),
		BufferBinding::write(d_gamma.storage()),
		BufferBinding::write(d_scale.storage()),
		BufferBinding::write(d_theta.storage()),
		BufferBinding::write(d_d_token.storage()),
	];
	let backward_constants = [
		PushConstant::U32(geometry.batch),
		PushConstant::U32(geometry.length),
		PushConstant::U32(geometry.heads),
		PushConstant::U32(geometry.head_dim),
		PushConstant::U32(geometry.state_size),
		PushConstant::U32(geometry.angles),
		PushConstant::U32(chunks_u32),
		PushConstant::U32(u32::from(config.has_d)),
	];
	let finalize_buffers = [
		BufferBinding::read(dt.storage()),
		BufferBinding::read(trap.storage()),
		BufferBinding::read(angle.storage()),
		BufferBinding::read(d_gamma.storage()),
		BufferBinding::read(d_scale.storage()),
		BufferBinding::read(d_theta.storage()),
		BufferBinding::write(d_dt.storage()),
		BufferBinding::write(d_trap.storage()),
		BufferBinding::write(d_angle_head.storage()),
	];
	let finalize_constants = [
		PushConstant::U32(geometry.batch),
		PushConstant::U32(geometry.length),
		PushConstant::U32(geometry.heads),
		PushConstant::U32(geometry.angles),
	];
	let group_count = checked_count(&[batch, length, groups, state_size], operation)?;
	let angle_count = checked_count(&[batch, length, angles], operation)?;
	let bias_count = checked_count(&[heads, state_size], operation)?;
	let reduce_count = group_count
		.checked_add(angle_count)
		.and_then(|count| count.checked_add(bias_count))
		.and_then(|count| count.checked_add(heads))
		.ok_or_else(|| {
			Error::invalid_argument(format!("{operation} chunk reduction size overflows usize"))
		})?;
	let reduce_count = shader_u32(reduce_count, "chunk reduction size", operation)?;
	let reduce_buffers = [
		BufferBinding::read(d_c_head.storage()),
		BufferBinding::read(d_b_head.storage()),
		BufferBinding::read(d_angle_head.storage()),
		BufferBinding::read(d_d_token.storage()),
		BufferBinding::write(d_c.storage()),
		BufferBinding::write(d_b.storage()),
		BufferBinding::write(d_angle.storage()),
		BufferBinding::write(d_c_bias.storage()),
		BufferBinding::write(d_b_bias.storage()),
		BufferBinding::write(d_d.storage()),
	];
	let reduce_constants = [
		PushConstant::U32(geometry.batch),
		PushConstant::U32(geometry.length),
		PushConstant::U32(geometry.heads),
		PushConstant::U32(geometry.groups),
		PushConstant::U32(geometry.state_size),
		PushConstant::U32(geometry.angles),
		PushConstant::U32(reduce_count),
	];
	let dispatches = [
		ComputeDispatch {
			kernel: KernelId::MlMamba3SisoBackwardPrepareF32,
			buffers: &prepare_buffers,
			push_constants: &prepare_constants,
			workgroups: [geometry.batch_heads.div_ceil(64), 1, 1],
		},
		ComputeDispatch {
			kernel: KernelId::MlMamba3SisoChunkSummaryF32,
			buffers: &summary_buffers,
			push_constants: &summary_constants,
			workgroups: [chunk_workgroups, 1, 1],
		},
		ComputeDispatch {
			kernel: KernelId::MlMamba3SisoChunkPrefixF32,
			buffers: &prefix_buffers,
			push_constants: &prefix_constants,
			workgroups: [geometry.batch_heads, 1, 1],
		},
		ComputeDispatch {
			kernel: KernelId::MlMamba3SisoChunkDpreF32,
			buffers: &dpre_buffers,
			push_constants: &dpre_constants,
			workgroups: [chunk_workgroups, 1, 1],
		},
		ComputeDispatch {
			kernel: KernelId::MlMamba3SisoChunkBackwardSummaryF32,
			buffers: &backward_summary_buffers,
			push_constants: &backward_summary_constants,
			workgroups: [chunk_workgroups, 1, 1],
		},
		ComputeDispatch {
			kernel: KernelId::MlMamba3SisoChunkBackwardPrefixF32,
			buffers: &backward_prefix_buffers,
			push_constants: &backward_prefix_constants,
			workgroups: [geometry.batch_heads, 1, 1],
		},
		ComputeDispatch {
			kernel: KernelId::MlMamba3SisoChunkBackwardF32,
			buffers: &backward_buffers,
			push_constants: &backward_constants,
			workgroups: [chunk_workgroups, 1, 1],
		},
		ComputeDispatch {
			kernel: KernelId::MlMamba3SisoChunkFinalizeF32,
			buffers: &finalize_buffers,
			push_constants: &finalize_constants,
			workgroups: [geometry.batch_heads.div_ceil(64), 1, 1],
		},
		ComputeDispatch {
			kernel: KernelId::MlMamba3SisoChunkReduceF32,
			buffers: &reduce_buffers,
			push_constants: &reduce_constants,
			workgroups: [reduce_count.div_ceil(256), 1, 1],
		},
	];
	let semantic_inputs = [
		output_gradient,
		c,
		b,
		x,
		z,
		adt,
		dt,
		trap,
		angle,
		c_bias,
		b_bias,
		d,
	];
	let semantic_outputs = [
		&d_c, &d_b, &d_x, &d_z, &d_adt, &d_dt, &d_trap, &d_angle, &d_c_bias, &d_b_bias, &d_d,
	];
	let attributes = [OpAttribute::UnsignedInteger {
		name: "config_identity".into(),
		value: config.identity(operation)?,
	}];
	engine.record_split_semantic(
		&dispatches,
		SemanticDispatch {
			contract,
			inputs: &semantic_inputs,
			outputs: &semantic_outputs,
			attributes: &attributes,
		},
	)?;
	Ok(SsmBackward {
		c: d_c,
		b: d_b,
		x: d_x,
		z: d_z,
		adt: d_adt,
		dt: d_dt,
		trap: d_trap,
		angle: d_angle,
		c_bias: d_c_bias,
		b_bias: d_b_bias,
		d: d_d,
	})
}

#[allow(
	clippy::too_many_arguments,
	reason = "preserves the donor upstream plus eleven-operand adjoint boundary"
)]
fn mamba3_siso_backward_generic(
	output_gradient: &Matrix,
	c: &Matrix,
	b: &Matrix,
	x: &Matrix,
	z: &Matrix,
	adt: &Matrix,
	dt: &Matrix,
	trap: &Matrix,
	angle: &Matrix,
	c_bias: &Matrix,
	b_bias: &Matrix,
	d: &Matrix,
	config: SsmConfig,
	geometry: &Geometry,
) -> Result<SsmBackward> {
	let contract = crate::core::operation::ml::MAMBA3_SISO_BACKWARD;
	let operation = contract.name();
	let engine = x.engine_handle();
	let allocate = |shape: Vec<usize>| -> Result<Matrix> {
		let count = checked_count(&shape, operation)?;
		shader_u32(count, "generic SISO tensor element count", operation)?;
		Matrix::allocate(engine, shape, count, DType::F32)
	};
	let batch = geometry.batch as usize;
	let length = geometry.length as usize;
	let heads = geometry.heads as usize;
	let groups = geometry.groups as usize;
	let head_dim = geometry.head_dim as usize;
	let state_size = geometry.state_size as usize;
	let angles = geometry.angles as usize;
	let batch_heads = geometry.batch_heads as usize;
	let chunks = length.div_ceil(32);

	let entry = allocate(vec![batch_heads, chunks, head_dim, state_size])?;
	let theta_entry = allocate(vec![batch_heads, chunks, angles])?;
	let chunk_buffer = allocate(vec![batch_heads, 32, head_dim, state_size])?;
	let d_c_head = allocate(vec![batch, length, heads, state_size])?;
	let d_b_head = allocate(vec![batch, length, heads, state_size])?;
	let d_x = allocate(vec![batch, length, heads, head_dim])?;
	let d_z = allocate(vec![batch, length, heads, head_dim])?;
	let d_adt = allocate(vec![batch, length, heads])?;
	let d_dt = allocate(vec![batch, length, heads])?;
	let d_trap_partial = allocate(vec![batch, length, heads])?;
	let d_angle_head = allocate(vec![batch, heads, length, angles])?;
	let d_d_token = allocate(vec![batch, length, heads])?;
	let d_c = allocate(vec![batch, length, groups, state_size])?;
	let d_b = allocate(vec![batch, length, groups, state_size])?;
	let d_angle = allocate(vec![batch, length, angles])?;
	let d_c_bias = allocate(vec![heads, state_size])?;
	let d_b_bias = allocate(vec![heads, state_size])?;
	let d_d = allocate(vec![heads])?;
	let d_trap = allocate(vec![batch, length, heads])?;

	let generic_buffers = [
		BufferBinding::read(c.storage()),
		BufferBinding::read(b.storage()),
		BufferBinding::read(x.storage()),
		BufferBinding::read(z.storage()),
		BufferBinding::read(adt.storage()),
		BufferBinding::read(dt.storage()),
		BufferBinding::read(trap.storage()),
		BufferBinding::read(angle.storage()),
		BufferBinding::read(c_bias.storage()),
		BufferBinding::read(b_bias.storage()),
		BufferBinding::read(d.storage()),
		BufferBinding::read(output_gradient.storage()),
		BufferBinding::read_write(entry.storage()),
		BufferBinding::read_write(theta_entry.storage()),
		BufferBinding::read_write(chunk_buffer.storage()),
		BufferBinding::write(d_c_head.storage()),
		BufferBinding::write(d_b_head.storage()),
		BufferBinding::write(d_x.storage()),
		BufferBinding::write(d_z.storage()),
		BufferBinding::write(d_adt.storage()),
		BufferBinding::read_write(d_dt.storage()),
		BufferBinding::read_write(d_trap_partial.storage()),
		BufferBinding::write(d_angle_head.storage()),
		BufferBinding::write(d_d_token.storage()),
	];
	let flags = u32::from(config.has_z) | (u32::from(config.has_d) << 1);
	let generic_constants = [
		PushConstant::U32(geometry.batch),
		PushConstant::U32(geometry.length),
		PushConstant::U32(geometry.heads),
		PushConstant::U32(geometry.groups),
		PushConstant::U32(geometry.head_dim),
		PushConstant::U32(geometry.state_size),
		PushConstant::U32(geometry.angles),
		PushConstant::U32(flags),
	];

	let group_count = checked_count(&[batch, length, groups, state_size], operation)?;
	let angle_count = checked_count(&[batch, length, angles], operation)?;
	let bias_count = checked_count(&[heads, state_size], operation)?;
	let trap_count = checked_count(&[batch, length, heads], operation)?;
	let reduce_count = group_count
		.checked_add(angle_count)
		.and_then(|count| count.checked_add(bias_count))
		.and_then(|count| count.checked_add(heads))
		.and_then(|count| count.checked_add(trap_count))
		.ok_or_else(|| {
			Error::invalid_argument(format!(
				"{operation} generic reduction size overflows usize"
			))
		})?;
	let reduce_count = shader_u32(reduce_count, "generic reduction size", operation)?;
	let reduce_buffers = [
		BufferBinding::read(d_c_head.storage()),
		BufferBinding::read(d_b_head.storage()),
		BufferBinding::read(d_angle_head.storage()),
		BufferBinding::read(d_d_token.storage()),
		BufferBinding::read(trap.storage()),
		BufferBinding::read(d_trap_partial.storage()),
		BufferBinding::write(d_c.storage()),
		BufferBinding::write(d_b.storage()),
		BufferBinding::write(d_angle.storage()),
		BufferBinding::write(d_c_bias.storage()),
		BufferBinding::write(d_b_bias.storage()),
		BufferBinding::write(d_d.storage()),
		BufferBinding::write(d_trap.storage()),
	];
	let reduce_constants = [
		PushConstant::U32(geometry.batch),
		PushConstant::U32(geometry.length),
		PushConstant::U32(geometry.heads),
		PushConstant::U32(geometry.groups),
		PushConstant::U32(geometry.state_size),
		PushConstant::U32(geometry.angles),
		PushConstant::U32(reduce_count),
	];
	let dispatches = [
		ComputeDispatch {
			kernel: KernelId::MlMamba3SisoBackwardGenericF32,
			buffers: &generic_buffers,
			push_constants: &generic_constants,
			workgroups: [geometry.batch_heads, 1, 1],
		},
		ComputeDispatch {
			kernel: KernelId::MlMamba3SisoBackwardGenericReduceF32,
			buffers: &reduce_buffers,
			push_constants: &reduce_constants,
			workgroups: [reduce_count.div_ceil(256), 1, 1],
		},
	];
	let semantic_inputs = [
		output_gradient,
		c,
		b,
		x,
		z,
		adt,
		dt,
		trap,
		angle,
		c_bias,
		b_bias,
		d,
	];
	let semantic_outputs = [
		&d_c, &d_b, &d_x, &d_z, &d_adt, &d_dt, &d_trap, &d_angle, &d_c_bias, &d_b_bias, &d_d,
	];
	let attributes = [OpAttribute::UnsignedInteger {
		name: "config_identity".into(),
		value: config.identity(operation)?,
	}];
	engine.record_split_semantic(
		&dispatches,
		SemanticDispatch {
			contract,
			inputs: &semantic_inputs,
			outputs: &semantic_outputs,
			attributes: &attributes,
		},
	)?;
	Ok(SsmBackward {
		c: d_c,
		b: d_b,
		x: d_x,
		z: d_z,
		adt: d_adt,
		dt: d_dt,
		trap: d_trap,
		angle: d_angle,
		c_bias: d_c_bias,
		b_bias: d_b_bias,
		d: d_d,
	})
}
