//! Private lowering for selective state-space operations.

use crate::{
	DType, Error, Matrix, OpAttribute, Result,
	runtime::{BufferBinding, ComputeDispatch, KernelId, PushConstant, SemanticDispatch},
};

use super::common::{shader_u32, validate_f32_same_engine};
use crate::ml::matrix::SsmConfig;

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

fn checked_count(shape: &[usize], operation: &'static str) -> Result<usize> {
	shape.iter().try_fold(1_usize, |count, extent| {
		count.checked_mul(*extent).ok_or_else(|| {
			Error::invalid_argument(format!("{operation} tensor size overflows usize"))
		})
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
	if geometry.length > 16
		|| geometry.head_dim > 16
		|| geometry.state_size > 32
		|| geometry.angles > 8
	{
		return Err(Error::missing_capability(
			"oa::ml::matrix::mamba3_siso_backward currently admits the donor short route (L/P <= 16, N <= 32, A <= 8)",
		));
	}

	let engine = x.engine_handle();
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
		.ok_or_else(|| {
			Error::invalid_argument(format!("{operation} reduction size exceeds u32"))
		})?;
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
		.ok_or_else(|| {
			Error::invalid_argument(format!("{operation} group reduction exceeds u32"))
		})?;
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
