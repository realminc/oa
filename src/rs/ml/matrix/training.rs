//! Stateless state-slot accumulation and exact categorical cursor sampling.
//! Donors: core flat add.gen.slang, Philox uniform and dense sample_logits.
use crate::runtime::{BufferBinding, ComputeDispatch, KernelId, PushConstant, SemanticDispatch};
use crate::{DType, Error, Matrix, OpAttribute, Result};

pub(in crate::ml) fn training_accumulate(
	target: &Matrix,
	value: &Matrix,
	cursor: &Matrix,
	scale: f32,
) -> Result<()> {
	if target.dtype() != DType::F32
		|| target.shape() != value.shape()
		|| value.dtype() != DType::F32
		|| !target.engine_handle().same_as(value.engine_handle())
		|| cursor.dtype() != DType::U32
		|| cursor.shape() != [2]
		|| !target.engine_handle().same_as(cursor.engine_handle())
		|| !scale.is_finite()
	{
		return Err(Error::invalid_argument(
			"compiled accumulation shape/dtype/Engine mismatch",
		));
	}
	let count = u32::try_from(target.num_elements())
		.map_err(|_| Error::resource_exhausted("gradient too large"))?;
	let buffers = [
		BufferBinding::read_write(target.storage()),
		BufferBinding::read(value.storage()),
		BufferBinding::read(cursor.storage()),
	];
	let push = [PushConstant::U32(count), PushConstant::F32(scale)];
	target.engine_handle().record_semantic(
		ComputeDispatch {
			kernel: KernelId::MlTrainingAccumulateF32,
			buffers: &buffers,
			push_constants: &push,
			workgroups: KernelId::MlTrainingAccumulateF32.linear_workgroups(count),
		},
		SemanticDispatch {
			contract: crate::core::operation::ml::TRAINING_ACCUMULATE,
			inputs: &[target, value, cursor],
			outputs: &[target],
			attributes: &[OpAttribute::Float {
				name: "scale".into(),
				value: f64::from(scale),
			}],
		},
	)
}

pub(in crate::ml) fn sample_categorical_cursor(logits: &Matrix, cursor: &Matrix) -> Result<Matrix> {
	let [rows, categories] = logits.shape() else {
		return Err(Error::invalid_argument(
			"categorical cursor requires [rows,categories]",
		));
	};
	if *rows == 0
		|| *categories < 2
		|| logits.dtype() != DType::F32
		|| cursor.dtype() != DType::U32
		|| cursor.shape() != [7]
		|| !logits.engine_handle().same_as(cursor.engine_handle())
	{
		return Err(Error::invalid_argument(
			"categorical cursor shape/dtype/Engine mismatch",
		));
	}
	let rows =
		u32::try_from(*rows).map_err(|_| Error::resource_exhausted("categorical rows overflow"))?;
	let categories = u32::try_from(*categories)
		.map_err(|_| Error::resource_exhausted("categorical categories overflow"))?;
	let output = Matrix::allocate(
		logits.engine_handle(),
		vec![rows as usize],
		rows as usize,
		DType::I32,
	)?;
	let buffers = [
		BufferBinding::read(logits.storage()),
		BufferBinding::read(cursor.storage()),
		BufferBinding::write(output.storage()),
	];
	let push = [PushConstant::U32(rows), PushConstant::U32(categories)];
	logits.engine_handle().record_semantic(
		ComputeDispatch {
			kernel: KernelId::MlSampleCategoricalCursorF32,
			buffers: &buffers,
			push_constants: &push,
			workgroups: KernelId::MlSampleCategoricalCursorF32.linear_workgroups(rows),
		},
		SemanticDispatch {
			contract: crate::core::operation::ml::SAMPLE_CATEGORICAL_CURSOR,
			inputs: &[logits, cursor],
			outputs: &[&output],
			attributes: &[],
		},
	)?;
	Ok(output)
}
