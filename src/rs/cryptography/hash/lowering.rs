//! Private semantic-to-Vulkan lowering for batch hashing.

use crate::{
	DType, Error, Matrix, OpAttribute, Result,
	runtime::{BufferBinding, ComputeDispatch, KernelId, PushConstant, SemanticDispatch},
};

#[derive(Clone, Copy)]
pub(super) enum ShakeKind {
	Shake128,
	Shake256,
}

pub(super) fn shake(input: &Matrix, output_length: usize, kind: ShakeKind) -> Result<Matrix> {
	let operation = match kind {
		ShakeKind::Shake128 => crate::core::operation::cryptography::hash::SHAKE128,
		ShakeKind::Shake256 => crate::core::operation::cryptography::hash::SHAKE256,
	};
	let kernel = match kind {
		ShakeKind::Shake128 => KernelId::CryptographyShake128U8,
		ShakeKind::Shake256 => KernelId::CryptographyShake256U8,
	};
	let rate = match kind {
		ShakeKind::Shake128 => 168,
		ShakeKind::Shake256 => 136,
	};
	let default_output_length = match kind {
		ShakeKind::Shake128 => 16,
		ShakeKind::Shake256 => 32,
	};
	let [rows, message_len] = validate_byte_rows(input, operation.name())?;
	let output_length = if output_length == 0 {
		default_output_length
	} else {
		output_length
	};
	let output_row_bytes = output_length
		.checked_add(7)
		.ok_or_else(|| Error::invalid_argument("SHAKE output length overflows usize"))?
		/ 8 * 8;
	let element_count = rows
		.checked_mul(output_row_bytes)
		.ok_or_else(|| Error::invalid_argument("SHAKE output size overflows usize"))?;
	shader_u32(element_count, "output byte size", operation.name())?;
	let output = Matrix::allocate(
		input.engine_handle(),
		vec![rows, output_row_bytes],
		element_count,
		DType::U8,
	)?;
	let rows_u32 = shader_u32(rows, "batch size", operation.name())?;
	let message_len_u32 = shader_u32(message_len, "message length", operation.name())?;
	let num_blocks = message_len
		.checked_div(rate)
		.and_then(|blocks| blocks.checked_add(1))
		.ok_or_else(|| Error::invalid_argument("SHAKE block count overflows usize"))?;
	let padded_message_len = num_blocks
		.checked_mul(rate)
		.ok_or_else(|| Error::invalid_argument("SHAKE padded message size overflows usize"))?;
	shader_u32(
		padded_message_len,
		"padded message length",
		operation.name(),
	)?;
	let num_blocks = shader_u32(num_blocks, "block count", operation.name())?;
	let squeeze_u64 = shader_u32(output_row_bytes / 8, "squeeze length", operation.name())?;
	let buffers = [
		BufferBinding::read(input.storage()),
		BufferBinding::write(output.storage()),
	];
	let push_constants = [
		PushConstant::U32(rows_u32),
		PushConstant::U32(message_len_u32),
		PushConstant::U32(num_blocks),
		PushConstant::U32(squeeze_u64),
	];
	let inputs = [input];
	let outputs = [&output];
	let attributes = [OpAttribute::UnsignedInteger {
		name: "output_length".to_owned(),
		value: u64::try_from(output_length)
			.map_err(|_| Error::invalid_argument("SHAKE output length exceeds u64"))?,
	}];
	input.engine_handle().record_semantic(
		ComputeDispatch {
			kernel,
			buffers: &buffers,
			push_constants: &push_constants,
			workgroups: kernel.linear_workgroups(rows_u32),
		},
		SemanticDispatch {
			contract: operation,
			inputs: &inputs,
			outputs: &outputs,
			attributes: &attributes,
		},
	)?;
	Ok(output)
}

pub(super) fn keccak_f1600(input: &Matrix) -> Result<Matrix> {
	const OPERATION: &str = crate::core::operation::cryptography::hash::KECCAK_F1600.name();
	let [rows, row_bytes] = validate_byte_rows(input, OPERATION)?;
	if row_bytes != 200 {
		return Err(Error::invalid_argument(format!(
			"{OPERATION} requires 200-byte rows; found {row_bytes}"
		)));
	}
	let output = Matrix::allocate(
		input.engine_handle(),
		input.shape().to_vec(),
		input.num_elements(),
		DType::U8,
	)?;
	let count = shader_u32(rows, "batch size", OPERATION)?;
	let kernel = KernelId::CryptographyKeccakF1600U8;
	let buffers = [
		BufferBinding::read(input.storage()),
		BufferBinding::write(output.storage()),
	];
	let push_constants = [PushConstant::U32(count)];
	let inputs = [input];
	let outputs = [&output];
	input.engine_handle().record_semantic(
		ComputeDispatch {
			kernel,
			buffers: &buffers,
			push_constants: &push_constants,
			workgroups: kernel.linear_workgroups(count),
		},
		SemanticDispatch {
			contract: crate::core::operation::cryptography::hash::KECCAK_F1600,
			inputs: &inputs,
			outputs: &outputs,
			attributes: &[],
		},
	)?;
	Ok(output)
}

pub(super) fn merkle_root(input: &Matrix) -> Result<Matrix> {
	const OPERATION: &str = crate::core::operation::cryptography::hash::MERKLE_ROOT.name();
	let [leaf_count, hash_bytes] = validate_byte_rows(input, OPERATION)?;
	if hash_bytes != 32 {
		return Err(Error::invalid_argument(format!(
			"{OPERATION} requires 32-byte leaf hashes; found {hash_bytes}"
		)));
	}
	if !leaf_count.is_power_of_two() {
		return Err(Error::invalid_argument(format!(
			"{OPERATION} requires a power-of-two leaf count; found {leaf_count}"
		)));
	}

	if leaf_count == 1 {
		let output = Matrix::allocate(input.engine_handle(), vec![1, 32], 32, DType::U8)?;
		let kernel = KernelId::CryptographyMerkleCopyU8;
		let buffers = [
			BufferBinding::read(input.storage()),
			BufferBinding::write(output.storage()),
		];
		let push_constants = [PushConstant::U32(8)];
		let inputs = [input];
		let outputs = [&output];
		input.engine_handle().record_semantic(
			ComputeDispatch {
				kernel,
				buffers: &buffers,
				push_constants: &push_constants,
				workgroups: kernel.linear_workgroups(8),
			},
			SemanticDispatch {
				contract: crate::core::operation::cryptography::hash::MERKLE_ROOT,
				inputs: &inputs,
				outputs: &outputs,
				attributes: &[],
			},
		)?;
		return Ok(output);
	}
	if leaf_count == 2 {
		let output = Matrix::allocate(input.engine_handle(), vec![1, 32], 32, DType::U8)?;
		let kernel = KernelId::CryptographyMerkleReduceU8;
		let buffers = [
			BufferBinding::read(input.storage()),
			BufferBinding::write(output.storage()),
		];
		let push_constants = [PushConstant::U32(1), PushConstant::U32(32)];
		let inputs = [input];
		let outputs = [&output];
		input.engine_handle().record_semantic(
			ComputeDispatch {
				kernel,
				buffers: &buffers,
				push_constants: &push_constants,
				workgroups: kernel.linear_workgroups(1),
			},
			SemanticDispatch {
				contract: crate::core::operation::cryptography::hash::MERKLE_ROOT,
				inputs: &inputs,
				outputs: &outputs,
				attributes: &[],
			},
		)?;
		return Ok(output);
	}

	let mut levels = Vec::new();
	let mut count = leaf_count / 2;
	loop {
		levels.push(Matrix::allocate(
			input.engine_handle(),
			vec![count, 32],
			count
				.checked_mul(32)
				.ok_or_else(|| Error::invalid_argument("Merkle level size overflows usize"))?,
			DType::U8,
		)?);
		if count == 1 {
			break;
		}
		count /= 2;
	}

	{
		let binding_sets = levels
			.iter()
			.enumerate()
			.map(|(index, output)| {
				let source = if index == 0 {
					input
				} else {
					&levels[index - 1]
				};
				[
					BufferBinding::read(source.storage()),
					BufferBinding::write(output.storage()),
				]
			})
			.collect::<Vec<_>>();
		let push_sets = levels
			.iter()
			.map(|level| {
				Ok([
					PushConstant::U32(shader_u32(level.shape()[0], "level count", OPERATION)?),
					PushConstant::U32(32),
				])
			})
			.collect::<Result<Vec<_>>>()?;
		let kernel = KernelId::CryptographyMerkleReduceU8;
		let dispatches = binding_sets
			.iter()
			.zip(&push_sets)
			.map(|(buffers, push_constants)| {
				let PushConstant::U32(count) = push_constants[0] else {
					unreachable!();
				};
				ComputeDispatch {
					kernel,
					buffers,
					push_constants,
					workgroups: kernel.linear_workgroups(count),
				}
			})
			.collect::<Vec<_>>();
		let inputs = [input];
		let outputs = [&levels[levels.len() - 1]];
		input.engine_handle().record_split_semantic(
			&dispatches,
			SemanticDispatch {
				contract: crate::core::operation::cryptography::hash::MERKLE_ROOT,
				inputs: &inputs,
				outputs: &outputs,
				attributes: &[],
			},
		)?;
	}
	levels
		.pop()
		.ok_or_else(|| Error::internal("Merkle lowering produced no output"))
}

fn validate_byte_rows(input: &Matrix, operation: &'static str) -> Result<[usize; 2]> {
	if input.dtype() != DType::U8 {
		return Err(Error::invalid_argument(format!(
			"{operation} requires a U8 Matrix; found {}",
			input.dtype().token()
		)));
	}
	let [rows, columns] = input.shape() else {
		return Err(Error::invalid_argument(format!(
			"{operation} requires rank-2 [batch, bytes] input; found {:?}",
			input.shape()
		)));
	};
	if *rows == 0 {
		return Err(Error::invalid_argument(format!(
			"{operation} requires a non-zero batch size"
		)));
	}
	shader_u32(input.num_elements(), "input byte size", operation)?;
	Ok([*rows, *columns])
}

fn shader_u32(value: usize, label: &str, operation: &'static str) -> Result<u32> {
	u32::try_from(value)
		.map_err(|_| Error::invalid_argument(format!("{operation} {label} exceeds u32")))
}
