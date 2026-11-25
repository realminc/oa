//! Precision-policy provider: FP16-storage / FP32-accumulate lowering.
//!
//! The public matrix operations keep their F32 contracts; when the
//! `OA_FORCE_PRECISION=FP16` math mode is active and the admitted device has
//! enabled `shaderFloat16` + `storageBuffer16BitAccess`, this provider lowers
//! the operation as one split-semantic group: F32→F16 storage casts into
//! private workspaces followed by the dedicated FP16-storage kernel. Master
//! weights, accumulation and operation outputs stay F32 — only operand
//! storage narrows, matching the FP16 math-mode contract.

use crate::{Error, Result};

use crate::Matrix;
use crate::runtime::{
	BufferBinding, ComputeDispatch, KernelId, PushConstant, RuntimeHandle, SemanticDispatch,
};

use super::blas::{GemmKernel, GemmPrecision, MatmulPlan};

/// Whether the FP16-storage math mode is requested and admitted on this
/// runtime's device contract. Returns false — the caller's F32 route stays —
/// when either condition fails.
fn fp16s_admitted(runtime: &RuntimeHandle) -> bool {
	if forced_precision() != GemmPrecision::F16 {
		return false;
	}
	runtime.matmul_device_contract().caps_mask & super::blas::CAP_F16_STORAGE != 0
}

/// `OA_FORCE_PRECISION` request, shared by all precision-provider lowerings.
fn forced_precision() -> GemmPrecision {
	match crate::core::env_flag::EnvFlag::get_string("OA_FORCE_PRECISION", "").as_str() {
		"FP16" => GemmPrecision::F16,
		"BF16" => GemmPrecision::Bf16,
		"Q8" => GemmPrecision::Q8,
		"FP32" => GemmPrecision::F32,
		_ => GemmPrecision::Auto,
	}
}

/// Record the FP16-storage lowering of one `matmul` plan: two F32→F16
/// storage casts into private workspaces plus the selected FP16-storage GEMM.
///
/// Returns `Ok(false)` for plans that did not select an F16s variant so the
/// caller records its ordinary single-dispatch route instead.
pub(crate) fn record_matmul_nt_f16(
	runtime: &RuntimeHandle,
	left: &Matrix,
	right: &Matrix,
	output: &Matrix,
	plan: &MatmulPlan,
	dims: [u32; 3],
) -> Result<bool> {
	let [m, n, k] = dims;
	let gemm_kernel = match plan.kernel {
		GemmKernel::F16F32 | GemmKernel::TiledWideF16F32 => plan.kernel_id,
		_ => return Ok(false),
	};
	runtime.require_kernel(KernelId::MatrixCastF32F16)?;
	runtime.require_kernel(gemm_kernel)?;
	let left_count = usize::try_from(u64::from(m) * u64::from(k))
		.map_err(|_| Error::invalid_argument("matmul f16 left workspace exceeds usize"))?;
	let right_count = usize::try_from(u64::from(n) * u64::from(k))
		.map_err(|_| Error::invalid_argument("matmul f16 right workspace exceeds usize"))?;
	let cast_a_count = m
		.checked_mul(k)
		.ok_or_else(|| Error::invalid_argument("matmul f16 left cast count exceeds u32"))?;
	let cast_b_count = n
		.checked_mul(k)
		.ok_or_else(|| Error::invalid_argument("matmul f16 right cast count exceeds u32"))?;
	let left_f16 = Matrix::allocate_output(runtime, vec![left_count], left_count, crate::DType::F16)?;
	let right_f16 =
		Matrix::allocate_output(runtime, vec![right_count], right_count, crate::DType::F16)?;
	let cast_a_push = [PushConstant::U32(cast_a_count)];
	let cast_b_push = [PushConstant::U32(cast_b_count)];
	let gemm_push = [
		PushConstant::U32(m),
		PushConstant::U32(n),
		PushConstant::U32(k),
	];
	let cast_a_buffers = [
		BufferBinding::read(left.storage()),
		BufferBinding::write(left_f16.storage()),
	];
	let cast_b_buffers = [
		BufferBinding::read(right.storage()),
		BufferBinding::write(right_f16.storage()),
	];
	let gemm_buffers = [
		BufferBinding::read(left_f16.storage()),
		BufferBinding::read(right_f16.storage()),
		BufferBinding::write(output.storage()),
	];
	let dispatches = [
		ComputeDispatch {
			kernel: KernelId::MatrixCastF32F16,
			buffers: &cast_a_buffers,
			push_constants: &cast_a_push,
			workgroups: KernelId::MatrixCastF32F16.linear_workgroups(cast_a_count),
			indirect: None,
		},
		ComputeDispatch {
			kernel: KernelId::MatrixCastF32F16,
			buffers: &cast_b_buffers,
			push_constants: &cast_b_push,
			workgroups: KernelId::MatrixCastF32F16.linear_workgroups(cast_b_count),
			indirect: None,
		},
		ComputeDispatch {
			kernel: gemm_kernel,
			buffers: &gemm_buffers,
			push_constants: &gemm_push,
			workgroups: plan.grid,
			indirect: None,
		},
	];
	let inputs = [left, right];
	let outputs = [output];
	runtime.record_split_semantic(
		&dispatches,
		SemanticDispatch {
			contract: crate::core::operation::matrix::MATMUL,
			inputs: &inputs,
			outputs: &outputs,
			attributes: &[],
		},
	)?;
	Ok(true)
}

/// Record the FP16-storage lowering of one full `sum`: an F32→F16 storage
/// cast into a private workspace plus the FP16-storage single-workgroup
/// reduction. Returns `Ok(false)` when the F16 math mode is not requested or
/// not admitted, so the caller records the ordinary F32 route.
pub(crate) fn record_sum_f16(
	runtime: &RuntimeHandle,
	input: &Matrix,
	output: &Matrix,
	element_count: u32,
	dim: i32,
) -> Result<bool> {
	if !fp16s_admitted(runtime) {
		return Ok(false);
	}
	runtime.require_kernel(KernelId::MatrixCastF32F16)?;
	runtime.require_kernel(KernelId::MatrixSumF16)?;
	let input_f16 = Matrix::allocate_output(
		runtime,
		vec![element_count as usize],
		element_count as usize,
		crate::DType::F16,
	)?;
	let cast_push = [PushConstant::U32(element_count)];
	let cast_buffers = [
		BufferBinding::read(input.storage()),
		BufferBinding::write(input_f16.storage()),
	];
	let sum_buffers = [
		BufferBinding::read(input_f16.storage()),
		BufferBinding::write(output.storage()),
	];
	let dispatches = [
		ComputeDispatch {
			kernel: KernelId::MatrixCastF32F16,
			buffers: &cast_buffers,
			push_constants: &cast_push,
			workgroups: KernelId::MatrixCastF32F16.linear_workgroups(element_count),
			indirect: None,
		},
		ComputeDispatch {
			kernel: KernelId::MatrixSumF16,
			buffers: &sum_buffers,
			push_constants: &cast_push,
			workgroups: [1, 1, 1],
			indirect: None,
		},
	];
	let inputs = [input];
	let outputs = [output];
	let attributes = [crate::OpAttribute::SignedInteger {
		name: "dim".into(),
		value: i64::from(dim),
	}];
	runtime.record_split_semantic(
		&dispatches,
		SemanticDispatch {
			contract: crate::core::operation::matrix::SUM,
			inputs: &inputs,
			outputs: &outputs,
			attributes: &attributes,
		},
	)?;
	Ok(true)
}

/// Record the FP16-storage lowering of one ML `linear`: F32→F16 casts on
/// input and weight, the FP16-storage GEMM into a private F32 workspace, and
/// the established row-broadcast bias tail into the F32 output.
///
/// Returns `Ok(false)` when the FP16 math mode is not admitted so the caller
/// records the incumbent fused `MlLinearF32` dispatch.
pub(crate) fn record_linear_f16(
	runtime: &RuntimeHandle,
	input: &Matrix,
	weight: &Matrix,
	bias: &Matrix,
	output: &Matrix,
	contract: crate::OperationContract,
	dims: [u32; 3],
) -> Result<bool> {
	if !fp16s_admitted(runtime) {
		return Ok(false);
	}
	let [batch, input_features, output_features] = dims;
	runtime.require_kernel(KernelId::MatrixCastF32F16)?;
	runtime.require_kernel(KernelId::MatrixMatmulNtF16)?;
	runtime.require_kernel(KernelId::MatrixBiasAddBroadcastF32)?;
	let input_count = usize::try_from(u64::from(batch) * u64::from(input_features))
		.map_err(|_| Error::invalid_argument("linear f16 input workspace exceeds usize"))?;
	let weight_count = usize::try_from(u64::from(output_features) * u64::from(input_features))
		.map_err(|_| Error::invalid_argument("linear f16 weight workspace exceeds usize"))?;
	let output_count = usize::try_from(u64::from(batch) * u64::from(output_features))
		.map_err(|_| Error::invalid_argument("linear f16 workspace exceeds usize"))?;
	let input_f16 =
		Matrix::allocate_output(runtime, vec![input_count], input_count, crate::DType::F16)?;
	let weight_f16 =
		Matrix::allocate_output(runtime, vec![weight_count], weight_count, crate::DType::F16)?;
	let gemm_out =
		Matrix::allocate_output(runtime, vec![output_count], output_count, crate::DType::F32)?;
	let cast_a_count = batch
		.checked_mul(input_features)
		.ok_or_else(|| Error::invalid_argument("linear f16 input cast count exceeds u32"))?;
	let cast_b_count = output_features
		.checked_mul(input_features)
		.ok_or_else(|| Error::invalid_argument("linear f16 weight cast count exceeds u32"))?;
	let gemm_grid = [batch.div_ceil(64), output_features.div_ceil(64), 1];
	let element_count = batch
		.checked_mul(output_features)
		.ok_or_else(|| Error::invalid_argument("linear f16 output count exceeds u32"))?;
	// Row-broadcast bias tail over [batch, out] + [out]: the right operand's
	// row stride is zero and column stride is one.
	let broadcast_push: [PushConstant; 26] = std::array::from_fn(|index| {
		PushConstant::U32(match index {
			0 => element_count,
			1 => 2,
			2 => batch,
			3 => output_features,
			4..=9 => 1,
			10 => output_features,
			11 => 1,
			12..=17 => 0,
			18 => 0,
			19 => 1,
			_ => 0,
		})
	});
	let cast_a_buffers = [
		BufferBinding::read(input.storage()),
		BufferBinding::write(input_f16.storage()),
	];
	let cast_b_buffers = [
		BufferBinding::read(weight.storage()),
		BufferBinding::write(weight_f16.storage()),
	];
	let gemm_buffers = [
		BufferBinding::read(input_f16.storage()),
		BufferBinding::read(weight_f16.storage()),
		BufferBinding::write(gemm_out.storage()),
	];
	let bias_buffers = [
		BufferBinding::read(gemm_out.storage()),
		BufferBinding::read(bias.storage()),
		BufferBinding::write(output.storage()),
	];
	let dispatches = [
		ComputeDispatch {
			kernel: KernelId::MatrixCastF32F16,
			buffers: &cast_a_buffers,
			push_constants: &[PushConstant::U32(cast_a_count)],
			workgroups: KernelId::MatrixCastF32F16.linear_workgroups(cast_a_count),
			indirect: None,
		},
		ComputeDispatch {
			kernel: KernelId::MatrixCastF32F16,
			buffers: &cast_b_buffers,
			push_constants: &[PushConstant::U32(cast_b_count)],
			workgroups: KernelId::MatrixCastF32F16.linear_workgroups(cast_b_count),
			indirect: None,
		},
		ComputeDispatch {
			kernel: KernelId::MatrixMatmulNtF16,
			buffers: &gemm_buffers,
			push_constants: &[
				PushConstant::U32(batch),
				PushConstant::U32(output_features),
				PushConstant::U32(input_features),
			],
			workgroups: gemm_grid,
			indirect: None,
		},
		ComputeDispatch {
			kernel: KernelId::MatrixBiasAddBroadcastF32,
			buffers: &bias_buffers,
			push_constants: &broadcast_push,
			workgroups: KernelId::MatrixBiasAddBroadcastF32.linear_workgroups(element_count),
			indirect: None,
		},
	];
	let inputs = [input, weight, bias];
	let outputs = [output];
	runtime.record_split_semantic(
		&dispatches,
		SemanticDispatch {
			contract,
			inputs: &inputs,
			outputs: &outputs,
			attributes: &[],
		},
	)?;
	Ok(true)
}
