//! Tile-template route admission (OA-CC-13).
//!
//! A canonical `matrix.mat_mul_nt` node may substitute an admitted
//! `BlockProgram` specialization for the incumbent registry pipeline
//! through the `JitNode` override path. Route selection is still
//! env-gated experimental vocabulary; measured candidate selection and
//! artifact keys are OA-CC-12/13 deliverables.

use std::sync::Arc;

use crate::runtime::{
	BufferAccess, Device, PushConstant,
	dna::{
		ir::tile::{BiasAct, Binding, BlockProgram},
		kernel::KernelId,
	},
	graph::executable::{ComputeNode, JitNode},
};

/// Substitute an admitted `mat_mul_nt` tile template for the incumbent
/// registry pipeline on one dispatch node. Every failure mode — gate
/// off, non-tiled kernel, unexpected dispatch ABI, compile or pipeline
/// rejection — leaves the canonical kernel route unchanged.
pub(in crate::runtime) fn specialize_gemm_node(device: &Arc<Device>, node: &mut ComputeNode) {
	if node.kernel != KernelId::MatrixMatMulNtTiledF32 || node.jit.is_some() {
		return;
	}
	let [left, right, output] = node.buffers.as_slice() else {
		return;
	};
	if left.access != BufferAccess::Read
		|| right.access != BufferAccess::Read
		|| output.access != BufferAccess::Write
		|| !device.tile_gemm_enabled()
	{
		return;
	}
	let [
		PushConstant::U32(m),
		PushConstant::U32(n),
		PushConstant::U32(_k),
	] = node.push_constants.as_slice()
	else {
		return;
	};
	// Donor schedule re-expression: `matrix_mat_mul_nt_tiled_f32` (BK=16,
	// 4×4 fragments, vec4 staging). The shape→tile map is the `OA_TILE_GEMM`
	// seed candidate; `OA_TILE_GEMM_SEARCH` admits the bounded 32/64
	// row-column set so the CC-12 route record selects by measurement.
	// `group_id.y` rides the 65535 workgroup bound; a candidate past it
	// is not offered, and a shape with no admissible candidate keeps the
	// incumbent route, which selects its own grid per variant.
	let (seed_rows, seed_columns) = gemm_candidate(*m, *n);
	let search = device.tile_gemm_search_enabled();
	let mut candidates: Vec<(bool, crate::Result<BlockProgram>)> = Vec::new();
	for tile_rows in [32u32, 64] {
		for tile_columns in [32u32, 64] {
			let enabled = if (tile_rows, tile_columns) == (seed_rows, seed_columns) {
				device.tile_gemm_enabled()
			} else {
				search
			};
			let program = BlockProgram::mat_mul_nt(tile_rows, tile_columns, 16, 4, 4, 4);
			if let Ok(program) = &program
				&& program.workgroups(*m, *n)[1] > u32::from(u16::MAX)
			{
				continue;
			}
			candidates.push((enabled, program));
		}
	}
	let Some(program) = tile_candidate(device, candidates) else {
		return;
	};
	let workgroups = program.workgroups(*m, *n);
	node.fusion_signature = Some(device.tile_signature(&program));
	let Some(key) = device.jit_admit_tile_gemm(&program) else {
		return;
	};
	node.jit = Some(JitNode {
		key,
		buffers: node.buffers.clone(),
		push_constants: node.push_constants.clone(),
	});
	node.workgroups = workgroups;
}

/// `OA_TILE_GEMM_BIAS` substitution for the BLAS epilogue donors —
/// `MatrixMatMulNtBias*F32` share the `mat_mul_nt` schedule with the
/// `[left, right, bias, output]` binding order, `[m, n, k]` push ABI
/// and a fixed inline activation on the single output store.
pub(in crate::runtime) fn specialize_gemm_bias_node(device: &Arc<Device>, node: &mut ComputeNode) {
	if node.jit.is_some() || !device.tile_gemm_bias_enabled() {
		return;
	}
	let act = match node.kernel {
		KernelId::MatrixMatMulNtBiasF32 => None,
		KernelId::MatrixMatMulNtBiasReluF32 => Some(BiasAct::Relu),
		KernelId::MatrixMatMulNtBiasGeluF32 => Some(BiasAct::Gelu),
		KernelId::MatrixMatMulNtBiasSiluF32 => Some(BiasAct::Silu),
		_ => return,
	};
	let [left, right, bias, output] = node.buffers.as_slice() else {
		return;
	};
	if left.access != BufferAccess::Read
		|| right.access != BufferAccess::Read
		|| bias.access != BufferAccess::Read
		|| output.access != BufferAccess::Write
	{
		return;
	}
	let [
		PushConstant::U32(m),
		PushConstant::U32(n),
		PushConstant::U32(_k),
	] = node.push_constants.as_slice()
	else {
		return;
	};
	let (tile_rows, tile_columns) = gemm_candidate(*m, *n);
	let Ok(program) = BlockProgram::mat_mul_nt_bias(tile_rows, tile_columns, 16, 4, 4, 4, act) else {
		return;
	};
	let workgroups = program.workgroups(*m, *n);
	if workgroups[1] > u32::from(u16::MAX) {
		return;
	}
	node.fusion_signature = Some(device.tile_signature(&program));
	let Some(key) = device.jit_admit_tile(&program) else {
		return;
	};
	node.jit = Some(JitNode {
		key,
		buffers: node.buffers.clone(),
		push_constants: node.push_constants.clone(),
	});
	node.workgroups = workgroups;
}

/// Substitute the `mat_mul_nt_gemv` tile template for the incumbent
/// registry pipeline (OA-CC-13). The emitted program preserves the
/// donor ABI — `[left, right, output]` bindings, `[m, n, k]` push
/// scalars, one workgroup per `(row, column)` output — so
/// `node.workgroups` stays untouched. Every failure mode keeps the
/// canonical route.
pub(in crate::runtime) fn specialize_gemv_node(device: &Arc<Device>, node: &mut ComputeNode) {
	if node.kernel != KernelId::MatrixMatMulNtGemvF32 || node.jit.is_some() {
		return;
	}
	let [left, right, output] = node.buffers.as_slice() else {
		return;
	};
	if left.access != BufferAccess::Read
		|| right.access != BufferAccess::Read
		|| output.access != BufferAccess::Write
		|| !device.tile_gemv_enabled()
		|| node.push_constants.len() != 3
	{
		return;
	}
	let Ok(program) = BlockProgram::mat_mul_nt_gemv(256) else {
		return;
	};
	node.fusion_signature = Some(device.tile_signature(&program));
	let Some(key) = device.jit_admit_tile_gemv(&program) else {
		return;
	};
	node.jit = Some(JitNode {
		key,
		buffers: node.buffers.clone(),
		push_constants: node.push_constants.clone(),
	});
}

/// `OA_TILE_IMAGE` substitution for `ImageResize{Nearest,Bilinear}F32`
/// nodes — the gather anchor (OA-CC-33): donor ABI `[input, output]` +
/// `[batch, channels, in_h, in_w, out_h, out_w]` push over the donor's
/// `[ceil(out_w/16), ceil(out_h/16), batch*channels]` grid.
pub(in crate::runtime) fn specialize_image_resize_node(
	device: &Arc<Device>,
	node: &mut ComputeNode,
) {
	if node.jit.is_some() || !device.tile_image_enabled() {
		return;
	}
	let nearest = match node.kernel {
		KernelId::ImageResizeNearestF32 => true,
		KernelId::ImageResizeBilinearF32 => false,
		_ => return,
	};
	let [input, output] = node.buffers.as_slice() else {
		return;
	};
	if input.access != BufferAccess::Read || output.access != BufferAccess::Write {
		return;
	}
	let [
		PushConstant::U32(_batch),
		PushConstant::U32(_channels),
		PushConstant::U32(_in_h),
		PushConstant::U32(_in_w),
		PushConstant::U32(_out_h),
		PushConstant::U32(_out_w),
	] = node.push_constants.as_slice()
	else {
		return;
	};
	let Ok(program) = BlockProgram::image_resize(nearest) else {
		return;
	};
	node.fusion_signature = Some(device.tile_signature(&program));
	let Some(key) = device.jit_admit_tile_image(&program) else {
		return;
	};
	node.jit = Some(JitNode {
		key,
		buffers: node.buffers.clone(),
		push_constants: node.push_constants.clone(),
	});
}

/// `OA_TILE_IMAGE` substitution for `ImagePadF32` nodes — the
/// signed-offset bordered fetch (OA-CC-33): donor ABI `[input, output]`
/// + nine u32 pushes + f32 `border_value`.
pub(in crate::runtime) fn specialize_image_pad_node(device: &Arc<Device>, node: &mut ComputeNode) {
	if node.kernel != KernelId::ImagePadF32 || node.jit.is_some() || !device.tile_image_enabled() {
		return;
	}
	let [input, output] = node.buffers.as_slice() else {
		return;
	};
	if input.access != BufferAccess::Read
		|| output.access != BufferAccess::Write
		|| node.push_constants.len() != 10
		|| !matches!(node.push_constants.last(), Some(PushConstant::F32(_)))
	{
		return;
	}
	let program = BlockProgram::image_pad();
	node.fusion_signature = Some(device.tile_signature(&program));
	let Some(key) = device.jit_admit_tile_image(&program) else {
		return;
	};
	node.jit = Some(JitNode {
		key,
		buffers: node.buffers.clone(),
		push_constants: node.push_constants.clone(),
	});
}

/// `OA_TILE_IMAGE` substitution for `ImageWarp{Affine,Perspective}F32`
/// nodes — the matrix-coordinate bordered gather: donor ABI `[input,
/// transform, output]` + eight u32 pushes + f32 `border_value`.
pub(in crate::runtime) fn specialize_image_warp_node(device: &Arc<Device>, node: &mut ComputeNode) {
	if node.jit.is_some() || !device.tile_image_enabled() {
		return;
	}
	let perspective = match node.kernel {
		KernelId::ImageWarpAffineF32 => false,
		KernelId::ImageWarpPerspectiveF32 => true,
		_ => return,
	};
	let [input, transform, output] = node.buffers.as_slice() else {
		return;
	};
	if input.access != BufferAccess::Read
		|| transform.access != BufferAccess::Read
		|| output.access != BufferAccess::Write
		|| node.push_constants.len() != 9
		|| !matches!(node.push_constants.last(), Some(PushConstant::F32(_)))
	{
		return;
	}
	let program = BlockProgram::image_warp(perspective);
	node.fusion_signature = Some(device.tile_signature(&program));
	let Some(key) = device.jit_admit_tile_image(&program) else {
		return;
	};
	node.jit = Some(JitNode {
		key,
		buffers: node.buffers.clone(),
		push_constants: node.push_constants.clone(),
	});
}

/// `OA_TILE_IMAGE` substitution for `ImageRemapF32` nodes — the map-fed
/// bordered gather: donor ABI `[input, map, output]` + nine u32 pushes +
/// f32 `border_value`.
pub(in crate::runtime) fn specialize_image_remap_node(
	device: &Arc<Device>,
	node: &mut ComputeNode,
) {
	if node.kernel != KernelId::ImageRemapF32 || node.jit.is_some() || !device.tile_image_enabled() {
		return;
	}
	let [input, map, output] = node.buffers.as_slice() else {
		return;
	};
	if input.access != BufferAccess::Read
		|| map.access != BufferAccess::Read
		|| output.access != BufferAccess::Write
		|| node.push_constants.len() != 10
		|| !matches!(node.push_constants.last(), Some(PushConstant::F32(_)))
	{
		return;
	}
	let program = BlockProgram::image_remap();
	node.fusion_signature = Some(device.tile_signature(&program));
	let Some(key) = device.jit_admit_tile_image(&program) else {
		return;
	};
	node.jit = Some(JitNode {
		key,
		buffers: node.buffers.clone(),
		push_constants: node.push_constants.clone(),
	});
}

/// `OA_TILE_IMAGE` substitution for `ImageFlipF32` nodes — the
/// index-remap gather: donor ABI `[input, output]` +
/// `[batch, channels, height, width, horizontal, vertical]` push.
pub(in crate::runtime) fn specialize_image_flip_node(device: &Arc<Device>, node: &mut ComputeNode) {
	if node.kernel != KernelId::ImageFlipF32 || node.jit.is_some() || !device.tile_image_enabled() {
		return;
	}
	let [input, output] = node.buffers.as_slice() else {
		return;
	};
	if input.access != BufferAccess::Read
		|| output.access != BufferAccess::Write
		|| node.push_constants.len() != 6
	{
		return;
	}
	let program = BlockProgram::image_flip();
	node.fusion_signature = Some(device.tile_signature(&program));
	let Some(key) = device.jit_admit_tile_image(&program) else {
		return;
	};
	node.jit = Some(JitNode {
		key,
		buffers: node.buffers.clone(),
		push_constants: node.push_constants.clone(),
	});
}

/// `OA_TILE_IMAGE` substitution for `ImageCropF32`/`ImageCenterCropF32`
/// and `ImageRotateF32` nodes — the pure index-remap gathers (OA-CC-33).
pub(in crate::runtime) fn specialize_image_remap_index_node(
	device: &Arc<Device>,
	node: &mut ComputeNode,
) {
	if node.jit.is_some() || !device.tile_image_enabled() {
		return;
	}
	let (program, pushes) = match node.kernel {
		KernelId::ImageCropF32 | KernelId::ImageCenterCropF32 => (BlockProgram::image_crop(), 8),
		KernelId::ImageRotateF32 => (BlockProgram::image_rotate(), 7),
		_ => return,
	};
	let [input, output] = node.buffers.as_slice() else {
		return;
	};
	if input.access != BufferAccess::Read
		|| output.access != BufferAccess::Write
		|| node.push_constants.len() != pushes
	{
		return;
	}
	node.fusion_signature = Some(device.tile_signature(&program));
	let Some(key) = device.jit_admit_tile_image(&program) else {
		return;
	};
	node.jit = Some(JitNode {
		key,
		buffers: node.buffers.clone(),
		push_constants: node.push_constants.clone(),
	});
}

/// `OA_TILE_IMAGE` substitution for the `filter.slang` window donors —
/// re-emitted as serial per-lane folds over `image_sample_bordered`
/// taps (OA-CC-33 filter anchor): the runtime-K conv, the uniform
/// blur, the nested separable conv, and the morphology Min/Max
/// family. `median_blur` stays incumbent-only — its insertion-sort
/// kernel does not re-express as a fold.
pub(in crate::runtime) fn specialize_image_filter_node(
	device: &Arc<Device>,
	node: &mut ComputeNode,
) {
	if node.jit.is_some() || !device.tile_image_enabled() {
		return;
	}
	let program = match node.kernel {
		KernelId::ImageConvolve2dF32 => BlockProgram::image_convolve(),
		KernelId::ImageSeparableConvolve2dF32 => BlockProgram::image_separable_convolve(),
		KernelId::ImageAverageBlurF32 => BlockProgram::image_average_blur(),
		KernelId::ImageErodeF32 => BlockProgram::image_morphology_once(false),
		KernelId::ImageDilateF32 => BlockProgram::image_morphology_once(true),
		KernelId::ImageMorphologyOpenF32 => BlockProgram::image_morphology_twice(false),
		KernelId::ImageMorphologyCloseF32 => BlockProgram::image_morphology_twice(true),
		KernelId::ImageMorphologyGradientF32 => BlockProgram::image_morphology_gradient(),
		KernelId::ImageMorphologyTopHatF32 => BlockProgram::image_morphology_hat(true),
		KernelId::ImageMorphologyBlackHatF32 => BlockProgram::image_morphology_hat(false),
		KernelId::ImageSobelF32 => BlockProgram::image_directional_linear(0x4000_0000, 0x3F80_0000),
		KernelId::ImageScharrF32 => BlockProgram::image_directional_linear(0x4120_0000, 0x4040_0000),
		KernelId::ImageLaplacianF32 => BlockProgram::image_laplacian(),
		KernelId::ImageSharpenF32 => BlockProgram::image_sharpen(),
		KernelId::ImageGaussianBlurF32 => BlockProgram::image_gaussian(),
		KernelId::ImageBilateralFilterF32 => BlockProgram::image_bilateral(),
		KernelId::ImageUnsharpMaskF32 => BlockProgram::image_unsharp(),
		KernelId::ImageAdaptiveThresholdMeanF32 => BlockProgram::image_adaptive_threshold(false),
		KernelId::ImageAdaptiveThresholdGaussianF32 => BlockProgram::image_adaptive_threshold(true),
		KernelId::ImageThresholdBinaryF32 => BlockProgram::image_pointwise(0),
		KernelId::ImageThresholdBinaryInvF32 => BlockProgram::image_pointwise(1),
		KernelId::ImageThresholdTruncateF32 => BlockProgram::image_pointwise(2),
		KernelId::ImageThresholdToZeroF32 => BlockProgram::image_pointwise(3),
		KernelId::ImageThresholdToZeroInvF32 => BlockProgram::image_pointwise(4),
		KernelId::ImageInRangeF32 => BlockProgram::image_pointwise(5),
		KernelId::ImageClampF32 => BlockProgram::image_pointwise(6),
		KernelId::ImageInvertF32 => BlockProgram::image_pointwise(7),
		KernelId::ImageBrightnessContrastF32 => BlockProgram::image_pointwise(8),
		KernelId::ImageGammaContrastF32 => BlockProgram::image_pointwise(9),
		KernelId::ImageSolarizeF32 => BlockProgram::image_pointwise(10),
		KernelId::ImagePosterizeF32 => BlockProgram::image_pointwise(11),
		KernelId::ImageSegmentationOverlayF32 => BlockProgram::image_segmentation_overlay(),
		KernelId::ImageNormalizeF32 => BlockProgram::image_normalize(),
		KernelId::ImageConvertColorF32 => BlockProgram::image_convert_color(),
		KernelId::ImageResizeNormalizeF32 => BlockProgram::image_resize_normalize(),
		KernelId::ImageGrayscaleF32 => BlockProgram::image_grayscale(),
		KernelId::ImageChannelReorderF32 => BlockProgram::image_channel_reorder(),
		KernelId::ImageColorTwistF32 => BlockProgram::image_color_twist(),
		KernelId::ImageAlphaBlendF32 | KernelId::ImageCompositeF32 | KernelId::ImageEraseF32 => {
			BlockProgram::image_composite()
		}
		_ => return,
	};
	// ABI per donor class: the window filters bind
	// `[input, kernel_x, kernel_y, output]` with 16 push words, the
	// flat pointwise map binds `[input, output]` with 6, and the
	// segmentation overlay binds `[image, mask, palette, output]`
	// with 6.
	let (buffers, pushes) = match node.kernel {
		KernelId::ImageSegmentationOverlayF32 => (4usize, 6usize),
		KernelId::ImageNormalizeF32 => (2, 10),
		KernelId::ImageConvertColorF32 => (2, 4),
		KernelId::ImageResizeNormalizeF32 => (2, 13),
		KernelId::ImageGrayscaleF32 => (2, 4),
		KernelId::ImageChannelReorderF32 => (2, 8),
		KernelId::ImageColorTwistF32 => (3, 4),
		KernelId::ImageAlphaBlendF32 | KernelId::ImageCompositeF32 | KernelId::ImageEraseF32 => (4, 12),
		KernelId::ImageThresholdBinaryF32
		| KernelId::ImageThresholdBinaryInvF32
		| KernelId::ImageThresholdTruncateF32
		| KernelId::ImageThresholdToZeroF32
		| KernelId::ImageThresholdToZeroInvF32
		| KernelId::ImageInRangeF32
		| KernelId::ImageClampF32
		| KernelId::ImageInvertF32
		| KernelId::ImageBrightnessContrastF32
		| KernelId::ImageGammaContrastF32
		| KernelId::ImageSolarizeF32
		| KernelId::ImagePosterizeF32 => (2, 6),
		_ => (4, 16),
	};
	if node.buffers.len() != buffers
		|| node.push_constants.len() != pushes
		|| node.buffers[0].access != BufferAccess::Read
		|| node.buffers[buffers - 1].access != BufferAccess::Write
	{
		return;
	}
	node.fusion_signature = Some(device.tile_signature(&program));
	let Some(key) = device.jit_admit_tile_image(&program) else {
		return;
	};
	node.jit = Some(JitNode {
		key,
		buffers: node.buffers.clone(),
		push_constants: node.push_constants.clone(),
	});
}

/// `OA_TILE_AUDIO` substitution for `AudioMelFilterbankF32` nodes —
/// the per-(mel, frame, channel) dot product re-emitted as a serial
/// per-lane fold over `freq_bins` with the donor's log-scale tail
/// (OA-CC-32): donor ABI `[input, output, filterbank]` + five u32
/// pushes, donor grid `[num_mels/32, frames, channels]`.
pub(in crate::runtime) fn specialize_audio_filterbank_node(
	device: &Arc<Device>,
	node: &mut ComputeNode,
) {
	if node.kernel != KernelId::AudioMelFilterbankF32 || node.jit.is_some() {
		return;
	}
	let [input, output, filterbank] = node.buffers.as_slice() else {
		return;
	};
	if input.access != BufferAccess::Read
		|| output.access != BufferAccess::Write
		|| filterbank.access != BufferAccess::Read
		|| !device.tile_audio_enabled()
	{
		return;
	}
	let [
		PushConstant::U32(_),
		PushConstant::U32(_),
		PushConstant::U32(_),
		PushConstant::U32(num_mels),
		PushConstant::U32(_),
	] = node.push_constants.as_slice()
	else {
		return;
	};
	if node.workgroups[0] != num_mels.div_ceil(32) || node.workgroups[1] < 1 || node.workgroups[2] < 1
	{
		return;
	}
	let program = BlockProgram::audio_mel_filterbank();
	node.fusion_signature = Some(device.tile_signature(&program));
	let Some(key) = device.jit_admit_tile_audio(&program) else {
		return;
	};
	node.jit = Some(JitNode {
		key,
		buffers: node.buffers.clone(),
		push_constants: node.push_constants.clone(),
	});
}

/// `OA_TILE_AUDIO` substitution for `AudioMelNormalizeF32` nodes —
/// the channel-wise mel normalization re-emitted as a row-cooperative
/// block program (OA-CC-32): one workgroup per channel, lanes striding
/// the channel run, shared-tree sum and square-sum reductions, donor
/// ABI `[input, output]` + `[channels, values_per_channel]` push.
pub(in crate::runtime) fn specialize_audio_normalize_node(
	device: &Arc<Device>,
	node: &mut ComputeNode,
) {
	if node.kernel != KernelId::AudioMelNormalizeF32 || node.jit.is_some() {
		return;
	}
	let [input, output] = node.buffers.as_slice() else {
		return;
	};
	if input.access != BufferAccess::Read
		|| output.access != BufferAccess::Write
		|| !device.tile_audio_enabled()
	{
		return;
	}
	let [PushConstant::U32(channels), PushConstant::U32(_values)] = node.push_constants.as_slice()
	else {
		return;
	};
	// Donor grid: one workgroup per channel.
	if node.workgroups != [*channels, 1, 1] {
		return;
	}
	let Ok(program) = BlockProgram::channel_normalize(256) else {
		return;
	};
	node.fusion_signature = Some(device.tile_signature(&program));
	let Some(key) = device.jit_admit_tile_audio(&program) else {
		return;
	};
	node.jit = Some(JitNode {
		key,
		buffers: node.buffers.clone(),
		push_constants: node.push_constants.clone(),
	});
}

/// Substitute an admitted `softmax` tile template for the incumbent
/// registry pipeline on one dispatch node (OA-CC-14). The emitted
/// program preserves the donor ABI — two buffers,
/// `[outer_size, dim_size, inner_size]` push scalars, one workgroup per
/// `(outer, inner)` slice — so `node.workgroups` is already correct and
/// stays untouched. Every failure mode keeps the canonical route.
pub(in crate::runtime) fn specialize_softmax_node(device: &Arc<Device>, node: &mut ComputeNode) {
	if node.kernel != KernelId::MatrixSoftmaxF32 || node.jit.is_some() {
		return;
	}
	let [input, output] = node.buffers.as_slice() else {
		return;
	};
	if input.access != BufferAccess::Read
		|| output.access != BufferAccess::Write
		|| !device.tile_softmax_enabled()
	{
		return;
	}
	let [
		PushConstant::U32(outer),
		PushConstant::U32(_dim),
		PushConstant::U32(inner),
	] = node.push_constants.as_slice()
	else {
		return;
	};
	// One workgroup per (outer, inner) slice — the donor's dispatch
	// shape; a different grid keeps the incumbent route.
	if node.workgroups != [outer.saturating_mul(*inner), 1, 1] {
		return;
	}
	let Ok(program) = BlockProgram::softmax(256) else {
		return;
	};
	node.fusion_signature = Some(device.tile_signature(&program));
	let Some(key) = device.jit_admit_tile_softmax(&program) else {
		return;
	};
	node.jit = Some(JitNode {
		key,
		buffers: node.buffers.clone(),
		push_constants: node.push_constants.clone(),
	});
}

/// `OA_TILE_LOG_SOFTMAX` substitution for `matrix.log_softmax` nodes —
/// same donor ABI as `softmax`.
pub(in crate::runtime) fn specialize_log_softmax_node(
	device: &Arc<Device>,
	node: &mut ComputeNode,
) {
	if node.kernel != KernelId::MatrixLogSoftmaxF32 || node.jit.is_some() {
		return;
	}
	let [input, output] = node.buffers.as_slice() else {
		return;
	};
	if input.access != BufferAccess::Read
		|| output.access != BufferAccess::Write
		|| !device.tile_log_softmax_enabled()
	{
		return;
	}
	let [
		PushConstant::U32(outer),
		PushConstant::U32(_dim),
		PushConstant::U32(inner),
	] = node.push_constants.as_slice()
	else {
		return;
	};
	if node.workgroups != [outer.saturating_mul(*inner), 1, 1] {
		return;
	}
	let Ok(program) = BlockProgram::log_softmax(256) else {
		return;
	};
	node.fusion_signature = Some(device.tile_signature(&program));
	let Some(key) = device.jit_admit_tile(&program) else {
		return;
	};
	node.jit = Some(JitNode {
		key,
		buffers: node.buffers.clone(),
		push_constants: node.push_constants.clone(),
	});
}

/// `OA_TILE_SOFTMAX_BACKWARD` substitution for `matrix.softmax`'s
/// adjoint nodes — same `(outer, inner)` dispatch shape as forward.
pub(in crate::runtime) fn specialize_softmax_backward_node(
	device: &Arc<Device>,
	node: &mut ComputeNode,
) {
	if node.kernel != KernelId::MatrixSoftmaxBackwardF32 || node.jit.is_some() {
		return;
	}
	let [forward_output, output_gradient, input_gradient] = node.buffers.as_slice() else {
		return;
	};
	if forward_output.access != BufferAccess::Read
		|| output_gradient.access != BufferAccess::Read
		|| input_gradient.access != BufferAccess::Write
		|| !device.tile_softmax_backward_enabled()
	{
		return;
	}
	let [
		PushConstant::U32(outer),
		PushConstant::U32(_dim),
		PushConstant::U32(inner),
	] = node.push_constants.as_slice()
	else {
		return;
	};
	if node.workgroups != [outer.saturating_mul(*inner), 1, 1] {
		return;
	}
	let Ok(program) = BlockProgram::softmax_backward(256) else {
		return;
	};
	node.fusion_signature = Some(device.tile_signature(&program));
	let Some(key) = device.jit_admit_tile(&program) else {
		return;
	};
	node.jit = Some(JitNode {
		key,
		buffers: node.buffers.clone(),
		push_constants: node.push_constants.clone(),
	});
}

/// `OA_TILE_LOG_SOFTMAX_BACKWARD` substitution for `matrix.log_softmax`'s
/// adjoint nodes.
pub(in crate::runtime) fn specialize_log_softmax_backward_node(
	device: &Arc<Device>,
	node: &mut ComputeNode,
) {
	if node.kernel != KernelId::MatrixLogSoftmaxBackwardF32 || node.jit.is_some() {
		return;
	}
	let [forward_output, output_gradient, input_gradient] = node.buffers.as_slice() else {
		return;
	};
	if forward_output.access != BufferAccess::Read
		|| output_gradient.access != BufferAccess::Read
		|| input_gradient.access != BufferAccess::Write
		|| !device.tile_log_softmax_backward_enabled()
	{
		return;
	}
	let [
		PushConstant::U32(outer),
		PushConstant::U32(_dim),
		PushConstant::U32(inner),
	] = node.push_constants.as_slice()
	else {
		return;
	};
	if node.workgroups != [outer.saturating_mul(*inner), 1, 1] {
		return;
	}
	let Ok(program) = BlockProgram::log_softmax_backward(256) else {
		return;
	};
	node.fusion_signature = Some(device.tile_signature(&program));
	let Some(key) = device.jit_admit_tile(&program) else {
		return;
	};
	node.jit = Some(JitNode {
		key,
		buffers: node.buffers.clone(),
		push_constants: node.push_constants.clone(),
	});
}

/// `OA_TILE_LINEAR` substitution for `ml.linear` nodes — the bias
/// epilogue template with the donor's `[batch, input, output]` push
/// ABI and `GroupX`-row dispatch orientation.
pub(in crate::runtime) fn specialize_linear_node(device: &Arc<Device>, node: &mut ComputeNode) {
	if node.kernel != KernelId::MlLinearF32 || node.jit.is_some() {
		return;
	}
	let [input, weight, bias, output] = node.buffers.as_slice() else {
		return;
	};
	if input.access != BufferAccess::Read
		|| weight.access != BufferAccess::Read
		|| bias.access != BufferAccess::Read
		|| output.access != BufferAccess::Write
		|| !device.tile_linear_enabled()
	{
		return;
	}
	let [
		PushConstant::U32(batch),
		PushConstant::U32(_in),
		PushConstant::U32(out),
	] = node.push_constants.as_slice()
	else {
		return;
	};
	let (tile_rows, tile_columns) = gemm_candidate(*batch, *out);
	let Ok(program) = BlockProgram::linear(tile_rows, tile_columns, 16, 4, 4, 4) else {
		return;
	};
	if node.workgroups != program.workgroups(*batch, *out) {
		return;
	}
	node.fusion_signature = Some(device.tile_signature(&program));
	let Some(key) = device.jit_admit_tile(&program) else {
		return;
	};
	node.jit = Some(JitNode {
		key,
		buffers: node.buffers.clone(),
		push_constants: node.push_constants.clone(),
	});
}

/// CC-12 multi-candidate selection over `(enabled, program)` pairs:
/// gated candidates are probed least-samples-first until each carries
/// `MIN_CANDIDATE_SAMPLES` measured specialized dispatches, then the
/// measured median minimum holds. A candidate whose probes keep
/// losing the incumbent hysteresis stops being re-probed after
/// `CANDIDATE_PROBE_BUDGET` admission attempts — the measured winner
/// holds instead of starving the route. Single-candidate inputs skip
/// the stats lookup entirely, keeping today's behavior.
fn tile_candidate(
	device: &Arc<Device>,
	candidates: Vec<(bool, crate::Result<BlockProgram>)>,
) -> Option<BlockProgram> {
	let candidates: Vec<BlockProgram> = candidates
		.into_iter()
		.filter_map(|(enabled, program)| enabled.then(|| program.ok()).flatten())
		.collect();
	let mut candidates = candidates.into_iter();
	let first = candidates.next()?;
	if candidates.len() == 0 {
		return Some(first);
	}
	let candidates: Vec<BlockProgram> = std::iter::once(first).chain(candidates).collect();
	let stats: Vec<(Option<u64>, usize, u64)> = candidates
		.iter()
		.map(|program| device.tile_route_stats(device.tile_signature(program)))
		.collect();
	select_tile_candidate(&stats).map(|index| candidates[index].clone())
}

/// The pure selection core for `tile_candidate` over
/// `(specialized p50 ns, contributing samples, admission attempts)`
/// rows in candidate order — probe the least-sampled candidate until
/// every candidate has `MIN_CANDIDATE_SAMPLES` measured dispatches or
/// has exhausted `CANDIDATE_PROBE_BUDGET` admission attempts, then
/// hold the measured median minimum; a fully declined set falls back
/// to the first candidate so the incumbent hysteresis can run tape.
pub(in crate::runtime) fn select_tile_candidate(
	candidates: &[(Option<u64>, usize, u64)],
) -> Option<usize> {
	if candidates.is_empty() {
		return None;
	}
	const MIN_CANDIDATE_SAMPLES: usize = 4;
	const CANDIDATE_PROBE_BUDGET: u64 = 16;
	let mut probing: Option<(usize, usize)> = None;
	let mut measured: Option<(u64, usize)> = None;
	for (index, &(p50, samples, records)) in candidates.iter().enumerate() {
		if samples >= MIN_CANDIDATE_SAMPLES
			&& let Some(p50) = p50
		{
			if measured.is_none_or(|(best, _)| p50 < best) {
				measured = Some((p50, index));
			}
			continue;
		}
		if records < CANDIDATE_PROBE_BUDGET && probing.is_none_or(|(least, _)| samples < least) {
			probing = Some((samples, index));
		}
	}
	probing
		.map(|(_, index)| index)
		.or_else(|| measured.map(|(_, index)| index))
		.or(Some(0))
}

/// `OA_TILE_SUM` substitution for `matrix.sum` nodes — the single
/// workgroup cooperative reduce preserves the `[1,1,1]` dispatch and
/// `[element_count]` push ABI.
pub(in crate::runtime) fn specialize_sum_node(device: &Arc<Device>, node: &mut ComputeNode) {
	if node.kernel != KernelId::MatrixSumF32 || node.jit.is_some() {
		return;
	}
	let [input, output] = node.buffers.as_slice() else {
		return;
	};
	if input.access != BufferAccess::Read
		|| output.access != BufferAccess::Write
		|| !(device.tile_sum_enabled() || device.tile_sum_wave_enabled())
	{
		return;
	}
	let [PushConstant::U32(_count)] = node.push_constants.as_slice() else {
		return;
	};
	if node.workgroups != [1, 1, 1] {
		return;
	}
	// The wave candidate changes sum's reduction order versus the
	// shared tree — `tile_reduce_wave_offered` applies the per-kernel
	// flag, the explicit `OA_TILE_REDUCE_ORDER` mode, and the queried
	// subgroup arithmetic support.
	let wave_sum = device.tile_reduce_wave_offered(device.tile_sum_wave_enabled());
	let Some(program) = tile_candidate(
		device,
		vec![
			(device.tile_sum_enabled(), BlockProgram::sum(256)),
			(wave_sum, BlockProgram::sum_wave(256)),
		],
	) else {
		return;
	};
	node.fusion_signature = Some(device.tile_signature(&program));
	let Some(key) = device.jit_admit_tile(&program) else {
		return;
	};
	node.jit = Some(JitNode {
		key,
		buffers: node.buffers.clone(),
		push_constants: node.push_constants.clone(),
	});
}

/// `OA_TILE_SUM_AXIS` substitution for keep-dim `matrix.sum` nodes —
/// the donor's flat-thread schedule preserves the
/// `ceil(outer*inner/256)` dispatch, so the emitted program must also
/// use 256 lanes.
pub(in crate::runtime) fn specialize_sum_axis_node(device: &Arc<Device>, node: &mut ComputeNode) {
	if node.kernel != KernelId::MatrixSumAxisF32 || node.jit.is_some() {
		return;
	}
	let [input, output] = node.buffers.as_slice() else {
		return;
	};
	if input.access != BufferAccess::Read
		|| output.access != BufferAccess::Write
		|| !device.tile_sum_axis_enabled()
	{
		return;
	}
	let [
		PushConstant::U32(outer),
		PushConstant::U32(dim),
		PushConstant::U32(inner),
	] = node.push_constants.as_slice()
	else {
		return;
	};
	let slices = outer.saturating_mul(*inner);
	if node.workgroups != [slices.div_ceil(256), 1, 1] {
		return;
	}
	// Variant selected by row length: once the reduced axis holds at
	// least two elements per lane, one workgroup per `(outer, inner)`
	// slice parallelizes the serial scan the flat donor assigns to a
	// single thread. The slice grid must fit the dispatch-x bound.
	let cooperative = *dim >= 2 * 256 && slices <= u32::from(u16::MAX);
	let Ok(program) = (if cooperative {
		BlockProgram::sum_axis_cooperative(256)
	} else {
		BlockProgram::sum_axis(256)
	}) else {
		return;
	};
	node.fusion_signature = Some(device.tile_signature(&program));
	let Some(key) = device.jit_admit_tile(&program) else {
		return;
	};
	node.jit = Some(JitNode {
		key,
		buffers: node.buffers.clone(),
		push_constants: node.push_constants.clone(),
	});
	if cooperative {
		node.workgroups = [slices, 1, 1];
	}
}

/// `OA_TILE_LINEAR_TRIPLE` substitution for fused `MlLinearTripleF32`
/// nodes — ten buffers `[input, w0..w2, b0..b2, o0..o2]`, donor push
/// `[batch, input, output]` and a z=3 grid.
pub(in crate::runtime) fn specialize_linear_triple_node(
	device: &Arc<Device>,
	node: &mut ComputeNode,
) {
	if node.kernel != KernelId::MlLinearTripleF32 || node.jit.is_some() {
		return;
	}
	if node.buffers.len() != 10
		|| node.buffers[..7]
			.iter()
			.any(|usage| usage.access != BufferAccess::Read)
		|| node.buffers[7..]
			.iter()
			.any(|usage| usage.access != BufferAccess::Write)
		|| !device.tile_linear_triple_enabled()
	{
		return;
	}
	let [
		PushConstant::U32(batch),
		PushConstant::U32(_in),
		PushConstant::U32(out),
	] = node.push_constants.as_slice()
	else {
		return;
	};
	let (tile_rows, tile_columns) = gemm_candidate(*batch, *out);
	let Ok(program) = BlockProgram::linear_triple(tile_rows, tile_columns, 16, 4, 4, 4) else {
		return;
	};
	if node.workgroups[..2] != program.workgroups(*batch, *out)[..2] || node.workgroups[2] != 3 {
		return;
	}
	node.fusion_signature = Some(device.tile_signature(&program));
	let Some(key) = device.jit_admit_tile(&program) else {
		return;
	};
	node.jit = Some(JitNode {
		key,
		buffers: node.buffers.clone(),
		push_constants: node.push_constants.clone(),
	});
}

/// `OA_TILE_QKV` substitution for `MlQkvProjectionBiasF32` DNN
/// provider nodes — ten buffers `[input, qw, kw, vw, qb, kb, vb, qout,
/// kout, vout]`, donor push `[m, n, k]` and the donor's 32-column grid.
pub(in crate::runtime) fn specialize_qkv_projection_node(
	device: &Arc<Device>,
	node: &mut ComputeNode,
) {
	if node.kernel != KernelId::MlQkvProjectionBiasF32 || node.jit.is_some() {
		return;
	}
	if node.buffers.len() != 10
		|| node.buffers[..7]
			.iter()
			.any(|usage| usage.access != BufferAccess::Read)
		|| node.buffers[7..]
			.iter()
			.any(|usage| usage.access != BufferAccess::Write)
		|| !device.tile_qkv_enabled()
	{
		return;
	}
	let [
		PushConstant::U32(m),
		PushConstant::U32(n),
		PushConstant::U32(_k),
	] = node.push_constants.as_slice()
	else {
		return;
	};
	let Ok(program) = BlockProgram::qkv_projection_bias(32, 32, 32, 1, 4, 1) else {
		return;
	};
	if node.workgroups != program.workgroups(*m, *n) {
		return;
	}
	node.fusion_signature = Some(device.tile_signature(&program));
	let Some(key) = device.jit_admit_tile(&program) else {
		return;
	};
	node.jit = Some(JitNode {
		key,
		buffers: node.buffers.clone(),
		push_constants: node.push_constants.clone(),
	});
}

/// `OA_TILE_GATE_UP` substitution for `MlGateUpSwigluBiasF32` DNN
/// provider nodes — six buffers `[input, gate_w, up_w, gate_b, up_b,
/// output]`, donor push `[m, n, k]` and the donor's 32-column grid.
pub(in crate::runtime) fn specialize_gate_up_node(device: &Arc<Device>, node: &mut ComputeNode) {
	if node.kernel != KernelId::MlGateUpSwigluBiasF32 || node.jit.is_some() {
		return;
	}
	if node.buffers.len() != 6
		|| node.buffers[..5]
			.iter()
			.any(|usage| usage.access != BufferAccess::Read)
		|| node.buffers[5].access != BufferAccess::Write
		|| !device.tile_gate_up_enabled()
	{
		return;
	}
	let [
		PushConstant::U32(m),
		PushConstant::U32(n),
		PushConstant::U32(_k),
	] = node.push_constants.as_slice()
	else {
		return;
	};
	let Ok(program) = BlockProgram::gate_up_swiglu_bias(32, 32, 32, 1, 4, 1) else {
		return;
	};
	if node.workgroups != program.workgroups(*m, *n) {
		return;
	}
	node.fusion_signature = Some(device.tile_signature(&program));
	let Some(key) = device.jit_admit_tile(&program) else {
		return;
	};
	node.jit = Some(JitNode {
		key,
		buffers: node.buffers.clone(),
		push_constants: node.push_constants.clone(),
	});
}

/// `OA_TILE_RESIDUAL_NORM` substitution for fused
/// `MlLinearResidualNormF32` nodes — eleven buffers `[input, weight,
/// bias, residual, norm_weight, norm_bias, linear_out, sum_out, output,
/// normalized, inv_stddev]`, donor push `[batch, input, output]` plus
/// f32 `epsilon`. The in-tile norm needs one column tile to cover the
/// whole row, so the route only substitutes when `output` fits inside
/// `tile_columns` — the matcher already caps it at 64.
pub(in crate::runtime) fn specialize_linear_residual_norm_node(
	device: &Arc<Device>,
	node: &mut ComputeNode,
) {
	if node.kernel != KernelId::MlLinearResidualNormF32 || node.jit.is_some() {
		return;
	}
	if node.buffers.len() != 11
		|| node.buffers[..6]
			.iter()
			.any(|usage| usage.access != BufferAccess::Read)
		|| node.buffers[6..]
			.iter()
			.any(|usage| usage.access != BufferAccess::Write)
		|| !device.tile_residual_norm_enabled()
	{
		return;
	}
	let [
		PushConstant::U32(batch),
		PushConstant::U32(_in),
		PushConstant::U32(out),
		PushConstant::F32(_epsilon),
	] = node.push_constants.as_slice()
	else {
		return;
	};
	let (tile_rows, tile_columns) = gemm_candidate(*batch, *out);
	if *out > tile_columns {
		return;
	}
	let Ok(program) = BlockProgram::linear_residual_norm(tile_rows, tile_columns, 16, 4, 4, 4) else {
		return;
	};
	if node.workgroups != program.workgroups(*batch, *out) {
		return;
	}
	node.fusion_signature = Some(device.tile_signature(&program));
	let Some(key) = device.jit_admit_tile(&program) else {
		return;
	};
	node.jit = Some(JitNode {
		key,
		buffers: node.buffers.clone(),
		push_constants: node.push_constants.clone(),
	});
}

/// `OA_TILE_GROUPED_LINEAR` substitution for `MlGroupedLinearMF32`
/// nodes — five buffers `[input, weight, bias, offsets, output]`,
/// donor push `[rows, outputs, inner, experts]` and the donor's fixed
/// `[rows/32, outputs/32, 1]` grid.
pub(in crate::runtime) fn specialize_grouped_linear_node(
	device: &Arc<Device>,
	node: &mut ComputeNode,
) {
	if node.kernel != KernelId::MlGroupedLinearMF32 || node.jit.is_some() {
		return;
	}
	if node.buffers.len() != 5
		|| node.buffers[..4]
			.iter()
			.any(|usage| usage.access != BufferAccess::Read)
		|| node.buffers[4].access != BufferAccess::Write
		|| !device.tile_grouped_linear_enabled()
	{
		return;
	}
	let [
		PushConstant::U32(rows),
		PushConstant::U32(outputs),
		PushConstant::U32(_inner),
		PushConstant::U32(_experts),
	] = node.push_constants.as_slice()
	else {
		return;
	};
	let Ok(program) = BlockProgram::grouped_linear_m() else {
		return;
	};
	if node.workgroups != program.workgroups(*rows, *outputs) {
		return;
	}
	node.fusion_signature = Some(device.tile_signature(&program));
	let Some(key) = device.jit_admit_tile(&program) else {
		return;
	};
	node.jit = Some(JitNode {
		key,
		buffers: node.buffers.clone(),
		push_constants: node.push_constants.clone(),
	});
}

/// `OA_TILE_GROUPED_GEMM` substitution for `MlGroupedGemmMF32` nodes —
/// four buffers `[input, weight, offsets, output]` and the same
/// push/grid contract as `grouped_linear_m`.
pub(in crate::runtime) fn specialize_grouped_gemm_node(
	device: &Arc<Device>,
	node: &mut ComputeNode,
) {
	if node.kernel != KernelId::MlGroupedGemmMF32 || node.jit.is_some() {
		return;
	}
	if node.buffers.len() != 4
		|| node.buffers[..3]
			.iter()
			.any(|usage| usage.access != BufferAccess::Read)
		|| node.buffers[3].access != BufferAccess::Write
		|| !device.tile_grouped_gemm_enabled()
	{
		return;
	}
	let [
		PushConstant::U32(rows),
		PushConstant::U32(outputs),
		PushConstant::U32(_inner),
		PushConstant::U32(_experts),
	] = node.push_constants.as_slice()
	else {
		return;
	};
	let Ok(program) = BlockProgram::grouped_gemm_m() else {
		return;
	};
	if node.workgroups != program.workgroups(*rows, *outputs) {
		return;
	}
	node.fusion_signature = Some(device.tile_signature(&program));
	let Some(key) = device.jit_admit_tile(&program) else {
		return;
	};
	node.jit = Some(JitNode {
		key,
		buffers: node.buffers.clone(),
		push_constants: node.push_constants.clone(),
	});
}

/// `OA_TILE_GROUPED_BACKWARD` substitution for the grouped
/// data-adjoint nodes — `MlGroupedLinearMDataBackwardF32` and
/// `MlGroupedGemmMDataBackwardF32` share one donor body and the same
/// four-buffer `[output_gradient, weight, offsets, input_gradient]`
/// ABI, donor push `[rows, outputs, inputs, experts]` and the donor's
/// fixed `[rows/32, inputs/32, 1]` grid.
pub(in crate::runtime) fn specialize_grouped_data_backward_node(
	device: &Arc<Device>,
	node: &mut ComputeNode,
) {
	if !matches!(
		node.kernel,
		KernelId::MlGroupedLinearMDataBackwardF32 | KernelId::MlGroupedGemmMDataBackwardF32
	) || node.jit.is_some()
	{
		return;
	}
	if node.buffers.len() != 4
		|| node.buffers[..3]
			.iter()
			.any(|usage| usage.access != BufferAccess::Read)
		|| node.buffers[3].access != BufferAccess::Write
		|| !device.tile_grouped_backward_enabled()
	{
		return;
	}
	let [
		PushConstant::U32(rows),
		PushConstant::U32(_outputs),
		PushConstant::U32(inputs),
		PushConstant::U32(_experts),
	] = node.push_constants.as_slice()
	else {
		return;
	};
	let Ok(program) = BlockProgram::grouped_linear_m_data_backward() else {
		return;
	};
	if node.workgroups != program.workgroups(*rows, *inputs) {
		return;
	}
	node.fusion_signature = Some(device.tile_signature(&program));
	let Some(key) = device.jit_admit_tile(&program) else {
		return;
	};
	node.jit = Some(JitNode {
		key,
		buffers: node.buffers.clone(),
		push_constants: node.push_constants.clone(),
	});
}

/// Kernels that can produce the `dy` operand of a fused
/// `grouped_backward_prologue` pair — the producer half of
/// `lower_grouped_backward_prologue`'s vocabulary: activation adjoints,
/// the gated `silu_mul` adjoint, and the pointwise combines. Eager tile
/// specialization must not attach `jit` to a grouped data adjoint
/// following one of these — a jit node is an opaque effect boundary to
/// the fusion region proof and would suppress the fused prologue route
/// entirely.
pub(in crate::runtime) fn pairs_with_backward_prologue(kernel: KernelId) -> bool {
	matches!(
		kernel,
		KernelId::MlReluBackwardF32
			| KernelId::MlGeluBackwardF32
			| KernelId::MlSiluBackwardF32
			| KernelId::MlTanhBackwardF32
			| KernelId::MlSigmoidBackwardF32
			| KernelId::MlMishBackwardF32
			| KernelId::MlEluBackwardF32
			| KernelId::MlLeakyReluBackwardF32
			| KernelId::MlSoftplusBackwardF32
			| KernelId::MlSiluMulBackwardF32
			| KernelId::MatrixAddF32
			| KernelId::MatrixSubF32
			| KernelId::MatrixMulF32
			| KernelId::MatrixMulBroadcastF32
	)
}

/// Kernels that may produce the `dy` input a `MlLinearBackwardF32`
/// consumer consumes through the `LinearBackwardBmmChain` scatter
/// merge — the two 16×16 bmm producers the matcher accepts.
pub(in crate::runtime) fn pairs_with_backward_bmm(kernel: KernelId) -> bool {
	matches!(
		kernel,
		KernelId::MlBmmTiled16F32 | KernelId::MlBmmTnTiled16F32
	)
}

/// `OA_TILE_GROUPED_BACKWARD` substitution for the fused
/// `grouped_backward_prologue` node — the pair-fused
/// producer+data-adjoint carries `[upstream, saved, dy, weight,
/// offsets, input_gradient]` buffers (writes interleaved at slots 2
/// and 5), a `[rows, outputs, inputs, experts, prologue_kind, alpha]`
/// push block and a one-dimensional `ceil(rows/32)` grid — the fused
/// kernel walks column groups and expert segments inside the
/// workgroup.
pub(in crate::runtime) fn specialize_grouped_backward_prologue_node(
	device: &Arc<Device>,
	node: &mut ComputeNode,
) {
	if node.kernel != KernelId::MlGroupedBackwardPrologueF32
		|| node.jit.is_some()
		|| !device.tile_grouped_backward_enabled()
	{
		return;
	}
	let expected_access = [
		BufferAccess::Read,
		BufferAccess::Read,
		BufferAccess::Write,
		BufferAccess::Read,
		BufferAccess::Read,
		BufferAccess::Write,
	];
	if node.buffers.len() != expected_access.len()
		|| !node
			.buffers
			.iter()
			.zip(expected_access)
			.all(|(usage, access)| usage.access == access)
	{
		return;
	}
	let [
		PushConstant::U32(rows),
		PushConstant::U32(_),
		PushConstant::U32(_),
		PushConstant::U32(_),
		PushConstant::U32(_),
		PushConstant::F32(_),
	] = node.push_constants.as_slice()
	else {
		return;
	};
	let Ok(program) = BlockProgram::grouped_backward_prologue() else {
		return;
	};
	if node.workgroups != [rows.div_ceil(32), 1, 1] {
		return;
	}
	node.fusion_signature = Some(device.tile_signature(&program));
	let Some(key) = device.jit_admit_tile(&program) else {
		return;
	};
	node.jit = Some(JitNode {
		key,
		buffers: node.buffers.clone(),
		push_constants: node.push_constants.clone(),
	});
}

/// `OA_TILE_GROUPED_BACKWARD` substitution for the grouped
/// parameter-adjoint nodes — `MlGroupedLinearMParameterBackwardF32`
/// (five buffers, `dB` output) and `MlGroupedGemmMParameterBackwardF32`
/// (four buffers) share the z-dispatched `[outputs/32, inputs/32,
/// experts]` grid and `[rows, outputs, inputs, experts]` push.
pub(in crate::runtime) fn specialize_grouped_parameter_backward_node(
	device: &Arc<Device>,
	node: &mut ComputeNode,
) {
	let linear = node.kernel == KernelId::MlGroupedLinearMParameterBackwardF32;
	if !linear && node.kernel != KernelId::MlGroupedGemmMParameterBackwardF32 || node.jit.is_some() {
		return;
	}
	let buffers = if linear { 5 } else { 4 };
	if node.buffers.len() != buffers
		|| node.buffers[..3]
			.iter()
			.any(|usage| usage.access != BufferAccess::Read)
		|| node.buffers[3..]
			.iter()
			.any(|usage| usage.access != BufferAccess::Write)
		|| !device.tile_grouped_backward_enabled()
	{
		return;
	}
	let [
		PushConstant::U32(_rows),
		PushConstant::U32(outputs),
		PushConstant::U32(inputs),
		PushConstant::U32(experts),
	] = node.push_constants.as_slice()
	else {
		return;
	};
	let Ok(program) = (if linear {
		BlockProgram::grouped_linear_m_parameter_backward()
	} else {
		BlockProgram::grouped_gemm_m_parameter_backward()
	}) else {
		return;
	};
	let [x, y, _] = program.workgroups(*outputs, *inputs);
	if node.workgroups != [x, y, *experts] {
		return;
	}
	node.fusion_signature = Some(device.tile_signature(&program));
	let Some(key) = device.jit_admit_tile(&program) else {
		return;
	};
	node.jit = Some(JitNode {
		key,
		buffers: node.buffers.clone(),
		push_constants: node.push_constants.clone(),
	});
}

/// `OA_TILE_GROUPED_LINEAR_ACT` substitution for fused
/// `MlGroupedLinearMActF32` nodes — six buffers
/// `[input, weight, bias, offsets, output, act_output]`, donor push
/// `[rows, outputs, inner, experts, act_kind]` and the donor's fixed
/// `[rows/32, (outputs/2)/32, 1]` grid.
pub(in crate::runtime) fn specialize_grouped_linear_act_node(
	device: &Arc<Device>,
	node: &mut ComputeNode,
) {
	if node.kernel != KernelId::MlGroupedLinearMActF32 || node.jit.is_some() {
		return;
	}
	if node.buffers.len() != 6
		|| node.buffers[..4]
			.iter()
			.any(|usage| usage.access != BufferAccess::Read)
		|| node.buffers[4..]
			.iter()
			.any(|usage| usage.access != BufferAccess::Write)
		|| !device.tile_grouped_linear_act_enabled()
	{
		return;
	}
	let [
		PushConstant::U32(rows),
		PushConstant::U32(outputs),
		PushConstant::U32(_inner),
		PushConstant::U32(_experts),
		PushConstant::U32(9),
	] = node.push_constants.as_slice()
	else {
		return;
	};
	if *outputs == 0 || !outputs.is_multiple_of(2) {
		return;
	}
	let Ok(program) = BlockProgram::grouped_linear_m_act() else {
		return;
	};
	if node.workgroups != program.workgroups(*rows, *outputs / 2) {
		return;
	}
	node.fusion_signature = Some(device.tile_signature(&program));
	let Some(key) = device.jit_admit_tile(&program) else {
		return;
	};
	node.jit = Some(JitNode {
		key,
		buffers: node.buffers.clone(),
		push_constants: node.push_constants.clone(),
	});
}

/// `OA_TILE_LINEAR_CONCAT` substitution for fused `MlLinearConcatF32`
/// nodes — seven buffers `[src0, src1, src2, weight, bias, concat,
/// output]`, donor push `[batch, inner, output, w0, w1, w2]` and the
/// linear node's own dispatch grid.
pub(in crate::runtime) fn specialize_linear_concat_node(
	device: &Arc<Device>,
	node: &mut ComputeNode,
) {
	if node.kernel != KernelId::MlLinearConcatF32 || node.jit.is_some() {
		return;
	}
	if node.buffers.len() != 7
		|| node.buffers[..5]
			.iter()
			.any(|usage| usage.access != BufferAccess::Read)
		|| node.buffers[5..]
			.iter()
			.any(|usage| usage.access != BufferAccess::Write)
		|| !device.tile_linear_concat_enabled()
	{
		return;
	}
	let [
		PushConstant::U32(batch),
		PushConstant::U32(_inner),
		PushConstant::U32(out),
		PushConstant::U32(_w0),
		PushConstant::U32(_w1),
		PushConstant::U32(_w2),
	] = node.push_constants.as_slice()
	else {
		return;
	};
	let (tile_rows, tile_columns) = gemm_candidate(*batch, *out);
	let Ok(program) = BlockProgram::linear_concat(tile_rows, tile_columns, 16, 4, 4, 4) else {
		return;
	};
	if node.workgroups != program.workgroups(*batch, *out) {
		return;
	}
	node.fusion_signature = Some(device.tile_signature(&program));
	let Some(key) = device.jit_admit_tile(&program) else {
		return;
	};
	node.jit = Some(JitNode {
		key,
		buffers: node.buffers.clone(),
		push_constants: node.push_constants.clone(),
	});
}

/// A GEMM-family template constructor `(tile_rows, tile_columns,
/// tile_k, fragment_rows, fragment_columns, vec)`.
type GemmTemplate = fn(u32, u32, u32, u32, u32, u32) -> crate::Result<BlockProgram>;

/// `OA_TILE_LINEAR_ACT` substitution for the fused `linear` epilogue
/// family — `MlLinearActF32`, `MlLinearResidualF32` and
/// `MlLinearResidualActF32` share the donor's
/// `[input, weight, bias, residual?, linear_out, sum_out?, act_out?]`
/// binding order and `[batch, input, output, act_kind?, alpha?]` push
/// ABI. Fused nodes only exist after `fuse_epilogue_pairs`, so this
/// hook runs at the fusion push sites rather than `from_dispatches`.
/// `kind 9` (gated silu-mul) and the residual-norm pair are different
/// donors and stay incumbent.
pub(in crate::runtime) fn specialize_fused_linear_node(
	device: &Arc<Device>,
	node: &mut ComputeNode,
) {
	if node.jit.is_some() || !device.tile_linear_act_enabled() {
		return;
	}
	let (reads, writes, scalars, halved, template): (usize, usize, usize, bool, GemmTemplate) =
		match node.kernel {
			KernelId::MlLinearActF32 => (3usize, 2usize, 5usize, false, BlockProgram::linear_act),
			KernelId::MlLinearResidualF32 => (4, 2, 3, false, BlockProgram::linear_residual),
			KernelId::MlLinearResidualActF32 => (4, 3, 5, false, BlockProgram::linear_residual_act),
			KernelId::MlLinearGatedActF32 => (3, 2, 5, true, BlockProgram::linear_gated_act),
			_ => return,
		};
	if node.buffers.len() != reads + writes || node.push_constants.len() != scalars {
		return;
	}
	if node.buffers[..reads]
		.iter()
		.any(|usage| usage.access != BufferAccess::Read)
		|| node.buffers[reads..]
			.iter()
			.any(|usage| usage.access != BufferAccess::Write)
	{
		return;
	}
	let (PushConstant::U32(batch), PushConstant::U32(out)) =
		(&node.push_constants[0], &node.push_constants[2])
	else {
		return;
	};
	let Some(columns) = (if halved {
		(*out).checked_div(2)
	} else {
		Some(*out)
	})
	.filter(|columns| *columns > 0) else {
		return;
	};
	let (tile_rows, tile_columns) = gemm_candidate(*batch, columns);
	let Ok(program) = template(tile_rows, tile_columns, 16, 4, 4, 4) else {
		return;
	};
	if node.workgroups != program.workgroups(*batch, columns) {
		return;
	}
	node.fusion_signature = Some(device.tile_signature(&program));
	let Some(key) = device.jit_admit_tile(&program) else {
		return;
	};
	node.jit = Some(JitNode {
		key,
		buffers: node.buffers.clone(),
		push_constants: node.push_constants.clone(),
	});
}

/// `OA_TILE_RMS_NORM` substitution for `ml.rms_norm` nodes.
pub(in crate::runtime) fn specialize_rms_norm_node(device: &Arc<Device>, node: &mut ComputeNode) {
	if node.kernel != KernelId::MlRmsNormF32 || node.jit.is_some() {
		return;
	}
	let [input, weight, output] = node.buffers.as_slice() else {
		return;
	};
	if input.access != BufferAccess::Read
		|| weight.access != BufferAccess::Read
		|| output.access != BufferAccess::Write
		|| !device.tile_rms_norm_enabled()
	{
		return;
	}
	let [
		PushConstant::U32(rows),
		PushConstant::U32(_columns),
		PushConstant::F32(_epsilon),
	] = node.push_constants.as_slice()
	else {
		return;
	};
	if node.workgroups != [*rows, 1, 1] {
		return;
	}
	let Ok(program) = BlockProgram::rms_norm(256) else {
		return;
	};
	node.fusion_signature = Some(device.tile_signature(&program));
	let Some(key) = device.jit_admit_tile(&program) else {
		return;
	};
	node.jit = Some(JitNode {
		key,
		buffers: node.buffers.clone(),
		push_constants: node.push_constants.clone(),
	});
}

/// `OA_TILE_LAYER_NORM` substitution for `ml.layer_norm` nodes.
pub(in crate::runtime) fn specialize_layer_norm_node(device: &Arc<Device>, node: &mut ComputeNode) {
	if node.kernel != KernelId::MlLayerNormF32 || node.jit.is_some() {
		return;
	}
	let [input, weight, bias, output, normalized, inverse_stddev] = node.buffers.as_slice() else {
		return;
	};
	if input.access != BufferAccess::Read
		|| weight.access != BufferAccess::Read
		|| bias.access != BufferAccess::Read
		|| output.access != BufferAccess::Write
		|| normalized.access != BufferAccess::Write
		|| inverse_stddev.access != BufferAccess::Write
		|| !device.tile_layer_norm_enabled()
	{
		return;
	}
	let [
		PushConstant::U32(rows),
		PushConstant::U32(_columns),
		PushConstant::F32(_epsilon),
	] = node.push_constants.as_slice()
	else {
		return;
	};
	if node.workgroups != [*rows, 1, 1] {
		return;
	}
	let Ok(program) = BlockProgram::layer_norm(256) else {
		return;
	};
	node.fusion_signature = Some(device.tile_signature(&program));
	let Some(key) = device.jit_admit_tile(&program) else {
		return;
	};
	node.jit = Some(JitNode {
		key,
		buffers: node.buffers.clone(),
		push_constants: node.push_constants.clone(),
	});
}

/// `OA_TILE_MOE` substitution for the fused `MlMoeRouterSelectF32`
/// router node — six buffers `[logits, selection, probabilities,
/// values, indices, mask]`, push `[rows, experts, k]` and the donor's
/// `[rows, 1, 1]` row-per-workgroup grid.
pub(in crate::runtime) fn specialize_router_select_node(
	device: &Arc<Device>,
	node: &mut ComputeNode,
) {
	if node.kernel != KernelId::MlMoeRouterSelectF32 || node.jit.is_some() {
		return;
	}
	if node.buffers.len() != 6
		|| node.buffers[..2]
			.iter()
			.any(|usage| usage.access != BufferAccess::Read)
		|| node.buffers[2..]
			.iter()
			.any(|usage| usage.access != BufferAccess::Write)
		|| !device.tile_moe_enabled()
	{
		return;
	}
	let [
		PushConstant::U32(rows),
		PushConstant::U32(_experts),
		PushConstant::U32(_k),
	] = node.push_constants.as_slice()
	else {
		return;
	};
	let program = BlockProgram::moe_router_select();
	if node.workgroups != program.workgroups(*rows, 1) {
		return;
	}
	node.fusion_signature = Some(device.tile_signature(&program));
	let Some(key) = device.jit_admit_tile(&program) else {
		return;
	};
	node.jit = Some(JitNode {
		key,
		buffers: node.buffers.clone(),
		push_constants: node.push_constants.clone(),
	});
}

/// `OA_TILE_MOE` substitution for the flat per-thread MoE plumbing
/// donors. Each candidate keeps the donor's 256-lane dispatch-index
/// schedule: block IR re-emits the same `group_id * 256 + lane`
/// indexing with the donor's serial route/column loops. Admission is
/// an exact-ABI substitution, not a cooperative port — none of these
/// programs share workgroup state.
pub(in crate::runtime) fn specialize_moe_plumbing_node(
	device: &Arc<Device>,
	node: &mut ComputeNode,
) {
	if node.jit.is_some() || !device.tile_moe_enabled() {
		return;
	}
	let (program, reads, scalar_count) = match node.kernel {
		KernelId::MlMoeGatherF32 => (BlockProgram::moe_gather(), 2, 2),
		KernelId::MlMoeGatherBackwardF32 => (BlockProgram::moe_gather_backward(), 2, 4),
		KernelId::MlMoeCombineF32 => (BlockProgram::moe_combine(), 3, 4),
		KernelId::MlMoeCombineBackwardF32 => (BlockProgram::moe_combine_backward(), 5, 4),
		KernelId::MlMoeRouteWeightsF32 => (BlockProgram::moe_route_weights(), 2, 3),
		KernelId::MlMoeRouteWeightsBackwardF32 => (BlockProgram::moe_route_weights_backward(), 4, 3),
		_ => return,
	};
	if node.buffers.len() != reads + (program.bindings.len() - reads)
		|| node.push_constants.len() != scalar_count
		|| node.buffers[..reads]
			.iter()
			.any(|usage| usage.access != BufferAccess::Read)
		|| node.buffers[reads..]
			.iter()
			.any(|usage| usage.access != BufferAccess::Write)
	{
		return;
	}
	node.fusion_signature = Some(device.tile_signature(&program));
	let Some(key) = device.jit_admit_tile(&program) else {
		return;
	};
	node.jit = Some(JitNode {
		key,
		buffers: node.buffers.clone(),
		push_constants: node.push_constants.clone(),
	});
}

/// `OA_TILE_CONV` substitution for the conv1d implicit-GEMM prologue
/// — `MlConv1dIm2ColF32` keeps the donor's flat `ceil(count/256)`
/// grid, `[input, columns]` binding and eight-scalar ABI, so the
/// substitution needs no grid rewrite.
pub(in crate::runtime) fn specialize_conv_node(device: &Arc<Device>, node: &mut ComputeNode) {
	if node.jit.is_some() || !device.tile_conv_enabled() {
		return;
	}
	let (program, accesses, scalar_count) = match node.kernel {
		KernelId::MlConv1dIm2ColF32 | KernelId::MlConvTranspose1dBackwardF32 => (
			BlockProgram::conv_1d_im2_col(),
			&[BufferAccess::Read, BufferAccess::Write][..],
			8,
		),
		KernelId::MlConv1dBackwardF32 => (
			BlockProgram::conv_1d_backward(),
			&[BufferAccess::Read, BufferAccess::Read, BufferAccess::Write][..],
			9,
		),
		KernelId::MlConvTranspose1dF32 => (
			BlockProgram::conv_transpose_1d(),
			&[BufferAccess::Read, BufferAccess::Read, BufferAccess::Write][..],
			9,
		),
		KernelId::MlConv2dBackwardF32 => (
			BlockProgram::conv_2d_backward(),
			&[BufferAccess::Read, BufferAccess::Read, BufferAccess::Write][..],
			11,
		),
		KernelId::MlConv2dParameterBackwardF32 => (
			BlockProgram::conv_2d_parameter_backward(),
			&[
				BufferAccess::Read,
				BufferAccess::Read,
				BufferAccess::Write,
				BufferAccess::Write,
			][..],
			12,
		),
		KernelId::MlConvTranspose2dF32 => (
			BlockProgram::conv_transpose_2d(),
			&[BufferAccess::Read, BufferAccess::Read, BufferAccess::Write][..],
			10,
		),
		KernelId::MlConvTranspose2dBiasAddF32 => (
			BlockProgram::conv_transpose_2d_bias_add(),
			&[BufferAccess::ReadWrite, BufferAccess::Read][..],
			4,
		),
		KernelId::MlConvTranspose2dBackwardF32 => (
			BlockProgram::conv_transpose_2d_backward(),
			&[BufferAccess::Read, BufferAccess::Read, BufferAccess::Write][..],
			10,
		),
		KernelId::MlConvTranspose2dParameterBackwardF32 => (
			BlockProgram::conv_transpose_2d_parameter_backward(),
			&[
				BufferAccess::Read,
				BufferAccess::Read,
				BufferAccess::Write,
				BufferAccess::Write,
			][..],
			11,
		),
		KernelId::MlConv1dBiasTransposeF32 => (
			BlockProgram::conv_1d_bias_transpose(),
			&[BufferAccess::Read, BufferAccess::Read, BufferAccess::Write][..],
			3,
		),
		KernelId::MlConv1dBiasTransposeReluF32 => (
			BlockProgram::conv_1d_bias_transpose_relu(),
			&[BufferAccess::Read, BufferAccess::Read, BufferAccess::Write][..],
			3,
		),
		KernelId::MlConv1dBiasTransposeResidualReluF32 => (
			BlockProgram::conv_1d_bias_transpose_residual_relu(),
			&[
				BufferAccess::Read,
				BufferAccess::Read,
				BufferAccess::Write,
				BufferAccess::Read,
				BufferAccess::Write,
			][..],
			3,
		),
		KernelId::MlConv1dBiasCopyF32 => (
			BlockProgram::conv_1d_bias_copy(),
			&[BufferAccess::Read, BufferAccess::Read, BufferAccess::Write][..],
			2,
		),
		KernelId::MlConv1dBiasTransposeNormF32 => (
			BlockProgram::conv_1d_bias_transpose_norm(false),
			&[
				BufferAccess::Read,
				BufferAccess::Read,
				BufferAccess::Read,
				BufferAccess::Read,
				BufferAccess::Write,
				BufferAccess::Write,
			][..],
			4,
		),
		KernelId::MlConv1dBiasTransposeNormReluF32 => (
			BlockProgram::conv_1d_bias_transpose_norm(true),
			&[
				BufferAccess::Read,
				BufferAccess::Read,
				BufferAccess::Read,
				BufferAccess::Read,
				BufferAccess::Write,
				BufferAccess::Write,
			][..],
			4,
		),
		KernelId::MlConv1dBiasAddF32 => (
			BlockProgram::conv_1d_bias_add(),
			&[BufferAccess::ReadWrite, BufferAccess::Read][..],
			2,
		),
		KernelId::MlConv1dTransposeF32 => (
			BlockProgram::conv_1d_transpose(),
			&[BufferAccess::Read, BufferAccess::Write][..],
			3,
		),
		KernelId::MlConv2dF32 => (
			BlockProgram::conv_2d(),
			&[
				BufferAccess::Read,
				BufferAccess::Read,
				BufferAccess::Read,
				BufferAccess::Write,
			][..],
			11,
		),
		KernelId::MlConv2dReluF32 => (
			BlockProgram::conv_2d_relu(),
			&[
				BufferAccess::Read,
				BufferAccess::Read,
				BufferAccess::Read,
				BufferAccess::Write,
			][..],
			11,
		),
		KernelId::MlConv1dParameterBackwardF32 => (
			BlockProgram::conv_1d_parameter_backward(),
			&[
				BufferAccess::Read,
				BufferAccess::Read,
				BufferAccess::Write,
				BufferAccess::Write,
			][..],
			10,
		),
		_ => return,
	};
	if node.buffers.len() != accesses.len()
		|| node.push_constants.len() != scalar_count
		|| node
			.buffers
			.iter()
			.zip(accesses)
			.any(|(usage, access)| usage.access != *access)
	{
		return;
	}
	node.fusion_signature = Some(device.tile_signature(&program));
	let Some(key) = device.jit_admit_tile(&program) else {
		return;
	};
	node.jit = Some(JitNode {
		key,
		buffers: node.buffers.clone(),
		push_constants: node.push_constants.clone(),
	});
}

/// `OA_TILE_POOL` substitution for the pooling donors — 16×16-decomposed
/// 3D grids preserved verbatim, including the saved `indices` argmax
/// buffer ABI and the fused `max_pool_2d_relu_backward` saved-output
/// binding.
pub(in crate::runtime) fn specialize_pool_node(device: &Arc<Device>, node: &mut ComputeNode) {
	if node.jit.is_some() || !device.tile_pool_enabled() {
		return;
	}
	let (program, accesses, scalar_count) = match node.kernel {
		KernelId::MlAvgPool2dF32 => (
			BlockProgram::avg_pool_2d(),
			&[BufferAccess::Read, BufferAccess::Write][..],
			9,
		),
		KernelId::MlAvgPool2dBackwardF32 => (
			BlockProgram::avg_pool_2d_backward(),
			&[BufferAccess::Read, BufferAccess::Write][..],
			9,
		),
		KernelId::MlMaxPool2dF32 => (
			BlockProgram::max_pool_2d(),
			&[BufferAccess::Read, BufferAccess::Write, BufferAccess::Write][..],
			9,
		),
		KernelId::MlMaxPool2dBackwardF32 => (
			BlockProgram::max_pool_2d_backward(),
			&[BufferAccess::Read, BufferAccess::Read, BufferAccess::Write][..],
			9,
		),
		KernelId::MlMaxPool2dReluBackwardF32 => (
			BlockProgram::max_pool_2d_relu_backward(),
			&[
				BufferAccess::Read,
				BufferAccess::Read,
				BufferAccess::Read,
				BufferAccess::Write,
			][..],
			9,
		),
		KernelId::MlAdaptiveAvgPool2dF32 => (
			BlockProgram::adaptive_avg_pool_2d(),
			&[BufferAccess::Read, BufferAccess::Write][..],
			6,
		),
		KernelId::MlAdaptiveAvgPool2dBackwardF32 => (
			BlockProgram::adaptive_avg_pool_2d_backward(),
			&[BufferAccess::Read, BufferAccess::Write][..],
			6,
		),
		_ => return,
	};
	if node.buffers.len() != accesses.len()
		|| node.push_constants.len() != scalar_count
		|| node
			.buffers
			.iter()
			.zip(accesses)
			.any(|(usage, access)| usage.access != *access)
	{
		return;
	}
	node.fusion_signature = Some(device.tile_signature(&program));
	let Some(key) = device.jit_admit_tile(&program) else {
		return;
	};
	node.jit = Some(JitNode {
		key,
		buffers: node.buffers.clone(),
		push_constants: node.push_constants.clone(),
	});
}

/// `OA_TILE_LAYER_NORM_BACKWARD` / `OA_TILE_RMS_NORM_BACKWARD`
/// substitutions for the norm data-adjoint nodes — row-per-workgroup
/// `GroupX` grids with a `[rows, columns]` scalar tail (RMSNorm adds
/// `float epsilon`).
pub(in crate::runtime) fn specialize_norm_backward_node(
	device: &Arc<Device>,
	node: &mut ComputeNode,
) {
	if node.jit.is_some() {
		return;
	}
	let program = match node.kernel {
		KernelId::MlLayerNormBackwardF32 if device.tile_layer_norm_backward_enabled() => {
			BlockProgram::layer_norm_backward(256)
		}
		KernelId::MlRmsNormBackwardF32 if device.tile_rms_norm_backward_enabled() => {
			BlockProgram::rms_norm_backward(256)
		}
		_ => return,
	};
	let Ok(program) = program else {
		return;
	};
	let reads = program
		.bindings
		.iter()
		.take_while(|binding| matches!(binding, Binding::Input(_)))
		.count();
	if node.buffers.len() != program.bindings.len()
		|| node.push_constants.len() != program.push_scalars.len() + program.push_f32_scalars.len()
		|| node.buffers[..reads]
			.iter()
			.any(|usage| usage.access != BufferAccess::Read)
		|| node.buffers[reads..]
			.iter()
			.any(|usage| usage.access != BufferAccess::Write)
	{
		return;
	}
	let Some(&PushConstant::U32(rows)) = node.push_constants.first() else {
		return;
	};
	if node.workgroups != [rows, 1, 1] {
		return;
	}
	node.fusion_signature = Some(device.tile_signature(&program));
	let Some(key) = device.jit_admit_tile(&program) else {
		return;
	};
	node.jit = Some(JitNode {
		key,
		buffers: node.buffers.clone(),
		push_constants: node.push_constants.clone(),
	});
}

/// `OA_TILE_LINEAR_BACKWARD` substitutions for the linear data-adjoint
/// family — `MlLinearBackwardF32` plus the fused
/// `MlLinearBackwardPrologueF32`/`MlLinearBackwardNormF32`/
/// `MlLinearBackwardNormAddF32` regions. All share the donor's
/// 32×32×16 tile geometry and the `[batch, input_features,
/// output_features]` push tail.
pub(in crate::runtime) fn specialize_linear_backward_node(
	device: &Arc<Device>,
	node: &mut ComputeNode,
) {
	if node.jit.is_some() {
		return;
	}
	if !device.tile_linear_backward_enabled() {
		return;
	}
	let program = match node.kernel {
		KernelId::MlLinearBackwardF32 => BlockProgram::linear_backward(),
		KernelId::MlLinearBackwardPrologueF32 => BlockProgram::linear_backward_prologue(),
		KernelId::MlLinearBackwardNormF32 => BlockProgram::linear_backward_norm(),
		KernelId::MlLinearBackwardNormAddF32 => BlockProgram::linear_backward_norm_add(),
		KernelId::MlLinearBackwardBmmChainF32 => BlockProgram::linear_backward_bmm_chain(),
		_ => return,
	};
	let reads = program
		.bindings
		.iter()
		.take_while(|binding| matches!(binding, Binding::Input(_)))
		.count();
	if node.buffers.len() != program.bindings.len()
		|| node.push_constants.len() != program.push_scalars.len() + program.push_f32_scalars.len()
		|| node.buffers[..reads]
			.iter()
			.any(|usage| usage.access != BufferAccess::Read)
		|| node.buffers[reads..]
			.iter()
			.any(|usage| usage.access != BufferAccess::Write)
	{
		return;
	}
	if node.kernel == KernelId::MlLinearBackwardBmmChainF32 {
		// The bmm chain grids `[1, ceil(rows/16), batch]` — the fused
		// node's pushes lead with `[batch, rows, inner, columns,
		// input_features, kind]`.
		let (Some(&PushConstant::U32(batch)), Some(&PushConstant::U32(rows))) =
			(node.push_constants.first(), node.push_constants.get(1))
		else {
			return;
		};
		if node.workgroups != [1, rows.div_ceil(16), batch] {
			return;
		}
	} else {
		let (Some(&PushConstant::U32(batch)), Some(&PushConstant::U32(input_features))) =
			(node.push_constants.first(), node.push_constants.get(1))
		else {
			return;
		};
		if node.workgroups != program.workgroups(batch, input_features) {
			return;
		}
	}
	node.fusion_signature = Some(device.tile_signature(&program));
	let Some(key) = device.jit_admit_tile(&program) else {
		return;
	};
	node.jit = Some(JitNode {
		key,
		buffers: node.buffers.clone(),
		push_constants: node.push_constants.clone(),
	});
}

/// `OA_TILE_MAX` substitutions for the full-tensor max family —
/// `MatrixMaxF32`/`MatrixArgmaxF32` run one workgroup over the whole
/// extent; `MatrixMaxBackwardF32` is the flat per-thread adjoint.
pub(in crate::runtime) fn specialize_max_node(device: &Arc<Device>, node: &mut ComputeNode) {
	if node.jit.is_some() {
		return;
	}
	// `tile_reduce_wave_offered` applies the per-kernel flag, the
	// explicit `OA_TILE_REDUCE_ORDER` mode, and the queried subgroup
	// arithmetic support.
	let wave_max = device.tile_reduce_wave_offered(device.tile_max_wave_enabled());
	let program = match node.kernel {
		KernelId::MatrixMaxF32 => tile_candidate(
			device,
			vec![
				(device.tile_max_enabled(), BlockProgram::max(256)),
				(wave_max, BlockProgram::max_wave(256)),
			],
		),
		KernelId::MatrixArgmaxF32 => tile_candidate(
			device,
			vec![
				(device.tile_max_enabled(), BlockProgram::argmax(256)),
				(wave_max, BlockProgram::argmax_wave(256)),
			],
		),
		KernelId::MatrixMaxBackwardF32 => tile_candidate(
			device,
			vec![(device.tile_max_enabled(), Ok(BlockProgram::max_backward()))],
		),
		_ => return,
	};
	let Some(program) = program else {
		return;
	};
	let reads = program
		.bindings
		.iter()
		.take_while(|binding| matches!(binding, Binding::Input(_)))
		.count();
	if node.buffers.len() != program.bindings.len()
		|| node.push_constants.len() != program.push_scalars.len() + program.push_f32_scalars.len()
		|| node.buffers[..reads]
			.iter()
			.any(|usage| usage.access != BufferAccess::Read)
		|| node.buffers[reads..]
			.iter()
			.any(|usage| usage.access != BufferAccess::Write)
	{
		return;
	}
	if matches!(
		node.kernel,
		KernelId::MatrixMaxF32 | KernelId::MatrixArgmaxF32
	) && node.workgroups != [1, 1, 1]
	{
		return;
	}
	node.fusion_signature = Some(device.tile_signature(&program));
	let Some(key) = device.jit_admit_tile(&program) else {
		return;
	};
	node.jit = Some(JitNode {
		key,
		buffers: node.buffers.clone(),
		push_constants: node.push_constants.clone(),
	});
}

/// `OA_TILE_LAYER_NORM_BACKWARD` substitution for the fused
/// `MlLayerNormBackwardSumF32` node — nine buffers `[normalized,
/// inverse_stddev, weight, addend_a, addend_b, addend_c,
/// input_gradient, weight_contribution, dy]`, `[rows, columns,
/// addend_count]` push, `[rows, 1, 1]` grid.
pub(in crate::runtime) fn specialize_layer_norm_backward_sum_node(
	device: &Arc<Device>,
	node: &mut ComputeNode,
) {
	if node.kernel != KernelId::MlLayerNormBackwardSumF32
		|| node.jit.is_some()
		|| !device.tile_layer_norm_backward_enabled()
	{
		return;
	}
	let Ok(program) = BlockProgram::layer_norm_backward_sum(256) else {
		return;
	};
	if node.buffers.len() != 9
		|| node.push_constants.len() != 3
		|| node.buffers[..6]
			.iter()
			.any(|usage| usage.access != BufferAccess::Read)
		|| node.buffers[6..]
			.iter()
			.any(|usage| usage.access != BufferAccess::Write)
	{
		return;
	}
	let Some(&PushConstant::U32(rows)) = node.push_constants.first() else {
		return;
	};
	if node.workgroups != [rows, 1, 1] {
		return;
	}
	node.fusion_signature = Some(device.tile_signature(&program));
	let Some(key) = device.jit_admit_tile(&program) else {
		return;
	};
	node.jit = Some(JitNode {
		key,
		buffers: node.buffers.clone(),
		push_constants: node.push_constants.clone(),
	});
}

/// `OA_TILE_EMBEDDING` substitution for the fused
/// `MlEmbeddingBackwardSumF32` node — four buffers `[indices,
/// addend_a, addend_b, weight_gradient]`, `[index_count,
/// num_embeddings, embedding_dim]` push, the consumer's flat linear
/// grid.
pub(in crate::runtime) fn specialize_embedding_backward_sum_node(
	device: &Arc<Device>,
	node: &mut ComputeNode,
) {
	if node.kernel != KernelId::MlEmbeddingBackwardSumF32
		|| node.jit.is_some()
		|| !device.tile_embedding_enabled()
	{
		return;
	}
	let program = BlockProgram::embedding_backward_sum();
	if node.buffers.len() != 4
		|| node.push_constants.len() != 3
		|| node.buffers[..3]
			.iter()
			.any(|usage| usage.access != BufferAccess::Read)
		|| node.buffers[3].access != BufferAccess::Write
	{
		return;
	}
	node.fusion_signature = Some(device.tile_signature(&program));
	let Some(key) = device.jit_admit_tile(&program) else {
		return;
	};
	node.jit = Some(JitNode {
		key,
		buffers: node.buffers.clone(),
		push_constants: node.push_constants.clone(),
	});
}

/// `OA_TILE_EMBEDDING` substitution for the unfused
/// `MlEmbeddingBackwardF32`/`MlEmbeddingBackwardU8F32` scatter
/// adjoints — three buffers `[indices, output_gradient,
/// weight_gradient]`, `[index_count, num_embeddings, embedding_dim]`
/// push, the consumer's flat linear grid.
pub(in crate::runtime) fn specialize_embedding_backward_node(
	device: &Arc<Device>,
	node: &mut ComputeNode,
) {
	if node.jit.is_some() || !device.tile_embedding_enabled() {
		return;
	}
	let program = match node.kernel {
		KernelId::MlEmbeddingBackwardF32 => BlockProgram::embedding_backward(),
		KernelId::MlEmbeddingBackwardU8F32 => BlockProgram::embedding_backward_u8(),
		_ => return,
	};
	if node.buffers.len() != 3
		|| node.push_constants.len() != 3
		|| node.buffers[..2]
			.iter()
			.any(|usage| usage.access != BufferAccess::Read)
		|| node.buffers[2].access != BufferAccess::Write
	{
		return;
	}
	node.fusion_signature = Some(device.tile_signature(&program));
	let Some(key) = device.jit_admit_tile(&program) else {
		return;
	};
	node.jit = Some(JitNode {
		key,
		buffers: node.buffers.clone(),
		push_constants: node.push_constants.clone(),
	});
}

/// `OA_TILE_SUM` substitution for the `MatrixSumBackwardF32` broadcast
/// adjoint — `[output_gradient, input_gradient]` buffers,
/// `[outer_size, dim_size, inner_size]` push, flat linear grid.
pub(in crate::runtime) fn specialize_sum_backward_node(
	device: &Arc<Device>,
	node: &mut ComputeNode,
) {
	if node.kernel != KernelId::MatrixSumBackwardF32
		|| node.jit.is_some()
		|| !device.tile_sum_enabled()
	{
		return;
	}
	let program = BlockProgram::sum_backward();
	if node.buffers.len() != 2
		|| node.push_constants.len() != 3
		|| node.buffers[0].access != BufferAccess::Read
		|| node.buffers[1].access != BufferAccess::Write
	{
		return;
	}
	node.fusion_signature = Some(device.tile_signature(&program));
	let Some(key) = device.jit_admit_tile(&program) else {
		return;
	};
	node.jit = Some(JitNode {
		key,
		buffers: node.buffers.clone(),
		push_constants: node.push_constants.clone(),
	});
}

/// `OA_TILE_CROSS_ENTROPY` substitution for the cross-entropy
/// forward/backward nodes — the 2D-flattened `[rows.min(65535),
/// rows.div_ceil(65535), 1]` row-per-workgroup grid with a
/// `[rows, classes, target_dtype]` scalar tail (`+ valid_count` on the
/// masked adjoint).
pub(in crate::runtime) fn specialize_cross_entropy_node(
	device: &Arc<Device>,
	node: &mut ComputeNode,
) {
	if node.jit.is_some() || !device.tile_cross_entropy_enabled() {
		return;
	}
	let program = match node.kernel {
		KernelId::MlCrossEntropyF32 => BlockProgram::cross_entropy(256, false),
		KernelId::MlMaskedCrossEntropyF32 => BlockProgram::cross_entropy(256, true),
		KernelId::MlCrossEntropyBackwardF32 => BlockProgram::cross_entropy_backward(256, false),
		KernelId::MlMaskedCrossEntropyBackwardF32 => BlockProgram::cross_entropy_backward(256, true),
		_ => return,
	};
	let Ok(program) = program else {
		return;
	};
	let reads = program
		.bindings
		.iter()
		.take_while(|binding| matches!(binding, Binding::Input(_)))
		.count();
	if node.buffers.len() != program.bindings.len()
		|| node.push_constants.len() != program.push_scalars.len()
		|| node.buffers[..reads]
			.iter()
			.any(|usage| usage.access != BufferAccess::Read)
		|| node.buffers[reads..]
			.iter()
			.any(|usage| usage.access != BufferAccess::Write)
	{
		return;
	}
	let Some(&PushConstant::U32(rows)) = node.push_constants.first() else {
		return;
	};
	if node.workgroups != [rows.min(65535), rows.div_ceil(65535), 1] {
		return;
	}
	node.fusion_signature = Some(device.tile_signature(&program));
	let Some(key) = device.jit_admit_tile(&program) else {
		return;
	};
	node.jit = Some(JitNode {
		key,
		buffers: node.buffers.clone(),
		push_constants: node.push_constants.clone(),
	});
}

/// Deterministic output-tile seed candidates. A tile that is wider than
/// the problem's masked coverage wastes half of every workgroup's
/// fragment work — the measured 2026-11-17 `nlp-*` regression — so the
/// narrow axis takes the first smaller fit. Measured per-class route
/// records replace this map in OA-CC-12.
fn gemm_candidate(m: u32, n: u32) -> (u32, u32) {
	let tile_rows = if m <= 32 { 32 } else { 64 };
	let tile_columns = if n <= 32 { 32 } else { 64 };
	(tile_rows, tile_columns)
}

/// `OA_TILE_ATTENTION` substitution for the head repack and
/// score-softmax donors — `MlSplitHeadsF32`/`MlMergeHeadsF32` are flat
/// repack gathers over `[B*S, H, S, D]` and `[B*H, S, D]` element
/// domains, and `MlSdpaSoftmaxF32`/`MlSdpaSoftmaxN32F32` are the
/// workgroup-per-row transformed-score softmax at 256/32 lanes. The
/// softmax record ABI interleaves its `scale` f32 scalar mid-vector;
/// the tile program declares u32 scalars first, so the admitted
/// `JitNode` repacks `[rows, columns, has_mask, causal, scale]`.
pub(in crate::runtime) fn specialize_attention_node(device: &Arc<Device>, node: &mut ComputeNode) {
	if node.jit.is_some() || !device.tile_attention_enabled() {
		return;
	}
	match node.kernel {
		KernelId::MlSplitHeadsF32 | KernelId::MlMergeHeadsF32 => {
			let program = if node.kernel == KernelId::MlSplitHeadsF32 {
				BlockProgram::split_heads()
			} else {
				BlockProgram::merge_heads()
			};
			let [
				PushConstant::U32(batch),
				PushConstant::U32(sequence),
				PushConstant::U32(heads),
				PushConstant::U32(dim),
			] = node.push_constants.as_slice()
			else {
				return;
			};
			let Some(count) = batch
				.checked_mul(*sequence)
				.and_then(|value| value.checked_mul(*heads))
				.and_then(|value| value.checked_mul(*dim))
			else {
				return;
			};
			if node.buffers.len() != 2
				|| node.buffers[0].access != BufferAccess::Read
				|| node.buffers[1].access != BufferAccess::Write
				|| node.workgroups != [count.div_ceil(256), 1, 1]
			{
				return;
			}
			node.fusion_signature = Some(device.tile_signature(&program));
			let Some(key) = device.jit_admit_tile(&program) else {
				return;
			};
			node.jit = Some(JitNode {
				key,
				buffers: node.buffers.clone(),
				push_constants: node.push_constants.clone(),
			});
		}
		KernelId::MlSmallCausalSdpaF32 => {
			let program = BlockProgram::small_causal_sdpa();
			let [
				PushConstant::U32(batch),
				PushConstant::U32(sequence),
				PushConstant::U32(heads),
				PushConstant::U32(dim),
				PushConstant::F32(_scale),
			] = node.push_constants.as_slice()
			else {
				return;
			};
			// The composite stages `S*D` floats into three shared tiles and
			// keeps the donor's workgroup-per-batch-head grid — the fused
			// matcher already bounds `dim <= 128`, `sequence <= 128` and
			// `sequence*dim <= 2048`.
			let Some(bh) = batch.checked_mul(*heads) else {
				return;
			};
			if *batch == 0
				|| *heads == 0
				|| *dim == 0
				|| *dim > 128
				|| *sequence == 0
				|| *sequence > 128
				|| sequence.checked_mul(*dim).is_none_or(|v| v > 2048)
				|| node.buffers.len() != 10
				|| node.workgroups != [bh, 1, 1]
			{
				return;
			}
			node.fusion_signature = Some(device.tile_signature(&program));
			let Some(key) = device.jit_admit_tile(&program) else {
				return;
			};
			node.jit = Some(JitNode {
				key,
				buffers: node.buffers.clone(),
				push_constants: node.push_constants.clone(),
			});
		}
		KernelId::MlFlashAttentionCausalF32
		| KernelId::MlFlashAttentionCausalBackwardQF32
		| KernelId::MlFlashAttentionCausalBackwardKvF32 => {
			let program = match node.kernel {
				KernelId::MlFlashAttentionCausalF32 => BlockProgram::flash_causal(),
				KernelId::MlFlashAttentionCausalBackwardQF32 => BlockProgram::flash_causal_backward_q(),
				_ => BlockProgram::flash_causal_backward_kv(),
			};
			let [
				PushConstant::U32(batch_heads),
				PushConstant::U32(sequence),
				PushConstant::U32(_dim),
				PushConstant::F32(_scale),
			] = node.push_constants.as_slice()
			else {
				return;
			};
			let Some(rows) = batch_heads.checked_mul(*sequence) else {
				return;
			};
			let (buffers, reads) = match node.kernel {
				KernelId::MlFlashAttentionCausalF32 => (5, 3),
				KernelId::MlFlashAttentionCausalBackwardQF32 => (7, 6),
				_ => (8, 6),
			};
			// The donors stage `sequence` scores into a 1024-float shared
			// tile; longer sequences stay on the incumbent kernels.
			if *sequence == 0
				|| *sequence > 1024
				|| node.buffers.len() != buffers
				|| node
					.buffers
					.iter()
					.take(reads)
					.any(|binding| binding.access != BufferAccess::Read)
				|| node
					.buffers
					.iter()
					.skip(reads)
					.any(|binding| binding.access != BufferAccess::Write)
				|| node.workgroups != [rows, 1, 1]
			{
				return;
			}
			node.fusion_signature = Some(device.tile_signature(&program));
			let Some(key) = device.jit_admit_tile(&program) else {
				return;
			};
			node.jit = Some(JitNode {
				key,
				buffers: node.buffers.clone(),
				push_constants: node.push_constants.clone(),
			});
		}
		KernelId::MlSdpaSoftmaxF32 | KernelId::MlSdpaSoftmaxN32F32 => {
			let lanes = if node.kernel == KernelId::MlSdpaSoftmaxN32F32 {
				32
			} else {
				256
			};
			let Ok(program) = BlockProgram::sdpa_softmax(lanes) else {
				return;
			};
			let [
				PushConstant::U32(rows),
				PushConstant::U32(_columns),
				PushConstant::F32(_scale),
				PushConstant::U32(_has_mask),
				PushConstant::U32(_causal),
			] = node.push_constants.as_slice()
			else {
				return;
			};
			if node.buffers.len() != 3
				|| node.buffers[0].access != BufferAccess::Read
				|| node.buffers[1].access != BufferAccess::Read
				|| node.buffers[2].access != BufferAccess::Write
				|| node.workgroups != [*rows, 1, 1]
			{
				return;
			}
			node.fusion_signature = Some(device.tile_signature(&program));
			let Some(key) = device.jit_admit_tile(&program) else {
				return;
			};
			node.jit = Some(JitNode {
				key,
				buffers: node.buffers.clone(),
				push_constants: vec![
					node.push_constants[0],
					node.push_constants[1],
					node.push_constants[3],
					node.push_constants[4],
					node.push_constants[2],
				],
			});
		}
		_ => {}
	}
}

/// `OA_TILE_VIDEO` substitution for the `to_matrix` conversion donors —
/// `VideoToMatrix{I420,Nv12,P010}` share one 16×16 per-lane schedule
/// with plane-buffer inputs followed by the f32 output, the
/// `[u32 scalars, f32 scalars]` push ABI, and `group_id.z` as the
/// channel axis. The emitted region keeps the donor's typed plane
/// loads as first-class lane expressions.
pub(in crate::runtime) fn specialize_video_node(device: &Arc<Device>, node: &mut ComputeNode) {
	if node.jit.is_some() || !device.tile_video_enabled() {
		return;
	}
	let (planar, sample_bits) = match node.kernel {
		KernelId::VideoToMatrixI420 => (true, 8),
		KernelId::VideoToMatrixNv12 => (false, 8),
		KernelId::VideoToMatrixP010 => (false, 10),
		_ => return,
	};
	let inputs = if planar { 3 } else { 2 };
	if node.buffers.len() != inputs + 1 {
		return;
	}
	if node.buffers[..inputs]
		.iter()
		.any(|binding| binding.access != BufferAccess::Read)
		|| node.buffers[inputs].access != BufferAccess::Write
	{
		return;
	}
	let Ok(program) = BlockProgram::video_to_matrix(planar, sample_bits) else {
		return;
	};
	node.fusion_signature = Some(device.tile_signature(&program));
	let Some(key) = device.jit_admit_tile_video(&program) else {
		return;
	};
	node.jit = Some(JitNode {
		key,
		buffers: node.buffers.clone(),
		push_constants: node.push_constants.clone(),
	});
}
