//! Reverse traversal, gradient accumulation, and semantic backward provenance.
//!
//! Mechanical extraction from `tape.rs`; admitted OA formulas and ordering
//! remain unchanged. GPU dispatch belongs to the canonical operation providers.

use super::{ACTIVE_TAPES, GradNode, GradientTape};
use crate::matrix::{autograd::GradientContext, sum_to_shape};
use crate::ml::autograd::node::operation::BackwardStatus;
use crate::ml::{flow as flow_lowering, matrix as ml_matrix};
use crate::{DType, Error, Matrix, Result, matrix};
use std::{collections::HashMap, rc::Rc};

impl GradientTape {
	/// Record the reverse-mode operations leading to one scalar loss.
	///
	/// The current checkpoint supports any recorded FP32 scalar result whose
	/// reverse path is composed entirely from admitted operations. Gradients
	/// accumulate on stable [`crate::ml::Parameter`] handles. This method records
	/// but does not submit or wait.
	///
	/// # Errors
	///
	/// Returns an error when the tape was already consumed, `root` was not
	/// produced by this tape, a saved parameter changed before backward, or
	/// gradient operation validation/allocation/recording fails.
	pub fn backward(&self, root: &Matrix) -> Result<()> {
		let self_weak = Rc::downgrade(&self.recording);
		let another_tape_is_active = ACTIVE_TAPES.with(|tapes| {
			tapes
				.borrow()
				.iter()
				.any(|candidate| candidate.strong_count() != 0 && !candidate.ptr_eq(&self_weak))
		});
		if another_tape_is_active {
			return Err(Error::failed_precondition(
				"gradient backward requires every other tape on this thread to be closed",
			));
		}
		self.close();
		if self.recording.consumed.get() {
			return Err(Error::failed_precondition(
				"gradient tape has already been consumed",
			));
		}
		if !root.shape().is_empty() || root.dtype() != DType::F32 {
			return Err(Error::invalid_argument(
				"gradient tape root must be an FP32 scalar",
			));
		}
		let nodes = self.recording.nodes.borrow();
		let root_found = nodes
			.iter()
			.any(|entry| entry.node.produces(root.value_id()));
		if !root_found {
			return Err(Error::failed_precondition(
				"gradient tape root was not produced by this tape",
			));
		}
		for entry in nodes.iter() {
			entry.node.validate_versions()?;
		}
		for watched in self.recording.watched_parameters.borrow().iter() {
			watched.parameter.validate_version(watched.version)?;
		}
		self.recording.consumed.set(true);

		let engine = root.engine_handle().clone();
		let mut gradients = HashMap::<u64, Matrix>::new();
		gradients.insert(
			root.value_id(),
			Matrix::from_slice_handle(&engine, vec![], &[1.0_f32])?,
		);
		for entry in nodes.iter().rev() {
			let node = &entry.node;
			let Some(reached_output_id) = reached_output_id(node, &gradients) else {
				continue;
			};
			let attachment = engine.attach_semantic_autograd(reached_output_id, entry.sequence)?;
			let backward_first = engine.semantic_operation_count();
			match node {
				GradNode::Matrix(node) => {
					if !node.backward(&mut MatrixGradients(&mut gradients))? {
						continue;
					}
				}
				GradNode::Operation(node) => {
					if matches!(node.backward(&mut gradients)?, BackwardStatus::Skipped) {
						continue;
					}
				}
				GradNode::FlowLinearState {
					clean,
					noise,
					time,
					output_id,
				} => {
					let Some(output_gradient) = gradients.remove(output_id) else {
						continue;
					};
					let one_minus_time = matrix::add_scalar(&matrix::scale(time, -1.0)?, 1.0)?;
					let clean_gradient = matrix::mul(&output_gradient, &one_minus_time)?;
					let noise_gradient = matrix::mul(&output_gradient, time)?;
					let velocity = matrix::sub(noise, clean)?;
					let time_gradient = matrix::mul(&output_gradient, &velocity)?;
					accumulate_value_gradient(
						&mut gradients,
						clean.value_id(),
						sum_to_shape(&clean_gradient, clean.shape())?,
					)?;
					accumulate_value_gradient(
						&mut gradients,
						noise.value_id(),
						sum_to_shape(&noise_gradient, noise.shape())?,
					)?;
					accumulate_value_gradient(
						&mut gradients,
						time.value_id(),
						sum_to_flow_time_shape(&time_gradient, time.shape(), clean.shape())?,
					)?;
				}
				GradNode::FlowLinearVelocity {
					clean,
					noise,
					output_id,
				} => {
					let Some(output_gradient) = gradients.remove(output_id) else {
						continue;
					};
					accumulate_value_gradient(
						&mut gradients,
						clean.value_id(),
						matrix::scale(&output_gradient, -1.0)?,
					)?;
					accumulate_value_gradient(&mut gradients, noise.value_id(), output_gradient)?;
				}
				GradNode::FlowEulerStep {
					state,
					velocity,
					delta_time,
					output_id,
				} => {
					let Some(output_gradient) = gradients.remove(output_id) else {
						continue;
					};
					accumulate_value_gradient(&mut gradients, state.value_id(), output_gradient.clone())?;
					accumulate_value_gradient(
						&mut gradients,
						velocity.value_id(),
						matrix::scale(&output_gradient, *delta_time)?,
					)?;
				}
				GradNode::FlowMaskedMse {
					prediction,
					target,
					mask,
					denominator,
					output_id,
				} => {
					let Some(output_gradient) = gradients.remove(output_id) else {
						continue;
					};
					let prediction_gradient = flow_lowering::masked_mse_backward(
						prediction,
						target,
						mask,
						denominator,
						&output_gradient,
					)?;
					accumulate_value_gradient(&mut gradients, prediction.value_id(), prediction_gradient)?;
				}

				GradNode::Linear {
					input,
					output_id,
					weight,
					weight_value,
					bias,
					..
				} => {
					let Some(output_gradient) = gradients.remove(output_id) else {
						continue;
					};
					let (input_gradient, weight_gradient, bias_gradient) =
						ml_matrix::linear_backward(input, weight_value, &output_gradient)?;
					accumulate_value_gradient(&mut gradients, input.value_id(), input_gradient)?;
					weight.accumulate_gradient(weight_gradient)?;
					if let Some(bias) = bias {
						bias.accumulate_gradient(bias_gradient)?;
					}
				}
				GradNode::Conv2d {
					input,
					output_id,
					weight,
					weight_value,
					bias,
					bias_value,
					stride,
					padding,
					groups,
					..
				} => {
					let Some(output_gradient) = gradients.remove(output_id) else {
						continue;
					};
					let (input_gradient, weight_gradient, bias_gradient) =
						crate::ml::matrix::conv_2d_backward(
							input,
							weight_value,
							&output_gradient,
							*stride,
							*padding,
							*groups,
						)?;
					accumulate_value_gradient(&mut gradients, input.value_id(), input_gradient)?;
					accumulate_value_gradient(
						&mut gradients,
						weight_value.value_id(),
						weight_gradient.clone(),
					)?;
					accumulate_value_gradient(&mut gradients, bias_value.value_id(), bias_gradient.clone())?;
					if let Some(weight) = weight {
						weight.accumulate_gradient(weight_gradient)?;
					}
					if let Some(bias) = bias {
						bias.accumulate_gradient(bias_gradient)?;
					}
				}
				GradNode::Conv1d {
					input,
					output_id,
					weight,
					weight_value,
					bias,
					bias_value,
					stride,
					padding,
					dilation,
					..
				} => {
					let Some(output_gradient) = gradients.remove(output_id) else {
						continue;
					};
					let (input_gradient, weight_gradient, bias_gradient) =
						crate::ml::matrix::conv_1d_backward(
							input,
							weight_value,
							&output_gradient,
							*stride,
							*padding,
							*dilation,
						)?;
					accumulate_value_gradient(&mut gradients, input.value_id(), input_gradient)?;
					accumulate_value_gradient(
						&mut gradients,
						weight_value.value_id(),
						weight_gradient.clone(),
					)?;
					accumulate_value_gradient(&mut gradients, bias_value.value_id(), bias_gradient.clone())?;
					if let Some(weight) = weight {
						weight.accumulate_gradient(weight_gradient)?;
					}
					if let Some(bias) = bias {
						bias.accumulate_gradient(bias_gradient)?;
					}
				}
				GradNode::ConvTranspose1d {
					input,
					output_id,
					weight,
					weight_value,
					stride,
					padding,
					dilation,
					..
				} => {
					let Some(output_gradient) = gradients.remove(output_id) else {
						continue;
					};
					let (input_gradient, weight_gradient) = crate::ml::matrix::conv_transpose_1d_backward(
						input,
						weight_value,
						&output_gradient,
						*stride,
						*padding,
						*dilation,
					)?;
					accumulate_value_gradient(&mut gradients, input.value_id(), input_gradient)?;
					accumulate_value_gradient(
						&mut gradients,
						weight_value.value_id(),
						weight_gradient.clone(),
					)?;
					if let Some(weight) = weight {
						weight.accumulate_gradient(weight_gradient)?;
					}
				}
				GradNode::ConvTranspose2d {
					input,
					output_id,
					weight,
					weight_value,
					bias,
					bias_value,
					stride,
					padding,
					..
				} => {
					let Some(output_gradient) = gradients.remove(output_id) else {
						continue;
					};
					let (input_gradient, weight_gradient, bias_gradient) =
						crate::ml::matrix::conv_transpose_2d_backward(
							input,
							weight_value,
							&output_gradient,
							*stride,
							*padding,
						)?;
					accumulate_value_gradient(&mut gradients, input.value_id(), input_gradient)?;
					accumulate_value_gradient(
						&mut gradients,
						weight_value.value_id(),
						weight_gradient.clone(),
					)?;
					accumulate_value_gradient(&mut gradients, bias_value.value_id(), bias_gradient.clone())?;
					if let Some(weight) = weight {
						weight.accumulate_gradient(weight_gradient)?;
					}
					if let Some(bias) = bias {
						bias.accumulate_gradient(bias_gradient)?;
					}
				}
				GradNode::Embedding {
					indices,
					output_id,
					weight,
					weight_value,
					..
				} => {
					let Some(output_gradient) = gradients.remove(output_id) else {
						continue;
					};
					let weight_gradient =
						ml_matrix::embedding_backward(indices, &output_gradient, weight_value)?;
					weight.accumulate_gradient(weight_gradient)?;
				}
				GradNode::LayerNorm {
					input,
					normalized,
					inverse_stddev,
					output_id,
					weight,
					weight_value,
					bias,
					..
				} => {
					let Some(output_gradient) = gradients.remove(output_id) else {
						continue;
					};
					let (input_gradient, weight_gradient, bias_gradient) = ml_matrix::layer_norm_backward(
						input,
						weight_value,
						normalized,
						inverse_stddev,
						&output_gradient,
					)?;
					accumulate_value_gradient(&mut gradients, input.value_id(), input_gradient)?;
					weight.accumulate_gradient(weight_gradient)?;
					bias.accumulate_gradient(bias_gradient)?;
				}
				GradNode::BatchNorm2d {
					input,
					mean,
					variance,
					output_id,
					weight,
					weight_value,
					bias,
					bias_value,
					epsilon,
					training,
					..
				} => {
					let Some(output_gradient) = gradients.remove(output_id) else {
						continue;
					};
					let (input_gradient, weight_gradient, bias_gradient) =
						crate::ml::matrix::batch_norm_2d_backward(
							input,
							weight_value,
							mean,
							variance,
							&output_gradient,
							*epsilon,
							*training,
						)?;
					accumulate_value_gradient(&mut gradients, input.value_id(), input_gradient)?;
					accumulate_value_gradient(
						&mut gradients,
						weight_value.value_id(),
						weight_gradient.clone(),
					)?;
					accumulate_value_gradient(&mut gradients, bias_value.value_id(), bias_gradient.clone())?;
					if let Some(weight) = weight {
						weight.accumulate_gradient(weight_gradient)?;
					}
					if let Some(bias) = bias {
						bias.accumulate_gradient(bias_gradient)?;
					}
				}
				GradNode::RmsNorm {
					input,
					output_id,
					weight,
					weight_value,
					epsilon,
					..
				} => {
					let Some(output_gradient) = gradients.remove(output_id) else {
						continue;
					};
					let (input_gradient, weight_gradient) =
						ml_matrix::rms_norm_backward(input, weight_value, &output_gradient, *epsilon)?;
					accumulate_value_gradient(&mut gradients, input.value_id(), input_gradient)?;
					accumulate_value_gradient(
						&mut gradients,
						weight_value.value_id(),
						weight_gradient.clone(),
					)?;
					if let Some(weight) = weight {
						weight.accumulate_gradient(weight_gradient)?;
					}
				}
				GradNode::RmsNormGated(node) => {
					let Some(output_gradient) = gradients.remove(&node.output_id) else {
						continue;
					};
					let result = ml_matrix::rms_norm_gated_backward(
						&node.input,
						&node.weight_value,
						node.bias_value.as_ref(),
						&node.gate,
						&output_gradient,
						node.epsilon,
					)?;
					accumulate_value_gradient(&mut gradients, node.input.value_id(), result.input)?;
					accumulate_value_gradient(
						&mut gradients,
						node.weight_value.value_id(),
						result.weight.clone(),
					)?;
					accumulate_value_gradient(&mut gradients, node.gate.value_id(), result.gate)?;
					if let Some(weight) = &node.weight {
						weight.accumulate_gradient(result.weight)?;
					}
					if let (Some(bias_value), Some(bias_gradient)) = (&node.bias_value, result.bias) {
						accumulate_value_gradient(
							&mut gradients,
							bias_value.value_id(),
							bias_gradient.clone(),
						)?;
						if let Some(bias) = &node.bias {
							bias.accumulate_gradient(bias_gradient)?;
						}
					}
				}
				GradNode::ChannelNorm(node) => {
					let Some(output_gradient) = gradients.remove(&node.output_id) else {
						continue;
					};
					let result = ml_matrix::channel_norm_backward_saved(
						&node.input,
						&node.weight_value,
						node.output.as_ref(),
						&output_gradient,
						node.epsilon,
					)?;
					accumulate_value_gradient(&mut gradients, node.input.value_id(), result.input)?;
					accumulate_value_gradient(
						&mut gradients,
						node.weight_value.value_id(),
						result.weight.clone(),
					)?;
					accumulate_value_gradient(
						&mut gradients,
						node.bias_value.value_id(),
						result.bias.clone(),
					)?;
					if let Some(weight) = &node.weight {
						weight.accumulate_gradient(result.weight)?;
					}
					if let Some(bias) = &node.bias {
						bias.accumulate_gradient(result.bias)?;
					}
				}

				GradNode::Bmm {
					left,
					right,
					output_id,
				} => {
					let Some(output_gradient) = gradients.remove(output_id) else {
						continue;
					};
					let left_gradient = ml_matrix::bmm_nt(&output_gradient, right)?;
					let right_gradient = ml_matrix::bmm_tn(left, &output_gradient)?;
					accumulate_value_gradient(&mut gradients, left.value_id(), left_gradient)?;
					accumulate_value_gradient(&mut gradients, right.value_id(), right_gradient)?;
				}
				GradNode::BmmNt {
					left,
					right,
					output_id,
				} => {
					let Some(output_gradient) = gradients.remove(output_id) else {
						continue;
					};
					let left_gradient = ml_matrix::bmm(&output_gradient, right)?;
					let right_gradient = ml_matrix::bmm_tn(&output_gradient, left)?;
					accumulate_value_gradient(&mut gradients, left.value_id(), left_gradient)?;
					accumulate_value_gradient(&mut gradients, right.value_id(), right_gradient)?;
				}
				GradNode::BmmTn {
					left,
					right,
					output_id,
				} => {
					let Some(output_gradient) = gradients.remove(output_id) else {
						continue;
					};
					let left_gradient = ml_matrix::bmm_nt(right, &output_gradient)?;
					let right_gradient = ml_matrix::bmm(left, &output_gradient)?;
					accumulate_value_gradient(&mut gradients, left.value_id(), left_gradient)?;
					accumulate_value_gradient(&mut gradients, right.value_id(), right_gradient)?;
				}
				GradNode::SplitHeads {
					input,
					output_id,
					batch,
					sequence_length,
					num_heads,
				} => {
					let Some(output_gradient) = gradients.remove(output_id) else {
						continue;
					};
					let (input_gradient, _) = ml_matrix::merge_heads_dispatch(
						&output_gradient,
						*batch,
						*sequence_length,
						*num_heads,
					)?;
					accumulate_value_gradient(&mut gradients, input.value_id(), input_gradient)?;
				}
				GradNode::MergeHeads {
					input,
					output_id,
					batch,
					sequence_length,
					num_heads,
				} => {
					let Some(output_gradient) = gradients.remove(output_id) else {
						continue;
					};
					let (input_gradient, _) = ml_matrix::split_heads_dispatch(
						&output_gradient,
						*batch,
						*sequence_length,
						*num_heads,
					)?;
					accumulate_value_gradient(&mut gradients, input.value_id(), input_gradient)?;
				}
				GradNode::SoftmaxScaledMasked {
					input,
					output,
					output_id,
					scale,
				} => {
					let Some(output_gradient) = gradients.remove(output_id) else {
						continue;
					};
					let input_gradient =
						ml_matrix::softmax_scaled_masked_backward(output, &output_gradient, *scale)?;
					accumulate_value_gradient(&mut gradients, input.value_id(), input_gradient)?;
				}
				GradNode::ScaledDotProductAttention {
					query,
					key,
					value,
					probabilities,
					output_id,
					scale,
				} => {
					let Some(output_gradient) = gradients.remove(output_id) else {
						continue;
					};
					let probability_gradient = ml_matrix::bmm_nt(&output_gradient, value)?;
					let [batch_heads, sequence_length, _] = probabilities.shape() else {
						return Err(Error::internal(
							"saved SDPA probabilities lost rank-three shape",
						));
					};
					let rows = batch_heads
						.checked_mul(*sequence_length)
						.ok_or_else(|| Error::internal("saved SDPA probability rows overflow usize"))?;
					let score_gradient = ml_matrix::softmax_scaled_masked_backward(
						&probabilities.reshape([rows, *sequence_length])?,
						&probability_gradient.reshape([rows, *sequence_length])?,
						*scale,
					)?
					.reshape([*batch_heads, *sequence_length, *sequence_length])?;
					let query_gradient = ml_matrix::bmm(&score_gradient, key)?;
					let key_gradient = ml_matrix::bmm_tn(&score_gradient, query)?;
					let value_gradient = ml_matrix::bmm_tn(probabilities, &output_gradient)?;
					accumulate_value_gradient(&mut gradients, query.value_id(), query_gradient)?;
					accumulate_value_gradient(&mut gradients, key.value_id(), key_gradient)?;
					accumulate_value_gradient(&mut gradients, value.value_id(), value_gradient)?;
				}
				GradNode::FlashAttention {
					query,
					key,
					value,
					output,
					log_sum_exp,
					output_id,
					scale,
				} => {
					let Some(output_gradient) = gradients.remove(output_id) else {
						continue;
					};
					let (query_grad, key_grad, value_grad) = ml_matrix::flash_attention_causal_backward(
						query,
						key,
						value,
						output,
						log_sum_exp,
						&output_gradient,
						*scale,
					)?;
					accumulate_value_gradient(&mut gradients, query.value_id(), query_grad)?;
					accumulate_value_gradient(&mut gradients, key.value_id(), key_grad)?;
					accumulate_value_gradient(&mut gradients, value.value_id(), value_grad)?;
				}
				GradNode::MoeRouteWeights {
					probabilities,
					expert_indices,
					output,
					output_id,
				} => {
					let Some(output_gradient) = gradients.remove(output_id) else {
						continue;
					};
					let probability_gradient = crate::ml::matrix::moe_route_weights_backward(
						&output_gradient,
						probabilities,
						expert_indices,
						output,
					)?;
					accumulate_value_gradient(
						&mut gradients,
						probabilities.value_id(),
						probability_gradient,
					)?;
				}
				GradNode::MoeGather {
					input,
					inverse,
					output_id,
				} => {
					let Some(output_gradient) = gradients.remove(output_id) else {
						continue;
					};
					let input_gradient =
						crate::ml::matrix::moe_gather_backward(&output_gradient, inverse, input.shape()[0])?;
					accumulate_value_gradient(&mut gradients, input.value_id(), input_gradient)?;
				}
				GradNode::MoeCombine {
					packed,
					route_gate,
					inverse,
					packed_slot,
					output_id,
				} => {
					let Some(output_gradient) = gradients.remove(output_id) else {
						continue;
					};
					let result = crate::ml::matrix::moe_combine_backward(
						&output_gradient,
						packed,
						route_gate,
						inverse,
						packed_slot,
					)?;
					accumulate_value_gradient(&mut gradients, packed.value_id(), result.packed)?;
					accumulate_value_gradient(&mut gradients, route_gate.value_id(), result.route_gate)?;
				}
				GradNode::GroupedGemmM {
					input,
					weight,
					weight_value,
					offsets,
					output_id,
					..
				} => {
					let Some(output_gradient) = gradients.remove(output_id) else {
						continue;
					};
					let result = crate::ml::matrix::grouped_gemm_m_backward(
						&output_gradient,
						input,
						weight_value,
						offsets,
					)?;
					accumulate_value_gradient(&mut gradients, input.value_id(), result.input)?;
					accumulate_value_gradient(
						&mut gradients,
						weight_value.value_id(),
						result.weight.clone(),
					)?;
					if let Some(weight) = weight {
						weight.accumulate_gradient(result.weight)?;
					}
				}
				GradNode::GroupedLinearM {
					input,
					weight,
					weight_value,
					bias,
					bias_value,
					offsets,
					output_id,
					..
				} => {
					let Some(output_gradient) = gradients.remove(output_id) else {
						continue;
					};
					let result = crate::ml::matrix::grouped_linear_m_backward(
						&output_gradient,
						input,
						weight_value,
						offsets,
					)?;
					accumulate_value_gradient(&mut gradients, input.value_id(), result.input)?;
					accumulate_value_gradient(
						&mut gradients,
						weight_value.value_id(),
						result.weight.clone(),
					)?;
					accumulate_value_gradient(&mut gradients, bias_value.value_id(), result.bias.clone())?;
					if let Some(weight) = weight {
						weight.accumulate_gradient(result.weight)?;
					}
					if let Some(bias) = bias {
						bias.accumulate_gradient(result.bias)?;
					}
				}
				GradNode::Mamba3Siso(node) => {
					let Some(output_gradient) = gradients.remove(&node.output_id) else {
						continue;
					};
					let result = crate::ml::matrix::mamba3_siso_backward(
						&output_gradient,
						&node.c,
						&node.b,
						&node.x,
						&node.z,
						&node.adt,
						&node.dt,
						&node.trap,
						&node.angle,
						&node.c_bias,
						&node.b_bias,
						&node.d,
						node.config,
					)?;
					for (input, gradient) in [
						(&node.c, result.c),
						(&node.b, result.b),
						(&node.x, result.x),
						(&node.z, result.z),
						(&node.adt, result.adt),
						(&node.dt, result.dt),
						(&node.trap, result.trap),
						(&node.angle, result.angle),
						(&node.c_bias, result.c_bias),
						(&node.b_bias, result.b_bias),
						(&node.d, result.d),
					] {
						accumulate_value_gradient(&mut gradients, input.value_id(), gradient)?;
					}
				}
				GradNode::Mamba3Mimo(node) => {
					let Some(output_gradient) = gradients.remove(&node.output_id) else {
						continue;
					};
					let [
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
					] = &node.inputs;
					let result = crate::ml::matrix::mamba3_mimo_backward(
						&output_gradient,
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
						node.config,
					)?;
					for (input, gradient) in node.inputs.iter().zip([
						result.c,
						result.b,
						result.x,
						result.z,
						result.adt,
						result.dt,
						result.trap,
						result.angle,
						result.c_bias,
						result.b_bias,
						result.d,
						result.mimo_x,
						result.mimo_z,
						result.mimo_o,
						result.norm_weight,
					]) {
						accumulate_value_gradient(&mut gradients, input.value_id(), gradient)?;
					}
				}
				GradNode::Mamba3Preprocess(node) => {
					let output_gradients = crate::ml::matrix::Mamba3PreprocessResult {
						x: take_or_zero_gradient(&mut gradients, &node.outputs[0])?,
						z: take_or_zero_gradient(&mut gradients, &node.outputs[1])?,
						bh: take_or_zero_gradient(&mut gradients, &node.outputs[2])?,
						ch: take_or_zero_gradient(&mut gradients, &node.outputs[3])?,
						dt: take_or_zero_gradient(&mut gradients, &node.outputs[4])?,
						adt: take_or_zero_gradient(&mut gradients, &node.outputs[5])?,
						trap: take_or_zero_gradient(&mut gradients, &node.outputs[6])?,
						angle: take_or_zero_gradient(&mut gradients, &node.outputs[7])?,
					};
					let result = crate::ml::matrix::mamba3_preprocess_backward(
						&node.projected,
						&node.dt_bias,
						&output_gradients,
						node.config,
					)?;
					accumulate_value_gradient(&mut gradients, node.projected.value_id(), result.projected)?;
					accumulate_value_gradient(&mut gradients, node.dt_bias.value_id(), result.dt_bias)?;
				}
				GradNode::RnnScan(node) => {
					let Some(output_gradient) = gradients.remove(&node.output_id) else {
						continue;
					};
					let (gates_i_gradient, weight_gradient, bias_gradient) =
						crate::ml::matrix::rnn_scan_backward(
							&output_gradient,
							&node.gates_i,
							&node.hidden_previous,
							&node.weight_value,
							&node.bias_value,
							node.has_bias,
						)?;
					accumulate_value_gradient(&mut gradients, node.gates_i.value_id(), gates_i_gradient)?;
					accumulate_value_gradient(
						&mut gradients,
						node.weight_value.value_id(),
						weight_gradient.clone(),
					)?;
					if node.has_bias {
						accumulate_value_gradient(
							&mut gradients,
							node.bias_value.value_id(),
							bias_gradient.clone(),
						)?;
					}
					if let Some(weight) = &node.weight {
						weight.accumulate_gradient(weight_gradient)?;
					}
					if let Some(bias) = &node.bias {
						bias.accumulate_gradient(bias_gradient)?;
					}
				}
				GradNode::RnnCell(node) => {
					let Some(output_gradient) = gradients.remove(&node.output_id) else {
						continue;
					};
					let (gates_i_gradient, hidden_gradient, weight_gradient, bias_gradient) =
						crate::ml::matrix::rnn_cell_backward(
							&node.gates_i,
							&node.gates_h,
							&node.hidden,
							&output_gradient,
							&node.weight_value,
							&node.bias_value,
							node.has_bias,
						)?;
					accumulate_value_gradient(&mut gradients, node.gates_i.value_id(), gates_i_gradient)?;
					accumulate_value_gradient(&mut gradients, node.hidden.value_id(), hidden_gradient)?;
					accumulate_value_gradient(
						&mut gradients,
						node.weight_value.value_id(),
						weight_gradient.clone(),
					)?;
					if node.has_bias {
						accumulate_value_gradient(
							&mut gradients,
							node.bias_value.value_id(),
							bias_gradient.clone(),
						)?;
					}
					if let Some(weight) = &node.weight {
						weight.accumulate_gradient(weight_gradient)?;
					}
					if let Some(bias) = &node.bias {
						bias.accumulate_gradient(bias_gradient)?;
					}
				}
				GradNode::GruScan(node) => {
					let Some(output_gradient) = gradients.remove(&node.output_id) else {
						continue;
					};
					let (gates_i_gradient, weight_gradient, bias_gradient) =
						crate::ml::matrix::gru_scan_backward(
							&output_gradient,
							&node.gates_i,
							&node.hidden_previous,
							&node.weight_value,
							&node.bias_value,
							node.has_bias,
						)?;
					accumulate_value_gradient(&mut gradients, node.gates_i.value_id(), gates_i_gradient)?;
					accumulate_value_gradient(
						&mut gradients,
						node.weight_value.value_id(),
						weight_gradient.clone(),
					)?;
					if node.has_bias {
						accumulate_value_gradient(
							&mut gradients,
							node.bias_value.value_id(),
							bias_gradient.clone(),
						)?;
					}
					if let Some(weight) = &node.weight {
						weight.accumulate_gradient(weight_gradient)?;
					}
					if let Some(bias) = &node.bias {
						bias.accumulate_gradient(bias_gradient)?;
					}
				}
				GradNode::GruCell(node) => {
					let Some(output_gradient) = gradients.remove(&node.output_id) else {
						continue;
					};
					let (gates_i_gradient, hidden_gradient, weight_gradient, bias_gradient) =
						crate::ml::matrix::gru_cell_backward(
							&node.gates_i,
							&node.gates_h,
							&node.hidden,
							&output_gradient,
							&node.weight_value,
							&node.bias_value,
							node.has_bias,
						)?;
					accumulate_value_gradient(&mut gradients, node.gates_i.value_id(), gates_i_gradient)?;
					accumulate_value_gradient(&mut gradients, node.hidden.value_id(), hidden_gradient)?;
					accumulate_value_gradient(
						&mut gradients,
						node.weight_value.value_id(),
						weight_gradient.clone(),
					)?;
					if node.has_bias {
						accumulate_value_gradient(
							&mut gradients,
							node.bias_value.value_id(),
							bias_gradient.clone(),
						)?;
					}
					if let Some(weight) = &node.weight {
						weight.accumulate_gradient(weight_gradient)?;
					}
					if let Some(bias) = &node.bias {
						bias.accumulate_gradient(bias_gradient)?;
					}
				}
			}
			if let Some((forward, generation)) = attachment {
				engine.complete_semantic_autograd(forward, entry.sequence, generation, backward_first)?;
			}
		}
		for watched in self.recording.watched_parameters.borrow().iter() {
			if let Some(gradient) = gradients.remove(&watched.data_id) {
				watched.parameter.accumulate_gradient(gradient)?;
			}
		}
		Ok(())
	}
}

fn reached_output_id(node: &GradNode, gradients: &HashMap<u64, Matrix>) -> Option<u64> {
	match node {
		GradNode::Mamba3Preprocess(node) => node
			.outputs
			.iter()
			.map(Matrix::value_id)
			.find(|output_id| gradients.contains_key(output_id)),
		_ => gradients
			.contains_key(&node.output_id())
			.then(|| node.output_id()),
	}
}

fn take_or_zero_gradient(gradients: &mut HashMap<u64, Matrix>, output: &Matrix) -> Result<Matrix> {
	if let Some(gradient) = gradients.remove(&output.value_id()) {
		return Ok(gradient);
	}
	Matrix::from_slice_handle(
		output.engine_handle(),
		output.shape().to_vec(),
		&vec![0.0_f32; output.element_count()],
	)
}

pub(in crate::ml::autograd) fn accumulate_value_gradient(
	gradients: &mut HashMap<u64, Matrix>,
	value_id: u64,
	gradient: Matrix,
) -> Result<()> {
	if let Some(previous) = gradients.remove(&value_id) {
		gradients.insert(value_id, matrix::add(&previous, &gradient)?);
	} else {
		gradients.insert(value_id, gradient);
	}
	Ok(())
}

fn sum_to_flow_time_shape(
	gradient: &Matrix,
	time_shape: &[usize],
	state_shape: &[usize],
) -> Result<Matrix> {
	if time_shape.len() == 1
		&& state_shape.len() > 1
		&& time_shape[0] != 1
		&& time_shape[0] == state_shape[0]
	{
		let mut result = gradient.clone();
		for axis in 1..state_shape.len() {
			result = matrix::sum(
				&result,
				i32::try_from(axis).map_err(|_| Error::internal("flow time adjoint axis exceeds i32"))?,
			)?;
		}
		return result.reshape(time_shape.to_vec());
	}
	sum_to_shape(gradient, time_shape)
}

// One borrowed adapter over the tape's existing gradient map. No extra storage.
struct MatrixGradients<'a>(&'a mut HashMap<u64, Matrix>);

impl GradientContext for MatrixGradients<'_> {
	fn take_output(&mut self, output_id: u64) -> Option<Matrix> {
		self.0.remove(&output_id)
	}
	fn accumulate(&mut self, value_id: u64, gradient: Matrix) -> Result<()> {
		accumulate_value_gradient(self.0, value_id, gradient)
	}
}
