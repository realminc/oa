use std::{
	cell::{Cell, RefCell},
	collections::HashMap,
	rc::{Rc, Weak},
};

use crate::{
	DType, Error, Matrix, Result,
	core::autograd::{self as core_autograd, MatrixNode},
	matrix,
};

use crate::ml::lowering::{attention as attention_lowering, loss, matrix as matrix_lowering};

use super::node::Node;

thread_local! {
	static ACTIVE_TAPES: RefCell<Vec<Weak<TapeInner>>> = const { RefCell::new(Vec::new()) };
	static NEXT_SEQUENCE: Cell<u64> = const { Cell::new(0) };
}

struct TapeEntry {
	sequence: u64,
	node: Node,
}

struct TapeInner {
	nodes: RefCell<Vec<TapeEntry>>,
	consumed: Cell<bool>,
}

/// Thread-affine reverse-mode recording scope.
///
/// Constructing a tape selects it for operations on the current thread.
/// [`GradientTape::backward`] closes the recording scope and records the
/// corresponding backward operations into the originating engine's eager batch.
#[must_use]
pub struct GradientTape {
	inner: Rc<TapeInner>,
	active: Cell<bool>,
	matrix_recording: RefCell<Option<core_autograd::RecordingSelection>>,
}

impl GradientTape {
	/// Begin a reverse-mode recording scope on the current thread.
	pub fn new() -> Self {
		let inner = Rc::new(TapeInner {
			nodes: RefCell::new(Vec::new()),
			consumed: Cell::new(false),
		});
		ACTIVE_TAPES.with(|tapes| tapes.borrow_mut().push(Rc::downgrade(&inner)));
		let weak = Rc::downgrade(&inner);
		let matrix_recording = core_autograd::select(move |node| {
			let Some(tape) = weak.upgrade() else {
				return Ok(());
			};
			let node = match node {
				MatrixNode::Reshape { input, output_id } => Node::Reshape { input, output_id },
				MatrixNode::Add {
					left,
					right,
					output_id,
				} => Node::Add {
					left,
					right,
					output_id,
				},
				MatrixNode::Dropout {
					input,
					output_id,
					probability,
					seed,
				} => Node::Dropout {
					input,
					output_id,
					probability,
					seed,
				},
				MatrixNode::Softmax {
					input,
					output,
					output_id,
					dim,
				} => Node::Softmax {
					input,
					output,
					output_id,
					dim,
				},
				MatrixNode::LogSoftmax {
					input,
					output,
					output_id,
					dim,
				} => Node::LogSoftmax {
					input,
					output,
					output_id,
					dim,
				},
				MatrixNode::Sum {
					input,
					output_id,
					dim,
				} => Node::Sum {
					input,
					output_id,
					dim,
				},
			};
			record_node_for(&tape, node)
		});
		Self {
			inner,
			active: Cell::new(true),
			matrix_recording: RefCell::new(Some(matrix_recording)),
		}
	}

	/// Close recording without constructing a backward pass.
	pub fn close(&self) {
		if !self.active.replace(false) {
			return;
		}
		self.matrix_recording.borrow_mut().take();
		ACTIVE_TAPES.with(|tapes| {
			tapes.borrow_mut().retain(|candidate| {
				candidate.strong_count() != 0 && !candidate.ptr_eq(&Rc::downgrade(&self.inner))
			});
		});
	}

	/// Record the reverse-mode operations leading to one scalar loss.
	///
	/// The current checkpoint supports the admitted scalar mean-loss roots and any recorded
	/// chain of admitted Linear, Embedding, and reshape operations leading into
	/// it. Gradients accumulate on stable [`crate::ml::Parameter`] handles. This method
	/// records but does not submit or wait.
	///
	/// # Errors
	///
	/// Returns an error when the tape was already consumed, `root` is not one of its
	/// scalar loss results, a saved parameter changed before backward, or
	/// gradient operation validation/allocation/recording fails.
	pub fn backward(&self, root: &Matrix) -> Result<()> {
		let self_weak = Rc::downgrade(&self.inner);
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
		if self.inner.consumed.get() {
			return Err(Error::failed_precondition(
				"gradient tape has already been consumed",
			));
		}
		if !root.shape().is_empty() || root.dtype() != DType::F32 {
			return Err(Error::invalid_argument(
				"gradient tape root must be an FP32 scalar",
			));
		}
		let nodes = self.inner.nodes.borrow();
		let root_found = nodes
			.iter()
			.any(|entry| entry.node.is_loss_root(root.value_id()));
		if !root_found {
			return Err(Error::failed_precondition(
				"gradient tape root was not produced by this tape",
			));
		}
		for entry in nodes.iter() {
			match &entry.node {
				Node::Linear {
					weight,
					weight_version,
					bias,
					bias_version,
					..
				} => {
					weight.validate_version(*weight_version)?;
					if let (Some(bias), Some(bias_version)) = (bias, bias_version) {
						bias.validate_version(*bias_version)?;
					}
				}
				Node::Conv2d {
					weight,
					weight_version,
					bias,
					bias_version,
					..
				} => {
					if let (Some(weight), Some(weight_version)) = (weight, weight_version) {
						weight.validate_version(*weight_version)?;
					}
					if let (Some(bias), Some(bias_version)) = (bias, bias_version) {
						bias.validate_version(*bias_version)?;
					}
				}
				Node::Conv1d {
					weight,
					weight_version,
					bias,
					bias_version,
					..
				} => {
					if let (Some(weight), Some(weight_version)) = (weight, weight_version) {
						weight.validate_version(*weight_version)?;
					}
					if let (Some(bias), Some(bias_version)) = (bias, bias_version) {
						bias.validate_version(*bias_version)?;
					}
				}
				Node::ConvTranspose1d {
					weight,
					weight_version,
					..
				} => {
					if let (Some(weight), Some(weight_version)) = (weight, weight_version) {
						weight.validate_version(*weight_version)?;
					}
				}
				Node::ConvTranspose2d {
					weight,
					weight_version,
					bias,
					bias_version,
					..
				} => {
					if let (Some(weight), Some(weight_version)) = (weight, weight_version) {
						weight.validate_version(*weight_version)?;
					}
					if let (Some(bias), Some(bias_version)) = (bias, bias_version) {
						bias.validate_version(*bias_version)?;
					}
				}
				Node::Embedding {
					weight,
					weight_version,
					..
				} => weight.validate_version(*weight_version)?,
				Node::LayerNorm {
					weight,
					weight_version,
					bias,
					bias_version,
					..
				} => {
					weight.validate_version(*weight_version)?;
					bias.validate_version(*bias_version)?;
				}
				Node::BatchNorm2d {
					weight: Some(weight),
					weight_version: Some(weight_version),
					bias: Some(bias),
					bias_version: Some(bias_version),
					..
				} => {
					weight.validate_version(*weight_version)?;
					bias.validate_version(*bias_version)?;
				}
				Node::RmsNorm {
					weight: Some(weight),
					weight_version: Some(weight_version),
					..
				} => weight.validate_version(*weight_version)?,
				Node::GruCell(node) => {
					if let (Some(weight), Some(version)) = (&node.weight, node.weight_version) {
						weight.validate_version(version)?;
					}
					if let (Some(bias), Some(version)) = (&node.bias, node.bias_version) {
						bias.validate_version(version)?;
					}
				}
				Node::GruScan(node) => {
					if let (Some(weight), Some(version)) = (&node.weight, node.weight_version) {
						weight.validate_version(version)?;
					}
					if let (Some(bias), Some(version)) = (&node.bias, node.bias_version) {
						bias.validate_version(version)?;
					}
				}
				Node::RnnCell(node) => {
					if let (Some(weight), Some(version)) = (&node.weight, node.weight_version) {
						weight.validate_version(version)?;
					}
					if let (Some(bias), Some(version)) = (&node.bias, node.bias_version) {
						bias.validate_version(version)?;
					}
				}
				Node::RnnScan(node) => {
					if let (Some(weight), Some(version)) = (&node.weight, node.weight_version) {
						weight.validate_version(version)?;
					}
					if let (Some(bias), Some(version)) = (&node.bias, node.bias_version) {
						bias.validate_version(version)?;
					}
				}
				Node::Add { .. }
				| Node::Reshape { .. }
				| Node::Dropout { .. }
				| Node::Gelu { .. }
				| Node::RmsNorm { .. }
				| Node::BatchNorm2d { .. }
				| Node::Rope { .. }
				| Node::Silu { .. }
				| Node::Relu { .. }
				| Node::Tanh { .. }
				| Node::Sigmoid { .. }
				| Node::LeakyRelu { .. }
				| Node::Elu { .. }
				| Node::Mish { .. }
				| Node::Softplus { .. }
				| Node::Softmax { .. }
				| Node::LogSoftmax { .. }
				| Node::AvgPool2d { .. }
				| Node::MaxPool2d { .. }
				| Node::AdaptiveAvgPool2d { .. }
				| Node::Upsample2d { .. }
				| Node::Sum { .. }
				| Node::Swiglu { .. }
				| Node::Bmm { .. }
				| Node::BmmNt { .. }
				| Node::BmmTn { .. }
				| Node::SplitHeads { .. }
				| Node::MergeHeads { .. }
				| Node::SoftmaxScaledMasked { .. }
				| Node::ScaledDotProductAttention { .. }
				| Node::FlashAttention { .. }
				| Node::MoeRouteWeights { .. }
				| Node::CrossEntropy { .. }
				| Node::MaskedCrossEntropy { .. }
				| Node::SmoothL1 { .. }
				| Node::Mse { .. }
				| Node::L1 { .. }
				| Node::Bce { .. } => {}
			}
		}
		self.inner.consumed.set(true);

		let mut gradients = HashMap::<u64, Matrix>::new();
		let engine = root.engine_handle().clone();
		for entry in nodes.iter().rev() {
			let node = &entry.node;
			let reaches_root =
				node.is_loss_root(root.value_id()) || gradients.contains_key(&node.output_id());
			if !reaches_root {
				continue;
			}
			let attachment = if matches!(node, Node::Reshape { .. }) {
				None
			} else {
				engine.attach_semantic_autograd(node.output_id(), entry.sequence)?
			};
			let backward_first = engine.semantic_operation_count();
			match node {
				Node::Add {
					left,
					right,
					output_id,
				} => {
					let Some(output_gradient) = gradients.remove(output_id) else {
						continue;
					};
					accumulate_value_gradient(
						&mut gradients,
						left.value_id(),
						output_gradient.clone(),
					)?;
					accumulate_value_gradient(&mut gradients, right.value_id(), output_gradient)?;
				}
				Node::Dropout {
					input,
					output_id,
					probability,
					seed,
				} => {
					let Some(output_gradient) = gradients.remove(output_id) else {
						continue;
					};
					let input_gradient =
						matrix::dropout_backward(&output_gradient, *probability, *seed)?;
					accumulate_value_gradient(&mut gradients, input.value_id(), input_gradient)?;
				}
				Node::CrossEntropy {
					logits,
					targets,
					output_id,
				} if *output_id == root.value_id() => {
					let gradient = loss::cross_entropy_backward(logits, targets)?;
					accumulate_value_gradient(&mut gradients, logits.value_id(), gradient)?;
				}
				Node::MaskedCrossEntropy {
					logits,
					targets,
					mask,
					valid_count,
					output_id,
				} if *output_id == root.value_id() => {
					let gradient =
						loss::masked_cross_entropy_backward(logits, targets, mask, *valid_count)?;
					accumulate_value_gradient(&mut gradients, logits.value_id(), gradient)?;
				}
				Node::SmoothL1 {
					prediction,
					target,
					output_id,
				} if *output_id == root.value_id() => {
					let gradient = loss::smooth_l1_backward(prediction, target)?;
					accumulate_value_gradient(&mut gradients, prediction.value_id(), gradient)?;
				}
				Node::Mse {
					prediction,
					target,
					output_id,
				} if *output_id == root.value_id() => {
					let gradient = loss::mse_backward(prediction, target)?;
					accumulate_value_gradient(&mut gradients, prediction.value_id(), gradient)?;
				}
				Node::L1 {
					prediction,
					target,
					output_id,
				} if *output_id == root.value_id() => {
					let gradient = loss::l1_backward(prediction, target)?;
					accumulate_value_gradient(&mut gradients, prediction.value_id(), gradient)?;
				}
				Node::Bce {
					prediction,
					target,
					output_id,
				} if *output_id == root.value_id() => {
					let gradient = loss::bce_backward(prediction, target)?;
					accumulate_value_gradient(&mut gradients, prediction.value_id(), gradient)?;
				}
				Node::Linear {
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
						matrix_lowering::linear_backward(input, weight_value, &output_gradient)?;
					accumulate_value_gradient(&mut gradients, input.value_id(), input_gradient)?;
					weight.accumulate_gradient(weight_gradient)?;
					if let Some(bias) = bias {
						bias.accumulate_gradient(bias_gradient)?;
					}
				}
				Node::Conv2d {
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
					accumulate_value_gradient(
						&mut gradients,
						bias_value.value_id(),
						bias_gradient.clone(),
					)?;
					if let Some(weight) = weight {
						weight.accumulate_gradient(weight_gradient)?;
					}
					if let Some(bias) = bias {
						bias.accumulate_gradient(bias_gradient)?;
					}
				}
				Node::Conv1d {
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
					accumulate_value_gradient(
						&mut gradients,
						bias_value.value_id(),
						bias_gradient.clone(),
					)?;
					if let Some(weight) = weight {
						weight.accumulate_gradient(weight_gradient)?;
					}
					if let Some(bias) = bias {
						bias.accumulate_gradient(bias_gradient)?;
					}
				}
				Node::ConvTranspose1d {
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
					let (input_gradient, weight_gradient) =
						crate::ml::matrix::conv_transpose_1d_backward(
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
				Node::ConvTranspose2d {
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
					accumulate_value_gradient(
						&mut gradients,
						bias_value.value_id(),
						bias_gradient.clone(),
					)?;
					if let Some(weight) = weight {
						weight.accumulate_gradient(weight_gradient)?;
					}
					if let Some(bias) = bias {
						bias.accumulate_gradient(bias_gradient)?;
					}
				}
				Node::Embedding {
					indices,
					output_id,
					weight,
					weight_value,
					..
				} => {
					let Some(output_gradient) = gradients.remove(output_id) else {
						continue;
					};
					let weight_gradient = matrix_lowering::embedding_backward(
						indices,
						&output_gradient,
						weight_value,
					)?;
					weight.accumulate_gradient(weight_gradient)?;
				}
				Node::LayerNorm {
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
					let (input_gradient, weight_gradient, bias_gradient) =
						matrix_lowering::layer_norm_backward(
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
				Node::BatchNorm2d {
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
					accumulate_value_gradient(
						&mut gradients,
						bias_value.value_id(),
						bias_gradient.clone(),
					)?;
					if let Some(weight) = weight {
						weight.accumulate_gradient(weight_gradient)?;
					}
					if let Some(bias) = bias {
						bias.accumulate_gradient(bias_gradient)?;
					}
				}
				Node::RmsNorm {
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
					let (input_gradient, weight_gradient) = matrix_lowering::rms_norm_backward(
						input,
						weight_value,
						&output_gradient,
						*epsilon,
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
				Node::Rope {
					input,
					output_id,
					num_heads,
					head_dim,
					theta_base,
					position_offset,
				} => {
					let Some(output_gradient) = gradients.remove(output_id) else {
						continue;
					};
					let input_gradient = matrix_lowering::rope_backward(
						&output_gradient,
						*num_heads,
						*head_dim,
						*theta_base,
						*position_offset,
					)?;
					accumulate_value_gradient(&mut gradients, input.value_id(), input_gradient)?;
				}
				Node::Gelu { input, output_id } => {
					let Some(output_gradient) = gradients.remove(output_id) else {
						continue;
					};
					let input_gradient = matrix_lowering::gelu_backward(input, &output_gradient)?;
					accumulate_value_gradient(&mut gradients, input.value_id(), input_gradient)?;
				}
				Node::Silu { input, output_id } => {
					let Some(output_gradient) = gradients.remove(output_id) else {
						continue;
					};
					let input_gradient = matrix_lowering::silu_backward(input, &output_gradient)?;
					accumulate_value_gradient(&mut gradients, input.value_id(), input_gradient)?;
				}
				Node::Relu {
					input,
					output,
					output_id,
				} => {
					let Some(output_gradient) = gradients.remove(output_id) else {
						continue;
					};
					let input_gradient = matrix_lowering::relu_backward(output, &output_gradient)?;
					accumulate_value_gradient(&mut gradients, input.value_id(), input_gradient)?;
				}
				Node::Tanh {
					input,
					output,
					output_id,
				} => {
					let Some(output_gradient) = gradients.remove(output_id) else {
						continue;
					};
					let input_gradient = matrix_lowering::tanh_backward(output, &output_gradient)?;
					accumulate_value_gradient(&mut gradients, input.value_id(), input_gradient)?;
				}
				Node::Sigmoid {
					input,
					output,
					output_id,
				} => {
					let Some(output_gradient) = gradients.remove(output_id) else {
						continue;
					};
					let input_gradient =
						matrix_lowering::sigmoid_backward(output, &output_gradient)?;
					accumulate_value_gradient(&mut gradients, input.value_id(), input_gradient)?;
				}
				Node::LeakyRelu {
					input,
					alpha,
					output_id,
				} => {
					let Some(output_gradient) = gradients.remove(output_id) else {
						continue;
					};
					let input_gradient =
						matrix_lowering::leaky_relu_backward(input, &output_gradient, *alpha)?;
					accumulate_value_gradient(&mut gradients, input.value_id(), input_gradient)?;
				}
				Node::Elu {
					input,
					output,
					alpha,
					output_id,
				} => {
					let Some(output_gradient) = gradients.remove(output_id) else {
						continue;
					};
					let input_gradient =
						matrix_lowering::elu_backward(output, &output_gradient, *alpha)?;
					accumulate_value_gradient(&mut gradients, input.value_id(), input_gradient)?;
				}
				Node::Mish { input, output_id } => {
					let Some(output_gradient) = gradients.remove(output_id) else {
						continue;
					};
					let input_gradient = matrix_lowering::mish_backward(input, &output_gradient)?;
					accumulate_value_gradient(&mut gradients, input.value_id(), input_gradient)?;
				}
				Node::Softplus {
					input,
					output,
					output_id,
				} => {
					let Some(output_gradient) = gradients.remove(output_id) else {
						continue;
					};
					let input_gradient =
						matrix_lowering::softplus_backward(output, &output_gradient)?;
					accumulate_value_gradient(&mut gradients, input.value_id(), input_gradient)?;
				}
				Node::Softmax {
					input,
					output,
					output_id,
					dim,
				} => {
					let Some(output_gradient) = gradients.remove(output_id) else {
						continue;
					};
					let input_gradient = matrix::softmax_backward(output, &output_gradient, *dim)?;
					accumulate_value_gradient(&mut gradients, input.value_id(), input_gradient)?;
				}
				Node::LogSoftmax {
					input,
					output,
					output_id,
					dim,
				} => {
					let Some(output_gradient) = gradients.remove(output_id) else {
						continue;
					};
					let input_gradient =
						matrix::log_softmax_backward(output, &output_gradient, *dim)?;
					accumulate_value_gradient(&mut gradients, input.value_id(), input_gradient)?;
				}
				Node::AvgPool2d {
					input,
					output_id,
					kernel_size,
					stride,
					padding,
				} => {
					let Some(output_gradient) = gradients.remove(output_id) else {
						continue;
					};
					let input_gradient = crate::ml::matrix::avg_pool_2d_backward(
						input,
						&output_gradient,
						*kernel_size,
						*stride,
						*padding,
					)?;
					accumulate_value_gradient(&mut gradients, input.value_id(), input_gradient)?;
				}
				Node::MaxPool2d {
					input,
					indices,
					output_id,
					kernel_size,
					stride,
					padding,
				} => {
					let Some(output_gradient) = gradients.remove(output_id) else {
						continue;
					};
					let input_gradient = crate::ml::matrix::max_pool_2d_backward(
						input,
						indices,
						&output_gradient,
						*kernel_size,
						*stride,
						*padding,
					)?;
					accumulate_value_gradient(&mut gradients, input.value_id(), input_gradient)?;
				}
				Node::AdaptiveAvgPool2d {
					input,
					output_id,
					output_height,
					output_width,
				} => {
					let Some(output_gradient) = gradients.remove(output_id) else {
						continue;
					};
					let input_gradient = crate::ml::matrix::adaptive_avg_pool_2d_backward(
						input,
						&output_gradient,
						*output_height,
						*output_width,
					)?;
					accumulate_value_gradient(&mut gradients, input.value_id(), input_gradient)?;
				}
				Node::Upsample2d {
					input,
					output_id,
					scale_factor,
					mode,
				} => {
					let Some(output_gradient) = gradients.remove(output_id) else {
						continue;
					};
					let input_gradient = crate::ml::matrix::upsample_2d_backward(
						input,
						&output_gradient,
						*scale_factor,
						*mode,
					)?;
					accumulate_value_gradient(&mut gradients, input.value_id(), input_gradient)?;
				}
				Node::Sum {
					input,
					output_id,
					dim,
				} => {
					let Some(output_gradient) = gradients.remove(output_id) else {
						continue;
					};
					let input_gradient = matrix::sum_backward(input, &output_gradient, *dim)?;
					accumulate_value_gradient(&mut gradients, input.value_id(), input_gradient)?;
				}
				Node::Swiglu {
					gate,
					up,
					output_id,
				} => {
					let Some(output_gradient) = gradients.remove(output_id) else {
						continue;
					};
					let (gate_gradient, up_gradient) =
						matrix_lowering::swiglu_backward(gate, up, &output_gradient)?;
					accumulate_value_gradient(&mut gradients, gate.value_id(), gate_gradient)?;
					accumulate_value_gradient(&mut gradients, up.value_id(), up_gradient)?;
				}
				Node::Bmm {
					left,
					right,
					output_id,
				} => {
					let Some(output_gradient) = gradients.remove(output_id) else {
						continue;
					};
					let left_gradient = attention_lowering::bmm_nt(&output_gradient, right)?;
					let right_gradient = attention_lowering::bmm_tn(left, &output_gradient)?;
					accumulate_value_gradient(&mut gradients, left.value_id(), left_gradient)?;
					accumulate_value_gradient(&mut gradients, right.value_id(), right_gradient)?;
				}
				Node::BmmNt {
					left,
					right,
					output_id,
				} => {
					let Some(output_gradient) = gradients.remove(output_id) else {
						continue;
					};
					let left_gradient = attention_lowering::bmm(&output_gradient, right)?;
					let right_gradient = attention_lowering::bmm_tn(&output_gradient, left)?;
					accumulate_value_gradient(&mut gradients, left.value_id(), left_gradient)?;
					accumulate_value_gradient(&mut gradients, right.value_id(), right_gradient)?;
				}
				Node::BmmTn {
					left,
					right,
					output_id,
				} => {
					let Some(output_gradient) = gradients.remove(output_id) else {
						continue;
					};
					let left_gradient = attention_lowering::bmm_nt(right, &output_gradient)?;
					let right_gradient = attention_lowering::bmm(left, &output_gradient)?;
					accumulate_value_gradient(&mut gradients, left.value_id(), left_gradient)?;
					accumulate_value_gradient(&mut gradients, right.value_id(), right_gradient)?;
				}
				Node::SplitHeads {
					input,
					output_id,
					batch,
					sequence_length,
					num_heads,
				} => {
					let Some(output_gradient) = gradients.remove(output_id) else {
						continue;
					};
					let input_gradient = attention_lowering::merge_heads(
						&output_gradient,
						*batch,
						*sequence_length,
						*num_heads,
					)?
					.output;
					accumulate_value_gradient(&mut gradients, input.value_id(), input_gradient)?;
				}
				Node::MergeHeads {
					input,
					output_id,
					batch,
					sequence_length,
					num_heads,
				} => {
					let Some(output_gradient) = gradients.remove(output_id) else {
						continue;
					};
					let input_gradient = attention_lowering::split_heads(
						&output_gradient,
						*batch,
						*sequence_length,
						*num_heads,
					)?
					.output;
					accumulate_value_gradient(&mut gradients, input.value_id(), input_gradient)?;
				}
				Node::SoftmaxScaledMasked {
					input,
					output,
					output_id,
					scale,
				} => {
					let Some(output_gradient) = gradients.remove(output_id) else {
						continue;
					};
					let input_gradient = attention_lowering::softmax_scaled_masked_backward(
						output,
						&output_gradient,
						*scale,
					)?;
					accumulate_value_gradient(&mut gradients, input.value_id(), input_gradient)?;
				}
				Node::ScaledDotProductAttention {
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
					let probability_gradient = attention_lowering::bmm_nt(&output_gradient, value)?;
					let [batch_heads, sequence_length, _] = probabilities.shape() else {
						return Err(Error::internal(
							"saved SDPA probabilities lost rank-three shape",
						));
					};
					let rows = batch_heads.checked_mul(*sequence_length).ok_or_else(|| {
						Error::internal("saved SDPA probability rows overflow usize")
					})?;
					let score_gradient = attention_lowering::softmax_scaled_masked_backward(
						&probabilities.reshape([rows, *sequence_length])?,
						&probability_gradient.reshape([rows, *sequence_length])?,
						*scale,
					)?
					.reshape([*batch_heads, *sequence_length, *sequence_length])?;
					let query_gradient = attention_lowering::bmm(&score_gradient, key)?;
					let key_gradient = attention_lowering::bmm_tn(&score_gradient, query)?;
					let value_gradient =
						attention_lowering::bmm_tn(probabilities, &output_gradient)?;
					accumulate_value_gradient(&mut gradients, query.value_id(), query_gradient)?;
					accumulate_value_gradient(&mut gradients, key.value_id(), key_gradient)?;
					accumulate_value_gradient(&mut gradients, value.value_id(), value_gradient)?;
				}
				Node::FlashAttention {
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
					let result = attention_lowering::flash_attention_causal_backward(
						query,
						key,
						value,
						output,
						log_sum_exp,
						&output_gradient,
						*scale,
					)?;
					accumulate_value_gradient(&mut gradients, query.value_id(), result.query)?;
					accumulate_value_gradient(&mut gradients, key.value_id(), result.key)?;
					accumulate_value_gradient(&mut gradients, value.value_id(), result.value)?;
				}
				Node::MoeRouteWeights {
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
				Node::RnnScan(node) => {
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
					accumulate_value_gradient(
						&mut gradients,
						node.gates_i.value_id(),
						gates_i_gradient,
					)?;
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
				Node::RnnCell(node) => {
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
					accumulate_value_gradient(
						&mut gradients,
						node.gates_i.value_id(),
						gates_i_gradient,
					)?;
					accumulate_value_gradient(
						&mut gradients,
						node.hidden.value_id(),
						hidden_gradient,
					)?;
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
				Node::GruScan(node) => {
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
					accumulate_value_gradient(
						&mut gradients,
						node.gates_i.value_id(),
						gates_i_gradient,
					)?;
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
				Node::GruCell(node) => {
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
					accumulate_value_gradient(
						&mut gradients,
						node.gates_i.value_id(),
						gates_i_gradient,
					)?;
					accumulate_value_gradient(
						&mut gradients,
						node.hidden.value_id(),
						hidden_gradient,
					)?;
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
				Node::Reshape { input, output_id } => {
					let Some(output_gradient) = gradients.remove(output_id) else {
						continue;
					};
					let input_gradient = output_gradient.reshape(input.shape().to_vec())?;
					accumulate_value_gradient(&mut gradients, input.value_id(), input_gradient)?;
				}
				Node::CrossEntropy { .. }
				| Node::MaskedCrossEntropy { .. }
				| Node::SmoothL1 { .. }
				| Node::Mse { .. }
				| Node::L1 { .. }
				| Node::Bce { .. } => {}
			}
			if let Some((forward, generation)) = attachment {
				engine.complete_semantic_autograd(
					forward,
					entry.sequence,
					generation,
					backward_first,
				)?;
			}
		}
		Ok(())
	}
}

impl Default for GradientTape {
	fn default() -> Self {
		Self::new()
	}
}

impl Drop for GradientTape {
	fn drop(&mut self) {
		self.close();
	}
}

pub(super) fn record_node(node: Node) -> Result<()> {
	ACTIVE_TAPES.with(|tapes| {
		let active = tapes.borrow().last().and_then(Weak::upgrade);
		if let Some(active) = active {
			record_node_for(&active, node)?;
		}
		Ok(())
	})
}

fn record_node_for(tape: &TapeInner, node: Node) -> Result<()> {
	let sequence = NEXT_SEQUENCE.with(|next| {
		let sequence = next
			.get()
			.checked_add(1)
			.ok_or_else(|| Error::resource_exhausted("autograd sequence exhausted"))?;
		next.set(sequence);
		Ok(sequence)
	})?;
	tape.nodes.borrow_mut().push(TapeEntry { sequence, node });
	Ok(())
}

fn accumulate_value_gradient(
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
