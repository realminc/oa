use std::{
	cell::{Cell, RefCell},
	collections::HashMap,
	rc::{Rc, Weak},
};

use crate::ml::Parameter;
use crate::{
	DType, Error, Matrix, Result,
	core::autograd::{self as core_autograd, MatrixNode},
	matrix,
};

use crate::ml::lowering::{
	attention as attention_lowering, flow as flow_lowering, loss, matrix as matrix_lowering,
};

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
	watched_parameters: RefCell<Vec<WatchedParameter>>,
	consumed: Cell<bool>,
}

struct WatchedParameter {
	parameter: Parameter,
	data_id: u64,
	version: u64,
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
			watched_parameters: RefCell::new(Vec::new()),
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
				MatrixNode::Mul {
					left,
					right,
					output_id,
				} => Node::Mul {
					left,
					right,
					output_id,
				},
				MatrixNode::Div {
					left,
					right,
					output_id,
				} => Node::Div {
					left,
					right,
					output_id,
				},
				MatrixNode::Scale {
					input,
					output_id,
					scalar,
				} => Node::Scale {
					input,
					output_id,
					scalar,
				},
				MatrixNode::Reciprocal {
					input,
					output,
					output_id,
				} => Node::Reciprocal {
					input,
					output,
					output_id,
				},
				MatrixNode::Exp {
					input,
					output,
					output_id,
				} => Node::Exp {
					input,
					output,
					output_id,
				},
				MatrixNode::Log { input, output_id } => Node::Log { input, output_id },
				MatrixNode::Abs { input, output_id } => Node::Abs { input, output_id },
				MatrixNode::Copy { input, output_id } => Node::Copy { input, output_id },
				MatrixNode::Sqrt {
					input,
					output,
					output_id,
				} => Node::Sqrt {
					input,
					output,
					output_id,
				},
				MatrixNode::ClampMax {
					input,
					output_id,
					maximum,
				} => Node::ClampMax {
					input,
					output_id,
					maximum,
				},
				MatrixNode::ClampMin {
					input,
					output_id,
					minimum,
				} => Node::ClampMin {
					input,
					output_id,
					minimum,
				},
				MatrixNode::Sub {
					left,
					right,
					output_id,
				} => Node::Sub {
					left,
					right,
					output_id,
				},
				MatrixNode::Slice {
					input,
					output_id,
					dim,
					start,
					end,
				} => Node::Slice {
					input,
					output_id,
					dim,
					start,
					end,
				},
				MatrixNode::RepeatInterleave {
					input,
					output_id,
					repeats,
					dim,
				} => Node::RepeatInterleave {
					input,
					output_id,
					repeats,
					dim,
				},
				MatrixNode::Concat {
					inputs,
					output_id,
					dim,
					sizes,
				} => Node::Concat {
					inputs,
					output_id,
					dim,
					sizes,
				},
				MatrixNode::Transpose {
					input,
					output_id,
					dim0,
					dim1,
				} => Node::Transpose {
					input,
					output_id,
					dim0,
					dim1,
				},
				MatrixNode::Gather {
					input,
					indices,
					output_id,
				} => Node::Gather {
					input,
					indices,
					output_id,
				},
				MatrixNode::GatherLastDim {
					input,
					indices,
					output_id,
					input_width,
				} => Node::GatherLastDim {
					input,
					indices,
					output_id,
					input_width,
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
				MatrixNode::MatMulNt {
					left,
					right,
					output_id,
				} => Node::MatMulNt {
					left,
					right,
					output_id,
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
			.any(|entry| entry.node.produces(root.value_id()));
		if !root_found {
			return Err(Error::failed_precondition(
				"gradient tape root was not produced by this tape",
			));
		}
		for entry in nodes.iter() {
			match &entry.node {
				Node::GroupedGemmM {
					weight,
					weight_version,
					..
				} => {
					if let (Some(weight), Some(weight_version)) = (weight, weight_version) {
						weight.validate_version(*weight_version)?;
					}
				}
				Node::GroupedLinearM {
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
				Node::RmsNormGated(node) => {
					if let (Some(weight), Some(version)) = (&node.weight, node.weight_version) {
						weight.validate_version(version)?;
					}
					if let (Some(bias), Some(version)) = (&node.bias, node.bias_version) {
						bias.validate_version(version)?;
					}
				}
				Node::ChannelNorm(node) => {
					if let (Some(weight), Some(version)) = (&node.weight, node.weight_version) {
						weight.validate_version(version)?;
					}
					if let (Some(bias), Some(version)) = (&node.bias, node.bias_version) {
						bias.validate_version(version)?;
					}
				}
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
				| Node::Mul { .. }
				| Node::Div { .. }
				| Node::Scale { .. }
				| Node::Reciprocal { .. }
				| Node::Exp { .. }
				| Node::Log { .. }
				| Node::Abs { .. }
				| Node::Copy { .. }
				| Node::Sqrt { .. }
				| Node::ClampMax { .. }
				| Node::ClampMin { .. }
				| Node::Sub { .. }
				| Node::Slice { .. }
				| Node::RepeatInterleave { .. }
				| Node::Concat { .. }
				| Node::Transpose { .. }
				| Node::Gather { .. }
				| Node::GatherLastDim { .. }
				| Node::FlowLinearState { .. }
				| Node::FlowLinearVelocity { .. }
				| Node::FlowEulerStep { .. }
				| Node::FlowMaskedMse { .. }
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
				| Node::MatMulNt { .. }
				| Node::Swiglu { .. }
				| Node::SiluMul { .. }
				| Node::Bmm { .. }
				| Node::BmmNt { .. }
				| Node::BmmTn { .. }
				| Node::SplitHeads { .. }
				| Node::MergeHeads { .. }
				| Node::SoftmaxScaledMasked { .. }
				| Node::ScaledDotProductAttention { .. }
				| Node::FlashAttention { .. }
				| Node::MoeRouteWeights { .. }
				| Node::MoeGather { .. }
				| Node::MoeCombine { .. }
				| Node::CrossEntropy { .. }
				| Node::MaskedCrossEntropy { .. }
				| Node::SmoothL1 { .. }
				| Node::Mse { .. }
				| Node::L1 { .. }
				| Node::Bce { .. }
				| Node::Mamba3Preprocess(_)
				| Node::Mamba3Siso(_)
				| Node::Mamba3Mimo(_)
				| Node::PpoClippedPolicy { .. } => {}
			}
		}
		for watched in self.inner.watched_parameters.borrow().iter() {
			watched.parameter.validate_version(watched.version)?;
		}
		self.inner.consumed.set(true);

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
						sum_to_shape(&output_gradient, left.shape())?,
					)?;
					accumulate_value_gradient(
						&mut gradients,
						right.value_id(),
						sum_to_shape(&output_gradient, right.shape())?,
					)?;
				}
				Node::Mul {
					left,
					right,
					output_id,
				} => {
					let Some(output_gradient) = gradients.remove(output_id) else {
						continue;
					};
					let left_gradient = matrix::mul(&output_gradient, right)?;
					let right_gradient = matrix::mul(&output_gradient, left)?;
					accumulate_value_gradient(
						&mut gradients,
						left.value_id(),
						sum_to_shape(&left_gradient, left.shape())?,
					)?;
					accumulate_value_gradient(
						&mut gradients,
						right.value_id(),
						sum_to_shape(&right_gradient, right.shape())?,
					)?;
				}
				Node::Div {
					left,
					right,
					output_id,
				} => {
					let Some(output_gradient) = gradients.remove(output_id) else {
						continue;
					};
					let left_gradient = matrix::div(&output_gradient, right)?;
					let denominator = matrix::mul(right, right)?;
					let right_gradient = matrix::scale(
						&matrix::div(&matrix::mul(&output_gradient, left)?, &denominator)?,
						-1.0,
					)?;
					accumulate_value_gradient(
						&mut gradients,
						left.value_id(),
						sum_to_shape(&left_gradient, left.shape())?,
					)?;
					accumulate_value_gradient(
						&mut gradients,
						right.value_id(),
						sum_to_shape(&right_gradient, right.shape())?,
					)?;
				}
				Node::Scale {
					input,
					output_id,
					scalar,
				} => {
					let Some(output_gradient) = gradients.remove(output_id) else {
						continue;
					};
					accumulate_value_gradient(
						&mut gradients,
						input.value_id(),
						matrix::scale(&output_gradient, *scalar)?,
					)?;
				}
				Node::Reciprocal {
					input,
					output,
					output_id,
				} => {
					let Some(output_gradient) = gradients.remove(output_id) else {
						continue;
					};
					let output_squared = matrix::mul(output, output)?;
					let input_gradient =
						matrix::scale(&matrix::mul(&output_gradient, &output_squared)?, -1.0)?;
					accumulate_value_gradient(&mut gradients, input.value_id(), input_gradient)?;
				}
				Node::Exp {
					input,
					output,
					output_id,
				} => {
					let Some(output_gradient) = gradients.remove(output_id) else {
						continue;
					};
					let input_gradient = matrix::mul(&output_gradient, output)?;
					accumulate_value_gradient(&mut gradients, input.value_id(), input_gradient)?;
				}
				Node::Log { input, output_id } => {
					let Some(output_gradient) = gradients.remove(output_id) else {
						continue;
					};
					let input_gradient = matrix::div(&output_gradient, input)?;
					accumulate_value_gradient(&mut gradients, input.value_id(), input_gradient)?;
				}
				Node::Abs { input, output_id } => {
					let Some(output_gradient) = gradients.remove(output_id) else {
						continue;
					};
					let positive = matrix::equal(&matrix::clamp_max(input, 0.0)?, 0.0)?;
					let negative = matrix::equal(&matrix::clamp_min(input, 0.0)?, 0.0)?;
					let sign = matrix::sub(&positive, &negative)?;
					let input_gradient = matrix::mul(&output_gradient, &sign)?;
					accumulate_value_gradient(&mut gradients, input.value_id(), input_gradient)?;
				}
				Node::Copy { input, output_id } => {
					let Some(output_gradient) = gradients.remove(output_id) else {
						continue;
					};
					accumulate_value_gradient(&mut gradients, input.value_id(), output_gradient)?;
				}
				Node::Sqrt {
					input,
					output,
					output_id,
				} => {
					let Some(output_gradient) = gradients.remove(output_id) else {
						continue;
					};
					let input_gradient = matrix::div(&output_gradient, &matrix::scale(output, 2.0)?)?;
					accumulate_value_gradient(&mut gradients, input.value_id(), input_gradient)?;
				}
				Node::ClampMax {
					input,
					output_id,
					maximum,
				} => {
					let Some(output_gradient) = gradients.remove(output_id) else {
						continue;
					};
					let difference = matrix::sub_scalar(input, *maximum)?;
					let mask = matrix::equal(&matrix::clamp_min(&difference, 0.0)?, 0.0)?;
					let input_gradient = matrix::mul(&output_gradient, &mask)?;
					accumulate_value_gradient(&mut gradients, input.value_id(), input_gradient)?;
				}
				Node::ClampMin {
					input,
					output_id,
					minimum,
				} => {
					let Some(output_gradient) = gradients.remove(output_id) else {
						continue;
					};
					let difference = matrix::sub_scalar(input, *minimum)?;
					let mask = matrix::equal(&matrix::clamp_max(&difference, 0.0)?, 0.0)?;
					let input_gradient = matrix::mul(&output_gradient, &mask)?;
					accumulate_value_gradient(&mut gradients, input.value_id(), input_gradient)?;
				}
				Node::Sub {
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
						sum_to_shape(&output_gradient, left.shape())?,
					)?;
					accumulate_value_gradient(
						&mut gradients,
						right.value_id(),
						sum_to_shape(&matrix::scale(&output_gradient, -1.0)?, right.shape())?,
					)?;
				}
				Node::Slice {
					input,
					output_id,
					dim,
					start,
					end,
				} => {
					let Some(output_gradient) = gradients.remove(output_id) else {
						continue;
					};
					let input_gradient =
						matrix::slice_backward(input.shape(), *dim, *start, *end, &output_gradient)?;
					accumulate_value_gradient(&mut gradients, input.value_id(), input_gradient)?;
				}
				Node::RepeatInterleave {
					input,
					output_id,
					repeats,
					dim,
				} => {
					let Some(output_gradient) = gradients.remove(output_id) else {
						continue;
					};
					let input_gradient =
						matrix::repeat_interleave_backward(input.shape(), *repeats, *dim, &output_gradient)?;
					accumulate_value_gradient(&mut gradients, input.value_id(), input_gradient)?;
				}
				Node::Concat {
					inputs,
					output_id,
					dim,
					sizes,
				} => {
					let Some(output_gradient) = gradients.remove(output_id) else {
						continue;
					};
					let mut start = 0_usize;
					for (input, size) in inputs.iter().zip(sizes) {
						let end = start
							.checked_add(*size)
							.ok_or_else(|| Error::internal("saved Concat interval overflows usize"))?;
						if *size != 0 {
							let dim = i32::try_from(*dim)
								.map_err(|_| Error::internal("saved Concat axis exceeds i32"))?;
							let start_i64 = i64::try_from(start)
								.map_err(|_| Error::internal("saved Concat interval exceeds i64"))?;
							let end_i64 = i64::try_from(end)
								.map_err(|_| Error::internal("saved Concat interval exceeds i64"))?;
							let input_gradient = matrix::slice(&output_gradient, dim, start_i64, end_i64)?;
							accumulate_value_gradient(&mut gradients, input.value_id(), input_gradient)?;
						}
						start = end;
					}
				}
				Node::Transpose {
					input,
					output_id,
					dim0,
					dim1,
				} => {
					let Some(output_gradient) = gradients.remove(output_id) else {
						continue;
					};
					let dim0 = i32::try_from(*dim0)
						.map_err(|_| Error::internal("saved Transpose dim0 exceeds i32"))?;
					let dim1 = i32::try_from(*dim1)
						.map_err(|_| Error::internal("saved Transpose dim1 exceeds i32"))?;
					let input_gradient = matrix::transpose(&output_gradient, dim0, dim1)?;
					accumulate_value_gradient(&mut gradients, input.value_id(), input_gradient)?;
				}
				Node::Gather {
					input,
					indices,
					output_id,
				} => {
					let Some(output_gradient) = gradients.remove(output_id) else {
						continue;
					};
					let input_gradient = matrix::gather_backward(input.shape(), indices, &output_gradient)?;
					accumulate_value_gradient(&mut gradients, input.value_id(), input_gradient)?;
				}
				Node::GatherLastDim {
					input,
					indices,
					output_id,
					input_width,
				} => {
					let Some(output_gradient) = gradients.remove(output_id) else {
						continue;
					};
					let input_gradient =
						matrix::gather_last_dim_backward(&output_gradient, indices, *input_width)?;
					accumulate_value_gradient(&mut gradients, input.value_id(), input_gradient)?;
				}
				Node::FlowLinearState {
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
				Node::FlowLinearVelocity {
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
				Node::FlowEulerStep {
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
				Node::FlowMaskedMse {
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
				Node::MatMulNt {
					left,
					right,
					output_id,
				} => {
					let Some(output_gradient) = gradients.remove(output_id) else {
						continue;
					};
					let [rows, inner] = left.shape() else {
						return Err(Error::internal("saved MatMulNt left rank changed"));
					};
					let [columns, right_inner] = right.shape() else {
						return Err(Error::internal("saved MatMulNt right rank changed"));
					};
					if inner != right_inner {
						return Err(Error::internal("saved MatMulNt inner extent changed"));
					}
					let output_gradient = output_gradient.reshape([1, *rows, *columns])?;
					let left_batched = left.reshape([1, *rows, *inner])?;
					let right_batched = right.reshape([1, *columns, *inner])?;
					let left_gradient = crate::ml::matrix::bmm(&output_gradient, &right_batched)?
						.reshape(left.shape().to_vec())?;
					let right_gradient = crate::ml::matrix::bmm_tn(&output_gradient, &left_batched)?
						.reshape(right.shape().to_vec())?;
					accumulate_value_gradient(&mut gradients, left.value_id(), left_gradient)?;
					accumulate_value_gradient(&mut gradients, right.value_id(), right_gradient)?;
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
					let input_gradient = matrix::dropout_backward(&output_gradient, *probability, *seed)?;
					accumulate_value_gradient(&mut gradients, input.value_id(), input_gradient)?;
				}
				Node::CrossEntropy {
					logits,
					targets,
					output_id,
				} => {
					let Some(output_gradient) = gradients.remove(output_id) else {
						continue;
					};
					let gradient = matrix::mul(
						&loss::cross_entropy_backward(logits, targets)?,
						&output_gradient,
					)?;
					accumulate_value_gradient(&mut gradients, logits.value_id(), gradient)?;
				}
				Node::MaskedCrossEntropy {
					logits,
					targets,
					mask,
					valid_count,
					output_id,
				} => {
					let Some(output_gradient) = gradients.remove(output_id) else {
						continue;
					};
					let gradient = matrix::mul(
						&loss::masked_cross_entropy_backward(logits, targets, mask, *valid_count)?,
						&output_gradient,
					)?;
					accumulate_value_gradient(&mut gradients, logits.value_id(), gradient)?;
				}
				Node::SmoothL1 {
					prediction,
					target,
					output_id,
				} => {
					let Some(output_gradient) = gradients.remove(output_id) else {
						continue;
					};
					let gradient = matrix::mul(
						&loss::smooth_l1_backward(prediction, target)?,
						&output_gradient,
					)?;
					accumulate_value_gradient(&mut gradients, prediction.value_id(), gradient)?;
				}
				Node::Mse {
					prediction,
					target,
					output_id,
				} => {
					let Some(output_gradient) = gradients.remove(output_id) else {
						continue;
					};
					let gradient = matrix::mul(&loss::mse_backward(prediction, target)?, &output_gradient)?;
					accumulate_value_gradient(&mut gradients, prediction.value_id(), gradient)?;
				}
				Node::L1 {
					prediction,
					target,
					output_id,
				} => {
					let Some(output_gradient) = gradients.remove(output_id) else {
						continue;
					};
					let gradient = matrix::mul(&loss::l1_backward(prediction, target)?, &output_gradient)?;
					accumulate_value_gradient(&mut gradients, prediction.value_id(), gradient)?;
				}
				Node::Bce {
					prediction,
					target,
					output_id,
				} => {
					let Some(output_gradient) = gradients.remove(output_id) else {
						continue;
					};
					let gradient = matrix::mul(&loss::bce_backward(prediction, target)?, &output_gradient)?;
					accumulate_value_gradient(&mut gradients, prediction.value_id(), gradient)?;
				}
				Node::PpoClippedPolicy {
					new_log_probability,
					old_log_probability,
					advantage,
					clip_epsilon,
					output_id,
				} => {
					let Some(output_gradient) = gradients.remove(output_id) else {
						continue;
					};
					let gradient = matrix::mul(
						&loss::ppo_clipped_policy_backward(
							new_log_probability,
							old_log_probability,
							advantage,
							*clip_epsilon,
						)?,
						&output_gradient,
					)?;
					accumulate_value_gradient(&mut gradients, new_log_probability.value_id(), gradient)?;
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
					accumulate_value_gradient(&mut gradients, bias_value.value_id(), bias_gradient.clone())?;
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
					accumulate_value_gradient(&mut gradients, bias_value.value_id(), bias_gradient.clone())?;
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
					accumulate_value_gradient(&mut gradients, bias_value.value_id(), bias_gradient.clone())?;
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
					let weight_gradient =
						matrix_lowering::embedding_backward(indices, &output_gradient, weight_value)?;
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
					accumulate_value_gradient(&mut gradients, bias_value.value_id(), bias_gradient.clone())?;
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
					let (input_gradient, weight_gradient) =
						matrix_lowering::rms_norm_backward(input, weight_value, &output_gradient, *epsilon)?;
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
				Node::RmsNormGated(node) => {
					let Some(output_gradient) = gradients.remove(&node.output_id) else {
						continue;
					};
					let result = matrix_lowering::rms_norm_gated_backward(
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
				Node::ChannelNorm(node) => {
					let Some(output_gradient) = gradients.remove(&node.output_id) else {
						continue;
					};
					let result = matrix_lowering::channel_norm_backward(
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
					let input_gradient = matrix_lowering::sigmoid_backward(output, &output_gradient)?;
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
					let input_gradient = matrix_lowering::elu_backward(output, &output_gradient, *alpha)?;
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
					let input_gradient = matrix_lowering::softplus_backward(output, &output_gradient)?;
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
					let input_gradient = matrix::log_softmax_backward(output, &output_gradient, *dim)?;
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
					let input_gradient =
						crate::ml::matrix::upsample_2d_backward(input, &output_gradient, *scale_factor, *mode)?;
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
				Node::SiluMul {
					input,
					intermediate_size,
					output_id,
				} => {
					let Some(output_gradient) = gradients.remove(output_id) else {
						continue;
					};
					let input_gradient =
						crate::ml::matrix::silu_mul_backward(input, &output_gradient, *intermediate_size)?;
					accumulate_value_gradient(&mut gradients, input.value_id(), input_gradient)?;
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
					let input_gradient =
						attention_lowering::softmax_scaled_masked_backward(output, &output_gradient, *scale)?;
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
					let rows = batch_heads
						.checked_mul(*sequence_length)
						.ok_or_else(|| Error::internal("saved SDPA probability rows overflow usize"))?;
					let score_gradient = attention_lowering::softmax_scaled_masked_backward(
						&probabilities.reshape([rows, *sequence_length])?,
						&probability_gradient.reshape([rows, *sequence_length])?,
						*scale,
					)?
					.reshape([*batch_heads, *sequence_length, *sequence_length])?;
					let query_gradient = attention_lowering::bmm(&score_gradient, key)?;
					let key_gradient = attention_lowering::bmm_tn(&score_gradient, query)?;
					let value_gradient = attention_lowering::bmm_tn(probabilities, &output_gradient)?;
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
				Node::MoeGather {
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
				Node::MoeCombine {
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
				Node::GroupedGemmM {
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
				Node::GroupedLinearM {
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
				Node::Mamba3Siso(node) => {
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
				Node::Mamba3Mimo(node) => {
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
				Node::Mamba3Preprocess(node) => {
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
				Node::Reshape { input, output_id } => {
					let Some(output_gradient) = gradients.remove(output_id) else {
						continue;
					};
					let input_gradient = output_gradient.reshape(input.shape().to_vec())?;
					accumulate_value_gradient(&mut gradients, input.value_id(), input_gradient)?;
				}
			}
			if let Some((forward, generation)) = attachment {
				engine.complete_semantic_autograd(forward, entry.sequence, generation, backward_first)?;
			}
		}
		for watched in self.inner.watched_parameters.borrow().iter() {
			if let Some(gradient) = gradients.remove(&watched.data_id) {
				watched.parameter.accumulate_gradient(gradient)?;
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

pub(in crate::ml) fn record_parameter_leaf(parameter: &Parameter) -> Result<()> {
	let (data, version, requires_grad) = parameter.snapshot();
	if !requires_grad {
		return Ok(());
	}
	ACTIVE_TAPES.with(|tapes| {
		let active = tapes.borrow().last().and_then(Weak::upgrade);
		if let Some(active) = active {
			let mut watched = active.watched_parameters.borrow_mut();
			if !watched
				.iter()
				.any(|candidate| candidate.parameter.same_as(parameter))
			{
				watched.push(WatchedParameter {
					parameter: parameter.clone(),
					data_id: data.value_id(),
					version,
				});
			}
		}
		Ok(())
	})
}

pub(in crate::ml) fn recording_active() -> bool {
	ACTIVE_TAPES.with(|tapes| {
		tapes
			.borrow()
			.last()
			.is_some_and(|candidate| candidate.strong_count() != 0)
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

fn reached_output_id(node: &Node, gradients: &HashMap<u64, Matrix>) -> Option<u64> {
	match node {
		Node::Mamba3Preprocess(node) => node
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

fn sum_to_shape(gradient: &Matrix, target_shape: &[usize]) -> Result<Matrix> {
	if gradient.shape() == target_shape {
		return Ok(gradient.clone());
	}
	if gradient.shape().len() < target_shape.len() {
		return Err(Error::internal(
			"broadcast adjoint target rank exceeds output-gradient rank",
		));
	}
	let gradient_shape = gradient.shape().to_vec();
	let leading = gradient_shape.len() - target_shape.len();
	let mut result = gradient.clone();
	for (axis, extent) in gradient_shape.iter().copied().enumerate() {
		let target_extent = axis
			.checked_sub(leading)
			.map_or(1, |target_axis| target_shape[target_axis]);
		if target_extent == 1 && extent > 1 {
			result = matrix::sum(
				&result,
				i32::try_from(axis).map_err(|_| Error::internal("broadcast adjoint axis exceeds i32"))?,
			)?;
		}
	}
	if result.shape() != target_shape {
		result = result.reshape(target_shape.to_vec())?;
	}
	Ok(result)
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
