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

use super::{Parameter, kernels};

thread_local! {
	static ACTIVE_TAPES: RefCell<Vec<Weak<TapeInner>>> = const { RefCell::new(Vec::new()) };
	static NEXT_SEQUENCE: Cell<u64> = const { Cell::new(0) };
}

struct TapeEntry {
	sequence: u64,
	node: Node,
}

enum Node {
	Add {
		left: Matrix,
		right: Matrix,
		output_id: u64,
	},
	Reshape {
		input: Matrix,
		output_id: u64,
	},
	Linear {
		input: Matrix,
		output_id: u64,
		weight: Parameter,
		weight_value: Matrix,
		weight_version: u64,
		bias: Parameter,
		bias_version: u64,
	},
	LayerNorm {
		input: Matrix,
		normalized: Matrix,
		inverse_stddev: Matrix,
		output_id: u64,
		weight: Parameter,
		weight_value: Matrix,
		weight_version: u64,
		bias: Parameter,
		bias_version: u64,
	},
	Gelu {
		input: Matrix,
		output_id: u64,
	},
	Swiglu {
		gate: Matrix,
		up: Matrix,
		output_id: u64,
	},
	Attention {
		query: Matrix,
		key: Matrix,
		value: Matrix,
		probabilities: Matrix,
		output_id: u64,
		sequence_length: usize,
		num_heads: usize,
	},
	Embedding {
		indices: Matrix,
		output_id: u64,
		weight: Parameter,
		weight_value: Matrix,
		weight_version: u64,
	},
	Rnn(Box<RnnNode>),
	CrossEntropy {
		logits: Matrix,
		targets: Matrix,
		output_id: u64,
	},
}

impl Node {
	const fn output_id(&self) -> u64 {
		match self {
			Self::Add { output_id, .. }
			| Self::Reshape { output_id, .. }
			| Self::Linear { output_id, .. }
			| Self::LayerNorm { output_id, .. }
			| Self::Gelu { output_id, .. }
			| Self::Swiglu { output_id, .. }
			| Self::Attention { output_id, .. }
			| Self::Embedding { output_id, .. }
			| Self::CrossEntropy { output_id, .. } => *output_id,
			Self::Rnn(node) => node.output_id,
		}
	}
}

struct RnnNode {
	input: Matrix,
	output: Matrix,
	hidden_previous: Matrix,
	output_id: u64,
	parameters: [Parameter; 4],
	weights: [Matrix; 2],
	versions: [u64; 4],
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
	/// The current checkpoint supports a mean cross-entropy root and any recorded
	/// chain of admitted Linear, Embedding, and reshape operations leading into
	/// it. Gradients accumulate on stable [`Parameter`] handles. This method
	/// records but does not submit or wait.
	///
	/// # Errors
	///
	/// Returns an error when the tape was already consumed, `root` is not its
	/// scalar cross-entropy result, a saved parameter changed before backward, or
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
		let root_found = nodes.iter().any(
			|entry| matches!(&entry.node, Node::CrossEntropy { output_id, .. } if *output_id == root.value_id()),
		);
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
					bias.validate_version(*bias_version)?;
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
				Node::Rnn(node) => {
					for (parameter, version) in node.parameters.iter().zip(&node.versions) {
						parameter.validate_version(*version)?;
					}
				}
				Node::Add { .. }
				| Node::Reshape { .. }
				| Node::Gelu { .. }
				| Node::Swiglu { .. }
				| Node::Attention { .. }
				| Node::CrossEntropy { .. } => {}
			}
		}
		self.inner.consumed.set(true);

		let mut gradients = HashMap::<u64, Matrix>::new();
		let engine = root.engine_handle().clone();
		for entry in nodes.iter().rev() {
			let node = &entry.node;
			let reaches_root = matches!(
				node,
				Node::CrossEntropy { output_id, .. } if *output_id == root.value_id()
			) || gradients.contains_key(&node.output_id());
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
				Node::CrossEntropy {
					logits,
					targets,
					output_id,
				} if *output_id == root.value_id() => {
					let gradient = kernels::cross_entropy_backward(logits, targets)?;
					accumulate_value_gradient(&mut gradients, logits.value_id(), gradient)?;
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
						kernels::linear_backward(input, weight_value, &output_gradient)?;
					accumulate_value_gradient(&mut gradients, input.value_id(), input_gradient)?;
					weight.accumulate_gradient(weight_gradient)?;
					bias.accumulate_gradient(bias_gradient)?;
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
						kernels::embedding_backward(indices, &output_gradient, weight_value)?;
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
						kernels::layer_norm_backward(
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
				Node::Gelu { input, output_id } => {
					let Some(output_gradient) = gradients.remove(output_id) else {
						continue;
					};
					let input_gradient = kernels::gelu_backward(input, &output_gradient)?;
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
						kernels::swiglu_backward(gate, up, &output_gradient)?;
					accumulate_value_gradient(&mut gradients, gate.value_id(), gate_gradient)?;
					accumulate_value_gradient(&mut gradients, up.value_id(), up_gradient)?;
				}
				Node::Attention {
					query,
					key,
					value,
					probabilities,
					output_id,
					sequence_length,
					num_heads,
				} => {
					let Some(output_gradient) = gradients.remove(output_id) else {
						continue;
					};
					let (query_gradient, key_gradient, value_gradient) =
						kernels::scaled_dot_product_attention_causal_backward(
							query,
							key,
							value,
							probabilities,
							&output_gradient,
							*sequence_length,
							*num_heads,
						)?;
					accumulate_value_gradient(&mut gradients, query.value_id(), query_gradient)?;
					accumulate_value_gradient(&mut gradients, key.value_id(), key_gradient)?;
					accumulate_value_gradient(&mut gradients, value.value_id(), value_gradient)?;
				}
				Node::Rnn(node) => {
					let Some(output_gradient) = gradients.remove(&node.output_id) else {
						continue;
					};
					let result = kernels::rnn_backward(
						&output_gradient,
						&node.input,
						&node.output,
						&node.hidden_previous,
						&node.weights[0],
						&node.weights[1],
					)?;
					accumulate_value_gradient(&mut gradients, node.input.value_id(), result.input)?;
					for (parameter, gradient) in node.parameters.iter().zip([
						result.weight_ih,
						result.weight_hh,
						result.bias_ih,
						result.bias_hh,
					]) {
						parameter.accumulate_gradient(gradient)?;
					}
				}
				Node::Reshape { input, output_id } => {
					let Some(output_gradient) = gradients.remove(output_id) else {
						continue;
					};
					let input_gradient = output_gradient.reshape(input.shape().to_vec())?;
					accumulate_value_gradient(&mut gradients, input.value_id(), input_gradient)?;
				}
				Node::CrossEntropy { .. } => {}
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

pub(super) fn record_rnn(
	input: &Matrix,
	output: &Matrix,
	hidden_previous: Matrix,
	parameters: [Parameter; 4],
	weights: [Matrix; 2],
	versions: [u64; 4],
) -> Result<()> {
	record_node(Node::Rnn(Box::new(RnnNode {
		input: input.clone(),
		output: output.clone(),
		hidden_previous,
		output_id: output.value_id(),
		parameters,
		weights,
		versions,
	})))
}

pub(super) fn record_embedding(
	indices: &Matrix,
	output: &Matrix,
	weight: Parameter,
	weight_value: Matrix,
	weight_version: u64,
) -> Result<()> {
	record_node(Node::Embedding {
		indices: indices.clone(),
		output_id: output.value_id(),
		weight,
		weight_value,
		weight_version,
	})
}

#[allow(clippy::too_many_arguments)]
pub(super) fn record_layer_norm(
	input: &Matrix,
	output: &Matrix,
	normalized: Matrix,
	inverse_stddev: Matrix,
	weight: Parameter,
	weight_value: Matrix,
	weight_version: u64,
	bias: Parameter,
	bias_version: u64,
) -> Result<()> {
	record_node(Node::LayerNorm {
		input: input.clone(),
		normalized,
		inverse_stddev,
		output_id: output.value_id(),
		weight,
		weight_value,
		weight_version,
		bias,
		bias_version,
	})
}

pub(super) fn record_gelu(input: &Matrix, output: &Matrix) -> Result<()> {
	record_node(Node::Gelu {
		input: input.clone(),
		output_id: output.value_id(),
	})
}

pub(super) fn record_swiglu(gate: &Matrix, up: &Matrix, output: &Matrix) -> Result<()> {
	record_node(Node::Swiglu {
		gate: gate.clone(),
		up: up.clone(),
		output_id: output.value_id(),
	})
}

pub(super) fn record_attention(
	query: &Matrix,
	key: &Matrix,
	value: &Matrix,
	output: &Matrix,
	probabilities: Matrix,
	sequence_length: usize,
	num_heads: usize,
) -> Result<()> {
	record_node(Node::Attention {
		query: query.clone(),
		key: key.clone(),
		value: value.clone(),
		probabilities,
		output_id: output.value_id(),
		sequence_length,
		num_heads,
	})
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

pub(super) fn record_linear(
	input: &Matrix,
	output: &Matrix,
	weight: Parameter,
	weight_value: Matrix,
	weight_version: u64,
	bias: Parameter,
	bias_version: u64,
) -> Result<()> {
	record_node(Node::Linear {
		input: input.clone(),
		output_id: output.value_id(),
		weight,
		weight_value,
		weight_version,
		bias,
		bias_version,
	})
}

pub(super) fn record_cross_entropy(
	logits: &Matrix,
	targets: &Matrix,
	output: &Matrix,
) -> Result<()> {
	record_node(Node::CrossEntropy {
		logits: logits.clone(),
		targets: targets.clone(),
		output_id: output.value_id(),
	})
}

fn record_node(node: Node) -> Result<()> {
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
