use std::{
	cell::{Cell, RefCell},
	rc::Rc,
};

use crate::{DType, Engine, Error, Matrix, Result, matrix as core_matrix};

use super::super::{Module, ModuleRegistry, NamedBuffer, Parameter, autograd, matrix, random};
use super::{Linear, RmsNorm, Swiglu};

/// Reduced host telemetry for the latest MoE routing decision.
#[derive(Clone, Debug, PartialEq)]
pub struct MoeRouteStats {
	load_fraction: Vec<f32>,
	mean_probability: Vec<f32>,
	entropy: f32,
	max_load_ratio: f32,
	dead_experts: usize,
}

impl MoeRouteStats {
	/// Return the fraction of token-slot assignments received by each expert.
	pub fn load_fraction(&self) -> &[f32] {
		&self.load_fraction
	}

	/// Return the mean unbiased router probability for each expert.
	pub fn mean_probability(&self) -> &[f32] {
		&self.mean_probability
	}

	/// Return normalized load entropy in `[0, 1]` (`1` is balanced).
	pub const fn entropy(&self) -> f32 {
		self.entropy
	}

	/// Return expert count times the maximum load fraction (`1` is balanced).
	pub const fn max_load_ratio(&self) -> f32 {
		self.max_load_ratio
	}

	/// Return the number of experts that received no token-slot assignments.
	pub const fn dead_experts(&self) -> usize {
		self.dead_experts
	}
}

/// Sparse top-k mixture-of-experts feed-forward module.
///
/// The implementation preserves OA's expert-major route plan and stacked
/// parameter layout. Routing decisions remain detached while gradients flow
/// through selected gate magnitudes, expert projections, RMSNorm, and the
/// residual input.
pub struct Moe {
	model_width: usize,
	hidden_width: usize,
	num_experts: usize,
	experts_per_token: usize,
	epsilon: f32,
	norm: Rc<RmsNorm>,
	router: Rc<Linear>,
	expert_gate_up_weight: Parameter,
	expert_gate_up_bias: Parameter,
	expert_down_weight: Parameter,
	expert_down_bias: Parameter,
	shared_experts: Vec<Rc<Swiglu>>,
	routing_bias: NamedBuffer,
	sparse_execution: Cell<bool>,
	balance_rate: Cell<f32>,
	aux_loss_alpha: Cell<f32>,
	router_z_loss_beta: Cell<f32>,
	last_aux_loss: RefCell<Matrix>,
	zero_aux_loss: Matrix,
	last_selection_mask: RefCell<Option<Matrix>>,
	last_gate_probabilities: RefCell<Option<Matrix>>,
	registry: ModuleRegistry,
}

impl Moe {
	/// Construct a deterministically initialized sparse MoE.
	///
	/// `experts_per_token` is clamped to `1..=num_experts`, matching OA C++.
	/// The current GPU route planner admits at most 256 experts.
	///
	/// # Errors
	///
	/// Returns an error for zero dimensions, more than 256 experts, invalid
	/// epsilon, shape overflow, allocation failure, or registration failure.
	pub fn with_seed(
		engine: &Engine,
		model_width: usize,
		hidden_width: usize,
		num_experts: usize,
		experts_per_token: usize,
		epsilon: f32,
		seed: u64,
	) -> Result<Self> {
		Self::with_seed_and_shared_experts(
			engine,
			model_width,
			hidden_width,
			num_experts,
			experts_per_token,
			epsilon,
			0,
			seed,
		)
	}

	/// Construct a deterministic sparse MoE with always-on shared SwiGLU experts.
	///
	/// Shared experts consume the same normalized input as routed experts and add
	/// their complete ungated deltas before the residual connection.
	///
	/// # Errors
	///
	/// Returns the same errors as [`Self::with_seed`], plus shared-expert
	/// allocation or registration failures.
	#[allow(
		clippy::too_many_arguments,
		reason = "the constructor preserves the donor MoE dimensions, policy, and seed"
	)]
	pub fn with_seed_and_shared_experts(
		engine: &Engine,
		model_width: usize,
		hidden_width: usize,
		num_experts: usize,
		experts_per_token: usize,
		epsilon: f32,
		num_shared_experts: usize,
		seed: u64,
	) -> Result<Self> {
		if model_width == 0 || hidden_width == 0 || num_experts == 0 || num_experts > 256 {
			return Err(Error::invalid_argument(
				"MoE dimensions must be nonzero and num_experts must not exceed 256",
			));
		}
		let gate_up_width = hidden_width
			.checked_mul(2)
			.ok_or_else(|| Error::invalid_argument("MoE gate/up width overflows usize"))?;
		let gate_up_count = num_experts
			.checked_mul(gate_up_width)
			.and_then(|value| value.checked_mul(model_width))
			.ok_or_else(|| Error::invalid_argument("MoE gate/up weight size overflows usize"))?;
		let gate_up_bias_count = num_experts
			.checked_mul(gate_up_width)
			.ok_or_else(|| Error::invalid_argument("MoE gate/up bias size overflows usize"))?;
		let down_count = num_experts
			.checked_mul(model_width)
			.and_then(|value| value.checked_mul(hidden_width))
			.ok_or_else(|| Error::invalid_argument("MoE down weight size overflows usize"))?;
		let down_bias_count = num_experts
			.checked_mul(model_width)
			.ok_or_else(|| Error::invalid_argument("MoE down bias size overflows usize"))?;
		let gate_limit = (1.0_f32 / model_width as f32).sqrt();
		let down_limit = (1.0_f32 / hidden_width as f32).sqrt();
		let norm = Rc::new(RmsNorm::new(engine, model_width, epsilon)?);
		let router = Rc::new(Linear::with_seed(engine, model_width, num_experts, seed)?);
		let gate_up_weight = Matrix::from_f32(
			engine,
			[num_experts, gate_up_width, model_width],
			&random::symmetric_uniform(gate_up_count, gate_limit, seed.wrapping_add(1)),
		)?;
		let gate_up_bias = Matrix::from_f32(
			engine,
			[num_experts, gate_up_width],
			&vec![0.0; gate_up_bias_count],
		)?;
		let down_weight = Matrix::from_f32(
			engine,
			[num_experts, model_width, hidden_width],
			&random::symmetric_uniform(down_count, down_limit, seed.wrapping_add(2)),
		)?;
		let down_bias = Matrix::from_f32(
			engine,
			[num_experts, model_width],
			&vec![0.0; down_bias_count],
		)?;
		let mut shared_experts = Vec::with_capacity(num_shared_experts);
		for index in 0..num_shared_experts {
			let index = u64::try_from(index)
				.map_err(|_| Error::resource_exhausted("shared expert index exceeds u64"))?;
			shared_experts.push(Rc::new(Swiglu::with_seed(
				engine,
				model_width,
				hidden_width,
				true,
				seed.wrapping_add(3).wrapping_add(index.wrapping_mul(3)),
			)?));
		}
		Self::from_parts(
			norm,
			router,
			gate_up_weight,
			gate_up_bias,
			down_weight,
			down_bias,
			experts_per_token,
			shared_experts,
		)
	}

	/// Construct a sparse MoE from exact donor-layout FP32 matrices.
	///
	/// # Errors
	///
	/// Returns an error unless the matrices form `norm[D]`, router weight
	/// `[E,D]`, router bias `[E]`, gate/up weight `[E,2H,D]`, gate/up bias
	/// `[E,2H]`, down weight `[E,D,H]`, and down bias `[E,D]` on one engine.
	#[allow(
		clippy::too_many_arguments,
		reason = "the constructor exposes the exact seven-matrix donor checkpoint layout"
	)]
	pub fn from_matrices(
		norm_weight: Matrix,
		router_weight: Matrix,
		router_bias: Matrix,
		gate_up_weight: Matrix,
		gate_up_bias: Matrix,
		down_weight: Matrix,
		down_bias: Matrix,
		experts_per_token: usize,
		epsilon: f32,
	) -> Result<Self> {
		let norm = Rc::new(RmsNorm::from_matrix(norm_weight, epsilon)?);
		let router = Rc::new(Linear::from_matrices(router_weight, router_bias)?);
		Self::from_parts(
			norm,
			router,
			gate_up_weight,
			gate_up_bias,
			down_weight,
			down_bias,
			experts_per_token,
			Vec::new(),
		)
	}

	#[allow(
		clippy::too_many_arguments,
		reason = "the owner retains the exact stacked donor parameter set"
	)]
	fn from_parts(
		norm: Rc<RmsNorm>,
		router: Rc<Linear>,
		gate_up_weight: Matrix,
		gate_up_bias: Matrix,
		down_weight: Matrix,
		down_bias: Matrix,
		experts_per_token: usize,
		shared_experts: Vec<Rc<Swiglu>>,
	) -> Result<Self> {
		let model_width = norm.dimension();
		let num_experts = router.output_features();
		let norm_value = norm.weight().data();
		let router_weight = router.weight().data();
		let router_bias = router
			.bias()
			.expect("MoE router is constructed with a trainable bias")
			.data();
		let [gate_experts, gate_up_width, gate_input] = gate_up_weight.shape() else {
			return Err(Error::invalid_argument(
				"MoE gate/up weight must have shape [E, 2H, D]",
			));
		};
		if gate_up_width % 2 != 0 {
			return Err(Error::invalid_argument(
				"MoE gate/up output width must be even",
			));
		}
		let hidden_width = gate_up_width / 2;
		if num_experts == 0
			|| num_experts > 256
			|| router.input_features() != model_width
			|| *gate_experts != num_experts
			|| *gate_input != model_width
			|| hidden_width == 0
			|| gate_up_bias.shape() != [num_experts, *gate_up_width]
			|| down_weight.shape() != [num_experts, model_width, hidden_width]
			|| down_bias.shape() != [num_experts, model_width]
			|| [
				&router_weight,
				&router_bias,
				&gate_up_weight,
				&gate_up_bias,
				&down_weight,
				&down_bias,
			]
			.iter()
			.any(|matrix| {
				matrix.dtype() != DType::F32
					|| !norm_value.engine_handle().same_as(matrix.engine_handle())
			}) {
			return Err(Error::invalid_argument(
				"MoE matrices do not match the donor stacked expert layout on one engine",
			));
		}
		let experts_per_token = experts_per_token.clamp(1, num_experts);
		let expert_gate_up_weight = Parameter::new("expert_gate_up_weight", gate_up_weight)?;
		let expert_gate_up_bias = Parameter::new("expert_gate_up_bias", gate_up_bias)?;
		let expert_down_weight = Parameter::new("expert_down_weight", down_weight)?;
		let expert_down_bias = Parameter::new("expert_down_bias", down_bias)?;
		let mut registry = ModuleRegistry::new();
		registry.register_parameter("expert_gate_up_weight", expert_gate_up_weight.clone())?;
		registry.register_parameter("expert_gate_up_bias", expert_gate_up_bias.clone())?;
		registry.register_parameter("expert_down_weight", expert_down_weight.clone())?;
		registry.register_parameter("expert_down_bias", expert_down_bias.clone())?;
		registry.register_module("norm", norm.clone())?;
		registry.register_module("router", router.clone())?;
		for (index, expert) in shared_experts.iter().enumerate() {
			if expert.input_features() != model_width || expert.intermediate_size() != hidden_width
			{
				return Err(Error::invalid_argument(
					"shared MoE expert dimensions do not match the routed experts",
				));
			}
			for parameter in expert.all_parameters()? {
				if !norm_value
					.engine_handle()
					.same_as(parameter.data().engine_handle())
				{
					return Err(Error::invalid_argument(
						"shared MoE experts must belong to the routed expert engine",
					));
				}
			}
			registry.register_module(format!("shared_expert_{index}"), expert.clone())?;
		}
		registry.register_buffer(
			"routing_bias",
			Matrix::from_slice_handle(
				norm_value.engine_handle(),
				vec![1, num_experts],
				&vec![0.0; num_experts],
			)?,
			true,
		)?;
		let routing_bias = registry
			.buffer_handle("routing_bias")
			.ok_or_else(|| Error::internal("MoE routing-bias registration was lost"))?;
		let zero_aux_loss =
			Matrix::from_slice_handle(norm_value.engine_handle(), vec![], &[0.0_f32])?;
		Ok(Self {
			model_width,
			hidden_width,
			num_experts,
			experts_per_token,
			epsilon: norm.epsilon(),
			norm,
			router,
			expert_gate_up_weight,
			expert_gate_up_bias,
			expert_down_weight,
			expert_down_bias,
			shared_experts,
			routing_bias,
			sparse_execution: Cell::new(true),
			balance_rate: Cell::new(0.0),
			aux_loss_alpha: Cell::new(0.0),
			router_z_loss_beta: Cell::new(0.0),
			last_aux_loss: RefCell::new(zero_aux_loss.clone()),
			zero_aux_loss,
			last_selection_mask: RefCell::new(None),
			last_gate_probabilities: RefCell::new(None),
			registry,
		})
	}

	/// Apply top-k routed experts and add their delta to the input.
	///
	/// # Errors
	///
	/// Returns an error unless `input` is nonempty F32 `[T,D]` on this module's
	/// engine, or a routing, projection, activation, or recording operation fails.
	pub fn forward(&self, input: &Matrix) -> Result<Matrix> {
		let [tokens, width] = input.shape() else {
			return Err(Error::invalid_argument("MoE input must have shape [T, D]"));
		};
		if *tokens == 0 || *width != self.model_width || input.dtype() != DType::F32 {
			return Err(Error::invalid_argument(
				"MoE input must be nonempty F32 [T, model_width]",
			));
		}
		let normalized = self.norm.forward(input)?;
		let logits = self.router.forward(&normalized)?;
		let probabilities = core_matrix::softmax(&logits, 1)?;
		let selection_logits = if self.balance_rate.get() > 0.0 {
			core_matrix::add(&logits, &self.routing_bias.data())?
		} else {
			logits.clone()
		};
		let top_k = core_matrix::top_k(&selection_logits, self.experts_per_token as i32, 1)?;
		let selection_mask = core_matrix::top_k_mask(&top_k.indices, self.num_experts)?;
		let mut delta = if self.sparse_execution.get() {
			let route_gate = matrix::moe_route_weights(&probabilities, &top_k.indices)?;
			let plan = core_matrix::moe_expert_plan(&top_k.indices, self.num_experts)?;
			let packed = matrix::moe_gather(&normalized, &plan.packed_token, &plan.inverse)?;
			let gate_up = self.grouped_projection(
				&packed,
				&self.expert_gate_up_weight,
				&self.expert_gate_up_bias,
				&plan.offsets,
			)?;
			let hidden = matrix::silu_mul(&gate_up, self.hidden_width)?;
			let packed_output = self.grouped_projection(
				&hidden,
				&self.expert_down_weight,
				&self.expert_down_bias,
				&plan.offsets,
			)?;
			matrix::moe_combine(
				&packed_output,
				&route_gate,
				&plan.inverse,
				&plan.packed_slot,
			)?
		} else {
			let gate_unnormalized = core_matrix::mul(&probabilities, &selection_mask)?;
			let denominator = core_matrix::sum(&gate_unnormalized, 1)?;
			let gate =
				core_matrix::mul(&gate_unnormalized, &core_matrix::reciprocal(&denominator)?)?;
			self.dense_expert_delta(&normalized, &gate)?
		};
		for shared_expert in &self.shared_experts {
			delta = core_matrix::add(&delta, &shared_expert.forward(&normalized)?)?;
		}
		let mut aux_loss = None;
		if self.aux_loss_alpha.get() > 0.0 {
			let reciprocal_tokens = 1.0 / *tokens as f32;
			let load_fraction =
				core_matrix::scale(&core_matrix::sum(&selection_mask, 0)?, reciprocal_tokens)?;
			let mean_probability =
				core_matrix::scale(&core_matrix::sum(&probabilities, 0)?, reciprocal_tokens)?;
			let per_expert = core_matrix::mul(&load_fraction, &mean_probability)?;
			let total = core_matrix::sum(&per_expert, 1)?;
			aux_loss = Some(core_matrix::scale(
				&total,
				self.aux_loss_alpha.get() * self.num_experts as f32 / self.experts_per_token as f32,
			)?);
		}
		if self.router_z_loss_beta.get() > 0.0 {
			let log_probabilities = core_matrix::log_softmax(&logits, 1)?;
			let first_logit = core_matrix::slice(&logits, 1, 0, 1)?;
			let first_log_probability = core_matrix::slice(&log_probabilities, 1, 0, 1)?;
			let log_sum_exp = core_matrix::sub(&first_logit, &first_log_probability)?;
			let squared = core_matrix::mul(&log_sum_exp, &log_sum_exp)?;
			let mean = core_matrix::scale(&core_matrix::sum(&squared, 0)?, 1.0 / *tokens as f32)?;
			let z_loss = core_matrix::scale(&mean, self.router_z_loss_beta.get())?;
			aux_loss = Some(if let Some(switch_loss) = aux_loss {
				core_matrix::add(&switch_loss, &z_loss)?
			} else {
				z_loss
			});
		}
		let aux_loss = if let Some(aux_loss) = aux_loss {
			core_matrix::reshape(&aux_loss, Vec::new())?
		} else {
			self.zero_aux_loss.clone()
		};
		*self.last_selection_mask.borrow_mut() = Some(selection_mask);
		*self.last_gate_probabilities.borrow_mut() = Some(probabilities);
		*self.last_aux_loss.borrow_mut() = aux_loss;
		core_matrix::add(input, &delta)
	}

	fn dense_expert_delta(&self, input: &Matrix, gate: &Matrix) -> Result<Matrix> {
		for parameter in [
			&self.expert_gate_up_weight,
			&self.expert_gate_up_bias,
			&self.expert_down_weight,
			&self.expert_down_bias,
		] {
			autograd::record_parameter_leaf(parameter)?;
		}
		let gate_up_weight = self.expert_gate_up_weight.data();
		let gate_up_bias = self.expert_gate_up_bias.data();
		let down_weight = self.expert_down_weight.data();
		let down_bias = self.expert_down_bias.data();
		let mut combined = None;
		for expert in 0..self.num_experts {
			let start = i64::try_from(expert)
				.map_err(|_| Error::resource_exhausted("MoE expert index exceeds i64"))?;
			let end = start + 1;
			let expert_gate_up_weight = core_matrix::slice(&gate_up_weight, 0, start, end)?
				.reshape([2 * self.hidden_width, self.model_width])?;
			let expert_gate_up_bias = core_matrix::slice(&gate_up_bias, 0, start, end)?
				.reshape([2 * self.hidden_width])?;
			let expert_down_weight = core_matrix::slice(&down_weight, 0, start, end)?
				.reshape([self.model_width, self.hidden_width])?;
			let expert_down_bias =
				core_matrix::slice(&down_bias, 0, start, end)?.reshape([self.model_width])?;
			let gate_up = core_matrix::add(
				&core_matrix::mat_mul_nt(input, &expert_gate_up_weight)?,
				&expert_gate_up_bias,
			)?;
			let hidden = matrix::silu_mul(&gate_up, self.hidden_width)?;
			let output = core_matrix::add(
				&core_matrix::mat_mul_nt(&hidden, &expert_down_weight)?,
				&expert_down_bias,
			)?;
			let expert_gate = core_matrix::slice(gate, 1, start, end)?;
			let weighted = core_matrix::mul(&output, &expert_gate)?;
			combined = Some(if let Some(previous) = combined {
				core_matrix::add(&previous, &weighted)?
			} else {
				weighted
			});
		}
		combined.ok_or_else(|| Error::internal("MoE dense oracle has no experts"))
	}

	fn grouped_projection(
		&self,
		input: &Matrix,
		weight: &Parameter,
		bias: &Parameter,
		offsets: &Matrix,
	) -> Result<Matrix> {
		let (weight_value, weight_version, _) = weight.snapshot();
		let (bias_value, bias_version, _) = bias.snapshot();
		matrix::grouped_linear_m_parameterized(
			input,
			(weight.clone(), weight_value, weight_version),
			(bias.clone(), bias_value, bias_version),
			offsets,
		)
	}

	/// Return the residual model width.
	pub const fn model_width(&self) -> usize {
		self.model_width
	}

	/// Return the expert hidden width.
	pub const fn hidden_width(&self) -> usize {
		self.hidden_width
	}

	/// Return the routed expert count.
	pub const fn num_experts(&self) -> usize {
		self.num_experts
	}

	/// Return the clamped routes selected per token.
	pub const fn experts_per_token(&self) -> usize {
		self.experts_per_token
	}

	/// Select sparse grouped execution or the dense all-expert correctness oracle.
	///
	/// Sparse execution is enabled by default. Dense execution evaluates every
	/// expert and exists for differential correctness testing, not performance.
	pub fn set_sparse_execution(&self, sparse: bool) {
		self.sparse_execution.set(sparse);
	}

	/// Return whether grouped sparse expert execution is selected.
	pub fn sparse_execution(&self) -> bool {
		self.sparse_execution.get()
	}

	/// Return the number of always-on shared SwiGLU experts.
	pub fn num_shared_experts(&self) -> usize {
		self.shared_experts.len()
	}

	/// Return the RMSNorm epsilon.
	pub const fn epsilon(&self) -> f32 {
		self.epsilon
	}

	/// Set the auxiliary-loss-free routing-bias update rate.
	///
	/// Non-finite or non-positive values disable balancing, matching the donor's
	/// clamped opt-in policy.
	pub fn set_balance_rate(&self, gamma: f32) {
		self.balance_rate.set(if gamma.is_finite() && gamma > 0.0 {
			gamma
		} else {
			0.0
		});
	}

	/// Return the current auxiliary-loss-free routing-bias update rate.
	pub fn balance_rate(&self) -> f32 {
		self.balance_rate.get()
	}

	/// Return the persistent device-resident `[1, E]` routing-bias buffer.
	pub fn routing_bias(&self) -> Matrix {
		self.routing_bias.data()
	}

	/// Set the Switch/GShard routing auxiliary-loss coefficient.
	///
	/// Non-finite or non-positive values disable the loss. The default is zero,
	/// so the module's established forward and gradient graph remain unchanged.
	pub fn set_aux_loss_alpha(&self, alpha: f32) {
		self.aux_loss_alpha
			.set(if alpha.is_finite() && alpha > 0.0 {
				alpha
			} else {
				0.0
			});
	}

	/// Return the current Switch/GShard auxiliary-loss coefficient.
	pub fn aux_loss_alpha(&self) -> f32 {
		self.aux_loss_alpha.get()
	}

	/// Set the router z-loss coefficient.
	///
	/// Non-finite or non-positive values disable the loss. The donor objective is
	/// `beta * mean(logsumexp(router_logits)^2)` and is evaluated through a stable
	/// LogSoftmax identity.
	pub fn set_router_z_loss_beta(&self, beta: f32) {
		self.router_z_loss_beta
			.set(if beta.is_finite() && beta > 0.0 {
				beta
			} else {
				0.0
			});
	}

	/// Return the current router z-loss coefficient.
	pub fn router_z_loss_beta(&self) -> f32 {
		self.router_z_loss_beta.get()
	}

	/// Return the scalar auxiliary loss recorded by the latest forward.
	///
	/// Add this value to the task loss before [`crate::ml::GradientTape::backward`].
	/// Before the first forward, and after a forward while the coefficient is
	/// disabled, this returns a stable zero scalar.
	pub fn aux_loss(&self) -> Matrix {
		self.last_aux_loss.borrow().clone()
	}

	/// Queue one routing-bias update from the latest selection mask.
	///
	/// This method does not submit, wait, or read the bias on the host. It is a
	/// no-op while balancing is disabled or before the first forward.
	///
	/// # Errors
	///
	/// Returns an error when the device-resident update cannot be recorded.
	pub fn update_routing_bias(&self) -> Result<()> {
		let gamma = self.balance_rate.get();
		let Some(selection_mask) = self.last_selection_mask() else {
			return Ok(());
		};
		if gamma == 0.0 {
			return Ok(());
		}
		core_matrix::moe_routing_bias_update(
			&selection_mask,
			&self.routing_bias.data(),
			self.experts_per_token,
			gamma,
		)
	}

	/// Return the selection mask from the latest recorded forward, if any.
	pub fn last_selection_mask(&self) -> Option<Matrix> {
		self.last_selection_mask.borrow().clone()
	}

	/// Return router probabilities from the latest recorded forward, if any.
	pub fn last_gate_probabilities(&self) -> Option<Matrix> {
		self.last_gate_probabilities.borrow().clone()
	}

	/// Reduce and read routing telemetry from the latest forward.
	///
	/// This is an explicit host-observation boundary. It reads only two reduced
	/// `[E]` vectors; it never transfers the token-by-expert routing matrices.
	/// Before the first forward, all per-expert values and aggregate statistics
	/// are zero.
	///
	/// # Errors
	///
	/// Returns an error when reduction, submission, completion, or readback fails.
	pub fn route_stats(&self) -> Result<MoeRouteStats> {
		let Some(selection_mask) = self.last_selection_mask() else {
			return Ok(MoeRouteStats {
				load_fraction: vec![0.0; self.num_experts],
				mean_probability: vec![0.0; self.num_experts],
				entropy: 0.0,
				max_load_ratio: 0.0,
				dead_experts: self.num_experts,
			});
		};
		let probabilities = self
			.last_gate_probabilities()
			.expect("selection mask and gate probabilities are committed together");
		let load = core_matrix::sum(&selection_mask, 0)?.reshape([self.num_experts])?;
		let probability_sum = core_matrix::sum(&probabilities, 0)?.reshape([self.num_experts])?;
		let mut load_fraction = load.read_f32()?;
		let mut mean_probability = probability_sum.read_f32()?;
		let load_total = load_fraction
			.iter()
			.map(|value| f64::from(*value))
			.sum::<f64>();
		if load_total > 0.0 {
			for value in &mut load_fraction {
				*value = (f64::from(*value) / load_total) as f32;
			}
		}
		let tokens = probabilities.shape()[0] as f32;
		for value in &mut mean_probability {
			*value /= tokens;
		}
		let mut entropy = 0.0_f64;
		let mut max_load = 0.0_f64;
		let mut dead_experts = 0;
		for &fraction in &load_fraction {
			let fraction = f64::from(fraction);
			if fraction > 0.0 {
				entropy -= fraction * fraction.ln();
			} else {
				dead_experts += 1;
			}
			max_load = max_load.max(fraction);
		}
		let entropy = if self.num_experts > 1 {
			(entropy / (self.num_experts as f64).ln()) as f32
		} else {
			1.0
		};
		Ok(MoeRouteStats {
			load_fraction,
			mean_probability,
			entropy,
			max_load_ratio: (max_load * self.num_experts as f64) as f32,
			dead_experts,
		})
	}

	/// Return the stacked routed-expert gate/up weight.
	pub fn expert_gate_up_weight(&self) -> Parameter {
		self.expert_gate_up_weight.clone()
	}

	/// Return the stacked routed-expert gate/up bias.
	pub fn expert_gate_up_bias(&self) -> Parameter {
		self.expert_gate_up_bias.clone()
	}

	/// Return the stacked routed-expert down-projection weight.
	pub fn expert_down_weight(&self) -> Parameter {
		self.expert_down_weight.clone()
	}

	/// Return the stacked routed-expert down-projection bias.
	pub fn expert_down_bias(&self) -> Parameter {
		self.expert_down_bias.clone()
	}
}

impl Module for Moe {
	fn forward(&self, input: &Matrix) -> Result<Matrix> {
		Moe::forward(self, input)
	}

	fn registry(&self) -> &ModuleRegistry {
		&self.registry
	}
}
