use pyo3::prelude::*;

use crate::{error::python_error, matrix::PythonMatrix, runtime::PythonEngine};

use super::super::autograd::PythonParameter;

// ── MoeRouteStats ─────────────────────────────────────────────────────────────

/// Reduced telemetry from the most recent MoE routing decision.
#[pyclass(name = "MoeRouteStats")]
#[derive(Clone)]
pub(crate) struct PythonMoeRouteStats {
	inner: oa::ml::nn::MoeRouteStats,
}

#[pymethods]
impl PythonMoeRouteStats {
	/// Fraction of token-slot assignments received by each expert.
	#[getter]
	pub fn load_fraction(&self) -> Vec<f32> {
		self.inner.load_fraction().to_vec()
	}

	/// Mean unbiased router probability for each expert.
	#[getter]
	pub fn mean_probability(&self) -> Vec<f32> {
		self.inner.mean_probability().to_vec()
	}

	/// Normalized load entropy in `[0, 1]` where `1` is balanced.
	#[getter]
	pub fn entropy(&self) -> f32 {
		self.inner.entropy()
	}

	/// Expert count times the maximum load fraction (`1` is balanced).
	#[getter]
	pub fn max_load_ratio(&self) -> f32 {
		self.inner.max_load_ratio()
	}

	/// Number of experts that received no token-slot assignments.
	#[getter]
	pub fn dead_experts(&self) -> usize {
		self.inner.dead_experts()
	}

	pub fn __repr__(&self) -> String {
		format!(
			"MoeRouteStats(entropy={:.3}, max_load_ratio={:.3}, dead_experts={})",
			self.inner.entropy(),
			self.inner.max_load_ratio(),
			self.inner.dead_experts(),
		)
	}
}

// ── Moe ───────────────────────────────────────────────────────────────────────

/// Sparse top-k mixture-of-experts feed-forward module.
#[pyclass(name = "Moe", unsendable)]
pub(crate) struct PythonMoe {
	pub(crate) inner: oa::ml::nn::Moe,
}

#[pymethods]
impl PythonMoe {
	/// Construct a deterministically initialized sparse MoE.
	#[new]
	#[pyo3(signature = (engine, model_width, hidden_width, num_experts, experts_per_token, epsilon = 1e-5, seed = 0))]
	#[allow(clippy::too_many_arguments)]
	pub fn new(
		engine: &PythonEngine,
		model_width: usize,
		hidden_width: usize,
		num_experts: usize,
		experts_per_token: usize,
		epsilon: f32,
		seed: u64,
	) -> PyResult<Self> {
		oa::ml::nn::Moe::with_seed(
			&engine.inner,
			model_width,
			hidden_width,
			num_experts,
			experts_per_token,
			epsilon,
			seed,
		)
		.map(|inner| Self { inner })
		.map_err(python_error)
	}

	/// Construct with always-on shared SwiGLU experts alongside the routed set.
	#[staticmethod]
	#[pyo3(signature = (engine, model_width, hidden_width, num_experts, experts_per_token, num_shared_experts, epsilon = 1e-5, seed = 0))]
	#[allow(clippy::too_many_arguments)]
	pub fn with_shared_experts(
		engine: &PythonEngine,
		model_width: usize,
		hidden_width: usize,
		num_experts: usize,
		experts_per_token: usize,
		num_shared_experts: usize,
		epsilon: f32,
		seed: u64,
	) -> PyResult<Self> {
		oa::ml::nn::Moe::with_seed_and_shared_experts(
			&engine.inner,
			model_width,
			hidden_width,
			num_experts,
			experts_per_token,
			epsilon,
			num_shared_experts,
			seed,
		)
		.map(|inner| Self { inner })
		.map_err(python_error)
	}

	/// Apply top-k routed experts and add their delta to the input.
	pub fn forward(&self, input: &PythonMatrix) -> PyResult<PythonMatrix> {
		self
			.inner
			.forward(&input.inner)
			.map(PythonMatrix::wrap)
			.map_err(python_error)
	}

	#[getter]
	pub fn model_width(&self) -> usize {
		self.inner.model_width()
	}

	#[getter]
	pub fn hidden_width(&self) -> usize {
		self.inner.hidden_width()
	}

	#[getter]
	pub fn num_experts(&self) -> usize {
		self.inner.num_experts()
	}

	#[getter]
	pub fn experts_per_token(&self) -> usize {
		self.inner.experts_per_token()
	}

	#[getter]
	pub fn epsilon(&self) -> f32 {
		self.inner.epsilon()
	}

	#[getter]
	pub fn num_shared_experts(&self) -> usize {
		self.inner.num_shared_experts()
	}

	/// Select sparse grouped execution (default) or the dense correctness oracle.
	pub fn set_sparse_execution(&self, sparse: bool) {
		self.inner.set_sparse_execution(sparse);
	}

	pub fn sparse_execution(&self) -> bool {
		self.inner.sparse_execution()
	}

	/// Set the auxiliary-loss-free routing-bias update rate.
	pub fn set_balance_rate(&self, gamma: f32) {
		self.inner.set_balance_rate(gamma);
	}

	pub fn balance_rate(&self) -> f32 {
		self.inner.balance_rate()
	}

	/// Return the device-resident `[1, E]` routing-bias buffer.
	pub fn routing_bias(&self) -> PythonMatrix {
		PythonMatrix::wrap(self.inner.routing_bias())
	}

	/// Set the Switch/GShard routing auxiliary-loss coefficient.
	pub fn set_aux_loss_alpha(&self, alpha: f32) {
		self.inner.set_aux_loss_alpha(alpha);
	}

	pub fn aux_loss_alpha(&self) -> f32 {
		self.inner.aux_loss_alpha()
	}

	/// Set the router z-loss coefficient.
	pub fn set_router_z_loss_beta(&self, beta: f32) {
		self.inner.set_router_z_loss_beta(beta);
	}

	pub fn router_z_loss_beta(&self) -> f32 {
		self.inner.router_z_loss_beta()
	}

	/// Return the scalar auxiliary loss from the latest forward.
	pub fn aux_loss(&self) -> PythonMatrix {
		PythonMatrix::wrap(self.inner.aux_loss())
	}

	/// Queue one routing-bias update from the latest selection mask.
	pub fn update_routing_bias(&self) -> PyResult<()> {
		self.inner.update_routing_bias().map_err(python_error)
	}

	/// Return the selection mask from the latest forward, if any.
	pub fn last_selection_mask(&self) -> Option<PythonMatrix> {
		self.inner.last_selection_mask().map(PythonMatrix::wrap)
	}

	/// Return router probabilities from the latest forward, if any.
	pub fn last_gate_probabilities(&self) -> Option<PythonMatrix> {
		self.inner.last_gate_probabilities().map(PythonMatrix::wrap)
	}

	/// Read routing telemetry (explicit host-observation boundary).
	pub fn route_stats(&self) -> PyResult<PythonMoeRouteStats> {
		self
			.inner
			.route_stats()
			.map(|inner| PythonMoeRouteStats { inner })
			.map_err(python_error)
	}

	pub fn expert_gate_up_weight(&self) -> PythonParameter {
		PythonParameter {
			inner: self.inner.expert_gate_up_weight(),
		}
	}

	pub fn expert_gate_up_bias(&self) -> PythonParameter {
		PythonParameter {
			inner: self.inner.expert_gate_up_bias(),
		}
	}

	pub fn expert_down_weight(&self) -> PythonParameter {
		PythonParameter {
			inner: self.inner.expert_down_weight(),
		}
	}

	pub fn expert_down_bias(&self) -> PythonParameter {
		PythonParameter {
			inner: self.inner.expert_down_bias(),
		}
	}

	pub fn train(&self) {
		use oa::ml::Module as _;
		self.inner.train(true);
	}

	pub fn eval(&self) {
		use oa::ml::Module as _;
		self.inner.train(false);
	}

	pub fn all_parameters(&self) -> PyResult<Vec<PythonParameter>> {
		use oa::ml::Module as _;
		self
			.inner
			.all_parameters()
			.map(|params| {
				params
					.into_iter()
					.map(|inner| PythonParameter { inner })
					.collect()
			})
			.map_err(python_error)
	}

	pub fn __repr__(&self) -> String {
		format!(
			"Moe(model_width={}, hidden_width={}, num_experts={}, experts_per_token={})",
			self.inner.model_width(),
			self.inner.hidden_width(),
			self.inner.num_experts(),
			self.inner.experts_per_token(),
		)
	}
}

pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
	module.add_class::<PythonMoeRouteStats>()?;
	module.add_class::<PythonMoe>()?;
	Ok(())
}
