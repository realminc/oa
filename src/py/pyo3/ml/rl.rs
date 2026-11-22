use pyo3::prelude::*;

use crate::{error::python_error, matrix::PythonMatrix};

// ── PpoLossConfig ─────────────────────────────────────────────────────────────

/// Coefficients for the PPO composite objective.
#[pyclass(name = "PpoLossConfig")]
#[derive(Clone)]
pub(crate) struct PythonPpoLossConfig {
	pub(crate) inner: oa::ml::loss::PpoLossConfig,
}

#[pymethods]
impl PythonPpoLossConfig {
	#[new]
	#[pyo3(signature = (clip_epsilon = 0.2, value_coefficient = 0.5, entropy_coefficient = 0.01))]
	pub fn new(clip_epsilon: f32, value_coefficient: f32, entropy_coefficient: f32) -> Self {
		Self {
			inner: oa::ml::loss::PpoLossConfig {
				clip_epsilon,
				value_coefficient,
				entropy_coefficient,
			},
		}
	}

	#[getter]
	pub fn clip_epsilon(&self) -> f32 {
		self.inner.clip_epsilon
	}

	#[getter]
	pub fn value_coefficient(&self) -> f32 {
		self.inner.value_coefficient
	}

	#[getter]
	pub fn entropy_coefficient(&self) -> f32 {
		self.inner.entropy_coefficient
	}

	pub fn __repr__(&self) -> String {
		format!(
			"PpoLossConfig(clip_epsilon={}, value_coef={}, entropy_coef={})",
			self.inner.clip_epsilon, self.inner.value_coefficient, self.inner.entropy_coefficient,
		)
	}
}

// ── PpoLossResult ─────────────────────────────────────────────────────────────

/// PPO loss components retained for metrics and reverse mode.
#[pyclass(name = "PpoLossResult", unsendable)]
pub(crate) struct PythonPpoLossResult {
	inner: oa::ml::loss::PpoLossResult,
}

#[pymethods]
impl PythonPpoLossResult {
	/// Mean negative clipped surrogate objective.
	#[getter]
	pub fn policy_loss(&self) -> PythonMatrix {
		PythonMatrix::wrap(self.inner.policy_loss.clone())
	}

	/// Mean-squared critic error.
	#[getter]
	pub fn value_loss(&self) -> PythonMatrix {
		PythonMatrix::wrap(self.inner.value_loss.clone())
	}

	/// Mean policy entropy.
	#[getter]
	pub fn entropy(&self) -> PythonMatrix {
		PythonMatrix::wrap(self.inner.entropy.clone())
	}

	/// Total composite loss.
	#[getter]
	pub fn total_loss(&self) -> PythonMatrix {
		PythonMatrix::wrap(self.inner.total_loss.clone())
	}

	pub fn __repr__(&self) -> &str {
		"PpoLossResult"
	}
}

// ── DqnLossConfig ─────────────────────────────────────────────────────────────

/// Discount factor for the DQN Bellman target.
#[pyclass(name = "DqnLossConfig")]
#[derive(Clone)]
pub(crate) struct PythonDqnLossConfig {
	pub(crate) inner: oa::ml::loss::DqnLossConfig,
}

#[pymethods]
impl PythonDqnLossConfig {
	#[new]
	#[pyo3(signature = (discount = 0.99))]
	pub fn new(discount: f32) -> Self {
		Self {
			inner: oa::ml::loss::DqnLossConfig { discount },
		}
	}

	#[getter]
	pub fn discount(&self) -> f32 {
		self.inner.discount
	}

	pub fn __repr__(&self) -> String {
		format!("DqnLossConfig(discount={})", self.inner.discount)
	}
}

// ── DqnLossResult ─────────────────────────────────────────────────────────────

/// Selected action values, detached targets, and scalar DQN loss.
#[pyclass(name = "DqnLossResult", unsendable)]
pub(crate) struct PythonDqnLossResult {
	inner: oa::ml::loss::DqnLossResult,
}

#[pymethods]
impl PythonDqnLossResult {
	/// Q values selected by the batch action indices, shaped `[batch]`.
	#[getter]
	pub fn selected_q(&self) -> PythonMatrix {
		PythonMatrix::wrap(self.inner.selected_q.clone())
	}

	/// Detached Bellman targets, shaped `[batch]`.
	#[getter]
	pub fn target_q(&self) -> PythonMatrix {
		PythonMatrix::wrap(self.inner.target_q.clone())
	}

	/// Mean Smooth L1 loss.
	#[getter]
	pub fn loss(&self) -> PythonMatrix {
		PythonMatrix::wrap(self.inner.loss.clone())
	}

	pub fn __repr__(&self) -> &str {
		"DqnLossResult"
	}
}

// ── SacLossConfig ─────────────────────────────────────────────────────────────

/// Discount and entropy temperature for SAC losses.
#[pyclass(name = "SacLossConfig")]
#[derive(Clone)]
pub(crate) struct PythonSacLossConfig {
	pub(crate) inner: oa::ml::loss::SacLossConfig,
}

#[pymethods]
impl PythonSacLossConfig {
	#[new]
	#[pyo3(signature = (discount = 0.99, entropy_coefficient = 0.2))]
	pub fn new(discount: f32, entropy_coefficient: f32) -> Self {
		Self {
			inner: oa::ml::loss::SacLossConfig {
				discount,
				entropy_coefficient,
			},
		}
	}

	#[getter]
	pub fn discount(&self) -> f32 {
		self.inner.discount
	}

	#[getter]
	pub fn entropy_coefficient(&self) -> f32 {
		self.inner.entropy_coefficient
	}

	pub fn __repr__(&self) -> String {
		format!(
			"SacLossConfig(discount={}, entropy_coef={})",
			self.inner.discount, self.inner.entropy_coefficient,
		)
	}
}

// ── SacCriticLossResult ───────────────────────────────────────────────────────

/// Detached target and twin-critic SAC loss components.
#[pyclass(name = "SacCriticLossResult", unsendable)]
pub(crate) struct PythonSacCriticLossResult {
	inner: oa::ml::loss::SacCriticLossResult,
}

#[pymethods]
impl PythonSacCriticLossResult {
	/// Detached entropy-regularized Bellman target, shaped `[batch]`.
	#[getter]
	pub fn target_q(&self) -> PythonMatrix {
		PythonMatrix::wrap(self.inner.target_q.clone())
	}

	/// Mean-squared loss for the first critic.
	#[getter]
	pub fn q1_loss(&self) -> PythonMatrix {
		PythonMatrix::wrap(self.inner.q1_loss.clone())
	}

	/// Mean-squared loss for the second critic.
	#[getter]
	pub fn q2_loss(&self) -> PythonMatrix {
		PythonMatrix::wrap(self.inner.q2_loss.clone())
	}

	/// Sum of both critic losses.
	#[getter]
	pub fn total_loss(&self) -> PythonMatrix {
		PythonMatrix::wrap(self.inner.total_loss.clone())
	}

	pub fn __repr__(&self) -> &str {
		"SacCriticLossResult"
	}
}

// ── GaeConfig ────────────────────────────────────────────────────────────────

/// Discount parameters for generalized advantage estimation.
#[pyclass(name = "GaeConfig")]
#[derive(Clone)]
pub(crate) struct PythonGaeConfig {
	pub(crate) inner: oa::ml::advantage::GaeConfig,
}

#[pymethods]
impl PythonGaeConfig {
	#[new]
	#[pyo3(signature = (gamma = 0.99, lambda = 0.95))]
	pub fn new(gamma: f32, lambda: f32) -> Self {
		Self {
			inner: oa::ml::advantage::GaeConfig { gamma, lambda },
		}
	}

	#[getter]
	pub fn gamma(&self) -> f32 {
		self.inner.gamma
	}

	#[getter]
	pub fn lambda(&self) -> f32 {
		self.inner.lambda
	}

	pub fn __repr__(&self) -> String {
		format!(
			"GaeConfig(gamma={}, lambda={})",
			self.inner.gamma, self.inner.lambda,
		)
	}
}

// ── GaeResult ─────────────────────────────────────────────────────────────────

/// Generalized advantages and matching value targets.
#[pyclass(name = "GaeResult", unsendable)]
pub(crate) struct PythonGaeResult {
	inner: oa::ml::advantage::GaeResult,
}

#[pymethods]
impl PythonGaeResult {
	/// Advantage estimates with the same `[time, environments]` shape as reward.
	#[getter]
	pub fn advantage(&self) -> PythonMatrix {
		PythonMatrix::wrap(self.inner.advantage.clone())
	}

	/// Returns computed as `advantage + value`.
	#[getter]
	pub fn returns(&self) -> PythonMatrix {
		PythonMatrix::wrap(self.inner.returns.clone())
	}

	pub fn __repr__(&self) -> &str {
		"GaeResult"
	}
}

// ── PolicyResult ──────────────────────────────────────────────────────────────

/// Categorical action and differentiable policy statistics.
#[pyclass(name = "PolicyResult", unsendable)]
pub(crate) struct PythonPolicyResult {
	pub(crate) inner: oa::ml::PolicyResult,
}

#[pymethods]
impl PythonPolicyResult {
	/// I32 action index per environment.
	#[getter]
	pub fn action(&self) -> PythonMatrix {
		PythonMatrix::wrap(self.inner.action.clone())
	}

	/// Log probability of each selected action.
	#[getter]
	pub fn log_probability(&self) -> PythonMatrix {
		PythonMatrix::wrap(self.inner.log_probability.clone())
	}

	/// Categorical entropy per environment.
	#[getter]
	pub fn entropy(&self) -> PythonMatrix {
		PythonMatrix::wrap(self.inner.entropy.clone())
	}

	/// Critic value passed through without storage copy.
	#[getter]
	pub fn value(&self) -> PythonMatrix {
		PythonMatrix::wrap(self.inner.value.clone())
	}

	pub fn __repr__(&self) -> &str {
		"PolicyResult"
	}
}

// ── ContinuousPolicyResult ────────────────────────────────────────────────────

/// Bounded continuous action and differentiable policy statistics.
#[pyclass(name = "ContinuousPolicyResult", unsendable)]
pub(crate) struct PythonContinuousPolicyResult {
	inner: oa::ml::ContinuousPolicyResult,
}

#[pymethods]
impl PythonContinuousPolicyResult {
	/// Action after tanh squashing and affine mapping to the range.
	#[getter]
	pub fn action(&self) -> PythonMatrix {
		PythonMatrix::wrap(self.inner.action.clone())
	}

	/// Unsquashed normal sample retained for exact policy reevaluation.
	#[getter]
	pub fn raw_action(&self) -> PythonMatrix {
		PythonMatrix::wrap(self.inner.raw_action.clone())
	}

	/// Summed corrected log probability per environment.
	#[getter]
	pub fn log_probability(&self) -> PythonMatrix {
		PythonMatrix::wrap(self.inner.log_probability.clone())
	}

	/// Summed base-normal entropy per environment.
	#[getter]
	pub fn entropy(&self) -> PythonMatrix {
		PythonMatrix::wrap(self.inner.entropy.clone())
	}

	/// Critic value passed through without storage copy.
	#[getter]
	pub fn value(&self) -> PythonMatrix {
		PythonMatrix::wrap(self.inner.value.clone())
	}

	pub fn __repr__(&self) -> &str {
		"ContinuousPolicyResult"
	}
}

// ── RL loss functions ─────────────────────────────────────────────────────────

#[pyfunction]
pub(crate) fn ml_loss_ppo_clipped_policy(
	new_log_probability: &PythonMatrix,
	old_log_probability: &PythonMatrix,
	advantage: &PythonMatrix,
	clip_epsilon: f32,
) -> PyResult<PythonMatrix> {
	oa::ml::loss::ppo_clipped_policy(
		&new_log_probability.inner,
		&old_log_probability.inner,
		&advantage.inner,
		clip_epsilon,
	)
	.map(PythonMatrix::wrap)
	.map_err(python_error)
}

#[pyfunction]
pub(crate) fn ml_loss_ppo(
	new_log_probability: &PythonMatrix,
	old_log_probability: &PythonMatrix,
	advantage: &PythonMatrix,
	value: &PythonMatrix,
	target_return: &PythonMatrix,
	entropy: &PythonMatrix,
	config: &PythonPpoLossConfig,
) -> PyResult<PythonPpoLossResult> {
	oa::ml::loss::ppo(
		&new_log_probability.inner,
		&old_log_probability.inner,
		&advantage.inner,
		&value.inner,
		&target_return.inner,
		&entropy.inner,
		config.inner,
	)
	.map(|inner| PythonPpoLossResult { inner })
	.map_err(python_error)
}

#[pyfunction]
#[allow(clippy::too_many_arguments)]
pub(crate) fn ml_loss_dqn(
	q: &PythonMatrix,
	action: &PythonMatrix,
	reward: &PythonMatrix,
	next_q: &PythonMatrix,
	terminated: &PythonMatrix,
	truncated: &PythonMatrix,
	config: &PythonDqnLossConfig,
) -> PyResult<PythonDqnLossResult> {
	oa::ml::loss::dqn(
		&q.inner,
		&action.inner,
		&reward.inner,
		&next_q.inner,
		&terminated.inner,
		&truncated.inner,
		config.inner,
	)
	.map(|inner| PythonDqnLossResult { inner })
	.map_err(python_error)
}

#[pyfunction]
#[allow(clippy::too_many_arguments)]
pub(crate) fn ml_loss_sac_critic(
	q1: &PythonMatrix,
	q2: &PythonMatrix,
	reward: &PythonMatrix,
	next_q1: &PythonMatrix,
	next_q2: &PythonMatrix,
	next_log_probability: &PythonMatrix,
	terminated: &PythonMatrix,
	truncated: &PythonMatrix,
	config: &PythonSacLossConfig,
) -> PyResult<PythonSacCriticLossResult> {
	oa::ml::loss::sac_critic(
		&q1.inner,
		&q2.inner,
		&reward.inner,
		&next_q1.inner,
		&next_q2.inner,
		&next_log_probability.inner,
		&terminated.inner,
		&truncated.inner,
		config.inner,
	)
	.map(|inner| PythonSacCriticLossResult { inner })
	.map_err(python_error)
}

#[pyfunction]
pub(crate) fn ml_loss_sac_actor(
	q1: &PythonMatrix,
	q2: &PythonMatrix,
	log_probability: &PythonMatrix,
	entropy_coefficient: f32,
) -> PyResult<PythonMatrix> {
	oa::ml::loss::sac_actor(
		&q1.inner,
		&q2.inner,
		&log_probability.inner,
		entropy_coefficient,
	)
	.map(PythonMatrix::wrap)
	.map_err(python_error)
}

// ── Advantage functions ───────────────────────────────────────────────────────

#[pyfunction]
pub(crate) fn ml_advantage_normalize(
	advantage: &PythonMatrix,
	epsilon: f32,
) -> PyResult<PythonMatrix> {
	oa::ml::advantage::normalize(&advantage.inner, epsilon)
		.map(PythonMatrix::wrap)
		.map_err(python_error)
}

#[pyfunction]
#[allow(clippy::too_many_arguments)]
pub(crate) fn ml_advantage_gae(
	reward: &PythonMatrix,
	value: &PythonMatrix,
	next_value: &PythonMatrix,
	terminated: &PythonMatrix,
	truncated: &PythonMatrix,
	config: &PythonGaeConfig,
) -> PyResult<PythonGaeResult> {
	oa::ml::advantage::gae(
		&reward.inner,
		&value.inner,
		&next_value.inner,
		&terminated.inner,
		&truncated.inner,
		config.inner,
	)
	.map(|inner| PythonGaeResult { inner })
	.map_err(python_error)
}

// ── Policy functions ──────────────────────────────────────────────────────────

#[pyfunction]
pub(crate) fn ml_policy_evaluate_categorical(
	logits: &PythonMatrix,
	action: &PythonMatrix,
	value: &PythonMatrix,
) -> PyResult<PythonPolicyResult> {
	oa::ml::policy::evaluate_categorical(&logits.inner, &action.inner, &value.inner)
		.map(|inner| PythonPolicyResult { inner })
		.map_err(python_error)
}

#[pyfunction]
pub(crate) fn ml_policy_sample_categorical(
	logits: &PythonMatrix,
	value: &PythonMatrix,
	seed: u64,
) -> PyResult<PythonPolicyResult> {
	oa::ml::policy::sample_categorical(&logits.inner, &value.inner, seed)
		.map(|inner| PythonPolicyResult { inner })
		.map_err(python_error)
}

#[pyfunction]
#[allow(clippy::too_many_arguments)]
pub(crate) fn ml_policy_evaluate_tanh_normal(
	mean: &PythonMatrix,
	log_stddev: &PythonMatrix,
	raw_action: &PythonMatrix,
	value: &PythonMatrix,
	minimum: f32,
	maximum: f32,
	epsilon: f32,
) -> PyResult<PythonContinuousPolicyResult> {
	oa::ml::policy::evaluate_tanh_normal(
		&mean.inner,
		&log_stddev.inner,
		&raw_action.inner,
		&value.inner,
		minimum,
		maximum,
		epsilon,
	)
	.map(|inner| PythonContinuousPolicyResult { inner })
	.map_err(python_error)
}

#[pyfunction]
#[allow(clippy::too_many_arguments)]
pub(crate) fn ml_policy_sample_tanh_normal(
	mean: &PythonMatrix,
	log_stddev: &PythonMatrix,
	value: &PythonMatrix,
	minimum: f32,
	maximum: f32,
	seed: u64,
	epsilon: f32,
) -> PyResult<PythonContinuousPolicyResult> {
	oa::ml::policy::sample_tanh_normal(
		&mean.inner,
		&log_stddev.inner,
		&value.inner,
		minimum,
		maximum,
		seed,
		epsilon,
	)
	.map(|inner| PythonContinuousPolicyResult { inner })
	.map_err(python_error)
}

pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
	module.add_class::<PythonPpoLossConfig>()?;
	module.add_class::<PythonPpoLossResult>()?;
	module.add_class::<PythonDqnLossConfig>()?;
	module.add_class::<PythonDqnLossResult>()?;
	module.add_class::<PythonSacLossConfig>()?;
	module.add_class::<PythonSacCriticLossResult>()?;
	module.add_class::<PythonGaeConfig>()?;
	module.add_class::<PythonGaeResult>()?;
	module.add_class::<PythonPolicyResult>()?;
	module.add_class::<PythonContinuousPolicyResult>()?;
	macro_rules! add_functions {
		($($f:ident),+ $(,)?) => { $(module.add_function(wrap_pyfunction!($f, module)?)?;)+ };
	}
	add_functions!(
		ml_loss_ppo_clipped_policy,
		ml_loss_ppo,
		ml_loss_dqn,
		ml_loss_sac_critic,
		ml_loss_sac_actor,
		ml_advantage_normalize,
		ml_advantage_gae,
		ml_policy_evaluate_categorical,
		ml_policy_sample_categorical,
		ml_policy_evaluate_tanh_normal,
		ml_policy_sample_tanh_normal,
	);
	Ok(())
}
