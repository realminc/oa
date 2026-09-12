use crate::{DType, Error, Matrix, OpAttribute, Result, matrix};

use crate::runtime::SemanticDispatch;

mod session;

pub use session::{Environment, EnvironmentExecution};

/// Structural domain of one reinforcement-learning environment field.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum EnvironmentSpaceKind {
	/// Bounded or unbounded floating-point values.
	Box,
	/// One integer category per environment.
	Discrete,
	/// Zero-or-one values.
	Binary,
}

/// Checked shape, dtype, and range contract for one environment field.
///
/// `shape` excludes the leading environment-batch dimension. An empty shape
/// therefore represents one scalar per environment.
#[derive(Clone, Debug)]
pub struct EnvironmentSpace {
	name: String,
	kind: EnvironmentSpaceKind,
	shape: Vec<usize>,
	dtype: DType,
	minimum: f64,
	maximum: f64,
	cardinality: usize,
}

impl EnvironmentSpace {
	/// Construct a floating-point Box space.
	///
	/// # Errors
	///
	/// Returns an error for an empty name, unsupported dtype, zero/overflowing
	/// dimensions, NaN bounds, or reversed bounds.
	pub fn continuous(
		name: impl Into<String>,
		shape: impl Into<Vec<usize>>,
		dtype: DType,
		minimum: f64,
		maximum: f64,
	) -> Result<Self> {
		Self::checked(
			name.into(),
			EnvironmentSpaceKind::Box,
			shape.into(),
			dtype,
			minimum,
			maximum,
			0,
		)
	}

	/// Construct a scalar integer category space over `[0, cardinality)`.
	///
	/// # Errors
	///
	/// Returns an error for an empty name, zero cardinality, unsupported dtype,
	/// or a cardinality whose inclusive upper bound cannot be represented exactly.
	pub fn discrete(name: impl Into<String>, cardinality: usize, dtype: DType) -> Result<Self> {
		let maximum = cardinality
			.checked_sub(1)
			.ok_or_else(|| Error::invalid_argument("discrete cardinality must be positive"))?;
		if maximum as u128 > 1_u128 << 53 {
			return Err(Error::invalid_argument(
				"discrete cardinality exceeds exact f64 bound representation",
			));
		}
		Self::checked(
			name.into(),
			EnvironmentSpaceKind::Discrete,
			Vec::new(),
			dtype,
			0.0,
			maximum as f64,
			cardinality,
		)
	}

	/// Construct a U8 binary scalar or shaped field.
	///
	/// # Errors
	///
	/// Returns an error for an empty name or zero/overflowing dimensions.
	pub fn binary(name: impl Into<String>, shape: impl Into<Vec<usize>>) -> Result<Self> {
		Self::checked(
			name.into(),
			EnvironmentSpaceKind::Binary,
			shape.into(),
			DType::U8,
			0.0,
			1.0,
			2,
		)
	}

	#[allow(clippy::too_many_arguments)]
	fn checked(
		name: String,
		kind: EnvironmentSpaceKind,
		shape: Vec<usize>,
		dtype: DType,
		minimum: f64,
		maximum: f64,
		cardinality: usize,
	) -> Result<Self> {
		if name.is_empty() {
			return Err(Error::invalid_argument(
				"environment field name must not be empty",
			));
		}
		if shape.contains(&0) {
			return Err(Error::invalid_argument(format!(
				"environment field {name:?} dimensions must be positive"
			)));
		}
		shape.iter().try_fold(1_usize, |elements, dimension| {
			elements.checked_mul(*dimension).ok_or_else(|| {
				Error::invalid_argument(format!(
					"environment field {name:?} shape product overflows usize"
				))
			})
		})?;
		if minimum.is_nan() || maximum.is_nan() || minimum > maximum {
			return Err(Error::invalid_argument(format!(
				"environment field {name:?} bounds must be ordered and not NaN"
			)));
		}
		let valid_kind = match kind {
			EnvironmentSpaceKind::Box => dtype == DType::F32 && cardinality == 0,
			EnvironmentSpaceKind::Discrete => {
				matches!(dtype, DType::U8 | DType::I32 | DType::U32)
					&& shape.is_empty()
					&& cardinality > 0
					&& minimum == 0.0
					&& maximum == (cardinality - 1) as f64
			}
			EnvironmentSpaceKind::Binary => {
				dtype == DType::U8 && cardinality == 2 && minimum == 0.0 && maximum == 1.0
			}
		};
		if !valid_kind {
			return Err(Error::invalid_argument(format!(
				"environment field {name:?} kind and dtype contract is invalid"
			)));
		}
		Ok(Self {
			name,
			kind,
			shape,
			dtype,
			minimum,
			maximum,
			cardinality,
		})
	}

	/// Return the field name.
	pub fn name(&self) -> &str {
		&self.name
	}

	/// Return the structural space kind.
	pub const fn kind(&self) -> EnvironmentSpaceKind {
		self.kind
	}

	/// Return the per-environment shape without the batch dimension.
	pub fn shape(&self) -> &[usize] {
		&self.shape
	}

	/// Return the required Matrix dtype.
	pub const fn dtype(&self) -> DType {
		self.dtype
	}

	/// Return the inclusive lower bound.
	pub const fn minimum(&self) -> f64 {
		self.minimum
	}

	/// Return the inclusive upper bound.
	pub const fn maximum(&self) -> f64 {
		self.maximum
	}

	/// Return the category count, or zero for a Box space.
	pub const fn cardinality(&self) -> usize {
		self.cardinality
	}

	/// Return the number of dense values stored per environment.
	pub fn elements_per_environment(&self) -> usize {
		self.shape.iter().copied().product()
	}

	/// Return the exact Matrix shape for a positive environment count.
	///
	/// # Errors
	///
	/// Returns an error when `environments` is zero.
	pub fn batched_shape(&self, environments: u32) -> Result<Vec<usize>> {
		if environments == 0 {
			return Err(Error::invalid_argument(format!(
				"environment field {:?} requires a positive environment count",
				self.name
			)));
		}
		let mut shape = Vec::with_capacity(self.shape.len() + 1);
		shape.push(environments as usize);
		shape.extend_from_slice(&self.shape);
		Ok(shape)
	}

	/// Validate one device Matrix against this batched field contract.
	///
	/// # Errors
	///
	/// Returns an error for zero environments or a shape/dtype mismatch.
	pub fn validate_matrix(&self, matrix: &Matrix, environments: u32) -> Result<()> {
		let expected = self.batched_shape(environments)?;
		if matrix.dtype() != self.dtype || matrix.shape() != expected {
			return Err(Error::invalid_argument(format!(
				"environment field {:?} requires {:?} {:?}; found {:?} {:?}",
				self.name,
				self.dtype,
				expected,
				matrix.dtype(),
				matrix.shape()
			)));
		}
		Ok(())
	}
}

/// Complete single-agent environment schema shared by native and adapter paths.
#[derive(Clone, Debug)]
pub struct EnvironmentSpec {
	observation: EnvironmentSpace,
	action: EnvironmentSpace,
	reward: EnvironmentSpace,
	terminated: EnvironmentSpace,
	truncated: EnvironmentSpace,
}

impl EnvironmentSpec {
	/// Construct the five durable Gymnasium-compatible field contracts.
	///
	/// # Errors
	///
	/// Returns an error unless reward is a scalar Box and both episode-boundary
	/// fields are scalar Binary spaces.
	pub fn new(
		observation: EnvironmentSpace,
		action: EnvironmentSpace,
		reward: EnvironmentSpace,
		terminated: EnvironmentSpace,
		truncated: EnvironmentSpace,
	) -> Result<Self> {
		if reward.kind != EnvironmentSpaceKind::Box || !reward.shape.is_empty() {
			return Err(Error::invalid_argument(
				"environment reward must be one floating scalar per environment",
			));
		}
		if terminated.kind != EnvironmentSpaceKind::Binary
			|| !terminated.shape.is_empty()
			|| truncated.kind != EnvironmentSpaceKind::Binary
			|| !truncated.shape.is_empty()
		{
			return Err(Error::invalid_argument(
				"environment termination and truncation must be binary scalars",
			));
		}
		Ok(Self {
			observation,
			action,
			reward,
			terminated,
			truncated,
		})
	}

	/// Return the observation field contract.
	pub const fn observation(&self) -> &EnvironmentSpace {
		&self.observation
	}

	/// Return the action field contract.
	pub const fn action(&self) -> &EnvironmentSpace {
		&self.action
	}

	/// Return the scalar reward field contract.
	pub const fn reward(&self) -> &EnvironmentSpace {
		&self.reward
	}

	/// Return the task-terminal field contract.
	pub const fn terminated(&self) -> &EnvironmentSpace {
		&self.terminated
	}

	/// Return the external-truncation field contract.
	pub const fn truncated(&self) -> &EnvironmentSpace {
		&self.truncated
	}

	/// Validate one batched reset observation without reading it back.
	///
	/// # Errors
	///
	/// Returns an error for a zero environment count or shape/dtype mismatch.
	pub fn validate_reset(&self, observation: &Matrix, environments: u32) -> Result<()> {
		self.observation.validate_matrix(observation, environments)
	}

	/// Validate one batched action without reading it back.
	///
	/// # Errors
	///
	/// Returns an error for a zero environment count or shape/dtype mismatch.
	pub fn validate_action(&self, action: &Matrix, environments: u32) -> Result<()> {
		self.action.validate_matrix(action, environments)
	}

	/// Validate action and all five transition values without readback.
	///
	/// # Errors
	///
	/// Returns an error for a zero environment count or any shape/dtype mismatch.
	pub fn validate_transition(
		&self,
		action: &Matrix,
		transition: &EnvironmentTransition,
		environments: u32,
	) -> Result<()> {
		self.observation
			.validate_matrix(&transition.observation, environments)?;
		self.action.validate_matrix(action, environments)?;
		self.observation
			.validate_matrix(&transition.next_observation, environments)?;
		self.reward
			.validate_matrix(&transition.reward, environments)?;
		self.terminated
			.validate_matrix(&transition.terminated, environments)?;
		self.truncated
			.validate_matrix(&transition.truncated, environments)
	}
}

/// One batched environment step result preserving termination and truncation.
pub struct EnvironmentTransition {
	observation: Matrix,
	next_observation: Matrix,
	reward: Matrix,
	terminated: Matrix,
	truncated: Matrix,
}

impl EnvironmentTransition {
	/// Construct an owned transition value; [`EnvironmentSpec::validate_transition`]
	/// proves its complete field contract together with the action.
	pub fn new(
		observation: Matrix,
		next_observation: Matrix,
		reward: Matrix,
		terminated: Matrix,
		truncated: Matrix,
	) -> Self {
		Self {
			observation,
			next_observation,
			reward,
			terminated,
			truncated,
		}
	}

	/// Return the observation before the action.
	pub const fn observation(&self) -> &Matrix {
		&self.observation
	}

	/// Return the observation after the action.
	pub const fn next_observation(&self) -> &Matrix {
		&self.next_observation
	}

	/// Return one reward per environment.
	pub const fn reward(&self) -> &Matrix {
		&self.reward
	}

	/// Return task-terminal flags, distinct from truncation.
	pub const fn terminated(&self) -> &Matrix {
		&self.terminated
	}

	/// Return external/time-limit flags, distinct from termination.
	pub const fn truncated(&self) -> &Matrix {
		&self.truncated
	}
}

/// Normalize and symmetrically clip FP32 observations.
///
/// Computes `clamp((observation - mean) / (stddev + epsilon), -clip, clip)`
/// through differentiable Core Matrix operations under one Environment
/// semantic identity.
///
/// # Errors
///
/// Returns an error for empty/non-FP32 inputs, invalid epsilon/clip, incompatible
/// broadcasting or ownership, or composite lowering failure.
pub fn normalize_observation(
	observation: &Matrix,
	mean: &Matrix,
	stddev: &Matrix,
	epsilon: f32,
	clip: f32,
) -> Result<Matrix> {
	const CONTRACT: crate::OperationContract = crate::core::operation::ml::NORMALIZE_OBSERVATION;
	if [observation, mean, stddev]
		.into_iter()
		.any(|input| input.dtype() != DType::F32 || input.num_elements() == 0)
		|| !epsilon.is_finite()
		|| epsilon <= 0.0
		|| !clip.is_finite()
		|| clip <= 0.0
	{
		return Err(Error::invalid_argument(format!(
			"{} requires nonempty FP32 inputs and positive finite epsilon/clip",
			CONTRACT.name()
		)));
	}
	let lowering = observation.engine_handle().begin_semantic_lowering()?;
	let centered = matrix::sub(observation, mean)?;
	let denominator = matrix::add_scalar(stddev, epsilon)?;
	let normalized = matrix::div(&centered, &denominator)?;
	let result = clamp(&normalized, -clip, clip)?;
	let attributes = [
		OpAttribute::Float {
			name: "epsilon".into(),
			value: f64::from(epsilon),
		},
		OpAttribute::Float {
			name: "clip".into(),
			value: f64::from(clip),
		},
	];
	lowering.commit(SemanticDispatch {
		contract: CONTRACT,
		inputs: &[observation, mean, stddev],
		outputs: &[&result],
		attributes: &attributes,
	})?;
	Ok(result)
}

/// Affinely map FP32 actions between two finite ranges.
///
/// # Errors
///
/// Returns an error for an empty/non-FP32 input, unordered/non-finite bounds,
/// or Matrix/composite lowering failure.
pub fn scale_action(
	action: &Matrix,
	source_minimum: f32,
	source_maximum: f32,
	target_minimum: f32,
	target_maximum: f32,
	clamp_source: bool,
) -> Result<Matrix> {
	const CONTRACT: crate::OperationContract = crate::core::operation::ml::SCALE_ACTION;
	if action.dtype() != DType::F32
		|| action.num_elements() == 0
		|| !source_minimum.is_finite()
		|| !source_maximum.is_finite()
		|| !target_minimum.is_finite()
		|| !target_maximum.is_finite()
		|| source_minimum >= source_maximum
		|| target_minimum >= target_maximum
	{
		return Err(Error::invalid_argument(format!(
			"{} requires nonempty FP32 actions and ordered finite bounds",
			CONTRACT.name()
		)));
	}
	let lowering = action.engine_handle().begin_semantic_lowering()?;
	let source = if clamp_source {
		clamp(action, source_minimum, source_maximum)?
	} else {
		action.clone()
	};
	let scale = (target_maximum - target_minimum) / (source_maximum - source_minimum);
	let result = matrix::add_scalar(
		&matrix::scale(&matrix::sub_scalar(&source, source_minimum)?, scale)?,
		target_minimum,
	)?;
	let attributes = [
		OpAttribute::Float {
			name: "source_minimum".into(),
			value: f64::from(source_minimum),
		},
		OpAttribute::Float {
			name: "source_maximum".into(),
			value: f64::from(source_maximum),
		},
		OpAttribute::Float {
			name: "target_minimum".into(),
			value: f64::from(target_minimum),
		},
		OpAttribute::Float {
			name: "target_maximum".into(),
			value: f64::from(target_maximum),
		},
		OpAttribute::Boolean {
			name: "clamp".into(),
			value: clamp_source,
		},
	];
	lowering.commit(SemanticDispatch {
		contract: CONTRACT,
		inputs: &[action],
		outputs: &[&result],
		attributes: &attributes,
	})?;
	Ok(result)
}

/// Clip FP32 rewards to an inclusive finite range.
///
/// # Errors
///
/// Returns an error for an empty/non-FP32 input, unordered/non-finite bounds,
/// or Matrix/composite lowering failure.
pub fn clip_reward(reward: &Matrix, minimum: f32, maximum: f32) -> Result<Matrix> {
	const CONTRACT: crate::OperationContract = crate::core::operation::ml::CLIP_REWARD;
	if reward.dtype() != DType::F32
		|| reward.num_elements() == 0
		|| !minimum.is_finite()
		|| !maximum.is_finite()
		|| minimum > maximum
	{
		return Err(Error::invalid_argument(format!(
			"{} requires nonempty FP32 rewards and ordered finite bounds",
			CONTRACT.name()
		)));
	}
	let lowering = reward.engine_handle().begin_semantic_lowering()?;
	let result = clamp(reward, minimum, maximum)?;
	let attributes = [
		OpAttribute::Float {
			name: "minimum".into(),
			value: f64::from(minimum),
		},
		OpAttribute::Float {
			name: "maximum".into(),
			value: f64::from(maximum),
		},
	];
	lowering.commit(SemanticDispatch {
		contract: CONTRACT,
		inputs: &[reward],
		outputs: &[&result],
		attributes: &attributes,
	})?;
	Ok(result)
}

fn clamp(input: &Matrix, minimum: f32, maximum: f32) -> Result<Matrix> {
	matrix::clamp_max(&matrix::clamp_min(input, minimum)?, maximum)
}
