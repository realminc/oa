//! Parametric activation modules: learned-slope PReLU and alpha-scaled ELU.

use crate::ml::matrix::relu;
use crate::{Engine, Error, Matrix, Result, matrix};

use super::super::{Module, ModuleRegistry, Parameter, autograd};

/// Parametric ReLU: `max(0, x) - slope * relu(-x)` elementwise.
///
/// `num_slope_parameters` is `1` for a shared slope or `channels` for the
/// per-channel `nn.PReLU` form; channelwise slopes broadcast over the channel
/// (second) axis of rank >= 2 inputs. The slope is a trainable `Parameter`
/// registered as `slope`, initialized to `0.25` matching the PyTorch default.
pub struct Prelu {
	num_slope_parameters: usize,
	slope: Parameter,
	registry: ModuleRegistry,
}

impl Prelu {
	/// Construct with `num_slope_parameters` slopes initialized to `0.25`.
	///
	/// # Errors
	///
	/// Returns an error when `num_slope_parameters` is zero or allocation fails.
	pub fn new_on(engine: &Engine, num_slope_parameters: usize) -> Result<Self> {
		if num_slope_parameters == 0 {
			return Err(Error::invalid_argument(
				"prelu requires at least one slope parameter",
			));
		}
		let data = Matrix::new_from_f32_on(
			engine,
			[num_slope_parameters],
			&vec![0.25_f32; num_slope_parameters],
		)?;
		let slope = Parameter::new("slope", data)?;
		let mut registry = ModuleRegistry::new();
		registry.register_parameter("slope", slope.clone())?;
		Ok(Self {
			num_slope_parameters,
			slope,
			registry,
		})
	}

	/// Construct on the current engine.
	///
	/// # Errors
	///
	/// Returns an error under the same contract as [`Self::new_on`].
	pub fn new(num_slope_parameters: usize) -> Result<Self> {
		Self::new_on(&Engine::current()?, num_slope_parameters)
	}

	/// Return the learned slope parameter.
	pub fn slope(&self) -> Parameter {
		self.slope.clone()
	}

	/// Apply `max(0, x) - slope * relu(-x)`.
	///
	/// # Errors
	///
	/// Returns an error when a channelwise slope count does not match the
	/// channel axis, or the composed elementwise ops fail.
	pub fn forward(&self, input: &Matrix) -> Result<Matrix> {
		if self.num_slope_parameters > 1 {
			let channels = input.shape().get(1).copied().unwrap_or(0);
			if channels != self.num_slope_parameters {
				return Err(Error::invalid_argument(format!(
					"prelu has {} slopes for channel axis {}; found {:?}",
					self.num_slope_parameters,
					channels,
					input.shape(),
				)));
			}
		}
		let slope_shape = if self.num_slope_parameters == 1 {
			vec![1_usize]
		} else {
			let mut shape = vec![1_usize; input.shape().len()];
			shape[1] = self.num_slope_parameters;
			shape
		};
		autograd::record_parameter_leaf(&self.slope)?;
		let slope = self.slope.data().reshape(slope_shape)?;
		let negative = relu(&matrix::neg(input)?)?;
		matrix::sub(&relu(input)?, &matrix::mul(&slope, &negative)?)
	}
}

impl Module for Prelu {
	fn forward(&self, input: &Matrix) -> Result<Matrix> {
		Prelu::forward(self, input)
	}

	fn registry(&self) -> &ModuleRegistry {
		&self.registry
	}
}

/// Exponential linear unit: `x` for `x > 0`, `alpha * (e^x - 1)` otherwise.
///
/// Stateless composed contract: `relu(x) + alpha * (exp(-relu(-x)) - 1)`,
/// so autograd flows through the elementwise adjoints.
pub struct Elu {
	alpha: f32,
	registry: ModuleRegistry,
}

impl Elu {
	/// Construct with the `alpha` negative-range scale (torch default `1.0`).
	///
	/// # Errors
	///
	/// Returns an error when `alpha` is not finite.
	pub fn new(alpha: f32) -> Result<Self> {
		if !alpha.is_finite() {
			return Err(Error::invalid_argument("elu alpha must be finite"));
		}
		Ok(Self {
			alpha,
			registry: ModuleRegistry::new(),
		})
	}

	/// Return the configured `alpha`.
	pub const fn alpha(&self) -> f32 {
		self.alpha
	}

	/// Apply the composed ELU formula.
	///
	/// # Errors
	///
	/// Returns an error when the composed elementwise ops fail.
	pub fn forward(&self, input: &Matrix) -> Result<Matrix> {
		let negative = relu(&matrix::neg(input)?)?;
		let branch = matrix::scale(
			&matrix::sub_scalar(&matrix::exp(&matrix::neg(&negative)?)?, 1.0)?,
			self.alpha,
		)?;
		matrix::add(&relu(input)?, &branch)
	}
}

impl Module for Elu {
	fn forward(&self, input: &Matrix) -> Result<Matrix> {
		Elu::forward(self, input)
	}

	fn registry(&self) -> &ModuleRegistry {
		&self.registry
	}
}
