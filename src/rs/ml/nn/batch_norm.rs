use crate::{DType, Engine, Error, Matrix, Result};

use super::super::{Module, ModuleRegistry, NamedBuffer, Parameter, autograd, matrix};

/// Trainable FP32 BatchNorm over channels of NCHW matrices.
pub struct BatchNorm2d {
	num_features: usize,
	epsilon: f32,
	momentum: f32,
	weight: Parameter,
	bias: Parameter,
	running_mean: NamedBuffer,
	running_variance: NamedBuffer,
	registry: ModuleRegistry,
}

impl BatchNorm2d {
	/// Construct BatchNorm2d with epsilon `1e-5` and momentum `0.1`.
	///
	/// # Errors
	///
	/// Returns an error when `num_features` is zero or state allocation fails.
	pub fn new(engine: &Engine, num_features: usize) -> Result<Self> {
		Self::with_options(engine, num_features, 1e-5, 0.1)
	}

	/// Construct BatchNorm2d with explicit numerical and running-state options.
	///
	/// # Errors
	///
	/// Returns an error when `num_features` is zero, epsilon is not finite and
	/// positive, momentum is not finite and in `[0, 1]`, or state allocation
	/// fails.
	pub fn with_options(
		engine: &Engine,
		num_features: usize,
		epsilon: f32,
		momentum: f32,
	) -> Result<Self> {
		if num_features == 0 {
			return Err(Error::invalid_argument(
				"BatchNorm2d feature count must be nonzero",
			));
		}
		let weight = Matrix::from_f32(engine, [num_features], &vec![1.0; num_features])?;
		let bias = Matrix::from_f32(engine, [num_features], &vec![0.0; num_features])?;
		let running_mean = Matrix::from_f32(engine, [num_features], &vec![0.0; num_features])?;
		let running_variance = Matrix::from_f32(engine, [num_features], &vec![1.0; num_features])?;
		Self::from_matrices(
			weight,
			bias,
			running_mean,
			running_variance,
			epsilon,
			momentum,
		)
	}

	/// Construct BatchNorm2d from exact affine parameters and running state.
	///
	/// # Errors
	///
	/// Returns an error unless all four inputs are equal, nonempty, same-engine
	/// F32 vectors and the numerical options satisfy [`Self::with_options`].
	pub fn from_matrices(
		weight: Matrix,
		bias: Matrix,
		running_mean: Matrix,
		running_variance: Matrix,
		epsilon: f32,
		momentum: f32,
	) -> Result<Self> {
		let [num_features] = weight.shape() else {
			return Err(Error::invalid_argument(
				"BatchNorm2d weight must have rank one",
			));
		};
		if *num_features == 0
			|| bias.shape() != [*num_features]
			|| running_mean.shape() != [*num_features]
			|| running_variance.shape() != [*num_features]
			|| [
				weight.dtype(),
				bias.dtype(),
				running_mean.dtype(),
				running_variance.dtype(),
			]
			.into_iter()
			.any(|dtype| dtype != DType::F32)
			|| !weight.engine_handle().same_as(bias.engine_handle())
			|| !weight.engine_handle().same_as(running_mean.engine_handle())
			|| !weight
				.engine_handle()
				.same_as(running_variance.engine_handle())
			|| !epsilon.is_finite()
			|| epsilon <= 0.0
			|| !momentum.is_finite()
			|| !(0.0..=1.0).contains(&momentum)
		{
			return Err(Error::invalid_argument(
				"BatchNorm2d requires equal nonempty same-engine F32 state vectors, finite positive epsilon, and momentum in [0, 1]",
			));
		}
		let num_features = *num_features;
		let weight = Parameter::new("weight", weight)?;
		let bias = Parameter::new("bias", bias)?;
		let mut registry = ModuleRegistry::new();
		registry.register_parameter("weight", weight.clone())?;
		registry.register_parameter("bias", bias.clone())?;
		registry.register_buffer("running_mean", running_mean, true)?;
		registry.register_buffer("running_variance", running_variance, true)?;
		let running_mean = registry
			.buffer_handle("running_mean")
			.ok_or_else(|| Error::internal("BatchNorm2d running mean registration was lost"))?;
		let running_variance = registry
			.buffer_handle("running_variance")
			.ok_or_else(|| Error::internal("BatchNorm2d running variance registration was lost"))?;
		Ok(Self {
			num_features,
			epsilon,
			momentum,
			weight,
			bias,
			running_mean,
			running_variance,
			registry,
		})
	}

	/// Normalize one NCHW Matrix according to the module's current mode.
	///
	/// Training computes batch statistics and functionally advances both running
	/// buffers. Evaluation reads the current running statistics without changing
	/// them. This records GPU work but does not submit or wait.
	///
	/// # Errors
	///
	/// Returns an error when the input contract, state, or runtime recording fails.
	pub fn forward(&self, input: &Matrix) -> Result<Matrix> {
		let (weight_value, weight_version, weight_requires_grad) = self.weight.snapshot();
		let (bias_value, bias_version, bias_requires_grad) = self.bias.snapshot();
		let training = self.is_training();
		let (output, mean, variance) = if training {
			let result =
				matrix::batch_norm_2d_forward(input, &weight_value, &bias_value, self.epsilon)?;
			let running_mean = self.running_mean.data();
			let running_variance = self.running_variance.data();
			let (updated_mean, updated_variance) = matrix::batch_norm_2d_running_update(
				&running_mean,
				&running_variance,
				&result.mean,
				&result.variance,
				self.momentum,
			)?;
			self.running_mean.replace_data(updated_mean)?;
			self.running_variance.replace_data(updated_variance)?;
			(result.output, result.mean, result.variance)
		} else {
			let mean = self.running_mean.data();
			let variance = self.running_variance.data();
			let output = matrix::batch_norm_2d_with_stats_forward(
				input,
				&mean,
				&variance,
				&weight_value,
				&bias_value,
				self.epsilon,
			)?;
			(output, mean, variance)
		};
		autograd::record_batch_norm_2d(
			input,
			&output,
			mean,
			variance,
			weight_requires_grad.then(|| (self.weight.clone(), weight_version)),
			weight_value,
			bias_requires_grad.then(|| (self.bias.clone(), bias_version)),
			bias_value,
			self.epsilon,
			training,
		)?;
		Ok(output)
	}

	/// Return the number of normalized channels.
	pub const fn num_features(&self) -> usize {
		self.num_features
	}

	/// Return the numerical-stability epsilon.
	pub const fn epsilon(&self) -> f32 {
		self.epsilon
	}

	/// Return the running-statistics update momentum.
	pub const fn momentum(&self) -> f32 {
		self.momentum
	}

	/// Return the stable affine weight parameter.
	pub fn weight(&self) -> Parameter {
		self.weight.clone()
	}

	/// Return the stable affine bias parameter.
	pub fn bias(&self) -> Parameter {
		self.bias.clone()
	}

	/// Return a handle to the current running channel mean.
	pub fn running_mean(&self) -> Matrix {
		self.running_mean.data()
	}

	/// Return a handle to the current running channel variance.
	pub fn running_variance(&self) -> Matrix {
		self.running_variance.data()
	}

	/// Return the affine parameters in deterministic weight, bias order.
	pub fn parameters(&self) -> [Parameter; 2] {
		[self.weight.clone(), self.bias.clone()]
	}
}

impl Module for BatchNorm2d {
	fn forward(&self, input: &Matrix) -> Result<Matrix> {
		BatchNorm2d::forward(self, input)
	}

	fn registry(&self) -> &ModuleRegistry {
		&self.registry
	}
}
