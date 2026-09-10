use crate::{DType, Engine, Error, Matrix, Result};

use super::super::{Module, ModuleRegistry, Parameter, autograd, lowering::matrix as dispatch};

/// Trainable FP32 RMSNorm over the final input dimension.
pub struct RmsNorm {
	dimension: usize,
	epsilon: f32,
	weight: Parameter,
	registry: ModuleRegistry,
}

impl RmsNorm {
	/// Construct RMSNorm with unit weight.
	///
	/// # Errors
	///
	/// Returns an error when `dimension` is zero, `epsilon` is not finite and
	/// positive, or parameter allocation/upload fails.
	pub fn new(engine: &Engine, dimension: usize, epsilon: f32) -> Result<Self> {
		if dimension == 0 {
			return Err(Error::invalid_argument("RMSNorm dimension must be nonzero"));
		}
		let weight = Matrix::from_f32(engine, [dimension], &vec![1.0; dimension])?;
		Self::from_matrix(weight, epsilon)
	}

	/// Construct RMSNorm from an exact weight Matrix.
	///
	/// # Errors
	///
	/// Returns an error unless weight is a nonempty F32 vector and `epsilon` is
	/// finite and positive.
	pub fn from_matrix(weight: Matrix, epsilon: f32) -> Result<Self> {
		let [dimension] = weight.shape() else {
			return Err(Error::invalid_argument("RMSNorm weight must be rank one"));
		};
		if *dimension == 0 || weight.dtype() != DType::F32 || !epsilon.is_finite() || epsilon <= 0.0
		{
			return Err(Error::invalid_argument(
				"RMSNorm requires a nonempty F32 weight vector and finite positive epsilon",
			));
		}
		let dimension = *dimension;
		let weight = Parameter::new("weight", weight)?;
		let mut registry = ModuleRegistry::new();
		registry.register_parameter("weight", weight.clone())?;
		Ok(Self {
			dimension,
			epsilon,
			weight,
			registry,
		})
	}

	/// Normalize each row by its root mean square and apply weight.
	///
	/// # Errors
	///
	/// Returns an error for an incompatible input or runtime recording failure.
	pub fn forward(&self, input: &Matrix) -> Result<Matrix> {
		let (weight, version, requires_grad) = self.weight.snapshot();
		let output = dispatch::rms_norm(input, &weight, self.epsilon)?;
		let parameter = requires_grad.then(|| (self.weight.clone(), version));
		autograd::record_rms_norm(input, &output, parameter, weight, self.epsilon)?;
		Ok(output)
	}

	/// Return the normalized final-dimension width.
	pub const fn dimension(&self) -> usize {
		self.dimension
	}

	/// Return the numerical-stability epsilon.
	pub const fn epsilon(&self) -> f32 {
		self.epsilon
	}

	/// Return the stable trainable weight handle.
	pub fn weight(&self) -> Parameter {
		self.weight.clone()
	}

	/// Return the one registered parameter.
	pub fn parameters(&self) -> [Parameter; 1] {
		[self.weight.clone()]
	}
}

impl Module for RmsNorm {
	fn forward(&self, input: &Matrix) -> Result<Matrix> {
		RmsNorm::forward(self, input)
	}

	fn registry(&self) -> &ModuleRegistry {
		&self.registry
	}
}
