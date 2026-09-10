use crate::{DType, Engine, Error, Matrix, Result};

use super::super::{Module, ModuleRegistry, Parameter, autograd, lowering::matrix as dispatch};

/// Trainable FP32 LayerNorm over the final input dimension.
pub struct LayerNorm {
	normalized_shape: usize,
	epsilon: f32,
	weight: Parameter,
	bias: Parameter,
	registry: ModuleRegistry,
}

impl LayerNorm {
	/// Construct a LayerNorm with unit weight and zero bias.
	///
	/// # Errors
	///
	/// Returns an error when `normalized_shape` is zero, `epsilon` is not finite
	/// and positive, or parameter allocation/upload fails.
	pub fn new(engine: &Engine, normalized_shape: usize, epsilon: f32) -> Result<Self> {
		if normalized_shape == 0 {
			return Err(Error::invalid_argument(
				"layer norm normalized shape must be nonzero",
			));
		}
		if !epsilon.is_finite() || epsilon <= 0.0 {
			return Err(Error::invalid_argument(
				"layer norm epsilon must be finite and positive",
			));
		}
		let weight = Matrix::from_f32(engine, [normalized_shape], &vec![1.0; normalized_shape])?;
		let bias = Matrix::from_f32(engine, [normalized_shape], &vec![0.0; normalized_shape])?;
		Self::from_matrices(weight, bias, epsilon)
	}

	/// Construct a LayerNorm from exact weight and bias matrices.
	///
	/// # Errors
	///
	/// Returns an error unless weight and bias are equal, nonempty, same-engine
	/// FP32 vectors and `epsilon` is finite and positive.
	pub fn from_matrices(weight: Matrix, bias: Matrix, epsilon: f32) -> Result<Self> {
		let [normalized_shape] = weight.shape() else {
			return Err(Error::invalid_argument(
				"layer norm weight must be rank one",
			));
		};
		if *normalized_shape == 0
			|| bias.shape() != [*normalized_shape]
			|| weight.dtype() != DType::F32
			|| bias.dtype() != DType::F32
			|| !weight.engine_handle().same_as(bias.engine_handle())
			|| !epsilon.is_finite()
			|| epsilon <= 0.0
		{
			return Err(Error::invalid_argument(
				"layer norm requires nonempty same-engine FP32 weight/bias vectors and finite positive epsilon",
			));
		}
		let normalized_shape = *normalized_shape;
		let weight = Parameter::new("weight", weight)?;
		let bias = Parameter::new("bias", bias)?;
		let mut registry = ModuleRegistry::new();
		registry.register_parameter("weight", weight.clone())?;
		registry.register_parameter("bias", bias.clone())?;
		Ok(Self {
			normalized_shape,
			epsilon,
			weight,
			bias,
			registry,
		})
	}

	/// Normalize each row over the final dimension and apply affine parameters.
	///
	/// # Errors
	///
	/// Returns an error when the input is empty, is not FP32, its final dimension
	/// differs from the normalized shape, it belongs to another engine, or runtime
	/// recording fails.
	pub fn forward(&self, input: &Matrix) -> Result<Matrix> {
		let (weight, weight_version, weight_requires_grad) = self.weight.snapshot();
		let (bias, bias_version, bias_requires_grad) = self.bias.snapshot();
		let result = dispatch::layer_norm(input, &weight, &bias, self.epsilon)?;
		if weight_requires_grad || bias_requires_grad {
			autograd::record_layer_norm(
				input,
				&result.output,
				result.normalized,
				result.inverse_stddev,
				self.weight.clone(),
				weight,
				weight_version,
				self.bias.clone(),
				bias_version,
			)?;
		}
		Ok(result.output)
	}

	/// Return the normalized final-dimension width.
	pub const fn normalized_shape(&self) -> usize {
		self.normalized_shape
	}

	/// Return the numerical-stability epsilon.
	pub const fn epsilon(&self) -> f32 {
		self.epsilon
	}

	/// Return the stable trainable weight handle.
	pub fn weight(&self) -> Parameter {
		self.weight.clone()
	}

	/// Return the stable trainable bias handle.
	pub fn bias(&self) -> Parameter {
		self.bias.clone()
	}

	/// Return the parameters in deterministic weight, bias order.
	pub fn parameters(&self) -> [Parameter; 2] {
		[self.weight.clone(), self.bias.clone()]
	}
}

impl Module for LayerNorm {
	fn forward(&self, input: &Matrix) -> Result<Matrix> {
		LayerNorm::forward(self, input)
	}

	fn registry(&self) -> &ModuleRegistry {
		&self.registry
	}
}
