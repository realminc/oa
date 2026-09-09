use crate::{Engine, Error, Matrix, Result};

use super::super::{Module, ModuleRegistry, Parameter, autograd, kernels, random};

/// Trainable FP32 affine projection using `[output, input]` weight layout.
pub struct Linear {
	input_features: usize,
	output_features: usize,
	weight: Parameter,
	bias: Parameter,
	registry: ModuleRegistry,
}

impl Linear {
	/// Construct a deterministically initialized linear layer.
	///
	/// `seed` controls an independent SplitMix64 Xavier-uniform initializer.
	///
	/// # Errors
	///
	/// Returns an error when either feature count is zero, shape arithmetic
	/// overflows, or parameter allocation/upload fails.
	pub fn with_seed(
		engine: &Engine,
		input_features: usize,
		output_features: usize,
		seed: u64,
	) -> Result<Self> {
		if input_features == 0 || output_features == 0 {
			return Err(Error::invalid_argument(
				"linear feature counts must be nonzero",
			));
		}
		let weight_count = input_features
			.checked_mul(output_features)
			.ok_or_else(|| Error::invalid_argument("linear weight size overflows usize"))?;
		let denominator = input_features
			.checked_add(output_features)
			.ok_or_else(|| Error::invalid_argument("linear Xavier extent overflows usize"))?;
		let limit = (6.0_f32 / denominator as f32).sqrt();
		let weights = random::symmetric_uniform(weight_count, limit, seed);
		let weight = Matrix::from_f32(engine, [output_features, input_features], &weights)?;
		let bias = Matrix::from_f32(engine, [output_features], &vec![0.0; output_features])?;
		Self::from_matrices(weight, bias)
	}

	/// Construct a linear layer from exact FP32 weight and bias matrices.
	///
	/// # Errors
	///
	/// Returns an error unless weight has shape `[O, I]`, bias has shape `[O]`,
	/// both are nonempty FP32 matrices, and both belong to the same engine.
	pub fn from_matrices(weight: Matrix, bias: Matrix) -> Result<Self> {
		let [output_features, input_features] = weight.shape() else {
			return Err(Error::invalid_argument("linear weight must have rank two"));
		};
		if *input_features == 0
			|| *output_features == 0
			|| bias.shape() != [*output_features]
			|| weight.dtype() != crate::DType::F32
			|| bias.dtype() != crate::DType::F32
			|| !weight.engine_handle().same_as(bias.engine_handle())
		{
			return Err(Error::invalid_argument(
				"linear requires nonempty same-engine FP32 weight [O, I] and bias [O]",
			));
		}
		let input_features = *input_features;
		let output_features = *output_features;
		let weight = Parameter::new("weight", weight)?;
		let bias = Parameter::new("bias", bias)?;
		let mut registry = ModuleRegistry::new();
		registry.register_parameter("weight", weight.clone())?;
		registry.register_parameter("bias", bias.clone())?;
		Ok(Self {
			input_features,
			output_features,
			weight,
			bias,
			registry,
		})
	}

	/// Apply this layer to a rank-two FP32 matrix.
	///
	/// # Errors
	///
	/// Returns an error when the input contract or runtime recording fails.
	pub fn forward(&self, input: &Matrix) -> Result<Matrix> {
		let (weight, weight_version, weight_requires_grad) = self.weight.snapshot();
		let (bias, bias_version, bias_requires_grad) = self.bias.snapshot();
		let output = kernels::linear(input, &weight, &bias)?;
		if weight_requires_grad || bias_requires_grad {
			autograd::record_linear(
				input,
				&output,
				self.weight.clone(),
				weight,
				weight_version,
				self.bias.clone(),
				bias_version,
			)?;
		}
		Ok(output)
	}

	/// Return this layer's input feature count.
	pub const fn input_features(&self) -> usize {
		self.input_features
	}

	/// Return this layer's output feature count.
	pub const fn output_features(&self) -> usize {
		self.output_features
	}

	/// Return the stable trainable weight handle.
	pub fn weight(&self) -> Parameter {
		self.weight.clone()
	}

	/// Return the stable trainable bias handle.
	pub fn bias(&self) -> Parameter {
		self.bias.clone()
	}

	/// Return this layer's parameters in deterministic weight, bias order.
	pub fn parameters(&self) -> [Parameter; 2] {
		[self.weight.clone(), self.bias.clone()]
	}
}

impl Module for Linear {
	fn forward(&self, input: &Matrix) -> Result<Matrix> {
		Linear::forward(self, input)
	}

	fn registry(&self) -> &ModuleRegistry {
		&self.registry
	}
}
