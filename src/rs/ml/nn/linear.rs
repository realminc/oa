use crate::{Engine, Error, Matrix, Result};

use super::super::{
	Module, ModuleRegistry, Parameter, autograd, lowering::matrix as dispatch, random,
};

/// Trainable FP32 affine projection using `[output, input]` weight layout.
pub struct Linear {
	input_features: usize,
	output_features: usize,
	weight: Parameter,
	bias: LinearBias,
	registry: ModuleRegistry,
}

enum LinearBias {
	Parameter(Parameter),
	Zero(Matrix),
}

impl LinearBias {
	fn snapshot(&self) -> (Matrix, Option<(Parameter, u64)>, bool) {
		match self {
			Self::Parameter(parameter) => {
				let (value, version, requires_grad) = parameter.snapshot();
				(
					value,
					requires_grad.then(|| (parameter.clone(), version)),
					requires_grad,
				)
			}
			Self::Zero(value) => (value.clone(), None, false),
		}
	}
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
		Self::with_seed_and_bias(engine, input_features, output_features, true, seed)
	}

	/// Construct a deterministically initialized linear layer with optional bias.
	///
	/// A bias-free layer retains a private zero vector required by the current
	/// physical GEMM ABI, but exposes and registers no fake trainable parameter.
	///
	/// # Errors
	///
	/// Returns an error when either feature count is zero, shape arithmetic
	/// overflows, or parameter allocation/upload fails.
	pub fn with_seed_and_bias(
		engine: &Engine,
		input_features: usize,
		output_features: usize,
		trainable_bias: bool,
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
		Self::from_parts(weight, bias, trainable_bias)
	}

	/// Construct a linear layer from exact FP32 weight and bias matrices.
	///
	/// # Errors
	///
	/// Returns an error unless weight has shape `[O, I]`, bias has shape `[O]`,
	/// both are nonempty FP32 matrices, and both belong to the same engine.
	pub fn from_matrices(weight: Matrix, bias: Matrix) -> Result<Self> {
		Self::from_parts(weight, bias, true)
	}

	/// Construct a bias-free linear layer from an exact FP32 weight matrix.
	///
	/// # Errors
	///
	/// Returns an error unless weight is a nonempty FP32 `[O, I]` matrix or the
	/// private physical zero vector cannot be allocated.
	pub fn from_weight(weight: Matrix) -> Result<Self> {
		let [output_features, input_features] = weight.shape() else {
			return Err(Error::invalid_argument("linear weight must have rank two"));
		};
		if *input_features == 0 || *output_features == 0 || weight.dtype() != crate::DType::F32 {
			return Err(Error::invalid_argument(
				"bias-free linear requires nonempty FP32 weight [O, I]",
			));
		}
		let output_features = *output_features;
		let zero = Matrix::allocate(
			weight.engine_handle(),
			vec![output_features],
			output_features,
			crate::DType::F32,
		)?;
		Self::from_parts(weight, zero, false)
	}

	fn from_parts(weight: Matrix, bias: Matrix, trainable_bias: bool) -> Result<Self> {
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
		let bias = if trainable_bias {
			LinearBias::Parameter(Parameter::new("bias", bias)?)
		} else {
			LinearBias::Zero(bias)
		};
		let mut registry = ModuleRegistry::new();
		registry.register_parameter("weight", weight.clone())?;
		if let LinearBias::Parameter(parameter) = &bias {
			registry.register_parameter("bias", parameter.clone())?;
		}
		Ok(Self {
			input_features,
			output_features,
			weight,
			bias,
			registry,
		})
	}

	/// Apply this layer to an FP32 Matrix with rank at least two.
	///
	/// Leading dimensions are preserved and the final input-feature dimension
	/// is replaced by the configured output-feature dimension.
	///
	/// # Errors
	///
	/// Returns an error when the input contract or runtime recording fails.
	pub fn forward(&self, input: &Matrix) -> Result<Matrix> {
		let (weight, weight_version, weight_requires_grad) = self.weight.snapshot();
		let (bias, bias_parameter, bias_requires_grad) = self.bias.snapshot();
		let output = dispatch::linear(input, &weight, &bias)?;
		if weight_requires_grad || bias_requires_grad {
			autograd::record_linear(
				input,
				&output,
				self.weight.clone(),
				weight,
				weight_version,
				bias_parameter,
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
	pub fn bias(&self) -> Option<Parameter> {
		match &self.bias {
			LinearBias::Parameter(parameter) => Some(parameter.clone()),
			LinearBias::Zero(_) => None,
		}
	}

	/// Return whether this layer owns a trainable bias.
	pub fn has_bias(&self) -> bool {
		matches!(self.bias, LinearBias::Parameter(_))
	}

	/// Return this layer's parameters in deterministic weight, optional-bias order.
	pub fn parameters(&self) -> Vec<Parameter> {
		let mut parameters = vec![self.weight.clone()];
		if let Some(bias) = self.bias() {
			parameters.push(bias);
		}
		parameters
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
