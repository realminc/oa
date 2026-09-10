use crate::{Engine, Error, Matrix, Result};

use super::super::{
	Module, ModuleRegistry, Parameter, autograd, lowering::matrix as dispatch, matrix, random,
};

enum ProjectionBias {
	Parameter(Parameter),
	Zero(Matrix),
}

impl ProjectionBias {
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

/// Donor-compatible SwiGLU MLP with optional projection biases.
pub struct Swiglu {
	input_features: usize,
	intermediate_size: usize,
	weights: [Parameter; 3],
	biases: [ProjectionBias; 3],
	registry: ModuleRegistry,
}

impl Swiglu {
	/// Construct gate, up, and down projections with deterministic Xavier weights.
	///
	/// # Errors
	///
	/// Returns an error when dimensions are zero, shape arithmetic overflows, or
	/// parameter allocation and registration fail.
	pub fn with_seed(
		engine: &Engine,
		input_features: usize,
		intermediate_size: usize,
		bias: bool,
		seed: u64,
	) -> Result<Self> {
		if input_features == 0 || intermediate_size == 0 {
			return Err(Error::invalid_argument(
				"SwiGLU feature counts must be nonzero",
			));
		}
		let gate_weight = xavier_parameter(
			engine,
			"gate_weight",
			input_features,
			intermediate_size,
			seed,
		)?;
		let up_weight = xavier_parameter(
			engine,
			"up_weight",
			input_features,
			intermediate_size,
			seed.wrapping_add(1),
		)?;
		let down_weight = xavier_parameter(
			engine,
			"down_weight",
			intermediate_size,
			input_features,
			seed.wrapping_add(2),
		)?;
		let biases = [
			projection_bias(engine, "gate_bias", intermediate_size, bias)?,
			projection_bias(engine, "up_bias", intermediate_size, bias)?,
			projection_bias(engine, "down_bias", input_features, bias)?,
		];
		let weights = [gate_weight, up_weight, down_weight];
		let mut registry = ModuleRegistry::new();
		for (name, weight) in ["gate_weight", "up_weight", "down_weight"]
			.into_iter()
			.zip(&weights)
		{
			registry.register_parameter(name, weight.clone())?;
		}
		for (name, projection_bias) in ["gate_bias", "up_bias", "down_bias"]
			.into_iter()
			.zip(&biases)
		{
			if let ProjectionBias::Parameter(parameter) = projection_bias {
				registry.register_parameter(name, parameter.clone())?;
			}
		}
		Ok(Self {
			input_features,
			intermediate_size,
			weights,
			biases,
			registry,
		})
	}

	/// Apply gate/up projections, fused SwiGLU activation, and down projection.
	///
	/// Leading dimensions are flattened for projection and restored afterward.
	///
	/// # Errors
	///
	/// Returns an error unless the final F32 input dimension matches the
	/// configured input width, or a projection/recording operation fails.
	pub fn forward(&self, input: &Matrix) -> Result<Matrix> {
		let Some(&width) = input.shape().last() else {
			return Err(Error::invalid_argument(
				"SwiGLU input must have at least one dimension",
			));
		};
		if width != self.input_features {
			return Err(Error::invalid_argument(
				"SwiGLU input final dimension does not match input_features",
			));
		}
		let rows = input.num_elements() / width;
		let flat = if input.shape().len() == 2 {
			input.clone()
		} else {
			input.reshape([rows, width])?
		};
		let gate = project(&flat, &self.weights[0], &self.biases[0])?;
		let up = project(&flat, &self.weights[1], &self.biases[1])?;
		let activated = matrix::swiglu(&gate, &up)?;
		let output = project(&activated, &self.weights[2], &self.biases[2])?;
		if input.shape().len() == 2 {
			return Ok(output);
		}
		let mut output_shape = input.shape().to_vec();
		*output_shape
			.last_mut()
			.expect("nonempty input shape was validated") = self.input_features;
		output.reshape(output_shape)
	}

	/// Return the input and output projection width.
	pub const fn input_features(&self) -> usize {
		self.input_features
	}

	/// Return the gate and up projection width.
	pub const fn intermediate_size(&self) -> usize {
		self.intermediate_size
	}

	/// Return whether the three projection biases are trainable parameters.
	pub fn has_bias(&self) -> bool {
		matches!(&self.biases[0], ProjectionBias::Parameter(_))
	}
}

impl Module for Swiglu {
	fn forward(&self, input: &Matrix) -> Result<Matrix> {
		Swiglu::forward(self, input)
	}

	fn registry(&self) -> &ModuleRegistry {
		&self.registry
	}
}

fn xavier_parameter(
	engine: &Engine,
	name: &'static str,
	input_features: usize,
	output_features: usize,
	seed: u64,
) -> Result<Parameter> {
	let count = input_features
		.checked_mul(output_features)
		.ok_or_else(|| Error::invalid_argument("SwiGLU weight size overflows usize"))?;
	let denominator = input_features
		.checked_add(output_features)
		.ok_or_else(|| Error::invalid_argument("SwiGLU Xavier extent overflows usize"))?;
	let limit = (6.0_f32 / denominator as f32).sqrt();
	Parameter::new(
		name,
		Matrix::from_f32(
			engine,
			[output_features, input_features],
			&random::symmetric_uniform(count, limit, seed),
		)?,
	)
}

fn projection_bias(
	engine: &Engine,
	name: &'static str,
	size: usize,
	trainable: bool,
) -> Result<ProjectionBias> {
	let value = Matrix::from_f32(engine, [size], &vec![0.0; size])?;
	if trainable {
		Ok(ProjectionBias::Parameter(Parameter::new(name, value)?))
	} else {
		Ok(ProjectionBias::Zero(value))
	}
}

fn project(input: &Matrix, weight: &Parameter, bias: &ProjectionBias) -> Result<Matrix> {
	let (weight_value, weight_version, weight_requires_grad) = weight.snapshot();
	let (bias_value, bias_parameter, bias_requires_grad) = bias.snapshot();
	let output = dispatch::linear(input, &weight_value, &bias_value)?;
	if weight_requires_grad || bias_requires_grad {
		autograd::record_linear(
			input,
			&output,
			weight.clone(),
			weight_value,
			weight_version,
			bias_parameter,
		)?;
	}
	Ok(output)
}
