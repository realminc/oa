//! Bilinear layer (`nn.Bilinear`): `y = x1^T W_o x2 + b` per output feature.

use crate::{Engine, Error, Matrix, Result, matrix};

use super::super::{Module, ModuleRegistry, Parameter, random};

/// Trainable bilinear interaction `out[n, o] = sum_i,j x1[n,i] W[o,i,j] x2[n,j] + b[o]`.
///
/// The weight uses the donor `[O, I1, I2]` layout and Xavier-uniform
/// initialization matching `nn.Linear`.
pub struct Bilinear {
	in1_features: usize,
	in2_features: usize,
	out_features: usize,
	weight: Parameter,
	bias: Option<Parameter>,
	registry: ModuleRegistry,
}

impl Bilinear {
	/// Construct on the current engine with the thread-local seed source.
	///
	/// # Errors
	///
	/// Returns an error when a feature count is zero or initialization fails.
	pub fn new(in1_features: usize, in2_features: usize, out_features: usize) -> Result<Self> {
		Self::new_with_seed(in1_features, in2_features, out_features, true, 0)
	}

	/// Construct on the current engine with an explicit seed.
	///
	/// # Errors
	///
	/// Returns an error when a feature count is zero or initialization fails.
	pub fn new_with_seed(
		in1_features: usize,
		in2_features: usize,
		out_features: usize,
		trainable_bias: bool,
		seed: u64,
	) -> Result<Self> {
		Self::new_with_seed_on(
			&Engine::current()?,
			in1_features,
			in2_features,
			out_features,
			trainable_bias,
			seed,
		)
	}

	/// Construct on `engine` with an explicit seed.
	///
	/// # Errors
	///
	/// Returns an error when a feature count is zero or initialization fails.
	pub fn new_with_seed_on(
		engine: &Engine,
		in1_features: usize,
		in2_features: usize,
		out_features: usize,
		trainable_bias: bool,
		seed: u64,
	) -> Result<Self> {
		if in1_features == 0 || in2_features == 0 || out_features == 0 {
			return Err(Error::invalid_argument(
				"bilinear feature counts must be nonzero",
			));
		}
		let weight_count = in1_features
			.checked_mul(in2_features)
			.and_then(|count| count.checked_mul(out_features))
			.ok_or_else(|| Error::invalid_argument("bilinear weight size overflows usize"))?;
		let denominator = in1_features
			.checked_add(in2_features)
			.ok_or_else(|| Error::invalid_argument("bilinear Xavier extent overflows usize"))?;
		let limit = (6.0_f32 / denominator as f32).sqrt();
		let weights = random::symmetric_uniform(weight_count, limit, seed);
		let weight =
			Matrix::new_from_f32_on(engine, [out_features, in1_features, in2_features], &weights)?;
		let bias = Matrix::new_from_f32_on(engine, [out_features], &vec![0.0; out_features])?;
		Self::new_from_parts(weight, bias, trainable_bias)
	}

	/// Construct from exact FP32 weight and bias matrices.
	///
	/// # Errors
	///
	/// Returns an error unless weight is `[O, I1, I2]` and bias is `[O]` on the
	/// same engine.
	pub fn new_from_matrices(weight: Matrix, bias: Matrix) -> Result<Self> {
		let [out_features, in1_features, in2_features] = weight.shape().to_vec()[..] else {
			return Err(Error::invalid_argument(format!(
				"bilinear weight must be [O, I1, I2]; found {:?}",
				weight.shape()
			)));
		};
		if bias.shape() != [out_features] || !bias.engine_handle().same_as(weight.engine_handle()) {
			return Err(Error::invalid_argument(format!(
				"bilinear bias must be [O] on the weight engine; found {:?}",
				bias.shape()
			)));
		}
		Self::new_from_parts_unchecked(weight, bias, true, in1_features, in2_features, out_features)
	}

	fn new_from_parts(weight: Matrix, bias: Matrix, trainable_bias: bool) -> Result<Self> {
		let [out_features, in1_features, in2_features] = weight.shape().to_vec()[..] else {
			return Err(Error::invalid_argument(format!(
				"bilinear weight must be [O, I1, I2]; found {:?}",
				weight.shape()
			)));
		};
		Self::new_from_parts_unchecked(
			weight,
			bias,
			trainable_bias,
			in1_features,
			in2_features,
			out_features,
		)
	}

	fn new_from_parts_unchecked(
		weight: Matrix,
		bias: Matrix,
		trainable_bias: bool,
		in1_features: usize,
		in2_features: usize,
		out_features: usize,
	) -> Result<Self> {
		let weight = Parameter::new("weight", weight)?;
		let bias = if trainable_bias {
			Some(Parameter::new("bias", bias)?)
		} else {
			None
		};
		let mut registry = ModuleRegistry::new();
		registry.register_parameter("weight", weight.clone())?;
		if let Some(parameter) = &bias {
			registry.register_parameter("bias", parameter.clone())?;
		}
		Ok(Self {
			in1_features,
			in2_features,
			out_features,
			weight,
			bias,
			registry,
		})
	}

	/// Return the `[O, I1, I2]` trainable weight.
	pub fn weight(&self) -> Parameter {
		self.weight.clone()
	}

	/// Return the `[O]` trainable bias when present.
	pub fn bias(&self) -> Option<Parameter> {
		self.bias.clone()
	}

	/// Apply the bilinear map to `[N, I1]` and `[N, I2]` inputs.
	///
	/// # Errors
	///
	/// Returns an error for mismatched ranks, feature extents, batch counts, or
	/// engines, or when the composed dispatch fails.
	pub fn forward2(&self, input1: &Matrix, input2: &Matrix) -> Result<Matrix> {
		let operation = "Bilinear";
		for (input, features) in [(input1, self.in1_features), (input2, self.in2_features)] {
			if input.shape().len() != 2
				|| input.shape()[1] != features
				|| !input
					.engine_handle()
					.same_as(self.weight.data().engine_handle())
			{
				return Err(Error::invalid_argument(format!(
					"{operation} requires same-engine [N, {features}] inputs; found {:?}",
					input.shape()
				)));
			}
		}
		if input1.shape()[0] != input2.shape()[0] {
			return Err(Error::invalid_argument(format!(
				"{operation} batch extents must match; found {:?} and {:?}",
				input1.shape(),
				input2.shape()
			)));
		}
		let batch = input1.shape()[0];
		let outer = matrix::mul(
			&input1.reshape([batch, self.in1_features, 1])?,
			&input2.reshape([batch, 1, self.in2_features])?,
		)?
		.reshape([batch, self.in1_features * self.in2_features])?;
		let flat = self
			.weight
			.data()
			.reshape([self.out_features, self.in1_features * self.in2_features])?;
		let mut output = matrix::mat_mul_nt(&outer, &flat)?;
		if let Some(bias) = &self.bias {
			output = matrix::add(&output, &bias.data())?;
		}
		Ok(output)
	}
}

impl Module for Bilinear {
	fn forward(&self, input: &Matrix) -> Result<Matrix> {
		let _ = input;
		Err(Error::invalid_argument(
			"Bilinear requires two inputs; call forward2(input1, input2)",
		))
	}

	fn registry(&self) -> &ModuleRegistry {
		&self.registry
	}
}
