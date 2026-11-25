//! Inference-only linear layer over a block-quantized weight.

use crate::ml::matrix::{self as ml_matrix, QuantMatrix, Quantization};
use crate::{Error, Matrix, Result, matrix};

use super::super::module::{Module, ModuleRegistry};

/// Affine map `x @ W^T + b` whose weight stays packed in a [`QuantMatrix`].
///
/// `QuantizedLinear` is the model-level quantized-inference contract: the
/// weight is frozen at quantization time, exposes no trainable parameters,
/// and the forward pass dequantizes inside the dispatch rather than
/// materializing an F32 weight on the device. State-dict round-trips run the
/// standard quantize-on-export flow — checkpoints carry the F32 module and
/// `from_linear` re-derives the packed planes.
pub struct QuantizedLinear {
	input_features: usize,
	output_features: usize,
	weight: QuantMatrix,
	bias: Option<Matrix>,
	registry: ModuleRegistry,
}

impl QuantizedLinear {
	/// Quantize an existing [`super::Linear`]'s weight into `quantization`.
	///
	/// The bias, when the source module carries one, moves into this module
	/// unchanged as a persistent buffer.
	///
	/// # Errors
	///
	/// Returns an error when the weight fails the [`crate::ml::matrix::quantize`]
	/// contract or buffer registration fails.
	pub fn from_linear(linear: &super::Linear, quantization: Quantization) -> Result<Self> {
		let weight = ml_matrix::quantize(&linear.weight().data(), quantization)?;
		let bias = linear.bias().map(|bias| bias.data());
		Self::from_quantized(weight, bias)
	}

	/// Wrap an already-quantized `[N, K]` weight and optional `[N]` bias.
	///
	/// # Errors
	///
	/// Returns an error when the weight is not rank 2, the bias is not `[N]`,
	/// or buffer registration fails.
	pub fn from_quantized(weight: QuantMatrix, bias: Option<Matrix>) -> Result<Self> {
		if weight.shape().len() != 2 {
			return Err(Error::invalid_argument(
				"QuantizedLinear weight must have rank 2 [N,K]",
			));
		}
		let output_features = weight.shape()[0];
		let input_features = weight.shape()[1];
		if let Some(bias) = &bias
			&& bias.shape() != [output_features]
		{
			return Err(Error::invalid_argument(format!(
				"QuantizedLinear bias must have shape [{output_features}]; found {:?}",
				bias.shape(),
			)));
		}
		let mut registry = ModuleRegistry::new();
		if let Some(bias) = &bias {
			registry.register_buffer("bias", bias.clone(), true)?;
		}
		Ok(Self {
			input_features,
			output_features,
			weight,
			bias,
			registry,
		})
	}

	/// Return the accepted input feature count `K`.
	pub const fn input_features(&self) -> usize {
		self.input_features
	}

	/// Return the produced output feature count `N`.
	pub const fn output_features(&self) -> usize {
		self.output_features
	}

	/// Return the packed weight format.
	pub const fn quantization(&self) -> Quantization {
		self.weight.quantization()
	}

	/// Return the packed weight value.
	pub const fn weight(&self) -> &QuantMatrix {
		&self.weight
	}

	/// Return the optional additive bias, when present.
	pub const fn bias(&self) -> Option<&Matrix> {
		self.bias.as_ref()
	}
}

impl Module for QuantizedLinear {
	fn forward(&self, input: &Matrix) -> Result<Matrix> {
		if input.shape().last() != Some(&self.input_features) {
			return Err(Error::invalid_argument(format!(
				"QuantizedLinear requires input last dimension {}; found {:?}",
				self.input_features,
				input.shape(),
			)));
		}
		let output = ml_matrix::matmul(input, &self.weight)?;
		if let Some(bias) = &self.bias {
			return matrix::add(&output, bias);
		}
		Ok(output)
	}

	fn registry(&self) -> &ModuleRegistry {
		&self.registry
	}
}
