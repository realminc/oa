use std::rc::Rc;

use crate::{DType, Engine, Error, Matrix, Result};

use super::super::{Module, ModuleRegistry, Parameter, autograd, kernels, random};

struct RnnLayer {
	weight_ih: Parameter,
	weight_hh: Parameter,
	bias_ih: Parameter,
	bias_hh: Parameter,
	registry: ModuleRegistry,
}

impl RnnLayer {
	fn with_seed(
		engine: &Engine,
		input_size: usize,
		hidden_size: usize,
		seed: u64,
	) -> Result<Self> {
		let weight_ih_count = hidden_size
			.checked_mul(input_size)
			.ok_or_else(|| Error::invalid_argument("RNN input weight size overflows usize"))?;
		let weight_hh_count = hidden_size
			.checked_mul(hidden_size)
			.ok_or_else(|| Error::invalid_argument("RNN recurrent weight size overflows usize"))?;
		let xavier_extent = input_size
			.checked_add(hidden_size)
			.ok_or_else(|| Error::invalid_argument("RNN Xavier extent overflows usize"))?;
		let input_limit = (6.0_f32 / xavier_extent as f32).sqrt();
		let hidden_limit = (3.0_f32 / hidden_size as f32).sqrt();
		Self::from_matrices(
			Matrix::from_f32(
				engine,
				[hidden_size, input_size],
				&random::symmetric_uniform(weight_ih_count, input_limit, seed),
			)?,
			Matrix::from_f32(
				engine,
				[hidden_size, hidden_size],
				&random::symmetric_uniform(
					weight_hh_count,
					hidden_limit,
					seed.wrapping_add(0x9e37_79b9_7f4a_7c15),
				),
			)?,
			Matrix::from_f32(engine, [hidden_size], &vec![0.0; hidden_size])?,
			Matrix::from_f32(engine, [hidden_size], &vec![0.0; hidden_size])?,
		)
	}

	fn from_matrices(
		weight_ih: Matrix,
		weight_hh: Matrix,
		bias_ih: Matrix,
		bias_hh: Matrix,
	) -> Result<Self> {
		let [hidden_size, input_size] = weight_ih.shape() else {
			return Err(Error::invalid_argument(
				"RNN input weight must have rank two",
			));
		};
		if *hidden_size == 0
			|| *input_size == 0
			|| *hidden_size > 1024
			|| weight_hh.shape() != [*hidden_size, *hidden_size]
			|| bias_ih.shape() != [*hidden_size]
			|| bias_hh.shape() != [*hidden_size]
			|| [
				weight_ih.dtype(),
				weight_hh.dtype(),
				bias_ih.dtype(),
				bias_hh.dtype(),
			] != [DType::F32; 4]
			|| !weight_ih.engine_handle().same_as(weight_hh.engine_handle())
			|| !weight_ih.engine_handle().same_as(bias_ih.engine_handle())
			|| !weight_ih.engine_handle().same_as(bias_hh.engine_handle())
		{
			return Err(Error::invalid_argument(
				"RNN requires same-engine F32 weights [H, I]/[H, H], biases [H], and 1 <= H <= 1024",
			));
		}
		let weight_ih = Parameter::new("weight_ih", weight_ih)?;
		let weight_hh = Parameter::new("weight_hh", weight_hh)?;
		let bias_ih = Parameter::new("bias_ih", bias_ih)?;
		let bias_hh = Parameter::new("bias_hh", bias_hh)?;
		let mut registry = ModuleRegistry::new();
		for (name, parameter) in [
			("weight_ih", &weight_ih),
			("weight_hh", &weight_hh),
			("bias_ih", &bias_ih),
			("bias_hh", &bias_hh),
		] {
			registry.register_parameter(name, parameter.clone())?;
		}
		Ok(Self {
			weight_ih,
			weight_hh,
			bias_ih,
			bias_hh,
			registry,
		})
	}

	fn forward(&self, input: &Matrix) -> Result<Matrix> {
		let (weight_ih, weight_ih_version, weight_ih_grad) = self.weight_ih.snapshot();
		let (weight_hh, weight_hh_version, weight_hh_grad) = self.weight_hh.snapshot();
		let (bias_ih, bias_ih_version, bias_ih_grad) = self.bias_ih.snapshot();
		let (bias_hh, bias_hh_version, bias_hh_grad) = self.bias_hh.snapshot();
		let result = kernels::rnn(input, &weight_ih, &weight_hh, &bias_ih, &bias_hh)?;
		if weight_ih_grad || weight_hh_grad || bias_ih_grad || bias_hh_grad {
			autograd::record_rnn(
				input,
				&result.output,
				result.hidden_previous,
				[
					self.weight_ih.clone(),
					self.weight_hh.clone(),
					self.bias_ih.clone(),
					self.bias_hh.clone(),
				],
				[weight_ih, weight_hh],
				[
					weight_ih_version,
					weight_hh_version,
					bias_ih_version,
					bias_hh_version,
				],
			)?;
		}
		Ok(result.output)
	}

	fn parameters(&self) -> [Parameter; 4] {
		[
			self.weight_ih.clone(),
			self.weight_hh.clone(),
			self.bias_ih.clone(),
			self.bias_hh.clone(),
		]
	}
}

impl Module for RnnLayer {
	fn forward(&self, input: &Matrix) -> Result<Matrix> {
		RnnLayer::forward(self, input)
	}

	fn registry(&self) -> &ModuleRegistry {
		&self.registry
	}
}

/// Stacked, batch-first Elman recurrent network.
///
/// Input uses `[batch, sequence, input]`; output uses
/// `[batch, sequence, hidden]`. Each layer starts from a zero hidden state.
pub struct Rnn {
	input_size: usize,
	hidden_size: usize,
	layers: Vec<Rc<RnnLayer>>,
	registry: ModuleRegistry,
}

impl Rnn {
	/// Construct a deterministically initialized stacked RNN with biases.
	///
	/// # Errors
	///
	/// Returns an error when a dimension or layer count is zero, hidden size
	/// exceeds the current 1024-element scan limit, arithmetic overflows, or
	/// parameter allocation/upload fails.
	pub fn with_seed(
		engine: &Engine,
		input_size: usize,
		hidden_size: usize,
		num_layers: usize,
		seed: u64,
	) -> Result<Self> {
		if input_size == 0 || hidden_size == 0 || hidden_size > 1024 || num_layers == 0 {
			return Err(Error::invalid_argument(
				"RNN requires nonzero input/hidden/layer counts and hidden size <= 1024",
			));
		}
		let mut layers = Vec::with_capacity(num_layers);
		let mut registry = ModuleRegistry::new();
		for index in 0..num_layers {
			let layer_input = if index == 0 { input_size } else { hidden_size };
			let layer = Rc::new(RnnLayer::with_seed(
				engine,
				layer_input,
				hidden_size,
				seed.wrapping_add((index as u64).wrapping_mul(0xd1b5_4a32_d192_ed03)),
			)?);
			registry.register_module(format!("layer{index}"), layer.clone())?;
			layers.push(layer);
		}
		Ok(Self {
			input_size,
			hidden_size,
			layers,
			registry,
		})
	}

	/// Construct a one-layer RNN from exact matrices.
	///
	/// Parameter shapes are `weight_ih [H, I]`, `weight_hh [H, H]`, and two
	/// biases `[H]`.
	///
	/// # Errors
	///
	/// Returns an error when shape, dtype, engine ownership, or hidden-size
	/// contracts differ.
	pub fn from_matrices(
		weight_ih: Matrix,
		weight_hh: Matrix,
		bias_ih: Matrix,
		bias_hh: Matrix,
	) -> Result<Self> {
		let [hidden_size, input_size] = weight_ih.shape() else {
			return Err(Error::invalid_argument(
				"RNN input weight must have rank two",
			));
		};
		let input_size = *input_size;
		let hidden_size = *hidden_size;
		let layer = Rc::new(RnnLayer::from_matrices(
			weight_ih, weight_hh, bias_ih, bias_hh,
		)?);
		let mut registry = ModuleRegistry::new();
		registry.register_module("layer0", layer.clone())?;
		Ok(Self {
			input_size,
			hidden_size,
			layers: vec![layer],
			registry,
		})
	}

	/// Evaluate the complete sequence from a zero hidden state per layer.
	///
	/// # Errors
	///
	/// Returns an error unless input is same-engine F32 `[B, S, I]` with nonzero
	/// dimensions and the declared input width, or when runtime recording fails.
	pub fn forward(&self, input: &Matrix) -> Result<Matrix> {
		let [_, _, input_size] = input.shape() else {
			return Err(Error::invalid_argument(
				"RNN input must have shape [B, S, I]",
			));
		};
		if *input_size != self.input_size {
			return Err(Error::invalid_argument(format!(
				"RNN expected input width {}; found {input_size}",
				self.input_size
			)));
		}
		let mut output = input.clone();
		for layer in &self.layers {
			output = layer.forward(&output)?;
		}
		Ok(output)
	}

	pub const fn input_size(&self) -> usize {
		self.input_size
	}

	pub const fn hidden_size(&self) -> usize {
		self.hidden_size
	}

	pub fn num_layers(&self) -> usize {
		self.layers.len()
	}

	/// Return every parameter through the registered layer tree.
	pub fn all_parameters(&self) -> Result<Vec<Parameter>> {
		Module::all_parameters(self)
	}

	/// Return one layer's parameters in weight-ih, weight-hh, bias-ih, bias-hh
	/// order.
	pub fn layer_parameters(&self, layer: usize) -> Option<[Parameter; 4]> {
		self.layers.get(layer).map(|layer| layer.parameters())
	}
}

impl Module for Rnn {
	fn forward(&self, input: &Matrix) -> Result<Matrix> {
		Rnn::forward(self, input)
	}

	fn registry(&self) -> &ModuleRegistry {
		&self.registry
	}
}
