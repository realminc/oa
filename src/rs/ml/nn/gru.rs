use std::rc::Rc;

use crate::{DType, Engine, Error, Matrix, Result};

use super::super::{
	Module, ModuleRegistry, Parameter, autograd, lowering::matrix as dispatch, matrix, random,
};

enum GruBias {
	Parameter(Parameter),
	Zero(Matrix),
}

impl GruBias {
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

struct GruLayer {
	input_size: usize,
	hidden_size: usize,
	weight_ih: Parameter,
	weight_hh: Parameter,
	bias_ih: GruBias,
	bias_hh: GruBias,
	registry: ModuleRegistry,
}

impl GruLayer {
	fn with_seed(
		engine: &Engine,
		input_size: usize,
		hidden_size: usize,
		bias: bool,
		seed: u64,
	) -> Result<Self> {
		let gate_size = hidden_size
			.checked_mul(3)
			.ok_or_else(|| Error::invalid_argument("GRU gate size overflows usize"))?;
		let weight_ih = xavier_matrix(engine, gate_size, input_size, seed)?;
		let weight_hh = xavier_matrix(
			engine,
			gate_size,
			hidden_size,
			seed.wrapping_add(0x9e37_79b9_7f4a_7c15),
		)?;
		let bias_ih = bias
			.then(|| Matrix::from_f32(engine, [gate_size], &vec![0.0; gate_size]))
			.transpose()?;
		let bias_hh = bias
			.then(|| Matrix::from_f32(engine, [gate_size], &vec![0.0; gate_size]))
			.transpose()?;
		Self::from_values(weight_ih, weight_hh, bias_ih, bias_hh)
	}

	fn from_values(
		weight_ih: Matrix,
		weight_hh: Matrix,
		bias_ih: Option<Matrix>,
		bias_hh: Option<Matrix>,
	) -> Result<Self> {
		let [gate_size, input_size] = weight_ih.shape() else {
			return Err(Error::invalid_argument(
				"GRU input weight must have shape [3H, I]",
			));
		};
		if *gate_size == 0 || gate_size % 3 != 0 || *input_size == 0 {
			return Err(Error::invalid_argument(
				"GRU input weight requires nonzero shape [3H, I]",
			));
		}
		let hidden_size = gate_size / 3;
		let same_bias_mode = bias_ih.is_some() == bias_hh.is_some();
		let engine = weight_ih.engine_handle().clone();
		if hidden_size > 1024
			|| weight_hh.shape() != [*gate_size, hidden_size]
			|| weight_ih.dtype() != DType::F32
			|| weight_hh.dtype() != DType::F32
			|| !engine.same_as(weight_hh.engine_handle())
			|| !same_bias_mode
			|| bias_ih.as_ref().is_some_and(|value| {
				value.shape() != [*gate_size]
					|| value.dtype() != DType::F32
					|| !engine.same_as(value.engine_handle())
			}) || bias_hh.as_ref().is_some_and(|value| {
			value.shape() != [*gate_size]
				|| value.dtype() != DType::F32
				|| !engine.same_as(value.engine_handle())
		}) {
			return Err(Error::invalid_argument(
				"GRU requires same-engine F32 weights [3H, I]/[3H, H], paired biases [3H], and 1 <= H <= 1024",
			));
		}
		let input_size = *input_size;
		let gate_size = *gate_size;
		let weight_ih = Parameter::new("weight_ih", weight_ih)?;
		let weight_hh = Parameter::new("weight_hh", weight_hh)?;
		let bias_ih = match bias_ih {
			Some(value) => GruBias::Parameter(Parameter::new("bias_ih", value)?),
			None => GruBias::Zero(Matrix::allocate(
				&engine,
				vec![gate_size],
				gate_size,
				DType::F32,
			)?),
		};
		let bias_hh = match bias_hh {
			Some(value) => GruBias::Parameter(Parameter::new("bias_hh", value)?),
			None => GruBias::Zero(Matrix::allocate(
				&engine,
				vec![gate_size],
				gate_size,
				DType::F32,
			)?),
		};
		let mut registry = ModuleRegistry::new();
		registry.register_parameter("weight_ih", weight_ih.clone())?;
		registry.register_parameter("weight_hh", weight_hh.clone())?;
		if let GruBias::Parameter(parameter) = &bias_ih {
			registry.register_parameter("bias_ih", parameter.clone())?;
		}
		if let GruBias::Parameter(parameter) = &bias_hh {
			registry.register_parameter("bias_hh", parameter.clone())?;
		}
		Ok(Self {
			input_size,
			hidden_size,
			weight_ih,
			weight_hh,
			bias_ih,
			bias_hh,
			registry,
		})
	}

	fn forward(&self, input: &Matrix) -> Result<Matrix> {
		let [batch, sequence_length, input_size] = input.shape() else {
			return Err(Error::invalid_argument(
				"GRU input must have shape [B, S, I]",
			));
		};
		if *batch == 0 || *sequence_length == 0 || *input_size != self.input_size {
			return Err(Error::invalid_argument(format!(
				"GRU requires nonempty input [B, S, {}]; found {:?}",
				self.input_size,
				input.shape()
			)));
		}
		let rows = batch
			.checked_mul(*sequence_length)
			.ok_or_else(|| Error::invalid_argument("GRU flattened row count overflows usize"))?;
		let flat_input = input.reshape([rows, self.input_size])?;
		let flat_gates = self.input_projection(&flat_input)?;
		let gates = flat_gates.reshape([*batch, *sequence_length, 3 * self.hidden_size])?;
		let (weight_hh, weight_hh_version, _) = self.weight_hh.snapshot();
		let (bias_hh, bias_hh_parameter, _) = self.bias_hh.snapshot();
		matrix::gru_scan_parameterized(
			&gates,
			(self.weight_hh.clone(), weight_hh, weight_hh_version),
			bias_hh_parameter,
			&bias_hh,
			self.has_bias(),
		)
	}

	fn input_projection(&self, input: &Matrix) -> Result<Matrix> {
		let (weight_ih, weight_ih_version, weight_ih_requires_grad) = self.weight_ih.snapshot();
		let (bias_ih, bias_ih_parameter, bias_ih_requires_grad) = self.bias_ih.snapshot();
		let gates = dispatch::linear(input, &weight_ih, &bias_ih)?;
		if weight_ih_requires_grad || bias_ih_requires_grad {
			autograd::record_linear(
				input,
				&gates,
				self.weight_ih.clone(),
				weight_ih,
				weight_ih_version,
				bias_ih_parameter,
			)?;
		}
		Ok(gates)
	}

	fn step(&self, input: &Matrix, hidden: &Matrix) -> Result<Matrix> {
		let [batch, input_size] = input.shape() else {
			return Err(Error::invalid_argument(
				"GRU cell input must have shape [B, I]",
			));
		};
		if *batch == 0 || *input_size != self.input_size || hidden.shape() != [*batch, self.hidden_size]
		{
			return Err(Error::invalid_argument(format!(
				"GRU cell requires input [B, {}] and hidden [B, {}]; found {:?} and {:?}",
				self.input_size,
				self.hidden_size,
				input.shape(),
				hidden.shape()
			)));
		}
		let gates = self.input_projection(input)?;
		let (weight_hh, weight_hh_version, _) = self.weight_hh.snapshot();
		let (bias_hh, bias_hh_parameter, _) = self.bias_hh.snapshot();
		matrix::gru_cell_parameterized(
			&gates,
			hidden,
			(self.weight_hh.clone(), weight_hh, weight_hh_version),
			bias_hh_parameter,
			&bias_hh,
			self.has_bias(),
		)
	}

	fn parameters(&self) -> Vec<Parameter> {
		let mut parameters = vec![self.weight_ih.clone(), self.weight_hh.clone()];
		if let GruBias::Parameter(parameter) = &self.bias_ih {
			parameters.push(parameter.clone());
		}
		if let GruBias::Parameter(parameter) = &self.bias_hh {
			parameters.push(parameter.clone());
		}
		parameters
	}

	fn has_bias(&self) -> bool {
		matches!(self.bias_ih, GruBias::Parameter(_))
	}
}

impl Module for GruLayer {
	fn forward(&self, input: &Matrix) -> Result<Matrix> {
		GruLayer::forward(self, input)
	}

	fn registry(&self) -> &ModuleRegistry {
		&self.registry
	}
}

/// One gated recurrent unit cell with an explicit caller-owned hidden state.
pub struct GruCell {
	layer: GruLayer,
}

impl GruCell {
	/// Construct a deterministically initialized GRU cell.
	///
	/// # Errors
	///
	/// Returns an error when dimensions are zero, hidden size exceeds 1024,
	/// arithmetic overflows, or parameter allocation fails.
	pub fn with_seed(
		engine: &Engine,
		input_size: usize,
		hidden_size: usize,
		bias: bool,
		seed: u64,
	) -> Result<Self> {
		if input_size == 0 || hidden_size == 0 || hidden_size > 1024 {
			return Err(Error::invalid_argument(
				"GRU cell requires nonzero input/hidden sizes and hidden size <= 1024",
			));
		}
		Ok(Self {
			layer: GruLayer::with_seed(engine, input_size, hidden_size, bias, seed)?,
		})
	}

	/// Construct a biased GRU cell from exact matrices.
	///
	/// # Errors
	///
	/// Returns an error unless weights use `[3H, I]` and `[3H, H]`, biases use
	/// `[3H]`, and all values satisfy the F32 engine and hidden-size contract.
	pub fn from_matrices(
		weight_ih: Matrix,
		weight_hh: Matrix,
		bias_ih: Matrix,
		bias_hh: Matrix,
	) -> Result<Self> {
		Ok(Self {
			layer: GruLayer::from_values(weight_ih, weight_hh, Some(bias_ih), Some(bias_hh))?,
		})
	}

	/// Construct a bias-free GRU cell from exact weight matrices.
	///
	/// # Errors
	///
	/// Returns an error unless both weights satisfy the F32 engine, shape, and
	/// hidden-size contract.
	pub fn from_weights(weight_ih: Matrix, weight_hh: Matrix) -> Result<Self> {
		Ok(Self {
			layer: GruLayer::from_values(weight_ih, weight_hh, None, None)?,
		})
	}

	/// Create a zero hidden state `[batch, hidden]` on this cell's engine.
	///
	/// # Errors
	///
	/// Returns an error when batch is zero, shape arithmetic overflows, or
	/// allocation fails.
	pub fn zero_state(&self, batch: usize) -> Result<Matrix> {
		if batch == 0 {
			return Err(Error::invalid_argument("GRU cell batch must be nonzero"));
		}
		let count = batch
			.checked_mul(self.layer.hidden_size)
			.ok_or_else(|| Error::invalid_argument("GRU cell state size overflows usize"))?;
		let (weight, _, _) = self.layer.weight_hh.snapshot();
		Matrix::allocate(
			weight.engine_handle(),
			vec![batch, self.layer.hidden_size],
			count,
			DType::F32,
		)
	}

	/// Apply one recurrent step to `[B, I]` input and `[B, H]` hidden state.
	///
	/// # Errors
	///
	/// Returns an error when the input/hidden contract or runtime recording fails.
	pub fn step(&self, input: &Matrix, hidden: &Matrix) -> Result<Matrix> {
		self.layer.step(input, hidden)
	}

	/// Return the configured input width.
	pub const fn input_size(&self) -> usize {
		self.layer.input_size
	}

	/// Return the recurrent hidden width.
	pub const fn hidden_size(&self) -> usize {
		self.layer.hidden_size
	}

	/// Return whether the two biases are trainable parameters.
	pub fn has_bias(&self) -> bool {
		self.layer.has_bias()
	}

	/// Return parameters in weight-ih, weight-hh, bias-ih, bias-hh order.
	pub fn all_parameters(&self) -> Result<Vec<Parameter>> {
		Module::all_parameters(self)
	}
}

impl Module for GruCell {
	fn forward(&self, input: &Matrix) -> Result<Matrix> {
		let [batch, _] = input.shape() else {
			return Err(Error::invalid_argument(
				"GRU cell input must have shape [B, I]",
			));
		};
		self.step(input, &self.zero_state(*batch)?)
	}

	fn registry(&self) -> &ModuleRegistry {
		&self.layer.registry
	}
}

/// Stacked batch-first gated recurrent unit using whole-sequence GPU scans.
///
/// Input uses `[batch, sequence, input]`; output uses
/// `[batch, sequence, hidden]`. Every layer starts from a zero hidden state.
pub struct Gru {
	input_size: usize,
	hidden_size: usize,
	layers: Vec<Rc<GruLayer>>,
	registry: ModuleRegistry,
}

impl Gru {
	/// Construct a deterministically initialized stacked GRU.
	///
	/// # Errors
	///
	/// Returns an error when a dimension or layer count is zero, hidden size
	/// exceeds the 1024-element scan limit, arithmetic overflows, or parameter
	/// allocation and registration fail.
	pub fn with_seed(
		engine: &Engine,
		input_size: usize,
		hidden_size: usize,
		num_layers: usize,
		bias: bool,
		seed: u64,
	) -> Result<Self> {
		if input_size == 0 || hidden_size == 0 || hidden_size > 1024 || num_layers == 0 {
			return Err(Error::invalid_argument(
				"GRU requires nonzero input/hidden/layer counts and hidden size <= 1024",
			));
		}
		let mut layers = Vec::with_capacity(num_layers);
		let mut registry = ModuleRegistry::new();
		for index in 0..num_layers {
			let layer_input = if index == 0 { input_size } else { hidden_size };
			let layer = Rc::new(GruLayer::with_seed(
				engine,
				layer_input,
				hidden_size,
				bias,
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

	/// Construct a one-layer biased GRU from exact parameter matrices.
	///
	/// # Errors
	///
	/// Returns an error unless weights use `[3H, I]` and `[3H, H]`, biases use
	/// `[3H]`, and all values are nonempty same-engine F32 matrices with
	/// `H <= 1024`.
	pub fn from_matrices(
		weight_ih: Matrix,
		weight_hh: Matrix,
		bias_ih: Matrix,
		bias_hh: Matrix,
	) -> Result<Self> {
		Self::from_one_layer(weight_ih, weight_hh, Some(bias_ih), Some(bias_hh))
	}

	/// Construct a one-layer bias-free GRU from exact weight matrices.
	///
	/// # Errors
	///
	/// Returns an error unless weights use `[3H, I]` and `[3H, H]` and are
	/// nonempty same-engine F32 matrices with `H <= 1024`.
	pub fn from_weights(weight_ih: Matrix, weight_hh: Matrix) -> Result<Self> {
		Self::from_one_layer(weight_ih, weight_hh, None, None)
	}

	fn from_one_layer(
		weight_ih: Matrix,
		weight_hh: Matrix,
		bias_ih: Option<Matrix>,
		bias_hh: Option<Matrix>,
	) -> Result<Self> {
		let [gate_size, input_size] = weight_ih.shape() else {
			return Err(Error::invalid_argument(
				"GRU input weight must have shape [3H, I]",
			));
		};
		if gate_size % 3 != 0 {
			return Err(Error::invalid_argument(
				"GRU input weight first extent must be divisible by three",
			));
		}
		let input_size = *input_size;
		let hidden_size = gate_size / 3;
		let layer = Rc::new(GruLayer::from_values(
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

	/// Evaluate the complete sequence with one input projection and one recurrent
	/// scan dispatch per layer.
	///
	/// # Errors
	///
	/// Returns an error unless input is same-engine F32 `[B, S, I]` with nonzero
	/// extents and the configured input width, or when recording fails.
	pub fn forward(&self, input: &Matrix) -> Result<Matrix> {
		let [_, _, input_size] = input.shape() else {
			return Err(Error::invalid_argument(
				"GRU input must have shape [B, S, I]",
			));
		};
		if *input_size != self.input_size {
			return Err(Error::invalid_argument(format!(
				"GRU expected input width {}; found {input_size}",
				self.input_size
			)));
		}
		let mut output = input.clone();
		for layer in &self.layers {
			output = layer.forward(&output)?;
		}
		Ok(output)
	}

	/// Return the configured input width.
	pub const fn input_size(&self) -> usize {
		self.input_size
	}

	/// Return the recurrent hidden width.
	pub const fn hidden_size(&self) -> usize {
		self.hidden_size
	}

	/// Return the number of stacked recurrent layers.
	pub fn num_layers(&self) -> usize {
		self.layers.len()
	}

	/// Return whether layer biases are trainable parameters.
	pub fn has_bias(&self) -> bool {
		self.layers.first().is_some_and(|layer| layer.has_bias())
	}

	/// Return every parameter through the registered layer tree.
	pub fn all_parameters(&self) -> Result<Vec<Parameter>> {
		Module::all_parameters(self)
	}

	/// Return one layer's parameters in weight-ih, weight-hh, bias-ih, bias-hh
	/// order. Bias-free layers return only the two weights.
	pub fn layer_parameters(&self, layer: usize) -> Option<Vec<Parameter>> {
		self.layers.get(layer).map(|layer| layer.parameters())
	}
}

impl Module for Gru {
	fn forward(&self, input: &Matrix) -> Result<Matrix> {
		Gru::forward(self, input)
	}

	fn registry(&self) -> &ModuleRegistry {
		&self.registry
	}
}

fn xavier_matrix(
	engine: &Engine,
	output_features: usize,
	input_features: usize,
	seed: u64,
) -> Result<Matrix> {
	let count = output_features
		.checked_mul(input_features)
		.ok_or_else(|| Error::invalid_argument("GRU weight size overflows usize"))?;
	let denominator = output_features
		.checked_add(input_features)
		.ok_or_else(|| Error::invalid_argument("GRU Xavier extent overflows usize"))?;
	let limit = (6.0_f32 / denominator as f32).sqrt();
	Matrix::from_f32(
		engine,
		[output_features, input_features],
		&random::symmetric_uniform(count, limit, seed),
	)
}
