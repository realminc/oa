//! ONNX model import and forward evaluation.
//!
//! [`OnnxGraph`] parses an ONNX `ModelProto` from its protobuf wire encoding
//! without external dependencies, uploads FP32 initializers to the device,
//! and evaluates the node list in order over [`Matrix`] values. The admitted
//! operator subset is `MatMul`, `Gemm`, `Add`, `Sub`, `Mul`, `Relu`,
//! `Softmax`, `Flatten`, and `Transpose` — the coverage needed for
//! feed-forward reference models; unsupported operators fail closed with a
//! clear error rather than silently dropping nodes.

use std::collections::BTreeMap;
use std::path::Path;

use crate::{Engine, Error, Matrix, Result, matrix};

const ONNX_TENSOR_FLOAT: i64 = 1;

#[derive(Debug)]
enum Wire<'a> {
	Varint(u64),
	Fixed32(u32),
	Fixed64,
	Bytes(&'a [u8]),
}

fn read_fields<'a>(
	mut bytes: &'a [u8],
	mut field: impl FnMut(u64, Wire<'a>) -> Result<()>,
) -> Result<()> {
	while !bytes.is_empty() {
		let tag = varint(&mut bytes)?;
		let number = tag >> 3;
		let value = match tag & 7 {
			0 => Wire::Varint(varint(&mut bytes)?),
			1 => {
				let (head, tail) = bytes.split_at_checked(8).ok_or_else(truncated)?;
				bytes = tail;
				let _ = head;
				Wire::Fixed64
			}
			2 => {
				let length = varint(&mut bytes)? as usize;
				let (head, tail) = bytes.split_at_checked(length).ok_or_else(truncated)?;
				bytes = tail;
				Wire::Bytes(head)
			}
			5 => {
				let (head, tail) = bytes.split_at_checked(4).ok_or_else(truncated)?;
				bytes = tail;
				Wire::Fixed32(u32::from_le_bytes(head.try_into().expect("4 bytes")))
			}
			wire => {
				return Err(Error::data_loss(format!(
					"ONNX field {number} uses unsupported wire type {wire}"
				)));
			}
		};
		field(number, value)?;
	}
	Ok(())
}

fn varint(bytes: &mut &[u8]) -> Result<u64> {
	let mut value = 0_u64;
	for shift in (0..70).step_by(7) {
		let byte = *bytes.first().ok_or_else(truncated)?;
		*bytes = &bytes[1..];
		value |= u64::from(byte & 0x7f) << shift;
		if byte & 0x80 == 0 {
			return Ok(value);
		}
	}
	Err(Error::data_loss("ONNX varint exceeds ten bytes"))
}

fn truncated() -> Error {
	Error::data_loss("truncated ONNX protobuf payload")
}

fn text(wire: Wire<'_>) -> Result<String> {
	let Wire::Bytes(bytes) = wire else {
		return Err(Error::data_loss(
			"ONNX string field has the wrong wire type",
		));
	};
	String::from_utf8(bytes.to_vec()).map_err(|_| Error::data_loss("ONNX name is not UTF-8"))
}

#[derive(Default)]
struct OnnxTensor {
	name: String,
	dims: Vec<usize>,
	values: Vec<f32>,
	data_type: i64,
}

#[derive(Default)]
struct OnnxAttribute {
	name: String,
	f: f32,
	i: i64,
	ints: Vec<i64>,
}

#[derive(Default)]
struct OnnxNode {
	op_type: String,
	inputs: Vec<String>,
	output: String,
	attributes: Vec<OnnxAttribute>,
}

/// One imported ONNX inference graph over device-resident F32 tensors.
pub struct OnnxGraph {
	input_name: String,
	output_name: String,
	initializers: BTreeMap<String, Matrix>,
	nodes: Vec<OnnxNode>,
}

impl OnnxGraph {
	/// Load an ONNX model from disk onto the current engine.
	///
	/// # Errors
	///
	/// Returns an error for unreadable files, malformed protobuf payloads,
	/// non-F32 tensors, or unsupported operators in the node list.
	pub fn load(path: impl AsRef<Path>) -> Result<Self> {
		Self::load_on(&Engine::current()?, path)
	}

	/// Load an ONNX model from disk onto `engine`.
	///
	/// # Errors
	///
	/// Returns an error under the [`Self::load`] contract.
	pub fn load_on(engine: &Engine, path: impl AsRef<Path>) -> Result<Self> {
		let bytes =
			std::fs::read(path.as_ref()).map_err(|source| Error::io("ONNX model read", source))?;
		Self::from_bytes_on(engine, &bytes)
	}

	/// Import an ONNX model from its protobuf wire encoding in memory.
	///
	/// # Errors
	///
	/// Returns an error under the [`Self::load`] contract.
	pub fn from_bytes_on(engine: &Engine, bytes: &[u8]) -> Result<Self> {
		let mut graph_bytes = None;
		read_fields(bytes, |number, wire| {
			if number == 7 {
				let Wire::Bytes(payload) = wire else {
					return Err(Error::data_loss("ONNX graph field is not a message"));
				};
				graph_bytes = Some(payload);
			}
			Ok(())
		})?;
		let graph_bytes = graph_bytes.ok_or_else(|| Error::data_loss("ONNX model has no graph"))?;
		let mut input_name = String::new();
		let mut output_name = String::new();
		let mut initializers = BTreeMap::new();
		let mut nodes = Vec::new();
		read_fields(graph_bytes, |number, wire| {
			let Wire::Bytes(payload) = wire else {
				return Ok(());
			};
			match number {
				1 => nodes.push(parse_node(payload)?),
				5 => {
					let tensor = parse_tensor(payload)?;
					if tensor.data_type != ONNX_TENSOR_FLOAT {
						return Err(Error::invalid_argument(format!(
							"ONNX initializer {} uses dtype {}; only FLOAT is admitted",
							tensor.name, tensor.data_type,
						)));
					}
					let expected: usize = tensor.dims.iter().product();
					if tensor.values.len() != expected {
						return Err(Error::data_loss(format!(
							"ONNX initializer {} declares {expected} elements but carries {}",
							tensor.name,
							tensor.values.len(),
						)));
					}
					let matrix = Matrix::new_from_f32_on(engine, tensor.dims.clone(), &tensor.values)
						.map_err(|source| {
							Error::data_loss(format!(
								"ONNX initializer {} upload failed: {source}",
								tensor.name
							))
						})?;
					initializers.insert(tensor.name, matrix);
				}
				11 => {
					if let Some(name) = value_info_name(payload)?
						&& !initializers.contains_key(&name)
						&& input_name.is_empty()
					{
						input_name = name;
					}
				}
				12 => {
					if let Some(name) = value_info_name(payload)?
						&& output_name.is_empty()
					{
						output_name = name;
					}
				}
				_ => {}
			}
			Ok(())
		})?;
		if input_name.is_empty() {
			// Graph inputs that are not initializers arrive before outputs;
			// fall back to the first node input that is not an initializer.
			for node in &nodes {
				if let Some(name) = node
					.inputs
					.iter()
					.find(|name| !initializers.contains_key(*name))
				{
					input_name = name.clone();
					break;
				}
			}
		}
		if input_name.is_empty() || output_name.is_empty() {
			return Err(Error::invalid_argument(
				"ONNX graph requires exactly one non-initializer input and one output",
			));
		}
		for node in &nodes {
			if !matches!(
				node.op_type.as_str(),
				"MatMul" | "Gemm" | "Add" | "Sub" | "Mul" | "Relu" | "Softmax" | "Flatten" | "Transpose"
			) {
				return Err(Error::invalid_argument(format!(
					"ONNX operator {} is outside the admitted import subset",
					node.op_type,
				)));
			}
		}
		Ok(Self {
			input_name,
			output_name,
			initializers,
			nodes,
		})
	}

	/// Return the bound graph input name.
	pub fn input_name(&self) -> &str {
		&self.input_name
	}

	/// Return the graph output name.
	pub fn output_name(&self) -> &str {
		&self.output_name
	}

	/// Return the number of graph nodes evaluated per forward.
	pub fn num_nodes(&self) -> usize {
		self.nodes.len()
	}

	/// Evaluate the graph on `input`, returning the bound output tensor.
	///
	/// # Errors
	///
	/// Returns an error for a malformed graph environment or failed
	/// operation recording; unknown node inputs fail closed.
	pub fn forward(&self, input: &Matrix) -> Result<Matrix> {
		let mut environment = self.initializers.clone();
		environment.insert(self.input_name.clone(), input.clone());
		let mut produced = None;
		for node in &self.nodes {
			let value = self.evaluate(node, &environment)?;
			environment.insert(node.output.clone(), value.clone());
			produced = Some(value);
		}
		let value = environment
			.get(&self.output_name)
			.cloned()
			.or(produced)
			.ok_or_else(|| Error::data_loss("ONNX graph produced no output tensor"))?;
		Ok(value)
	}

	fn evaluate(&self, node: &OnnxNode, environment: &BTreeMap<String, Matrix>) -> Result<Matrix> {
		let operand = |index: usize| -> Result<&Matrix> {
			let name = node.inputs.get(index).ok_or_else(|| {
				Error::data_loss(format!(
					"ONNX {} node is missing input {index}",
					node.op_type
				))
			})?;
			environment.get(name).ok_or_else(|| {
				Error::data_loss(format!(
					"ONNX {} node references unbound tensor {name}",
					node.op_type,
				))
			})
		};
		let attr_i = |name: &str, default: i64| {
			node
				.attributes
				.iter()
				.find(|attribute| attribute.name == name)
				.map(|attribute| attribute.i)
				.unwrap_or(default)
		};
		let attr_f = |name: &str, default: f32| {
			node
				.attributes
				.iter()
				.find(|attribute| attribute.name == name)
				.map(|attribute| attribute.f)
				.unwrap_or(default)
		};
		match node.op_type.as_str() {
			"MatMul" => {
				let right = matrix::transpose(operand(1)?, -2, -1)?;
				matrix::mat_mul_nt(operand(0)?, &right)
			}
			"Gemm" => {
				let mut left = operand(0)?.clone();
				let mut right = operand(1)?.clone();
				if attr_i("transA", 0) == 1 {
					left = matrix::transpose(&left, -2, -1)?;
				}
				if attr_i("transB", 0) != 1 {
					right = matrix::transpose(&right, -2, -1)?;
				}
				let mut product = matrix::mat_mul_nt(&left, &right)?;
				let alpha = attr_f("alpha", 1.0);
				let beta = attr_f("beta", 1.0);
				if alpha != 1.0 {
					product = matrix::scale(&product, alpha)?;
				}
				if node.inputs.len() > 2 {
					let bias = operand(2)?.clone();
					let bias = if beta == 1.0 {
						bias
					} else {
						matrix::scale(&bias, beta)?
					};
					product = matrix::add(&product, &bias)?;
				}
				Ok(product)
			}
			"Add" => matrix::add(operand(0)?, operand(1)?),
			"Sub" => matrix::sub(operand(0)?, operand(1)?),
			"Mul" => matrix::mul(operand(0)?, operand(1)?),
			"Relu" => crate::ml::matrix::relu(operand(0)?),
			"Softmax" => matrix::softmax(operand(0)?, attr_i("axis", -1) as i32),
			"Transpose" => {
				let perm = self.perm_of(node)?;
				let rank = operand(0)?.shape().len();
				if perm.len() != rank || (0..rank).any(|axis| !perm.contains(&(axis as i64))) {
					return Err(Error::invalid_argument(
						"ONNX Transpose perm is not a permutation of the input rank",
					));
				}
				// Compose pairwise axis swaps into the requested permutation.
				let mut axes = (0..rank as i64).collect::<Vec<i64>>();
				let mut result = operand(0)?.clone();
				for (position, target) in perm.iter().enumerate() {
					let source = axes
						.iter()
						.position(|axis| axis == target)
						.expect("perm is a permutation");
					if source != position {
						result = matrix::transpose(&result, position as i32, source as i32)?;
						axes.swap(position, source);
					}
				}
				Ok(result)
			}
			"Flatten" => {
				let axis = attr_i("axis", 1).max(0) as usize;
				let shape = operand(0)?.shape().to_vec();
				if axis > shape.len() {
					return Err(Error::invalid_argument(
						"ONNX Flatten axis exceeds the input rank",
					));
				}
				let outer: usize = shape[..axis].iter().product();
				let inner: usize = shape[axis..].iter().product();
				matrix::reshape(operand(0)?, [outer, inner])
			}
			op => Err(Error::invalid_argument(format!(
				"ONNX operator {op} is outside the admitted import subset"
			))),
		}
	}

	fn perm_of(&self, node: &OnnxNode) -> Result<Vec<i64>> {
		node
			.attributes
			.iter()
			.find(|attribute| attribute.name == "perm")
			.map(|attribute| attribute.ints.clone())
			.ok_or_else(|| Error::invalid_argument("ONNX Transpose requires a perm attribute"))
	}
}

fn value_info_name(payload: &[u8]) -> Result<Option<String>> {
	let mut name = None;
	read_fields(payload, |number, wire| {
		if number == 1 {
			name = Some(text(wire)?);
		}
		Ok(())
	})?;
	Ok(name)
}

fn parse_tensor(payload: &[u8]) -> Result<OnnxTensor> {
	let mut tensor = OnnxTensor::default();
	read_fields(payload, |number, wire| {
		match number {
			1 => {
				// dims: packed or unpacked int64
				match wire {
					Wire::Varint(value) => tensor.dims.push(value as usize),
					Wire::Bytes(bytes) => {
						let mut rest = bytes;
						while !rest.is_empty() {
							tensor.dims.push(varint(&mut rest)? as usize);
						}
					}
					_ => return Err(Error::data_loss("ONNX dims field has the wrong wire type")),
				}
			}
			2 => {
				let Wire::Varint(value) = wire else {
					return Err(Error::data_loss("ONNX data_type is not a varint"));
				};
				tensor.data_type = value as i64;
			}
			4 => {
				// float_data: packed fixed32
				let Wire::Bytes(bytes) = wire else {
					return Err(Error::data_loss("ONNX float_data is not packed"));
				};
				if !bytes.len().is_multiple_of(4) {
					return Err(Error::data_loss("ONNX float_data is not four-byte aligned"));
				}
				for chunk in bytes.as_chunks::<4>().0 {
					tensor.values.push(f32::from_le_bytes(*chunk));
				}
			}
			8 => tensor.name = text(wire)?,
			9 => {
				// raw_data: little-endian typed payload; FLOAT = f32 words
				let Wire::Bytes(bytes) = wire else {
					return Err(Error::data_loss("ONNX raw_data is not bytes"));
				};
				if !bytes.len().is_multiple_of(4) {
					return Err(Error::data_loss("ONNX raw_data is not four-byte aligned"));
				}
				tensor.values.clear();
				for chunk in bytes.as_chunks::<4>().0 {
					tensor.values.push(f32::from_le_bytes(*chunk));
				}
			}
			_ => {}
		}
		Ok(())
	})?;
	Ok(tensor)
}

fn parse_attribute(payload: &[u8]) -> Result<OnnxAttribute> {
	let mut attribute = OnnxAttribute::default();
	read_fields(payload, |number, wire| {
		match number {
			1 => attribute.name = text(wire)?,
			2 => {
				let Wire::Fixed32(bits) = wire else {
					return Err(Error::data_loss("ONNX float attribute is not fixed32"));
				};
				attribute.f = f32::from_bits(bits);
			}
			3 => {
				let Wire::Varint(value) = wire else {
					return Err(Error::data_loss("ONNX int attribute is not a varint"));
				};
				attribute.i = value as i64;
			}
			7 => {
				// ints: packed or unpacked int64
				match wire {
					Wire::Varint(value) => attribute.ints.push(value as i64),
					Wire::Bytes(bytes) => {
						let mut rest = bytes;
						while !rest.is_empty() {
							attribute.ints.push(varint(&mut rest)? as i64);
						}
					}
					_ => {
						return Err(Error::data_loss(
							"ONNX ints attribute has the wrong wire type",
						));
					}
				}
			}
			_ => {}
		}
		Ok(())
	})?;
	Ok(attribute)
}

fn parse_node(payload: &[u8]) -> Result<OnnxNode> {
	let mut node = OnnxNode::default();
	read_fields(payload, |number, wire| {
		match number {
			1 => node.inputs.push(text(wire)?),
			2 => node.output = text(wire)?,
			4 => node.op_type = text(wire)?,
			5 => {
				let Wire::Bytes(inner) = wire else {
					return Err(Error::data_loss("ONNX attribute is not a message"));
				};
				node.attributes.push(parse_attribute(inner)?);
			}
			_ => {}
		}
		Ok(())
	})?;
	Ok(node)
}
