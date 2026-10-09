use std::{collections::HashMap, fs, path::Path};

use serde_json::Value;

use crate::{Error, Result};

const HEADER_PREFIX_SIZE: usize = 8;
const MAX_HEADER_SIZE: usize = 100 * 1024 * 1024;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ScalarType {
	Bool,
	U8,
	I8,
	I16,
	U16,
	I32,
	U32,
	I64,
	U64,
	F16,
	Bf16,
	F32,
	F64,
}

impl ScalarType {
	fn parse(token: &str) -> Result<Self> {
		match token {
			"BOOL" => Ok(Self::Bool),
			"U8" => Ok(Self::U8),
			"I8" => Ok(Self::I8),
			"I16" => Ok(Self::I16),
			"U16" => Ok(Self::U16),
			"I32" => Ok(Self::I32),
			"U32" => Ok(Self::U32),
			"I64" => Ok(Self::I64),
			"U64" => Ok(Self::U64),
			"F16" => Ok(Self::F16),
			"BF16" => Ok(Self::Bf16),
			"F32" => Ok(Self::F32),
			"F64" => Ok(Self::F64),
			_ => Err(Error::data_loss(format!(
				"SafeTensors dtype {token:?} is unsupported"
			))),
		}
	}

	const fn size_bytes(self) -> usize {
		match self {
			Self::Bool | Self::U8 | Self::I8 => 1,
			Self::I16 | Self::U16 | Self::F16 | Self::Bf16 => 2,
			Self::I32 | Self::U32 | Self::F32 => 4,
			Self::I64 | Self::U64 | Self::F64 => 8,
		}
	}
}

struct TensorInfo {
	dtype: ScalarType,
	shape: Vec<usize>,
	start: usize,
	end: usize,
}

/// Validated, host-owned SafeTensors source used by explicit model translators.
pub(crate) struct SafeTensorsSource {
	bytes: Vec<u8>,
	data_offset: usize,
	tensors: HashMap<String, TensorInfo>,
	metadata: HashMap<String, String>,
}

impl SafeTensorsSource {
	pub(crate) fn open(path: &Path) -> Result<Self> {
		let bytes = fs::read(path).map_err(|source| Error::io("read SafeTensors", source))?;
		let prefix: [u8; HEADER_PREFIX_SIZE] = bytes
			.get(..HEADER_PREFIX_SIZE)
			.and_then(|value| value.try_into().ok())
			.ok_or_else(|| Error::data_loss("SafeTensors file is missing its header size"))?;
		let header_size = usize::try_from(u64::from_le_bytes(prefix))
			.map_err(|_| Error::data_loss("SafeTensors header size exceeds usize"))?;
		if header_size == 0 || header_size > MAX_HEADER_SIZE {
			return Err(Error::data_loss(
				"SafeTensors header size is outside its bound",
			));
		}
		let data_offset = HEADER_PREFIX_SIZE
			.checked_add(header_size)
			.ok_or_else(|| Error::data_loss("SafeTensors header range overflows usize"))?;
		let header = bytes
			.get(HEADER_PREFIX_SIZE..data_offset)
			.ok_or_else(|| Error::data_loss("SafeTensors header is truncated"))?;
		let root = serde_json::from_slice::<Value>(header)
			.map_err(|source| Error::data_loss(format!("invalid SafeTensors JSON: {source}")))?;
		let entries = root
			.as_object()
			.ok_or_else(|| Error::data_loss("SafeTensors header must be a JSON object"))?;
		let payload_size = bytes.len() - data_offset;
		let mut tensors = HashMap::with_capacity(entries.len());
		let mut metadata = HashMap::new();
		for (name, value) in entries {
			if name == "__metadata__" {
				let values = value
					.as_object()
					.ok_or_else(|| Error::data_loss("SafeTensors metadata must be a string object"))?;
				for (key, value) in values {
					let value = value
						.as_str()
						.ok_or_else(|| Error::data_loss("SafeTensors metadata values must be strings"))?;
					metadata.insert(key.clone(), value.to_owned());
				}
				continue;
			}
			if name.is_empty() || name.as_bytes().contains(&0) {
				return Err(Error::data_loss("SafeTensors tensor name is invalid"));
			}
			let object = value
				.as_object()
				.ok_or_else(|| Error::data_loss(format!("SafeTensors tensor {name} is not an object")))?;
			let dtype = ScalarType::parse(
				object
					.get("dtype")
					.and_then(Value::as_str)
					.ok_or_else(|| Error::data_loss(format!("tensor {name} has no dtype")))?,
			)?;
			let shape = object
				.get("shape")
				.and_then(Value::as_array)
				.ok_or_else(|| Error::data_loss(format!("tensor {name} has no shape")))?
				.iter()
				.map(|extent| {
					extent
						.as_u64()
						.and_then(|value| usize::try_from(value).ok())
						.ok_or_else(|| Error::data_loss(format!("tensor {name} has an invalid extent")))
				})
				.collect::<Result<Vec<_>>>()?;
			let offsets = object
				.get("data_offsets")
				.and_then(Value::as_array)
				.filter(|values| values.len() == 2)
				.ok_or_else(|| Error::data_loss(format!("tensor {name} has invalid offsets")))?;
			let start = json_offset(&offsets[0], name)?;
			let end = json_offset(&offsets[1], name)?;
			if start > end || end > payload_size {
				return Err(Error::data_loss(format!(
					"tensor {name} range is outside the file"
				)));
			}
			let elements = shape.iter().try_fold(1_usize, |count, extent| {
				count
					.checked_mul(*extent)
					.ok_or_else(|| Error::data_loss(format!("tensor {name} element count overflows")))
			})?;
			let expected = elements
				.checked_mul(dtype.size_bytes())
				.ok_or_else(|| Error::data_loss(format!("tensor {name} byte count overflows")))?;
			if end - start != expected {
				return Err(Error::data_loss(format!(
					"tensor {name} byte range does not match its shape and dtype"
				)));
			}
			tensors.insert(
				name.clone(),
				TensorInfo {
					dtype,
					shape,
					start,
					end,
				},
			);
		}
		let mut ranges = tensors
			.values()
			.filter(|tensor| tensor.start != tensor.end)
			.map(|tensor| (tensor.start, tensor.end))
			.collect::<Vec<_>>();
		ranges.sort_unstable();
		if ranges.windows(2).any(|pair| pair[1].0 < pair[0].1) {
			return Err(Error::data_loss("SafeTensors tensor ranges overlap"));
		}
		Ok(Self {
			bytes,
			data_offset,
			tensors,
			metadata,
		})
	}

	pub(crate) fn names(&self) -> impl Iterator<Item = &str> {
		self.tensors.keys().map(String::as_str)
	}

	pub(crate) fn shape(&self, name: &str) -> Option<&[usize]> {
		self.tensors.get(name).map(|tensor| tensor.shape.as_slice())
	}

	pub(crate) fn f32_bytes(&self, name: &str) -> Result<&[u8]> {
		let tensor = self
			.tensors
			.get(name)
			.ok_or_else(|| Error::not_found(format!("SafeTensors tensor is missing: {name}")))?;
		if tensor.dtype != ScalarType::F32 {
			return Err(Error::invalid_argument(format!(
				"SafeTensors tensor {name} must be F32"
			)));
		}
		let start = self.data_offset + tensor.start;
		let end = self.data_offset + tensor.end;
		Ok(&self.bytes[start..end])
	}

	#[allow(dead_code, reason = "retained for the format-neutral source contract")]
	pub(crate) fn metadata(&self) -> &HashMap<String, String> {
		&self.metadata
	}
}

fn json_offset(value: &Value, name: &str) -> Result<usize> {
	value
		.as_u64()
		.and_then(|value| usize::try_from(value).ok())
		.ok_or_else(|| Error::data_loss(format!("tensor {name} has an invalid offset")))
}
