use std::{fs, path::Path};

use crate::{Engine, Error, Matrix, Result};

use super::{AdamW, Module, optimizer::AdamWRestore};

const MAGIC: &[u8; 8] = b"OARSML01";
const VERSION: u32 = 1;

/// Save registration-addressed model parameters and complete AdamW state.
///
/// This is an explicit host-observation boundary.
///
/// # Errors
///
/// Returns an error for ambiguous module ownership, optimizer/model mismatch,
/// arithmetic overflow, readback failure, or filesystem failure.
pub fn save_checkpoint(
	path: impl AsRef<Path>,
	model: &dyn Module,
	optimizer: &AdamW,
) -> Result<()> {
	let named = model.all_named_parameters()?;
	let optimizer_state = optimizer.checkpoint();
	if named.len() != optimizer_state.states.len() {
		return Err(Error::invalid_argument(
			"checkpoint optimizer does not cover the complete model parameter tree",
		));
	}
	let mut bytes = Vec::new();
	bytes.extend_from_slice(MAGIC);
	write_u32(&mut bytes, VERSION);
	write_u32(&mut bytes, optimizer_state.step);
	for value in [
		optimizer_state.learning_rate,
		optimizer_state.beta1,
		optimizer_state.beta2,
		optimizer_state.epsilon,
		optimizer_state.weight_decay,
	] {
		write_f32(&mut bytes, value);
	}
	write_u32(
		&mut bytes,
		checked_u32(named.len(), "checkpoint parameter count")?,
	);
	for entry in named {
		let parameter = entry.parameter();
		let Some((_, first_moment, second_moment)) = optimizer_state
			.states
			.iter()
			.find(|(candidate, _, _)| candidate.same_as(&parameter))
		else {
			return Err(Error::invalid_argument(
				"checkpoint optimizer parameter does not match the model tree",
			));
		};
		let data = parameter.data();
		let path_bytes = entry.path().as_bytes();
		write_u32(
			&mut bytes,
			checked_u32(path_bytes.len(), "checkpoint path length")?,
		);
		bytes.extend_from_slice(path_bytes);
		write_u32(
			&mut bytes,
			checked_u32(data.shape().len(), "checkpoint rank")?,
		);
		for extent in data.shape() {
			write_u64(
				&mut bytes,
				u64::try_from(*extent)
					.map_err(|_| Error::invalid_argument("checkpoint extent exceeds u64"))?,
			);
		}
		let count = checked_u32(data.num_elements(), "checkpoint element count")?;
		write_u32(&mut bytes, count);
		for matrix in [&data, first_moment, second_moment] {
			for value in matrix.read_f32()? {
				write_f32(&mut bytes, value);
			}
		}
	}
	fs::write(path, bytes).map_err(|source| Error::io("write ML checkpoint", source))
}

/// Load a complete model/AdamW checkpoint into existing matching owners.
///
/// The file is parsed and structurally validated before any parameter handle is
/// replaced. `engine` remains the sole owner of newly uploaded storage.
///
/// # Errors
///
/// Returns an error for malformed/truncated data, model path/shape mismatch,
/// optimizer mismatch, allocation failure, or filesystem failure.
pub fn load_checkpoint(
	engine: &Engine,
	path: impl AsRef<Path>,
	model: &dyn Module,
	optimizer: &mut AdamW,
) -> Result<()> {
	let bytes = fs::read(path).map_err(|source| Error::io("read ML checkpoint", source))?;
	let mut reader = Reader::new(&bytes);
	if reader.take(MAGIC.len())? != MAGIC || reader.u32()? != VERSION {
		return Err(Error::invalid_argument("unsupported ML checkpoint header"));
	}
	let step = reader.u32()?;
	let learning_rate = reader.f32()?;
	let beta1 = reader.f32()?;
	let beta2 = reader.f32()?;
	let epsilon = reader.f32()?;
	let weight_decay = reader.f32()?;
	let count = reader.usize_u32()?;
	let named = model.all_named_parameters()?;
	let optimizer_state = optimizer.checkpoint();
	if count != named.len() || count != optimizer_state.states.len() {
		return Err(Error::invalid_argument(
			"ML checkpoint parameter count mismatch",
		));
	}
	let engine_handle = engine.handle();
	if named.iter().any(|entry| {
		!entry
			.parameter()
			.data()
			.engine_handle()
			.same_as(&engine_handle)
	}) {
		return Err(Error::invalid_argument(
			"ML checkpoint engine does not own the destination model",
		));
	}
	let mut loaded = Vec::with_capacity(count);
	for expected in &named {
		let path_length = reader.usize_u32()?;
		let path = std::str::from_utf8(reader.take(path_length)?)
			.map_err(|_| Error::invalid_argument("ML checkpoint path is not UTF-8"))?;
		if path != expected.path() {
			return Err(Error::invalid_argument(format!(
				"ML checkpoint expected parameter {}, found {path}",
				expected.path()
			)));
		}
		let rank = reader.usize_u32()?;
		let parameter = expected.parameter();
		let expected_data = parameter.data();
		if rank != expected_data.shape().len() {
			return Err(Error::invalid_argument(format!(
				"ML checkpoint rank mismatch for {path}"
			)));
		}
		let mut shape = Vec::with_capacity(rank);
		for _ in 0..rank {
			shape.push(
				usize::try_from(reader.u64()?)
					.map_err(|_| Error::invalid_argument("checkpoint extent exceeds usize"))?,
			);
		}
		let element_count = reader.usize_u32()?;
		if expected_data.shape() != shape || expected_data.num_elements() != element_count {
			return Err(Error::invalid_argument(format!(
				"ML checkpoint shape mismatch for {path}"
			)));
		}
		let required_value_bytes = element_count
			.checked_mul(3)
			.and_then(|count| count.checked_mul(size_of::<f32>()))
			.ok_or_else(|| Error::invalid_argument("ML checkpoint value extent overflows usize"))?;
		if required_value_bytes > reader.remaining() {
			return Err(Error::invalid_argument("truncated ML checkpoint values"));
		}
		let mut matrices = Vec::with_capacity(3);
		for _ in 0..3 {
			let mut values = Vec::with_capacity(element_count);
			for _ in 0..element_count {
				values.push(reader.f32()?);
			}
			matrices.push(Matrix::from_f32(engine, shape.clone(), &values)?);
		}
		loaded.push((
			parameter,
			matrices.remove(0),
			matrices.remove(0),
			matrices.remove(0),
		));
	}
	if !reader.is_empty() {
		return Err(Error::invalid_argument(
			"ML checkpoint contains trailing data",
		));
	}
	for ((parameter, _, _), (loaded_parameter, _, _, _)) in
		optimizer_state.states.iter().zip(&loaded)
	{
		if !parameter.same_as(loaded_parameter) {
			return Err(Error::invalid_argument(
				"checkpoint optimizer order differs from module traversal",
			));
		}
	}
	let mut moments = Vec::with_capacity(loaded.len());
	for (parameter, data, first, second) in loaded {
		parameter.replace_data(data)?;
		parameter.clear_gradient();
		moments.push((first, second));
	}
	optimizer.restore_checkpoint(AdamWRestore {
		step,
		learning_rate,
		beta1,
		beta2,
		epsilon,
		weight_decay,
		moments,
	})
}

fn checked_u32(value: usize, label: &'static str) -> Result<u32> {
	u32::try_from(value).map_err(|_| Error::invalid_argument(format!("{label} exceeds u32")))
}
fn write_u32(bytes: &mut Vec<u8>, value: u32) {
	bytes.extend_from_slice(&value.to_le_bytes());
}
fn write_u64(bytes: &mut Vec<u8>, value: u64) {
	bytes.extend_from_slice(&value.to_le_bytes());
}
fn write_f32(bytes: &mut Vec<u8>, value: f32) {
	bytes.extend_from_slice(&value.to_le_bytes());
}

struct Reader<'a> {
	bytes: &'a [u8],
	offset: usize,
}
impl<'a> Reader<'a> {
	const fn new(bytes: &'a [u8]) -> Self {
		Self { bytes, offset: 0 }
	}
	fn take(&mut self, count: usize) -> Result<&'a [u8]> {
		let end = self
			.offset
			.checked_add(count)
			.ok_or_else(|| Error::invalid_argument("checkpoint offset overflow"))?;
		let value = self
			.bytes
			.get(self.offset..end)
			.ok_or_else(|| Error::invalid_argument("truncated ML checkpoint"))?;
		self.offset = end;
		Ok(value)
	}
	fn u32(&mut self) -> Result<u32> {
		Ok(u32::from_le_bytes(self.take(4)?.try_into().map_err(
			|_| Error::invalid_argument("invalid u32 checkpoint field"),
		)?))
	}
	fn u64(&mut self) -> Result<u64> {
		Ok(u64::from_le_bytes(self.take(8)?.try_into().map_err(
			|_| Error::invalid_argument("invalid u64 checkpoint field"),
		)?))
	}
	fn f32(&mut self) -> Result<f32> {
		Ok(f32::from_bits(self.u32()?))
	}
	fn usize_u32(&mut self) -> Result<usize> {
		usize::try_from(self.u32()?)
			.map_err(|_| Error::invalid_argument("checkpoint count exceeds usize"))
	}
	fn is_empty(&self) -> bool {
		self.offset == self.bytes.len()
	}
	fn remaining(&self) -> usize {
		self.bytes.len() - self.offset
	}
}
