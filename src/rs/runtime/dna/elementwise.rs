//! Domain-neutral FP32 fusion SSA and alternative physical lowerings.
//!
//! Resource identity and observability are admitted by runtime/fusion.rs.
//! This module carries slots, values and exact constant bits, never handles.
//! Tape limits are a backend restriction, not a limit on legal SSA programs.

pub(in crate::runtime) const REGISTER_COUNT: u32 = 8;
pub(in crate::runtime) const MAX_OPERAND_SLOTS: usize = 16;
pub(in crate::runtime) const MAX_STORE_SLOTS: usize = 6;
pub(in crate::runtime) const MAX_TAPE_STEPS: usize = 64;
#[derive(Clone, Debug, PartialEq, Eq)]
pub(in crate::runtime) enum Source {
	Input(usize),
	Value(usize),
	Constant(u32),
	Broadcast {
		input: usize,
		rank: u32,
		dimensions: [u32; 8],
		strides: [u32; 8],
	},
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(in crate::runtime) struct Instruction {
	pub opcode: u32,
	pub sources: Vec<Source>,
	/// Gather {rows,width,dtype}, or a backward alpha constant.
	pub parameters: Vec<u32>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(in crate::runtime) struct Program {
	pub element_count: u32,
	pub input_bytes: Vec<u64>,
	pub instructions: Vec<Instruction>,
	/// Output slot order is semantic materialization order, not a register ID.
	pub outputs: Vec<usize>,
}

impl Program {
	pub fn validate(&self) -> bool {
		let count = u64::from(self.element_count);
		if count == 0
			|| count * 4 > u64::from(u32::MAX)
			|| self.instructions.is_empty()
			|| self.instructions.len() > 256
			|| self.input_bytes.is_empty()
			|| self.input_bytes.len() > 64
			|| self.outputs.is_empty()
			|| self.outputs.len() > 64
			|| self
				.outputs
				.iter()
				.any(|&value| value >= self.instructions.len())
		{
			return false;
		}
		for (index, op) in self.instructions.iter().enumerate() {
			let (arity, parameters) = match op.opcode {
				4 => (2, 3),
				10..=16 | 40..=46 => (2, 0),
				20..=28 | 30..=33 => (1, 0),
				47..=48 => (2, 1),
				_ => return false,
			};
			if op.sources.len() != arity || op.parameters.len() != parameters {
				return false;
			}
			if op.opcode == 4 {
				let [Source::Input(table), Source::Input(indices)] = op.sources[..] else {
					return false;
				};
				let [rows, width, dtype] = op.parameters[..] else {
					return false;
				};
				if rows == 0
					|| width == 0
					|| dtype > 1
					|| !self.element_count.is_multiple_of(width)
					|| self.input_bytes.get(table).is_none_or(|&bytes| {
						u64::from(rows)
							.checked_mul(u64::from(width))
							.and_then(|v| v.checked_mul(4))
							.is_none_or(|need| need > bytes || need > u64::from(u32::MAX))
					}) || self.input_bytes.get(indices).is_none_or(|&bytes| {
					(count / u64::from(width) * if dtype == 0 { 1 } else { 4 }).next_multiple_of(4) > bytes
				}) {
					return false;
				}
				continue;
			}
			for source in &op.sources {
				match source {
					Source::Value(value) if *value < index => {}
					Source::Constant(_) => {}
					Source::Input(slot)
						if self
							.input_bytes
							.get(*slot)
							.is_some_and(|&bytes| count * 4 <= bytes) => {}
					Source::Broadcast {
						input,
						rank,
						dimensions,
						strides,
					} => {
						let rank = *rank as usize;
						if !(1..=8).contains(&rank) {
							return false;
						}
						let mut elements = 1u64;
						let mut last = 0u64;
						for (&dim, &stride) in dimensions[..rank].iter().zip(&strides[..rank]) {
							if dim == 0 {
								return false;
							}
							let Some(product) = elements.checked_mul(u64::from(dim)) else {
								return false;
							};
							elements = product;
							let Some(end) = last.checked_add(u64::from(dim - 1) * u64::from(stride)) else {
								return false;
							};
							last = end;
						}
						if elements != count
							|| last >= u64::from(u32::MAX) / 4
							|| self
								.input_bytes
								.get(*input)
								.is_none_or(|&bytes| (last + 1) * 4 > bytes)
						{
							return false;
						}
					}
					_ => return false,
				}
			}
		}
		true
	}

	/// Encode the existing interpreter ABI after resolving descriptor slots.
	pub fn tape(&self, inputs: &[u32], outputs: &[u32]) -> Option<Vec<u32>> {
		if !self.validate()
			|| inputs.len() != self.input_bytes.len()
			|| outputs.len() != self.outputs.len()
			|| inputs.len() > MAX_OPERAND_SLOTS
			|| outputs.len() > MAX_STORE_SLOTS
			|| inputs
				.iter()
				.chain(outputs)
				.any(|&slot| slot > u32::MAX - REGISTER_COUNT)
		{
			return None;
		}
		let mut last_use: Vec<usize> = (0..self.instructions.len()).collect();
		for (index, op) in self.instructions.iter().enumerate() {
			for source in &op.sources {
				if let Source::Value(value) = source {
					last_use[*value] = index;
				}
			}
		}
		let mut free: Vec<u32> = (0..REGISTER_COUNT).rev().collect();
		let mut registers = vec![0u32; self.instructions.len()];
		let mut tape = Vec::new();
		for (index, op) in self.instructions.iter().enumerate() {
			let dst = free.pop()?;
			let mut temporaries = Vec::new();
			let mut args = Vec::new();
			for source in &op.sources {
				args.push(match source {
					Source::Input(slot) => REGISTER_COUNT + inputs[*slot],
					Source::Value(value) => registers[*value],
					Source::Constant(bits) => {
						let register = free.pop()?;
						temporaries.push(register);
						tape.extend_from_slice(&[1, *bits, 0, register]);
						register
					}
					Source::Broadcast {
						input,
						rank,
						dimensions,
						strides,
					} => {
						let register = free.pop()?;
						temporaries.push(register);
						tape.extend_from_slice(&[3, REGISTER_COUNT + inputs[*input], *rank, register]);
						tape.extend_from_slice(dimensions);
						tape.extend_from_slice(strides);
						register
					}
				});
			}
			let second = args.get(1).copied().unwrap_or(0);
			tape.extend_from_slice(&[op.opcode, args[0], second, dst]);
			match op.opcode {
				4 => {
					tape.extend_from_slice(&op.parameters);
					tape.push(0);
				}
				47..=48 => {
					tape.extend_from_slice(&[op.parameters[0], 0, 0, 0]);
				}
				_ => {}
			}
			free.extend(temporaries);
			registers[index] = dst;
			for (slot, &value) in self.outputs.iter().enumerate() {
				if value == index {
					tape.extend_from_slice(&[2, dst, REGISTER_COUNT + outputs[slot], 0]);
				}
			}
			for (value, &last) in last_use.iter().enumerate().take(index + 1) {
				if last == index {
					free.push(registers[value]);
				}
			}
			if tape.len() / 4 > MAX_TAPE_STEPS {
				return None;
			}
		}
		Some(tape)
	}

	/// Full normalized fusion signature, including extents and input sizes.
	/// Route observations key on this signature through
	/// `vkcache::jit::signature_key`; artifact identity stays extent-free.
	pub fn canonical_key(&self) -> Vec<u8> {
		self.encode_key(false)
	}

	/// Emitted-code identity: structure, constants, binding count and outputs.
	/// Element count and input byte sizes are admission and dispatch facts;
	/// they never change the generated Slang text, so extents stay outside
	/// the artifact key and one compile serves every legal element count.
	pub fn artifact_key(&self) -> Vec<u8> {
		self.encode_key(true)
	}

	fn encode_key(&self, artifact: bool) -> Vec<u8> {
		let mut bytes = if artifact {
			b"oa.elementwise.artifact.v1\0".to_vec()
		} else {
			b"oa.elementwise.v1\0".to_vec()
		};
		let mut word = |value: u64| bytes.extend_from_slice(&value.to_le_bytes());
		if !artifact {
			word(u64::from(self.element_count));
		}
		word(self.input_bytes.len() as u64);
		if !artifact {
			for &size in &self.input_bytes {
				word(size);
			}
		}
		word(self.instructions.len() as u64);
		for op in &self.instructions {
			word(u64::from(op.opcode));
			word(op.sources.len() as u64);
			for source in &op.sources {
				match source {
					Source::Input(slot) => {
						word(0);
						word(*slot as u64);
					}
					Source::Value(value) => {
						word(1);
						word(*value as u64);
					}
					Source::Constant(bits) => {
						word(2);
						word(u64::from(*bits));
					}
					Source::Broadcast {
						input,
						rank,
						dimensions,
						strides,
					} => {
						word(3);
						word(*input as u64);
						word(u64::from(*rank));
						for &value in dimensions.iter().chain(strides) {
							word(u64::from(value));
						}
					}
				}
			}
			word(op.parameters.len() as u64);
			for &value in &op.parameters {
				word(u64::from(value));
			}
		}
		word(self.outputs.len() as u64);
		for &value in &self.outputs {
			word(value as u64);
		}
		bytes
	}
}

#[cfg(test)]
#[path = "../../../../test/rs/runtime/fusion_ir_unit.rs"]
mod tests;
