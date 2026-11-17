//! Dynamic-artifact SPIR-V Tools and exact ABI admission.
//! Donor: dna/kernel/jit.rs; stricter than static artifact reflection.
use super::jit_error;
use crate::Result;
use std::{
	collections::{BTreeMap, BTreeSet},
	io::Write,
	process::{Command, Stdio},
};

pub(super) fn validate_with_spirv_tools(words: &[u32], bounded: bool) -> Result<()> {
	let executable = std::env::var_os("SPIRV_VAL").unwrap_or_else(|| "spirv-val".into());
	let mut child = Command::new(executable)
		.args([
			"--target-env",
			if bounded { "vulkan1.2" } else { "vulkan1.3" },
			"-",
		])
		.stdin(Stdio::piped())
		.stdout(Stdio::piped())
		.stderr(Stdio::piped())
		.spawn()
		.map_err(jit_error)?;
	let bytes = words
		.iter()
		.flat_map(|word| word.to_le_bytes())
		.collect::<Vec<_>>();
	let write = child.stdin.take().expect("piped stdin").write_all(&bytes);
	let output = child.wait_with_output().map_err(jit_error)?;
	write.map_err(jit_error)?;
	if !output.status.success() {
		return Err(jit_error(format!(
			"SPIR-V validation: {}",
			String::from_utf8_lossy(&output.stderr)
		)));
	}
	Ok(())
}

// Fail closed rather than treating the general static artifact reflector as
// complete validation of arbitrary generated shaders.
pub(in crate::runtime) fn validate_program_abi(
	words: &[u32],
	bounded: bool,
	expected_bindings: usize,
) -> Result<()> {
	if !(2..=31).contains(&expected_bindings)
		|| words.len() < 5
		|| words[0] != 0x0723_0203
		|| words[1] != if bounded { 0x0001_0500 } else { 0x0001_0600 }
		|| crate::runtime::dna::kernel::reflect_push_constant_size(words)?
			!= ((expected_bindings + 1) * 4) as u32
	{
		return Err(jit_error("unexpected module header or push ABI"));
	}
	let mut entries = 0;
	let mut entry_id = None;
	let mut local_sizes = 0;
	let mut local_size_id = None;
	let mut bindings = Vec::new();
	let mut sets = Vec::new();
	let mut arrays = BTreeMap::new();
	let mut pointers = BTreeMap::new();
	let mut variables = BTreeMap::new();
	let mut constants = BTreeMap::new();
	let mut unsigned_words = BTreeSet::new();
	let mut structures = BTreeMap::new();
	let mut offsets = BTreeMap::new();
	let mut cursor = 5;
	while cursor < words.len() {
		let count = (words[cursor] >> 16) as usize;
		if count == 0 || count > words.len() - cursor {
			return Err(jit_error("invalid instruction"));
		}
		let instruction = &words[cursor..cursor + count];
		match instruction[0] as u16 {
			15 => {
				// OpEntryPoint: GLCompute, main
				if count < 5
					|| instruction[1] != 5
					|| instruction[3] != u32::from_le_bytes(*b"main")
					|| instruction[4] != 0
				{
					return Err(jit_error("unexpected entry point"));
				}
				entries += 1;
				entry_id = Some(instruction[2]);
			}
			16 if count == 6 && instruction[2..] == [17, 256, 1, 1] => {
				local_sizes += 1;
				local_size_id = Some(instruction[1]);
			}
			17 => {
				// Capability Shader, plus RuntimeDescriptorArray only on Core.
				if count != 2 || !(instruction[1] == 1 || (!bounded && instruction[1] == 5302)) {
					return Err(jit_error("unexpected shader capability"));
				}
			}
			71 if count == 4 && instruction[2] == 33 => bindings.push((instruction[1], instruction[3])),
			71 if count == 4 && instruction[2] == 34 => sets.push((instruction[1], instruction[3])),
			28 if count == 4 => {
				arrays.insert(instruction[1], Some(instruction[3]));
			}
			29 if count == 3 => {
				arrays.insert(instruction[1], None);
			}
			32 if count == 4 => {
				pointers.insert(instruction[1], (instruction[2], instruction[3]));
			}
			43 if count == 4 => {
				constants.insert(instruction[2], instruction[3]);
			}
			59 if count >= 4 => {
				variables.insert(instruction[2], (instruction[1], instruction[3]));
			}
			21 if count == 4 && instruction[2..] == [32, 0] => {
				unsigned_words.insert(instruction[1]);
			}
			30 if count >= 2 => {
				structures.insert(instruction[1], instruction[2..].to_vec());
			}
			72 if count == 5 && instruction[3] == 35 => {
				offsets.insert((instruction[1], instruction[2]), instruction[4]);
			}
			_ => {}
		}
		cursor += count;
	}
	if entries != 1
		|| local_sizes != 1
		|| entry_id != local_size_id
		|| bindings.len() != 1
		|| sets != bindings
		|| bindings[0].1 != 0
	{
		return Err(jit_error("unexpected entry/workgroup/descriptor ABI"));
	}
	let &(pointer, storage_class) = variables
		.get(&bindings[0].0)
		.ok_or_else(|| jit_error("missing descriptor variable"))?;
	let &(pointer_class, array) = pointers
		.get(&pointer)
		.ok_or_else(|| jit_error("missing descriptor pointer"))?;
	let length = arrays
		.get(&array)
		.ok_or_else(|| jit_error("descriptor is not an array"))?;
	if storage_class != 12
		|| pointer_class != 12
		|| if bounded {
			length.and_then(|id| constants.get(&id).copied()) != Some(expected_bindings as u32)
		} else {
			length.is_some()
		} {
		return Err(jit_error(
			"unexpected descriptor storage class or array extent",
		));
	}
	let pushes = variables
		.values()
		.filter(|(_, class)| *class == 9)
		.collect::<Vec<_>>();
	if pushes.len() != 1 {
		return Err(jit_error("expected one push block"));
	}
	let &(class, structure) = pointers
		.get(&pushes[0].0)
		.ok_or_else(|| jit_error("missing push pointer"))?;
	let fields = structures
		.get(&structure)
		.ok_or_else(|| jit_error("missing push structure"))?;
	if class != 9
		|| fields.len() != expected_bindings + 1
		|| fields.iter().any(|field| !unsigned_words.contains(field))
		|| (0..expected_bindings as u32 + 1)
			.any(|field| offsets.get(&(structure, field)) != Some(&(field * 4)))
	{
		return Err(jit_error("unexpected packed U32 push fields"));
	}
	Ok(())
}
