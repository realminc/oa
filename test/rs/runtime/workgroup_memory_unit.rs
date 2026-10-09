use super::{KernelId, SPIRV_MAGIC, reflect_workgroup_memory_size};

fn instruction(words: &mut Vec<u32>, opcode: u32, operands: &[u32]) {
	words.push(((operands.len() as u32 + 1) << 16) | opcode);
	words.extend_from_slice(operands);
}

#[test]
fn sizes_nested_arrays_vectors_and_multiple_variables() {
	let mut words = vec![SPIRV_MAGIC, 0x10500, 0, 20, 0];
	instruction(&mut words, 21, &[1, 32, 0]);
	instruction(&mut words, 43, &[1, 2, 256]);
	instruction(&mut words, 43, &[1, 3, 2]);
	instruction(&mut words, 23, &[4, 1, 4]);
	instruction(&mut words, 28, &[5, 4, 2]);
	instruction(&mut words, 28, &[6, 5, 3]);
	instruction(&mut words, 32, &[7, 4, 6]);
	instruction(&mut words, 59, &[7, 8, 4]);
	instruction(&mut words, 32, &[9, 4, 1]);
	instruction(&mut words, 59, &[9, 10, 4]);
	assert_eq!(reflect_workgroup_memory_size(&words).unwrap(), 8196);
	instruction(&mut words, 59, &[7, 11, 4]);
	assert_eq!(reflect_workgroup_memory_size(&words).unwrap(), 16388);
}

#[test]
fn rejects_overflow_and_truncated_artifacts() {
	let mut words = vec![SPIRV_MAGIC, 0x10500, 0, 10, 0];
	instruction(&mut words, 21, &[1, 32, 0]);
	instruction(&mut words, 43, &[1, 2, u32::MAX]);
	instruction(&mut words, 28, &[3, 1, 2]);
	instruction(&mut words, 32, &[4, 4, 3]);
	instruction(&mut words, 59, &[4, 5, 4]);
	assert!(reflect_workgroup_memory_size(&words).is_err());
	words.pop();
	assert!(reflect_workgroup_memory_size(&words).is_err());
}

#[test]
fn sizes_every_registered_profile_artifact() -> crate::Result<()> {
	for kernel in KernelId::ALL {
		kernel.artifact().workgroup_memory_size()?;
		kernel.bounded_artifact().workgroup_memory_size()?;
	}
	Ok(())
}
