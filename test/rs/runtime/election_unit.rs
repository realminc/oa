use super::{DeviceCapacityKey, capacity_frontier, preferred_capacity_index};

fn capacity(shared_memory_bytes: u32, local_heap_bytes: u64) -> DeviceCapacityKey {
	DeviceCapacityKey {
		shader_float16: false,
		shader_float64: false,
		shader_integer_dot_product: false,
		cooperative_matrix_compute: false,
		shared_memory_bytes,
		workgroup_invocations: 256,
		local_heap_bytes,
		storage_buffer_descriptors: 4096,
	}
}

#[test]
fn full_frontier_excludes_candidates_dominated_after_an_incomparable_one() {
	let capacities = [capacity(10, 1), capacity(1, 10), capacity(11, 2)];
	assert_eq!(capacity_frontier(&capacities), [false, true, true]);
	assert_eq!(preferred_capacity_index(&capacities, &[0, 1, 2]), Some(2));
}

#[test]
fn equal_capacities_prefer_original_enumeration_ordinal() {
	let capacities = [capacity(10, 10), capacity(10, 10)];
	assert_eq!(capacity_frontier(&capacities), [true, true]);
	assert_eq!(preferred_capacity_index(&capacities, &[4, 2]), Some(1));
	assert_eq!(preferred_capacity_index(&[], &[]), None);
}
