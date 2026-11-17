//! Host-only ownership/SSA oracles; resource numbers stand for physical
//! descriptor identities, so these proofs do not require a Vulkan device.

use super::*;
use crate::runtime::dna::kernel::{KernelEffects, KernelId};

fn compute(kernel: KernelId) -> ComputeNode {
	ComputeNode {
		operation: kernel.report_name(),
		kernel,
		buffers: Vec::new(),
		push_constants: Vec::new(),
		workgroups: [1, 1, 1],
		indirect: None,
		semantic_ops: Vec::new(),
		op_contract_hash: 0,
		jit: None,
		fusion_signature: None,
	}
}

#[test]
fn generated_effects_block_rng_state_and_atomic_nodes_without_read_write_bindings() {
	for (kernel, effect) in [
		(KernelId::MatrixPhiloxUniformF32, KernelEffects::RANDOM),
		(KernelId::MlAdamWGraphF32, KernelEffects::STATE_MUTATION),
		(KernelId::VisionConfusionMatrixI32, KernelEffects::ATOMIC),
		(KernelId::MatrixAddI32, KernelEffects::UNKNOWN),
	] {
		assert!(kernel.fusion_effects().contains(effect));
		let access = NodeAccess::from_compute(&compute(kernel));
		assert!(access.opaque);
		let graph = RegionGraph::new(vec![access.clone(), node(&[0], &[1])], BTreeSet::new());
		assert!(matches!(
			graph.contiguous(0..2, &BTreeSet::new(), &WriteAdmission::default()),
			Err(RegionDecline::EffectBoundary)
		));
		let candidate = chain(&[3])
			.contiguous(0..2, &BTreeSet::new(), &WriteAdmission::default())
			.unwrap();
		assert_eq!(
			candidate.accepts(&access, &WriteAdmission::default()),
			Err(RegionDecline::EffectBoundary)
		);
	}
}

#[test]
fn explicit_audit_allows_resource_proofs_but_does_not_establish_alias_legality() {
	for kernel in [
		KernelId::MatrixAddF32,
		KernelId::MatrixMulBroadcastF32,
		KernelId::MlLinearF32,
		KernelId::MlGroupedBackwardPrologueF32,
	] {
		assert!(kernel.fusion_effects().is_empty());
		assert!(!NodeAccess::from_compute(&compute(kernel)).opaque);
	}
	// An audited kernel binding a resource ReadWrite reaches the versioned
	// write admission — but reading a resource nobody wrote first is still
	// not a legal version.
	let mut audited = NodeAccess::from_compute(&compute(KernelId::MatrixAddF32));
	audited.bindings = vec![(0, BufferAccess::ReadWrite)];
	let graph = RegionGraph::new(vec![audited, node(&[0], &[1])], BTreeSet::new());
	assert!(matches!(
		graph.contiguous(0..2, &BTreeSet::new(), &WriteAdmission::default()),
		Err(RegionDecline::NonSsaWrite)
	));
}

fn node(reads: &[u32], writes: &[u32]) -> NodeAccess {
	NodeAccess {
		bindings: reads
			.iter()
			.map(|&id| (id, BufferAccess::Read))
			.chain(writes.iter().map(|&id| (id, BufferAccess::Write)))
			.collect(),
		opaque: false,
	}
}

fn chain(external: &[u32]) -> RegionGraph {
	RegionGraph::new(
		vec![node(&[0, 1], &[2]), node(&[2], &[3])],
		external.iter().copied().collect(),
	)
}

#[test]
fn unobserved_intermediate_can_stay_inside_lowering() {
	let graph = chain(&[3]);
	let candidate = graph
		.contiguous(0..2, &BTreeSet::new(), &WriteAdmission::default())
		.unwrap();
	assert_eq!(candidate.inputs, BTreeSet::from([0, 1]));
	assert_eq!(candidate.materialize, BTreeSet::from([3]));
	assert_eq!(
		candidate.accepts(&node(&[0, 1], &[3]), &WriteAdmission::default()),
		Ok(())
	);
}

#[test]
fn external_owner_including_autograd_save_requires_real_store() {
	let graph = chain(&[2, 3]);
	let candidate = graph
		.contiguous(0..2, &BTreeSet::new(), &WriteAdmission::default())
		.unwrap();
	assert_eq!(
		candidate.accepts(&node(&[0, 1], &[3]), &WriteAdmission::default()),
		Err(RegionDecline::MissingMaterialization)
	);
	assert_eq!(
		candidate.accepts(&node(&[0, 1], &[2, 3]), &WriteAdmission::default()),
		Ok(())
	);
}

#[test]
fn outside_reader_requires_store_even_without_external_owner() {
	let graph = RegionGraph::new(
		vec![node(&[0], &[1]), node(&[1], &[2]), node(&[1], &[3])],
		BTreeSet::new(),
	);
	let candidate = graph
		.contiguous(0..2, &BTreeSet::new(), &WriteAdmission::default())
		.unwrap();
	assert_eq!(candidate.materialize, BTreeSet::from([1]));
	assert_eq!(
		candidate.accepts(&node(&[0], &[2]), &WriteAdmission::default()),
		Err(RegionDecline::MissingMaterialization)
	);
}

#[test]
fn absorbed_anchor_or_later_slot_cannot_be_claimed_twice() {
	let graph = chain(&[3]);
	for index in [0, 1] {
		assert!(matches!(
			graph.contiguous(0..2, &BTreeSet::from([index]), &WriteAdmission::default()),
			Err(RegionDecline::ConsumedSlot)
		));
	}
	assert!(
		graph
			.contiguous(0..2, &BTreeSet::from([2]), &WriteAdmission::default())
			.is_ok()
	);
}

#[test]
fn malformed_ranges_decline_without_slicing_or_arithmetic_panics() {
	let graph = chain(&[3]);
	for (start, end) in [(0, 0), (0, 1), (1, 0), (0, 3), (usize::MAX, usize::MAX)] {
		assert!(matches!(
			graph.contiguous(start..end, &BTreeSet::new(), &WriteAdmission::default()),
			Err(RegionDecline::InvalidRange)
		));
	}
}

#[test]
fn earlier_reader_is_not_a_version_but_a_later_writer_is() {
	// A reader before the resource's first writer reads unwritten content.
	let graph = RegionGraph::new(
		vec![node(&[1], &[4]), node(&[0], &[1]), node(&[1], &[2])],
		BTreeSet::new(),
	);
	assert!(matches!(
		graph.contiguous(1..3, &BTreeSet::new(), &WriteAdmission::default()),
		Err(RegionDecline::NonSsaWrite)
	));
	// A writer strictly after the region simply overwrites the region's
	// version — the ordering is preserved and the region may be admitted.
	let graph = RegionGraph::new(
		vec![node(&[0], &[1]), node(&[1], &[2]), node(&[0], &[1])],
		BTreeSet::new(),
	);
	assert!(
		graph
			.contiguous(0..2, &BTreeSet::new(), &WriteAdmission::default())
			.is_ok()
	);
	// A writer *between* the region's positions but outside them breaks the
	// value ordering the single emission preserves: decline.
	let positions = BTreeSet::from([0, 2]);
	assert!(matches!(
		graph.window(&positions, &BTreeSet::new(), &WriteAdmission::default()),
		Err(RegionDecline::NonSsaWrite)
	));
}

#[test]
fn versioned_read_modify_write_on_a_region_input_is_admitted() {
	// The conv_1d_bias_tail shape: producer writes `projected` before the
	// region, the first member updates it in place (ReadWrite), the second
	// reads the new version.
	let mut rmw = node(&[], &[]);
	rmw.bindings = vec![(1, BufferAccess::ReadWrite), (9, BufferAccess::Read)];
	let graph = RegionGraph::new(
		vec![node(&[7], &[1]), rmw, node(&[1], &[2])],
		BTreeSet::new(),
	);
	let candidate = graph
		.contiguous(1..3, &BTreeSet::new(), &WriteAdmission::default())
		.unwrap();
	// `projected` is versioned: it is a write but not a strict input.
	assert!(candidate.writes.contains(&1));
	assert!(!candidate.inputs.contains(&1));
	assert!(candidate.inputs.contains(&9));
	// The emission may read the region's own version back.
	assert_eq!(
		candidate.accepts(&node(&[1, 9], &[2]), &WriteAdmission::default()),
		Ok(())
	);
	// An unobservable versioned write may be elided; an external owner or
	// outside reader requires republishing the new version.
	let mut rmw = node(&[], &[]);
	rmw.bindings = vec![(1, BufferAccess::ReadWrite), (9, BufferAccess::Read)];
	let graph = RegionGraph::new(
		vec![node(&[7], &[1]), rmw, node(&[1], &[2]), node(&[1], &[3])],
		BTreeSet::new(),
	);
	let candidate = graph
		.contiguous(1..3, &BTreeSet::new(), &WriteAdmission::default())
		.unwrap();
	assert!(candidate.materialize.contains(&1));
}

#[test]
fn partitioned_writes_admit_disjoint_extents_with_ordered_readers() {
	// The concat_linear shape: three parts write disjoint extents of the
	// shared buffer, the consumer reads it after all writes.
	let concat = 5;
	let nodes = vec![
		node(&[0], &[concat]),
		node(&[1], &[concat]),
		node(&[2], &[concat]),
		node(&[concat, 6], &[7]),
	];
	let admission = WriteAdmission::partitioned([concat]);
	let graph = RegionGraph::new(nodes.clone(), BTreeSet::new());
	assert!(graph.contiguous(0..4, &BTreeSet::new(), &admission).is_ok());
	// Undeclared, in-region writers ordered before their sole consumer are
	// a sequential version chain — the reader sees the completed buffer.
	let graph = RegionGraph::new(nodes.clone(), BTreeSet::new());
	assert!(
		graph
			.contiguous(0..4, &BTreeSet::new(), &WriteAdmission::default())
			.is_ok()
	);
	// A reader between writers observes a half-written buffer: decline
	// under the declaration and undeclared alike.
	let mut gap_reader = nodes.clone();
	gap_reader.insert(2, node(&[concat], &[8]));
	for admission in [
		WriteAdmission::partitioned([concat]),
		WriteAdmission::default(),
	] {
		let graph = RegionGraph::new(gap_reader.clone(), BTreeSet::new());
		assert!(matches!(
			graph.contiguous(0..5, &BTreeSet::new(), &admission),
			Err(RegionDecline::NonSsaWrite)
		));
	}
	// A hidden writer outside the region: decline.
	let mut hidden = nodes.clone();
	hidden.push(node(&[0], &[concat]));
	let graph = RegionGraph::new(hidden, BTreeSet::new());
	assert!(matches!(
		graph.contiguous(0..4, &BTreeSet::new(), &admission),
		Err(RegionDecline::NonSsaWrite)
	));
}

#[test]
fn declared_identity_writes_on_inputs_are_observable_neutral() {
	// The single-head SDPA shape: sources only read Q/K/V; the emission
	// reads and republishes them at identical addresses — a declared
	// identity write, not an aliased clobber.
	let graph = RegionGraph::new(
		vec![
			node(&[0, 1], &[3]),
			node(&[3, 3], &[4]),
			node(&[4, 2], &[5]),
		],
		BTreeSet::from([0, 1, 2, 5]),
	);
	let candidate = graph
		.contiguous(0..3, &BTreeSet::new(), &WriteAdmission::default())
		.unwrap();
	// Without the declaration a write on an input is an aliased write.
	let mut emission = node(&[0, 1, 2], &[3, 4, 5]);
	emission.bindings.extend([(0, BufferAccess::Write)]);
	assert_eq!(
		candidate.accepts(&emission, &WriteAdmission::default()),
		Err(RegionDecline::AliasedWrite)
	);
	// Declared identity republish on the exact input: admitted.
	assert_eq!(
		candidate.accepts(&emission, &WriteAdmission::identity([0])),
		Ok(())
	);
	// Identity does not license a write on a resource that is not a strict
	// input, and an undeclared resource still declines.
	let mut emission = node(&[0, 1, 2], &[3, 4, 5]);
	emission.bindings.extend([(7, BufferAccess::Write)]);
	assert_eq!(
		candidate.accepts(&emission, &WriteAdmission::identity([7])),
		Err(RegionDecline::UnexpectedWrite)
	);
}

#[test]
fn same_node_read_write_alias_is_rejected_even_with_separate_bindings() {
	let graph = RegionGraph::new(vec![node(&[0, 1], &[1]), node(&[1], &[2])], BTreeSet::new());
	assert!(matches!(
		graph.contiguous(0..2, &BTreeSet::new(), &WriteAdmission::default()),
		Err(RegionDecline::NonSsaWrite)
	));
}

#[test]
fn effectful_nodes_are_hard_boundaries() {
	let effect = NodeAccess {
		opaque: true,
		..node(&[0, 1], &[2])
	};
	let graph = RegionGraph::new(vec![effect, node(&[2], &[3])], BTreeSet::from([3]));
	assert!(matches!(
		graph.contiguous(0..2, &BTreeSet::new(), &WriteAdmission::default()),
		Err(RegionDecline::EffectBoundary)
	));
}

#[test]
fn unbound_reads_cannot_become_fused_inputs() {
	let graph = chain(&[3]);
	let candidate = graph
		.contiguous(0..2, &BTreeSet::new(), &WriteAdmission::default())
		.unwrap();
	// An eliminated source intermediate is not available to the replacement.
	// It must publish the intermediate itself before it may read it.
	assert_eq!(
		candidate.accepts(&node(&[2], &[3]), &WriteAdmission::default()),
		Err(RegionDecline::UnavailableRead)
	);
	assert_eq!(
		candidate.accepts(&node(&[7], &[3]), &WriteAdmission::default()),
		Err(RegionDecline::UnavailableRead)
	);
}

#[test]
fn sibling_reads_require_a_published_version_in_emission_order() {
	let candidate = chain(&[3])
		.contiguous(0..2, &BTreeSet::new(), &WriteAdmission::default())
		.unwrap();
	let admission = WriteAdmission::default();
	assert_eq!(
		candidate.accepts_ordered(&[node(&[0, 1], &[2]), node(&[2], &[3])], &admission),
		Ok(())
	);
	assert_eq!(
		candidate.accepts_ordered(&[node(&[2], &[3]), node(&[0, 1], &[2])], &admission),
		Err(RegionDecline::UnavailableRead)
	);
	assert_eq!(
		candidate.accepts_ordered(&[node(&[2], &[3])], &admission),
		Err(RegionDecline::UnavailableRead)
	);
}

#[test]
fn unexpected_duplicate_and_operand_alias_writes_decline() {
	let graph = chain(&[3]);
	let candidate = graph
		.contiguous(0..2, &BTreeSet::new(), &WriteAdmission::default())
		.unwrap();
	assert_eq!(
		candidate.accepts(&node(&[0, 1], &[3, 7]), &WriteAdmission::default()),
		Err(RegionDecline::UnexpectedWrite)
	);
	// Two write bindings on one published resource are the same kernel's
	// declared contract (disjoint extents), not a writer collision.
	assert_eq!(
		candidate.accepts(&node(&[0, 1], &[3, 3]), &WriteAdmission::default()),
		Ok(())
	);
	// A write to a strict region input without an identity declaration
	// stays an aliased write.
	assert_eq!(
		candidate.accepts(&node(&[0, 1], &[0, 3]), &WriteAdmission::default()),
		Err(RegionDecline::AliasedWrite)
	);
}

#[test]
fn repeated_write_bindings_to_one_published_resource_are_admitted() {
	// The small_causal_sdpa shape: the fused node's context and merge
	// outputs share one storage descriptor with disjoint extents — two
	// write bindings on resource 3 are one kernel's declared write
	// contract, not a collision between writers.
	let graph = RegionGraph::new(
		vec![
			node(&[0], &[1]),
			node(&[1], &[2]),
			node(&[2], &[3]),
			node(&[3], &[4]),
		],
		BTreeSet::from([4]),
	);
	let candidate = graph
		.contiguous(0..4, &BTreeSet::new(), &WriteAdmission::default())
		.unwrap();
	assert_eq!(
		candidate.accepts(&node(&[0], &[1, 2, 3, 3, 4]), &WriteAdmission::default()),
		Ok(())
	);
}

#[test]
fn repeated_read_binding_is_not_a_second_writer() {
	let graph = RegionGraph::new(
		vec![node(&[0, 0], &[1]), node(&[1], &[2])],
		BTreeSet::from([2]),
	);
	let candidate = graph
		.contiguous(0..2, &BTreeSet::new(), &WriteAdmission::default())
		.unwrap();
	assert_eq!(
		candidate.accepts(&node(&[0, 0], &[1, 2]), &WriteAdmission::default()),
		Ok(())
	);
}

#[test]
fn exhaustive_small_chains_preserve_exactly_the_observable_writes() {
	// Independent set oracle: enumerate external owners, outside consumers
	// and every replacement store set for a three-definition SSA chain.
	for external_mask in 0..8u32 {
		for reader_mask in 0..8u32 {
			let mut nodes = vec![node(&[0], &[1]), node(&[1], &[2]), node(&[2], &[3])];
			let outside_reads = (1..=3)
				.filter(|id| reader_mask & (1 << (id - 1)) != 0)
				.collect::<Vec<_>>();
			nodes.push(node(&outside_reads, &[4]));
			let external = (1..=3)
				.filter(|id| external_mask & (1 << (id - 1)) != 0)
				.collect();
			let graph = RegionGraph::new(nodes, external);
			let candidate = graph
				.contiguous(0..3, &BTreeSet::new(), &WriteAdmission::default())
				.unwrap();
			let required_mask = external_mask | reader_mask;
			for stores in 0..8u32 {
				let writes = (1..=3)
					.filter(|id| stores & (1 << (id - 1)) != 0)
					.collect::<Vec<_>>();
				assert_eq!(
					candidate
						.accepts(&node(&[0], &writes), &WriteAdmission::default())
						.is_ok(),
					stores & required_mask == required_mask
				);
			}
		}
	}
}
