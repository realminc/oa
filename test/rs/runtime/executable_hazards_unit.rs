use super::*;

#[test]
fn exclusive_queue_changes_depend_on_the_last_family_even_for_readers() {
	let read = AccessState::from_access(BufferAccess::Read);
	let write = AccessState::from_access(BufferAccess::Write);
	let accesses = [
		BTreeMap::from([(7, write)]),
		BTreeMap::from([(7, read)]),
		BTreeMap::from([(7, read), (8, write)]),
	];
	// Compute, decode and encode resolved to distinct families.
	let families = [0, 1, 2];

	let plan = plan_access_states(&accesses, |index| families[index]);
	assert!(plan[0].is_empty());
	assert_eq!(
		plan[1],
		[PlannedHazard {
			source_nodes: [0, 0],
			resource: 7,
			source: write,
			destination: read,
		}]
	);
	assert_eq!(
		plan[2],
		[PlannedHazard {
			source_nodes: [1, 1],
			resource: 7,
			source: read,
			destination: read,
		}]
	);
}

#[test]
fn later_indirect_reads_depend_on_the_writer_and_writes_order_both_reader_stages() {
	let read = AccessState::from_access(BufferAccess::Read);
	let write = AccessState::from_access(BufferAccess::Write);
	let indirect = AccessState::INDIRECT;
	let accesses =
		[write, read, indirect, indirect, read, write].map(|access| BTreeMap::from([(7, access)]));

	let plan = plan_access_states(&accesses, |_| 0);
	assert!(plan[0].is_empty());
	assert_eq!(plan[1][0].source_nodes, [0, 0]);
	assert_eq!(
		plan[2],
		[PlannedHazard {
			source_nodes: [0, 0],
			resource: 7,
			source: write,
			destination: indirect,
		}]
	);
	assert!(plan[3].is_empty());
	assert!(plan[4].is_empty());
	let mut shader_and_indirect_reads = read;
	shader_and_indirect_reads.merge(indirect);
	assert_eq!(
		plan[5],
		[PlannedHazard {
			source_nodes: [1, 4],
			resource: 7,
			source: shader_and_indirect_reads,
			destination: write,
		}]
	);
}

#[test]
fn queue_runs_split_only_at_resolved_family_transitions() {
	assert_eq!(
		queue_runs(&[0, 0, 0]),
		[QueueRun {
			family: 0,
			nodes: 0..3
		}]
	);
	assert_eq!(
		queue_runs(&[0, 0, 1, 1, 2]),
		[
			QueueRun {
				family: 0,
				nodes: 0..2
			},
			QueueRun {
				family: 1,
				nodes: 2..4
			},
			QueueRun {
				family: 2,
				nodes: 4..5
			},
		]
	);
	// A family recurs in several runs — decode, compute, encode would reuse
	// one video family twice.
	assert_eq!(
		queue_runs(&[1, 0, 1]),
		[
			QueueRun {
				family: 1,
				nodes: 0..1
			},
			QueueRun {
				family: 0,
				nodes: 1..2
			},
			QueueRun {
				family: 1,
				nodes: 2..3
			},
		]
	);
}

#[test]
fn queue_families_resolve_roles_to_enabled_queues() {
	let families = QueueFamilies {
		compute: 0,
		transfer: None,
		video_decode: Some(1),
		video_encode: Some(1),
	};
	assert_eq!(families.resolve(QueueAffinity::Compute).unwrap(), 0);
	// Transfer falls back to the compute family until a dedicated pool exists.
	assert_eq!(families.resolve(QueueAffinity::Transfer).unwrap(), 0);
	assert_eq!(families.matrix_copy_affinity(), QueueAffinity::Transfer);
	assert_eq!(families.resolve(QueueAffinity::VideoDecode).unwrap(), 1);
	assert_eq!(families.resolve(QueueAffinity::VideoEncode).unwrap(), 1);

	// A dedicated transfer family wins once enabled.
	let dedicated = QueueFamilies {
		compute: 0,
		transfer: Some(2),
		video_decode: None,
		video_encode: None,
	};
	assert_eq!(dedicated.resolve(QueueAffinity::Transfer).unwrap(), 2);
	assert_eq!(dedicated.matrix_copy_affinity(), QueueAffinity::Compute);
	// Video roles never fall back silently — they fail at capture.
	assert!(dedicated.resolve(QueueAffinity::VideoDecode).is_err());
	assert!(dedicated.resolve(QueueAffinity::VideoEncode).is_err());
}

#[test]
fn compute_copy_compute_resolves_to_one_run_on_the_compute_family() {
	let families = QueueFamilies {
		compute: 3,
		transfer: None,
		video_decode: None,
		video_encode: None,
	};
	let node = ComputeNode {
		operation: "test",
		kernel: KernelId::MatrixAddF32,
		buffers: Vec::new(),
		push_constants: Vec::new(),
		workgroups: [1, 1, 1],
		indirect: None,
		semantic_ops: Vec::new(),
		op_contract_hash: 0,
		jit: None,
		fusion_signature: None,
	};
	let graph = ExecutableGraph::from_nodes(vec![GraphNode::Compute(node)], families).unwrap();
	assert_eq!(graph.node_families(), &[3]);
	assert_eq!(
		graph.queue_runs(),
		[QueueRun {
			family: 3,
			nodes: 0..1
		}]
	);
}

/// A graph run whose resolved family has no enabled queue or pool must
/// fail at recording — fail-closed, never silently re-queued.
#[test]
#[ignore = "requires an admitted Vulkan compute device"]
fn a_run_resolving_to_an_unenabled_family_fails_at_recording() -> Result<()> {
	let engine = match crate::Engine::new() {
		Ok(engine) => engine,
		Err(_) => return Ok(()),
	};
	let device = engine.handle().device();
	let source = Buffer::host_visible_storage(&device, 64)?;
	let destination = Buffer::host_visible_storage(&device, 64)?;
	// No device enables a queue family 0xffffffff.
	let families = QueueFamilies {
		compute: device.physical().compute_queue_family,
		transfer: Some(u32::MAX),
		video_decode: None,
		video_encode: None,
	};
	let graph = ExecutableGraph::from_nodes(
		vec![GraphNode::BufferCopy(BufferCopyNode {
			operation: "test-copy",
			semantic_ops: Vec::new(),
			op_contract_hash: 0,
			queue: QueueAffinity::Transfer,
			buffers: [
				BufferUse {
					buffer: source,
					access: BufferAccess::Read,
				},
				BufferUse {
					buffer: destination,
					access: BufferAccess::Write,
				},
			],
			bytes: 64,
		})],
		families,
	)?;
	assert_eq!(
		graph.queue_runs(),
		[QueueRun {
			family: u32::MAX,
			nodes: 0..1
		}]
	);
	let error = match device.record_reusable_graph_range(&graph, 0..1) {
		Err(error) => error,
		Ok(_) => panic!("no queue or pool serves the resolved family"),
	};
	assert_eq!(error.kind(), crate::ErrorKind::MissingCapability);
	Ok(())
}
