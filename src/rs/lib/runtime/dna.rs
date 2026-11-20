//! Cross-domain compute planning, fusion and compilation.
//!
//! Owns compiler transformations and provider policy, not execution lifetime.
//! Engine owns orchestration; Device owns native resources and publication.
//! Shared semantic/executable representations live under runtime::graph.

pub(super) mod compiler;
pub(super) mod fusion;
pub(super) mod ir;
pub(super) mod kernel;
pub(super) mod provider;
pub(super) mod tile;
pub(super) mod tuning;

pub(super) use provider::dnn::{DnnPlan, DnnPolicy};

/// Keep a staged plane copy and its buffer-only consumers on the same run.
pub(super) fn schedule_video_plane_returns(nodes: &mut [super::graph::executable::GraphNode]) {
	use super::{
		device::video::PlaneCopyPhase,
		graph::executable::{GraphNode, VideoPlaneNode},
	};

	let mut index = 0;
	while index < nodes.len() {
		if matches!(
			nodes[index],
			GraphNode::VideoPlanes(VideoPlaneNode {
				phase: PlaneCopyPhase::Acquire,
				..
			})
		) {
			// The copy recorded the release back to decode already.
			// Postpone acquisition only across buffer-only compute work.
			let following = nodes[index + 1..]
				.iter()
				.take_while(|node| node.compute().is_some())
				.count();
			nodes[index..=index + following].rotate_left(1);
			index += following;
		}
		index += 1;
	}
}

/// Select native buffer copies after the compute fusers have consumed source dispatches.
pub(super) fn lower_buffer_copies(
	graph: &mut super::graph::executable::ExecutableGraph,
) -> crate::Result<()> {
	use super::graph::executable::{BufferCopyNode, ExecutableGraph, GraphNode};

	if graph
		.compute_nodes()
		.any(|node| node.kernel == kernel::KernelId::MatrixCopyF32)
	{
		let families = graph.queue_families();
		let queue = families.matrix_copy_affinity();
		let nodes = graph
			.nodes()
			.iter()
			.map(|node| {
				node
					.compute()
					.and_then(|node| BufferCopyNode::from_whole_copy(node, queue))
					.map_or_else(|| node.clone(), GraphNode::BufferCopy)
			})
			.collect();
		*graph = ExecutableGraph::from_nodes(nodes, families)?;
	}
	Ok(())
}
