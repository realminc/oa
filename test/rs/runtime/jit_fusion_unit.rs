use super::*;

/// Diagnostic tape timing above the production cutoff, using identical legality.
pub(in crate::runtime) fn diagnostic_elementwise_chain(
	device: &Arc<Device>,
	nodes: Vec<ComputeNode>,
	refs: &[&Storage],
) -> (Vec<ComputeNode>, Vec<Storage>) {
	fuse_elementwise_with_limit(device, nodes, refs, u32::MAX)
}

/// Use exactly the production matcher and observability checks, without tape limits.
pub(in crate::runtime) fn discover_programs(
	nodes: &[ComputeNode],
	refs: &[&Storage],
) -> Vec<(Program, Vec<Buffer>, Vec<Buffer>)> {
	let decoded = nodes.iter().map(decode_node).collect::<Vec<_>>();
	discover_regions(nodes, &decoded)
		.into_iter()
		.filter_map(|(start, end)| {
			if end - start < 2 {
				return None;
			}
			let candidate = build_program(nodes, &decoded, refs, start, end - 1).ok()?;
			Some((candidate.program, candidate.inputs, candidate.outputs))
		})
		.collect()
}
