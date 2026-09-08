use std::collections::BTreeMap;

use crate::{Error, Result};

use super::{BufferAccess, ComputeDispatch, PushConstant, shader::KernelId, vk};

/// Owned snapshot of concrete work ready for Vulkan command recording.
pub(in crate::runtime) struct ExecutableGraph {
	nodes: Vec<ComputeNode>,
	barriers: Vec<Vec<BufferHazard>>,
}

pub(in crate::runtime) struct ComputeNode {
	pub(in crate::runtime) operation: &'static str,
	pub(in crate::runtime) kernel: KernelId,
	pub(in crate::runtime) buffers: Vec<BufferUse>,
	pub(in crate::runtime) push_constants: Vec<PushConstant>,
	pub(in crate::runtime) workgroups: [u32; 3],
}

#[derive(Clone)]
pub(in crate::runtime) struct BufferUse {
	pub(in crate::runtime) buffer: vk::Buffer,
	pub(in crate::runtime) access: BufferAccess,
}

pub(in crate::runtime) struct BufferHazard {
	pub(in crate::runtime) buffer: vk::Buffer,
	pub(in crate::runtime) source: AccessState,
	pub(in crate::runtime) destination: AccessState,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(in crate::runtime) struct AccessState {
	read: bool,
	write: bool,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct PlannedHazard<Resource> {
	resource: Resource,
	source: AccessState,
	destination: AccessState,
}

impl ExecutableGraph {
	/// Resolve borrowed domain dispatches into one transactionally built graph.
	pub(in crate::runtime) fn from_dispatches(
		device: &vk::Device,
		dispatches: &[ComputeDispatch<'_>],
	) -> Result<Self> {
		if dispatches.is_empty() {
			return Err(Error::invalid_argument(
				"an executable graph requires at least one dispatch",
			));
		}

		let mut nodes = Vec::with_capacity(dispatches.len());
		for dispatch in dispatches {
			let mut buffers = Vec::with_capacity(dispatch.buffers.len());
			for binding in dispatch.buffers {
				if !binding.storage.belongs_to(device) {
					return Err(Error::invalid_argument(format!(
						"{} buffers must belong to the submitting engine",
						dispatch.operation
					)));
				}
				let buffer = binding.storage.buffer().cloned().ok_or_else(|| {
					Error::backend_failure(
						"Vulkan",
						"compute-dispatch storage validation",
						std::io::Error::other(format!(
							"{} references storage without a Vulkan buffer",
							dispatch.operation
						)),
					)
				})?;
				buffers.push(BufferUse {
					buffer,
					access: binding.access,
				});
			}
			nodes.push(ComputeNode {
				operation: dispatch.operation,
				kernel: dispatch.kernel,
				buffers,
				push_constants: dispatch.push_constants.to_vec(),
				workgroups: dispatch.workgroups,
			});
		}

		let barriers = plan_barriers(&nodes)?;
		Ok(Self { nodes, barriers })
	}

	pub(in crate::runtime) fn join(graphs: Vec<Self>) -> Result<Self> {
		let nodes = graphs
			.into_iter()
			.flat_map(|graph| graph.nodes)
			.collect::<Vec<_>>();
		if nodes.is_empty() {
			return Err(Error::invalid_argument(
				"an executable graph requires at least one dispatch",
			));
		}
		let barriers = plan_barriers(&nodes)?;
		Ok(Self { nodes, barriers })
	}

	pub(in crate::runtime) fn nodes(&self) -> &[ComputeNode] {
		&self.nodes
	}

	pub(in crate::runtime) fn barriers_before(&self, node: usize) -> &[BufferHazard] {
		&self.barriers[node]
	}
}

impl AccessState {
	const fn from_access(access: BufferAccess) -> Self {
		match access {
			BufferAccess::Read => Self {
				read: true,
				write: false,
			},
			BufferAccess::Write => Self {
				read: false,
				write: true,
			},
		}
	}

	fn merge(&mut self, other: Self) {
		self.read |= other.read;
		self.write |= other.write;
	}

	const fn conflicts_with(self, next: Self) -> bool {
		self.write || next.write
	}

	pub(in crate::runtime) fn vk_access(self) -> ash::vk::AccessFlags2 {
		let mut flags = ash::vk::AccessFlags2::empty();
		if self.read {
			flags |= ash::vk::AccessFlags2::SHADER_STORAGE_READ;
		}
		if self.write {
			flags |= ash::vk::AccessFlags2::SHADER_STORAGE_WRITE;
		}
		flags
	}
}

fn plan_barriers(nodes: &[ComputeNode]) -> Result<Vec<Vec<BufferHazard>>> {
	let mut resources = BTreeMap::<u32, vk::Buffer>::new();
	let accesses = nodes
		.iter()
		.map(|node| {
			let bindings = node.buffers.iter().map(|buffer_use| {
				let identity = buffer_use.buffer.descriptor_index();
				resources
					.entry(identity)
					.or_insert_with(|| buffer_use.buffer.clone());
				(identity, buffer_use.access)
			});
			aggregate_accesses(bindings)
		})
		.collect::<Vec<_>>();

	plan_access_states(&accesses)
		.into_iter()
		.map(|hazards| {
			hazards
				.into_iter()
				.map(|hazard| {
					let buffer = resources.get(&hazard.resource).cloned().ok_or_else(|| {
						Error::backend_failure(
							"Vulkan",
							"executable-graph hazard planning",
							std::io::Error::other(
								"planned hazard lost its retained buffer identity",
							),
						)
					})?;
					Ok(BufferHazard {
						buffer,
						source: hazard.source,
						destination: hazard.destination,
					})
				})
				.collect::<Result<Vec<_>>>()
		})
		.collect()
}

fn aggregate_accesses(
	accesses: impl IntoIterator<Item = (u32, BufferAccess)>,
) -> BTreeMap<u32, AccessState> {
	let mut aggregate = BTreeMap::new();
	for (identity, access) in accesses {
		let access = AccessState::from_access(access);
		aggregate
			.entry(identity)
			.and_modify(|state: &mut AccessState| state.merge(access))
			.or_insert(access);
	}
	aggregate
}

fn plan_access_states<Resource: Copy + Ord>(
	nodes: &[BTreeMap<Resource, AccessState>],
) -> Vec<Vec<PlannedHazard<Resource>>> {
	let mut previous = BTreeMap::<Resource, AccessState>::new();
	let mut plan = Vec::with_capacity(nodes.len());

	for current in nodes {
		let mut barriers = Vec::new();
		for (resource, destination) in current {
			if let Some(source) = previous.get(resource)
				&& source.conflicts_with(*destination)
			{
				barriers.push(PlannedHazard {
					resource: *resource,
					source: *source,
					destination: *destination,
				});
			}
		}
		previous.extend(current);
		plan.push(barriers);
	}

	plan
}

#[cfg(test)]
mod tests {
	use super::{AccessState, aggregate_accesses, plan_access_states};
	use crate::runtime::BufferAccess;

	fn state(access: BufferAccess) -> AccessState {
		AccessState::from_access(access)
	}

	fn plan(nodes: &[&[(u32, BufferAccess)]]) -> Vec<Vec<super::PlannedHazard<u32>>> {
		let accesses = nodes
			.iter()
			.map(|node| aggregate_accesses(node.iter().copied()))
			.collect::<Vec<_>>();
		plan_access_states(&accesses)
	}

	#[test]
	fn derives_raw_war_and_waw_without_read_after_read_barriers() {
		let raw = plan(&[&[(1, BufferAccess::Write)], &[(1, BufferAccess::Read)]]);
		assert_eq!(raw[1].len(), 1);
		assert_eq!(raw[1][0].source, state(BufferAccess::Write));
		assert_eq!(raw[1][0].destination, state(BufferAccess::Read));

		let war = plan(&[&[(1, BufferAccess::Read)], &[(1, BufferAccess::Write)]]);
		assert_eq!(war[1].len(), 1);
		assert_eq!(war[1][0].source, state(BufferAccess::Read));
		assert_eq!(war[1][0].destination, state(BufferAccess::Write));

		let waw = plan(&[&[(1, BufferAccess::Write)], &[(1, BufferAccess::Write)]]);
		assert_eq!(waw[1].len(), 1);

		let rar = plan(&[&[(1, BufferAccess::Read)], &[(1, BufferAccess::Read)]]);
		assert!(rar[1].is_empty());
	}

	#[test]
	fn retains_state_across_unrelated_nodes_and_merges_aliases() {
		let planned = plan(&[
			&[(7, BufferAccess::Write)],
			&[(9, BufferAccess::Read)],
			&[(7, BufferAccess::Read), (7, BufferAccess::Write)],
		]);

		assert!(planned[0].is_empty());
		assert!(planned[1].is_empty());
		assert_eq!(planned[2].len(), 1);
		assert_eq!(planned[2][0].resource, 7);
		assert_eq!(
			planned[2][0].destination,
			AccessState {
				read: true,
				write: true,
			}
		);
	}

	#[test]
	fn bundles_conflicts_in_stable_resource_order() {
		let planned = plan(&[
			&[(8, BufferAccess::Write), (2, BufferAccess::Write)],
			&[(8, BufferAccess::Read), (2, BufferAccess::Read)],
		]);

		assert_eq!(
			planned[1]
				.iter()
				.map(|hazard| hazard.resource)
				.collect::<Vec<_>>(),
			[2, 8]
		);
	}

	#[test]
	fn maps_declared_access_to_exact_vulkan_storage_flags() {
		assert_eq!(
			state(BufferAccess::Read).vk_access(),
			ash::vk::AccessFlags2::SHADER_STORAGE_READ
		);
		assert_eq!(
			state(BufferAccess::Write).vk_access(),
			ash::vk::AccessFlags2::SHADER_STORAGE_WRITE
		);
	}
}
