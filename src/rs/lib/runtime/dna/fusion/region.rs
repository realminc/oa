//! Contiguous-region resource proofs for incumbent pair lowerings.
//!
//! Donor: `runtime/fusion.rs::{storage_unobservable,write_is_exclusive,
//! elided_writes_unobservable}`. Adaptation: normalize the same physical
//! accesses once, then separate ownership/SSA proofs from kernel geometry.
//! This is not the tile IR or the captured DNN partition validator.

use std::{
	collections::{BTreeMap, BTreeSet},
	ops::Range,
};

use super::{BufferAccess, ComputeNode, Device, Storage, storage_unobservable};

/// Descriptor slots identify retained physical buffers within one Device,
/// as in ExecutableGraph's hazard planner. Never serialize these identifiers.
#[derive(Clone)]
struct NodeAccess {
	/// Deduplicated `(descriptor, access)` pairs in descriptor order. One
	/// resource may still appear as separate read and write bindings —
	/// the aliased-slot republish pattern — but exact duplicates carry no
	/// information and would double-count writes.
	bindings: Vec<(u32, BufferAccess)>,
	opaque: bool,
}

#[derive(Default)]
struct ResourceAccess {
	writers: Vec<usize>,
	readers: Vec<usize>,
}

pub(super) struct RegionGraph {
	nodes: Vec<NodeAccess>,
	resources: BTreeMap<u32, ResourceAccess>,
	external: BTreeSet<u32>,
}

/// Failed resource proof, distinct from a lowering's shape/ABI decline.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(usize)]
pub(super) enum RegionDecline {
	InvalidRange,
	ConsumedSlot,
	EffectBoundary,
	NonSsaWrite,
	MissingMaterialization,
	UnavailableRead,
	UnexpectedWrite,
	AliasedWrite,
	Provenance,
}

impl RegionDecline {
	pub(super) const ALL: [Self; 9] = [
		Self::InvalidRange,
		Self::ConsumedSlot,
		Self::EffectBoundary,
		Self::NonSsaWrite,
		Self::MissingMaterialization,
		Self::UnavailableRead,
		Self::UnexpectedWrite,
		Self::AliasedWrite,
		Self::Provenance,
	];

	pub(super) const fn name(self) -> &'static str {
		match self {
			Self::InvalidRange => "invalid_range",
			Self::ConsumedSlot => "consumed_slot",
			Self::EffectBoundary => "effect_boundary",
			Self::NonSsaWrite => "non_ssa_write",
			Self::MissingMaterialization => "missing_materialization",
			Self::UnavailableRead => "unavailable_read",
			Self::UnexpectedWrite => "unexpected_write",
			Self::AliasedWrite => "aliased_write",
			Self::Provenance => "provenance",
		}
	}
}

pub(super) struct RegionCandidate {
	/// Resources the region only reads: their content comes entirely from
	/// before the region (`reads - writes`).
	inputs: BTreeSet<u32>,
	/// Prior physical versions available at entry even when this region also
	/// writes the resource (e.g. an in-place bias followed by an activation).
	incoming_versions: BTreeSet<u32>,
	/// Resources the region's final version of: every in-region write.
	/// A versioned input (written before and inside the region) is both an
	/// input and a write.
	writes: BTreeSet<u32>,
	materialize: BTreeSet<u32>,
}

/// Lowering-declared write facts the descriptor model cannot derive. Each
/// entry is an audited lowering contract — the resource model still proves
/// writer/reader positions, ownership and materialization around it.
#[derive(Default)]
pub(super) struct WriteAdmission {
	/// Resources several region members write at disjoint extents — the
	/// concat-partition pattern. Every writer of such a resource must be
	/// inside the position set, and every reader must sit after all writers
	/// or outside the position range entirely.
	pub partitioned: BTreeSet<u32>,
	/// Emission writes on region inputs that republish the loaded bytes at
	/// identical addresses — observable-neutral by lowering audit (the
	/// single-head SDPA aliased slots). A declared resource only licenses a
	/// write on a strict input (`reads - writes`).
	pub identity: BTreeSet<u32>,
}

impl WriteAdmission {
	/// Disjoint-extent write admission for the concat pattern.
	pub(super) fn partitioned(resources: impl IntoIterator<Item = u32>) -> Self {
		Self {
			partitioned: resources.into_iter().collect(),
			identity: BTreeSet::new(),
		}
	}

	/// Identity-republish admission for writes on region inputs.
	pub(super) fn identity(resources: impl IntoIterator<Item = u32>) -> Self {
		Self {
			partitioned: BTreeSet::new(),
			identity: resources.into_iter().collect(),
		}
	}
}

impl NodeAccess {
	fn from_compute(node: &ComputeNode) -> Self {
		let mut bindings: Vec<(u32, BufferAccess)> = node
			.buffers
			.iter()
			.map(|usage| (usage.buffer.descriptor_index(), usage.access))
			.collect();
		if let Some(indirect) = &node.indirect {
			bindings.push((indirect.buffer.descriptor_index(), BufferAccess::Read));
		}
		bindings.sort_by_key(|&(descriptor, access)| (descriptor, access as u8));
		bindings.dedup();
		Self {
			bindings,
			// An explicit schema audit is necessary, never sufficient: pair
			// matchers and the resource/alias proofs still restrict admission.
			opaque: !node.kernel.fusion_effects().is_empty()
				|| node.indirect.is_some()
				|| node.jit.is_some(),
		}
	}
}

impl RegionGraph {
	pub(super) fn from_compute(device: &Device, nodes: &[ComputeNode], refs: &[&Storage]) -> Self {
		let mut external = BTreeSet::new();
		let mut seen = BTreeSet::new();
		let accesses = nodes
			.iter()
			.map(|node| {
				let mut access = NodeAccess::from_compute(node);
				for usage in &node.buffers {
					access.opaque |= !usage.buffer.belongs_to(device);
					if usage.access != BufferAccess::Read
						&& seen.insert(usage.buffer.descriptor_index())
						&& !storage_unobservable(refs, &usage.buffer)
					{
						external.insert(usage.buffer.descriptor_index());
					}
				}
				access
			})
			.collect();
		Self::new(accesses, external)
	}

	fn new(nodes: Vec<NodeAccess>, external: BTreeSet<u32>) -> Self {
		let mut resources = BTreeMap::<u32, ResourceAccess>::new();
		for (index, node) in nodes.iter().enumerate() {
			for &(resource, access) in &node.bindings {
				let entry = resources.entry(resource).or_default();
				if access != BufferAccess::Write && entry.readers.last() != Some(&index) {
					entry.readers.push(index);
				}
				if access != BufferAccess::Read && entry.writers.last() != Some(&index) {
					entry.writers.push(index);
				}
			}
		}
		Self {
			nodes,
			resources,
			external,
		}
	}

	pub(super) fn contiguous(
		&self,
		range: Range<usize>,
		consumed: &BTreeSet<usize>,
		admission: &WriteAdmission,
	) -> Result<RegionCandidate, RegionDecline> {
		self.window(&range.collect(), consumed, admission)
	}

	/// Region over an arbitrary set of node positions. Lookahead merges absorb
	/// non-adjacent nodes; the materialization check must then treat any reader
	/// outside the position set — including gap nodes between absorbed slots —
	/// as an external consumer.
	///
	/// Write versioning replaces strict SSA: an in-region write on a resource
	/// other nodes wrote *before* the position range is a new version of a
	/// region input (the read-modify-write and write-into-input patterns),
	/// and a writer *after* the range simply overwrites — both keep the plan's
	/// value ordering intact. Only a writer between the first and last
	/// absorbed positions but outside the position set breaks that ordering.
	/// Resources declared `partitioned` additionally admit several in-region
	/// writers covering disjoint extents, with every reader after every
	/// writer or outside the range.
	pub(super) fn window(
		&self,
		positions: &BTreeSet<usize>,
		consumed: &BTreeSet<usize>,
		admission: &WriteAdmission,
	) -> Result<RegionCandidate, RegionDecline> {
		if positions.len() < 2 || positions.iter().any(|&index| index >= self.nodes.len()) {
			return Err(RegionDecline::InvalidRange);
		}
		if positions.iter().any(|index| consumed.contains(index)) {
			return Err(RegionDecline::ConsumedSlot);
		}
		let first = *positions.iter().next().unwrap_or(&0);
		let last = *positions.iter().next_back().unwrap_or(&0);
		let mut writes = BTreeSet::new();
		let mut reads = BTreeSet::new();
		let mut partitioned = BTreeSet::new();
		let mut multi_writes = BTreeSet::new();
		for &index in positions {
			let node = &self.nodes[index];
			if node.opaque {
				return Err(RegionDecline::EffectBoundary);
			}
			for &(resource, access) in &node.bindings {
				if access != BufferAccess::Write {
					reads.insert(resource);
				}
				if access == BufferAccess::Read {
					continue;
				}
				let lifetime = &self.resources[&resource];
				if admission.partitioned.contains(&resource) {
					// Disjoint-extent writes: every writer must be inside the
					// region so no hidden contributor is dropped.
					if !lifetime.writers.iter().all(|w| positions.contains(w)) {
						return Err(RegionDecline::NonSsaWrite);
					}
					partitioned.insert(resource);
					continue;
				}
				if !writes.insert(resource) {
					// A second in-region writer is a sequential new version —
					// positions are totally ordered, so the last writer's value
					// is the one the replacement must republish. Unlike
					// declared partitioned writes this needs no extent proof,
					// but every writer and every gap reader is checked below.
					multi_writes.insert(resource);
					continue;
				}
				if lifetime
					.writers
					.iter()
					.any(|w| !positions.contains(w) && (first..=last).contains(w))
					|| lifetime
						.readers
						.first()
						.is_some_and(|&reader| reader <= lifetime.writers[0])
				{
					return Err(RegionDecline::NonSsaWrite);
				}
			}
		}
		for &resource in &multi_writes {
			let lifetime = &self.resources[&resource];
			// A writer inside the position range but outside the position set
			// would silently overwrite or be overwritten by the republished
			// version; writers strictly before or after the range are the
			// already-admitted versioned ordering.
			if lifetime
				.writers
				.iter()
				.any(|w| !positions.contains(w) && (first..=last).contains(w))
			{
				return Err(RegionDecline::NonSsaWrite);
			}
			let last_w = lifetime.writers.iter().max().copied().unwrap_or(0);
			if lifetime.readers.iter().any(|&r| {
				if positions.contains(&r) {
					// An in-region reader before the final version's writer —
					// unless it is itself a writer (sequential
					// read-modify-write) — may observe either an intermediate
					// version or partition-garbage; the descriptor model
					// cannot distinguish, so it declines conservatively.
					r < last_w && !lifetime.writers.contains(&r)
				} else {
					// A gap/external reader between the range start and the
					// final version's writer observes an intermediate
					// version the fused node no longer produces.
					(first..=last_w).contains(&r)
				}
			}) {
				return Err(RegionDecline::NonSsaWrite);
			}
		}
		for &resource in &partitioned {
			let lifetime = &self.resources[&resource];
			let (min_w, max_w) = (
				lifetime.writers.iter().min().copied().unwrap_or(0),
				lifetime.writers.iter().max().copied().unwrap_or(0),
			);
			// A reader inside the region must see the completed partition;
			// a gap reader between the first and last writer would observe
			// a half-written buffer.
			if lifetime.readers.iter().any(|&r| {
				if positions.contains(&r) {
					r <= max_w
				} else {
					(min_w..=max_w).contains(&r)
				}
			}) {
				return Err(RegionDecline::NonSsaWrite);
			}
			writes.insert(resource);
		}
		let materialize = writes
			.iter()
			.copied()
			.filter(|resource| {
				self.external.contains(resource)
					|| self.resources[resource]
						.readers
						.iter()
						.any(|reader| !positions.contains(reader))
			})
			.collect();
		Ok(RegionCandidate {
			inputs: reads.difference(&writes).copied().collect(),
			incoming_versions: writes
				.iter()
				.copied()
				.filter(|resource| {
					self.resources[resource]
						.writers
						.iter()
						.any(|&writer| writer < first)
				})
				.collect(),
			writes,
			materialize,
		})
	}
}

impl RegionCandidate {
	pub(super) fn accepts_compute<'a>(
		&self,
		sources: impl IntoIterator<Item = &'a ComputeNode>,
		fused: &ComputeNode,
		admission: &WriteAdmission,
	) -> Result<(), RegionDecline> {
		if super::merged_semantic_ops(
			&sources
				.into_iter()
				.map(|node| node.semantic_ops.as_slice())
				.collect::<Vec<_>>(),
		) != fused.semantic_ops
		{
			return Err(RegionDecline::Provenance);
		}
		self.accepts(&NodeAccess::from_compute(fused), admission)
	}

	/// One write binding is legal when it (a) republishes a region write —
	/// including a versioned input the region already rewrote — or (b) is a
	/// lowering-declared identity republish on a strict region input. The
	/// declaration licenses only that input alias; a write to anything else
	/// stays an unexpected write. `published` records which replacement node
	/// already wrote the resource: a single node's multiple write bindings
	/// to one descriptor are its declared write contract (disjoint extents,
	/// `physical_write` exclusivity), while a write collision across
	/// *different* sibling replacements stays an `AliasedWrite`.
	fn admits_write(
		&self,
		published: &mut BTreeMap<u32, usize>,
		replacement: usize,
		admission: &WriteAdmission,
		resource: u32,
	) -> Result<(), RegionDecline> {
		if self.inputs.contains(&resource) {
			if !admission.identity.contains(&resource) {
				return Err(RegionDecline::AliasedWrite);
			}
		} else if !self.writes.contains(&resource) {
			return Err(RegionDecline::UnexpectedWrite);
		}
		match published.entry(resource) {
			std::collections::btree_map::Entry::Vacant(slot) => {
				slot.insert(replacement);
			}
			std::collections::btree_map::Entry::Occupied(slot) if *slot.get() != replacement => {
				return Err(RegionDecline::AliasedWrite);
			}
			std::collections::btree_map::Entry::Occupied(_) => {}
		}
		Ok(())
	}

	fn accepts(
		&self,
		replacement: &NodeAccess,
		admission: &WriteAdmission,
	) -> Result<(), RegionDecline> {
		if replacement.opaque {
			return Err(RegionDecline::EffectBoundary);
		}
		let mut published = BTreeMap::new();
		for &(resource, access) in &replacement.bindings {
			if access != BufferAccess::Write {
				// A read is satisfied by a strict input or by the region's
				// own produced version — an emission may legitimately read
				// back a versioned input (conv_1d's biased `projected`).
				if !self.inputs.contains(&resource)
					&& !self.incoming_versions.contains(&resource)
					&& !replacement
						.bindings
						.iter()
						.any(|&(written, access)| written == resource && access != BufferAccess::Read)
				{
					return Err(RegionDecline::UnavailableRead);
				}
			}
			if access != BufferAccess::Read {
				self.admits_write(&mut published, 0, admission, resource)?;
			}
		}
		if !self
			.materialize
			.iter()
			.all(|resource| published.contains_key(resource))
		{
			return Err(RegionDecline::MissingMaterialization);
		}
		Ok(())
	}

	/// Admit a multi-node emission (e.g. fused + separate reduction) as one
	/// region replacement. Reads may additionally be satisfied by a sibling
	/// replacement's write — that buffer is region-internal and must not be
	/// elided if anything outside still needs it, which `materialize` covers.
	pub(super) fn accepts_many<'a>(
		&self,
		sources: impl IntoIterator<Item = &'a ComputeNode>,
		replacements: &[&ComputeNode],
		admission: &WriteAdmission,
	) -> Result<(), RegionDecline> {
		if replacements.is_empty() {
			return Err(RegionDecline::InvalidRange);
		}
		let merged_sources = super::merged_semantic_ops(
			&sources
				.into_iter()
				.map(|node| node.semantic_ops.as_slice())
				.collect::<Vec<_>>(),
		);
		let merged_replacements = super::merged_semantic_ops(
			&replacements
				.iter()
				.map(|node| node.semantic_ops.as_slice())
				.collect::<Vec<_>>(),
		);
		if merged_sources != merged_replacements {
			return Err(RegionDecline::Provenance);
		}
		let accesses: Vec<NodeAccess> = replacements
			.iter()
			.map(|node| NodeAccess::from_compute(node))
			.collect();
		self.accepts_ordered(&accesses, admission)
	}

	fn accepts_ordered(
		&self,
		accesses: &[NodeAccess],
		admission: &WriteAdmission,
	) -> Result<(), RegionDecline> {
		for access in accesses {
			if access.opaque {
				return Err(RegionDecline::EffectBoundary);
			}
		}
		let mut published = BTreeMap::new();
		for (replacement, access) in accesses.iter().enumerate() {
			for &(resource, access) in &access.bindings {
				if access != BufferAccess::Read {
					self.admits_write(&mut published, replacement, admission, resource)?;
				}
			}
		}
		for (replacement, access) in accesses.iter().enumerate() {
			for &(resource, access) in &access.bindings {
				if access != BufferAccess::Write
					&& !self.inputs.contains(&resource)
					&& !self.incoming_versions.contains(&resource)
					&& !published
						.get(&resource)
						.is_some_and(|&writer| writer <= replacement)
				{
					return Err(RegionDecline::UnavailableRead);
				}
			}
		}
		if !self
			.materialize
			.iter()
			.all(|resource| published.contains_key(resource))
		{
			return Err(RegionDecline::MissingMaterialization);
		}
		Ok(())
	}
}

#[cfg(test)]
#[path = "../../../../../../test/rs/runtime/fusion_region_unit.rs"]
mod tests;
