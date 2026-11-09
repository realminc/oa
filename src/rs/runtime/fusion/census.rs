//! Device-owned counters of fusion decisions and decline reasons.
//!
//! Every fusion layer records one outcome at its existing decision point;
//! the census never changes a routing decision. Elementwise-pass node
//! outcomes are exclusive: each node offered to the elementwise pass lands in
//! exactly one node bucket, so `examined_node_count` equals their sum. JIT
//! counters count admission requests, which can exceed fused regions because
//! a rejected longer segment retries as a shorter one. Counters use relaxed
//! atomics; a snapshot is a diagnostic, not a synchronization point.

use std::{
	fmt,
	sync::atomic::{AtomicU64, Ordering},
};

use crate::runtime::FusionStats;

/// Exclusive outcome of one node offered to the elementwise pass.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(in crate::runtime) enum NodeOutcome {
	/// Elementwise fusion is disabled or its tape kernel is unavailable.
	Disabled,
	/// The node is not in the fusible elementwise vocabulary.
	NotElementwise,
	/// The node is elementwise but excluded by `OA_DISABLE_RL_FUSION`.
	PolicyExcluded,
	/// The write is read before its producer or written by another node.
	Hazard,
	/// No admissible elementwise neighbour remains.
	Isolated,
	/// Adjacent elementwise neighbours have a different element count.
	ExtentBoundary,
	/// A gather or broadcast-mapped read consumes an in-region producer.
	CrossInvocation,
	/// The candidate failed program construction or validation.
	ProgramRejected,
	/// An operand or output belongs to another device.
	ForeignDevice,
	/// Above the tape element bound with specialization disabled.
	TapeBound,
	/// The tape encoder rejected register, operand, store or step capacity.
	TapeCapacity,
	/// Tape storage could not be allocated.
	Resource,
	/// Above the tape element bound and specialization declined the region.
	SpecializationDeclined,
	/// Fused into a tape-interpreter dispatch.
	TapeFused,
	/// Fused into a specialized JIT dispatch.
	SpecializedFused,
}

/// Outcome of one specialization admission request.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(in crate::runtime) enum JitOutcome {
	/// `OA_JIT_FUSION` is not set.
	Disabled,
	/// A new specialization pipeline was admitted.
	Admitted,
	/// An already admitted pipeline was reused.
	Reused,
	/// Route policy kept the tape incumbent or ran a tape probe.
	PolicyTape,
	/// This program failed admission earlier in the device lifetime.
	CachedNegative,
	/// The bounded admission table is full.
	Bound,
	/// The runtime Slang compiler could not be opened.
	CompilerUnavailable,
	/// Compilation, SPIR-V validation or ABI reflection rejected the program.
	CompileRejected,
	/// Descriptor layout, pipeline cache or pipeline creation failed.
	PipelineRejected,
	/// A new specialization was accepted onto the background compile queue.
	Queued,
	/// The program's background compile is still in flight.
	Pending,
}

#[derive(Clone, Copy)]
#[repr(usize)]
enum Counter {
	Examined,
	Disabled,
	NotElementwise,
	PolicyExcluded,
	Hazard,
	Isolated,
	ExtentBoundary,
	CrossInvocation,
	ProgramRejected,
	ForeignDevice,
	TapeBound,
	TapeCapacity,
	Resource,
	SpecializationDeclined,
	TapeFused,
	SpecializedFused,
	Pass,
	StructuralRegion,
	StructuralNode,
	TapeRegion,
	SpecializedRegion,
	JitDisabled,
	JitAdmitted,
	JitReused,
	JitPolicyTape,
	JitCachedNegative,
	JitBound,
	JitCompilerUnavailable,
	JitCompileRejected,
	JitPipelineRejected,
	JitQueued,
	JitPending,
	JitCompile,
	JitCompileNs,
	JitPipelineNs,
	DnnApplied,
	DnnFallback,
}

const COUNTER_COUNT: usize = Counter::DnnFallback as usize + 1;

impl NodeOutcome {
	const fn counter(self) -> Counter {
		match self {
			Self::Disabled => Counter::Disabled,
			Self::NotElementwise => Counter::NotElementwise,
			Self::PolicyExcluded => Counter::PolicyExcluded,
			Self::Hazard => Counter::Hazard,
			Self::Isolated => Counter::Isolated,
			Self::ExtentBoundary => Counter::ExtentBoundary,
			Self::CrossInvocation => Counter::CrossInvocation,
			Self::ProgramRejected => Counter::ProgramRejected,
			Self::ForeignDevice => Counter::ForeignDevice,
			Self::TapeBound => Counter::TapeBound,
			Self::TapeCapacity => Counter::TapeCapacity,
			Self::Resource => Counter::Resource,
			Self::SpecializationDeclined => Counter::SpecializationDeclined,
			Self::TapeFused => Counter::TapeFused,
			Self::SpecializedFused => Counter::SpecializedFused,
		}
	}
}

impl JitOutcome {
	const fn counter(self) -> Counter {
		match self {
			Self::Disabled => Counter::JitDisabled,
			Self::Admitted => Counter::JitAdmitted,
			Self::Reused => Counter::JitReused,
			Self::PolicyTape => Counter::JitPolicyTape,
			Self::CachedNegative => Counter::JitCachedNegative,
			Self::Bound => Counter::JitBound,
			Self::CompilerUnavailable => Counter::JitCompilerUnavailable,
			Self::CompileRejected => Counter::JitCompileRejected,
			Self::PipelineRejected => Counter::JitPipelineRejected,
			Self::Queued => Counter::JitQueued,
			Self::Pending => Counter::JitPending,
		}
	}
}

pub(in crate::runtime) struct FusionCensus {
	counters: [AtomicU64; COUNTER_COUNT],
}

impl FusionCensus {
	pub(in crate::runtime) fn new() -> Self {
		Self {
			counters: std::array::from_fn(|_| AtomicU64::new(0)),
		}
	}

	fn add(&self, counter: Counter, amount: u64) {
		// Wrapping needs 2^64 recorded decisions and would only corrupt a
		// diagnostic, never execution.
		self.counters[counter as usize].fetch_add(amount, Ordering::Relaxed);
	}

	fn get(&self, counter: Counter) -> u64 {
		self.counters[counter as usize].load(Ordering::Relaxed)
	}

	/// Record one elementwise pass offered `nodes` nodes.
	pub(in crate::runtime) fn pass(&self, nodes: usize) {
		self.add(Counter::Pass, 1);
		self.add(Counter::Examined, nodes as u64);
	}

	pub(in crate::runtime) fn nodes(&self, outcome: NodeOutcome, count: usize) {
		self.add(outcome.counter(), count as u64);
	}

	/// Record one fused elementwise region and its source nodes.
	pub(in crate::runtime) fn fused(&self, specialized: bool, nodes: usize) {
		let (region, outcome) = if specialized {
			(Counter::SpecializedRegion, NodeOutcome::SpecializedFused)
		} else {
			(Counter::TapeRegion, NodeOutcome::TapeFused)
		};
		self.add(region, 1);
		self.nodes(outcome, nodes);
	}

	/// Record structural replacements and the source nodes they absorbed.
	pub(in crate::runtime) fn structural(&self, regions: usize, nodes: usize) {
		self.add(Counter::StructuralRegion, regions as u64);
		self.add(Counter::StructuralNode, nodes as u64);
	}

	pub(in crate::runtime) fn jit(&self, outcome: JitOutcome) {
		self.add(outcome.counter(), 1);
	}

	pub(in crate::runtime) fn jit_compile(&self, elapsed: std::time::Duration) {
		self.add(Counter::JitCompile, 1);
		self.add(Counter::JitCompileNs, duration_ns(elapsed));
	}

	pub(in crate::runtime) fn jit_pipeline(&self, elapsed: std::time::Duration) {
		self.add(Counter::JitPipelineNs, duration_ns(elapsed));
	}

	/// Record one captured plan's DNN partition outcomes.
	pub(in crate::runtime) fn dnn(&self, applied: u32, fallback: u32) {
		self.add(Counter::DnnApplied, u64::from(applied));
		self.add(Counter::DnnFallback, u64::from(fallback));
	}

	pub(in crate::runtime) fn snapshot(&self) -> FusionStats {
		FusionStats {
			examined_node_count: self.get(Counter::Examined),
			disabled_node_count: self.get(Counter::Disabled),
			not_elementwise_node_count: self.get(Counter::NotElementwise),
			policy_excluded_node_count: self.get(Counter::PolicyExcluded),
			hazard_node_count: self.get(Counter::Hazard),
			isolated_node_count: self.get(Counter::Isolated),
			extent_boundary_node_count: self.get(Counter::ExtentBoundary),
			cross_invocation_node_count: self.get(Counter::CrossInvocation),
			program_rejected_node_count: self.get(Counter::ProgramRejected),
			foreign_device_node_count: self.get(Counter::ForeignDevice),
			tape_bound_node_count: self.get(Counter::TapeBound),
			tape_capacity_node_count: self.get(Counter::TapeCapacity),
			resource_node_count: self.get(Counter::Resource),
			specialization_declined_node_count: self.get(Counter::SpecializationDeclined),
			tape_fused_node_count: self.get(Counter::TapeFused),
			specialized_fused_node_count: self.get(Counter::SpecializedFused),
			pass_count: self.get(Counter::Pass),
			structural_region_count: self.get(Counter::StructuralRegion),
			structural_node_count: self.get(Counter::StructuralNode),
			tape_region_count: self.get(Counter::TapeRegion),
			specialized_region_count: self.get(Counter::SpecializedRegion),
			jit_disabled_count: self.get(Counter::JitDisabled),
			jit_admitted_count: self.get(Counter::JitAdmitted),
			jit_reused_count: self.get(Counter::JitReused),
			jit_policy_tape_count: self.get(Counter::JitPolicyTape),
			jit_cached_negative_count: self.get(Counter::JitCachedNegative),
			jit_bound_count: self.get(Counter::JitBound),
			jit_compiler_unavailable_count: self.get(Counter::JitCompilerUnavailable),
			jit_compile_rejected_count: self.get(Counter::JitCompileRejected),
			jit_pipeline_rejected_count: self.get(Counter::JitPipelineRejected),
			jit_queued_count: self.get(Counter::JitQueued),
			jit_pending_count: self.get(Counter::JitPending),
			jit_compile_count: self.get(Counter::JitCompile),
			jit_compile_ns: self.get(Counter::JitCompileNs),
			jit_pipeline_ns: self.get(Counter::JitPipelineNs),
			dnn_applied_partition_count: self.get(Counter::DnnApplied),
			dnn_fallback_partition_count: self.get(Counter::DnnFallback),
		}
	}
}

fn duration_ns(elapsed: std::time::Duration) -> u64 {
	u64::try_from(elapsed.as_nanos()).unwrap_or(u64::MAX)
}

impl FusionStats {
	/// Return nodes offered to the elementwise fusion pass.
	///
	/// Every examined node is counted in exactly one node outcome below, so
	/// this equals the sum of the `*_node_count` outcomes other than
	/// [`FusionStats::structural_node_count`].
	pub const fn examined_node_count(self) -> u64 {
		self.examined_node_count
	}

	/// Return nodes seen while elementwise fusion was disabled or unavailable.
	pub const fn disabled_node_count(self) -> u64 {
		self.disabled_node_count
	}

	/// Return nodes outside the fusible elementwise vocabulary.
	///
	/// Includes nodes already replaced by a dedicated structural kernel.
	pub const fn not_elementwise_node_count(self) -> u64 {
		self.not_elementwise_node_count
	}

	/// Return elementwise nodes excluded by the RL fusion opt-out.
	pub const fn policy_excluded_node_count(self) -> u64 {
		self.policy_excluded_node_count
	}

	/// Return elementwise nodes whose output is aliased or multiply written.
	pub const fn hazard_node_count(self) -> u64 {
		self.hazard_node_count
	}

	/// Return elementwise nodes with no admissible elementwise neighbour.
	pub const fn isolated_node_count(self) -> u64 {
		self.isolated_node_count
	}

	/// Return elementwise nodes separated from neighbours by element count.
	pub const fn extent_boundary_node_count(self) -> u64 {
		self.extent_boundary_node_count
	}

	/// Return nodes left unfused because a gather or broadcast read needs a
	/// materialized producer.
	pub const fn cross_invocation_node_count(self) -> u64 {
		self.cross_invocation_node_count
	}

	/// Return nodes whose candidate program failed construction or validation.
	pub const fn program_rejected_node_count(self) -> u64 {
		self.program_rejected_node_count
	}

	/// Return nodes whose buffers belong to another device.
	pub const fn foreign_device_node_count(self) -> u64 {
		self.foreign_device_node_count
	}

	/// Return nodes above the tape element bound while specialization is off.
	pub const fn tape_bound_node_count(self) -> u64 {
		self.tape_bound_node_count
	}

	/// Return nodes rejected by tape register, operand, store or step capacity.
	pub const fn tape_capacity_node_count(self) -> u64 {
		self.tape_capacity_node_count
	}

	/// Return nodes left unfused because fusion storage was unavailable.
	pub const fn resource_node_count(self) -> u64 {
		self.resource_node_count
	}

	/// Return nodes above the tape element bound that specialization declined.
	pub const fn specialization_declined_node_count(self) -> u64 {
		self.specialization_declined_node_count
	}

	/// Return nodes fused into tape-interpreter dispatches.
	pub const fn tape_fused_node_count(self) -> u64 {
		self.tape_fused_node_count
	}

	/// Return nodes fused into specialized dispatches.
	pub const fn specialized_fused_node_count(self) -> u64 {
		self.specialized_fused_node_count
	}

	/// Return completed elementwise fusion passes.
	///
	/// Eager work is fused once per flush and captured work once per capture;
	/// plan replay does not run fusion.
	pub const fn pass_count(self) -> u64 {
		self.pass_count
	}

	/// Return dedicated structural replacements made before elementwise fusion.
	pub const fn structural_region_count(self) -> u64 {
		self.structural_region_count
	}

	/// Return source nodes absorbed by dedicated structural replacements.
	pub const fn structural_node_count(self) -> u64 {
		self.structural_node_count
	}

	/// Return fused tape-interpreter dispatches created.
	pub const fn tape_region_count(self) -> u64 {
		self.tape_region_count
	}

	/// Return fused specialized dispatches created.
	pub const fn specialized_region_count(self) -> u64 {
		self.specialized_region_count
	}

	/// Return specialization requests made while `OA_JIT_FUSION` was unset.
	pub const fn jit_disabled_count(self) -> u64 {
		self.jit_disabled_count
	}

	/// Return requests that admitted a new specialized pipeline.
	pub const fn jit_admitted_count(self) -> u64 {
		self.jit_admitted_count
	}

	/// Return requests that reused an admitted specialized pipeline.
	pub const fn jit_reused_count(self) -> u64 {
		self.jit_reused_count
	}

	/// Return requests where measured route policy selected the tape route.
	pub const fn jit_policy_tape_count(self) -> u64 {
		self.jit_policy_tape_count
	}

	/// Return requests for a program that already failed admission.
	pub const fn jit_cached_negative_count(self) -> u64 {
		self.jit_cached_negative_count
	}

	/// Return requests declined because the admission table is full.
	pub const fn jit_bound_count(self) -> u64 {
		self.jit_bound_count
	}

	/// Return requests declined because the runtime compiler is unavailable.
	pub const fn jit_compiler_unavailable_count(self) -> u64 {
		self.jit_compiler_unavailable_count
	}

	/// Return requests rejected by compilation, validation or ABI reflection.
	pub const fn jit_compile_rejected_count(self) -> u64 {
		self.jit_compile_rejected_count
	}

	/// Return requests rejected during pipeline preparation.
	pub const fn jit_pipeline_rejected_count(self) -> u64 {
		self.jit_pipeline_rejected_count
	}

	/// Return programs accepted onto the background compile queue.
	pub const fn jit_queued_count(self) -> u64 {
		self.jit_queued_count
	}

	/// Return requests that found a background compile still in flight.
	pub const fn jit_pending_count(self) -> u64 {
		self.jit_pending_count
	}

	/// Return runtime compiler invocations, excluding restored artifacts.
	pub const fn jit_compile_count(self) -> u64 {
		self.jit_compile_count
	}

	/// Return host wall time spent in runtime compilation, in nanoseconds.
	pub const fn jit_compile_ns(self) -> u64 {
		self.jit_compile_ns
	}

	/// Return host wall time spent creating specialized pipelines, in nanoseconds.
	pub const fn jit_pipeline_ns(self) -> u64 {
		self.jit_pipeline_ns
	}

	/// Return captured-plan DNN partitions replaced by a provider route.
	pub const fn dnn_applied_partition_count(self) -> u64 {
		self.dnn_applied_partition_count
	}

	/// Return captured-plan DNN partitions retained on their source route.
	pub const fn dnn_fallback_partition_count(self) -> u64 {
		self.dnn_fallback_partition_count
	}
}

/// Stable space-separated `key=value` record in declaration order.
impl fmt::Display for FusionStats {
	fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
		let fields = [
			("examined", self.examined_node_count),
			("disabled", self.disabled_node_count),
			("not_elementwise", self.not_elementwise_node_count),
			("policy_excluded", self.policy_excluded_node_count),
			("hazard", self.hazard_node_count),
			("isolated", self.isolated_node_count),
			("extent_boundary", self.extent_boundary_node_count),
			("cross_invocation", self.cross_invocation_node_count),
			("program_rejected", self.program_rejected_node_count),
			("foreign_device", self.foreign_device_node_count),
			("tape_bound", self.tape_bound_node_count),
			("tape_capacity", self.tape_capacity_node_count),
			("resource", self.resource_node_count),
			(
				"specialization_declined",
				self.specialization_declined_node_count,
			),
			("tape_fused", self.tape_fused_node_count),
			("specialized_fused", self.specialized_fused_node_count),
			("passes", self.pass_count),
			("structural_regions", self.structural_region_count),
			("structural_nodes", self.structural_node_count),
			("tape_regions", self.tape_region_count),
			("specialized_regions", self.specialized_region_count),
			("jit_disabled", self.jit_disabled_count),
			("jit_admitted", self.jit_admitted_count),
			("jit_reused", self.jit_reused_count),
			("jit_policy_tape", self.jit_policy_tape_count),
			("jit_cached_negative", self.jit_cached_negative_count),
			("jit_bound", self.jit_bound_count),
			(
				"jit_compiler_unavailable",
				self.jit_compiler_unavailable_count,
			),
			("jit_compile_rejected", self.jit_compile_rejected_count),
			("jit_pipeline_rejected", self.jit_pipeline_rejected_count),
			("jit_queued", self.jit_queued_count),
			("jit_pending", self.jit_pending_count),
			("jit_compiles", self.jit_compile_count),
			("jit_compile_ns", self.jit_compile_ns),
			("jit_pipeline_ns", self.jit_pipeline_ns),
			("dnn_applied", self.dnn_applied_partition_count),
			("dnn_fallback", self.dnn_fallback_partition_count),
		];
		for (index, (key, value)) in fields.iter().enumerate() {
			if index > 0 {
				formatter.write_str(" ")?;
			}
			write!(formatter, "{key}={value}")?;
		}
		Ok(())
	}
}

#[cfg(test)]
#[path = "../../../../test/rs/runtime/fusion_census_unit.rs"]
mod tests;
