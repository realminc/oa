use super::*;

fn node_outcome_sum(stats: FusionStats) -> u64 {
	[
		stats.disabled_node_count(),
		stats.not_elementwise_node_count(),
		stats.policy_excluded_node_count(),
		stats.hazard_node_count(),
		stats.isolated_node_count(),
		stats.extent_boundary_node_count(),
		stats.cross_invocation_node_count(),
		stats.program_rejected_node_count(),
		stats.foreign_device_node_count(),
		stats.tape_bound_node_count(),
		stats.tape_capacity_node_count(),
		stats.resource_node_count(),
		stats.specialization_declined_node_count(),
		stats.tape_fused_node_count(),
		stats.specialized_fused_node_count(),
	]
	.iter()
	.sum()
}

#[test]
fn census_starts_empty_and_matches_default() {
	assert_eq!(FusionCensus::new().snapshot(), FusionStats::default());
}

#[test]
fn census_routes_every_outcome_to_one_counter() {
	let census = FusionCensus::new();
	let outcomes = [
		NodeOutcome::Disabled,
		NodeOutcome::NotElementwise,
		NodeOutcome::PolicyExcluded,
		NodeOutcome::Hazard,
		NodeOutcome::Isolated,
		NodeOutcome::ExtentBoundary,
		NodeOutcome::CrossInvocation,
		NodeOutcome::ProgramRejected,
		NodeOutcome::ForeignDevice,
		NodeOutcome::TapeBound,
		NodeOutcome::TapeCapacity,
		NodeOutcome::Resource,
		NodeOutcome::SpecializationDeclined,
	];
	census.pass(outcomes.len() + 5);
	for outcome in outcomes {
		census.nodes(outcome, 1);
	}
	census.fused(false, 3);
	census.fused(true, 2);
	let stats = census.snapshot();
	assert_eq!(stats.pass_count(), 1);
	assert_eq!(stats.examined_node_count(), node_outcome_sum(stats));
	assert_eq!(stats.tape_region_count(), 1);
	assert_eq!(stats.tape_fused_node_count(), 3);
	assert_eq!(stats.specialized_region_count(), 1);
	assert_eq!(stats.specialized_fused_node_count(), 2);
	for count in [
		stats.disabled_node_count(),
		stats.not_elementwise_node_count(),
		stats.policy_excluded_node_count(),
		stats.hazard_node_count(),
		stats.isolated_node_count(),
		stats.extent_boundary_node_count(),
		stats.cross_invocation_node_count(),
		stats.program_rejected_node_count(),
		stats.foreign_device_node_count(),
		stats.tape_bound_node_count(),
		stats.tape_capacity_node_count(),
		stats.resource_node_count(),
		stats.specialization_declined_node_count(),
	] {
		assert_eq!(count, 1);
	}
}

#[test]
fn census_records_structural_jit_and_dnn_outcomes() {
	let census = FusionCensus::new();
	census.structural(2, 5);
	for outcome in [
		JitOutcome::Disabled,
		JitOutcome::Admitted,
		JitOutcome::Reused,
		JitOutcome::Reused,
		JitOutcome::PolicyTape,
		JitOutcome::CachedNegative,
		JitOutcome::Bound,
		JitOutcome::CompilerUnavailable,
		JitOutcome::CompileRejected,
		JitOutcome::PipelineRejected,
		JitOutcome::Queued,
		JitOutcome::Pending,
		JitOutcome::Pending,
	] {
		census.jit(outcome);
	}
	census.jit_compile(std::time::Duration::from_nanos(700));
	census.jit_compile(std::time::Duration::from_nanos(300));
	census.jit_pipeline(std::time::Duration::from_nanos(50));
	census.dnn(3, 4);
	census.dnn(1, 0);
	let stats = census.snapshot();
	assert_eq!(stats.structural_region_count(), 2);
	assert_eq!(stats.structural_node_count(), 5);
	assert_eq!(stats.jit_disabled_count(), 1);
	assert_eq!(stats.jit_admitted_count(), 1);
	assert_eq!(stats.jit_reused_count(), 2);
	assert_eq!(stats.jit_policy_tape_count(), 1);
	assert_eq!(stats.jit_cached_negative_count(), 1);
	assert_eq!(stats.jit_bound_count(), 1);
	assert_eq!(stats.jit_compiler_unavailable_count(), 1);
	assert_eq!(stats.jit_compile_rejected_count(), 1);
	assert_eq!(stats.jit_pipeline_rejected_count(), 1);
	assert_eq!(stats.jit_queued_count(), 1);
	assert_eq!(stats.jit_pending_count(), 2);
	assert_eq!(stats.jit_compile_count(), 2);
	assert_eq!(stats.jit_compile_ns(), 1000);
	assert_eq!(stats.jit_pipeline_ns(), 50);
	assert_eq!(stats.dnn_applied_partition_count(), 4);
	assert_eq!(stats.dnn_fallback_partition_count(), 4);
	assert_eq!(stats.examined_node_count(), 0);
}

#[test]
fn display_is_a_stable_ordered_key_value_record() {
	let census = FusionCensus::new();
	census.pass(4);
	census.nodes(NodeOutcome::NotElementwise, 2);
	census.fused(false, 2);
	census.dnn(0, 1);
	let text = census.snapshot().to_string();
	assert!(text.starts_with("examined=4 disabled=0 not_elementwise=2 "));
	assert!(text.contains(" tape_fused=2 specialized_fused=0 passes=1 "));
	assert!(text.ends_with(" dnn_applied=0 dnn_fallback=1"));
	let keys = text
		.split(' ')
		.map(|field| field.split_once('=').expect("key=value").0)
		.collect::<Vec<_>>();
	assert_eq!(keys.len(), COUNTER_COUNT);
	let mut unique = keys.clone();
	unique.sort_unstable();
	unique.dedup();
	assert_eq!(unique.len(), keys.len());
}
