use super::*;

fn measured(specialized: u64, tape: u64) -> RouteObservation {
	let mut observation = RouteObservation::new();
	for _ in 0..MIN_ROUTE_SAMPLES {
		observation.push(ObservedRoute::Specialized, specialized);
		observation.push(ObservedRoute::Tape, tape);
	}
	observation
}

#[test]
fn winner_refreshes_losing_route_without_persisting_probe_as_winner() {
	let mut observation = measured(50, 100);
	observation.records = REFRESH_PERIOD;
	assert!(!select_route(&mut observation, true));
	assert!(
		observation.specialized,
		"tape refresh must not replace the winner"
	);
	observation.records += 1;
	assert!(select_route(&mut observation, true));

	observation = measured(150, 100);
	observation.records = REPROBE_PERIOD;
	assert!(select_route(&mut observation, true));
	assert!(
		!observation.specialized,
		"specialized probe must not replace the winner"
	);
	observation.records += 1;
	assert!(!select_route(&mut observation, true));
	assert!(select_route(&mut observation, false));
}

#[test]
fn live_windows_require_four_samples_and_hysteresis_preserves_winner() {
	let mut observation = measured(100, 100);
	observation.records = 1;
	assert!(
		!observation.specialized,
		"an unproven specialization is not the incumbent"
	);
	assert!(
		!select_route(&mut observation, true),
		"equal timings cannot promote specialization"
	);
	observation.specialized = false;
	assert!(!select_route(&mut observation, true));
	observation.specialized = true;
	assert!(select_route(&mut observation, true));
	observation.specialized_ns.truncate(MIN_ROUTE_SAMPLES - 1);
	assert!(select_route(&mut observation, true));
	observation.specialized_ns.push_back(100);
	observation.tape_ns.truncate(MIN_ROUTE_SAMPLES - 1);
	assert!(!select_route(&mut observation, true));
}

#[test]
fn persisted_evidence_requires_each_route_and_large_timings_compare_without_overflow() {
	let mut observation = measured(u64::MAX / 2, u64::MAX);
	observation.records = 1;
	assert!(select_route(&mut observation, true));
	observation.tape_ns.truncate(1);
	let record = observation.record(1);
	assert_eq!(record.sample_count, 1);
	let mut restored = RouteObservation::restore(&record);
	assert!(
		select_route(&mut restored, true),
		"under-sampled record must restart probing"
	);
	observation.specialized_ns.clear();
	observation.tape_ns.clear();
	observation.restored_samples = u32::MAX;
	assert_eq!(observation.record(1).sample_count, u32::MAX);
}
