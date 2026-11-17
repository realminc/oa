//! Measured specialization routing: sample windows, hysteresis and probe cadence.
//! Donor: device/jit.rs; policy constants, observations and selection unchanged.
use crate::runtime::device::vkcache::jit::{self, RouteObservationRecord};
use std::collections::VecDeque;

/// Timing samples kept per route; the median of the live window wins over
/// restored medians so policy follows the current workload.
const MAX_ROUTE_SAMPLES: usize = 32;
/// Live samples required before a persisted median alone decides probes.
const MIN_ROUTE_SAMPLES: usize = 4;
/// Records between re-probes of a demoted specialized route.
const REPROBE_PERIOD: u64 = 16;
/// Records between tape refresh probes while specialized holds.
const REFRESH_PERIOD: u64 = 32;
/// The route a fused dispatch executed, for route-observation timing.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum ObservedRoute {
	/// The canonical fused tape interpreter ran.
	Tape,
	/// An admitted specialization ran.
	Specialized,
}

/// Diagnostic route override from `OA_JIT_ROUTE`, honored only while
/// `OA_JIT_FUSION` is set: the flag-off process can never observe it, so
/// flag-raising tests serialized on the jit lock stay the only readers.
/// `tape` runs every fused region on the tape interpreter, `specialized`
/// admits the compiled pipeline without the hysteresis policy and
/// `unfused` declines fused regions at discovery so canonical dispatches
/// run. Unset or unrecognized values keep the adaptive policy.
#[derive(Clone, Copy, PartialEq, Eq)]
pub(in crate::runtime) enum JitRouteOverride {
	Tape,
	Specialized,
	Unfused,
}

/// Read per call like the `OA_DISABLE_*` env gates so flag-sensitive tests
/// can change the route between serialized phases.
pub(in crate::runtime) fn jit_route_override() -> Option<JitRouteOverride> {
	if !crate::EnvFlag::is_set("OA_JIT_FUSION") {
		return None;
	}
	match crate::EnvFlag::get_string("OA_JIT_ROUTE", "").as_str() {
		"" | "auto" => None,
		"tape" => Some(JitRouteOverride::Tape),
		"specialized" => Some(JitRouteOverride::Specialized),
		"unfused" => Some(JitRouteOverride::Unfused),
		value => {
			static WARNED: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
			if !WARNED.swap(true, std::sync::atomic::Ordering::Relaxed) {
				crate::log_warn!(
					crate::LogComponent::RUNTIME,
					"OA_JIT_ROUTE={value}: expected auto, tape, specialized or unfused; keeping auto"
				);
			}
			None
		}
	}
}

/// Measurement and selection state for one canonical fusion signature.
pub(in crate::runtime) struct RouteObservation {
	/// Admission calls (≈ recordings) of this signature; also persisted as
	/// `last_step`.
	pub(in crate::runtime) records: u64,
	/// Route the policy currently selects: `true` = specialized.
	pub(in crate::runtime) specialized: bool,
	/// Recent specialized-route GPU durations, newest last.
	pub(in crate::runtime) specialized_ns: VecDeque<u64>,
	/// Recent tape-route GPU durations, newest last.
	pub(in crate::runtime) tape_ns: VecDeque<u64>,
	/// Persisted medians restored from `.vkcache`; used only while the live
	/// window is empty.
	pub(in crate::runtime) restored_samples: u32,
	pub(in crate::runtime) restored_specialized_ns: u64,
	pub(in crate::runtime) restored_tape_ns: u64,
}

impl RouteObservation {
	pub(in crate::runtime) fn new() -> Self {
		Self {
			records: 0,
			// Initial specialized dispatches collect evidence; they are probes.
			// Tape remains the incumbent until specialization earns promotion.
			specialized: false,
			specialized_ns: VecDeque::new(),
			tape_ns: VecDeque::new(),
			restored_samples: 0,
			restored_specialized_ns: 0,
			restored_tape_ns: 0,
		}
	}

	pub(in crate::runtime) fn restore(record: &RouteObservationRecord) -> Self {
		Self {
			records: record.last_step,
			specialized: record.selected_route == jit::ROUTE_SPECIALIZED,
			specialized_ns: VecDeque::new(),
			tape_ns: VecDeque::new(),
			restored_samples: record.sample_count,
			restored_specialized_ns: record.specialized_p50_ns,
			restored_tape_ns: record.tape_p50_ns,
		}
	}

	pub(in crate::runtime) fn push(&mut self, route: ObservedRoute, ns: u64) {
		let window = match route {
			ObservedRoute::Specialized => &mut self.specialized_ns,
			ObservedRoute::Tape => &mut self.tape_ns,
		};
		if window.len() >= MAX_ROUTE_SAMPLES {
			window.pop_front();
		}
		window.push_back(ns);
	}

	fn median(window: &VecDeque<u64>) -> Option<u64> {
		if window.is_empty() {
			return None;
		}
		let mut sorted: Vec<u64> = window.iter().copied().collect();
		sorted.sort_unstable();
		Some(sorted[sorted.len() / 2])
	}

	pub(in crate::runtime) fn specialized_p50(&self) -> Option<u64> {
		Self::median(&self.specialized_ns).or_else(|| {
			(self.restored_samples >= MIN_ROUTE_SAMPLES as u32 && self.restored_specialized_ns > 0)
				.then_some(self.restored_specialized_ns)
		})
	}

	pub(in crate::runtime) fn tape_p50(&self) -> Option<u64> {
		Self::median(&self.tape_ns).or_else(|| {
			(self.restored_samples >= MIN_ROUTE_SAMPLES as u32 && self.restored_tape_ns > 0)
				.then_some(self.restored_tape_ns)
		})
	}

	pub(in crate::runtime) fn record(&self, signature_key: u64) -> RouteObservationRecord {
		RouteObservationRecord {
			signature_key,
			selected_route: if self.specialized {
				jit::ROUTE_SPECIALIZED
			} else {
				jit::ROUTE_TAPE
			},
			sample_count: {
				let count = |window: &VecDeque<u64>| {
					if window.is_empty() {
						self.restored_samples
					} else {
						window.len() as u32
					}
				};
				count(&self.specialized_ns).min(count(&self.tape_ns))
			},
			unfused_p50_ns: 0,
			tape_p50_ns: self.tape_p50().unwrap_or(0),
			specialized_p50_ns: self.specialized_p50().unwrap_or(0),
			last_step: self.records,
		}
	}
}

/// Hysteresis route selection: a route needs measured evidence before it
/// can hold the node, and the loser is re-probed at a bounded cadence so
/// drift can flip the decision but noise cannot. A restored median stands
/// in for a live window only while that window is empty; the live minimum
/// still gates every comparison.
pub(in crate::runtime) fn select_route(
	observation: &mut RouteObservation,
	tape_eligible: bool,
) -> bool {
	if !tape_eligible {
		return true;
	}
	let ready = |window: &VecDeque<u64>, restored: u64| {
		window.len() >= MIN_ROUTE_SAMPLES
			|| (window.is_empty()
				&& observation.restored_samples >= MIN_ROUTE_SAMPLES as u32
				&& restored > 0)
	};
	let specialized_ready = ready(
		&observation.specialized_ns,
		observation.restored_specialized_ns,
	);
	let tape_ready = ready(&observation.tape_ns, observation.restored_tape_ns);
	if !specialized_ready {
		return true;
	}
	if !tape_ready {
		return false;
	}
	let Some(specialized) = observation.specialized_p50() else {
		return true;
	};
	let Some(tape) = observation.tape_p50() else {
		return false;
	};
	if u128::from(specialized) * 10 < u128::from(tape) * 9 {
		observation.specialized = true;
	} else if u128::from(specialized) * 10 > u128::from(tape) * 11 {
		observation.specialized = false;
	}
	if observation.specialized {
		!observation.records.is_multiple_of(REFRESH_PERIOD)
	} else {
		observation.records.is_multiple_of(REPROBE_PERIOD)
	}
}

#[cfg(test)]
#[path = "../../../../../test/rs/runtime/fusion_tuning_unit.rs"]
mod tests;
