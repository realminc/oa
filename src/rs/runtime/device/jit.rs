//! Device-bound specialization state for the opt-in `OA_JIT_FUSION` route.
//!
//! One lock serializes restore, artifact admission, pipeline publication and
//! record-time lookup. Compilation is non-blocking: on a true cache miss the
//! canonical tape or unfused route records immediately while a bounded
//! background queue compiles the program. The worker owns the Slang session
//! outright and never borrows the thread-affine Engine or touches live
//! command recording; a finished compile only becomes dispatchable when
//! `pipeline` creates the Vulkan pipeline at the boundary of a new recording,
//! so an executable generation in flight is never patched. A failed admission
//! is remembered negatively so recurring submissions never pay the compiler
//! again, and every failure mode returns `None`, keeping the recorded
//! canonical route in place.
//!
//! Route observations (`vkcache` section 6) close the measurement loop:
//! recording timestamps each fused dispatch and the completed samples land
//! here through `observe`. `admit` then applies a hysteresis policy per
//! canonical signature — the specialized route must beat tape by a margin
//! before it keeps the node, and a demoted route is re-probed periodically so
//! a workload drift cannot pin the wrong selection. Probing only ever chooses
//! routes that already exist; above the tape eligibility bound the
//! specialized dispatch is the only fused route and no probe runs.

use std::collections::{BTreeMap, BTreeSet, HashMap, VecDeque};
use std::sync::Arc;
use std::time::Instant;

use crate::runtime::{
	Device,
	device::{
		features::ExecutionProfile,
		timestamp::{ObservedRoute, RouteTimings},
		vkcache::jit::{self, RouteObservationRecord},
	},
	dna::{
		elementwise::Program,
		kernel::jit::{CompiledChain, JitArtifacts, SlangCompiler},
	},
	fusion::census::{FusionCensus, JitOutcome},
};

use super::{pipeline::ComputePipeline, vkcache::VkCacheScope};

/// Timing samples kept per route; the median of the live window wins over
/// restored medians so policy follows the current workload.
const MAX_ROUTE_SAMPLES: usize = 32;
/// Live samples required before a persisted median alone decides probes.
const MIN_ROUTE_SAMPLES: usize = 4;
/// Records between re-probes of a demoted specialized route.
const REPROBE_PERIOD: u64 = 16;
/// Records between tape refresh probes while specialized holds.
const REFRESH_PERIOD: u64 = 32;
/// Bounded per-device observation state.
const MAX_OBSERVED_SIGNATURES: usize = 256;
/// Completed submissions whose query pools wait here for a lazy read.
const MAX_PENDING_TIMINGS: usize = 64;
/// Positive and negative admissions share one hard memory/compile-work bound.
const MAX_PIPELINE_ENTRIES: usize = 64;
/// Compilations waiting on the single background worker; together with the
/// admission bound this caps total compile work per device lifetime.
const MAX_QUEUED_COMPILATIONS: usize = 16;
/// Version 2 records the minimum qualified sample count across both routes.
/// Version 1 summed counts and could promote an under-sampled restored route.
const OBSERVATION_VERSION: u32 = 2;

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
struct RouteObservation {
	/// Admission calls (≈ recordings) of this signature; also persisted as
	/// `last_step`.
	records: u64,
	/// Route the policy currently selects: `true` = specialized.
	specialized: bool,
	/// Recent specialized-route GPU durations, newest last.
	specialized_ns: VecDeque<u64>,
	/// Recent tape-route GPU durations, newest last.
	tape_ns: VecDeque<u64>,
	/// Persisted medians restored from `.vkcache`; used only while the live
	/// window is empty.
	restored_samples: u32,
	restored_specialized_ns: u64,
	restored_tape_ns: u64,
}

impl RouteObservation {
	fn new() -> Self {
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

	fn restore(record: &RouteObservationRecord) -> Self {
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

	fn push(&mut self, route: ObservedRoute, ns: u64) {
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

	fn specialized_p50(&self) -> Option<u64> {
		Self::median(&self.specialized_ns).or_else(|| {
			(self.restored_samples >= MIN_ROUTE_SAMPLES as u32 && self.restored_specialized_ns > 0)
				.then_some(self.restored_specialized_ns)
		})
	}

	fn tape_p50(&self) -> Option<u64> {
		Self::median(&self.tape_ns).or_else(|| {
			(self.restored_samples >= MIN_ROUTE_SAMPLES as u32 && self.restored_tape_ns > 0)
				.then_some(self.restored_tape_ns)
		})
	}

	fn record(&self, signature_key: u64) -> RouteObservationRecord {
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

/// One queued specialization: everything the worker needs to compile the
/// program for the device's descriptor profile.
struct CompileRequest {
	program: Program,
	bounded: bool,
}

/// Folded back under the jit lock by the next admission, record lookup or
/// persistence snapshot. A failed outcome becomes a negative admission.
struct CompileResult {
	key: Vec<u8>,
	compiler_version: String,
	result: Result<CompiledChain, JitOutcome>,
}

/// Bounded background compiler: Slang compilation, `spirv-val` and ABI
/// reflection run off the admission path on one worker that owns the compiler
/// session. The worker never borrows the thread-affine Engine and never
/// touches command recording, device state or GPU resources.
struct CompileWorker {
	requests: std::sync::mpsc::SyncSender<CompileRequest>,
	results: std::sync::mpsc::Receiver<CompileResult>,
	/// The detached worker exits once the request channel closes; joining is
	/// never required because it owns every resource it touches.
	_thread: std::thread::JoinHandle<()>,
}

impl CompileWorker {
	fn spawn(compiler: SlangCompiler) -> std::io::Result<Self> {
		let (requests, inbox) = std::sync::mpsc::sync_channel(MAX_QUEUED_COMPILATIONS);
		let (results, outbox) = std::sync::mpsc::channel();
		let thread = std::thread::Builder::new()
			.name("oa-jit-compile".into())
			.spawn(move || compile_worker_main(compiler, inbox, results))?;
		Ok(Self {
			requests,
			results: outbox,
			_thread: thread,
		})
	}
}

fn compile_worker_main(
	compiler: SlangCompiler,
	inbox: std::sync::mpsc::Receiver<CompileRequest>,
	outbox: std::sync::mpsc::Sender<CompileResult>,
) {
	while let Ok(request) = inbox.recv() {
		let result = CompileResult {
			key: request.program.artifact_key(),
			compiler_version: compiler.version.clone(),
			result: compiler
				.compile_program(&request.program, request.bounded)
				.map_err(|error| {
					crate::log_warn!(
						crate::LogComponent::RUNTIME,
						"JIT specialization admission rejected: {error:?}"
					);
					JitOutcome::CompileRejected
				}),
		};
		if outbox.send(result).is_err() {
			return;
		}
	}
}

pub(super) struct JitFusion {
	/// `true` once `.vkcache` JIT sections have been replayed (successfully or
	/// not); restore runs exactly once on the first opt-in request.
	restored: bool,
	/// Lazily opened compiler, moved into the background worker on the first
	/// queued compile; `Some(Err)` poisons compilation for the rest of the
	/// device lifetime so a missing library is never retried per chain.
	compiler: Option<crate::Result<SlangCompiler>>,
	artifacts: JitArtifacts,
	/// `None` marks an admission that failed this session — compile, ABI,
	/// layout or pipeline creation — so it is never retried.
	pipelines: HashMap<Vec<u8>, Option<Arc<ComputePipeline>>>,
	/// Per-signature route measurements and the hysteresis selection they
	/// drive, keyed by `jit::signature_key` of the canonical program.
	observations: BTreeMap<u64, RouteObservation>,
	/// Recorded query pools awaiting a completed readback.
	pending: VecDeque<Arc<RouteTimings>>,
	/// Background compile worker, spawned lazily on the first true artifact
	/// miss, and the artifact keys it has not finished yet.
	worker: Option<CompileWorker>,
	in_flight: BTreeSet<Vec<u8>>,
}

impl JitFusion {
	pub(super) fn new() -> Self {
		Self {
			restored: false,
			compiler: None,
			artifacts: JitArtifacts::new(),
			pipelines: HashMap::new(),
			observations: BTreeMap::new(),
			pending: VecDeque::new(),
			worker: None,
			in_flight: BTreeSet::new(),
		}
	}

	/// Ensure the program's specialization is compiled or restored and a
	/// dispatchable pipeline exists. Returns the artifact key for the fused
	/// node on success; `None` leaves the canonical tape route in place —
	/// including while a background compile is still in flight. `tape_eligible`
	/// is `false` above the tape element bound, where the specialized route is
	/// the only fused route and no tape probe may run.
	pub(super) fn admit(
		&mut self,
		device: &Device,
		program: &Program,
		tape_eligible: bool,
	) -> Option<Vec<u8>> {
		let census = device.fusion_census();
		self.restore_once(device);
		self.drain_pending();
		if matches!(
			jit_route_override(),
			Some(JitRouteOverride::Tape) | Some(JitRouteOverride::Unfused)
		) {
			census.jit(JitOutcome::Disabled);
			return None;
		}
		let signature = jit::signature_key(&program.canonical_key());
		if jit_route_override() != Some(JitRouteOverride::Specialized)
			&& let Some(observation) = self.observation_mut(signature)
		{
			observation.records = observation.records.saturating_add(1);
			if !Self::select_route(observation, tape_eligible) {
				census.jit(JitOutcome::PolicyTape);
				return None;
			}
		}
		let key = program.artifact_key();
		if let Some(slot) = self.pipelines.get(&key) {
			census.jit(if slot.is_some() {
				JitOutcome::Reused
			} else {
				JitOutcome::CachedNegative
			});
			return slot.is_some().then_some(key);
		}
		self.drain_compiles(census);
		// A drained result can only land as a negative admission; pipeline
		// creation stays on the publication path below.
		if self.pipelines.contains_key(&key) {
			census.jit(JitOutcome::CachedNegative);
			return None;
		}
		let bounded = device.physical().profile == ExecutionProfile::Compatibility;
		if self.artifacts.get(program, bounded).is_some() {
			return self.publish(device, &key).map(|_| key);
		}
		if self.in_flight.contains(&key) {
			census.jit(JitOutcome::Pending);
			return None;
		}
		if self.pipelines.len() + self.in_flight.len() >= MAX_PIPELINE_ENTRIES
			|| self.artifacts.len() + self.in_flight.len() >= JitArtifacts::MAX_CHAINS
		{
			census.jit(JitOutcome::Bound);
			return None;
		}
		self.queue_compile(census, program, bounded);
		None
	}

	/// Queue `program` for background compilation. The canonical route keeps
	/// running until a later admission publishes the finished artifact, so a
	/// hot caller is never blocked on the compiler.
	fn queue_compile(&mut self, census: &FusionCensus, program: &Program, bounded: bool) {
		let key = program.artifact_key();
		if self.worker.is_none() {
			let Some(compiler) = self.worker_compiler() else {
				census.jit(JitOutcome::CompilerUnavailable);
				self.negative_admission(key);
				return;
			};
			match CompileWorker::spawn(compiler) {
				Ok(worker) => self.worker = Some(worker),
				Err(error) => {
					crate::log_warn!(
						crate::LogComponent::RUNTIME,
						"JIT specialization worker unavailable: {error}"
					);
					census.jit(JitOutcome::CompilerUnavailable);
					self.negative_admission(key);
					return;
				}
			}
		}
		let request = CompileRequest {
			program: program.clone(),
			bounded,
		};
		match self
			.worker
			.as_ref()
			.map(|worker| worker.requests.try_send(request))
		{
			Some(Ok(())) => {
				self.in_flight.insert(key);
				census.jit(JitOutcome::Queued);
			}
			Some(Err(std::sync::mpsc::TrySendError::Full(_))) => {
				census.jit(JitOutcome::Bound);
			}
			_ => {
				// The worker thread is gone; compilation cannot proceed.
				census.jit(JitOutcome::CompilerUnavailable);
				self.negative_admission(key);
				self.worker = None;
			}
		}
	}

	/// Resolve the compiler moved into the first worker. `Some(Err)` poisons
	/// the device lifetime; a fresh worker spawn after a dead one retries the
	/// open, which stays bounded by the negative admissions it produces.
	fn worker_compiler(&mut self) -> Option<SlangCompiler> {
		match self.compiler.take() {
			Some(Ok(compiler)) => Some(compiler),
			Some(Err(error)) => {
				self.compiler = Some(Err(error));
				None
			}
			None => match SlangCompiler::open_default() {
				Ok(compiler) => Some(compiler),
				Err(error) => {
					crate::log_warn!(
						crate::LogComponent::RUNTIME,
						"JIT specialization compiler unavailable: {error}"
					);
					self.compiler = Some(Err(error));
					None
				}
			},
		}
	}

	/// Remember a failed admission so recurring submissions never retry it.
	fn negative_admission(&mut self, key: Vec<u8>) {
		if self.pipelines.len() < MAX_PIPELINE_ENTRIES {
			self.pipelines.insert(key, None);
		}
	}

	/// Fold finished background compiles into the artifact table. A failure
	/// becomes a negative admission; a dead worker turns its in-flight keys
	/// into negative admissions too rather than retrying lost work blindly.
	fn drain_compiles(&mut self, census: &FusionCensus) {
		let mut dead = false;
		loop {
			let result = match self.worker.as_mut().map(|worker| worker.results.try_recv()) {
				Some(Ok(result)) => result,
				Some(Err(std::sync::mpsc::TryRecvError::Disconnected)) => {
					dead = true;
					break;
				}
				_ => break,
			};
			self.in_flight.remove(&result.key);
			match result.result {
				Ok(chain) => {
					census.jit_compile(chain.compile_time);
					if let Err(error) = self.artifacts.insert(&result.compiler_version, chain) {
						crate::log_warn!(
							crate::LogComponent::RUNTIME,
							"JIT specialization artifact rejected: {error}"
						);
						census.jit(JitOutcome::Bound);
						self.negative_admission(result.key);
					}
				}
				Err(outcome) => {
					census.jit(outcome);
					self.negative_admission(result.key);
				}
			}
		}
		if dead {
			self.worker = None;
			for key in std::mem::take(&mut self.in_flight) {
				census.jit(JitOutcome::CompilerUnavailable);
				self.negative_admission(key);
			}
		}
	}

	/// Create the dispatchable pipeline for a compiled artifact and admit it.
	/// Publication only ever happens here, inside a new admission or a new
	/// recording, so an executable generation in flight is never patched.
	fn publish(&mut self, device: &Device, key: &[u8]) -> Option<Arc<ComputePipeline>> {
		if self.pipelines.len() >= MAX_PIPELINE_ENTRIES {
			return None;
		}
		let census = device.fusion_census();
		let bounded = device.physical().profile == ExecutionProfile::Compatibility;
		let artifact = self.artifacts.get_key(bounded, key)?;
		let pipeline_started = Instant::now();
		let layout = if bounded {
			device.bounded_descriptor_layout(artifact.bindings)
		} else {
			Some(device.descriptor_layout())
		};
		let pipeline = layout
			.ok_or_else(|| jit_missing("descriptor layout"))
			.and_then(|layout| {
				device
					.pipeline_cache()
					.map_err(|_| jit_missing("pipeline cache"))
					.and_then(|cache| {
						ComputePipeline::from_words(
							device.raw_arc(),
							layout,
							artifact.words(),
							[256, 1, 1],
							device.physical().limits,
							cache.raw(),
						)
					})
			});
		census.jit_pipeline(pipeline_started.elapsed());
		match pipeline {
			Ok(pipeline) => {
				census.jit(JitOutcome::Admitted);
				let pipeline = Arc::new(pipeline);
				self.pipelines.insert(key.to_vec(), Some(pipeline.clone()));
				Some(pipeline)
			}
			Err(error) => {
				census.jit(JitOutcome::PipelineRejected);
				crate::log_warn!(
					crate::LogComponent::RUNTIME,
					"JIT specialization pipeline rejected: {error:?}"
				);
				self.pipelines.insert(key.to_vec(), None);
				None
			}
		}
	}

	/// Hysteresis route selection: a route needs measured evidence before it
	/// can hold the node, and the loser is re-probed at a bounded cadence so
	/// drift can flip the decision but noise cannot. A restored median stands
	/// in for a live window only while that window is empty; the live minimum
	/// still gates every comparison.
	fn select_route(observation: &mut RouteObservation, tape_eligible: bool) -> bool {
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

	/// Record-time lookup for a key that `admit` prepared. A compiled artifact
	/// publishes its dispatchable pipeline here, at the boundary of a new
	/// executable generation; a missing or declined entry fails closed to the
	/// canonical tape route.
	pub(super) fn pipeline(&mut self, device: &Device, key: &[u8]) -> Option<Arc<ComputePipeline>> {
		let census = device.fusion_census();
		self.drain_compiles(census);
		if let Some(slot) = self.pipelines.get(key) {
			return slot.clone();
		}
		self.publish(device, key)
	}

	/// Queue a recorded route-timing pool for lazy readback. The pool is
	/// bounded; the oldest unread pools are dropped when a workload stops
	/// submitting them.
	pub(super) fn observe(&mut self, timings: Arc<RouteTimings>) {
		if self.pending.len() >= MAX_PENDING_TIMINGS {
			self.pending.pop_front();
		}
		self.pending.push_back(timings);
	}

	/// Fold completed timing readbacks into the observation windows. Called
	/// under the jit lock at every admission, so samples land on
	/// the next lock acquisition after their submission completes.
	fn drain_pending(&mut self) {
		let mut index = 0;
		while index < self.pending.len() {
			let Some(durations) = self.pending[index].try_durations() else {
				index += 1;
				continue;
			};
			self.pending.remove(index);
			for (signature, route, duration) in durations {
				if let Some(observation) = self.observation_mut(signature) {
					observation.push(route, duration.as_nanos() as u64);
				}
			}
		}
	}

	fn observation_mut(&mut self, signature: u64) -> Option<&mut RouteObservation> {
		if !self.observations.contains_key(&signature)
			&& self.observations.len() >= MAX_OBSERVED_SIGNATURES
		{
			return None;
		}
		Some(
			self
				.observations
				.entry(signature)
				.or_insert_with(RouteObservation::new),
		)
	}

	fn restore_once(&mut self, device: &Device) {
		if self.restored {
			return;
		}
		self.restored = true;
		let scope = VkCacheScope::new(device.physical());
		self.artifacts = JitArtifacts::restore(&scope.path(), &scope).unwrap_or_else(|error| {
			crate::log_warn!(
				crate::LogComponent::RUNTIME,
				"discarding persisted JIT artifacts: {error}"
			);
			JitArtifacts::new()
		});
		if let Ok(file) = super::vkcache::VkCacheFile::open(&scope.path(), &scope)
			&& let Some(section) = file.section(super::vkcache::SECTION_FUSION_ROUTE_OBSERVATIONS)
			&& section.version == OBSERVATION_VERSION
		{
			match jit::decode_observations(&section.bytes) {
				Ok(records) => {
					for record in &records {
						if self.observations.len() >= MAX_OBSERVED_SIGNATURES {
							break;
						}
						self
							.observations
							.entry(record.signature_key)
							.or_insert_with(|| RouteObservation::restore(record));
					}
				}
				Err(error) => crate::log_warn!(
					crate::LogComponent::RUNTIME,
					"discarding persisted fusion route observations: {error}"
				),
			}
		}
	}

	/// Snapshot `.vkcache` section updates for the Engine's single atomic
	/// commit; empty when no artifacts exist.
	pub(super) fn section_updates(
		&mut self,
		census: &FusionCensus,
	) -> Vec<super::vkcache::SectionUpdate> {
		// Engine Drop snapshots completed host state only; it must not read GPU
		// queries or wait on the worker. Finished compiles fold into the
		// snapshot; anything still compiling simply recompiles next process.
		self.drain_compiles(census);
		let mut updates = self.artifacts.section_updates().unwrap_or_default();
		if !self.observations.is_empty() {
			let records: Vec<RouteObservationRecord> = self
				.observations
				.iter()
				.map(|(signature, observation)| observation.record(*signature))
				.collect();
			if let Ok(payload) = jit::encode_observations(&records) {
				updates.push(super::vkcache::SectionUpdate::replace(
					super::vkcache::SECTION_FUSION_ROUTE_OBSERVATIONS,
					OBSERVATION_VERSION,
					payload,
				));
			}
		}
		updates
	}
}

fn jit_missing(what: &str) -> crate::Error {
	crate::Error::missing_capability(format!("JIT specialization {what} unavailable"))
}

#[cfg(test)]
#[path = "../../../../test/rs/runtime/jit_policy_unit.rs"]
mod tests;
