//! Device-bound specialization admission and Vulkan pipeline publication.
//!
//! DNA owns IR, compilation and route policy. This adapter keeps per-device
//! admission/pipeline tables, reads completed GPU timings, and integrates their
//! snapshots with the existing device-scoped .vkcache commit. Publication still
//! occurs only at a new recording boundary; in-flight generations are unchanged.

use super::{pipeline::ComputePipeline, vkcache::VkCacheScope};
use crate::runtime::{
	Device,
	device::{
		timestamp::RouteTimings,
		vkcache::jit::{self, RouteObservationRecord},
	},
	dna::{
		compiler::{
			JitArtifacts, SlangCompiler,
			worker::{CompileRequest, CompileWorker},
		},
		fusion::census::{FusionCensus, JitOutcome},
		ir::elementwise::Program,
		tuning::{JitRouteOverride, RouteObservation, jit_route_override, select_route},
	},
};
use std::collections::{BTreeMap, BTreeSet, HashMap, VecDeque};
use std::sync::Arc;
use std::time::Instant;

/// Bounded per-device observation state.
const MAX_OBSERVED_SIGNATURES: usize = 256;
/// Completed submissions whose query pools wait here for a lazy read.
const MAX_PENDING_TIMINGS: usize = 64;
/// Positive and negative admissions share one hard memory/compile-work bound.
const MAX_PIPELINE_ENTRIES: usize = 64;
/// Version 2 records the minimum qualified sample count across both routes.
/// Version 1 summed counts and could promote an under-sampled restored route.
const OBSERVATION_VERSION: u32 = 2;

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
			if !select_route(observation, tape_eligible) {
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
		let bounded = cfg!(oa_profile = "compatibility");
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
		let bounded = cfg!(oa_profile = "compatibility");
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
#[path = "../../../../../test/rs/runtime/jit_policy_unit.rs"]
mod tests;
