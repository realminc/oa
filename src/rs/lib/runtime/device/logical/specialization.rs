//! Device-bound specialization admission and Vulkan pipeline publication.
//!
//! DNA owns IR, compilation and route policy. This adapter keeps per-device
//! admission/pipeline tables, reads completed GPU timings, and integrates their
//! snapshots with the existing device-scoped .vkcache commit. Publication still
//! occurs only at a new recording boundary; in-flight generations are unchanged.

use super::{pipeline::ComputePipeline, vkcache::VkCacheScope};
use crate::runtime::{
	Device,
	device::logical::{
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

pub(in crate::runtime::device) struct JitFusion {
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
	/// `true` once `.vkcache` block-artifact sections have been replayed;
	/// restore runs once on the first block admission that can open the
	/// compiler (compiler-version invalidation needs its identity).
	block_restored: bool,
	/// Validated block SPIR-V keyed by `block_key` — compiled this
	/// session or restored from `.vkcache`; persisted at Runtime drop.
	block_artifacts: BTreeMap<Vec<u8>, BlockArtifact>,
	/// Compiler identity recorded with block artifacts; empty until the
	/// block compiler first opens.
	block_compiler_version: String,
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
	pub(in crate::runtime::device) fn new() -> Self {
		Self {
			restored: false,
			compiler: None,
			artifacts: JitArtifacts::new(),
			pipelines: HashMap::new(),
			block_restored: false,
			block_artifacts: BTreeMap::new(),
			block_compiler_version: String::new(),
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
	pub(in crate::runtime::device) fn admit(
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

	/// Admit a tile-template `BlockProgram` through the same pipeline table
	/// the elementwise specializations use (OA-CC-13). Compilation is
	/// synchronous — the experimental `OA_TILE_GEMM` route admits once per
	/// program key and caches every failure mode negative, so a hot
	/// dispatch never re-enters the compiler. Route observations do not
	/// apply: candidate selection is an OA-CC-12 deliverable.
	pub(in crate::runtime::device) fn admit_block(
		&mut self,
		device: &Device,
		program: &crate::runtime::dna::ir::tile::BlockProgram,
	) -> Option<Vec<u8>> {
		let census = device.fusion_census();
		let bounded = cfg!(oa_profile = "compatibility");
		let key = block_key(program, bounded);
		self.restore_once(device);
		self.restore_blocks_once(device);
		self.drain_pending();
		if matches!(
			jit_route_override(),
			Some(JitRouteOverride::Tape) | Some(JitRouteOverride::Unfused)
		) {
			census.jit(JitOutcome::Disabled);
			return None;
		}
		// OA-CC-12: measured incumbent-vs-tile selection. The hysteresis
		// policy probes the losing route at a bounded cadence; a declined
		// admission lets the canonical kernel run so its timing records
		// under the node's `fusion_signature`.
		if jit_route_override() != Some(JitRouteOverride::Specialized)
			&& let Some(observation) = self.observation_mut(block_signature(program))
		{
			observation.records = observation.records.saturating_add(1);
			if !select_route(observation, true) {
				census.jit(JitOutcome::PolicyTape);
				return None;
			}
		}
		if let Some(slot) = self.pipelines.get(&key) {
			census.jit(if slot.is_some() {
				JitOutcome::Reused
			} else {
				JitOutcome::CachedNegative
			});
			return slot.is_some().then_some(key);
		}
		if self.pipelines.len() >= MAX_PIPELINE_ENTRIES {
			census.jit(JitOutcome::Bound);
			return None;
		}
		// Block-aware push ABI: the reflected push range must equal the
		// program's `buffers + u32 scalars + f32 scalars` u32 fields —
		// the record-time substitute writes exactly that. The program is
		// the authority; restored words are re-checked against it here.
		let expected_push =
			(program.bindings.len() + program.push_scalars.len() + program.push_f32_scalars.len()) as u32
				* 4;
		let pipeline = (|| {
			let words = match self.block_artifacts.get(&key) {
				Some(artifact) => artifact.words.clone(),
				None => {
					let source = program.slang_source(bounded).ok()?;
					let words = self
						.block_compiler()?
						.compile_source(&source, bounded)
						.ok()?;
					crate::runtime::dna::compiler::validate_with_spirv_tools(&words, bounded).ok()?;
					if self.block_artifacts.len() < MAX_PIPELINE_ENTRIES {
						self.block_artifacts.insert(
							key.clone(),
							BlockArtifact {
								words: words.clone(),
								push_bytes: expected_push,
								bindings: program.bindings.len() as u32,
								lanes: program.shape.lanes,
							},
						);
					}
					words
				}
			};
			if crate::runtime::dna::kernel::reflect_push_constant_size(&words).ok()? != expected_push {
				return None;
			}
			let layout = if bounded {
				device.bounded_descriptor_layout(program.bindings.len() as u32)
			} else {
				Some(device.descriptor_layout())
			}?;
			device.pipeline_cache().ok().and_then(|cache| {
				ComputePipeline::from_words(
					device.raw_arc(),
					layout,
					&words,
					[program.shape.lanes, 1, 1],
					device.physical().limits,
					cache.raw(),
				)
				.ok()
			})
		})();
		match pipeline {
			Some(pipeline) => {
				census.jit(JitOutcome::Admitted);
				self.pipelines.insert(key.clone(), Some(Arc::new(pipeline)));
				Some(key)
			}
			None => {
				census.jit(JitOutcome::PipelineRejected);
				self.negative_admission(key);
				None
			}
		}
	}

	/// Lazily opened compiler for the synchronous block route. Unlike the
	/// elementwise path the compiler stays resident here rather than
	/// moving into a background worker.
	fn block_compiler(&mut self) -> Option<&SlangCompiler> {
		if let Some(Ok(compiler)) = &self.compiler {
			self.block_compiler_version = compiler.version.clone();
		}
		if self.compiler.is_none() {
			self.compiler = Some(SlangCompiler::open_default());
		}
		self.compiler.as_ref()?.as_ref().ok()
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
	pub(in crate::runtime::device) fn pipeline(
		&mut self,
		device: &Device,
		key: &[u8],
	) -> Option<Arc<ComputePipeline>> {
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
	pub(in crate::runtime::device) fn observe(&mut self, timings: Arc<RouteTimings>) {
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

	/// Read-only candidate evidence for CC-12 multi-candidate
	/// selection: `(specialized median ns, contributing samples,
	/// admission attempts)` for one signature — the live window first,
	/// the `.vkcache` median while the window is still empty. Restores
	/// run once like the admit path so a fresh session sees persisted
	/// evidence without a prior admission.
	pub(in crate::runtime::device) fn route_stats(
		&mut self,
		device: &Device,
		signature: u64,
	) -> (Option<u64>, usize, u64) {
		self.restore_once(device);
		self.restore_blocks_once(device);
		match self.observations.get(&signature) {
			Some(observation) => (
				observation.specialized_p50(),
				observation.specialized_ns.len() + observation.restored_samples as usize,
				observation.records,
			),
			None => (None, 0, 0),
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

	/// Replay persisted block artifacts once — compiler-version
	/// invalidated, `spirv-val` revalidated, and push-size re-checked at
	/// admission against the live program. Records failing any check are
	/// skipped individually so one damaged entry cannot discard the rest.
	fn restore_blocks_once(&mut self, device: &Device) {
		if self.block_restored {
			return;
		}
		self.block_restored = true;
		let Some(compiler) = self.block_compiler() else {
			return;
		};
		let version = compiler.version.clone().into_bytes();
		let bounded = cfg!(oa_profile = "compatibility");
		let scope = VkCacheScope::new(device.physical());
		let file = match super::vkcache::VkCacheFile::open(&scope.path(), &scope) {
			Ok(file) => file,
			Err(_) => return,
		};
		let index = match file.section(super::vkcache::SECTION_BLOCK_ARTIFACT_INDEX) {
			Some(section) if section.version == 1 => jit::decode_index(&section.bytes),
			_ => return,
		};
		let table = match file.section(super::vkcache::SECTION_BLOCK_SPIRV_ARTIFACTS) {
			Some(section) if section.version == 1 => jit::decode_spirv(&section.bytes),
			_ => return,
		};
		let (index, table) = match (index, table) {
			(Ok(index), Ok(table)) => (index, table),
			_ => return,
		};
		let table: BTreeMap<u64, Vec<u32>> = table
			.into_iter()
			.map(|record| (record.artifact_id, record.words))
			.collect();
		let descriptor_profile = u32::from(bounded);
		for record in index {
			if self.block_artifacts.len() >= MAX_PIPELINE_ENTRIES {
				break;
			}
			if record.descriptor_profile != descriptor_profile
				|| record.emitter_version != BLOCK_EMITTER_VERSION
				|| record.spirv_environment != u32::from(!bounded)
				|| record.compiler != version
				|| !(2..=64).contains(&record.binding_count)
				|| record.push_bytes == 0
				|| record.push_bytes % 4 != 0
				|| !(32..=1024).contains(&record.workgroup_size)
				|| record.flags != 0
				|| jit::artifact_id(
					&record.key,
					record.descriptor_profile,
					record.emitter_version,
				) != record.artifact_id
			{
				continue;
			}
			let Some(words) = table.get(&record.artifact_id) else {
				continue;
			};
			if crate::runtime::dna::compiler::validate_with_spirv_tools(words, bounded).is_err()
				|| crate::runtime::dna::kernel::reflect_push_constant_size(words)
					.map(|size| size != record.push_bytes)
					.unwrap_or(true)
			{
				continue;
			}
			self.block_artifacts.insert(
				record.key,
				BlockArtifact {
					words: words.clone(),
					push_bytes: record.push_bytes,
					bindings: record.binding_count,
					lanes: record.workgroup_size,
				},
			);
		}
	}

	/// Snapshot `.vkcache` section updates for the Runtime's single atomic
	/// commit; empty when no artifacts exist.
	pub(in crate::runtime::device) fn section_updates(
		&mut self,
		census: &FusionCensus,
	) -> Vec<super::vkcache::SectionUpdate> {
		// Runtime Drop snapshots completed host state only; it must not read GPU
		// queries or wait on the worker. Finished compiles fold into the
		// snapshot; anything still compiling simply recompiles next process.
		self.drain_compiles(census);
		let mut updates = self.artifacts.section_updates().unwrap_or_default();
		if !self.block_artifacts.is_empty() {
			let bounded = cfg!(oa_profile = "compatibility");
			let descriptor_profile = u32::from(bounded);
			let mut index = Vec::with_capacity(self.block_artifacts.len());
			let mut spirv = Vec::with_capacity(self.block_artifacts.len());
			for (key, artifact) in &self.block_artifacts {
				let artifact_id = jit::artifact_id(key, descriptor_profile, BLOCK_EMITTER_VERSION);
				index.push(jit::JitIndexRecord {
					artifact_id,
					key: key.clone(),
					descriptor_profile,
					spirv_environment: u32::from(!bounded),
					workgroup_size: artifact.lanes,
					binding_count: artifact.bindings,
					push_bytes: artifact.push_bytes,
					emitter_version: BLOCK_EMITTER_VERSION,
					compiler: self.block_compiler_version.clone().into_bytes(),
					flags: 0,
				});
				spirv.push(jit::JitSpirvRecord {
					artifact_id,
					words: artifact.words.clone(),
				});
			}
			if let (Ok(index), Ok(spirv)) = (jit::encode_index(&index), jit::encode_spirv(&spirv)) {
				updates.push(super::vkcache::SectionUpdate::replace(
					super::vkcache::SECTION_BLOCK_ARTIFACT_INDEX,
					1,
					index,
				));
				updates.push(super::vkcache::SectionUpdate::replace(
					super::vkcache::SECTION_BLOCK_SPIRV_ARTIFACTS,
					1,
					spirv,
				));
			}
		}
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

/// Validated block-artifact words keyed by `block_key` — the SPIR-V
/// plus the reflected push size the program must still match at
/// admission. `bindings`/`lanes` live in the program, not the record.
struct BlockArtifact {
	words: Vec<u32>,
	push_bytes: u32,
	bindings: u32,
	lanes: u32,
}

/// Emitter generation for persisted block artifacts — bump when the
/// tile-IR emission contract changes so stale artifacts invalidate.
const BLOCK_EMITTER_VERSION: u32 = 2;

fn jit_missing(what: &str) -> crate::Error {
	crate::Error::missing_capability(format!("JIT specialization {what} unavailable"))
}

/// Deterministic admission key for a tile-template program: template
/// identity plus its full IR description and descriptor profile.
/// Compiler-version invalidation and `.vkcache` persistence are
/// OA-CC-12 artifact-key deliverables.
pub(in crate::runtime) fn block_key(
	program: &crate::runtime::dna::ir::tile::BlockProgram,
	bounded: bool,
) -> Vec<u8> {
	let mut hash = 0xcbf2_9ce4_8422_2325_u64;
	for byte in format!("oa.tile.v1:{bounded}:{program:?}").bytes() {
		hash ^= u64::from(byte);
		hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
	}
	format!("oa.tile.gemm_nt.{hash:016x}").into_bytes()
}

/// Route-observation signature for a block program: one problem class per
/// (template, geometry, descriptor profile), persisted through the shared
/// section-6 observation table. Shape scalars are push data, not program
/// identity — all shapes under one tile class share the record.
pub(in crate::runtime) fn block_signature(
	program: &crate::runtime::dna::ir::tile::BlockProgram,
) -> u64 {
	jit::signature_key(&block_key(program, cfg!(oa_profile = "compatibility")))
}

#[cfg(test)]
#[path = "../../../../../../test/rs/runtime/jit_policy_unit.rs"]
mod tests;
