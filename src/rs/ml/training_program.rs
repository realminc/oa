use std::time::Duration;

use crate::{
	DType, Engine, Event, ExecutionPlan, ExecutionPlanDiagnostics, Matrix, Result, SemanticGraph,
	runtime::CaptureAttempt,
};

use super::{AdamW, optimizer::AdamWProgramSignature};

/// Ordered compilation stage for a captured training program.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TrainingCompilationStage {
	/// Validate the backend-independent semantic graph.
	SemanticValidation,
	/// Reject kernels whose mutable state cannot advance safely during replay.
	ReplaySafety,
	/// Analyze decomposition of semantic operations into executable work.
	Decomposition,
	/// Analyze candidate DNN partitions and fusion regions.
	Fusion,
	/// Resolve execution placement inherited from the originating engine.
	Placement,
	/// Resolve precision policy inherited from recorded operations.
	Precision,
	/// Resolve concrete kernels inherited from recorded dispatches.
	KernelSelection,
	/// Validate semantic-to-executable lowering provenance.
	LoweringValidation,
	/// Analyze and, when profitable, materialize transient aliases.
	MemoryPlanning,
	/// Plan executable resource hazards and synchronization.
	SynchronizationPlanning,
	/// Record the reusable backend command.
	CommandRecording,
}

impl TrainingCompilationStage {
	/// Return the stable report token for this stage.
	pub const fn token(self) -> &'static str {
		match self {
			Self::SemanticValidation => "semantic_validation",
			Self::ReplaySafety => "replay_safety",
			Self::Decomposition => "decomposition",
			Self::Fusion => "fusion",
			Self::Placement => "placement",
			Self::Precision => "precision",
			Self::KernelSelection => "kernel_selection",
			Self::LoweringValidation => "lowering_validation",
			Self::MemoryPlanning => "memory_planning",
			Self::SynchronizationPlanning => "synchronization_planning",
			Self::CommandRecording => "command_recording",
		}
	}
}

/// Outcome of one training-program compilation stage.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TrainingCompilationState {
	/// The stage did not run during this capture transaction.
	NotRun,
	/// The stage consumed an already-resolved decision from recording.
	Inherited,
	/// The stage analyzed its input without changing the executable plan.
	Analyzed,
	/// The stage validated or changed the executable plan successfully.
	Applied,
	/// The stage failed and prevented capture commit.
	Failed,
}

impl TrainingCompilationState {
	/// Return the stable report token for this state.
	pub const fn token(self) -> &'static str {
		match self {
			Self::NotRun => "not_run",
			Self::Inherited => "inherited",
			Self::Analyzed => "analyzed",
			Self::Applied => "applied",
			Self::Failed => "failed",
		}
	}
}

/// Immutable evidence for one training-program compilation stage.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TrainingCompilationStageRecord {
	stage: TrainingCompilationStage,
	state: TrainingCompilationState,
	input_count: usize,
	output_count: usize,
}

impl TrainingCompilationStageRecord {
	/// Return the compilation stage.
	pub const fn stage(self) -> TrainingCompilationStage {
		self.stage
	}

	/// Return how the stage completed during capture.
	pub const fn state(self) -> TrainingCompilationState {
		self.state
	}

	/// Return the number of stage-specific input work items.
	pub const fn input_count(self) -> usize {
		self.input_count
	}

	/// Return the number of stage-specific output work items.
	pub const fn output_count(self) -> usize {
		self.output_count
	}
}

/// Fixed-shape forward, backward, and AdamW program captured for Vulkan replay.
///
/// The first capture records semantic work without executing it. Replays use one
/// cached command buffer and graph-resident optimizer state; no forward,
/// autograd, or optimizer graph is rebuilt on the host.
#[must_use]
pub struct TrainingProgram {
	plan: ExecutionPlan,
	loss: Matrix,
	optimizer: AdamWProgramSignature,
	compilation_stages: [TrainingCompilationStageRecord; 11],
	replays: u32,
}

pub(super) enum TrainingProgramCapture {
	Captured(Box<TrainingProgram>),
	Rejected {
		error: crate::Error,
		loss: Matrix,
		optimizer: AdamWProgramSignature,
	},
}

impl TrainingProgram {
	/// Capture one complete fixed-shape forward/backward/AdamW step.
	///
	/// `step` must construct and return a scalar F32 loss and record its backward
	/// pass. This method clears existing gradients and records the optimizer update
	/// after `step` returns. Capture performs no submission or waiting.
	///
	/// # Errors
	///
	/// Returns an error from execution capture, `step`, backward/optimizer
	/// recording, or when the returned value is not a scalar F32 loss or every
	/// optimizer parameter did not receive a stable gradient.
	pub fn capture(
		engine: &Engine,
		optimizer: &mut AdamW,
		step: impl FnOnce() -> Result<Matrix>,
	) -> Result<Self> {
		let (plan, loss) = engine.capture_observed_matrix(|| {
			optimizer.zero_grad();
			let loss = step()?;
			if !loss.shape().is_empty() || loss.dtype() != DType::F32 {
				return Err(crate::Error::invalid_argument(
					"training program requires a scalar F32 loss",
				));
			}
			optimizer.step()?;
			Ok(loss)
		})?;
		plan.validate_training_replay_safety()?;
		let optimizer = optimizer.program_signature()?;
		let compilation_stages = compilation_stage_records(plan.diagnostics());
		Ok(Self {
			plan,
			loss,
			optimizer,
			compilation_stages,
			replays: 0,
		})
	}

	pub(super) fn capture_preserving_recording(
		engine: &Engine,
		optimizer: &mut AdamW,
		step: impl FnOnce() -> Result<Matrix>,
	) -> Result<TrainingProgramCapture> {
		let attempt = engine.capture_observed_training_matrix_preserving(|| {
			optimizer.zero_grad();
			let loss = step()?;
			if !loss.shape().is_empty() || loss.dtype() != DType::F32 {
				return Err(crate::Error::invalid_argument(
					"training program requires a scalar F32 loss",
				));
			}
			optimizer.step()?;
			Ok(loss)
		})?;
		let optimizer_signature = optimizer.program_signature()?;
		Ok(match attempt {
			CaptureAttempt::Captured { plan, output: loss } => {
				let compilation_stages = compilation_stage_records(plan.diagnostics());
				TrainingProgramCapture::Captured(Box::new(Self {
					plan: *plan,
					loss,
					optimizer: optimizer_signature,
					compilation_stages,
					replays: 0,
				}))
			}
			CaptureAttempt::Rejected {
				error,
				output: loss,
			} => TrainingProgramCapture::Rejected {
				error,
				loss,
				optimizer: optimizer_signature,
			},
		})
	}

	/// Upload a new fixed-shape host batch into a captured input slot.
	///
	/// A pending prior replay is completed before the stable input buffer is
	/// overwritten. The cached command remains valid.
	///
	/// # Errors
	///
	/// Returns the input-slot validation, completion, or upload error.
	pub fn upload_input<T: crate::core::Element>(
		&mut self,
		captured: &Matrix,
		values: &[T],
	) -> Result<()> {
		self.plan.upload_matrix_input(captured, values)
	}

	/// Submit one replay without waiting.
	///
	/// Same-queue timeline ordering serializes repeated updates. Call
	/// [`TrainingProgram::upload_input`] before changing a reused input slot; it
	/// owns the required completion wait.
	///
	/// # Errors
	///
	/// Returns an error when the engine or optimizer differs, stable captured state
	/// was replaced, a counter is exhausted, or submission fails.
	pub fn replay(&mut self, engine: &Engine, optimizer: &mut AdamW) -> Result<Event> {
		self.replay_impl(engine, optimizer, false)
	}

	/// Submit one replay with device timestamps around the complete program.
	///
	/// # Errors
	///
	/// Returns the errors from [`TrainingProgram::replay`], or
	/// `MissingCapability` when the selected queue cannot provide timestamps.
	pub fn replay_timed(&mut self, engine: &Engine, optimizer: &mut AdamW) -> Result<Event> {
		self.replay_impl(engine, optimizer, true)
	}

	fn replay_impl(
		&mut self,
		engine: &Engine,
		optimizer: &mut AdamW,
		timed: bool,
	) -> Result<Event> {
		let expected_step = self
			.optimizer
			.base_step
			.checked_add(self.replays)
			.ok_or_else(|| crate::Error::resource_exhausted("training step counter exhausted"))?;
		let logical_step = expected_step
			.checked_add(1)
			.ok_or_else(|| crate::Error::resource_exhausted("training step counter exhausted"))?;
		optimizer.validate_program_replay(&self.optimizer, expected_step, logical_step)?;
		let event = if timed {
			engine.submit_timed(&self.plan)?
		} else {
			engine.submit(&self.plan)?
		};
		optimizer.complete_program_replay(&self.optimizer, logical_step)?;
		self.replays = self
			.replays
			.checked_add(1)
			.ok_or_else(|| crate::Error::resource_exhausted("training replay counter exhausted"))?;
		Ok(event)
	}

	/// Submit one replay, wait for its exact completion, and read its scalar loss.
	///
	/// # Errors
	///
	/// Returns the replay, completion, or loss-read error.
	pub fn replay_and_wait(&mut self, engine: &Engine, optimizer: &mut AdamW) -> Result<f32> {
		self.replay(engine, optimizer)?.wait()?;
		Ok(self.loss.read_f32()?[0])
	}

	/// Submit one timed replay, wait, and return its scalar loss and device time.
	///
	/// # Errors
	///
	/// Returns the timed replay, completion, timestamp, or loss-read error.
	pub fn replay_timed_and_wait(
		&mut self,
		engine: &Engine,
		optimizer: &mut AdamW,
	) -> Result<(f32, Duration)> {
		let event = self.replay_timed(engine, optimizer)?;
		let duration = event.device_duration()?;
		Ok((self.loss.read_f32()?[0], duration))
	}

	/// Return the captured scalar loss value.
	pub const fn loss(&self) -> &Matrix {
		&self.loss
	}

	/// Return graph, command-cache, submission, and input-upload diagnostics.
	pub fn diagnostics(&self) -> ExecutionPlanDiagnostics {
		self.plan.diagnostics()
	}

	/// Return the captured backend-independent training graph.
	pub const fn semantic_graph(&self) -> &SemanticGraph {
		self.plan.semantic_graph()
	}

	/// Return the backend-independent captured semantic graph as deterministic JSON.
	pub fn semantic_debug_report_json(&self, name: &str) -> String {
		self.plan.semantic_graph().debug_report_json(name)
	}

	/// Return the normalized captured executable graph as deterministic JSON.
	pub fn debug_report_json(&self, name: &str) -> String {
		self.plan.debug_report_json(name)
	}

	/// Return how many training steps this program submitted.
	pub const fn replay_count(&self) -> u32 {
		self.replays
	}

	/// Return whether the most recently submitted training replay has completed.
	///
	/// A captured program that has not replayed is complete.
	///
	/// # Errors
	///
	/// Returns an error when Vulkan cannot query the originating timeline.
	pub fn is_complete(&self) -> Result<bool> {
		self.plan.is_complete()
	}

	/// Wait for the most recently submitted replay and host retirement.
	///
	/// # Errors
	///
	/// Returns an error when Vulkan completion or host retirement waiting fails.
	pub fn wait(&self) -> Result<()> {
		self.plan.wait()
	}

	/// Wait for pending replay and consume the program, releasing captured state.
	///
	/// Capture constructs a new Rust value, so consuming reset replaces the C++
	/// donor's reusable empty shell. Ordinary drop remains non-blocking.
	///
	/// # Errors
	///
	/// Returns an error from [`TrainingProgram::wait`]. Submitted resources remain
	/// protected by engine retirement if the wait fails.
	pub fn reset(self) -> Result<()> {
		self.plan.reset()
	}

	/// Return the donor-compatible deterministic compilation report as JSON.
	///
	/// The report contains no Vulkan handles, addresses, descriptors, or mutable
	/// routing controls. It records capture-time stages, private DNN analysis,
	/// semantic lowering provenance, and logical memory-planning evidence.
	pub fn compilation_debug_report_json(&self, name: &str) -> String {
		let diagnostics = self.plan.diagnostics();
		let lowering = self.plan.semantic_lowering();
		let resources = self.plan.captured_resources();
		let mut output = String::new();
		output.push_str("{\n  \"schema\": \"oa.training_compilation.v2\",\n  \"name\": ");
		crate::core::push_json_string(&mut output, name);
		output.push_str(",\n  \"captured\": true,\n  \"stages\": [");
		for (index, stage) in self.compilation_stages.iter().enumerate() {
			output.push_str(if index == 0 { "\n" } else { ",\n" });
			crate::core::push_format(
				&mut output,
				format_args!(
					"    {{\"stage\": \"{}\", \"state\": \"{}\", \"input_count\": {}, \"output_count\": {}}}",
					stage.stage.token(),
					stage.state.token(),
					stage.input_count,
					stage.output_count
				),
			);
		}
		output.push_str("\n  ],\n  \"dnn_plan\": {\n");
		crate::core::push_format(
			&mut output,
			format_args!(
				"    \"graph_hash\": {},\n    \"source_operation_count\": {},\n    \"captured_operation_count\": {},\n    \"partition_count\": {},\n    \"recognized_partition_count\": {},\n    \"applied_partition_count\": {},\n    \"inherited_partition_count\": {},\n    \"fallback_partition_count\": {},\n    \"unexpected_fallback_count\": {},\n    \"fallback_reasons\": [",
				diagnostics.dnn_graph_hash(),
				diagnostics.semantic_operation_count(),
				diagnostics.dnn_captured_operation_count(),
				diagnostics.dnn_partition_count(),
				diagnostics.dnn_recognized_partition_count(),
				diagnostics.dnn_applied_partition_count(),
				diagnostics.dnn_inherited_partition_count(),
				diagnostics.dnn_fallback_partition_count(),
				diagnostics.dnn_unexpected_fallback_count(),
			),
		);
		for (index, reason) in self.plan.dnn_fallback_reasons().enumerate() {
			if index != 0 {
				output.push_str(", ");
			}
			crate::core::push_json_string(&mut output, reason);
		}
		output.push_str("]\n");
		output.push_str("  },\n  \"lowering_analysis\": {\n");
		crate::core::push_format(
			&mut output,
			format_args!(
				"    \"operation_count\": {},\n    \"schema_owned_node_count\": {},\n    \"compatibility_node_count\": {},\n    \"direct_operation_count\": {},\n    \"decomposed_operation_count\": {},\n    \"fused_operation_count\": {},\n    \"fused_node_count\": {},\n    \"maximum_nodes_per_operation\": {},\n    \"maximum_operations_per_node\": {},\n    \"operations\": [",
				diagnostics.semantic_operation_count(),
				lowering.schema_owned_node_count(),
				lowering.compatibility_node_count(),
				lowering.direct_op_count(),
				lowering.decomposed_op_count(),
				lowering.fused_op_count(),
				lowering.fused_node_count(),
				lowering.maximum_nodes_per_op(),
				lowering.maximum_ops_per_node(),
			),
		);
		for (index, operation) in self.plan.semantic_graph().operations().iter().enumerate() {
			output.push_str(if index == 0 { "\n" } else { ",\n" });
			crate::core::push_format(
				&mut output,
				format_args!(
					"      {{\"operation\": {}, \"executable_node_count\": {}}}",
					operation.id().index(),
					lowering.executable_node_count(operation.id())
				),
			);
		}
		if !self.plan.semantic_graph().operations().is_empty() {
			output.push('\n');
		}
		output.push_str("    ]\n  },\n  \"memory_analysis\": {\n");
		crate::core::push_format(
			&mut output,
			format_args!(
				"    \"resource_count\": {},\n    \"candidate_count\": {},\n    \"materialization_eligible_count\": {},\n    \"alias_group_count\": {},\n    \"potential_savings_bytes\": {},\n    \"materialized_savings_bytes\": {},\n    \"materialized\": {},\n    \"fallback_reason\": ",
				diagnostics.captured_resource_count(),
				diagnostics.alias_candidate_count(),
				diagnostics.alias_candidate_count(),
				diagnostics.planned_alias_group_count(),
				diagnostics.potential_alias_savings(),
				diagnostics.materialized_alias_savings(),
				if diagnostics.materialized_alias_savings() == 0 {
					"false"
				} else {
					"true"
				},
			),
		);
		crate::core::push_json_string(
			&mut output,
			self.plan
				.alias_materialization_fallback_reason()
				.unwrap_or(""),
		);
		output.push_str(",\n    \"resources\": [");
		for (index, resource) in resources.iter().enumerate() {
			output.push_str(if index == 0 { "\n" } else { ",\n" });
			crate::core::push_format(
				&mut output,
				format_args!(
					"      {{\"resource\": {}, \"bytes\": {}, \"placement\": \"host_upload\", \"has_lifetime\": true, \"first_access\": {}, \"last_access\": {}, \"semantic_external\": {}, \"stable_replay_input\": {}, \"stable_transient\": {}, \"observed_output\": {}, \"alias_candidate\": {}, \"alias_materialized\": {}}}",
					resource.resource(),
					resource.byte_len(),
					resource.first_access(),
					resource.last_access(),
					resource.semantic_external(),
					resource.stable_replay_input(),
					resource.stable_transient(),
					resource.observed_output(),
					resource.alias_candidate(),
					resource.alias_materialized(),
				),
			);
		}
		if !resources.is_empty() {
			output.push('\n');
		}
		output.push_str("    ]\n  }\n}\n");
		output
	}

	/// Return immutable stage-by-stage evidence from the capture transaction.
	///
	/// Explicit capture records commands lazily, so its command-recording stage is
	/// `NotRun`. Automatic [`crate::ml::TrainingLoop`] capture precompiles the
	/// reusable command before committing its preserved source recording.
	pub const fn compilation_stages(&self) -> &[TrainingCompilationStageRecord] {
		&self.compilation_stages
	}
}

fn compilation_stage_records(
	diagnostics: ExecutionPlanDiagnostics,
) -> [TrainingCompilationStageRecord; 11] {
	use TrainingCompilationStage as Stage;
	use TrainingCompilationState as State;

	let semantic = diagnostics.semantic_operation_count();
	let nodes = diagnostics.node_count();
	let memory_state = if diagnostics.materialized_alias_savings() == 0 {
		State::Analyzed
	} else {
		State::Applied
	};
	let memory_output = if diagnostics.materialized_alias_savings() == 0 {
		diagnostics.alias_candidate_count()
	} else {
		diagnostics.alias_materialized_count()
	};
	let command_state = if diagnostics.command_recording_count() == 0 {
		State::NotRun
	} else {
		State::Applied
	};
	[
		record(
			Stage::SemanticValidation,
			State::Applied,
			semantic,
			semantic,
		),
		record(Stage::ReplaySafety, State::Applied, nodes, nodes),
		record(
			Stage::Decomposition,
			State::Analyzed,
			semantic,
			diagnostics.schema_owned_node_count(),
		),
		record(
			Stage::Fusion,
			State::Analyzed,
			semantic,
			diagnostics.dnn_partition_count(),
		),
		record(Stage::Placement, State::Inherited, semantic, nodes),
		record(Stage::Precision, State::Inherited, semantic, nodes),
		record(Stage::KernelSelection, State::Inherited, semantic, nodes),
		record(Stage::LoweringValidation, State::Applied, semantic, nodes),
		record(
			Stage::MemoryPlanning,
			memory_state,
			diagnostics.captured_resource_count(),
			memory_output,
		),
		record(
			Stage::SynchronizationPlanning,
			State::Applied,
			nodes,
			diagnostics.barrier_count(),
		),
		record(Stage::CommandRecording, command_state, nodes, nodes),
	]
}

const fn record(
	stage: TrainingCompilationStage,
	state: TrainingCompilationState,
	input_count: usize,
	output_count: usize,
) -> TrainingCompilationStageRecord {
	TrainingCompilationStageRecord {
		stage,
		state,
		input_count,
		output_count,
	}
}
