//! Thread-safe live control and bounded observation for [`super::ItTraining`].
//!
//! The public [`TrainingSession`] is a cloneable command and observation handle.
//! The training iterator remains on its owning engine thread and applies queued
//! commands only at explicit safe points.

use std::{
	collections::VecDeque,
	sync::{Arc, Condvar, Mutex, MutexGuard},
	time::Duration,
};

use crate::{Error, ErrorKind, Result};

use super::{Optimizer, TrainingSnapshot};

type CommandHandler<'a> = Box<dyn FnMut() -> Result<()> + 'a>;
type RebuildHandler<'a> = Box<dyn FnMut(&TrainingCommand) -> Result<()> + 'a>;
type ValueGetter<'a> = Box<dyn Fn() -> TrainingValue + 'a>;
type ValueSetter<'a> = Box<dyn FnMut(&TrainingValue) -> Result<()> + 'a>;

/// Live state of an attached training lifecycle.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TrainingState {
	/// A new step may begin at the next safe point.
	Running,
	/// New steps remain blocked until a resume or stop command is applied.
	Paused,
	/// A stop was requested and no new step may begin.
	Stopping,
	/// Training reached its explicit successful terminal boundary.
	Completed,
	/// Training reached an explicit failure boundary.
	Failed,
}

impl TrainingState {
	pub(crate) const fn is_terminal(self) -> bool {
		matches!(self, Self::Completed | Self::Failed)
	}
}

/// Kind of command applied at a training safe point.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TrainingCommandKind {
	/// Pause before another step begins.
	Pause,
	/// Resume a paused lifecycle.
	Resume,
	/// Stop before another step begins.
	Stop,
	/// Invoke the attached checkpoint handler.
	Checkpoint,
	/// Invoke the attached evaluation handler.
	Evaluate,
	/// Change one registered parameter.
	SetParameter,
	/// Discard the captured program so the next automatic step recaptures it.
	RequestRecapture,
	/// Invoke the attached rebuild handler.
	RequestRebuild,
}

/// Whether a processed command changed the training lifecycle.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TrainingCommandDisposition {
	/// The command passed validation and was applied.
	Applied,
	/// The command was observed but rejected without mutation.
	Rejected,
}

/// Safe-point class of a live training parameter.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TrainingParameterClass {
	/// May change at any batch safe point.
	Hot,
	/// May change only while paused and requires explicit recapture policy.
	Recapture,
	/// May change only while paused and requires explicit rebuild policy.
	Rebuild,
	/// May be observed but never changed through the session.
	Immutable,
}

/// Discriminant of a [`TrainingValue`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TrainingValueKind {
	/// No payload.
	Empty,
	/// Boolean payload.
	Boolean,
	/// Signed integer payload.
	Integer,
	/// Floating-point payload.
	Float,
	/// UTF-8 string payload.
	String,
}

/// Typed value carried by a command or exposed parameter.
#[derive(Clone, Debug, Default, PartialEq)]
pub enum TrainingValue {
	/// No payload.
	#[default]
	Empty,
	/// Boolean payload.
	Boolean(bool),
	/// Signed integer payload.
	Integer(i64),
	/// Floating-point payload.
	Float(f64),
	/// UTF-8 string payload.
	String(String),
}

impl TrainingValue {
	/// Return this value's stable discriminant.
	pub const fn kind(&self) -> TrainingValueKind {
		match self {
			Self::Empty => TrainingValueKind::Empty,
			Self::Boolean(_) => TrainingValueKind::Boolean,
			Self::Integer(_) => TrainingValueKind::Integer,
			Self::Float(_) => TrainingValueKind::Float,
			Self::String(_) => TrainingValueKind::String,
		}
	}

	/// Convert an integer or float payload to a common numeric representation.
	pub fn as_number(&self) -> Option<f64> {
		match self {
			Self::Integer(value) => Some(*value as f64),
			Self::Float(value) => Some(*value),
			Self::Empty | Self::Boolean(_) | Self::String(_) => None,
		}
	}
}

/// One command submitted by a session client.
#[derive(Clone, Debug, PartialEq)]
pub struct TrainingCommand {
	/// Session-assigned sequence. Input values are overwritten by [`TrainingSession::enqueue`].
	pub sequence: u64,
	/// Required live revision, or zero to accept the current revision.
	pub expected_revision: u64,
	/// Requested action.
	pub kind: TrainingCommandKind,
	/// Parameter name for [`TrainingCommandKind::SetParameter`].
	pub parameter: String,
	/// Typed parameter or rebuild payload.
	pub value: TrainingValue,
}

impl TrainingCommand {
	fn new(kind: TrainingCommandKind, expected_revision: u64) -> Self {
		Self {
			sequence: 0,
			expected_revision,
			kind,
			parameter: String::new(),
			value: TrainingValue::Empty,
		}
	}
}

/// Cloneable failure evidence retained in a command result.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TrainingCommandError {
	kind: ErrorKind,
	message: String,
}

impl TrainingCommandError {
	/// Return the stable error category.
	pub const fn kind(&self) -> ErrorKind {
		self.kind
	}

	/// Return the contextual rejection message.
	pub fn message(&self) -> &str {
		&self.message
	}

	fn new(kind: ErrorKind, message: impl Into<String>) -> Self {
		Self {
			kind,
			message: message.into(),
		}
	}

	fn from_error(error: &Error) -> Self {
		Self::new(error.kind(), error.message())
	}
}

/// Durable result of one processed command.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TrainingCommandResult {
	/// Processed command sequence.
	pub sequence: u64,
	/// Revision after processing.
	pub revision: u64,
	/// Whether the command was applied.
	pub disposition: TrainingCommandDisposition,
	/// Live state after processing.
	pub state: TrainingState,
	/// Rejection evidence, or `None` for an applied command.
	pub error: Option<TrainingCommandError>,
}

/// One named scalar published with a completed-step snapshot.
#[derive(Clone, Debug, PartialEq)]
pub struct TrainingMetricSample {
	/// Stable metric name.
	pub name: String,
	/// Finite scalar value.
	pub value: f64,
	/// Completed training step associated with this sample.
	pub step: u64,
}

/// Bounded live-control snapshot published by an attached iterator.
#[derive(Clone, Debug, PartialEq)]
pub struct TrainingSessionSnapshot {
	/// Session revision represented by this snapshot.
	pub revision: u64,
	/// Lifecycle state represented by this snapshot.
	pub state: TrainingState,
	/// Completed training steps.
	pub step: u64,
	/// Current one-based epoch, or zero when epochs are disabled.
	pub epoch: u64,
	/// Learning rate used by the next optimizer step.
	pub learning_rate: f32,
	/// Most recently completed loss, or zero before a loss is available.
	pub loss: f32,
	/// Most recent device duration in milliseconds, or zero when untimed.
	pub gpu_ms: f64,
	/// Mean wall-clock milliseconds per completed step.
	pub wall_ms: f64,
	/// Current named scalar metrics.
	pub metrics: Vec<TrainingMetricSample>,
}

/// Bounded-history capacities for a [`TrainingSession`].
#[derive(Clone, Copy, Debug)]
pub struct TrainingSessionConfig {
	/// Maximum queued commands before non-blocking enqueue rejects work.
	pub command_capacity: usize,
	/// Maximum retained command results.
	pub result_capacity: usize,
	/// Maximum retained lifecycle snapshots.
	pub snapshot_capacity: usize,
}

impl Default for TrainingSessionConfig {
	fn default() -> Self {
		Self {
			command_capacity: 64,
			result_capacity: 128,
			snapshot_capacity: 256,
		}
	}
}

/// Engine-thread handlers invoked for application-owned commands.
#[derive(Default)]
pub struct TrainingSessionHandlers<'a> {
	checkpoint: Option<CommandHandler<'a>>,
	evaluate: Option<CommandHandler<'a>>,
	rebuild: Option<RebuildHandler<'a>>,
}

impl<'a> TrainingSessionHandlers<'a> {
	/// Install the checkpoint handler.
	pub fn checkpoint(mut self, handler: impl FnMut() -> Result<()> + 'a) -> Self {
		self.checkpoint = Some(Box::new(handler));
		self
	}

	/// Install the evaluation handler.
	pub fn evaluate(mut self, handler: impl FnMut() -> Result<()> + 'a) -> Self {
		self.evaluate = Some(Box::new(handler));
		self
	}

	/// Install the rebuild handler.
	pub fn rebuild(mut self, handler: impl FnMut(&TrainingCommand) -> Result<()> + 'a) -> Self {
		self.rebuild = Some(Box::new(handler));
		self
	}
}

/// Engine-thread descriptor for one session-visible parameter.
pub struct TrainingParameterDesc<'a> {
	name: String,
	parameter_class: TrainingParameterClass,
	kind: TrainingValueKind,
	minimum: Option<f64>,
	maximum: Option<f64>,
	get: ValueGetter<'a>,
	set: Option<ValueSetter<'a>>,
}

impl<'a> TrainingParameterDesc<'a> {
	/// Describe a mutable parameter and its engine-thread accessors.
	pub fn mutable(
		name: impl Into<String>,
		parameter_class: TrainingParameterClass,
		kind: TrainingValueKind,
		get: impl Fn() -> TrainingValue + 'a,
		set: impl FnMut(&TrainingValue) -> Result<()> + 'a,
	) -> Self {
		Self {
			name: name.into(),
			parameter_class,
			kind,
			minimum: None,
			maximum: None,
			get: Box::new(get),
			set: Some(Box::new(set)),
		}
	}

	/// Describe an immutable observable parameter.
	pub fn immutable(
		name: impl Into<String>,
		kind: TrainingValueKind,
		get: impl Fn() -> TrainingValue + 'a,
	) -> Self {
		Self {
			name: name.into(),
			parameter_class: TrainingParameterClass::Immutable,
			kind,
			minimum: None,
			maximum: None,
			get: Box::new(get),
			set: None,
		}
	}

	/// Declare an inclusive numeric range.
	pub fn numeric_range(mut self, minimum: f64, maximum: f64) -> Self {
		self.minimum = Some(minimum);
		self.maximum = Some(maximum);
		self
	}
}

struct Shared {
	state: Mutex<SessionState>,
	wake: Condvar,
}

struct SessionState {
	config: TrainingSessionConfig,
	state: TrainingState,
	revision: u64,
	next_sequence: u64,
	take_sequence: u64,
	attached: bool,
	commands: VecDeque<TrainingCommand>,
	results: VecDeque<TrainingCommandResult>,
	snapshots: VecDeque<TrainingSessionSnapshot>,
	parameters: Vec<(String, TrainingValue)>,
	pending_metrics: Vec<TrainingMetricSample>,
}

/// Cloneable, thread-safe command and observation handle for one training loop.
///
/// Attach it to exactly one [`super::ItTraining`] instance. Clones may enqueue
/// commands and read bounded evidence from UI, viewer, or service threads.
#[derive(Clone)]
pub struct TrainingSession {
	shared: Arc<Shared>,
}

impl TrainingSession {
	/// Construct an unattached live-control handle.
	pub fn new(mut config: TrainingSessionConfig) -> Self {
		config.command_capacity = config.command_capacity.max(1);
		config.result_capacity = config.result_capacity.max(1);
		config.snapshot_capacity = config.snapshot_capacity.max(1);
		Self {
			shared: Arc::new(Shared {
				state: Mutex::new(SessionState {
					config,
					state: TrainingState::Running,
					revision: 0,
					next_sequence: 1,
					take_sequence: 0,
					attached: false,
					commands: VecDeque::new(),
					results: VecDeque::new(),
					snapshots: VecDeque::new(),
					parameters: Vec::new(),
					pending_metrics: Vec::new(),
				}),
				wake: Condvar::new(),
			}),
		}
	}

	/// Enqueue one command without blocking.
	///
	/// # Errors
	///
	/// Returns [`ErrorKind::ResourceExhausted`] when the bounded queue is full.
	pub fn enqueue(&self, mut command: TrainingCommand) -> Result<u64> {
		let mut state = self.lock();
		if state.commands.len() >= state.config.command_capacity {
			return Err(Error::resource_exhausted(
				"training session command queue is full",
			));
		}
		let sequence = state.next_sequence;
		state.next_sequence = state
			.next_sequence
			.checked_add(1)
			.ok_or_else(|| Error::resource_exhausted("training command sequence exhausted"))?;
		command.sequence = sequence;
		state.commands.push_back(command);
		drop(state);
		self.shared.wake.notify_all();
		Ok(sequence)
	}

	/// Enqueue a pause request.
	pub fn pause(&self, expected_revision: u64) -> Result<u64> {
		self.enqueue(TrainingCommand::new(
			TrainingCommandKind::Pause,
			expected_revision,
		))
	}

	/// Enqueue a resume request.
	pub fn resume(&self, expected_revision: u64) -> Result<u64> {
		self.enqueue(TrainingCommand::new(
			TrainingCommandKind::Resume,
			expected_revision,
		))
	}

	/// Enqueue a stop request.
	pub fn stop(&self, expected_revision: u64) -> Result<u64> {
		self.enqueue(TrainingCommand::new(
			TrainingCommandKind::Stop,
			expected_revision,
		))
	}

	/// Enqueue a checkpoint request.
	pub fn checkpoint(&self, expected_revision: u64) -> Result<u64> {
		self.enqueue(TrainingCommand::new(
			TrainingCommandKind::Checkpoint,
			expected_revision,
		))
	}

	/// Enqueue an evaluation request.
	pub fn evaluate(&self, expected_revision: u64) -> Result<u64> {
		self.enqueue(TrainingCommand::new(
			TrainingCommandKind::Evaluate,
			expected_revision,
		))
	}

	/// Enqueue a typed parameter update.
	pub fn set_parameter(
		&self,
		name: impl Into<String>,
		value: TrainingValue,
		expected_revision: u64,
	) -> Result<u64> {
		let mut command = TrainingCommand::new(TrainingCommandKind::SetParameter, expected_revision);
		command.parameter = name.into();
		command.value = value;
		self.enqueue(command)
	}

	/// Enqueue a captured-program recapture request.
	pub fn request_recapture(&self, expected_revision: u64) -> Result<u64> {
		self.enqueue(TrainingCommand::new(
			TrainingCommandKind::RequestRecapture,
			expected_revision,
		))
	}

	/// Enqueue an application-owned rebuild request.
	pub fn request_rebuild(&self, value: TrainingValue, expected_revision: u64) -> Result<u64> {
		let mut command = TrainingCommand::new(TrainingCommandKind::RequestRebuild, expected_revision);
		command.value = value;
		self.enqueue(command)
	}

	/// Publish or replace a named finite scalar for subsequent snapshots.
	pub fn publish_metric(&self, name: impl Into<String>, value: f64) {
		let name = name.into();
		if name.is_empty() || !value.is_finite() {
			return;
		}
		let mut state = self.lock();
		if let Some(metric) = state
			.pending_metrics
			.iter_mut()
			.find(|metric| metric.name == name)
		{
			metric.value = value;
		} else {
			state.pending_metrics.push(TrainingMetricSample {
				name,
				value,
				step: 0,
			});
		}
	}

	/// Return the most recently cached value of a registered parameter.
	pub fn parameter(&self, name: &str) -> Option<TrainingValue> {
		self
			.lock()
			.parameters
			.iter()
			.find_map(|(candidate, value)| (candidate == name).then(|| value.clone()))
	}

	/// Return the current live state.
	pub fn state(&self) -> TrainingState {
		self.lock().state
	}

	/// Return the current optimistic-concurrency revision.
	pub fn revision(&self) -> u64 {
		self.lock().revision
	}

	/// Combine the live state and revision with the latest published metrics.
	pub fn current_snapshot(&self) -> TrainingSessionSnapshot {
		let state = self.lock();
		let mut snapshot = state
			.snapshots
			.back()
			.cloned()
			.unwrap_or_else(|| TrainingSessionSnapshot {
				revision: state.revision,
				state: state.state,
				step: 0,
				epoch: 0,
				learning_rate: 0.0,
				loss: 0.0,
				gpu_ms: 0.0,
				wall_ms: 0.0,
				metrics: Vec::new(),
			});
		snapshot.revision = state.revision;
		snapshot.state = state.state;
		snapshot
	}

	/// Return the latest published lifecycle snapshot.
	pub fn latest_snapshot(&self) -> Option<TrainingSessionSnapshot> {
		self.lock().snapshots.back().cloned()
	}

	/// Return retained results newer than an observer-owned sequence cursor.
	pub fn results_after(&self, sequence: u64) -> Vec<TrainingCommandResult> {
		self
			.lock()
			.results
			.iter()
			.filter(|result| result.sequence > sequence)
			.cloned()
			.collect()
	}

	/// Return results not previously consumed through this convenience cursor.
	///
	/// This does not delete retained results and therefore cannot steal evidence
	/// from independent observers using [`TrainingSession::results_after`].
	pub fn take_results(&self) -> Vec<TrainingCommandResult> {
		let mut state = self.lock();
		let results = state
			.results
			.iter()
			.filter(|result| result.sequence > state.take_sequence)
			.cloned()
			.collect::<Vec<_>>();
		if let Some(result) = results.last() {
			state.take_sequence = result.sequence;
		}
		results
	}

	fn lock(&self) -> MutexGuard<'_, SessionState> {
		self
			.shared
			.state
			.lock()
			.unwrap_or_else(std::sync::PoisonError::into_inner)
	}

	pub(crate) fn attach(&self, snapshot: TrainingSessionSnapshot) -> Result<()> {
		let mut state = self.lock();
		if state.attached {
			return Err(Error::already_exists(
				"training session is already attached to an iterator",
			));
		}
		state.attached = true;
		state.state = TrainingState::Running;
		state.revision = state
			.revision
			.checked_add(1)
			.ok_or_else(|| Error::resource_exhausted("training session revision exhausted"))?;
		let mut snapshot = snapshot;
		snapshot.revision = state.revision;
		snapshot.state = state.state;
		let capacity = state.config.snapshot_capacity;
		push_bounded(&mut state.snapshots, snapshot, capacity);
		drop(state);
		self.shared.wake.notify_all();
		Ok(())
	}

	pub(crate) fn drain_commands(&self) -> Vec<TrainingCommand> {
		self.lock().commands.drain(..).collect()
	}

	pub(crate) fn command_context(&self) -> (TrainingState, u64) {
		let state = self.lock();
		(state.state, state.revision)
	}

	pub(crate) fn record_result(
		&self,
		command: &TrainingCommand,
		transition: Option<TrainingState>,
		error: Option<TrainingCommandError>,
	) {
		let mut state = self.lock();
		if error.is_none() {
			if let Some(next) = transition {
				state.state = next;
			}
			state.revision = state.revision.saturating_add(1);
		}
		let result = TrainingCommandResult {
			sequence: command.sequence,
			revision: state.revision,
			disposition: if error.is_none() {
				TrainingCommandDisposition::Applied
			} else {
				TrainingCommandDisposition::Rejected
			},
			state: state.state,
			error,
		};
		let capacity = state.config.result_capacity;
		push_bounded(&mut state.results, result, capacity);
		drop(state);
		self.shared.wake.notify_all();
	}

	pub(crate) fn cache_parameter(&self, name: String, value: TrainingValue) -> Result<()> {
		let mut state = self.lock();
		if state
			.parameters
			.iter()
			.any(|(candidate, _)| candidate == &name)
		{
			return Err(Error::already_exists(format!(
				"training session parameter already exists: {name}"
			)));
		}
		state.parameters.push((name, value));
		Ok(())
	}

	pub(crate) fn update_parameter(&self, name: &str, value: TrainingValue) {
		if let Some((_, cached)) = self
			.lock()
			.parameters
			.iter_mut()
			.find(|(candidate, _)| candidate == name)
		{
			*cached = value;
		}
	}

	pub(crate) fn publish(
		&self,
		snapshot: TrainingSnapshot,
		learning_rate: f32,
		terminal: Option<TrainingState>,
	) {
		let mut state = self.lock();
		if let Some(terminal) = terminal {
			state.state = terminal;
			state.revision = state.revision.saturating_add(1);
		}
		let metrics = state
			.pending_metrics
			.iter()
			.cloned()
			.map(|mut metric| {
				metric.step = snapshot.step_count();
				metric
			})
			.collect();
		let live = TrainingSessionSnapshot {
			revision: state.revision,
			state: state.state,
			step: snapshot.step_count(),
			epoch: snapshot.epoch(),
			learning_rate,
			loss: snapshot.last_loss().unwrap_or(0.0),
			gpu_ms: snapshot
				.last_gpu_time()
				.unwrap_or(Duration::ZERO)
				.as_secs_f64()
				* 1000.0,
			wall_ms: snapshot.wall_ms_per_step(),
			metrics,
		};
		let capacity = state.config.snapshot_capacity;
		push_bounded(&mut state.snapshots, live, capacity);
		drop(state);
		self.shared.wake.notify_all();
	}

	pub(crate) fn wait_while_paused(&self) {
		let mut state = self.lock();
		while state.state == TrainingState::Paused && state.commands.is_empty() {
			state = self
				.shared
				.wake
				.wait(state)
				.unwrap_or_else(std::sync::PoisonError::into_inner);
		}
	}
}

impl Default for TrainingSession {
	fn default() -> Self {
		Self::new(TrainingSessionConfig::default())
	}
}

pub(crate) struct TrainingSessionAttachment<'a> {
	pub(crate) session: TrainingSession,
	handlers: TrainingSessionHandlers<'a>,
	parameters: Vec<TrainingParameterDesc<'a>>,
}

#[derive(Default)]
pub(crate) struct SessionActions {
	pub(crate) stop: bool,
	pub(crate) recapture: bool,
}

impl<'a> TrainingSessionAttachment<'a> {
	pub(crate) fn new(session: TrainingSession, handlers: TrainingSessionHandlers<'a>) -> Self {
		Self {
			session,
			handlers,
			parameters: Vec::new(),
		}
	}

	pub(crate) fn register_parameter(&mut self, desc: TrainingParameterDesc<'a>) -> Result<()> {
		if desc.name.is_empty() || desc.minimum.zip(desc.maximum).is_some_and(|(a, b)| a > b) {
			return Err(Error::invalid_argument(
				"training parameter requires a name and an ordered numeric range",
			));
		}
		if desc.parameter_class != TrainingParameterClass::Immutable && desc.set.is_none() {
			return Err(Error::invalid_argument(
				"mutable training parameter requires a setter",
			));
		}
		let value = (desc.get)();
		if value.kind() != desc.kind {
			return Err(Error::invalid_argument(
				"training parameter getter kind does not match its declaration",
			));
		}
		self.session.cache_parameter(desc.name.clone(), value)?;
		self.parameters.push(desc);
		Ok(())
	}

	pub(crate) fn poll(
		&mut self,
		optimizer: &mut dyn Optimizer,
		can_recapture: bool,
	) -> SessionActions {
		let mut actions = SessionActions::default();
		for command in self.session.drain_commands() {
			let (state, revision) = self.session.command_context();
			let mut transition = None;
			let error = if command.expected_revision != 0 && command.expected_revision != revision {
				Some(TrainingCommandError::new(
					ErrorKind::Aborted,
					"training session command revision is stale",
				))
			} else {
				match command.kind {
					TrainingCommandKind::Pause if state != TrainingState::Running => {
						Some(TrainingCommandError::new(
							ErrorKind::FailedPrecondition,
							"pause requires a running training session",
						))
					}
					TrainingCommandKind::Pause => {
						transition = Some(TrainingState::Paused);
						None
					}
					TrainingCommandKind::Resume if state != TrainingState::Paused => {
						Some(TrainingCommandError::new(
							ErrorKind::FailedPrecondition,
							"resume requires a paused training session",
						))
					}
					TrainingCommandKind::Resume => {
						transition = Some(TrainingState::Running);
						None
					}
					TrainingCommandKind::Stop if state.is_terminal() => Some(TrainingCommandError::new(
						ErrorKind::FailedPrecondition,
						"stop requires an active training session",
					)),
					TrainingCommandKind::Stop => {
						actions.stop = true;
						transition = Some(TrainingState::Stopping);
						None
					}
					TrainingCommandKind::Checkpoint => {
						invoke_handler(self.handlers.checkpoint.as_mut(), "checkpoint")
					}
					TrainingCommandKind::Evaluate => {
						invoke_handler(self.handlers.evaluate.as_mut(), "evaluation")
					}
					TrainingCommandKind::SetParameter => self.set_parameter(&command, state, optimizer),
					TrainingCommandKind::RequestRecapture if state != TrainingState::Paused => {
						Some(TrainingCommandError::new(
							ErrorKind::FailedPrecondition,
							"program recapture requires a paused session",
						))
					}
					TrainingCommandKind::RequestRecapture if !can_recapture => {
						Some(TrainingCommandError::new(
							ErrorKind::FailedPrecondition,
							"training program recapture requires automatic capture",
						))
					}
					TrainingCommandKind::RequestRecapture => {
						actions.recapture = true;
						None
					}
					TrainingCommandKind::RequestRebuild if state != TrainingState::Paused => {
						Some(TrainingCommandError::new(
							ErrorKind::FailedPrecondition,
							"training rebuild requires a paused session",
						))
					}
					TrainingCommandKind::RequestRebuild => match self.handlers.rebuild.as_mut() {
						Some(handler) => handler(&command)
							.err()
							.as_ref()
							.map(TrainingCommandError::from_error),
						None => Some(missing_handler("rebuild")),
					},
				}
			};
			self.session.record_result(&command, transition, error);
		}
		actions
	}

	fn set_parameter(
		&mut self,
		command: &TrainingCommand,
		state: TrainingState,
		optimizer: &mut dyn Optimizer,
	) -> Option<TrainingCommandError> {
		if command.parameter == "learning_rate" {
			let Some(value) = command.value.as_number() else {
				return Some(kind_mismatch());
			};
			let result = if !value.is_finite() || value.abs() > f64::from(f32::MAX) {
				Err(Error::out_of_range("learning rate is outside f32 range"))
			} else {
				optimizer.set_learning_rate(value as f32)
			};
			return match result {
				Ok(()) => {
					self.session.update_parameter(
						"learning_rate",
						TrainingValue::Float(f64::from(optimizer.learning_rate())),
					);
					None
				}
				Err(error) => Some(TrainingCommandError::from_error(&error)),
			};
		}
		let Some(parameter) = self
			.parameters
			.iter_mut()
			.find(|parameter| parameter.name == command.parameter)
		else {
			return Some(TrainingCommandError::new(
				ErrorKind::NotFound,
				format!("unknown training parameter: {}", command.parameter),
			));
		};
		if parameter.parameter_class == TrainingParameterClass::Immutable {
			return Some(TrainingCommandError::new(
				ErrorKind::FailedPrecondition,
				format!("training parameter is immutable: {}", command.parameter),
			));
		}
		if parameter.parameter_class != TrainingParameterClass::Hot && state != TrainingState::Paused {
			return Some(TrainingCommandError::new(
				ErrorKind::FailedPrecondition,
				"recapture/rebuild parameters require a paused session",
			));
		}
		if command.value.kind() != parameter.kind
			&& !(parameter.kind == TrainingValueKind::Float
				&& command.value.kind() == TrainingValueKind::Integer)
		{
			return Some(kind_mismatch());
		}
		if let Some(number) = command.value.as_number()
			&& (parameter.minimum.is_some_and(|minimum| number < minimum)
				|| parameter.maximum.is_some_and(|maximum| number > maximum))
		{
			return Some(TrainingCommandError::new(
				ErrorKind::OutOfRange,
				"training parameter value is outside its declared range",
			));
		}
		let Some(set) = parameter.set.as_mut() else {
			return Some(TrainingCommandError::new(
				ErrorKind::FailedPrecondition,
				"training parameter does not admit mutation",
			));
		};
		if let Err(error) = set(&command.value) {
			return Some(TrainingCommandError::from_error(&error));
		}
		self
			.session
			.update_parameter(&parameter.name, (parameter.get)());
		None
	}
}

fn invoke_handler(
	handler: Option<&mut CommandHandler<'_>>,
	name: &'static str,
) -> Option<TrainingCommandError> {
	match handler {
		Some(handler) => handler()
			.err()
			.as_ref()
			.map(TrainingCommandError::from_error),
		None => Some(missing_handler(name)),
	}
}

fn missing_handler(name: &'static str) -> TrainingCommandError {
	TrainingCommandError::new(
		ErrorKind::FailedPrecondition,
		format!("training session has no {name} handler"),
	)
}

fn kind_mismatch() -> TrainingCommandError {
	TrainingCommandError::new(
		ErrorKind::InvalidArgument,
		"training parameter value kind does not match its declaration",
	)
}

fn push_bounded<T>(queue: &mut VecDeque<T>, value: T, capacity: usize) {
	while queue.len() >= capacity {
		queue.pop_front();
	}
	queue.push_back(value);
}
