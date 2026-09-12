//! Training iteration, captured programs, metrics, and callbacks.
//!
//! `ItTraining` owns the mutable step lifecycle. `TrainingProgram` is the
//! immutable captured executable it may create and replay; they deliberately
//! remain separate types under one subsystem authority.

pub(super) use super::{AdamW, CheckpointOptimizer, Optimizer};

mod callbacks;
mod dqn;
mod iterator;
mod ppo;
mod program;
mod rollout;
mod sac;
mod schedule;
mod session;
mod target;

pub use callbacks::{
	CbCheckpoint, CbCsvLogger, CbEarlyStop, CbLrScheduler, CbPhase, CbProgressBar, CbSummary,
	CbValidation, Checkpoint, CsvLogger, EarlyStopMode, EarlyStopping, LearningRateScheduler,
	PhaseSchedule, ProgressBar, TrainingPhase, TrainingSummary, Validation, ValidationMetric,
	ValidationResult,
};

pub use dqn::{DqnTrainer, DqnTrainerConfig, DqnTrainerMetrics};
pub use sac::{SacTrainer, SacTrainerConfig, SacTrainerMetrics};

pub use iterator::{
	GpuTimingStats, ItTraining, ItTrainingConfig, LossAggregation, LossMetric, TrainingCallback,
	TrainingCallbackContext, TrainingControl, TrainingLoop, TrainingLoopConfig, TrainingMetric,
	TrainingSnapshot,
};
pub use ppo::{PpoTrainer, PpoTrainerConfig, PpoTrainerMetrics};
pub use rollout::{ItRolloutTraining, ItRolloutTrainingConfig, RolloutTrainingPhase};

pub use program::{
	TrainingCompilationStage, TrainingCompilationStageRecord, TrainingCompilationState,
	TrainingProgram,
};

pub use schedule::{
	CosineScheduler, CosineWarmRestartsScheduler, CyclicMode, CyclicScheduler,
	LinearWarmupCosineScheduler, LrScheduler, OneCycleScheduler, PlateauMode,
	ReduceOnPlateauScheduler, SequentialScheduler, WarmupScheduler,
};
pub use session::{
	TrainingCommand, TrainingCommandDisposition, TrainingCommandError, TrainingCommandKind,
	TrainingCommandResult, TrainingMetricSample, TrainingParameterClass, TrainingParameterDesc,
	TrainingSession, TrainingSessionConfig, TrainingSessionHandlers, TrainingSessionSnapshot,
	TrainingState, TrainingValue, TrainingValueKind,
};

pub(crate) use session::TrainingSessionAttachment;
