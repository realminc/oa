//! Machine-learning operations, differentiation, layers, and optimizers.

mod actor_critic;
pub mod advantage;
mod autograd;
mod checkpoint;
mod collector;
pub mod environment;
pub mod evaluation;
pub mod flow;
pub mod loss;
pub(crate) mod lowering;
pub mod matrix;
pub mod metric;
mod model_file;
mod module;
pub mod nlp;
pub mod nn;
pub mod optim;
mod optimizer;
mod parameter;
pub mod policy;
mod random;
pub mod replay;
pub mod rollout;
pub mod training;

pub use actor_critic::{
	ActorCritic, ActorCriticOutput, CategoricalActorCritic, CategoricalActorCriticConfig,
};
pub use autograd::GradientTape;
pub use checkpoint::{
	CheckpointManager, CheckpointManagerConfig, load_checkpoint, save_checkpoint,
};
pub use collector::{RolloutCollector, RolloutCollectorConfig, RolloutCollectorMetrics};
pub use environment::{
	EnvironmentSpace, EnvironmentSpaceKind, EnvironmentSpec, EnvironmentTransition,
};
pub use matrix::UpsampleMode;
pub use module::{Module, ModuleRegistry, NamedBuffer, NamedParameter, ScopedEval};
pub use optimizer::{Adam, AdamW, CheckpointOptimizer, Muon, NoOpOptimizer, Optimizer, Sgd};
pub use parameter::Parameter;
pub use policy::{ContinuousPolicyResult, PolicyResult};
pub use replay::{ReplayBatch, ReplayBuffer, ReplayConfig, ReplayTransition};
pub use rollout::{RolloutBatch, RolloutBuffer, RolloutConfig, RolloutTransition};
pub use training::*;
