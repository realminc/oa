//! Machine-learning operations, differentiation, layers, and optimizers.

mod autograd;
mod checkpoint;
mod environment;
pub mod loss;
mod lowering;
pub mod matrix;
pub mod metric;
mod model_file;
mod module;
pub mod nlp;
pub mod nn;
pub mod optim;
mod optimizer;
mod parameter;
mod random;
pub mod training;

pub use autograd::GradientTape;
pub use checkpoint::{
	CheckpointManager, CheckpointManagerConfig, load_checkpoint, save_checkpoint,
};
pub use environment::{
	EnvironmentSpace, EnvironmentSpaceKind, EnvironmentSpec, EnvironmentTransition,
};
pub use matrix::UpsampleMode;
pub use module::{Module, ModuleRegistry, NamedBuffer, NamedParameter, ScopedEval};
pub use optimizer::{Adam, AdamW, CheckpointOptimizer, Muon, NoOpOptimizer, Optimizer, Sgd};
pub use parameter::Parameter;
pub use training::*;
