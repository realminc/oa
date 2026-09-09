//! Machine-learning operations, differentiation, layers, and optimizers.

mod activation;
mod attention;
mod autograd;
mod checkpoint;
mod kernels;
pub mod loss;
mod module;
pub mod nlp;
pub mod nn;
mod optimizer;
mod parameter;
mod random;
mod training_loop;
mod training_program;

pub use activation::{gelu, swiglu};
pub use attention::scaled_dot_product_attention_causal;
pub use autograd::GradientTape;
pub use checkpoint::{load_checkpoint, save_checkpoint};
pub use module::{Module, ModuleRegistry, NamedBuffer, NamedParameter, ScopedEval};
pub use optimizer::AdamW;
pub use parameter::Parameter;
pub use training_loop::{
	LossAggregation, LossMetric, TrainingCallback, TrainingCallbackContext, TrainingControl,
	TrainingLoop, TrainingLoopConfig, TrainingMetric, TrainingSnapshot,
};
pub use training_program::{
	TrainingCompilationStage, TrainingCompilationStageRecord, TrainingCompilationState,
	TrainingProgram,
};
