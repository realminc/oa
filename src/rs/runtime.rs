//! Engine construction, execution ownership, storage, and completion.

mod dispatch;
mod dnn;
mod engine;
mod event;
mod executable_graph;
mod log;
mod plan;
mod semantic_graph;
mod session;
mod shader;
mod storage;
mod vk;

pub(crate) use dispatch::{
	AudioSemanticDispatch, AudioSemanticOutput, BufferAccess, BufferBinding, ComputeDispatch,
	ImageSemanticDispatch, ImageSemanticInput, OptionalSemanticDispatch, PushConstant,
	SemanticDispatch,
};
pub(crate) use engine::{
	CaptureAttempt, EngineHandle, RecordingTransaction, SemanticLoweringScope, VideoDecoderBackend,
};
pub use engine::{DeviceSelection, Engine, EngineBuilder};
pub use event::Event;
#[doc(hidden)]
pub use log::{__log_should_write, __log_write};
pub use log::{LogComponent, LogLevel, LogOptions};
pub use plan::{
	CapturedResourceDesc, ExecutionPlan, ExecutionPlanDiagnostics, SemanticStorageBinding,
};
pub use semantic_graph::{
	SemanticAccessMode, SemanticAliasDesc, SemanticAutogradDesc, SemanticGraph,
	SemanticLoweringAnalysis, SemanticOpDesc, SemanticOpId, SemanticValueAccess, SemanticValueDesc,
	SemanticValueId,
};
pub(crate) use shader::KernelId;
pub(crate) use storage::Storage;
pub(crate) use vk::NativeDecodedFrame;
