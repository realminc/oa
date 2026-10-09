//! Engine construction, execution ownership, storage, and completion.

mod device;
mod dispatch;
mod dnn;
mod engine;
mod event;
mod executable_graph;
mod instance;
mod log;
mod plan;
mod presentation;
mod presenter;
mod semantic_graph;
mod session;
mod shader;
mod storage;
mod window;

pub(crate) use device::dispatch::{BufferAccess, BufferBinding, ComputeDispatch, PushConstant};
pub use device::{DeviceHardwareInfo, DeviceInfo, DeviceKind, DeviceSoftwareInfo, DeviceSummary};
pub(crate) use dispatch::{
	AudioSemanticDispatch, AudioSemanticOutput, ImageSemanticDispatch, ImageSemanticInput,
	OptionalSemanticDispatch, SemanticDispatch,
};
pub(crate) use engine::{
	CaptureAttempt, EngineHandle, RecordingTransaction, SemanticLoweringScope, VideoDecoderBackend,
};
pub use engine::{DeviceSelection, Engine, EngineBuilder, EngineRequirements};
pub use event::Event;
#[doc(hidden)]
pub use log::{__log_should_write, __log_write};
pub use log::{LogComponent, LogLevel, LogOptions};
pub use plan::{
	CapturedResourceDesc, ExecutionPlan, ExecutionPlanDiagnostics, SemanticStorageBinding,
};
pub use presenter::Presenter;
pub use semantic_graph::{
	SemanticAccessMode, SemanticAliasDesc, SemanticAutogradDesc, SemanticGraph,
	SemanticLoweringAnalysis, SemanticOpDesc, SemanticOpId, SemanticValueAccess, SemanticValueDesc,
	SemanticValueId,
};
pub(crate) use shader::KernelId;
pub(crate) use storage::Storage;

pub use window::SdlWindow;

pub(in crate::runtime) use device::buffer::Buffer;
pub(crate) use device::buffer::MeshBuffer;
pub(in crate::runtime) use device::command::{RecordedCommandBuffer, ReusableCommandBuffer};
pub(crate) use device::logical::SceneDrawCmd;
pub(in crate::runtime) use device::retirement::{RetirementService, RetirementTicket};
pub(in crate::runtime) use device::secret_buffer::{SecretBinding, SecretBuffer, SecretErasure};
pub(in crate::runtime) use device::swapchain::PresentationSwapchain;
pub(in crate::runtime) use device::timestamp::TimestampPair;
pub(in crate::runtime) use device::video::DecodeSession;
pub(crate) use device::video::NativeDecodedFrame;
pub(in crate::runtime) use device::{Device, DevicePhysical};
pub(in crate::runtime) use instance::Instance;
pub use presentation::{PresentationCapabilities, PresentationSurface};

pub(crate) use device::pipeline::{
	FlatColorPipeline, GraphicsPipeline, StandardSurfaceBlendPipeline,
	StandardSurfaceDescriptorArena, StandardSurfaceDescriptorLayouts, StandardSurfacePipeline,
	UnlitDescriptorArena, UnlitDescriptorLayouts, UnlitPipeline,
};

pub(crate) use device::image::{NativeRgbaImage, RenderTarget};

#[cfg(test)]
#[path = "../../test/rs/runtime/bounded_add_unit.rs"]
mod bounded_add_unit;

#[cfg(test)]
#[path = "../../test/rs/runtime/compatibility_engine_unit.rs"]
mod compatibility_engine_unit;
