//! Engine construction, execution ownership, storage, and completion.

mod clock_calibration;
mod device;
mod dispatch;
mod dna;
mod engine;
mod event;
mod executable_graph;
mod fusion;
mod instance;
mod log;
mod plan;
mod presentation;
mod presenter;
mod semantic_graph;
mod session;
mod storage;
mod timer;
mod window;

pub use clock_calibration::{ClockCalibration, HostClockDomain};
pub(crate) use device::commands::SyncCommands;
pub(crate) use device::dispatch::{
	BufferAccess, BufferBinding, ComputeDispatch, IndirectDispatch, PushConstant,
};
pub use device::{DeviceKind, DeviceSummary};
pub(crate) use dispatch::{
	AudioSemanticDispatch, AudioSemanticOutput, ImageSemanticDispatch, ImageSemanticInput,
	OptionalSemanticDispatch, QuantSemanticDispatch, QuantSemanticInput, QuantSemanticOutput,
	SemanticDispatch,
};
pub(crate) use dna::kernel::KernelId;
pub(in crate::runtime) use dna::provider::blas::{
	GemmEpilogue, MatmulPreference, MatmulProblem, lower_mat_mul_nt_plan, plan_mat_mul_nt_problem,
};
pub(crate) use dna::provider::blas::{lower_mat_mul_nt, plan_mat_mul_nt};
pub(crate) use engine::{
	CaptureAttempt, EngineHandle, MlDsaDeviceKey, MlKemDeviceKey, MlKemSharedSecret,
	RecordingTransaction, SecretLoweringScope, SemanticLoweringScope, VideoDecoderBackend,
	VideoEncoderBackend,
};
pub use engine::{DeviceSelection, Engine, EngineBuilder};
pub use event::Event;
#[doc(hidden)]
pub use log::{__log_should_write, __log_write};
pub use log::{LogComponent, LogLevel, LogOptions};
pub use plan::{
	CapturedResourceDesc, ExecutionBindingDesc, ExecutionBindingRole, ExecutionPlan,
	ExecutionPlanDiagnostics, SemanticStorageBinding,
};
pub use presenter::Presenter;
pub use semantic_graph::{
	SemanticAccessMode, SemanticAliasDesc, SemanticAutogradDesc, SemanticGraph,
	SemanticLoweringAnalysis, SemanticOpDesc, SemanticOpId, SemanticValueAccess, SemanticValueDesc,
	SemanticValueId,
};
pub(crate) use storage::Storage;
pub use timer::{Timer, TimerDomain};
pub use types::{
	DeviceHardwareInfo, DeviceInfo, DeviceSoftwareInfo, EngineRequirements, FusionStats,
	PresentationCapabilities,
};

pub use window::SdlWindow;
pub(crate) use window::decoration as window_decoration;

pub(in crate::runtime) use device::buffer::Buffer;
pub(crate) use device::buffer::MeshBuffer;
pub(in crate::runtime) use device::command::{RecordedCommandBuffer, ReusableCommandBuffer};
pub(crate) use device::logical::SceneDrawCmd;
pub(in crate::runtime) use device::retirement::{
	DeferredRetirement, RetirementService, RetirementTicket,
};
pub(crate) use device::secret_buffer::SecretErasure;
pub(in crate::runtime) use device::secret_buffer::{SecretBinding, SecretBuffer};
pub(in crate::runtime) use device::swapchain::PresentationSwapchain;
pub(in crate::runtime) use device::timestamp::TimestampPair;
pub(crate) use device::video::NativeDecodedFrame;
pub(in crate::runtime) use device::video::{DecodeSession, EncodeSession};
pub(in crate::runtime) use device::{Device, DevicePhysical};
pub(in crate::runtime) use instance::Instance;
pub use presentation::PresentationSurface;

pub(crate) use device::pipeline::{
	CanvasGridPipeline, FlatColorPipeline, GraphicsPipeline, SurfaceBlendPipeline,
	SurfaceDescriptorArena, SurfaceDescriptorLayouts, SurfacePipeline, UnlitDescriptorArena,
	UnlitDescriptorLayouts, UnlitPipeline,
};

pub(crate) use device::image::{NativeRgbaImage, RenderTarget};

#[cfg(test)]
#[path = "../../test/rs/runtime/bounded_add_unit.rs"]
mod bounded_add_unit;

#[cfg(test)]
#[path = "../../test/rs/runtime/compatibility_engine_unit.rs"]
mod compatibility_engine_unit;

#[path = "runtime/parity.gen.rs"]
mod parity;
#[path = "runtime/type.gen.rs"]
mod types;
