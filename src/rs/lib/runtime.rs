//! Runtime construction, execution ownership, storage, and completion.

mod clock_calibration;
mod device;
mod dispatch;
mod dna;
#[doc(hidden)]
pub use dna::fusion::{EpilogueFusionDisabled, SingleHeadSdpaFusionEnabled};
mod event;
mod graph;
mod instance;
mod log;
mod plan;
mod presentation;
mod presenter;
#[allow(clippy::module_inception)]
mod runtime;
mod session;
mod storage;
mod timer;
mod window;

pub use clock_calibration::{ClockCalibration, HostClockDomain};
pub(crate) use device::logical::commands::{RenderingCommands, SyncCommands};
pub(crate) use device::logical::dispatch::{
	BufferAccess, BufferBinding, ComputeDispatch, IndirectDispatch, PushConstant,
};
pub use device::{DeviceKind, DeviceSummary};
pub(crate) use dispatch::{
	AudioSemanticDispatch, AudioSemanticInput, AudioSemanticOutput, ImageSemanticDispatch,
	ImageSemanticInput, ImageSemanticOutput, OptionalSemanticDispatch, QuantSemanticDispatch,
	QuantSemanticInput, QuantSemanticOutput, SemanticDispatch,
};
pub(crate) use dna::kernel::KernelId;
pub(in crate::runtime) use dna::provider::blas::{
	GemmEpilogue, MatmulPreference, MatmulProblem, lower_matmul_nt_plan, plan_matmul_nt_problem,
};
pub(crate) use dna::provider::blas::{lower_matmul_nt, plan_matmul_nt, plan_matmul_nt_lowered};
pub(crate) use dna::provider::precision::{
	record_linear_f16, record_matmul_nt_f16, record_sum_f16,
};
pub use event::Event;
pub use graph::semantic::{
	SemanticAccessMode, SemanticAliasDesc, SemanticAutogradDesc, SemanticGraph,
	SemanticLoweringAnalysis, SemanticOpDesc, SemanticOpId, SemanticValueAccess, SemanticValueDesc,
	SemanticValueId,
};
#[doc(hidden)]
pub use log::{__log_should_write, __log_write};
pub use log::{LogComponent, LogLevel, LogOptions};
pub use plan::{
	CapturedResourceDesc, ExecutionBindingDesc, ExecutionBindingRole, ExecutionPlan,
	ExecutionPlanDiagnostics, SemanticStorageBinding,
};
pub use presenter::Presenter;
pub(crate) use runtime::{
	CaptureAttempt, MlDsaDeviceKey, MlKemDeviceKey, MlKemSharedSecret, RecordingTransaction,
	RuntimeHandle, SecretLoweringScope, SemanticLoweringScope, VideoDecoderBackend,
	VideoEncoderBackend,
};
pub use runtime::{DeviceSelection, Runtime, RuntimeBuilder};
pub(crate) use storage::Storage;
pub use timer::{Timer, TimerDomain};
pub use types::{
	DeviceHardwareInfo, DeviceInfo, DeviceSoftwareInfo, FusionStats, PresentationCapabilities,
	RuntimeRequirements,
};

pub use window::SdlWindow;
pub(crate) use window::decoration as window_decoration;

pub(crate) use device::Device;
pub(in crate::runtime) use device::DevicePhysical;
pub(crate) use device::logical::SceneDrawCmd;
pub(in crate::runtime) use device::logical::buffer::Buffer;
pub(crate) use device::logical::buffer::MeshBuffer;
pub(in crate::runtime) use device::logical::command::{
	RecordedCommandBuffer, ReusableCommandBuffer,
};
pub(in crate::runtime) use device::logical::retirement::{
	DeferredRetirement, RetirementService, RetirementTicket,
};
pub(crate) use device::logical::secret_buffer::SecretErasure;
pub(in crate::runtime) use device::logical::secret_buffer::{SecretBinding, SecretBuffer};
pub(in crate::runtime) use device::logical::swapchain::PresentationSwapchain;
pub(in crate::runtime) use device::logical::timestamp::TimestampPair;
pub(crate) use device::logical::video::NativeDecodedFrame;
pub(in crate::runtime) use device::logical::video::{DecodeSession, EncodeSession};
pub(in crate::runtime) use instance::Instance;
pub use presentation::PresentationSurface;

pub(crate) use device::logical::pipeline::render::{
	SurfaceDescriptorArena, SurfaceDescriptorLayouts, UnlitDescriptorArena, UnlitDescriptorLayouts,
};
pub(crate) use device::logical::pipeline::ui::{UiLayouts, UiPipelines};
pub(crate) use device::logical::pipeline::{GraphicsPipeline, decode_spirv_bytes};

pub(crate) use device::logical::image::{NativeRgbaImage, RenderTarget};

#[cfg(test)]
#[path = "../../../test/rs/runtime/bounded_add_unit.rs"]
mod bounded_add_unit;

#[cfg(test)]
#[path = "../../../test/rs/runtime/compatibility_runtime_unit.rs"]
mod compatibility_runtime_unit;

#[path = "runtime/parity.gen.rs"]
mod parity;
#[path = "runtime/type.gen.rs"]
mod types;
