//! Engine construction, execution ownership, storage, and completion.

mod dispatch;
mod engine;
mod event;
mod executable_graph;
mod log;
mod plan;
mod session;
mod shader;
mod storage;
mod vk;

pub(crate) use dispatch::{BufferAccess, BufferBinding, ComputeDispatch, PushConstant};
pub(crate) use engine::EngineHandle;
pub use engine::{DeviceSelection, Engine, EngineBuilder};
pub use event::Event;
#[doc(hidden)]
pub use log::{__log_should_write, __log_write};
pub use log::{LogComponent, LogLevel, LogOptions};
pub use plan::ExecutionPlan;
pub(crate) use shader::KernelId;
pub(crate) use storage::Storage;
