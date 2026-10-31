use crate::runtime::Instance;

/// The enabled synchronization2 command spelling for this logical device.
/// Vulkan 1.2 must call KHR entry points even when the structures are aliases
/// of their Vulkan 1.3 counterparts.
#[derive(Clone)]
pub(crate) enum SyncCommands {
	Core,
	Khr,
}

/// The enabled dynamic-rendering command spelling. A Vulkan 1.2 graphics
/// device may use the KHR extension without exposing core 1.3 entry points.
pub(super) enum RenderingCommands {
	Core,
	Khr,
}

impl RenderingCommands {
	pub(super) fn new(_instance: &Instance, device: &vk::Device, api_version: u32) -> Self {
		let _ = device;
		if api_version >= vk::API_VERSION_1_3 {
			Self::Core
		} else {
			Self::Khr
		}
	}

	/// # Safety
	/// The command buffer must be recording outside a render pass, and the
	/// attachment resources and layouts must satisfy Vulkan dynamic rendering.
	pub(super) unsafe fn begin(
		&self,
		device: &vk::Device,
		command: vk::CommandBuffer,
		info: &vk::RenderingInfo<'_>,
	) -> std::result::Result<(), vk::CallError> {
		unsafe {
			match self {
				Self::Core => device.cmd_begin_rendering(command, info),
				Self::Khr => device.cmd_begin_rendering_khr(command, info),
			}
		}
	}

	/// # Safety
	/// The command buffer must be recording inside a dynamic-rendering scope.
	pub(super) unsafe fn end(
		&self,
		device: &vk::Device,
		command: vk::CommandBuffer,
	) -> std::result::Result<(), vk::CallError> {
		unsafe {
			match self {
				Self::Core => device.cmd_end_rendering(command),
				Self::Khr => device.cmd_end_rendering_khr(command),
			}
		}
	}
}

impl SyncCommands {
	pub(super) fn new(_instance: &Instance, device: &vk::Device, api_version: u32) -> Self {
		let _ = device;
		if api_version >= vk::API_VERSION_1_3 {
			Self::Core
		} else {
			Self::Khr
		}
	}

	/// # Safety
	/// The caller must provide a recording command buffer and valid dependency
	/// scopes/resources owned by this device.
	pub(crate) unsafe fn pipeline_barrier(
		&self,
		device: &vk::Device,
		command: vk::CommandBuffer,
		dependency: &vk::DependencyInfo<'_>,
	) -> std::result::Result<(), vk::CallError> {
		unsafe {
			match self {
				Self::Core => device.cmd_pipeline_barrier2(command, dependency),
				Self::Khr => device.cmd_pipeline_barrier2_khr(command, dependency),
			}
		}
	}

	/// # Safety
	/// The queue, command buffers, semaphores and fence must be valid for this
	/// device and obey the Vulkan submission synchronization rules.
	pub(super) unsafe fn submit(
		&self,
		device: &vk::Device,
		queue: vk::Queue,
		submits: &[vk::SubmitInfo2<'_>],
		fence: vk::Fence,
	) -> std::result::Result<(), vk::CallError> {
		unsafe {
			match self {
				Self::Core => device.queue_submit2(queue, submits, fence),
				Self::Khr => device.queue_submit2_khr(queue, submits, fence),
			}
		}
	}

	/// # Safety
	/// The query pool and index must be valid and the command buffer recording.
	pub(super) unsafe fn write_timestamp(
		&self,
		device: &vk::Device,
		command: vk::CommandBuffer,
		stage: vk::PipelineStageFlags2,
		pool: vk::QueryPool,
		query: u32,
	) -> std::result::Result<(), vk::CallError> {
		unsafe {
			match self {
				Self::Core => device.cmd_write_timestamp2(command, stage, pool, query),
				Self::Khr => device.cmd_write_timestamp2_khr(command, stage, pool, query),
			}
		}
	}
}

#[cfg(test)]
#[path = "../../../../test/rs/runtime/commands_unit.rs"]
mod tests;
