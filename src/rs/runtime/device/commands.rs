use crate::runtime::Instance;

/// The enabled synchronization2 command spelling for this logical device.
/// Vulkan 1.2 must call KHR entry points even when the structures are aliases
/// of their Vulkan 1.3 counterparts.
pub(super) enum SyncCommands {
	Core,
	Khr(ash::khr::synchronization2::Device),
}

/// The enabled dynamic-rendering command spelling. A Vulkan 1.2 graphics
/// device may use the KHR extension without exposing core 1.3 entry points.
pub(super) enum RenderingCommands {
	Core,
	Khr(ash::khr::dynamic_rendering::Device),
}

impl RenderingCommands {
	pub(super) fn new(instance: &Instance, device: &ash::Device, api_version: u32) -> Self {
		if api_version >= ash::vk::API_VERSION_1_3 {
			Self::Core
		} else {
			Self::Khr(ash::khr::dynamic_rendering::Device::new(
				instance.raw(),
				device,
			))
		}
	}

	/// # Safety
	/// The command buffer must be recording outside a render pass, and the
	/// attachment resources and layouts must satisfy Vulkan dynamic rendering.
	pub(super) unsafe fn begin(
		&self,
		device: &ash::Device,
		command: ash::vk::CommandBuffer,
		info: &ash::vk::RenderingInfo<'_>,
	) {
		unsafe {
			match self {
				Self::Core => device.cmd_begin_rendering(command, info),
				Self::Khr(khr) => khr.cmd_begin_rendering(command, info),
			}
		}
	}

	/// # Safety
	/// The command buffer must be recording inside a dynamic-rendering scope.
	pub(super) unsafe fn end(&self, device: &ash::Device, command: ash::vk::CommandBuffer) {
		unsafe {
			match self {
				Self::Core => device.cmd_end_rendering(command),
				Self::Khr(khr) => khr.cmd_end_rendering(command),
			}
		}
	}
}

impl SyncCommands {
	pub(super) fn new(instance: &Instance, device: &ash::Device, api_version: u32) -> Self {
		if api_version >= ash::vk::API_VERSION_1_3 {
			Self::Core
		} else {
			Self::Khr(ash::khr::synchronization2::Device::new(
				instance.raw(),
				device,
			))
		}
	}

	/// # Safety
	/// The caller must provide a recording command buffer and valid dependency
	/// scopes/resources owned by this device.
	pub(super) unsafe fn pipeline_barrier(
		&self,
		device: &ash::Device,
		command: ash::vk::CommandBuffer,
		dependency: &ash::vk::DependencyInfo<'_>,
	) {
		unsafe {
			match self {
				Self::Core => device.cmd_pipeline_barrier2(command, dependency),
				Self::Khr(khr) => khr.cmd_pipeline_barrier2(command, dependency),
			}
		}
	}

	/// # Safety
	/// The queue, command buffers, semaphores and fence must be valid for this
	/// device and obey the Vulkan submission synchronization rules.
	pub(super) unsafe fn submit(
		&self,
		device: &ash::Device,
		queue: ash::vk::Queue,
		submits: &[ash::vk::SubmitInfo2<'_>],
		fence: ash::vk::Fence,
	) -> std::result::Result<(), ash::vk::Result> {
		unsafe {
			match self {
				Self::Core => device.queue_submit2(queue, submits, fence),
				Self::Khr(khr) => khr.queue_submit2(queue, submits, fence),
			}
		}
	}

	/// # Safety
	/// The query pool and index must be valid and the command buffer recording.
	pub(super) unsafe fn write_timestamp(
		&self,
		device: &ash::Device,
		command: ash::vk::CommandBuffer,
		stage: ash::vk::PipelineStageFlags2,
		pool: ash::vk::QueryPool,
		query: u32,
	) {
		unsafe {
			match self {
				Self::Core => device.cmd_write_timestamp2(command, stage, pool, query),
				Self::Khr(khr) => khr.cmd_write_timestamp2(command, stage, pool, query),
			}
		}
	}
}

#[cfg(test)]
#[path = "../../../../test/rs/runtime/commands_unit.rs"]
mod tests;
