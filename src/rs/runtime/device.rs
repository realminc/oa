//! One retained device domain and its queried information.

pub(in crate::runtime) mod buffer;
pub(in crate::runtime) mod command;
pub(in crate::runtime) mod commands;
pub(in crate::runtime) mod descriptor;
pub(in crate::runtime) mod dispatch;
pub(in crate::runtime) mod image;
pub(in crate::runtime) mod pipeline;
#[cfg_attr(
	not(test),
	expect(
		dead_code,
		reason = "private ML-KEM qualification route; public secret operation admission remains planned"
	)
)]
pub(in crate::runtime) mod pqc;

pub(in crate::runtime) mod retirement;
#[cfg_attr(
	not(test),
	expect(
		dead_code,
		reason = "private PQC storage substrate; secret algorithms are not admitted"
	)
)]
pub(in crate::runtime) mod secret_buffer;
pub(in crate::runtime) mod swapchain;
pub(in crate::runtime) mod timeline;
pub(in crate::runtime) mod timestamp;
pub(in crate::runtime) mod video;

mod configuration;
pub(in crate::runtime) mod features;
pub(super) mod info;
pub(super) mod logical;
pub(in crate::runtime) mod physical;

use std::sync::Arc;

/// Bitwise membership test for generated `vk::*FlagBits*` newtypes.
///
/// The generated bindings expose `BitAnd`/`BitOr` but no named membership
/// helper; this restores the conventional `flags.contains(bit)` spelling for
/// private native code.
pub(in crate::runtime) trait NativeFlagBits:
	Copy + PartialEq + std::ops::BitAnd<Output = Self>
{
	fn contains(self, member: Self) -> bool {
		(self & member) == member
	}
}

macro_rules! impl_native_flag_bits {
	($($ty:ty),* $(,)?) => {
		$(impl NativeFlagBits for $ty {})*
	};
}

impl_native_flag_bits! {
	vk::FormatFeatureFlagBits,
	vk::ImageCreateFlagBits,
	vk::ImageUsageFlagBits,
	vk::QueueFlagBits,
	vk::ShaderStageFlagBits,
	vk::CompositeAlphaFlagBitsKHR,
	vk::SampleCountFlagBits,
	vk::VideoCapabilityFlagBitsKHR,
	vk::VideoChromaSubsamplingFlagBitsKHR,
	vk::VideoCodingControlFlagBitsKHR,
	vk::VideoComponentBitDepthFlagBitsKHR,
	vk::VideoDecodeCapabilityFlagBitsKHR,
	vk::VideoDecodeH264PictureLayoutFlagBitsKHR,
	vk::VideoEncodeFeedbackFlagBitsKHR,
	vk::VideoEncodeH265CtbSizeFlagBitsKHR,
	vk::VideoEncodeH265TransformBlockSizeFlagBitsKHR,
	vk::VideoEncodeRateControlModeFlagBitsKHR,
	vk::AccessFlagBits2,
	vk::PipelineStageFlagBits2,
	vk::BufferUsageFlagBits,
	vk::QueryResultFlagBits,
}

use configuration::DeviceConfiguration;
use logical::DeviceLogical;
pub(in crate::runtime) use physical::DevicePhysical;

/// Engine-private execution domain, retained by resources through `Arc<Device>`.
/// Queried facts and executor resources each have one authoritative owner.
/// Owned executor fields never retain this Arc, preventing ownership cycles.
pub(in crate::runtime) struct Device {
	// Drop executor resources before releasing their owning Vulkan instance.
	logical: DeviceLogical,
	physical: DevicePhysical,
	configuration: DeviceConfiguration,
	instance: super::instance::Instance,
}

impl Device {
	pub(in crate::runtime) fn new(
		instance: &super::instance::Instance,
		physical: DevicePhysical,
	) -> crate::Result<Arc<Self>> {
		let configuration = DeviceConfiguration::new(instance, &physical);
		let logical = DeviceLogical::new(instance, &physical, &configuration)?;
		Ok(Arc::new(Self {
			logical,
			physical,
			configuration,
			instance: instance.clone(),
		}))
	}

	pub(in crate::runtime) fn logical(&self) -> &DeviceLogical {
		&self.logical
	}

	pub(in crate::runtime) fn compute_shared_memory_bytes(&self) -> u32 {
		self.physical.limits.max_compute_shared_memory_size
	}

	/// Flatten linear work across two queried dispatch dimensions.
	pub(in crate::runtime) fn linear_dispatch_grid(
		&self,
		count: u32,
		group_size: u32,
	) -> crate::Result<[u32; 3]> {
		let limits = self.physical.limits;
		if group_size == 0
			|| group_size > limits.max_compute_work_group_size[0]
			|| group_size > limits.max_compute_work_group_invocations
		{
			return Err(crate::Error::invalid_argument(
				"linear dispatch group size exceeds device limits",
			));
		}
		let groups = count.div_ceil(group_size).max(1);
		let x = groups.min(limits.max_compute_work_group_count[0]);
		if x == 0 || limits.max_compute_work_group_count[1] == 0 {
			return Err(crate::Error::missing_capability(
				"device has no usable compute dispatch grid",
			));
		}
		let y = groups.div_ceil(x);
		if y > limits.max_compute_work_group_count[1] {
			return Err(crate::Error::invalid_argument(
				"linear work exceeds queried compute dispatch grid",
			));
		}
		Ok([x, y, 1])
	}
}

pub use info::{DeviceHardwareInfo, DeviceInfo, DeviceKind, DeviceSoftwareInfo, DeviceSummary};
