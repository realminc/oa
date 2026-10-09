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
}

pub use info::{DeviceHardwareInfo, DeviceInfo, DeviceKind, DeviceSoftwareInfo, DeviceSummary};
