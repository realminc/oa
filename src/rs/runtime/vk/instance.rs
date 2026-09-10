use std::sync::Arc;

use crate::{Error, Result};

const MIN_API_VERSION: u32 = ash::vk::API_VERSION_1_3;

#[derive(Clone)]
pub(in crate::runtime) struct Instance {
	inner: Arc<InstanceInner>,
}

struct InstanceInner {
	// The Vulkan library must remain loaded until after the instance is destroyed.
	entry: ash::Entry,
	handle: ash::Instance,
}

impl Instance {
	pub(in crate::runtime) fn new() -> Result<Self> {
		// SAFETY: `Entry` owns the loaded Vulkan library, remains alive in
		// `InstanceInner`, and therefore outlives every resolved function pointer.
		let entry = unsafe { ash::Entry::load() }
			.map_err(|source| Error::backend_unavailable("Vulkan", source))?;

		// SAFETY: the loaded entry owns a valid `vkGetInstanceProcAddr`. This call has
		// no application-provided pointers and only queries the loader's API version.
		let available_api_version = unsafe { entry.try_enumerate_instance_version() }
			.map_err(|source| Error::backend_failure("Vulkan", "instance-version query", source))?
			.unwrap_or(ash::vk::API_VERSION_1_0);

		if available_api_version < MIN_API_VERSION {
			return Err(Error::unsupported_backend_version(
				"Vulkan",
				"1.3",
				format_api_version(available_api_version),
			));
		}

		let app_info = ash::vk::ApplicationInfo::default()
			.application_name(c"oa")
			.application_version(ash::vk::make_api_version(0, 0, 1, 0))
			.engine_name(c"oa")
			.engine_version(ash::vk::make_api_version(0, 0, 1, 0))
			.api_version(MIN_API_VERSION);
		let create_info = ash::vk::InstanceCreateInfo::default().application_info(&app_info);

		// SAFETY: the create information references only values alive for this call.
		// No allocation callbacks are installed, so destruction uses the same `None`.
		let handle = unsafe { entry.create_instance(&create_info, None) }
			.map_err(|source| Error::backend_failure("Vulkan", "instance creation", source))?;

		Ok(Self {
			inner: Arc::new(InstanceInner { entry, handle }),
		})
	}

	pub(super) fn raw(&self) -> &ash::Instance {
		&self.inner.handle
	}

	pub(super) fn entry(&self) -> &ash::Entry {
		&self.inner.entry
	}
}

impl Drop for InstanceInner {
	fn drop(&mut self) {
		// SAFETY: the final `Arc` uniquely owns the instance. Device owners retain an
		// `Instance` clone, so no logical device can remain at this point.
		unsafe {
			self.handle.destroy_instance(None);
		}
	}
}

fn format_api_version(version: u32) -> String {
	format!(
		"{}.{}.{}",
		ash::vk::api_version_major(version),
		ash::vk::api_version_minor(version),
		ash::vk::api_version_patch(version)
	)
}

#[cfg(test)]
mod tests {
	use super::format_api_version;

	#[test]
	fn formats_vulkan_api_version() {
		assert_eq!(
			format_api_version(ash::vk::make_api_version(0, 1, 3, 281)),
			"1.3.281"
		);
	}
}
