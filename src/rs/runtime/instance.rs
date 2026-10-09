use std::{ffi::CString, sync::Arc};

use crate::{Error, Result};

const MIN_API_VERSION: u32 = ash::vk::API_VERSION_1_2;

#[derive(Clone)]
pub(in crate::runtime) struct Instance {
	inner: Arc<InstanceInner>,
}

struct InstanceInner {
	// The Vulkan library must remain loaded until after the instance is destroyed.
	entry: ash::Entry,
	handle: ash::Instance,
	surface_enabled: bool,
}

impl Instance {
	pub(in crate::runtime) fn new(required_extensions: &[String]) -> Result<Self> {
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
				"1.2",
				format_api_version(available_api_version),
			));
		}

		let app_info = ash::vk::ApplicationInfo::default()
			.application_name(c"oa")
			.application_version(ash::vk::make_api_version(0, 0, 1, 0))
			.engine_name(c"oa")
			.engine_version(ash::vk::make_api_version(0, 0, 1, 0))
			.api_version(available_api_version.min(ash::vk::API_VERSION_1_3));
		let extension_names = required_extensions
			.iter()
			.map(|name| {
				CString::new(name.as_str()).map_err(|_| {
					Error::invalid_argument(format!(
						"Vulkan instance extension contains an interior NUL: {name:?}"
					))
				})
			})
			.collect::<Result<Vec<_>>>()?;
		let extension_pointers = extension_names
			.iter()
			.map(|name| name.as_ptr())
			.collect::<Vec<_>>();
		let create_info = ash::vk::InstanceCreateInfo::default()
			.application_info(&app_info)
			.enabled_extension_names(&extension_pointers);

		// SAFETY: the create information references only values alive for this call.
		// No allocation callbacks are installed, so destruction uses the same `None`.
		let handle = unsafe { entry.create_instance(&create_info, None) }
			.map_err(|source| Error::backend_failure("Vulkan", "instance creation", source))?;

		let surface_enabled = required_extensions
			.iter()
			.any(|name| name == ash::khr::surface::NAME.to_string_lossy().as_ref());

		Ok(Self {
			inner: Arc::new(InstanceInner {
				entry,
				handle,
				surface_enabled,
			}),
		})
	}

	pub(in crate::runtime) fn raw(&self) -> &ash::Instance {
		&self.inner.handle
	}

	pub(in crate::runtime) fn entry(&self) -> &ash::Entry {
		&self.inner.entry
	}

	pub(in crate::runtime) fn surface_enabled(&self) -> bool {
		self.inner.surface_enabled
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
