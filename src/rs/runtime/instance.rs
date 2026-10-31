use std::{ffi::CString, sync::Arc};

use crate::{Error, Result};

const MIN_API_VERSION: u32 = vk::API_VERSION_1_2;

#[derive(Clone)]
pub(in crate::runtime) struct Instance {
	inner: Arc<InstanceInner>,
}

struct InstanceInner {
	// vk::Instance retains the loaded library and resolved dispatch internally.
	// The slot stays Some for the life of every Instance clone; Drop vacates it
	// only after the final Arc release to satisfy the consuming destroy contract.
	handle: Option<vk::Instance>,
	surface_enabled: bool,
}

impl Instance {
	pub(in crate::runtime) fn new(required_extensions: &[String]) -> Result<Self> {
		// SAFETY: `Entry` owns the loaded Vulkan library; the created `Instance`
		// retains a shared library reference and its own resolved dispatch, so no
		// separate loader object is stored here.
		let entry =
			unsafe { vk::Entry::new() }.map_err(|source| Error::backend_unavailable("Vulkan", source))?;

		let available_api_version = entry.enumerate_instance_version().map_err(|code| {
			Error::backend_dispatch_failure("Vulkan", "instance-version query", Ok(code))
		})?;

		if available_api_version < MIN_API_VERSION {
			return Err(Error::unsupported_backend_version(
				"Vulkan",
				"1.2",
				format_api_version(available_api_version),
			));
		}

		let app_info = vk::ApplicationInfo::default()
			.application_name(c"oa")
			.application_version(vk::make_api_version(0, 0, 1, 0))
			.engine_name(c"oa")
			.engine_version(vk::make_api_version(0, 0, 1, 0))
			.api_version(available_api_version.min(vk::API_VERSION_1_3));
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
		let create_info = vk::InstanceCreateInfo::default()
			.application_info(&app_info)
			.enabled_extension_names(&extension_pointers)
			.map_err(|source| Error::backend_failure("Vulkan", "instance create-info", source))?;

		// SAFETY: the create information references only values alive for this call.
		// No allocation callbacks are installed, so destruction uses the same `None`.
		let handle = unsafe { vk::Instance::new(&entry, &create_info, None) }
			.map_err(|source| Error::backend_failure("Vulkan", "instance creation", source))?;

		let surface_enabled = required_extensions
			.iter()
			.any(|name| name.as_str() == vk::KHR_SURFACE_EXTENSION_NAME.to_str().unwrap_or_default());

		Ok(Self {
			inner: Arc::new(InstanceInner {
				handle: Some(handle),
				surface_enabled,
			}),
		})
	}

	pub(in crate::runtime) fn raw(&self) -> &vk::Instance {
		// The handle outlives every Instance clone; Drop vacates it only after
		// the last Arc release, which no live clone can observe.
		self
			.inner
			.handle
			.as_ref()
			.expect("instance handle outlives its owners")
	}

	pub(in crate::runtime) fn surface_enabled(&self) -> bool {
		self.inner.surface_enabled
	}
}

impl Drop for InstanceInner {
	fn drop(&mut self) {
		// SAFETY: the final `Arc` uniquely owns the instance. Device owners retain an
		// `Instance` clone, so no logical device can remain at this point. Taking the
		// handle satisfies vk::Instance's consuming destruction contract.
		if let Some(instance) = self.handle.take() {
			unsafe {
				instance.destroy(None);
			}
		}
	}
}

fn format_api_version(version: u32) -> String {
	format!(
		"{}.{}.{}",
		vk::api_version_major(version),
		vk::api_version_minor(version),
		vk::api_version_patch(version)
	)
}
