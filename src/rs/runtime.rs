use ash::{Entry, Instance, vk as ash_vk};

use crate::error::Result;

/// Sole owner of OA's local Vulkan execution services.
pub struct Engine {
    _entry: Entry,
    instance: Instance,
}

impl Engine {
    /// Load Vulkan and create the engine's instance.
    ///
    /// Device selection and logical-device construction are not implemented by
    /// the current structural prototype.
    pub fn new() -> Result<Self> {
        // SAFETY: `Entry` retains the loaded Vulkan library for the lifetime of all
        // functions resolved through it, and the returned owner is stored in `Engine`.
        let entry = unsafe { Entry::load()? };

        let app_name = c"oa";
        let engine_name = c"oa";

        let app_info = ash_vk::ApplicationInfo::default()
            .application_name(app_name)
            .application_version(ash_vk::make_api_version(0, 0, 1, 0))
            .engine_name(engine_name)
            .engine_version(ash_vk::make_api_version(0, 0, 1, 0))
            .api_version(ash_vk::make_api_version(0, 1, 3, 0));

        let create_info = ash_vk::InstanceCreateInfo::default().application_info(&app_info);

        // SAFETY: `create_info` contains only default/null optional pointers and
        // references static C strings that remain valid for the duration of the call.
        let instance = unsafe { entry.create_instance(&create_info, None)? };

        Ok(Self {
            _entry: entry,
            instance,
        })
    }
}

impl Drop for Engine {
    fn drop(&mut self) {
        // SAFETY: `Engine` uniquely owns the instance. The current prototype creates
        // no child Vulkan objects, so none can remain live at this destruction point.
        unsafe {
            self.instance.destroy_instance(None);
        }
    }
}
