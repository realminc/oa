use ash::vk;

use crate::error::Result;

#[derive(Clone, Debug)]
pub struct PhysicalDeviceInfo {
    pub handle: vk::PhysicalDevice,
    pub name: String,
    pub vendor_id: u32,
    pub device_id: u32,
    pub device_type: vk::PhysicalDeviceType,
    pub api_version: u32,
}

pub fn enumerate_devices(instance: &ash::Instance) -> Result<Vec<PhysicalDeviceInfo>> {
    let devices = unsafe { instance.enumerate_physical_devices()? };

    let mut result = Vec::with_capacity(devices.len());

    for handle in devices {
        let props = unsafe { instance.get_physical_device_properties(handle) };

        let name = unsafe { std::ffi::CStr::from_ptr(props.device_name.as_ptr()) }
            .to_string_lossy()
            .into_owned();

        result.push(PhysicalDeviceInfo {
            handle,
            name,
            vendor_id: props.vendor_id,
            device_id: props.device_id,
            device_type: props.device_type,
            api_version: props.api_version,
        });
    }

    Ok(result)
}
