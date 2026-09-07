use ash::vk;

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("failed to load Vulkan: {0}")]
    VulkanLoad(#[from] ash::LoadingError),

    #[error("Vulkan error: {0:?}")]
    Vulkan(#[from] vk::Result),

    #[error("no suitable Vulkan device")]
    NoSuitableDevice,

    #[error("required Vulkan capability is unavailable: {0}")]
    MissingCapability(&'static str),
}

pub type Result<T> = std::result::Result<T, Error>;
