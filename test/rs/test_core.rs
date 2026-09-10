//! External contracts for the public core module.

#[macro_use]
#[path = "support/test_vk.rs"]
mod test_vk_fixture;

#[path = "core/test_api.rs"]
mod api;
#[path = "core/test_image.rs"]
mod image;
#[path = "core/test_memory.rs"]
mod memory;
#[path = "core/test_vlm.rs"]
mod vlm;
