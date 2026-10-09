//! ComposeImage — R8G8B8A8_UNORM storage image registered in the bindless heap.
//!
//! Created through `EngineHandle::create_compose_image`.  Owns the bindless
//! slot; releases it on drop.
//!
//! Donor reference: `Viewer` compose-image management in `renderBackend.cpp`

use std::cell::Cell;

use crate::Result;

/// Private compose storage image for the UI layer.
pub(super) struct ComposeImage {
	/// Index in the engine's bindless STORAGE_IMAGE heap (binding=1).
	pub(super) bindless_index: u32,
	pub(super) width: u32,
	pub(super) height: u32,
	engine_handle: crate::runtime::EngineHandle,
	/// Raw Vulkan image handle retained for barrier commands.
	raw_image: ash::vk::Image,
	layout: Cell<ash::vk::ImageLayout>,
}

impl ComposeImage {
	pub(super) fn new(
		engine: &crate::runtime::EngineHandle,
		width: u32,
		height: u32,
	) -> Result<Self> {
		let (raw_image, bindless_index) = engine.create_compose_image(width, height)?;
		Ok(Self {
			bindless_index,
			width,
			height,
			engine_handle: engine.clone(),
			raw_image,
			layout: Cell::new(ash::vk::ImageLayout::UNDEFINED),
		})
	}

	pub(super) const fn raw_image(&self) -> ash::vk::Image {
		self.raw_image
	}

	pub(super) fn layout(&self) -> ash::vk::ImageLayout {
		self.layout.get()
	}

	pub(super) fn set_layout(&self, layout: ash::vk::ImageLayout) {
		self.layout.set(layout);
	}
}

impl Drop for ComposeImage {
	fn drop(&mut self) {
		self
			.engine_handle
			.destroy_compose_image(self.raw_image, self.bindless_index);
	}
}
