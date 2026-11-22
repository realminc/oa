//! ComposeImage — R8G8B8A8_UNORM color-attachment image for the UI raster pass.
//!
//! Created through `EngineHandle::create_compose_image`. Owns the image and
//! its retained view; releases both through the engine registry on drop.

use std::cell::Cell;

use crate::Result;

/// Private compose color-attachment image for the UI layer.
pub(super) struct ComposeImage {
	pub(super) width: u32,
	pub(super) height: u32,
	engine_handle: crate::runtime::EngineHandle,
	/// Raw Vulkan image handle retained for barrier commands.
	raw_image: vk::Image,
	/// Color-attachment view retained for dynamic rendering.
	raw_view: vk::ImageView,
	layout: Cell<vk::ImageLayout>,
}

impl ComposeImage {
	pub(super) fn new(
		engine: &crate::runtime::EngineHandle,
		width: u32,
		height: u32,
	) -> Result<Self> {
		let (raw_image, raw_view) = engine.create_compose_image(width, height)?;
		Ok(Self {
			width,
			height,
			engine_handle: engine.clone(),
			raw_image,
			raw_view,
			layout: Cell::new(vk::ImageLayout::UNDEFINED),
		})
	}

	pub(super) const fn raw_image(&self) -> vk::Image {
		self.raw_image
	}

	pub(super) const fn raw_view(&self) -> vk::ImageView {
		self.raw_view
	}

	pub(super) fn layout(&self) -> vk::ImageLayout {
		self.layout.get()
	}

	pub(super) fn set_layout(&self, layout: vk::ImageLayout) {
		self.layout.set(layout);
	}
}

impl Drop for ComposeImage {
	fn drop(&mut self) {
		self.engine_handle.destroy_compose_image(self.raw_image);
	}
}
