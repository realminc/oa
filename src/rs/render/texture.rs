//! Packed buffer-backed texture values and explicit host boundaries.

use std::{
	path::Path,
	rc::Rc,
	sync::atomic::{AtomicU64, Ordering},
};

/// Raw Vulkan image and view info returned by [`Texture::render_target_image`].
type RenderTargetImageInfo = (
	ash::vk::Image,
	ash::vk::ImageView,
	u32,
	u32,
	Rc<dyn std::any::Any>,
);

#[derive(Clone, Copy)]
pub(super) struct RenderTargetLease {
	pub(super) slot: usize,
	pub(super) generation: u64,
}

use crate::{Color, DType, Engine, Error, Image, ImageFormat, ImageLayout, Matrix, Result};

/// Internal backing for a `Texture` value.
#[derive(Clone)]
enum TextureBacking {
	/// Buffer-backed: a U8 Matrix `[H, W, 4]` plus an uploaded native image.
	BufferBacked {
		data: Matrix,
		native: Rc<crate::runtime::NativeRgbaImage>,
	},
	/// Render-target-backed: a non-owning view of an already-allocated native
	/// color image (R8G8B8A8_UNORM, SHADER_READ_ONLY after draw).
	/// Readback is not available through this backing; use the owning
	/// `RenderFrame::consume_readback` instead.
	RenderTarget {
		engine_handle: crate::runtime::EngineHandle,
		owner: Rc<std::cell::RefCell<super::renderer::RendererState>>,
		lease: RenderTargetLease,
		image: ash::vk::Image,
		view: ash::vk::ImageView,
		width: u32,
		height: u32,
	},
}

/// Stable semantic identity for one texture value.
///
/// Assigned once at creation, never reused.  Used for semantic-graph lineage
/// tracking in the same way `ImageSemantic` and `MatrixSemantic` track Image and
/// Matrix values.
#[derive(Clone)]
pub(crate) struct TextureSemantic {
	pub(crate) id: u64,
}

impl TextureSemantic {
	fn new() -> crate::Result<Self> {
		Ok(Self {
			id: next_texture_value_id()?,
		})
	}
}

/// Packed RGBA8 texture value reusable by Render, Presenter, Image, and UI.
///
/// Two backings are currently admitted:
/// - **Buffer-backed**: uploaded from host bytes or a Matrix; supports `read_rgba8`.
/// - **Render-target-backed**: a non-owning alias of a `Renderer` slot's color
///   image; valid only while the originating `RenderFrame` is live.
///   Readback uses `RenderFrame::consume_readback`, not `read_rgba8`.
#[derive(Clone)]
pub struct Texture {
	backing: TextureBacking,
	semantic: Rc<TextureSemantic>,
}

impl Texture {
	/// Return the stable semantic identity for this texture value.
	///
	/// Used for semantic-graph lineage tracking and future UI/plot value
	/// identity checks.
	#[allow(dead_code)]
	pub(crate) fn value_id(&self) -> u64 {
		self.semantic.id
	}

	pub(crate) fn engine_handle(&self) -> &crate::runtime::EngineHandle {
		match &self.backing {
			TextureBacking::BufferBacked { data, .. } => data.engine_handle(),
			TextureBacking::RenderTarget { engine_handle, .. } => engine_handle,
		}
	}

	/// Return the texture width.
	pub fn width(&self) -> usize {
		match &self.backing {
			TextureBacking::BufferBacked { data, .. } => data.shape()[1],
			TextureBacking::RenderTarget { width, .. } => *width as usize,
		}
	}

	/// Return the texture height.
	pub fn height(&self) -> usize {
		match &self.backing {
			TextureBacking::BufferBacked { data, .. } => data.shape()[0],
			TextureBacking::RenderTarget { height, .. } => *height as usize,
		}
	}

	/// Return the packed byte dtype (always `U8` for RGBA8 textures).
	pub fn dtype(&self) -> DType {
		match &self.backing {
			TextureBacking::BufferBacked { data, .. } => data.dtype(),
			TextureBacking::RenderTarget { .. } => DType::U8,
		}
	}

	/// Read packed row-major RGBA8 pixels to host memory.
	///
	/// Only available for buffer-backed textures. Render-target-backed textures
	/// must use `RenderFrame::consume_readback` instead.
	///
	/// # Errors
	///
	/// Returns an error when submission, completion, mapping, or cache
	/// invalidation fails, or when called on a render-target-backed Texture.
	pub fn read_rgba8(&self) -> Result<Vec<u8>> {
		match &self.backing {
			TextureBacking::BufferBacked { data, .. } => data.read::<u8>(),
			TextureBacking::RenderTarget { .. } => Err(Error::failed_precondition(
				"render::Texture::read_rgba8 is not available for render-target-backed textures; \
				 use RenderFrame::consume_readback",
			)),
		}
	}

	pub(crate) fn native(&self) -> Option<Rc<crate::runtime::NativeRgbaImage>> {
		match &self.backing {
			TextureBacking::BufferBacked { native, .. } => Some(native.clone()),
			TextureBacking::RenderTarget { .. } => None,
		}
	}

	pub(crate) fn packed_storage(&self) -> Option<&crate::runtime::Storage> {
		match &self.backing {
			TextureBacking::BufferBacked { data, .. } => Some(data.storage()),
			TextureBacking::RenderTarget { .. } => None,
		}
	}

	pub(crate) const fn is_render_target(&self) -> bool {
		matches!(self.backing, TextureBacking::RenderTarget { .. })
	}

	pub(crate) fn mark_consumed(&self, event: &crate::Event) -> Result<()> {
		match &self.backing {
			TextureBacking::RenderTarget {
				engine_handle,
				owner,
				lease,
				..
			} => {
				if !engine_handle.owns_event(event) {
					return Err(Error::invalid_argument(
						"render target consumer event belongs to another Engine",
					));
				}
				owner
					.borrow_mut()
					.mark_texture_consumed(lease.slot, lease.generation, event)
			}
			TextureBacking::BufferBacked { .. } => Err(Error::failed_precondition(
				"consumer completion applies only to render-target-backed textures",
			)),
		}
	}

	/// Return the raw Vulkan image and view for render-target-backed presentation.
	pub(crate) fn render_target_image(&self) -> Option<RenderTargetImageInfo> {
		match &self.backing {
			TextureBacking::RenderTarget {
				owner,
				image,
				view,
				width,
				height,
				..
			} => Some((*image, *view, *width, *height, owner.clone())),
			TextureBacking::BufferBacked { .. } => None,
		}
	}

	pub(crate) fn lease_weak(&self) -> std::rc::Weak<TextureSemantic> {
		Rc::downgrade(&self.semantic)
	}

	pub(crate) fn lease_strong_count(&self) -> usize {
		Rc::strong_count(&self.semantic)
	}
}

/// Upload one exact packed row-major RGBA8 host image as a Texture.
///
/// # Errors
///
/// Returns an error for zero or overflowing dimensions, a mismatched byte
/// count, exhausted semantic identity, or device allocation/upload failure.
pub fn texture_from_rgba8(
	engine: &Engine,
	rgba: &[u8],
	width: usize,
	height: usize,
) -> Result<Texture> {
	if width == 0 || height == 0 {
		return Err(Error::invalid_argument(
			"render::texture_from_rgba8 dimensions must be non-zero",
		));
	}
	let expected = width
		.checked_mul(height)
		.and_then(|pixels| pixels.checked_mul(4))
		.ok_or_else(|| Error::out_of_range("texture byte count overflows usize"))?;
	if rgba.len() != expected {
		return Err(Error::invalid_argument(
			"render::texture_from_rgba8 requires exactly width * height * 4 bytes",
		));
	}
	let data = Matrix::from_slice(engine, [height, width, 4], rgba)?;
	texture_from_matrix(data, width, height)
}

/// Convert one packed RGBA8 HWC image into an independent Texture value.
///
/// The operation retains the texture/image semantic distinction and performs a
/// device-side deep copy. Other image layouts, channel orders, and scalar
/// representations require an explicit image conversion before this boundary.
///
/// # Errors
///
/// Returns an error unless `image` is an unbatched U8 HWC RGBA image or its
/// device copy cannot be recorded.
pub fn texture_from_image(image: &Image) -> Result<Texture> {
	if image.layout() != ImageLayout::Hwc
		|| image.format() != ImageFormat::Rgba
		|| image.dtype() != DType::U8
	{
		return Err(Error::invalid_argument(
			"render::texture_from_image requires an HWC RGBA U8 image",
		));
	}
	let data = crate::matrix::copy(image.as_matrix())?;
	let width = image.as_matrix().shape()[1];
	let height = image.as_matrix().shape()[0];
	texture_from_matrix(data, width, height)
}

/// Deep-copy a packed Texture into independent device storage.
///
/// This is the currently admitted whole-surface `FnTexture::blit` equivalent.
/// Rectangle and scaled blits remain unavailable because the C++ donor also
/// rejects them rather than silently selecting different sampling semantics.
///
/// # Errors
///
/// Returns an error when the device copy cannot be recorded.
pub fn blit(texture: &Texture) -> Result<Texture> {
	let data = match &texture.backing {
		TextureBacking::BufferBacked { data, .. } => crate::matrix::copy(data)?,
		TextureBacking::RenderTarget { .. } => {
			return Err(Error::failed_precondition(
				"render::blit is not available for render-target-backed textures",
			));
		}
	};
	texture_from_matrix(data, texture.width(), texture.height())
}

/// Return an independent packed Texture filled with one RGBA8 clear color.
///
/// The result has the source extent but does not alias source storage. This is
/// Rust's immutable-value spelling of the C++ donor's in-place clear.
///
/// # Errors
///
/// Returns an error for non-finite or out-of-range color components, exhausted
/// texture identity, or failed device allocation/upload.
pub fn clear(texture: &Texture, color: Color) -> Result<Texture> {
	let rgba = color.to_u32().to_be_bytes();
	let pixel_count = texture
		.width()
		.checked_mul(texture.height())
		.ok_or_else(|| Error::out_of_range("texture pixel count overflows usize"))?;
	let mut pixels = vec![
		0_u8;
		pixel_count.checked_mul(4).ok_or_else(|| {
			Error::out_of_range("texture clear byte count overflows usize")
		})?
	];
	for pixel in pixels.chunks_mut(4) {
		pixel.copy_from_slice(&rgba);
	}
	texture_from_rgba8_from_engine(texture, &pixels)
}

fn texture_from_rgba8_from_engine(texture: &Texture, rgba: &[u8]) -> Result<Texture> {
	let engine_handle = texture.engine_handle().clone();
	let element_count = rgba.len();
	let data = Matrix::allocate(
		&engine_handle,
		vec![texture.height(), texture.width(), 4],
		element_count,
		DType::U8,
	)?;
	data.write_values(rgba)?;
	texture_from_matrix(data, texture.width(), texture.height())
}

fn texture_from_matrix(data: Matrix, width: usize, height: usize) -> Result<Texture> {
	let w32 =
		u32::try_from(width).map_err(|_| Error::out_of_range("texture width exceeds Vulkan limits"))?;
	let h32 = u32::try_from(height)
		.map_err(|_| Error::out_of_range("texture height exceeds Vulkan limits"))?;
	let native = data
		.engine_handle()
		.upload_native_rgba_image(data.storage(), w32, h32)?;
	let semantic = Rc::new(TextureSemantic::new()?);
	// Emit a schema-owned Texture semantic node so the session graph records
	// this value and its extent, matching the Image/Audio semantic contract.
	let _ = data
		.engine_handle()
		.record_texture_semantic(semantic.id, width, height);
	Ok(Texture {
		backing: TextureBacking::BufferBacked { data, native },
		semantic,
	})
}

/// Create a non-owning render-target-backed Texture alias.
///
/// The caller is responsible for ensuring the backing image remains live
/// (via the `RenderFrame`'s `Weak` reference to `RendererState`).
pub(super) fn texture_from_render_target(
	engine_handle: crate::runtime::EngineHandle,
	owner: Rc<std::cell::RefCell<super::renderer::RendererState>>,
	lease: RenderTargetLease,
	image: ash::vk::Image,
	view: ash::vk::ImageView,
	width: u32,
	height: u32,
) -> Texture {
	// Render-target textures are aliases; a fresh semantic id distinguishes
	// them from their backing image and from each other across frames.
	let semantic = TextureSemantic::new().unwrap_or(TextureSemantic { id: 0 });
	Texture {
		backing: TextureBacking::RenderTarget {
			engine_handle,
			owner,
			lease,
			image,
			view,
			width,
			height,
		},
		semantic: Rc::new(semantic),
	}
}

/// Monotonic texture value identity counter.
fn next_texture_value_id() -> crate::Result<u64> {
	static NEXT_TEXTURE_VALUE_ID: AtomicU64 = AtomicU64::new(1);
	let id = NEXT_TEXTURE_VALUE_ID.fetch_add(1, Ordering::Relaxed);
	if id == u64::MAX {
		return Err(crate::Error::resource_exhausted(
			"texture semantic value identity space is exhausted",
		));
	}
	Ok(id)
}

/// Read and save one packed RGBA8 Texture using its path extension.
///
/// Supported extensions are `.jpg`, `.jpeg`, `.png`, `.webp`, `.bmp`, and
/// `.tga`. This is an explicit blocking readback and filesystem boundary.
///
/// # Errors
///
/// Returns an error for unsupported paths, failed device observation, codec
/// failure, or filesystem failure.
pub fn save_texture_file(texture: &Texture, path: impl AsRef<Path>, quality: u32) -> Result<()> {
	let width = u32::try_from(texture.width())
		.map_err(|_| Error::out_of_range("texture width exceeds codec limits"))?;
	let height = u32::try_from(texture.height())
		.map_err(|_| Error::out_of_range("texture height exceeds codec limits"))?;
	let rgba = texture.read_rgba8()?;
	crate::image::save_rgba_file(&rgba, width, height, path, quality)
}
