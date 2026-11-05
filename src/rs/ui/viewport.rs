//! `Viewport` — passive camera/target/view description for rendering.
//!
//! Donor: `oa/ui/viewport.h` and `oa/ui/viewport.cpp`.
//!
//! Architecture naming resolution:
//! - `Viewport` (ui): passive camera + target + viewport rectangle (data)
//! - `Viewer` (viewer): the windowed interactive application (controller)
//!
//! This is not a window or an interactive element. It declares what to
//! render and where without owning swapchains or handling input.

use crate::render::{Camera, Texture};
use crate::{Error, Result};

/// What kind of content the viewport describes.
///
/// Donor: `oa::ViewportMode`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum ViewportMode {
	/// 2D image/video with an orthographic camera.
	#[default]
	Image2D,
	/// 3D scene with a perspective camera.
	Scene3D,
	/// Matrix-as-heatmap visualization.
	Matrix,
	/// Video with a timeline (2D + time control).
	Video,
	/// Live camera feed.
	CameraStream,
}

/// Viewport rectangle and depth range.
///
/// Donor: `oa::ViewportDesc`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ViewportDesc {
	pub x: i32,
	pub y: i32,
	pub width: i32,
	pub height: i32,
	pub min_depth: f32,
	pub max_depth: f32,
}

impl Default for ViewportDesc {
	fn default() -> Self {
		Self {
			x: 0,
			y: 0,
			width: 1280,
			height: 720,
			min_depth: 0.0,
			max_depth: 1.0,
		}
	}
}

impl ViewportDesc {
	/// Return whether the rectangle is positive and the depth range is a
	/// finite ordered subset of `[0, 1]`.
	#[must_use]
	pub fn is_valid(&self) -> bool {
		self.width > 0
			&& self.height > 0
			&& self.min_depth.is_finite()
			&& self.max_depth.is_finite()
			&& self.min_depth >= 0.0
			&& self.min_depth <= self.max_depth
			&& self.max_depth <= 1.0
	}

	/// Return `width / height`.
	///
	/// # Errors
	///
	/// Returns [`ErrorKind::InvalidArgument`] when the description is invalid.
	///
	/// [`ErrorKind::InvalidArgument`]: crate::ErrorKind::InvalidArgument
	pub fn aspect_ratio(&self) -> Result<f32> {
		if !self.is_valid() {
			return Err(Error::invalid_argument(
				"ViewportDesc aspect ratio requires a valid viewport",
			));
		}
		Ok(self.width as f32 / self.height as f32)
	}
}

/// Scissor rectangle for clipping.
///
/// Donor: `oa::ScissorDesc`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ScissorDesc {
	pub x: i32,
	pub y: i32,
	pub width: i32,
	pub height: i32,
}

impl Default for ScissorDesc {
	fn default() -> Self {
		Self {
			x: 0,
			y: 0,
			width: 1280,
			height: 720,
		}
	}
}

impl ScissorDesc {
	/// Return whether the rectangle is positive.
	#[must_use]
	pub const fn is_valid(&self) -> bool {
		self.width > 0 && self.height > 0
	}

	/// Return whether `(x, y)` lies inside the rectangle.
	#[must_use]
	pub const fn contains(&self, x: i32, y: i32) -> bool {
		self.is_valid()
			&& x as i64 >= self.x as i64
			&& (x as i64) < self.x as i64 + self.width as i64
			&& y as i64 >= self.y as i64
			&& (y as i64) < self.y as i64 + self.height as i64
	}
}

/// Layer/filter flags for selective rendering.
///
/// Donor: `oa::RenderLayer`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct RenderLayer(pub u32);

impl RenderLayer {
	pub const NONE: Self = Self(0);
	pub const DEFAULT: Self = Self(1 << 0);
	pub const UI: Self = Self(1 << 1);
	pub const OVERLAY: Self = Self(1 << 2);
	pub const BACKGROUND: Self = Self(1 << 3);
	pub const DEBUG: Self = Self(1 << 4);
	pub const ALL: Self = Self(0xFFFF_FFFF);

	/// Return whether `other` shares any bit with this mask.
	#[must_use]
	pub const fn intersects(self, other: Self) -> bool {
		self.0 & other.0 != 0
	}
}

impl std::ops::BitOr for RenderLayer {
	type Output = Self;
	fn bitor(self, rhs: Self) -> Self {
		Self(self.0 | rhs.0)
	}
}

impl std::ops::BitOrAssign for RenderLayer {
	fn bitor_assign(&mut self, rhs: Self) {
		self.0 |= rhs.0;
	}
}

impl std::ops::BitAnd for RenderLayer {
	type Output = Self;
	fn bitand(self, rhs: Self) -> Self {
		Self(self.0 & rhs.0)
	}
}

/// Passive view description for rendering.
///
/// Declares camera, target, viewport, scissor, layer mask, and clear values.
/// `setup_image_2d`/`setup_image_aspect_fit` manufacture and retain an
/// orthographic camera internally; an externally borrowed camera supplied
/// later through [`Self::set_camera`] takes precedence again.
///
/// Donor: `oa::Viewport`.
pub struct Viewport<'a> {
	mode: ViewportMode,
	camera: Option<&'a Camera>,
	internal_camera: Option<Camera>,
	target: Option<&'a Texture>,
	viewport: ViewportDesc,
	scissor: ScissorDesc,
	use_scissor: bool,
	layer_mask: RenderLayer,
	clear_color: [f32; 4],
	clear_depth: f32,
}

impl<'a> Default for Viewport<'a> {
	fn default() -> Self {
		Self::with_mode(ViewportMode::Image2D)
	}
}

impl<'a> Viewport<'a> {
	/// Create a viewport for `mode`.
	#[must_use]
	pub fn with_mode(mode: ViewportMode) -> Self {
		Self {
			mode,
			camera: None,
			internal_camera: None,
			target: None,
			viewport: ViewportDesc::default(),
			scissor: ScissorDesc::default(),
			use_scissor: false,
			layer_mask: RenderLayer::DEFAULT,
			clear_color: [0.01, 0.01, 0.01, 1.0],
			clear_depth: 1.0,
		}
	}

	/// Set the content mode.
	pub fn set_mode(&mut self, mode: ViewportMode) {
		self.mode = mode;
	}

	/// Return the content mode.
	#[must_use]
	pub const fn mode(&self) -> ViewportMode {
		self.mode
	}

	/// Borrow an external camera, or `None` to clear one.
	pub fn set_camera(&mut self, camera: Option<&'a Camera>) {
		self.camera = camera;
	}

	/// Return the active camera — the borrowed external camera when present,
	/// otherwise the internally manufactured one.
	#[must_use]
	pub fn camera(&self) -> Option<&Camera> {
		self.camera.or(self.internal_camera.as_ref())
	}

	/// Borrow the render target, or `None` to clear it.
	pub fn set_target(&mut self, target: Option<&'a Texture>) {
		self.target = target;
	}

	/// Return the borrowed render target.
	#[must_use]
	pub const fn target(&self) -> Option<&'a Texture> {
		self.target
	}

	/// Set the viewport rectangle and depth range.
	///
	/// # Errors
	///
	/// Returns an error for non-positive dimensions or a depth range outside
	/// `[0, 1]`.
	pub fn set_viewport(&mut self, viewport: ViewportDesc) -> Result<()> {
		if !viewport.is_valid() {
			return Err(Error::invalid_argument(
				"Viewport::set_viewport requires positive dimensions and a finite [0,1] depth range",
			));
		}
		self.viewport = viewport;
		Ok(())
	}

	/// Return the configured viewport rectangle.
	#[must_use]
	pub const fn viewport(&self) -> ViewportDesc {
		self.viewport
	}

	/// Enable scissor clipping with `scissor`.
	///
	/// # Errors
	///
	/// Returns an error for non-positive dimensions.
	pub fn set_scissor(&mut self, scissor: ScissorDesc) -> Result<()> {
		if !scissor.is_valid() {
			return Err(Error::invalid_argument(
				"Viewport::set_scissor requires positive dimensions",
			));
		}
		self.scissor = scissor;
		self.use_scissor = true;
		Ok(())
	}

	/// Disable scissor clipping.
	pub fn disable_scissor(&mut self) {
		self.use_scissor = false;
	}

	/// Return whether scissor clipping is enabled.
	#[must_use]
	pub const fn is_scissor_enabled(&self) -> bool {
		self.use_scissor
	}

	/// Return the configured scissor rectangle.
	#[must_use]
	pub const fn scissor(&self) -> ScissorDesc {
		self.scissor
	}

	/// Set the layer visibility mask.
	pub fn set_layer_mask(&mut self, mask: RenderLayer) {
		self.layer_mask = mask;
	}

	/// Return the layer visibility mask.
	#[must_use]
	pub const fn layer_mask(&self) -> RenderLayer {
		self.layer_mask
	}

	/// Return whether `layer` passes the visibility mask.
	#[must_use]
	pub const fn is_layer_visible(&self, layer: RenderLayer) -> bool {
		self.layer_mask.intersects(layer)
	}

	/// Set the RGBA clear color.
	///
	/// # Errors
	///
	/// Returns an error for non-finite components.
	pub fn set_clear_color(&mut self, color: [f32; 4]) -> Result<()> {
		if !color.iter().all(|component| component.is_finite()) {
			return Err(Error::invalid_argument(
				"Viewport::set_clear_color requires finite components",
			));
		}
		self.clear_color = color;
		Ok(())
	}

	/// Return the RGBA clear color.
	#[must_use]
	pub const fn clear_color(&self) -> [f32; 4] {
		self.clear_color
	}

	/// Set the clear depth.
	///
	/// # Errors
	///
	/// Returns an error for a non-finite value or one outside `[0, 1]`.
	pub fn set_clear_depth(&mut self, depth: f32) -> Result<()> {
		if !depth.is_finite() || !(0.0..=1.0).contains(&depth) {
			return Err(Error::invalid_argument(
				"Viewport::set_clear_depth requires a finite value in [0,1]",
			));
		}
		self.clear_depth = depth;
		Ok(())
	}

	/// Return the clear depth.
	#[must_use]
	pub const fn clear_depth(&self) -> f32 {
		self.clear_depth
	}

	/// Return whether the viewport is complete and consistent: an active
	/// camera, a borrowed target, a valid viewport rectangle, a valid
	/// enabled scissor, and finite clear values.
	#[must_use]
	pub fn is_valid(&self) -> bool {
		self.camera().is_some()
			&& self.target.is_some()
			&& self.viewport.is_valid()
			&& (!self.use_scissor || self.scissor.is_valid())
			&& self
				.clear_color
				.iter()
				.all(|component| component.is_finite())
			&& self.clear_depth.is_finite()
			&& (0.0..=1.0).contains(&self.clear_depth)
	}

	/// Set up orthographic 2D image viewing over `target`.
	///
	/// Manufactures an internal orthographic camera matching the image
	/// extent; a later [`Self::set_camera`] overrides it again.
	///
	/// # Errors
	///
	/// Returns an error for non-positive image dimensions or an invalid
	/// viewport.
	pub fn setup_image_2d(
		&mut self,
		image_width: f32,
		image_height: f32,
		target: &'a Texture,
		viewport: ViewportDesc,
	) -> Result<()> {
		if !image_width.is_finite()
			|| !image_height.is_finite()
			|| image_width <= 0.0
			|| image_height <= 0.0
		{
			return Err(Error::invalid_argument(
				"Viewport::setup_image_2d requires finite positive image dimensions",
			));
		}
		if !viewport.is_valid() {
			return Err(Error::invalid_argument(
				"Viewport::setup_image_2d requires a valid viewport",
			));
		}
		self.internal_camera = Some(Camera::new_orthographic(
			image_width,
			image_height,
			0.0,
			1.0,
		));
		self.camera = None;
		self.mode = ViewportMode::Image2D;
		self.target = Some(target);
		self.viewport = viewport;
		self.scissor = ScissorDesc {
			x: viewport.x,
			y: viewport.y,
			width: viewport.width,
			height: viewport.height,
		};
		self.use_scissor = false;
		Ok(())
	}

	/// Set up aspect-fitted image display: a centered letterboxed viewport
	/// preserving the image aspect inside the window.
	///
	/// # Errors
	///
	/// Returns an error for non-positive dimensions, a window exceeding
	/// signed pixel coordinates, or a result that cannot be represented as a
	/// positive pixel rectangle.
	pub fn setup_image_aspect_fit(
		&mut self,
		image_width: f32,
		image_height: f32,
		window_width: f32,
		window_height: f32,
		target: &'a Texture,
	) -> Result<()> {
		if !image_width.is_finite()
			|| !image_height.is_finite()
			|| !window_width.is_finite()
			|| !window_height.is_finite()
			|| image_width <= 0.0
			|| image_height <= 0.0
			|| window_width <= 0.0
			|| window_height <= 0.0
		{
			return Err(Error::invalid_argument(
				"Viewport::setup_image_aspect_fit requires finite positive image and window dimensions",
			));
		}
		let max_pixel = f64::from(i32::MAX);
		if f64::from(window_width) > max_pixel || f64::from(window_height) > max_pixel {
			return Err(Error::out_of_range(
				"Viewport::setup_image_aspect_fit window exceeds signed pixel coordinates",
			));
		}
		let image_aspect = f64::from(image_width) / f64::from(image_height);
		let window_aspect = f64::from(window_width) / f64::from(window_height);
		let (display_width, display_height) = if image_aspect > window_aspect {
			(
				f64::from(window_width),
				f64::from(window_width) / image_aspect,
			)
		} else {
			(
				f64::from(window_height) * image_aspect,
				f64::from(window_height),
			)
		};
		let offset_x = (f64::from(window_width) - display_width) * 0.5;
		let offset_y = (f64::from(window_height) - display_height) * 0.5;
		if !display_width.is_finite()
			|| !display_height.is_finite()
			|| !offset_x.is_finite()
			|| !offset_y.is_finite()
			|| display_width < 0.5
			|| display_height < 0.5
			|| display_width > max_pixel
			|| display_height > max_pixel
			|| offset_x < 0.0
			|| offset_y < 0.0
			|| offset_x > max_pixel
			|| offset_y > max_pixel
		{
			return Err(Error::out_of_range(
				"Viewport::setup_image_aspect_fit result cannot be represented as a positive pixel rectangle",
			));
		}
		let viewport = ViewportDesc {
			x: offset_x.round() as i32,
			y: offset_y.round() as i32,
			width: display_width.round() as i32,
			height: display_height.round() as i32,
			min_depth: 0.0,
			max_depth: 1.0,
		};
		self.setup_image_2d(image_width, image_height, target, viewport)
	}
}
