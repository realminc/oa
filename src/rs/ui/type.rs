//! UI types: PixelRect, layout primitives, UiStyle, widget and chrome records.
//!
//! Donor references: `oa/ui/canvas.h`, `oa/ui/style.h`, `oa/ui/event.h`

// ── PixelRect ─────────────────────────────────────────────────────────────────

/// Axis-aligned pixel rectangle in physical screen coordinates.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct PixelRect {
	pub x: i32,
	pub y: i32,
	pub w: u32,
	pub h: u32,
}

impl PixelRect {
	pub const fn new(x: i32, y: i32, w: u32, h: u32) -> Self {
		Self { x, y, w, h }
	}

	/// True when the rectangle has non-zero area.
	pub const fn is_non_empty(&self) -> bool {
		self.w > 0 && self.h > 0
	}

	/// True when `(px, py)` is inside this rectangle.
	pub fn contains(&self, px: f32, py: f32) -> bool {
		px >= self.x as f32
			&& py >= self.y as f32
			&& px < (self.x + self.w as i32) as f32
			&& py < (self.y + self.h as i32) as f32
	}

	/// Return the bounding clip for `child` intersected with `self`.
	pub fn clip(&self, child: &PixelRect) -> PixelRect {
		let x0 = self.x.max(child.x);
		let y0 = self.y.max(child.y);
		let x1 = (self.x + self.w as i32).min(child.x + child.w as i32);
		let y1 = (self.y + self.h as i32).min(child.y + child.h as i32);
		if x1 <= x0 || y1 <= y0 {
			return PixelRect::default();
		}
		PixelRect {
			x: x0,
			y: y0,
			w: (x1 - x0) as u32,
			h: (y1 - y0) as u32,
		}
	}
}

pub use super::type_gen::{
	UiDropdownConfig, UiGridConfig, UiHeatmapConfig, UiPlotConfig, UiPopupConfig,
	UiPropertyRowConfig, UiScrollConfig, UiSplitConfig, UiTabBarResult, UiTooltipConfig,
	UiTreeRowConfig, UiTreeRowResult,
};
use crate::Color;

// ── UiStyle ───────────────────────────────────────────────────────────────────

/// Theme colors and metrics used by all widgets.
#[derive(Clone, Debug)]
pub struct UiStyle {
	/// Panel / window background.
	pub background: Color,
	/// Widget surface (button, slider track).
	pub surface: Color,
	/// Hovered and held surfaces.
	pub surface_hover: Color,
	pub surface_active: Color,
	/// Active/hovered widget accent.
	pub accent: Color,
	/// Hovered/held foreground accent.
	pub accent_hover: Color,
	/// Primary text color.
	pub text: Color,
	/// Secondary/muted text.
	pub text_secondary: Color,
	/// Disabled control text.
	pub text_disabled: Color,
	/// Widget border.
	pub border: Color,
	/// Strong visualization border.
	pub border_strong: Color,
	/// Widget corner radius in pixels.
	pub corner_radius: f32,
	/// Inner content padding in pixels.
	pub padding: f32,
	/// Base font size in pixels.
	pub font_size: f32,
}

impl Default for UiStyle {
	/// Realm dark theme — mirrors the donor default.
	fn default() -> Self {
		Self {
			background: Color::new(0.082, 0.082, 0.090, 1.0),
			surface: Color::new(0.141, 0.141, 0.161, 1.0),
			surface_hover: Color::new(0.133, 0.133, 0.133, 1.0),
			surface_active: Color::new(0.188, 0.188, 0.188, 1.0),
			accent: Color::new(0.388, 0.400, 0.945, 1.0),
			accent_hover: Color::new(0.506, 0.549, 0.973, 1.0),
			text: Color::new(0.961, 0.961, 0.961, 1.0),
			text_secondary: Color::new(0.600, 0.600, 0.620, 1.0),
			text_disabled: Color::new(0.400, 0.400, 0.400, 1.0),
			border: Color::new(0.250, 0.250, 0.280, 1.0),
			border_strong: Color::new(1.0, 1.0, 1.0, 0.15),
			corner_radius: 4.0,
			padding: 8.0,
			font_size: 14.0,
		}
	}
}

impl UiStyle {
	/// Donor media viewer palette (`oa::UiStyle::viewerDark`).
	pub fn viewer_dark() -> Self {
		Self {
			background: Color::new_from_u32(0x181818FF),
			surface: Color::new_from_u32(0x303030E8),
			surface_hover: Color::new_from_u32(0x424242F0),
			surface_active: Color::new_from_u32(0x505050F5),
			border: Color::new_from_u32(0xFFFFFF20),
			border_strong: Color::new_from_u32(0xFFFFFF38),
			text: Color::new_from_u32(0xFFFFFFFF),
			text_secondary: Color::new_from_u32(0xE6E6E6FF),
			text_disabled: Color::new_from_u32(0x777777FF),
			accent: Color::new_from_u32(0xF5F5F5FF),
			accent_hover: Color::new_from_u32(0xFFFFFFFF),
			corner_radius: 15.0,
			font_size: 15.0,
			..Self::default()
		}
	}
}

// ── Layout primitives ─────────────────────────────────────────────────────────

/// Sizing policy for one layout axis.
#[derive(Clone, Copy, Debug, Default)]
pub enum UiSizing {
	/// Expand to fill available parent space.
	#[default]
	Fill,
	/// Fixed pixel size.
	Fixed(f32),
	/// Shrink-wrap content.
	Hug,
}

/// Per-side padding in pixels.
#[derive(Clone, Copy, Debug, Default)]
pub struct UiEdge {
	pub top: f32,
	pub right: f32,
	pub bottom: f32,
	pub left: f32,
}

impl UiEdge {
	pub const fn all(v: f32) -> Self {
		Self {
			top: v,
			right: v,
			bottom: v,
			left: v,
		}
	}
}

/// Flow direction for a panel or row.
#[derive(Clone, Copy, Debug, Default)]
pub enum UiDirection {
	#[default]
	Column,
	Row,
}

/// Alignment of children along the cross-axis.
#[derive(Clone, Copy, Debug, Default)]
pub enum UiAlign {
	Start,
	Center,
	End,
	#[default]
	Stretch,
}

/// Layout descriptor passed to `begin_panel`.
#[derive(Clone, Debug)]
pub struct UiLayout {
	pub direction: UiDirection,
	pub align: UiAlign,
	pub gap: f32,
	pub padding: UiEdge,
	pub width: UiSizing,
	pub height: UiSizing,
}

impl Default for UiLayout {
	fn default() -> Self {
		Self {
			direction: UiDirection::Column,
			align: UiAlign::Stretch,
			gap: 4.0,
			padding: UiEdge::all(8.0),
			width: UiSizing::Fill,
			height: UiSizing::Hug,
		}
	}
}

// ── Tab bar ───────────────────────────────────────────────────────────────────

/// A single tab entry passed to [`crate::ui::Ui::tab_bar`].
///
/// Donor: `oa::UiTabItem`.
#[derive(Clone, Debug, Default)]
pub struct UiTabItem {
	/// Stable string identifier (not displayed).
	pub id: String,
	/// Visible tab label.
	pub label: String,
	/// Dirty indicator shown next to the label.
	pub dirty: bool,
	/// Whether the tab shows a close button.
	pub closable: bool,
	/// Whether the tab is enabled (greyed out when false).
	pub enabled: bool,
}

impl UiTabItem {
	pub fn new(id: impl Into<String>, label: impl Into<String>) -> Self {
		Self {
			id: id.into(),
			label: label.into(),
			closable: true,
			enabled: true,
			dirty: false,
		}
	}
}

/// Caller-owned tab bar presentation state.
///
/// Donor: `oa::UiTabBarState`.
#[derive(Clone, Debug, Default)]
pub struct UiTabBarState {
	/// Index of the currently selected tab, or -1 for none.
	pub selected: i32,
	/// Index of the first visible tab (for overflow scrolling).
	pub first_visible: i32,
}

// ── Tree row ──────────────────────────────────────────────────────────────────

impl UiTreeRowConfig {
	pub const fn new() -> Self {
		Self {
			depth: 0,
			indent: 16,
			disclosure_width: 18,
			has_children: false,
			open: false,
			selected: false,
			enabled: true,
		}
	}
}

// ── Node canvas ───────────────────────────────────────────────────────────────

/// Caller-owned view state for a node-graph canvas.
///
/// Donor: `oa::NodeCanvasState`.
#[derive(Clone, Debug)]
pub struct NodeCanvasState {
	/// World-space pan offset (origin of the viewport in world coords).
	pub pan: [f32; 2],
	/// Zoom factor (pixels per world unit).
	pub zoom: f32,
	/// Viewport dimensions in screen pixels.
	pub view_size: [f32; 2],
}

impl Default for NodeCanvasState {
	fn default() -> Self {
		Self {
			pan: [0.0, 0.0],
			zoom: 1.0,
			view_size: [0.0, 0.0],
		}
	}
}

/// World-space axis-aligned bounds for canvas hit tests and fit targets.
///
/// Donor: `oa::WorldAabb`.
#[derive(Clone, Copy, Debug, Default)]
pub struct WorldAabb {
	pub min: [f32; 2],
	pub max: [f32; 2],
}

impl WorldAabb {
	/// Return whether the bounds are finite and strictly positive in extent.
	#[must_use]
	pub fn is_valid(&self) -> bool {
		self.min[0].is_finite()
			&& self.min[1].is_finite()
			&& self.max[0].is_finite()
			&& self.max[1].is_finite()
			&& self.max[0] > self.min[0]
			&& self.max[1] > self.min[1]
	}

	/// Return whether `point` lies inside the bounds.
	#[must_use]
	pub fn contains(&self, point: [f32; 2]) -> bool {
		point[0] >= self.min[0]
			&& point[0] <= self.max[0]
			&& point[1] >= self.min[1]
			&& point[1] <= self.max[1]
	}

	/// Return whether both bounds are valid and overlap.
	#[must_use]
	pub fn intersects(&self, other: &Self) -> bool {
		self.is_valid()
			&& other.is_valid()
			&& self.min[0] < other.max[0]
			&& self.max[0] > other.min[0]
			&& self.min[1] < other.max[1]
			&& self.max[1] > other.min[1]
	}
}

/// Borrowed hit-test geometry for one node.
///
/// Donor: `oa::NodeCanvasHitItem`.
#[derive(Clone, Copy, Debug)]
pub struct NodeCanvasHitItem {
	pub id: u64,
	pub bounds: WorldAabb,
	pub layer: i32,
	pub enabled: bool,
}

/// The winning hit-test item and the resolved world point.
///
/// Donor: `oa::NodeCanvasHit`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct NodeCanvasHit {
	pub id: u64,
	pub index: usize,
	pub layer: i32,
	pub world_point: [f32; 2],
}

/// Resolved grid spacing for one minor tier and its major cadences.
///
/// Donor: `oa::NodeCanvasGrid`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct NodeCanvasGrid {
	pub minor_world_step: f32,
	pub minor_screen_step: f32,
	pub origin_screen: [f32; 2],
	pub major_every: u32,
	pub super_major_every: u32,
}

/// Node-graph canvas navigation, coordinate mapping, and hit tests.
///
/// Caller-owned; passed to [`crate::ui::Ui::node_canvas_grid`] to render the
/// background grid. Pan/zoom mutations cancel any running fit animation;
/// `fit_to_view` starts one that `step_animation` advances.
///
/// Donor: `oa::NodeCanvas`.
#[derive(Clone, Debug, Default)]
pub struct NodeCanvas {
	state: NodeCanvasState,
	anim_start: NodeCanvasState,
	anim_target: NodeCanvasState,
	animating: bool,
	anim_t: f32,
}

impl NodeCanvas {
	pub const ZOOM_MIN: f32 = 0.05;
	pub const ZOOM_MAX: f32 = 8.0;

	pub fn new() -> Self {
		Self::default()
	}

	/// Access the current canvas state.
	pub const fn state(&self) -> &NodeCanvasState {
		&self.state
	}

	/// Return whether a fit animation is still running.
	#[must_use]
	pub const fn is_animating(&self) -> bool {
		self.animating
	}

	/// Replace the complete canvas state and cancel any animation.
	///
	/// # Errors
	///
	/// Returns an error for non-finite pan/view state, a zoom outside
	/// [`Self::ZOOM_MIN`]..=[`Self::ZOOM_MAX`], or a non-positive view size.
	pub fn set_state(&mut self, state: NodeCanvasState) -> crate::Result<()> {
		if !is_valid_canvas_state(&state, true) {
			return Err(crate::Error::invalid_argument(
				"NodeCanvas::set_state requires finite pan, positive view size, and zoom in range",
			));
		}
		self.anim_start = state.clone();
		self.anim_target = state.clone();
		self.state = state;
		self.animating = false;
		self.anim_t = 0.0;
		Ok(())
	}

	/// Set the canvas view size (called once when the viewport is known).
	///
	/// # Errors
	///
	/// Returns an error for non-finite or non-positive dimensions.
	pub fn set_view_size(&mut self, w: f32, h: f32) -> crate::Result<()> {
		if !w.is_finite() || !h.is_finite() || w <= 0.0 || h <= 0.0 {
			return Err(crate::Error::invalid_argument(
				"NodeCanvas::set_view_size requires finite positive dimensions",
			));
		}
		self.state.view_size = [w, h];
		self.anim_start.view_size = self.state.view_size;
		self.anim_target.view_size = self.state.view_size;
		self.animating = false;
		self.anim_t = 0.0;
		Ok(())
	}

	/// Convert a world-space position to screen-space pixels.
	#[must_use]
	pub fn world_to_screen(&self, world: [f32; 2]) -> [f32; 2] {
		[
			(world[0] - self.state.pan[0]) * self.state.zoom + self.state.view_size[0] * 0.5,
			(world[1] - self.state.pan[1]) * self.state.zoom + self.state.view_size[1] * 0.5,
		]
	}

	/// Convert screen-space pixels to world-space coordinates.
	#[must_use]
	pub fn screen_to_world(&self, screen: [f32; 2]) -> [f32; 2] {
		[
			(screen[0] - self.state.view_size[0] * 0.5) / self.state.zoom + self.state.pan[0],
			(screen[1] - self.state.view_size[1] * 0.5) / self.state.zoom + self.state.pan[1],
		]
	}

	/// Return the world-space rectangle currently visible through the view.
	#[must_use]
	pub fn visible_world_rect(&self) -> WorldAabb {
		let half = [
			self.state.view_size[0] * 0.5 / self.state.zoom,
			self.state.view_size[1] * 0.5 / self.state.zoom,
		];
		WorldAabb {
			min: [self.state.pan[0] - half[0], self.state.pan[1] - half[1]],
			max: [self.state.pan[0] + half[0], self.state.pan[1] + half[1]],
		}
	}

	/// Find the topmost enabled item under a screen-space point.
	///
	/// Items are tested in order; on overlapping hits the item with the
	/// highest layer wins, with later items breaking ties.
	///
	/// # Errors
	///
	/// Returns an error for invalid canvas state, a non-finite screen point,
	/// zero item IDs, or invalid item bounds.
	pub fn hit_test(
		&self,
		screen: [f32; 2],
		items: &[NodeCanvasHitItem],
	) -> crate::Result<Option<NodeCanvasHit>> {
		if !is_valid_canvas_state(&self.state, true) {
			return Err(invalid_canvas_state());
		}
		if !screen[0].is_finite() || !screen[1].is_finite() {
			return Err(crate::Error::invalid_argument(
				"NodeCanvas::hit_test requires a finite screen point",
			));
		}
		for item in items {
			if item.id == 0 || !item.bounds.is_valid() {
				return Err(crate::Error::invalid_argument(
					"NodeCanvas::hit_test requires non-zero IDs and valid bounds",
				));
			}
		}
		let world = self.screen_to_world(screen);
		if !world[0].is_finite() || !world[1].is_finite() {
			return Err(crate::Error::out_of_range(
				"NodeCanvas::hit_test produced a non-finite world point",
			));
		}
		let mut hit: Option<NodeCanvasHit> = None;
		for (index, item) in items.iter().enumerate() {
			if !item.enabled || !item.bounds.contains(world) {
				continue;
			}
			if hit.is_none_or(|previous| item.layer >= previous.layer) {
				hit = Some(NodeCanvasHit {
					id: item.id,
					index,
					layer: item.layer,
					world_point: world,
				});
			}
		}
		Ok(hit)
	}

	/// Resolve the visible minor grid tier and its major cadences.
	///
	/// The donor preserves a Maya-style decimal hierarchy: zooming out
	/// promotes the minor tier by whole decades; zooming in never invents a
	/// sub-base grid. Donor call defaults are `8.0` minimum screen spacing,
	/// `10.0` base world step, `10` major, and `100` super-major.
	///
	/// # Errors
	///
	/// Returns an error for invalid canvas state, non-finite or out-of-range
	/// spacing, or undivisible major/super-major tiers.
	pub fn grid(
		&self,
		minimum_screen_spacing: f32,
		base_world_step: f32,
		major_every: u32,
		super_major_every: u32,
	) -> crate::Result<NodeCanvasGrid> {
		if !is_valid_canvas_state(&self.state, true) {
			return Err(invalid_canvas_state());
		}
		if !minimum_screen_spacing.is_finite()
			|| !base_world_step.is_finite()
			|| !(4.0..=512.0).contains(&minimum_screen_spacing)
			|| base_world_step <= 0.0
			|| !(2..=100).contains(&major_every)
			|| super_major_every < major_every * 2
			|| super_major_every > 10000
			|| !super_major_every.is_multiple_of(major_every)
		{
			return Err(crate::Error::invalid_argument(
				"NodeCanvas::grid requires finite spacing, a positive base step, and divisible major/super-major tiers",
			));
		}
		let ratio =
			f64::from(minimum_screen_spacing) / (f64::from(base_world_step) * f64::from(self.state.zoom));
		let exponent = ratio.log10().ceil().max(0.0);
		if !exponent.is_finite() || exponent < f64::from(i32::MIN) || exponent > f64::from(i32::MAX) {
			return Err(crate::Error::out_of_range(
				"NodeCanvas::grid spacing exponent is out of range",
			));
		}
		let world_step = f64::from(base_world_step) * 10.0_f64.powf(exponent);
		let screen_step = world_step * f64::from(self.state.zoom);
		let origin = self.world_to_screen([0.0, 0.0]);
		if !world_step.is_finite()
			|| !screen_step.is_finite()
			|| world_step <= 0.0
			|| screen_step <= 0.0
			|| world_step > f64::from(f32::MAX)
			|| screen_step > f64::from(f32::MAX)
			|| !origin[0].is_finite()
			|| !origin[1].is_finite()
		{
			return Err(crate::Error::out_of_range(
				"NodeCanvas::grid produced non-finite metrics",
			));
		}
		Ok(NodeCanvasGrid {
			minor_world_step: world_step as f32,
			minor_screen_step: screen_step as f32,
			origin_screen: origin,
			major_every,
			super_major_every,
		})
	}

	/// Pan by a screen-space delta.
	///
	/// # Errors
	///
	/// Returns an error for invalid canvas state, a non-finite delta, or a
	/// result outside the finite `f32` range.
	pub fn pan(&mut self, delta_screen: [f32; 2]) -> crate::Result<()> {
		if !is_valid_canvas_state(&self.state, false) {
			return Err(invalid_canvas_state());
		}
		if !delta_screen[0].is_finite() || !delta_screen[1].is_finite() {
			return Err(crate::Error::invalid_argument(
				"NodeCanvas::pan requires a finite screen delta",
			));
		}
		let next_x =
			f64::from(self.state.pan[0]) - f64::from(delta_screen[0]) / f64::from(self.state.zoom);
		let next_y =
			f64::from(self.state.pan[1]) - f64::from(delta_screen[1]) / f64::from(self.state.zoom);
		if !next_x.is_finite()
			|| !next_y.is_finite()
			|| next_x.abs() > f64::from(f32::MAX)
			|| next_y.abs() > f64::from(f32::MAX)
		{
			return Err(crate::Error::out_of_range(
				"NodeCanvas::pan result is out of range",
			));
		}
		self.state.pan = [next_x as f32, next_y as f32];
		self.animating = false;
		self.anim_t = 0.0;
		Ok(())
	}

	/// Zoom by `factor`, keeping the world point under `focus_screen` fixed.
	///
	/// # Errors
	///
	/// Returns an error for invalid canvas state, a non-positive or
	/// non-finite factor, a non-finite focus, or an out-of-range result.
	pub fn zoom_at(&mut self, factor: f32, focus_screen: [f32; 2]) -> crate::Result<()> {
		if !is_valid_canvas_state(&self.state, false) {
			return Err(invalid_canvas_state());
		}
		if !factor.is_finite()
			|| factor <= 0.0
			|| !focus_screen[0].is_finite()
			|| !focus_screen[1].is_finite()
		{
			return Err(crate::Error::invalid_argument(
				"NodeCanvas::zoom_at requires a finite positive factor and focus",
			));
		}
		let previous = self.state.clone();
		let world_before = self.screen_to_world(focus_screen);
		if !world_before[0].is_finite() || !world_before[1].is_finite() {
			return Err(crate::Error::out_of_range(
				"NodeCanvas::zoom_at focus is outside the finite world range",
			));
		}
		self.state.zoom = (self.state.zoom * factor).clamp(Self::ZOOM_MIN, Self::ZOOM_MAX);
		let world_after = self.screen_to_world(focus_screen);
		self.state.pan[0] -= world_after[0] - world_before[0];
		self.state.pan[1] -= world_after[1] - world_before[1];
		if !is_valid_canvas_state(&self.state, false) {
			self.state = previous;
			return Err(crate::Error::out_of_range(
				"NodeCanvas::zoom_at result is out of range",
			));
		}
		self.animating = false;
		self.anim_t = 0.0;
		Ok(())
	}

	/// Start a pan/zoom animation fitting `bounds` into the view.
	///
	/// # Errors
	///
	/// Returns an error for invalid canvas state, invalid bounds, or padding
	/// that leaves no usable view area.
	pub fn fit_to_view(&mut self, bounds: &WorldAabb, padding_pixels: f32) -> crate::Result<()> {
		if !is_valid_canvas_state(&self.state, true) {
			return Err(invalid_canvas_state());
		}
		if !bounds.is_valid()
			|| !padding_pixels.is_finite()
			|| padding_pixels < 0.0
			|| padding_pixels * 2.0 >= self.state.view_size[0]
			|| padding_pixels * 2.0 >= self.state.view_size[1]
		{
			return Err(crate::Error::invalid_argument(
				"NodeCanvas::fit_to_view requires valid bounds and usable padding",
			));
		}
		let extent_x = f64::from(bounds.max[0] - bounds.min[0]).max(1.0);
		let extent_y = f64::from(bounds.max[1] - bounds.min[1]).max(1.0);
		let mut target = self.state.clone();
		target.pan = [
			((f64::from(bounds.min[0]) + f64::from(bounds.max[0])) * 0.5) as f32,
			((f64::from(bounds.min[1]) + f64::from(bounds.max[1])) * 0.5) as f32,
		];
		target.zoom = ((f64::from(self.state.view_size[0] - padding_pixels * 2.0) / extent_x)
			.min(f64::from(self.state.view_size[1] - padding_pixels * 2.0) / extent_y)
			as f32)
			.clamp(Self::ZOOM_MIN, Self::ZOOM_MAX);
		if !is_valid_canvas_state(&target, true) {
			return Err(crate::Error::out_of_range(
				"NodeCanvas::fit_to_view produced an invalid target",
			));
		}
		self.anim_start = self.state.clone();
		self.anim_target = target;
		self.animating = true;
		self.anim_t = 0.0;
		Ok(())
	}

	/// Advance a running fit animation by `delta_ms` milliseconds.
	///
	/// A no-op when no animation is running. The donor eases with a smoothstep
	/// over a 200 ms duration.
	///
	/// # Errors
	///
	/// Returns an error for a non-finite or negative delta.
	pub fn step_animation(&mut self, delta_ms: f32) -> crate::Result<()> {
		if !delta_ms.is_finite() || delta_ms < 0.0 {
			return Err(crate::Error::invalid_argument(
				"NodeCanvas::step_animation requires a finite non-negative delta",
			));
		}
		if !self.animating {
			return Ok(());
		}
		self.anim_t += delta_ms * 0.005;
		if self.anim_t >= 1.0 {
			self.state.pan = self.anim_target.pan;
			self.state.zoom = self.anim_target.zoom;
			self.animating = false;
			self.anim_t = 1.0;
			return Ok(());
		}
		let t = self.anim_t * self.anim_t * (3.0 - 2.0 * self.anim_t);
		self.state.pan = [
			self.anim_start.pan[0] + (self.anim_target.pan[0] - self.anim_start.pan[0]) * t,
			self.anim_start.pan[1] + (self.anim_target.pan[1] - self.anim_start.pan[1]) * t,
		];
		self.state.zoom = self.anim_start.zoom + (self.anim_target.zoom - self.anim_start.zoom) * t;
		Ok(())
	}
}

fn is_valid_canvas_state(state: &NodeCanvasState, require_positive_view: bool) -> bool {
	state.pan[0].is_finite()
		&& state.pan[1].is_finite()
		&& state.view_size[0].is_finite()
		&& state.view_size[1].is_finite()
		&& state.zoom.is_finite()
		&& (NodeCanvas::ZOOM_MIN..=NodeCanvas::ZOOM_MAX).contains(&state.zoom)
		&& if require_positive_view {
			state.view_size[0] > 0.0 && state.view_size[1] > 0.0
		} else {
			state.view_size[0] >= 0.0 && state.view_size[1] >= 0.0
		}
}

fn invalid_canvas_state() -> crate::Error {
	crate::Error::failed_precondition(
		"NodeCanvas requires finite pan/view state and an admitted zoom",
	)
}

// ── UiScrollConfig / UiScrollRegion ─────────────────────────────────────────

impl Default for UiScrollConfig {
	fn default() -> Self {
		Self {
			wheel_step: 40.0,
			scrollbar_width: 6.0,
			scrollbar_gap: 2.0,
			show_scrollbar: true,
		}
	}
}

/// Return value from [`crate::ui::Ui::begin_scroll_panel`].
///
/// Donor: `oa::UiScrollRegion`.
#[derive(Clone, Debug, Default)]
pub struct UiScrollRegion {
	/// Visible viewport rectangle.
	pub viewport: PixelRect,
	/// Content area rectangle (may extend below viewport).
	pub content: PixelRect,
	/// Current vertical scroll offset in pixels.
	pub offset_y: i32,
	/// Maximum reachable vertical offset.
	pub max_offset_y: i32,
}

// ── UiVirtualRange ───────────────────────────────────────────────────────────

/// Half-open row range produced by [`crate::ui::Ui::virtual_rows`].
///
/// Donor: `oa::UiVirtualRange`.
#[derive(Clone, Copy, Debug, Default)]
pub struct UiVirtualRange {
	/// Index of the first row to render.
	pub first: i32,
	/// One past the last row to render.
	pub one_past_last: i32,
}

impl UiVirtualRange {
	/// True when no rows are in range.
	pub fn is_empty(self) -> bool {
		self.one_past_last <= self.first
	}
}

// ── UiSplitConfig / UiSplitRegion ────────────────────────────────────────────

impl Default for UiSplitConfig {
	fn default() -> Self {
		Self {
			direction: super::types::UiDirection::Row,
			handle_size: 4.0,
			minimum_first: 60.0,
			minimum_second: 60.0,
			keyboard_step: 8.0,
		}
	}
}

/// Return value from [`crate::ui::Ui::split_pane`].
///
/// Donor: `oa::UiSplitRegion`.
#[derive(Clone, Debug, Default)]
pub struct UiSplitRegion {
	/// Bounds of the first (top or left) pane.
	pub first: PixelRect,
	/// Bounds of the drag handle.
	pub handle: PixelRect,
	/// Bounds of the second (bottom or right) pane.
	pub second: PixelRect,
	/// Whether the ratio changed this frame.
	pub changed: bool,
}

// ── UiPropertyRowConfig / UiPropertyRegion ───────────────────────────────────

impl Default for UiPropertyRowConfig {
	fn default() -> Self {
		Self {
			label_fraction: 0.40,
			gap: 4.0,
			padding_x: 4.0,
			alternate: false,
		}
	}
}

/// Return value from [`crate::ui::Ui::property_row`].
///
/// Donor: `oa::UiPropertyRegion`.
#[derive(Clone, Debug, Default)]
pub struct UiPropertyRegion {
	/// Full row rectangle.
	pub row: PixelRect,
	/// Label sub-rectangle.
	pub label: PixelRect,
	/// Value sub-rectangle.
	pub value: PixelRect,
}

// ── UiPopupConfig ────────────────────────────────────────────────────────────

impl Default for UiPopupConfig {
	fn default() -> Self {
		Self {
			width: 160.0,
			height: 0.0,
			gap: 2.0,
			padding: 6.0,
		}
	}
}

// ── UiDropdownConfig ─────────────────────────────────────────────────────────

impl Default for UiDropdownConfig {
	fn default() -> Self {
		Self {
			max_visible_items: 8,
			popup_width: 0.0,
			popup_gap: 2.0,
		}
	}
}

// ── UiTooltipConfig ──────────────────────────────────────────────────────────

impl Default for UiTooltipConfig {
	fn default() -> Self {
		Self {
			delay_ms: 400.0,
			max_width: 240.0,
			gap: 8.0,
			padding: 6.0,
		}
	}
}

// ── UiHeatmapConfig ──────────────────────────────────────────────────────────

impl Default for UiHeatmapConfig {
	fn default() -> Self {
		Self {
			rows: 1,
			cols: 1,
			v_min: 0.0,
			v_max: 1.0,
			colormap: 1,
			auto_scale: true,
			show_grid: false,
		}
	}
}

// ── UiGridConfig ─────────────────────────────────────────────────────────────

impl Default for UiGridConfig {
	fn default() -> Self {
		let automatic = Color::new(0.0, 0.0, 0.0, 0.0);
		Self {
			origin: [0.0; 2],
			minor_spacing: [10.0; 2],
			major_every: 10,
			super_major_every: 100,
			minor_thickness: 0.65,
			major_thickness: 0.90,
			super_major_thickness: 1.0,
			axis_thickness: 1.25,
			opacity: 1.0,
			guide_fade_center: [0.0; 2],
			guide_fade_inner_radius: 0.0,
			guide_fade_outer_radius: 1.0,
			background_top: automatic,
			background_bottom: automatic,
			minor_color: automatic,
			major_color: automatic,
			super_major_color: automatic,
			axis_color: automatic,
			fill_background: true,
			draw_grid: true,
			draw_axes: true,
			radial_guide_fade: false,
			dither_background: false,
		}
	}
}

// ── UiPlotConfig ─────────────────────────────────────────────────────────────

impl Default for UiPlotConfig {
	fn default() -> Self {
		Self {
			color: Color::new(0.388, 0.400, 0.945, 1.0),
			x_min: 0.0,
			x_max: 1.0,
			y_min: 0.0,
			y_max: 1.0,
			auto_scale: true,
			show_grid: true,
			fill: false,
			antialias_samples: 4,
			line_width: 1.35,
			draw_surface: true,
		}
	}
}

// ── UiIcon ───────────────────────────────────────────────────────────────────

/// Named icon for icon-button widgets.
///
/// Donor: `oa::UiIcon`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum UiIcon {
	#[default]
	Previous,
	Next,
	Play,
	Pause,
	Menu,
	Minimize,
	Maximize,
	Restore,
	Close,
	Fullscreen,
	ExitFullscreen,
	Volume,
	Muted,
	Settings,
}

/// Display style for an icon button.
///
/// Donor: `oa::UiIconButtonStyle`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum UiIconButtonStyle {
	/// Semi-transparent overlay (e.g. video transport controls).
	#[default]
	Overlay,
	/// Inline header button.
	Header,
	/// Window chrome control (min/max/close).
	WindowControl,
}

/// Direction for a chevron button.
///
/// Donor: `oa::UiChevronDirection`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum UiChevronDirection {
	#[default]
	Previous,
	Next,
}

// ── UiAccessibility ──────────────────────────────────────────────────────────

/// Accessibility role for a widget.
///
/// Donor: `oa::UiAccessibilityRole`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum UiAccessibilityRole {
	#[default]
	Button,
	Checkbox,
	Slider,
	TextField,
	ComboBox,
	MenuItem,
	Tab,
	TreeItem,
	Splitter,
	Timeline,
}

/// Opaque accessibility snapshot — a flat list of node descriptors.
///
/// Donor: `oa::Ui::accessibilitySnapshot` return.
#[derive(Clone, Debug, Default)]
pub struct UiAccessibilitySnapshot {
	/// Number of nodes in the snapshot.
	pub node_count: usize,
}
