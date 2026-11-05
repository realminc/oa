//! Retained GPU-composed UI layer.
//!
//! `Ui` accumulates draw commands during `begin_frame` / `end_frame` and
//! replays them into a private R8G8B8A8_UNORM compose image via compute
//! dispatches in `record_render`.  The compose image is blitted to the
//! swapchain by the existing `Presenter::present_texture` path.
//!
//! Call pattern (every frame):
//!
//! ```rust,ignore
//! ui.begin_frame(delta_ms);
//! ui.begin_panel("panel", rect);
//!   if ui.button("Run") { /* … */ }
//!   ui.slider_f32("LR", &mut lr, 1e-5, 1e-3);
//!   ui.plot_line("loss", &loss_data);
//! ui.end_panel();
//! ui.end_frame();
//! let rgba = ui.render_rgba8()?; // explicit headless observation
//! ```
//!
//! Donor reference: `oa/ui/ui.h`, `oa/ui/ui.cpp`

pub(crate) mod command;
mod compose_image;
mod cv;
mod detection_overlay;
mod icons;
mod image;
mod layout;
mod pipeline;
mod text;
#[path = "ui/type.gen.rs"]
mod type_gen;
#[path = "ui/type.rs"]
pub mod types;
mod viewport;

use crate::Color;
use crate::input::{InputEvent, InputEventKind, InputState, Key, MotionSpeed};
use crate::{
	Error, Result,
	runtime::{EngineHandle, SyncCommands},
};
use command::{UiCmd, UiCmdList, clip_words, div_ceil, visible_blit_region};
use compose_image::ComposeImage;
pub use cv::{CvBbox, CvFrame, CvOverlay};
pub use detection_overlay::{DetectionOverlay, DetectionOverlayItem};
pub use image::{IMAGE_MAX_PLANES, ImageDtype, ImagePlanes};
use layout::{LayoutStack, PanelScope, default_item_height, resolve_width, separator_rect};
use pipeline::{UiFrameBindings, UiPipelines};
pub use text::{FontId, GlyphBuffer, GlyphInstance, PositionedGlyph, TextAtlas, TextLayout};
pub use type_gen::{
	CvBboxesConfig, DetectionOverlayConfig, GlyphInfo, TextLayoutConfig, UiDropdownConfig,
	UiGridConfig, UiHeatmapConfig, UiPlotConfig, UiPopupConfig, UiPropertyRowConfig, UiScrollConfig,
	UiSplitConfig, UiTabBarResult, UiTooltipConfig, UiTreeRowConfig, UiTreeRowResult,
};
pub use types::{
	NodeCanvas, NodeCanvasGrid, NodeCanvasHit, NodeCanvasHitItem, NodeCanvasState, PixelRect,
	UiAccessibilityRole, UiAccessibilitySnapshot, UiAlign, UiChevronDirection, UiDirection, UiEdge,
	UiIcon, UiIconButtonStyle, UiLayout, UiPropertyRegion, UiScrollRegion, UiSizing, UiSplitRegion,
	UiStyle, UiTabBarState, UiTabItem, UiVirtualRange, WorldAabb,
};
pub use viewport::{RenderLayer, ScissorDesc, Viewport, ViewportDesc, ViewportMode};

// ── Ui ────────────────────────────────────────────────────────────────────────

/// GPU-retained UI compositor.
///
/// Borrows one `Engine` for its lifetime; must be closed explicitly before
/// the engine or any resource it references is released.
pub struct Ui<'engine> {
	engine: &'engine crate::Engine,
	pub(crate) state: Option<UiState>,
}

struct WidgetMotion {
	start: f32,
	current: f32,
	target: f32,
	elapsed_ms: f32,
	last_seen: u64,
}

pub(crate) struct UiState {
	pub(crate) engine_handle: EngineHandle,
	/// Immutable core/KHR dispatch table; Engine retains the device lifetime.
	sync_commands: SyncCommands,
	pipelines: std::rc::Rc<UiPipelines>,
	compose: std::rc::Rc<ComposeImage>,
	text_atlas: std::rc::Rc<TextAtlas>,
	style_stack: Vec<UiStyle>,
	layout: LayoutStack,
	pub(crate) cmds: UiCmdList,
	input: InputState,
	content_scale: f32,
	motion_speed: MotionSpeed,
	frame_delta_ms: f32,
	frame_index: u64,
	widget_motion: std::collections::HashMap<u64, WidgetMotion>,
	/// Interaction state: focused widget id hash.
	focused: u64,
	/// Hovered widget id hash (set during begin_frame routing).
	hovered: u64,
	/// Pressed widget id hash (button/slider drag).
	pressed: u64,
	/// Sources referenced by commands being assembled for the current frame.
	frame_matrices: Vec<crate::Matrix>,
	frame_textures: Vec<crate::Texture>,
	frame_storages: Vec<crate::runtime::Storage>,
	/// Whether we are inside a begin_frame / end_frame pair.
	in_frame: bool,
	/// Text characters queued by TextInput events for the focused widget.
	input_pending_chars: Vec<char>,
	/// Backspace presses queued for the focused widget.
	input_pending_backspace: u32,
	/// First deferred recording error from infallible widget convenience APIs.
	recording_error: Option<Error>,
	/// Id hash of the currently open popup, or 0 for none.
	popup_open: u64,
	/// Anchor rect of the currently open popup.
	popup_anchor: PixelRect,
	/// Scroll offset keyed by panel-id hash.
	scroll_offsets: std::collections::HashMap<u64, i32>,
}

impl<'engine> Ui<'engine> {
	/// Initialise the UI layer.  Creates pipelines and the compose image.
	///
	/// `width` and `height` should match the presentation surface extent.
	pub fn init(engine: &'engine crate::Engine, width: u32, height: u32) -> Result<Self> {
		Self::init_with_style(engine, width, height, UiStyle::default())
	}

	/// Initialise with an explicit style theme.
	pub fn init_with_style(
		engine: &'engine crate::Engine,
		width: u32,
		height: u32,
		style: UiStyle,
	) -> Result<Self> {
		if width == 0 || height == 0 {
			return Err(Error::invalid_argument(
				"Ui compose extent must be non-zero",
			));
		}
		let handle = engine.handle();
		handle.require_ui_bindings()?;
		let compose = {
			let device = handle.raw_device();
			// The pipeline layout is the engine's shared bindless layout.
			let layout = handle.ui_pipeline_layout();
			let pipelines = std::rc::Rc::new(handle.with_pipeline_cache(|cache| {
				UiPipelines::new(&device, layout, handle.ui_uses_bounded_bindings(), cache)
			})?);
			let compose = std::rc::Rc::new(ComposeImage::new(&handle, width, height)?);
			let text_atlas = std::rc::Rc::new(TextAtlas::new(&handle)?);
			UiState {
				sync_commands: handle.ui_sync_commands(),
				engine_handle: handle,
				pipelines,
				compose,
				text_atlas,
				style_stack: vec![style],
				layout: LayoutStack::default(),
				cmds: UiCmdList::default(),
				input: InputState::default(),
				content_scale: 1.0,
				motion_speed: MotionSpeed::Normal,
				frame_delta_ms: 0.0,
				frame_index: 0,
				widget_motion: std::collections::HashMap::new(),
				focused: 0,
				hovered: 0,
				pressed: 0,
				frame_matrices: Vec::new(),
				frame_textures: Vec::new(),
				frame_storages: Vec::new(),
				in_frame: false,
				input_pending_chars: Vec::new(),
				input_pending_backspace: 0,
				recording_error: None,
				popup_open: 0,
				popup_anchor: PixelRect::default(),
				scroll_offsets: std::collections::HashMap::new(),
			}
		};
		Ok(Self {
			engine,
			state: Some(compose),
		})
	}

	/// Recreate the compose image after a surface resize.
	pub fn resize(&mut self, width: u32, height: u32) -> Result<()> {
		if width == 0 || height == 0 {
			return Err(Error::invalid_argument(
				"Ui compose extent must be non-zero",
			));
		}
		let state = self
			.state
			.as_mut()
			.ok_or_else(|| Error::failed_precondition("Ui is closed"))?;
		let new_compose = ComposeImage::new(&state.engine_handle, width, height)?;
		state.compose = std::rc::Rc::new(new_compose);
		Ok(())
	}

	/// Logical-to-pixel scale used by donor viewer controls.
	pub fn content_scale(&self) -> f32 {
		self.state.as_ref().map_or(1.0, |state| state.content_scale)
	}

	/// Set the logical-to-framebuffer scale for controls and symbolic icons.
	pub fn set_content_scale(&mut self, scale: f32) -> Result<()> {
		if !scale.is_finite() || scale <= 0.0 {
			return Err(Error::invalid_argument(
				"Ui content scale must be finite and positive",
			));
		}
		let state = self
			.state
			.as_mut()
			.ok_or_else(|| Error::failed_precondition("Ui is closed"))?;
		state.content_scale = scale;
		Ok(())
	}

	/// Select donor widget transition timing without recreating GPU resources.
	pub fn set_motion_speed(&mut self, speed: MotionSpeed) -> Result<()> {
		let state = self
			.state
			.as_mut()
			.ok_or_else(|| Error::failed_precondition("Ui is closed"))?;
		if state.motion_speed != speed {
			state.motion_speed = speed;
			for motion in state.widget_motion.values_mut() {
				motion.start = motion.current;
				motion.elapsed_ms = 0.0;
			}
		}
		Ok(())
	}

	// ── Per-frame ──────────────────────────────────────────────────────────────

	/// Begin a new frame.  Must be balanced by `end_frame`.
	pub fn begin_frame(&mut self, delta_ms: f32) {
		let Some(state) = self.state.as_mut() else {
			return;
		};
		state.frame_delta_ms = if delta_ms.is_finite() {
			delta_ms.clamp(0.0, 1000.0)
		} else {
			0.0
		};
		state.frame_index = state.frame_index.wrapping_add(1);
		state
			.widget_motion
			.retain(|_, motion| state.frame_index.wrapping_sub(motion.last_seen) <= 600);
		state.frame_matrices.clear();
		state.frame_textures.clear();
		state.frame_storages.clear();
		state.cmds.clear_all();
		state.hovered = 0;
		state.in_frame = true;
		// Emit a clear command to start each frame from a blank compose image.
		let bg = state
			.style_stack
			.last()
			.map_or(Color::new(0.0, 0.0, 0.0, 1.0), |s| s.background);
		state.cmds.push(UiCmd::Clear { color: bg });
	}

	/// Route a platform event.  Returns `true` if a widget consumed it.
	pub fn route_event(&mut self, event: &InputEvent) -> bool {
		let Some(state) = self.state.as_mut() else {
			return false;
		};
		match event.kind {
			InputEventKind::MouseMove => {
				state.input.pointer_x = event.mouse_x;
				state.input.pointer_y = event.mouse_y;
				state.input.mouse_x = event.mouse_x;
				state.input.mouse_y = event.mouse_y;
				state.input.mouse_dx = event.mouse_dx;
				state.input.mouse_dy = event.mouse_dy;
			}
			InputEventKind::MouseDown => {
				state.input.pointer_x = event.mouse_x;
				state.input.pointer_y = event.mouse_y;
				state.input.mouse_x = event.mouse_x;
				state.input.mouse_y = event.mouse_y;
				if event.button == 1 {
					state.input.l_button = true;
					state.input.pointer_down = true;
					state.input.l_pressed = true;
					state.input.l_click_count = event.click_count;
				} else if event.button == 2 {
					state.input.m_button = true;
				} else if event.button == 3 {
					state.input.r_button = true;
				}
				// If the pressed widget matches hovered, consume.
				if state.hovered != 0 && event.button == 1 {
					state.pressed = state.hovered;
					return true;
				}
			}
			InputEventKind::MouseUp => {
				state.input.pointer_x = event.mouse_x;
				state.input.pointer_y = event.mouse_y;
				if event.button == 1 {
					state.input.l_button = false;
					state.input.pointer_down = false;
					state.input.l_released = true;
				} else if event.button == 2 {
					state.input.m_button = false;
				} else if event.button == 3 {
					state.input.r_button = false;
				}
			}
			InputEventKind::MouseScroll => {
				state.input.scroll_x += event.scroll_x;
				state.input.scroll_y += event.scroll_y;
			}
			InputEventKind::KeyDown => {
				state.input.modifiers = event.modifiers;
				state.input.shift = event.shift();
				state.input.ctrl = event.ctrl();
				if event.key == Key::Tab {
					// Tab focus advance — handled implicitly by widget hashing.
					return state.focused != 0;
				}
				if event.key == Key::Backspace && state.focused != 0 {
					state.input_pending_backspace += 1;
					return true;
				}
				if event.key == Key::Escape {
					state.focused = 0;
				}
			}
			InputEventKind::KeyChar => {
				// Committed UTF-8 text from the platform IME / SDL TextInput event.
				if state.focused != 0 && !event.text.is_empty() {
					state.input_pending_chars.extend(event.text.chars());
					return true;
				}
			}
			InputEventKind::KeyUp => {
				state.input.modifiers = event.modifiers;
				state.input.shift = event.shift();
				state.input.ctrl = event.ctrl();
			}
			InputEventKind::WindowResize => {}
			_ => {}
		}
		false
	}

	/// End the current frame.  All panel scopes must have been closed.
	pub fn end_frame(&mut self) {
		let Some(state) = self.state.as_mut() else {
			return;
		};
		state.in_frame = false;
		if state.input.l_released {
			state.pressed = 0;
		}
		state.input.l_pressed = false;
		state.input.l_released = false;
		state.input.scroll_x = 0.0;
		state.input.scroll_y = 0.0;
		state.input_pending_chars.clear();
		state.input_pending_backspace = 0;
	}

	/// Current input snapshot.
	pub fn input(&self) -> &InputState {
		self.state.as_ref().map_or(&InputState::EMPTY, |s| &s.input)
	}

	// ── Style ──────────────────────────────────────────────────────────────────

	pub fn push_style(&mut self, style: UiStyle) {
		let Some(state) = self.state.as_mut() else {
			return;
		};
		state.style_stack.push(style);
	}

	pub fn pop_style(&mut self) {
		let Some(state) = self.state.as_mut() else {
			return;
		};
		if state.style_stack.len() > 1 {
			state.style_stack.pop();
		}
	}

	fn current_style_ref(state: &UiState) -> &UiStyle {
		state.style_stack.last().expect("style stack is non-empty")
	}

	/// Resolve the effective draw clip for `rect`: the current panel scope's
	/// clip intersected with `rect` and the compose bounds.
	///
	/// Donor: `oa::Ui::Impl::clipFor`.
	fn compose_clip(state: &UiState, rect: PixelRect) -> PixelRect {
		let bounds = PixelRect {
			x: 0,
			y: 0,
			w: state.compose.width,
			h: state.compose.height,
		};
		let clipped = bounds.clip(&rect);
		state
			.layout
			.top()
			.map_or(clipped, |scope| scope.clip().clip(&clipped))
	}

	/// Return the current ambient widget style.
	///
	/// Donor: `oa::Ui::currentStyle`.
	#[must_use]
	pub fn current_style(&self) -> UiStyle {
		self.state.as_ref().map_or_else(UiStyle::default, |state| {
			Self::current_style_ref(state).clone()
		})
	}

	// ── Layout ─────────────────────────────────────────────────────────────────

	/// Begin a new panel scope.  Must be balanced by `end_panel`.
	pub fn begin_panel(&mut self, _id: &str, rect: PixelRect) {
		self.begin_panel_with_layout(_id, rect, UiLayout::default());
	}

	/// Begin a panel with an explicit layout descriptor.
	pub fn begin_panel_with_layout(&mut self, _id: &str, rect: PixelRect, layout: UiLayout) {
		let Some(state) = self.state.as_mut() else {
			return;
		};
		let style = Self::current_style_ref(state).clone();
		let clip = state
			.layout
			.top()
			.map_or(rect, |parent| parent.clip().clip(&rect));
		// Draw panel background.
		state.cmds.push(UiCmd::Rect {
			rect,
			color: style.background,
			clip,
			corner_radius: style.corner_radius,
		});
		let mut scope = PanelScope::new(rect, layout);
		scope.clip_override = Some(clip);
		state.layout.push(scope);
	}

	/// End the innermost panel scope.
	pub fn end_panel(&mut self) {
		let Some(state) = self.state.as_mut() else {
			return;
		};
		state.layout.pop();
	}

	/// Explicit spacing in the current flow direction.
	pub fn spacing(&mut self, pixels: f32) {
		let Some(state) = self.state.as_mut() else {
			return;
		};
		if let Some(scope) = state.layout.top_mut() {
			match scope.layout.direction {
				UiDirection::Column => scope.cursor[1] += pixels as i32,
				UiDirection::Row => scope.cursor[0] += pixels as i32,
			}
		}
	}

	/// One-pixel separator line in the current flow direction.
	pub fn separator(&mut self) {
		let Some(state) = self.state.as_mut() else {
			return;
		};
		let (sep_rect, sep_clip) = {
			let Some(scope) = state.layout.top() else {
				return;
			};
			(separator_rect(scope), scope.clip())
		};
		let color = Self::current_style_ref(state).border;
		state.cmds.push(UiCmd::Rect {
			rect: sep_rect,
			color,
			clip: sep_clip,
			corner_radius: 0.0,
		});
		if let Some(scope) = state.layout.top_mut() {
			scope.allocate(sep_rect.w, sep_rect.h);
		}
	}

	/// Begin an explicit row (horizontal flow) scope.
	pub fn begin_row(&mut self) {
		let Some(state) = self.state.as_mut() else {
			return;
		};
		if let Some(parent) = state.layout.top() {
			let row_rect = PixelRect {
				x: parent.cursor[0],
				y: parent.cursor[1],
				w: parent.inner.w,
				h: 0,
			};
			let row_layout = UiLayout {
				direction: UiDirection::Row,
				..UiLayout::default()
			};
			state.layout.push(PanelScope::new(row_rect, row_layout));
		}
	}

	/// End the innermost row scope and advance the parent cursor by row height.
	pub fn end_row(&mut self) {
		let Some(state) = self.state.as_mut() else {
			return;
		};
		let row_h = state.layout.top().map_or(0, |s| s.row_width);
		state.layout.pop();
		if let Some(parent) = state.layout.top_mut() {
			parent.cursor[1] += row_h as i32 + parent.layout.gap as i32;
		}
	}

	// ── Widgets ────────────────────────────────────────────────────────────────

	/// Button.  Returns `true` on click (pointer up inside).
	pub fn button(&mut self, label: &str) -> bool {
		let Some(state) = self.state.as_mut() else {
			return false;
		};
		let style = Self::current_style_ref(state).clone();
		let id = widget_hash(label, state.layout.top().map(|s| s.outer));

		let (rect, clip) = allocate_widget(state, style.padding);

		// Hit test.
		let hovered = point_in_rect(state.input.pointer_x, state.input.pointer_y, rect);
		let activated = hovered && was_pressed_and_released(state, id);

		if hovered {
			state.hovered = id;
		}

		// Background.
		let bg = if state.pressed == id {
			Color::new(
				style.accent.r * 0.7,
				style.accent.g * 0.7,
				style.accent.b * 0.7,
				1.0,
			)
		} else if hovered {
			style.surface
		} else {
			Color::new(
				style.surface.r * 0.85,
				style.surface.g * 0.85,
				style.surface.b * 0.85,
				1.0,
			)
		};
		state.cmds.push(UiCmd::Rect {
			rect,
			color: bg,
			clip,
			corner_radius: style.corner_radius,
		});
		state.cmds.push(UiCmd::RectOutline {
			rect,
			color: style.border,
			thickness: 1,
			clip,
			corner_radius: style.corner_radius,
		});
		let text_config = TextLayoutConfig {
			font: FontId::SansSemibold,
			size: style.font_size,
			..TextLayoutConfig::default()
		};
		Self::emit_text_or_record(state, label, rect, style.text, text_config);

		activated
	}

	/// Checkbox.  Returns `true` when the value changes.
	pub fn checkbox(&mut self, label: &str, value: &mut bool) -> bool {
		let Some(state) = self.state.as_mut() else {
			return false;
		};
		let style = Self::current_style_ref(state).clone();
		let id = widget_hash(label, state.layout.top().map(|s| s.outer));

		let (rect, clip) = allocate_widget(state, style.padding);
		let hovered = point_in_rect(state.input.pointer_x, state.input.pointer_y, rect);
		let activated = hovered && was_pressed_and_released(state, id);
		if activated {
			*value = !*value;
		}
		if hovered {
			state.hovered = id;
		}

		let bg = if *value { style.accent } else { style.surface };
		state.cmds.push(UiCmd::Rect {
			rect,
			color: bg,
			clip,
			corner_radius: style.corner_radius,
		});
		state.cmds.push(UiCmd::RectOutline {
			rect,
			color: style.border,
			thickness: 1,
			clip,
			corner_radius: style.corner_radius,
		});
		let text_config = TextLayoutConfig {
			font: FontId::Sans,
			size: style.font_size,
			..TextLayoutConfig::default()
		};
		Self::emit_text_or_record(state, label, rect, style.text, text_config);
		activated
	}

	/// F32 slider.  Returns `true` when the value changes.
	pub fn slider_f32(&mut self, label: &str, value: &mut f32, min: f32, max: f32) -> bool {
		if min >= max {
			return false;
		}
		let Some(state) = self.state.as_mut() else {
			return false;
		};
		let style = Self::current_style_ref(state).clone();
		let id = widget_hash(label, state.layout.top().map(|s| s.outer));

		let (rect, clip) = allocate_widget(state, style.padding);
		let hovered = point_in_rect(state.input.pointer_x, state.input.pointer_y, rect);
		if hovered {
			state.hovered = id;
		}

		let mut changed = false;
		if state.pressed == id && state.input.pointer_down {
			let fraction =
				((state.input.pointer_x - rect.x as f32) / rect.w.max(1) as f32).clamp(0.0, 1.0);
			let new_val = min + fraction * (max - min);
			if (*value - new_val).abs() > 1e-6 {
				*value = new_val;
				changed = true;
			}
		}
		if hovered && state.input.pointer_down {
			state.pressed = id;
		}

		// Track background.
		state.cmds.push(UiCmd::Rect {
			rect,
			color: style.surface,
			clip,
			corner_radius: style.corner_radius,
		});
		// Fill.
		let fraction = ((*value - min) / (max - min)).clamp(0.0, 1.0);
		let fill_w = (rect.w as f32 * fraction) as u32;
		if fill_w > 0 {
			let fill_rect = PixelRect {
				x: rect.x,
				y: rect.y,
				w: fill_w,
				h: rect.h,
			};
			state.cmds.push(UiCmd::Rect {
				rect: fill_rect,
				color: style.accent,
				clip,
				corner_radius: style.corner_radius,
			});
		}
		let text = format!("{label}: {:.3}", *value);
		let text_config = TextLayoutConfig {
			font: FontId::Mono,
			size: style.font_size - 1.0,
			..TextLayoutConfig::default()
		};
		Self::emit_text_or_record(state, &text, rect, style.text, text_config);
		changed
	}

	/// Progress bar with a normalized fraction `[0, 1]`.
	pub fn progress_bar(&mut self, fraction: f32) {
		let overlay = if fraction.is_finite() {
			format!("{:.0}%", fraction.clamp(0.0, 1.0) * 100.0)
		} else {
			String::new()
		};
		self.progress_bar_with_overlay(fraction, &overlay);
	}

	/// Progress bar with an explicit text overlay.
	pub fn progress_bar_with_overlay(&mut self, fraction: f32, overlay: &str) {
		let Some(state) = self.state.as_mut() else {
			return;
		};
		let style = Self::current_style_ref(state).clone();
		let (rect, clip) = allocate_widget(state, style.padding);
		// Background.
		state.cmds.push(UiCmd::Rect {
			rect,
			color: style.surface,
			clip,
			corner_radius: style.corner_radius,
		});
		// Fill.
		let fill_w = (rect.w as f32 * fraction.clamp(0.0, 1.0)) as u32;
		if fill_w > 0 {
			let fill_rect = PixelRect {
				x: rect.x,
				y: rect.y,
				w: fill_w,
				h: rect.h,
			};
			state.cmds.push(UiCmd::Rect {
				rect: fill_rect,
				color: style.accent,
				clip,
				corner_radius: style.corner_radius,
			});
		}
		let text_config = TextLayoutConfig {
			font: FontId::Sans,
			size: style.font_size - 1.0,
			..TextLayoutConfig::default()
		};
		Self::emit_text_or_record(state, overlay, rect, style.text, text_config);
	}

	// ── Additional widgets ────────────────────────────────────────────────────

	/// Passive text label in the current panel flow.
	///
	/// Does not interact; advances the cursor by one item height.
	/// Donor: `oa::Ui::label`.
	pub fn label(&mut self, text: &str) {
		let Some(state) = self.state.as_mut() else {
			return;
		};
		let style = Self::current_style_ref(state).clone();
		let (rect, _clip) = allocate_widget(state, style.padding);
		if rect.w == 0 || rect.h == 0 {
			return;
		}
		let cfg = TextLayoutConfig {
			font: FontId::SansSemibold,
			size: style.font_size,
			..TextLayoutConfig::default()
		};
		Self::emit_text_or_record(state, text, rect, style.text_secondary, cfg);
	}

	/// Integer slider.  Returns `true` when the value changes.
	///
	/// Donor: `oa::Ui::sliderI32`.
	pub fn slider_i32(&mut self, label: &str, value: &mut i32, min: i32, max: i32) -> bool {
		if min >= max {
			return false;
		}
		let Some(state) = self.state.as_mut() else {
			return false;
		};
		let style = Self::current_style_ref(state).clone();
		let id = widget_hash(label, state.layout.top().map(|s| s.outer));

		let (rect, clip) = allocate_widget(state, style.padding);
		let hovered = point_in_rect(state.input.pointer_x, state.input.pointer_y, rect);
		if hovered {
			state.hovered = id;
		}

		let mut changed = false;
		if state.pressed == id && state.input.pointer_down {
			let fraction =
				((state.input.pointer_x - rect.x as f32) / rect.w.max(1) as f32).clamp(0.0, 1.0);
			let new_val = min + (fraction * (max - min) as f32).round() as i32;
			let new_val = new_val.clamp(min, max);
			if *value != new_val {
				*value = new_val;
				changed = true;
			}
		}
		if hovered && state.input.pointer_down {
			state.pressed = id;
		}

		// Track.
		state.cmds.push(UiCmd::Rect {
			rect,
			color: style.surface,
			clip,
			corner_radius: style.corner_radius,
		});
		// Fill.
		let fraction = ((*value - min) as f32 / (max - min) as f32).clamp(0.0, 1.0);
		let fill_w = (rect.w as f32 * fraction) as u32;
		if fill_w > 0 {
			state.cmds.push(UiCmd::Rect {
				rect: PixelRect {
					x: rect.x,
					y: rect.y,
					w: fill_w,
					h: rect.h,
				},
				color: style.accent,
				clip,
				corner_radius: style.corner_radius,
			});
		}
		let text = format!("{label}: {}", *value);
		let text_config = TextLayoutConfig {
			font: FontId::Mono,
			size: style.font_size - 1.0,
			..TextLayoutConfig::default()
		};
		Self::emit_text_or_record(state, &text, rect, style.text, text_config);
		changed
	}

	/// Single-line text input field.  Returns `true` when `value` changes.
	///
	/// This implementation records the field chrome; keyboard entry is handled
	/// when the field is focused (last clicked).  Text is committed on every
	/// character via SDL text-input events routed through `route_event`.
	///
	/// Donor: `oa::Ui::inputText`.
	pub fn input_text(&mut self, label: &str, value: &mut String) -> bool {
		let Some(state) = self.state.as_mut() else {
			return false;
		};
		let style = Self::current_style_ref(state).clone();
		let id = widget_hash(label, state.layout.top().map(|s| s.outer));

		let (rect, clip) = allocate_widget(state, style.padding);
		let hovered = point_in_rect(state.input.pointer_x, state.input.pointer_y, rect);
		if hovered && state.input.l_pressed {
			state.focused = id;
		}
		let focused = state.focused == id;

		// Background — brighter when focused.
		let bg = if focused {
			style.surface
		} else {
			Color::new(
				style.surface.r * 0.75,
				style.surface.g * 0.75,
				style.surface.b * 0.75,
				1.0,
			)
		};
		state.cmds.push(UiCmd::Rect {
			rect,
			color: bg,
			clip,
			corner_radius: style.corner_radius,
		});
		let border_color = if focused { style.accent } else { style.border };
		state.cmds.push(UiCmd::RectOutline {
			rect,
			color: border_color,
			thickness: 1,
			clip,
			corner_radius: style.corner_radius,
		});

		// Text content.
		let cfg = TextLayoutConfig {
			font: FontId::Mono,
			size: style.font_size - 1.0,
			..TextLayoutConfig::default()
		};
		let inner = PixelRect {
			x: rect.x + style.padding as i32,
			y: rect.y,
			w: rect.w.saturating_sub(style.padding as u32 * 2),
			h: rect.h,
		};
		Self::emit_text_or_record(state, value, inner, style.text, cfg);

		// Accumulate text input when focused.
		let mut changed = false;
		if focused {
			for ch in state.input_pending_chars.drain(..) {
				value.push(ch);
				changed = true;
			}
			if state.input_pending_backspace > 0 {
				for _ in 0..state.input_pending_backspace {
					value.pop();
				}
				changed = state.input_pending_backspace > 0;
			}
			state.input_pending_backspace = 0;
		}
		changed
	}

	/// Color swatch — a filled rectangle of the given color at the given size.
	///
	/// Placed inline in the current panel flow.  Does not return a value.
	/// Donor: `oa::Ui::colorSwatch`.
	pub fn color_swatch(&mut self, color: Color, size: [f32; 2]) {
		let Some(state) = self.state.as_mut() else {
			return;
		};
		let style = Self::current_style_ref(state).clone();
		let clip = state.layout.top().map(|s| s.clip()).unwrap_or_default();
		let Some(scope) = state.layout.top_mut() else {
			return;
		};
		let w = size[0].max(1.0) as u32;
		let h = size[1].max(1.0) as u32;
		let rect = scope.allocate(w, h);
		state.cmds.push(UiCmd::Rect {
			rect,
			color,
			clip,
			corner_radius: style.corner_radius,
		});
		state.cmds.push(UiCmd::RectOutline {
			rect,
			color: style.border,
			thickness: 1,
			clip,
			corner_radius: style.corner_radius,
		});
	}

	/// Tab bar widget.  Returns interaction results; mutates `state.selected`.
	///
	/// Donor: `oa::Ui::tabBar`.
	pub fn tab_bar(
		&mut self,
		_id: &str,
		rect: PixelRect,
		items: &[UiTabItem],
		state_out: &mut UiTabBarState,
	) -> UiTabBarResult {
		let mut result = UiTabBarResult {
			bar: rect,
			tabs: rect,
			first_visible: state_out.first_visible,
			one_past_last: items.len() as i32,
			activated_index: -1,
			close_requested_index: -1,
			selection_changed: false,
		};
		let Some(state) = self.state.as_mut() else {
			return result;
		};
		let style = Self::current_style_ref(state).clone();
		let clip = rect;

		// Bar background.
		state.cmds.push(UiCmd::Rect {
			rect,
			color: style.background,
			clip,
			corner_radius: 0.0,
		});
		// Bottom border.
		state.cmds.push(UiCmd::RectOutline {
			rect,
			color: style.border,
			thickness: 1,
			clip,
			corner_radius: 0.0,
		});

		if items.is_empty() {
			return result;
		}

		let tab_h = rect.h;
		let min_w: i32 = 88;
		let max_w: i32 = 220;
		let available_w = rect.w as i32;
		let n = items.len() as i32;
		let tab_w = (available_w / n).clamp(min_w, max_w);

		let mut x = rect.x;
		for (i, item) in items.iter().enumerate() {
			let i = i as i32;
			if i < state_out.first_visible {
				continue;
			}
			if x + tab_w > rect.x + rect.w as i32 {
				result.one_past_last = i;
				break;
			}
			let tab_rect = PixelRect {
				x,
				y: rect.y,
				w: tab_w as u32,
				h: tab_h,
			};
			let selected = state_out.selected == i;
			let hovered = point_in_rect(state.input.pointer_x, state.input.pointer_y, tab_rect);

			// Click → select.
			let id = widget_hash(&item.id, Some(rect));
			let clicked = hovered && was_pressed_and_released(state, id);
			if hovered {
				state.hovered = id;
			}
			if hovered && state.input.pointer_down {
				state.pressed = id;
			}
			if clicked && state_out.selected != i {
				state_out.selected = i;
				result.activated_index = i;
				result.selection_changed = true;
			}

			let bg = if selected {
				style.surface
			} else if hovered {
				Color::new(
					style.surface.r * 0.85,
					style.surface.g * 0.85,
					style.surface.b * 0.85,
					1.0,
				)
			} else {
				style.background
			};
			state.cmds.push(UiCmd::Rect {
				rect: tab_rect,
				color: bg,
				clip,
				corner_radius: 0.0,
			});
			if selected {
				// Accent bottom edge for selected tab.
				let sel_line = PixelRect {
					x: tab_rect.x,
					y: tab_rect.y + tab_rect.h as i32 - 2,
					w: tab_rect.w,
					h: 2,
				};
				state.cmds.push(UiCmd::Rect {
					rect: sel_line,
					color: style.accent,
					clip,
					corner_radius: 0.0,
				});
			}
			// Separator.
			state.cmds.push(UiCmd::RectOutline {
				rect: tab_rect,
				color: style.border,
				thickness: 1,
				clip,
				corner_radius: 0.0,
			});

			// Tab label text.
			let label_text = if item.dirty {
				format!("● {}", item.label)
			} else {
				item.label.clone()
			};
			let cfg = TextLayoutConfig {
				font: FontId::Sans,
				size: style.font_size - 1.0,
				..TextLayoutConfig::default()
			};
			let text_color = if item.enabled {
				style.text
			} else {
				style.text_secondary
			};
			Self::emit_text_or_record(state, &label_text, tab_rect, text_color, cfg);

			x += tab_w;
		}

		result
	}

	/// Tree row — one row in a hierarchical list.
	///
	/// Donor: `oa::Ui::treeRow`.
	pub fn tree_row(
		&mut self,
		id: &str,
		rect: PixelRect,
		label: &str,
		config: UiTreeRowConfig,
	) -> UiTreeRowResult {
		let mut result = UiTreeRowResult {
			row: rect,
			open: config.open,
			..UiTreeRowResult::default()
		};
		let Some(state) = self.state.as_mut() else {
			return result;
		};
		let style = Self::current_style_ref(state).clone();
		let clip = rect;
		let widget_id = widget_hash(id, Some(rect));

		let hovered = point_in_rect(state.input.pointer_x, state.input.pointer_y, rect);
		if hovered {
			state.hovered = widget_id;
		}
		let clicked = hovered && was_pressed_and_released(state, widget_id);
		if hovered && state.input.l_pressed {
			state.pressed = widget_id;
		}
		if clicked {
			result.activated = true;
		}

		// Background.
		let bg = if config.selected {
			Color::new(
				style.accent.r * 0.25,
				style.accent.g * 0.25,
				style.accent.b * 0.35,
				1.0,
			)
		} else if hovered {
			Color::new(
				style.surface.r * 1.1,
				style.surface.g * 1.1,
				style.surface.b * 1.1,
				1.0,
			)
		} else {
			Color::new(0.0, 0.0, 0.0, 0.0)
		};
		if bg.a > 0.0 {
			state.cmds.push(UiCmd::Rect {
				rect,
				color: bg,
				clip,
				corner_radius: 0.0,
			});
		}

		// Disclosure triangle.
		let indent = config.indent * config.depth;
		let disc_x = rect.x + indent;
		let disc_w = config.disclosure_width.max(0) as u32;
		let disc_rect = PixelRect {
			x: disc_x,
			y: rect.y,
			w: disc_w,
			h: rect.h,
		};

		if config.has_children {
			// Render a simple solid triangle as a small rect (GPU text support for
			// actual ▶/▼ glyphs not yet available in coverage atlas).
			let arrow_rect = PixelRect {
				x: disc_x + 4,
				y: rect.y + rect.h as i32 / 2 - 4,
				w: 8,
				h: 8,
			};
			let arrow_color = if config.open {
				style.text
			} else {
				style.text_secondary
			};
			state.cmds.push(UiCmd::Rect {
				rect: arrow_rect,
				color: arrow_color,
				clip,
				corner_radius: 1.0,
			});

			// Clicking the disclosure area toggles open.
			if clicked && point_in_rect(state.input.pointer_x, state.input.pointer_y, disc_rect) {
				result.open = !config.open;
				result.open_changed = true;
			}
		}
		result.disclosure = disc_rect;

		// Label text.
		let label_x = disc_x + disc_w as i32;
		let label_rect = PixelRect {
			x: label_x,
			y: rect.y,
			w: (rect.x + rect.w as i32 - label_x).max(0) as u32,
			h: rect.h,
		};
		result.label = label_rect;
		let text_color = if config.enabled {
			style.text
		} else {
			style.text_secondary
		};
		let cfg = TextLayoutConfig {
			font: FontId::Sans,
			size: style.font_size - 1.0,
			..TextLayoutConfig::default()
		};
		Self::emit_text_or_record(state, label, label_rect, text_color, cfg);

		result
	}

	/// Node-graph canvas background grid.
	///
	/// Renders a multi-level dot/line grid in `rect`, respecting the canvas
	/// pan/zoom from `canvas`.  Does not participate in panel flow (explicit
	/// rect).
	///
	/// Donor: `oa::Ui::nodeCanvasGrid`.
	pub fn node_canvas_grid(&mut self, canvas: &NodeCanvas, rect: PixelRect) {
		let Some(state) = self.state.as_mut() else {
			return;
		};
		let style = Self::current_style_ref(state).clone();

		// Canvas background.
		let bg = Color::new(
			style.background.r * 0.88,
			style.background.g * 0.88,
			style.background.b * 0.88,
			1.0,
		);
		state.cmds.push(UiCmd::Rect {
			rect,
			color: bg,
			clip: rect,
			corner_radius: 0.0,
		});

		// Grid lines.
		let zoom = canvas.state().zoom;
		let pan = canvas.state().pan;
		let base_step = 10.0_f32;
		let step_px = base_step * zoom;
		if step_px < 4.0 {
			return; // too dense, skip
		}

		// Compute world coordinate of rect top-left corner.
		let world_origin = canvas.screen_to_world([rect.x as f32, rect.y as f32]);

		// Align to grid.
		let start_wx = (world_origin[0] / base_step).floor() * base_step;
		let start_wy = (world_origin[1] / base_step).floor() * base_step;

		let minor_color = Color::new(style.border.r, style.border.g, style.border.b, 0.35);
		let major_color = Color::new(style.border.r, style.border.g, style.border.b, 0.65);

		// Vertical lines.
		let mut wx = start_wx;
		loop {
			let sx = canvas.world_to_screen([wx, 0.0])[0];
			if sx > (rect.x + rect.w as i32) as f32 {
				break;
			}
			if sx >= rect.x as f32 {
				let is_major = (wx / base_step).abs() % 10.0 < 0.5;
				let color = if is_major { major_color } else { minor_color };
				let line = PixelRect {
					x: sx as i32,
					y: rect.y,
					w: 1,
					h: rect.h,
				};
				state.cmds.push(UiCmd::Rect {
					rect: line,
					color,
					clip: rect,
					corner_radius: 0.0,
				});
			}
			wx += base_step;
			if wx - start_wx > (rect.w as f32 / zoom + base_step * 2.0) {
				break;
			}
		}

		// Horizontal lines.
		let _ = pan; // already incorporated through world_to_screen
		let mut wy = start_wy;
		loop {
			let sy = canvas.world_to_screen([0.0, wy])[1];
			if sy > (rect.y + rect.h as i32) as f32 {
				break;
			}
			if sy >= rect.y as f32 {
				let is_major = (wy / base_step).abs() % 10.0 < 0.5;
				let color = if is_major { major_color } else { minor_color };
				let line = PixelRect {
					x: rect.x,
					y: sy as i32,
					w: rect.w,
					h: 1,
				};
				state.cmds.push(UiCmd::Rect {
					rect: line,
					color,
					clip: rect,
					corner_radius: 0.0,
				});
			}
			wy += base_step;
			if wy - start_wy > (rect.h as f32 / zoom + base_step * 2.0) {
				break;
			}
		}
	}

	/// Single-series GPU line plot of a CPU F32 slice.
	///
	/// The data must already be uploaded into the bindless heap as a storage
	/// buffer.  Callers uploading host data should use the engine's buffer
	/// upload path and pass the resulting bindless index.
	pub fn plot_line_bindless(
		&mut self,
		values_idx: u32,
		count: u32,
		y_min: f32,
		y_max: f32,
		color: Color,
	) {
		let Some(state) = self.state.as_mut() else {
			return;
		};
		let style = Self::current_style_ref(state).clone();
		let (rect, clip) = allocate_plot(state, style.padding);
		state.cmds.push(UiCmd::PlotLine {
			values_idx,
			count,
			rect,
			clip,
			x_min: 0.0,
			x_max: count.saturating_sub(1).max(1) as f32,
			y_min,
			y_max,
			color,
			fill: false,
			explicit_xy: false,
			antialias_samples: 4,
			line_width: 1.35,
		});
	}

	/// Draw one F32 line series into an explicit pixel rectangle.
	///
	/// The samples are uploaded through this UI's Engine and retained until the
	/// exact composition submission completes; callers never handle descriptor
	/// indices or Vulkan objects.
	pub fn plot_line_at(
		&mut self,
		values: &[f32],
		rect: PixelRect,
		y_min: f32,
		y_max: f32,
		color: Color,
		line_width: f32,
	) -> Result<()> {
		if values.is_empty() {
			return Ok(());
		}
		if rect.w == 0 || rect.h == 0 || !y_min.is_finite() || !y_max.is_finite() || y_min >= y_max {
			return Err(Error::invalid_argument(
				"UI line plot requires a positive rectangle and finite increasing Y limits",
			));
		}
		if !line_width.is_finite() || line_width <= 0.0 {
			return Err(Error::invalid_argument(
				"UI line width must be finite and positive",
			));
		}
		let matrix = crate::Matrix::new_from_f32(self.engine, [values.len()], values)?;
		let descriptor = matrix
			.engine_handle()
			.storage_descriptor_index(matrix.storage())?;
		let state = self
			.state
			.as_mut()
			.ok_or_else(|| Error::failed_precondition("Ui is closed"))?;
		state.cmds.push(UiCmd::PlotLine {
			values_idx: descriptor,
			count: u32::try_from(values.len())
				.map_err(|_| Error::out_of_range("UI plot sample count exceeds u32"))?,
			rect,
			clip: PixelRect {
				x: 0,
				y: 0,
				w: state.compose.width,
				h: state.compose.height,
			},
			x_min: 0.0,
			x_max: values.len().saturating_sub(1).max(1) as f32,
			y_min,
			y_max,
			color,
			fill: false,
			explicit_xy: false,
			antialias_samples: 4,
			line_width,
		});
		state.frame_matrices.push(matrix);
		Ok(())
	}

	/// Draw one F32 line series with the donor chart widget contract.
	///
	/// Donor: `oa::Ui::plotLine`. Y bounds come from `config` or resolve from
	/// finite samples when `auto_scale` is set; a degenerate resolved range is
	/// expanded by a small margin. `show_grid`/`draw_surface` paint the chart
	/// grid first; `fill` shades under the series. The samples are uploaded
	/// through this UI's Engine and retained until the exact composition
	/// submission completes.
	///
	/// A series whose auto-scaled range contains no finite sample draws
	/// nothing, matching the donor's silent skip.
	pub fn plot_line(
		&mut self,
		values: &[f32],
		rect: PixelRect,
		config: &UiPlotConfig,
	) -> Result<()> {
		if values.is_empty() {
			return Ok(());
		}
		if rect.w == 0 || rect.h == 0 {
			return Err(Error::invalid_argument(
				"UI line plot requires a positive rectangle",
			));
		}
		if !config.x_min.is_finite() || !config.x_max.is_finite() {
			return Err(Error::invalid_argument(
				"UI line plot requires finite X limits",
			));
		}
		if !config.auto_scale && (!config.y_min.is_finite() || !config.y_max.is_finite()) {
			return Err(Error::invalid_argument(
				"UI line plot requires finite Y limits",
			));
		}
		let (mut y_min, mut y_max) = if config.auto_scale {
			let mut lo = f32::INFINITY;
			let mut hi = f32::NEG_INFINITY;
			for &value in values {
				if !value.is_finite() {
					continue;
				}
				lo = lo.min(value);
				hi = hi.max(value);
			}
			(lo, hi)
		} else {
			(config.y_min, config.y_max)
		};
		if !y_min.is_finite() || !y_max.is_finite() {
			return Ok(());
		}
		if y_max <= y_min {
			let margin = (y_min.abs() * 0.05).max(1.0e-4);
			y_min -= margin;
			y_max += margin;
		}
		if config.draw_surface || config.show_grid {
			self.grid(
				rect,
				chart_grid_config(
					rect,
					self.content_scale(),
					config.draw_surface,
					config.show_grid,
				),
			);
		}
		let matrix = crate::Matrix::new_from_f32(self.engine, [values.len()], values)?;
		let descriptor = matrix
			.engine_handle()
			.storage_descriptor_index(matrix.storage())?;
		let state = self
			.state
			.as_mut()
			.ok_or_else(|| Error::failed_precondition("Ui is closed"))?;
		let clip = Self::compose_clip(state, rect);
		if clip.w == 0 || clip.h == 0 {
			return Ok(());
		}
		state.cmds.push(UiCmd::PlotLine {
			values_idx: descriptor,
			count: u32::try_from(values.len())
				.map_err(|_| Error::out_of_range("UI plot sample count exceeds u32"))?,
			rect,
			clip,
			x_min: config.x_min,
			x_max: config.x_max,
			y_min,
			y_max,
			color: config.color,
			fill: config.fill,
			explicit_xy: false,
			antialias_samples: match config.antialias_samples {
				1 | 8 => config.antialias_samples,
				_ => 4,
			},
			line_width: if config.line_width.is_finite() {
				config.line_width.clamp(0.5, 16.0)
			} else {
				1.35
			},
		});
		state.frame_matrices.push(matrix);
		Ok(())
	}

	/// Draw an anti-aliased screen-space line.
	pub fn line_at(
		&mut self,
		from: [f32; 2],
		to: [f32; 2],
		color: Color,
		thickness: f32,
		bounds: PixelRect,
	) -> Result<()> {
		if from.into_iter().chain(to).any(|value| !value.is_finite())
			|| !thickness.is_finite()
			|| thickness <= 0.0
			|| bounds.w == 0
			|| bounds.h == 0
		{
			return Err(Error::invalid_argument(
				"UI line requires finite coordinates, positive thickness, and positive bounds",
			));
		}
		let state = self
			.state
			.as_mut()
			.ok_or_else(|| Error::failed_precondition("Ui is closed"))?;
		state.cmds.push(UiCmd::Line {
			x0: from[0],
			y0: from[1],
			x1: to[0],
			y1: to[1],
			color,
			thickness,
			bounds,
		});
		Ok(())
	}

	/// Blit one packed buffer-backed RGBA8 Texture into the compose image.
	pub fn image_at(&mut self, texture: &crate::Texture, rect: PixelRect) -> Result<()> {
		self.image_at_channel(texture, rect, 0)
	}

	/// Blit a packed RGBA8 Texture, optionally exposing one channel as grayscale.
	pub(crate) fn image_at_channel(
		&mut self,
		texture: &crate::Texture,
		rect: PixelRect,
		channel: u32,
	) -> Result<()> {
		if channel > 4 {
			return Err(Error::invalid_argument("UI image channel must be in 0..=4"));
		}
		if rect.w == 0 || rect.h == 0 {
			return Err(Error::invalid_argument(
				"UI image rectangle must be positive",
			));
		}
		if !texture.engine_handle().same_as(&self.engine.handle()) {
			return Err(Error::invalid_argument(
				"UI Texture belongs to a different Engine",
			));
		}
		let storage = texture.packed_storage().ok_or_else(|| {
			Error::failed_precondition("UI image currently requires a buffer-backed Texture")
		})?;
		let descriptor = texture.engine_handle().storage_descriptor_index(storage)?;
		let state = self
			.state
			.as_mut()
			.ok_or_else(|| Error::failed_precondition("Ui is closed"))?;
		state.cmds.push(UiCmd::BlitRgba {
			src_idx: descriptor,
			src_w: u32::try_from(texture.width())
				.map_err(|_| Error::out_of_range("Texture width exceeds UI ABI"))?,
			src_h: u32::try_from(texture.height())
				.map_err(|_| Error::out_of_range("Texture height exceeds UI ABI"))?,
			dst_x: rect.x,
			dst_y: rect.y,
			dst_w: rect.w,
			dst_h: rect.h,
			clip: PixelRect {
				x: 0,
				y: 0,
				w: state.compose.width,
				h: state.compose.height,
			},
			channel,
		});
		state.frame_textures.push(texture.clone());
		Ok(())
	}

	/// Measure the advance width of `text` at `config.size`, in pixels.
	pub(crate) fn measure_text(&self, text: &str, config: TextLayoutConfig) -> Result<f32> {
		let state = self
			.state
			.as_ref()
			.ok_or_else(|| Error::failed_precondition("Ui is closed"))?;
		let shaped = TextLayout::shape(
			&state.text_atlas,
			text,
			[0.0, 0.0],
			config,
			Color::default(),
		)?;
		Ok(shaped.iter().map(|g| g.advance).sum())
	}

	/// Shape and draw text at a pixel-space baseline origin.
	pub fn text_at(
		&mut self,
		text: &str,
		origin: [f32; 2],
		config: TextLayoutConfig,
		color: Color,
		clip: PixelRect,
	) -> Result<()> {
		let state = self
			.state
			.as_mut()
			.ok_or_else(|| Error::failed_precondition("Ui is closed"))?;
		let positioned = TextLayout::shape(&state.text_atlas, text, origin, config, color)?;
		if positioned.is_empty() {
			return Ok(());
		}
		let mut bytes = Vec::with_capacity(positioned.len() * 48);
		for glyph in positioned {
			let Some(info) = state
				.text_atlas
				.find_glyph(glyph.font, glyph.codepoint, config.size)
			else {
				continue;
			};
			let scale = config.size / info.raster_size;
			for value in [
				0.0_f32,
				0.0,
				glyph.x + info.bearing_x * scale,
				glyph.y - info.bearing_y * scale,
				info.atlas_w * scale,
				info.atlas_h * scale,
			] {
				bytes.extend_from_slice(&value.to_le_bytes());
			}
			for value in [
				info.atlas_x as u32,
				info.atlas_y as u32,
				info.atlas_w as u32,
				info.atlas_h as u32,
				glyph.color.to_u32(),
				0,
			] {
				bytes.extend_from_slice(&value.to_le_bytes());
			}
		}
		if bytes.is_empty() {
			return Ok(());
		}
		let storage = state.engine_handle.create_storage(&bytes)?;
		let glyph_idx = state.engine_handle.storage_descriptor_index(&storage)?;
		let count = u32::try_from(bytes.len() / 48)
			.map_err(|_| Error::out_of_range("glyph count exceeds u32"))?;
		state.cmds.push(UiCmd::Glyphs {
			glyph_idx,
			atlas_idx: state.text_atlas.descriptor(),
			count,
			dst: None,
			clip,
			atlas_w: state.text_atlas.width(),
			atlas_h: state.text_atlas.height(),
		});
		state.frame_storages.push(storage);
		Ok(())
	}

	// ── Low-level draw primitives ─────────────────────────────────────────────
	// These are used by plot::Figure and viewer modules that need to bypass
	// the layout system and emit commands with explicit rects.

	/// Alpha-blend a filled rect at an explicit position.
	pub fn rect_raw(&mut self, rect: PixelRect, color: Color, clip: PixelRect, corner_radius: f32) {
		if let Some(state) = self.state.as_mut() {
			state.cmds.push(UiCmd::Rect {
				rect,
				color,
				clip,
				corner_radius,
			});
		}
	}

	/// Alpha-blend a rect outline at an explicit position.
	pub fn rect_outline_raw(
		&mut self,
		rect: PixelRect,
		color: Color,
		thickness: u32,
		clip: PixelRect,
	) {
		self.rect_outline_rounded_at(rect, color, thickness, clip, 0.0);
	}

	/// Draw a rounded outline using the retained compositor.
	pub fn rect_outline_rounded_at(
		&mut self,
		rect: PixelRect,
		color: Color,
		thickness: u32,
		clip: PixelRect,
		radius: f32,
	) {
		if let Some(state) = self.state.as_mut() {
			state.cmds.push(UiCmd::RectOutline {
				rect,
				color,
				thickness,
				clip,
				corner_radius: radius,
			});
		}
	}

	/// Alpha-blend a filled rect clipped to the current panel scope.
	///
	/// Donor: `oa::Ui::rect`. Nonpositive or out-of-scope rects draw nothing.
	/// `corner_radius` must be finite and non-negative; it is clamped to half
	/// the smaller side exactly like the donor.
	pub fn rect(&mut self, rect: PixelRect, color: Color, corner_radius: f32) -> Result<()> {
		if !corner_radius.is_finite() || corner_radius < 0.0 {
			return Err(Error::invalid_argument(
				"Ui::rect requires a finite non-negative corner radius",
			));
		}
		if rect.w == 0 || rect.h == 0 {
			return Ok(());
		}
		let Some(state) = self.state.as_mut() else {
			return Ok(());
		};
		let clip = Self::compose_clip(state, rect);
		if clip.w == 0 || clip.h == 0 {
			return Ok(());
		}
		state.cmds.push(UiCmd::Rect {
			rect,
			color,
			clip,
			corner_radius: corner_radius.min(0.5 * rect.w.min(rect.h) as f32),
		});
		Ok(())
	}

	/// Alpha-blend a rect outline clipped to the current panel scope.
	///
	/// Donor: `oa::Ui::rectOutline`.
	pub fn rect_outline(
		&mut self,
		rect: PixelRect,
		color: Color,
		thickness: u32,
		corner_radius: f32,
	) -> Result<()> {
		if !corner_radius.is_finite() || corner_radius < 0.0 {
			return Err(Error::invalid_argument(
				"Ui::rect_outline requires a finite non-negative corner radius",
			));
		}
		if rect.w == 0 || rect.h == 0 {
			return Ok(());
		}
		let Some(state) = self.state.as_mut() else {
			return Ok(());
		};
		let clip = Self::compose_clip(state, rect);
		if clip.w == 0 || clip.h == 0 {
			return Ok(());
		}
		state.cmds.push(UiCmd::RectOutline {
			rect,
			color,
			thickness: thickness.max(1),
			clip,
			corner_radius: corner_radius.min(0.5 * rect.w.min(rect.h) as f32),
		});
		Ok(())
	}

	/// Draw normalized rectangle records from a GPU detection buffer in one
	/// dispatch.
	///
	/// `dst` maps the records' normalized source-image coordinates into the
	/// compose image; `clip` bounds rasterization and is intersected with
	/// `dst`. Records carry an optional per-instance `0xRRGGBBAA` color that
	/// falls back to `color`. Donor: `oa::Ui::rectOutlines`.
	pub fn rect_outlines(
		&mut self,
		detections: &crate::vision::DetectionBuffer,
		dst: PixelRect,
		clip: PixelRect,
		color: Color,
		thickness: u32,
	) -> Result<()> {
		if !detections.engine_handle().same_as(&self.engine.handle()) {
			return Err(Error::invalid_argument(
				"Ui::rect_outlines requires a DetectionBuffer from its engine",
			));
		}
		if detections.count() == 0 || dst.w == 0 || dst.h == 0 || clip.w == 0 || clip.h == 0 {
			return Ok(());
		}
		let clip = dst.clip(&clip);
		if clip.w == 0 || clip.h == 0 {
			return Ok(());
		}
		let rect_idx = detections.bindless_index()?;
		let state = self
			.state
			.as_mut()
			.ok_or_else(|| Error::failed_precondition("Ui is closed"))?;
		state.cmds.push(UiCmd::RectOutlines {
			rect_idx,
			count: detections.count(),
			dst,
			clip,
			thickness: thickness.max(1),
			rgba: color.to_u32(),
		});
		state
			.frame_storages
			.push(detections.storage().storage().clone());
		Ok(())
	}

	/// Draw a source-anchored glyph batch from one GPU record buffer through
	/// this `Ui`'s coverage atlas.
	///
	/// `dst` maps the records' normalized source-image anchors into the
	/// compose image while offsets and glyph extents stay in pixels; `clip`
	/// bounds rasterization and is intersected with `dst`. Donor:
	/// `oa::Ui::glyphs`.
	pub fn glyphs(&mut self, buffer: &GlyphBuffer, dst: PixelRect, clip: PixelRect) -> Result<()> {
		if !buffer.engine_handle().same_as(&self.engine.handle()) {
			return Err(Error::invalid_argument(
				"Ui::glyphs requires a GlyphBuffer from its engine",
			));
		}
		if buffer.count() == 0 || dst.w == 0 || dst.h == 0 || clip.w == 0 || clip.h == 0 {
			return Ok(());
		}
		let clip = dst.clip(&clip);
		if clip.w == 0 || clip.h == 0 {
			return Ok(());
		}
		let glyph_idx = buffer.bindless_index()?;
		let state = self
			.state
			.as_mut()
			.ok_or_else(|| Error::failed_precondition("Ui is closed"))?;
		state.cmds.push(UiCmd::Glyphs {
			glyph_idx,
			atlas_idx: state.text_atlas.descriptor(),
			count: buffer.count(),
			dst: Some(dst),
			clip,
			atlas_w: state.text_atlas.width(),
			atlas_h: state.text_atlas.height(),
		});
		state
			.frame_storages
			.push(buffer.storage().storage().clone());
		Ok(())
	}

	/// Return this `Ui`'s shared coverage atlas for shaping record buffers.
	///
	/// Glyph records store atlas coordinates; this atlas is the only producer
	/// of those coordinates, so [`Ui::glyphs`] consumers shape against the
	/// same instance their rasterization samples.
	///
	/// # Errors
	///
	/// Returns an error when the `Ui` is closed.
	pub fn text_atlas(&self) -> Result<&TextAtlas> {
		Ok(
			self
				.state
				.as_ref()
				.ok_or_else(|| Error::failed_precondition("Ui is closed"))?
				.text_atlas
				.as_ref(),
		)
	}

	// ── Record and submit ──────────────────────────────────────────────────────

	// ── Tier 1: Timeline widgets ───────────────────────────────────────────────

	/// Playback timeline scrub bar.  Returns `true` while scrubbing.
	///
	/// `fraction` is clamped to `[0, 1]`.  `out_active` is set to `true` while
	/// the pointer is held down on the bar.
	///
	/// Donor: `oa::Ui::timeline`.
	pub fn timeline(
		&mut self,
		id: &str,
		rect: PixelRect,
		fraction: &mut f32,
		out_active: Option<&mut bool>,
	) -> bool {
		let Some(state) = self.state.as_mut() else {
			return false;
		};
		let style = Self::current_style_ref(state).clone();
		let widget_id = widget_hash(id, Some(rect));
		let clip = rect;

		let hovered = point_in_rect(state.input.pointer_x, state.input.pointer_y, rect);
		if hovered && state.input.pointer_down {
			state.pressed = widget_id;
		}
		if hovered {
			state.hovered = widget_id;
		}
		let dragging = state.pressed == widget_id && state.input.pointer_down;
		let mut changed = false;
		if dragging {
			let new_frac =
				((state.input.pointer_x - rect.x as f32) / rect.w.max(1) as f32).clamp(0.0, 1.0);
			if (*fraction - new_frac).abs() > 1e-5 {
				*fraction = new_frac;
				changed = true;
			}
		}
		if let Some(active) = out_active {
			*active = dragging;
		}

		// Track.
		let track_h = (4.0 * state.content_scale).round().max(1.0) as u32;
		let track_h = track_h.min(rect.h);
		let track_rect = PixelRect {
			x: rect.x,
			y: rect.y + (rect.h - track_h) as i32 / 2,
			w: rect.w,
			h: track_h,
		};
		state.cmds.push(UiCmd::Rect {
			rect: track_rect,
			color: Color {
				a: 0.78,
				..style.surface
			},
			clip,
			corner_radius: track_h as f32 * 0.5,
		});
		// Filled portion.
		let frac = fraction.clamp(0.0, 1.0);
		let fill_w = (track_rect.w as f32 * frac) as u32;
		if fill_w > 0 {
			state.cmds.push(UiCmd::Rect {
				rect: PixelRect {
					x: track_rect.x,
					y: track_rect.y,
					w: fill_w,
					h: track_rect.h,
				},
				color: style.accent,
				clip,
				corner_radius: track_h as f32 * 0.5,
			});
		}
		// Thumb.
		let base_diameter = ((20.0 * state.content_scale).round().max(1.0) as u32).min(rect.h);
		let hover_mix = animate_hover(state, widget_id, hovered);
		let diameter = ((if dragging {
			18.0
		} else {
			20.0 + 4.0 * hover_mix
		}) * state.content_scale)
			.round()
			.max(1.0) as u32;
		let diameter = diameter.min(rect.h);
		let thumb_cx = rect.x
			+ base_diameter as i32 / 2
			+ (rect.w.saturating_sub(base_diameter) as f32 * frac).round() as i32;
		let thumb_rect = PixelRect::new(
			thumb_cx - diameter as i32 / 2,
			rect.y + (rect.h - diameter) as i32 / 2,
			diameter,
			diameter,
		);
		state.cmds.push(UiCmd::Rect {
			rect: thumb_rect,
			color: if dragging || hovered {
				style.accent_hover
			} else {
				style.text
			},
			clip,
			corner_radius: diameter as f32 * 0.5,
		});
		changed
	}

	/// Waveform timeline — like `timeline` but draws the waveform envelope
	/// behind the scrub bar.
	///
	/// `envelope` must be a 1-D slice of pre-computed RMS/peak amplitudes in
	/// `[0, 1]`.  One value per horizontal pixel is ideal; fewer values are
	/// stretched.
	///
	/// Donor: `oa::Ui::waveformTimeline`.
	pub fn waveform_timeline(
		&mut self,
		id: &str,
		rect: PixelRect,
		envelope: &[f32],
		fraction: &mut f32,
	) -> bool {
		let Some(state) = self.state.as_mut() else {
			return false;
		};
		let style = Self::current_style_ref(state).clone();
		let clip = rect;

		// Waveform background.
		state.cmds.push(UiCmd::Rect {
			rect,
			color: Color::new(
				style.surface.r * 0.6,
				style.surface.g * 0.6,
				style.surface.b * 0.6,
				1.0,
			),
			clip,
			corner_radius: 0.0,
		});

		// Draw envelope bars.
		if !envelope.is_empty() {
			let n = envelope.len();
			let bar_w = (rect.w as f32 / n as f32).max(1.0);
			for (i, &amp) in envelope.iter().enumerate() {
				let amp = amp.clamp(0.0, 1.0);
				let bar_h = (rect.h as f32 * amp).max(1.0) as u32;
				let bar_x = rect.x + (i as f32 * bar_w) as i32;
				let bar_y = rect.y + (rect.h - bar_h) as i32 / 2;
				let bar_rect = PixelRect {
					x: bar_x,
					y: bar_y,
					w: bar_w as u32,
					h: bar_h,
				};
				state.cmds.push(UiCmd::Rect {
					rect: bar_rect,
					color: style.text_secondary,
					clip,
					corner_radius: 0.0,
				});
			}
		}

		// Overlay scrub bar on top of waveform.
		self.timeline(id, rect, fraction, None)
	}

	// ── Tier 2: Container widgets ──────────────────────────────────────────────

	/// Scrollable panel.  Returns scroll region info.
	///
	/// `content_height` is the total height of the content in pixels.
	/// Caller provides the panel `id` for scroll-offset persistence.
	///
	/// Donor: `oa::Ui::beginScrollPanel` / `endScrollPanel`.
	pub fn begin_scroll_panel(
		&mut self,
		id: &str,
		rect: PixelRect,
		content_height: i32,
		config: UiScrollConfig,
	) -> UiScrollRegion {
		if rect.w == 0
			|| rect.h == 0
			|| rect.w > i32::MAX as u32
			|| rect.h > i32::MAX as u32
			|| content_height < 0
			|| !config.wheel_step.is_finite()
			|| config.wheel_step <= 0.0
			|| !config.scrollbar_width.is_finite()
			|| config.scrollbar_width > i32::MAX as f32
			|| (config.show_scrollbar && config.scrollbar_width <= 0.0)
			|| !config.scrollbar_gap.is_finite()
			|| config.scrollbar_gap > i32::MAX as f32
			|| config.scrollbar_gap < 0.0
		{
			return UiScrollRegion::default();
		}
		let panel_id = widget_hash(id, Some(rect));
		let viewport_height = i32::try_from(rect.h).unwrap_or(i32::MAX);
		let content_height = content_height.max(viewport_height);
		let max_offset = content_height.saturating_sub(viewport_height);

		let Some(state) = self.state.as_mut() else {
			return UiScrollRegion {
				viewport: rect,
				content: rect,
				..Default::default()
			};
		};

		// Scroll on wheel.
		if point_in_rect(state.input.pointer_x, state.input.pointer_y, rect) {
			let dy = -state.input.scroll_y;
			if dy.abs() > 0.5 {
				let entry = state.scroll_offsets.entry(panel_id).or_insert(0);
				*entry = entry
					.saturating_add((dy * config.wheel_step) as i32)
					.clamp(0, max_offset);
			}
		}
		let offset_y = *state.scroll_offsets.get(&panel_id).unwrap_or(&0);
		let offset_y = offset_y.clamp(0, max_offset);
		state.scroll_offsets.insert(panel_id, offset_y);

		// Content area.
		let scrollbar_width = config.scrollbar_width.ceil().max(1.0) as u32;
		let scrollbar_gap = config.scrollbar_gap.ceil() as u32;
		let show_scrollbar = config.show_scrollbar
			&& max_offset > 0
			&& rect.w > scrollbar_width.saturating_add(scrollbar_gap);
		let sb_room = if show_scrollbar {
			(scrollbar_width + scrollbar_gap) as i32
		} else {
			0
		};
		let content_rect = PixelRect {
			x: rect.x,
			y: rect.y.saturating_sub(offset_y),
			w: (rect.w as i32 - sb_room).max(0) as u32,
			h: content_height as u32,
		};
		let viewport = PixelRect {
			h: rect.h,
			w: content_rect.w,
			..rect
		};
		let panel_clip = state
			.layout
			.top()
			.map_or(rect, |parent| parent.clip().clip(&rect));
		let clip = panel_clip.clip(&viewport);

		// Draw panel background.
		let style = Self::current_style_ref(state).clone();
		state.cmds.push(UiCmd::Rect {
			rect,
			color: style.background,
			clip: panel_clip,
			corner_radius: 0.0,
		});

		// Scrollbar.
		if show_scrollbar {
			let sb_x = rect
				.x
				.saturating_add(rect.w as i32 - scrollbar_width as i32);
			let sb_track = PixelRect {
				x: sb_x,
				y: rect.y,
				w: scrollbar_width,
				h: rect.h,
			};
			state.cmds.push(UiCmd::Rect {
				rect: sb_track,
				color: Color::new(
					style.surface.r * 0.7,
					style.surface.g * 0.7,
					style.surface.b * 0.7,
					1.0,
				),
				clip: panel_clip,
				corner_radius: scrollbar_width as f32 * 0.5,
			});
			let thumb_h = ((rect.h as f32 * rect.h as f32 / content_height as f32) as u32)
				.max(16)
				.min(rect.h);
			let thumb_y =
				rect.y + (offset_y as f32 / max_offset as f32 * (rect.h - thumb_h) as f32) as i32;
			state.cmds.push(UiCmd::Rect {
				rect: PixelRect {
					x: sb_x,
					y: thumb_y,
					w: scrollbar_width,
					h: thumb_h,
				},
				color: style.accent,
				clip: panel_clip,
				corner_radius: scrollbar_width as f32 * 0.5,
			});
		}

		// Push a panel scope for content at the scrolled offset.
		let layout = UiLayout {
			padding: UiEdge::all(0.0),
			..UiLayout::default()
		};
		let mut scope = PanelScope::new(content_rect, layout);
		scope.clip_override = Some(clip);
		scope.scroll_viewport = Some(viewport);
		scope.scroll_offset_y = offset_y;
		state.layout.push(scope);

		UiScrollRegion {
			viewport,
			content: content_rect,
			offset_y,
			max_offset_y: max_offset,
		}
	}

	/// End the innermost scroll panel.
	///
	/// Donor: `oa::Ui::endScrollPanel`.
	pub fn end_scroll_panel(&mut self) {
		let Some(state) = self.state.as_mut() else {
			return;
		};
		if state
			.layout
			.top()
			.is_some_and(|scope| scope.scroll_viewport.is_some())
		{
			state.layout.pop();
		}
	}

	/// Virtual row helper — returns the range of row indices that are visible
	/// inside the current scroll panel's viewport.
	///
	/// Donor: `oa::Ui::virtualRows`.
	pub fn virtual_rows(
		&mut self,
		item_count: i32,
		row_height: i32,
		row_gap: i32,
		overscan_rows: i32,
	) -> UiVirtualRange {
		let Some(scope) = self.state.as_ref().and_then(|state| state.layout.top()) else {
			return UiVirtualRange::default();
		};
		let Some(viewport) = scope.scroll_viewport else {
			return UiVirtualRange::default();
		};
		virtual_row_range(
			item_count,
			row_height,
			row_gap,
			overscan_rows,
			scope.scroll_offset_y,
			viewport.h,
			scope.layout.padding.top,
		)
	}

	/// Two-pane split view.  Mutates `ratio` (first pane fraction in `[0, 1]`).
	///
	/// Donor: `oa::Ui::splitPane`.
	pub fn split_pane(
		&mut self,
		id: &str,
		rect: PixelRect,
		ratio: &mut f32,
		config: UiSplitConfig,
	) -> UiSplitRegion {
		let total = match config.direction {
			UiDirection::Row => rect.w,
			UiDirection::Column => rect.h,
		};
		if rect.w == 0
			|| rect.h == 0
			|| total > i32::MAX as u32
			|| !config.handle_size.is_finite()
			|| config.handle_size < 1.0
			|| !config.minimum_first.is_finite()
			|| config.minimum_first < 0.0
			|| !config.minimum_second.is_finite()
			|| config.minimum_second < 0.0
			|| !config.keyboard_step.is_finite()
			|| config.keyboard_step <= 0.0
			|| !ratio.is_finite()
		{
			return UiSplitRegion::default();
		}
		let widget_id = widget_hash(id, Some(rect));
		let handle_px = config.handle_size.ceil() as i32;
		let available = total as i32 - handle_px;
		let minimum_first = config.minimum_first.ceil() as i64;
		let minimum_second = config.minimum_second.ceil() as i64;
		if available <= 0 || minimum_first + minimum_second > i64::from(available) {
			return UiSplitRegion::default();
		}
		let clamp_ratio = |r: f32| {
			let lo = minimum_first as f32 / available as f32;
			let hi = 1.0 - minimum_second as f32 / available as f32;
			r.clamp(lo, hi)
		};
		let clamped = clamp_ratio(*ratio);
		let mut changed = clamped != *ratio;
		*ratio = clamped;

		let Some(state) = self.state.as_mut() else {
			return UiSplitRegion::default();
		};
		let style = Self::current_style_ref(state).clone();

		let (_, handle_rect, _) = split_rects(rect, config.direction, handle_px as u32, *ratio);

		let hovered = point_in_rect(state.input.pointer_x, state.input.pointer_y, handle_rect);
		if hovered && state.input.pointer_down {
			state.pressed = widget_id;
		}
		let dragging = state.pressed == widget_id && state.input.pointer_down;
		if dragging {
			let new_ratio = match config.direction {
				UiDirection::Row => {
					(state.input.pointer_x - rect.x as f32 - handle_px as f32 * 0.5) / available as f32
				}
				UiDirection::Column => {
					(state.input.pointer_y - rect.y as f32 - handle_px as f32 * 0.5) / available as f32
				}
			};
			let clamped = clamp_ratio(new_ratio);
			if (*ratio - clamped).abs() > 1e-5 {
				*ratio = clamped;
				changed = true;
			}
		}

		let (first_rect, handle_rect, second_rect) =
			split_rects(rect, config.direction, handle_px as u32, *ratio);

		// Handle visual.
		let handle_color = if dragging || hovered {
			style.accent
		} else {
			style.border
		};
		state.cmds.push(UiCmd::Rect {
			rect: handle_rect,
			color: handle_color,
			clip: rect,
			corner_radius: 0.0,
		});

		UiSplitRegion {
			first: first_rect,
			handle: handle_rect,
			second: second_rect,
			changed,
		}
	}

	/// Property row — a two-column row with a label and a value area.
	///
	/// Donor: `oa::Ui::propertyRow`.
	pub fn property_row(
		&mut self,
		_id: &str,
		rect: PixelRect,
		label: &str,
		value: &str,
		config: UiPropertyRowConfig,
	) -> UiPropertyRegion {
		let Some(state) = self.state.as_mut() else {
			return UiPropertyRegion::default();
		};
		let style = Self::current_style_ref(state).clone();
		let clip = rect;

		let label_w = (rect.w as f32 * config.label_fraction) as u32;
		let gap = config.gap as i32;
		let label_rect = PixelRect {
			x: rect.x + config.padding_x as i32,
			y: rect.y,
			w: label_w.saturating_sub(config.padding_x as u32),
			h: rect.h,
		};
		let value_rect = PixelRect {
			x: rect.x + label_w as i32 + gap,
			y: rect.y,
			w: (rect.w as i32 - label_w as i32 - gap - config.padding_x as i32).max(0) as u32,
			h: rect.h,
		};

		if config.alternate {
			state.cmds.push(UiCmd::Rect {
				rect,
				color: Color::new(
					style.surface.r * 0.7,
					style.surface.g * 0.7,
					style.surface.b * 0.7,
					0.5,
				),
				clip,
				corner_radius: 0.0,
			});
		}

		let text_cfg = TextLayoutConfig {
			font: FontId::Sans,
			size: style.font_size - 1.0,
			..Default::default()
		};
		Self::emit_text_or_record(state, label, label_rect, style.text_secondary, text_cfg);
		Self::emit_text_or_record(state, value, value_rect, style.text, text_cfg);

		UiPropertyRegion {
			row: rect,
			label: label_rect,
			value: value_rect,
		}
	}

	// ── Popup system ───────────────────────────────────────────────────────────

	/// Register a popup to open at the current pointer position.
	///
	/// Donor: `oa::Ui::openPopup`.
	pub fn open_popup(&mut self, id: &str) {
		let Some(state) = self.state.as_mut() else {
			return;
		};
		let popup_id = widget_hash(id, None);
		state.popup_open = popup_id;
		state.popup_anchor = PixelRect {
			x: state.input.pointer_x as i32,
			y: state.input.pointer_y as i32,
			w: 0,
			h: 0,
		};
	}

	/// Register a popup to open anchored to `anchor`.
	///
	/// Donor: `oa::Ui::openPopup(id, anchor)`.
	pub fn open_popup_at(&mut self, id: &str, anchor: PixelRect) {
		let Some(state) = self.state.as_mut() else {
			return;
		};
		let popup_id = widget_hash(id, None);
		state.popup_open = popup_id;
		state.popup_anchor = anchor;
	}

	/// Close the currently open popup.
	///
	/// Donor: `oa::Ui::closePopup`.
	pub fn close_popup(&mut self) {
		let Some(state) = self.state.as_mut() else {
			return;
		};
		state.popup_open = 0;
	}

	/// True when the popup with `id` is currently open.
	///
	/// Donor: `oa::Ui::isPopupOpen`.
	pub fn is_popup_open(&mut self, id: &str) -> bool {
		let Some(state) = self.state.as_ref() else {
			return false;
		};
		state.popup_open == widget_hash(id, None)
	}

	/// Begin a popup panel.  Returns `true` if the popup is open and was laid
	/// out; the caller must call `end_popup` if and only if `begin_popup`
	/// returned `true`.
	///
	/// Donor: `oa::Ui::beginPopup`.
	pub fn begin_popup(&mut self, id: &str, config: UiPopupConfig) -> bool {
		let popup_id = widget_hash(id, None);
		let Some(state) = self.state.as_mut() else {
			return false;
		};
		if state.popup_open != popup_id {
			return false;
		}
		let anchor = state.popup_anchor;
		let style = Self::current_style_ref(state).clone();

		// Donor placement: anchor left, prefer below, fall back above, then clamp.
		let width = i64::from(state.compose.width);
		let height = i64::from(state.compose.height);
		let margin = 4_i64.min(width.min(height) / 2);
		let popup_w = (config.width.max(anchor.w as f32) as i64).clamp(1, (width - 2 * margin).max(1));
		let popup_h = (if config.height > 0.0 {
			config.height as i64
		} else {
			200
		})
		.clamp(1, (height - 2 * margin).max(1));
		let below = i64::from(anchor.y) + i64::from(anchor.h) + config.gap.max(0.0) as i64;
		let above = i64::from(anchor.y) - config.gap.max(0.0) as i64 - popup_h;
		let y = if below + popup_h <= height - margin {
			below
		} else if above >= margin {
			above
		} else {
			below.clamp(margin, (height - margin - popup_h).max(margin))
		};
		let popup_rect = PixelRect::new(
			i64::from(anchor.x).clamp(margin, (width - margin - popup_w).max(margin)) as i32,
			y as i32,
			popup_w as u32,
			popup_h as u32,
		);
		if state.input.l_pressed
			&& !point_in_rect(state.input.pointer_x, state.input.pointer_y, popup_rect)
		{
			state.popup_open = 0;
			return false;
		}

		// Shadow / background.
		state.cmds.push(UiCmd::Rect {
			rect: popup_rect,
			color: style.background,
			clip: popup_rect,
			corner_radius: style.corner_radius,
		});
		state.cmds.push(UiCmd::RectOutline {
			rect: popup_rect,
			color: style.border,
			thickness: 1,
			clip: popup_rect,
			corner_radius: style.corner_radius,
		});

		let inner = UiLayout {
			padding: UiEdge::all(config.padding),
			gap: 4.0 * state.content_scale,
			..UiLayout::default()
		};
		state.layout.push(PanelScope::new(popup_rect, inner));
		true
	}

	/// End the popup panel.
	///
	/// Donor: `oa::Ui::endPopup`.
	pub fn end_popup(&mut self) {
		self.end_panel();
	}

	/// Menu item inside an open popup.  Returns `true` when clicked.
	///
	/// Donor: `oa::Ui::menuItem`.
	pub fn menu_item(&mut self, label: &str, selected: bool, enabled: bool) -> bool {
		let Some(state) = self.state.as_mut() else {
			return false;
		};
		let style = Self::current_style_ref(state).clone();
		let id = widget_hash(label, state.layout.top().map(|s| s.outer));
		let height = ((style.font_size + 8.0) * state.content_scale)
			.ceil()
			.max(1.0) as u32;
		let (rect, clip) = allocate_widget_explicit(state, style.padding, height);

		let hovered = enabled && point_in_rect(state.input.pointer_x, state.input.pointer_y, rect);
		if hovered && state.input.l_pressed {
			state.pressed = id;
		}
		let held = enabled && state.pressed == id && state.input.pointer_down;
		let activated = hovered && state.input.l_released && state.pressed == id;
		if activated {
			// Close popup on selection.
			state.popup_open = 0;
		}
		if hovered {
			state.hovered = id;
		}

		let bg = if selected {
			Color::new(
				style.accent.r,
				style.accent.g,
				style.accent.b,
				if hovered { 0.35 } else { 0.22 },
			)
		} else if hovered || held {
			if held {
				style.surface_active
			} else {
				style.surface_hover
			}
		} else {
			Color::new(0.0, 0.0, 0.0, 0.0)
		};
		if bg.a > 0.0 {
			state.cmds.push(UiCmd::Rect {
				rect,
				color: bg,
				clip,
				corner_radius: (style.corner_radius * 0.65).min(rect.h as f32 * 0.5),
			});
		}
		let text_color = if enabled {
			style.text
		} else {
			style.text_disabled
		};
		let cfg = TextLayoutConfig {
			font: FontId::Sans,
			size: style.font_size * state.content_scale,
			..Default::default()
		};
		if selected {
			state.cmds.push(UiCmd::Rect {
				rect: PixelRect::new(rect.x, rect.y, rect.w.min(3), rect.h),
				color: style.accent,
				clip,
				corner_radius: 1.5,
			});
		}
		let padding = ((6.0 + if selected { 4.0 } else { 0.0 }) * state.content_scale).round() as u32;
		let text_rect = PixelRect::new(
			rect.x + padding as i32,
			rect.y,
			rect.w.saturating_sub(padding),
			rect.h,
		);
		Self::emit_text_or_record(state, label, text_rect, text_color, cfg);
		activated
	}

	// ── Tier 2: Dropdown ──────────────────────────────────────────────────────

	/// Dropdown selection widget.  Returns `true` when `selected` changes.
	///
	/// `items` is the list of option strings.  `selected` is the current index
	/// (may be -1 for no selection).
	///
	/// Donor: `oa::Ui::dropdown`.
	pub fn dropdown(
		&mut self,
		label: &str,
		items: &[&str],
		selected: &mut i32,
		_config: UiDropdownConfig,
	) -> bool {
		let Some(state) = self.state.as_mut() else {
			return false;
		};
		let style = Self::current_style_ref(state).clone();
		let id = widget_hash(label, state.layout.top().map(|s| s.outer));
		let popup_id = widget_hash(&format!("__dd_{label}"), None);

		let (rect, clip) = allocate_widget(state, style.padding);
		let hovered = point_in_rect(state.input.pointer_x, state.input.pointer_y, rect);
		if hovered {
			state.hovered = id;
		}

		let clicked = hovered && was_pressed_and_released(state, id);
		if clicked {
			if state.popup_open == popup_id {
				state.popup_open = 0;
			} else {
				state.popup_open = popup_id;
				state.popup_anchor = rect;
			}
		}

		// Dropdown button.
		let is_open = state.popup_open == popup_id;
		let bg = if is_open || hovered {
			style.surface
		} else {
			Color::new(
				style.surface.r * 0.85,
				style.surface.g * 0.85,
				style.surface.b * 0.85,
				1.0,
			)
		};
		state.cmds.push(UiCmd::Rect {
			rect,
			color: bg,
			clip,
			corner_radius: style.corner_radius,
		});
		state.cmds.push(UiCmd::RectOutline {
			rect,
			color: if is_open { style.accent } else { style.border },
			thickness: 1,
			clip,
			corner_radius: style.corner_radius,
		});
		let display = if *selected >= 0 && (*selected as usize) < items.len() {
			items[*selected as usize]
		} else {
			label
		};
		let cfg = TextLayoutConfig {
			font: FontId::Sans,
			size: style.font_size - 1.0,
			..Default::default()
		};
		Self::emit_text_or_record(state, display, rect, style.text, cfg);

		// Popup item list.
		let mut changed = false;
		if is_open && !items.is_empty() {
			let item_h = default_item_height(style.padding) as i32;
			let popup_h = (items.len() as i32 * (item_h + 2) + 8).min(300);
			let popup_rect = PixelRect {
				x: rect.x,
				y: rect.y + rect.h as i32 + 2,
				w: rect.w,
				h: popup_h as u32,
			};
			// Close on outside click.
			if state.input.l_pressed
				&& !point_in_rect(state.input.pointer_x, state.input.pointer_y, popup_rect)
				&& !point_in_rect(state.input.pointer_x, state.input.pointer_y, rect)
			{
				state.popup_open = 0;
			} else {
				state.cmds.push(UiCmd::Rect {
					rect: popup_rect,
					color: style.background,
					clip: popup_rect,
					corner_radius: style.corner_radius,
				});
				state.cmds.push(UiCmd::RectOutline {
					rect: popup_rect,
					color: style.border,
					thickness: 1,
					clip: popup_rect,
					corner_radius: style.corner_radius,
				});
				for (i, &item) in items.iter().enumerate() {
					let item_rect = PixelRect {
						x: popup_rect.x + 4,
						y: popup_rect.y + 4 + i as i32 * (item_h + 2),
						w: popup_rect.w.saturating_sub(8),
						h: item_h as u32,
					};
					let hov = point_in_rect(state.input.pointer_x, state.input.pointer_y, item_rect);
					let iid = widget_hash(&format!("__ddi_{i}_{label}"), None);
					if hov {
						state.hovered = iid;
					}
					if hov && was_pressed_and_released(state, iid) {
						*selected = i as i32;
						state.popup_open = 0;
						changed = true;
					}
					let item_bg = if *selected == i as i32 {
						Color::new(
							style.accent.r * 0.3,
							style.accent.g * 0.3,
							style.accent.b * 0.4,
							1.0,
						)
					} else if hov {
						style.surface
					} else {
						Color::new(0.0, 0.0, 0.0, 0.0)
					};
					if item_bg.a > 0.0 {
						state.cmds.push(UiCmd::Rect {
							rect: item_rect,
							color: item_bg,
							clip: popup_rect,
							corner_radius: 0.0,
						});
					}
					let icfg = TextLayoutConfig {
						font: FontId::Sans,
						size: style.font_size - 1.0,
						..Default::default()
					};
					Self::emit_text_or_record(state, item, item_rect, style.text, icfg);
				}
			}
		}
		changed
	}

	// ── Tier 2: Tooltip ───────────────────────────────────────────────────────

	/// Tooltip shown near the current pointer when the previous widget is
	/// hovered.  Caller should call this immediately after the widget it
	/// annotates; it is a no-op when nothing is hovered.
	///
	/// Donor: `oa::Ui::tooltip`.
	pub fn tooltip(&mut self, text: &str, _config: UiTooltipConfig) {
		let Some(state) = self.state.as_mut() else {
			return;
		};
		if state.hovered == 0 {
			return;
		}
		let style = Self::current_style_ref(state).clone();
		let px = state.input.pointer_x;
		let py = state.input.pointer_y;
		let pad = _config.padding;
		let max_w = _config.max_width as u32;
		let tip_h = (style.font_size + pad * 2.0) as u32;
		let tip_rect = PixelRect {
			x: px as i32 + _config.gap as i32,
			y: py as i32 - tip_h as i32 - _config.gap as i32,
			w: max_w,
			h: tip_h,
		};
		state.cmds.push(UiCmd::Rect {
			rect: tip_rect,
			color: Color::new(
				style.surface.r * 1.15,
				style.surface.g * 1.15,
				style.surface.b * 1.15,
				0.95,
			),
			clip: tip_rect,
			corner_radius: style.corner_radius,
		});
		state.cmds.push(UiCmd::RectOutline {
			rect: tip_rect,
			color: style.border,
			thickness: 1,
			clip: tip_rect,
			corner_radius: style.corner_radius,
		});
		let cfg = TextLayoutConfig {
			font: FontId::Sans,
			size: style.font_size - 1.0,
			..Default::default()
		};
		Self::emit_text_or_record(state, text, tip_rect, style.text, cfg);
	}

	// ── Tier 2: Heatmap (Ui-level) ────────────────────────────────────────────

	/// Inline heatmap widget placed in the current panel flow.
	///
	/// Each element in `data` maps to one cell.  Values are scaled from
	/// `[v_min, v_max]` to a color in the chosen colormap.
	///
	/// Donor: `oa::Ui::heatmap`.
	pub fn heatmap_inline(
		&mut self,
		_label: &str,
		data: &[f32],
		config: UiHeatmapConfig,
	) -> Result<()> {
		if config.rows <= 0 || config.cols <= 0 {
			return Err(Error::invalid_argument("heatmap rows and cols must be > 0"));
		}
		let n = (config.rows * config.cols) as usize;
		let data_slice = if data.len() >= n { &data[..n] } else { data };

		let (v_min, v_max) = if config.auto_scale && !data_slice.is_empty() {
			let lo = data_slice.iter().cloned().fold(f32::INFINITY, f32::min);
			let hi = data_slice.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
			if lo < hi {
				(lo, hi)
			} else {
				(config.v_min, config.v_max)
			}
		} else {
			(config.v_min, config.v_max)
		};
		let range = (v_max - v_min).max(1e-9);

		let style;
		let rect;
		let clip;
		{
			let Some(state) = self.state.as_mut() else {
				return Ok(());
			};
			style = Self::current_style_ref(state).clone();
			let alloc_h = (config.rows as u32) * (default_item_height(style.padding));
			let (r, c) = allocate_widget_explicit(state, style.padding, alloc_h);
			rect = r;
			clip = c;
		}

		let cell_w = (rect.w / config.cols as u32).max(1);
		let cell_h = (rect.h / config.rows as u32).max(1);

		for row in 0..config.rows {
			for col in 0..config.cols {
				let idx = (row * config.cols + col) as usize;
				let val = if idx < data_slice.len() {
					data_slice[idx]
				} else {
					0.0
				};
				let t = ((val - v_min) / range).clamp(0.0, 1.0);
				// Simple 3-stop colormap (cold=dark-blue, mid=orange, hot=yellow-white).
				let color = colormap_sample(t, config.colormap);
				let cx = rect.x + col * cell_w as i32;
				let cy = rect.y + row * cell_h as i32;
				let cell_rect = PixelRect {
					x: cx,
					y: cy,
					w: cell_w,
					h: cell_h,
				};
				if let Some(state) = self.state.as_mut() {
					state.cmds.push(UiCmd::Rect {
						rect: cell_rect,
						color,
						clip,
						corner_radius: 0.0,
					});
					if config.show_grid {
						state.cmds.push(UiCmd::RectOutline {
							rect: cell_rect,
							color: style.border,
							thickness: 1,
							clip,
							corner_radius: 0.0,
						});
					}
				}
			}
		}
		Ok(())
	}

	// ── Tier 3: Text variants ─────────────────────────────────────────────────

	/// Flow text widget (multi-word content, regular weight).
	///
	/// Equivalent to `label` but uses regular Sans weight for body text.
	///
	/// Donor: `oa::Ui::text`.
	pub fn text(&mut self, content: &str) {
		let Some(state) = self.state.as_mut() else {
			return;
		};
		let style = Self::current_style_ref(state).clone();
		let (rect, _clip) = allocate_widget(state, style.padding);
		if rect.w == 0 || rect.h == 0 {
			return;
		}
		let cfg = TextLayoutConfig {
			font: FontId::Sans,
			size: style.font_size,
			..Default::default()
		};
		Self::emit_text_or_record(state, content, rect, style.text, cfg);
	}

	/// Formatted label (printf-style convenience wrapper around `label`).
	///
	/// Donor: `oa::Ui::labelFmt`.
	pub fn label_fmt(&mut self, text: &str) {
		// In Rust the formatting is done at the call site with format!(); this
		// method exists for API parity so callers that pass pre-formatted strings
		// do not need to branch.
		self.label(text);
	}

	// ── Tier 3: Specialized buttons ───────────────────────────────────────────

	/// Donor geometric icon button with circular hover fill and cubic transitions.
	/// Icons use GPU lines, outlines and filled triangles, never Unicode substitutes.
	/// Donor: `oa::Ui::iconButton`.
	pub fn icon_button(
		&mut self,
		id: &str,
		rect: PixelRect,
		icon: UiIcon,
		enabled: bool,
		style_kind: UiIconButtonStyle,
	) -> bool {
		let Some(state) = self.state.as_mut() else {
			return false;
		};
		if rect.w == 0
			|| rect.h == 0
			|| rect.w > state.compose.width
			|| rect.h > state.compose.height
			|| rect.w > i32::MAX as u32
			|| rect.h > i32::MAX as u32
			|| rect.x.checked_add(rect.w as i32).is_none()
			|| rect.y.checked_add(rect.h as i32).is_none()
		{
			if state.recording_error.is_none() {
				state.recording_error = Some(Error::invalid_argument(
					"icon_button requires bounded nonempty geometry",
				));
			}
			return false;
		}
		let widget_id = widget_hash(id, Some(rect));
		let hovered = enabled && point_in_rect(state.input.pointer_x, state.input.pointer_y, rect);
		if hovered {
			state.hovered = widget_id;
		}
		if hovered && state.input.l_pressed {
			state.pressed = widget_id;
		}
		let activated = hovered && state.input.l_released && state.pressed == widget_id;
		let held = enabled && state.pressed == widget_id && state.input.pointer_down;
		let hover_mix = animate_hover(state, widget_id, hovered);
		let window_control = style_kind == UiIconButtonStyle::WindowControl;
		let inset = if window_control {
			(5.0 * state.content_scale).round() as u32
		} else {
			0
		};
		let fill_rect = if inset.saturating_mul(2) < rect.w.min(rect.h) {
			PixelRect::new(
				rect.x + inset as i32,
				rect.y + inset as i32,
				rect.w - inset * 2,
				rect.h - inset * 2,
			)
		} else {
			rect
		};
		let fill_alpha = if !enabled {
			0.0
		} else if held || activated {
			0.30
		} else {
			0.15 * hover_mix
		};
		if fill_alpha > 0.0 {
			state.cmds.push(UiCmd::Rect {
				rect: fill_rect,
				color: Color::new(1.0, 1.0, 1.0, fill_alpha),
				clip: rect,
				corner_radius: fill_rect.w.min(fill_rect.h) as f32 * 0.5,
			});
		}
		if state.focused == widget_id {
			state.cmds.push(UiCmd::RectOutline {
				rect,
				color: Self::current_style_ref(state).accent,
				thickness: 1,
				clip: rect,
				corner_radius: rect.w.min(rect.h) as f32 * 0.5,
			});
		}
		let alpha = if !enabled {
			0.38
		} else if style_kind == UiIconButtonStyle::Overlay {
			0.94
		} else if held || activated {
			1.0
		} else {
			0.68 + 0.32 * hover_mix
		};
		let color = Color::new(1.0, 1.0, 1.0, alpha);
		let transport = if enabled {
			Color::new(1.0, 1.0, 1.0, 1.0)
		} else {
			color
		};
		let viewport = PixelRect::new(0, 0, state.compose.width, state.compose.height);
		let clip = state
			.layout
			.top()
			.map_or(viewport, |panel| viewport.clip(&panel.clip()))
			.clip(&rect);
		if clip.w != 0 && clip.h != 0 {
			icons::draw(
				&mut state.cmds,
				rect,
				clip,
				icon,
				state.content_scale,
				window_control,
				[color, transport],
			);
		}
		activated
	}

	/// Render a dense FP32 matrix directly on the GPU, retaining its storage.
	/// Donor: Ui::heatmap/drawHeatmap.slang. Explicit limits avoid host auto-scaling.
	pub fn heatmap_matrix_at(
		&mut self,
		values: &crate::Matrix,
		rect: PixelRect,
		range: [f32; 2],
		colormap: u32,
	) -> Result<()> {
		let state = self
			.state
			.as_mut()
			.ok_or_else(|| Error::failed_precondition("Ui is closed"))?;
		if values.dtype() != crate::DType::F32
			|| values.shape().len() != 2
			|| values.shape().contains(&0)
			|| !values.engine_handle().same_as(&state.engine_handle)
			|| !range.iter().all(|v| v.is_finite())
			|| range[0] >= range[1]
			|| colormap > 3
		{
			return Err(Error::invalid_argument(
				"heatmap requires an owned dense FP32 matrix and finite increasing range",
			));
		}
		let rows = u32::try_from(values.shape()[0])
			.map_err(|_| Error::out_of_range("heatmap rows exceed u32"))?;
		let cols = u32::try_from(values.shape()[1])
			.map_err(|_| Error::out_of_range("heatmap columns exceed u32"))?;
		if rect.w == 0
			|| rect.h == 0
			|| rect.w > state.compose.width
			|| rect.h > state.compose.height
			|| rect
				.x
				.checked_add(
					i32::try_from(rect.w).map_err(|_| Error::out_of_range("heatmap width exceeds i32"))?,
				)
				.is_none()
			|| rect
				.y
				.checked_add(
					i32::try_from(rect.h).map_err(|_| Error::out_of_range("heatmap height exceeds i32"))?,
				)
				.is_none()
			|| u64::from(rows) * u64::from(cols) > u64::from(u32::MAX / 4)
			|| cols.checked_mul(rect.w).is_none()
			|| rows.checked_mul(rect.h).is_none()
		{
			return Err(Error::out_of_range(
				"heatmap geometry or indexing exceeds donor ABI",
			));
		}
		let viewport = PixelRect::new(0, 0, state.compose.width, state.compose.height);
		let clip = state
			.layout
			.top()
			.map_or(viewport, |panel| viewport.clip(&panel.clip()))
			.clip(&rect);
		let values_idx = state
			.engine_handle
			.storage_descriptor_index(values.storage())?;
		let push = [
			values_idx,
			rows,
			cols,
			0,
			rect.x as u32,
			rect.y as u32,
			rect.w,
			rect.h,
			range[0].to_bits(),
			range[1].to_bits(),
			colormap,
			0,
			0,
			0,
			clip.x as u32,
			clip.y as u32,
			clip.w,
			clip.h,
		];
		state.cmds.push(UiCmd::Heatmap {
			values_idx,
			rect,
			push,
		});
		state.frame_matrices.push(values.clone());
		Ok(())
	}

	/// Render a retained `[bins, 2]` min/max envelope without readback.
	/// Donor: Ui::waveformTimeline/drawWaveform.slang; fraction marks playback.
	pub fn waveform_matrix_at(
		&mut self,
		values: &crate::Matrix,
		rect: PixelRect,
		fraction: f32,
		played: Color,
		remaining: Color,
	) -> Result<()> {
		let state = self
			.state
			.as_mut()
			.ok_or_else(|| Error::failed_precondition("Ui is closed"))?;
		if values.dtype() != crate::DType::F32
			|| values.shape().len() != 2
			|| values.shape()[1] != 2
			|| values.shape()[0] == 0
			|| !values.engine_handle().same_as(&state.engine_handle)
			|| !fraction.is_finite()
			|| !played
				.to_array()
				.iter()
				.chain(remaining.to_array().iter())
				.all(|v| v.is_finite())
		{
			return Err(Error::invalid_argument(
				"waveform requires an owned FP32 [bins, 2] envelope and finite colors/fraction",
			));
		}
		let bins = u32::try_from(values.shape()[0])
			.map_err(|_| Error::out_of_range("waveform bins exceed u32"))?;
		if rect.w == 0
			|| rect.h == 0
			|| rect.w > state.compose.width
			|| rect.h > state.compose.height
			|| rect
				.x
				.checked_add(
					i32::try_from(rect.w).map_err(|_| Error::out_of_range("waveform width exceeds i32"))?,
				)
				.is_none()
			|| rect
				.y
				.checked_add(
					i32::try_from(rect.h).map_err(|_| Error::out_of_range("waveform height exceeds i32"))?,
				)
				.is_none()
			|| bins > u32::MAX / 8
			|| bins.checked_mul(rect.w).is_none()
		{
			return Err(Error::out_of_range(
				"waveform geometry or indexing exceeds donor ABI",
			));
		}
		let viewport = PixelRect::new(0, 0, state.compose.width, state.compose.height);
		let clip = state
			.layout
			.top()
			.map_or(viewport, |panel| viewport.clip(&panel.clip()))
			.clip(&rect);
		let values_idx = state
			.engine_handle
			.storage_descriptor_index(values.storage())?;
		let push = [
			values_idx,
			bins,
			0,
			rect.x as u32,
			rect.y as u32,
			rect.w,
			rect.h,
			fraction.clamp(0.0, 1.0).to_bits(),
			played.to_u32(),
			remaining.to_u32(),
			clip.x as u32,
			clip.y as u32,
			clip.w,
			clip.h,
		];
		state.cmds.push(UiCmd::Waveform {
			values_idx,
			rect,
			push,
		});
		state.frame_matrices.push(values.clone());
		Ok(())
	}

	/// GPU envelope scrubber with the donor min/max bars and playhead.
	/// Donor: `oa::Ui::waveformTimeline`; source storage is retained until completion.
	pub fn waveform_matrix_timeline(
		&mut self,
		id: &str,
		rect: PixelRect,
		values: &crate::Matrix,
		fraction: &mut f32,
	) -> Result<bool> {
		let state = self
			.state
			.as_mut()
			.ok_or_else(|| Error::failed_precondition("Ui is closed"))?;
		let widget_id = widget_hash(id, Some(rect));
		let hovered = point_in_rect(state.input.pointer_x, state.input.pointer_y, rect);
		if hovered {
			state.hovered = widget_id;
		}
		if hovered && state.input.l_pressed {
			state.pressed = widget_id;
			state.focused = widget_id;
		}
		let held = state.pressed == widget_id && state.input.pointer_down;
		let activated = hovered && state.input.l_released && state.pressed == widget_id;
		let mut changed = false;
		if held || activated {
			let next = ((state.input.pointer_x - rect.x as f32) / rect.w.max(1) as f32).clamp(0.0, 1.0);
			changed = next != *fraction;
			*fraction = next;
		}
		let style = Self::current_style_ref(state).clone();
		let scale = state.content_scale;
		let remaining = Color {
			a: 0.60,
			..style.text_secondary
		};
		self.waveform_matrix_at(values, rect, *fraction, style.accent, remaining)?;
		let base = ((20.0 * scale).round().max(1.0) as u32).min(rect.h);
		let center = rect.x
			+ base as i32 / 2
			+ (rect.w.saturating_sub(base) as f32 * fraction.clamp(0.0, 1.0)).round() as i32;
		let width = ((if hovered || held { 3.0 } else { 2.0 }) * scale)
			.round()
			.max(1.0) as u32;
		let width = width.min(rect.w);
		let x = (center - width as i32 / 2).clamp(rect.x, rect.x + rect.w as i32 - width as i32);
		self.rect_raw(
			PixelRect::new(x, rect.y, width, rect.h),
			if hovered || held {
				style.accent_hover
			} else {
				style.text
			},
			rect,
			0.0,
		);
		Ok(changed)
	}

	/// Apply the donor premultiplied rounded mask after drawing a viewport.
	pub fn mask_rounded_rect(&mut self, rect: PixelRect, radius: f32, outside: Color) -> Result<()> {
		let state = self
			.state
			.as_mut()
			.ok_or_else(|| Error::failed_precondition("Ui is closed"))?;
		if rect.w == 0
			|| rect.h == 0
			|| rect.w > state.compose.width
			|| rect.h > state.compose.height
			|| rect.x.checked_add(rect.w as i32).is_none()
			|| rect.y.checked_add(rect.h as i32).is_none()
			|| !radius.is_finite()
			|| radius < 0.0
			|| !outside.to_array().iter().all(|v| v.is_finite())
		{
			return Err(Error::invalid_argument(
				"rounded mask requires bounded geometry and finite radius/color",
			));
		}
		let viewport = PixelRect::new(0, 0, state.compose.width, state.compose.height);
		let clip = state
			.layout
			.top()
			.map_or(viewport, |panel| viewport.clip(&panel.clip()))
			.clip(&rect);
		if clip.w != 0 && clip.h != 0 {
			state.cmds.push(UiCmd::RoundedMask {
				rect,
				radius,
				outside,
				clip,
			});
		}
		Ok(())
	}

	/// Chevron (prev/next) button.  Returns `true` on click.
	///
	/// Donor: `oa::Ui::chevronButton`.
	pub fn chevron_button(
		&mut self,
		id: &str,
		rect: PixelRect,
		direction: UiChevronDirection,
		enabled: bool,
	) -> bool {
		let icon = match direction {
			UiChevronDirection::Previous => UiIcon::Previous,
			UiChevronDirection::Next => UiIcon::Next,
		};
		self.icon_button(id, rect, icon, enabled, UiIconButtonStyle::Header)
	}

	/// Text button with explicit rect and text config.
	///
	/// Donor: `oa::Ui::textButton`.
	pub fn text_button(
		&mut self,
		id: &str,
		rect: PixelRect,
		text: &str,
		text_color: Color,
		enabled: bool,
	) -> bool {
		let Some(state) = self.state.as_mut() else {
			return false;
		};
		let style = Self::current_style_ref(state).clone();
		let widget_id = widget_hash(id, Some(rect));
		let clip = rect;

		let hovered = enabled && point_in_rect(state.input.pointer_x, state.input.pointer_y, rect);
		let activated = hovered && was_pressed_and_released(state, widget_id);
		if hovered {
			state.hovered = widget_id;
		}
		if hovered && state.input.pointer_down {
			state.pressed = widget_id;
		}

		let bg = if state.pressed == widget_id {
			Color::new(
				style.accent.r * 0.7,
				style.accent.g * 0.7,
				style.accent.b * 0.7,
				1.0,
			)
		} else if hovered {
			style.surface
		} else {
			Color::new(0.0, 0.0, 0.0, 0.0)
		};
		if bg.a > 0.0 {
			state.cmds.push(UiCmd::Rect {
				rect,
				color: bg,
				clip,
				corner_radius: style.corner_radius,
			});
		}
		let effective_color = if enabled {
			text_color
		} else {
			style.text_secondary
		};
		let cfg = TextLayoutConfig {
			font: FontId::Sans,
			size: style.font_size,
			..Default::default()
		};
		Self::emit_text_or_record(state, text, rect, effective_color, cfg);
		activated
	}

	// ── Tier 3: Regular grid ──────────────────────────────────────────────────

	/// Draw the donor decimal grid and gradient in one GPU dispatch.
	///
	/// Invalid geometry/configuration is reported by the next render boundary,
	/// matching other infallible widget APIs. Donor: `oa::Ui::grid`.
	pub fn grid(&mut self, rect: PixelRect, config: UiGridConfig) {
		let Some(state) = self.state.as_mut() else {
			return;
		};
		let thicknesses = [
			config.minor_thickness,
			config.major_thickness,
			config.super_major_thickness,
			config.axis_thickness,
		];
		let colors = [
			config.background_top,
			config.background_bottom,
			config.minor_color,
			config.major_color,
			config.super_major_color,
			config.axis_color,
		];
		let valid = rect.w != 0
			&& rect.h != 0
			&& rect.w <= i32::MAX as u32
			&& rect.h <= i32::MAX as u32
			&& thicknesses
				.iter()
				.all(|v| v.is_finite() && *v > 0.0 && *v <= 16.0)
			&& config
				.origin
				.iter()
				.chain(config.guide_fade_center.iter())
				.all(|v| v.is_finite())
			&& config
				.minor_spacing
				.iter()
				.all(|v| v.is_finite() && *v > 0.0)
			&& config.opacity.is_finite()
			&& (0.0..=1.0).contains(&config.opacity)
			&& (2..=100).contains(&config.major_every)
			&& (config.major_every.saturating_mul(2)..=10000).contains(&config.super_major_every)
			&& config
				.super_major_every
				.is_multiple_of(config.major_every.max(1))
			&& config.guide_fade_inner_radius.is_finite()
			&& config.guide_fade_inner_radius >= 0.0
			&& config.guide_fade_outer_radius.is_finite()
			&& config.guide_fade_outer_radius > config.guide_fade_inner_radius
			&& config.guide_fade_outer_radius <= 1.0e7
			&& colors
				.iter()
				.all(|c| [c.r, c.g, c.b, c.a].iter().all(|v| v.is_finite()));
		if !valid {
			if state.recording_error.is_none() {
				state.recording_error = Some(Error::invalid_argument(
					"ui::grid requires finite geometry, positive spacing/thickness/fade radii, opacity in [0,1], and divisible tiers",
				));
			}
			return;
		}
		// Wide checked geometry avoids overflow before clipping off-screen rectangles.
		let parent = state.layout.top().map_or(
			PixelRect::new(0, 0, state.compose.width, state.compose.height),
			|panel| panel.clip(),
		);
		let left = i64::from(rect.x).max(i64::from(parent.x)).max(0);
		let top = i64::from(rect.y).max(i64::from(parent.y)).max(0);
		let right = (i64::from(rect.x) + i64::from(rect.w))
			.min(i64::from(parent.x) + i64::from(parent.w))
			.min(i64::from(state.compose.width));
		let bottom = (i64::from(rect.y) + i64::from(rect.h))
			.min(i64::from(parent.y) + i64::from(parent.h))
			.min(i64::from(state.compose.height));
		if right <= left || bottom <= top {
			return;
		}
		let clip = PixelRect::new(
			left as i32,
			top as i32,
			(right - left) as u32,
			(bottom - top) as u32,
		);
		let style = Self::current_style_ref(state);
		let light = style.background.r + style.background.g + style.background.b > 1.5;
		let resolve = |requested: Color, fallback: Color| {
			if requested.a <= 0.0 {
				fallback
			} else {
				requested
			}
		};
		let line = |requested: Color, fallback: Color| {
			let mut color = resolve(requested, fallback);
			color.a *= config.opacity;
			color.to_u32()
		};
		let top_color = Color::new(
			style.surface.r + (1.0 - style.surface.r) * 0.055,
			style.surface.g + (1.0 - style.surface.g) * 0.055,
			style.surface.b + (1.0 - style.surface.b) * 0.055,
			style.surface.a,
		);
		let bottom_color = Color::new(
			style.surface.r * 0.955,
			style.surface.g * 0.955,
			style.surface.b * 0.955,
			style.surface.a,
		);
		let minor = if light {
			Color::new(0.0, 0.0, 0.0, 0.045)
		} else {
			Color::new(1.0, 1.0, 1.0, 0.045)
		};
		let major = if light {
			Color::new(0.0, 0.0, 0.0, 0.14)
		} else {
			Color::new(0.78, 0.80, 0.82, 0.16)
		};
		let super_major = if light {
			Color::new(0.12, 0.12, 0.12, 0.24)
		} else {
			Color::new(0.03, 0.03, 0.03, 0.62)
		};
		let axis = Color::new(0.0, 0.0, 0.0, if light { 0.60 } else { 0.82 });
		let flags = u32::from(config.fill_background)
			| (u32::from(config.draw_axes) << 1)
			| (u32::from(config.draw_grid) << 2)
			| (u32::from(config.radial_guide_fade) << 3)
			| (u32::from(config.dither_background) << 4);
		// Exactly 120 bytes, matching the donor/reflected shader ABI.
		let push = [
			0,
			clip.x as u32,
			clip.y as u32,
			clip.w,
			clip.h,
			config.origin[0].to_bits(),
			config.origin[1].to_bits(),
			config.minor_spacing[0].to_bits(),
			config.minor_spacing[1].to_bits(),
			config.major_every,
			config.super_major_every,
			config.minor_thickness.to_bits(),
			config.major_thickness.to_bits(),
			config.super_major_thickness.to_bits(),
			config.axis_thickness.to_bits(),
			resolve(config.background_top, top_color).to_u32(),
			resolve(config.background_bottom, bottom_color).to_u32(),
			line(config.minor_color, minor),
			line(config.major_color, major),
			line(config.super_major_color, super_major),
			line(config.axis_color, axis),
			config.guide_fade_center[0].to_bits(),
			config.guide_fade_center[1].to_bits(),
			config.guide_fade_inner_radius.to_bits(),
			config.guide_fade_outer_radius.to_bits(),
			flags,
			clip.x as u32,
			clip.y as u32,
			clip.w,
			clip.h,
		];
		state.cmds.push(UiCmd::CanvasGrid { rect: clip, push });
	}

	// ── Tier 3: Planar image ──────────────────────────────────────────────────

	/// Composite all channels of a planar multi-dtype image into one image.
	///
	/// Inside a panel the image scales to the panel rectangle; otherwise it
	/// draws at native size at `(dst_x, dst_y)`. Plane buffers stay retained
	/// until the frame completes.
	///
	/// Donor: `oa::Ui::imagePlanar`.
	pub fn image_planar(
		&mut self,
		planes: &crate::ui::ImagePlanes,
		dst_x: i32,
		dst_y: i32,
	) -> Result<()> {
		let state = self
			.state
			.as_mut()
			.ok_or_else(|| Error::failed_precondition("Ui is closed"))?;
		if !planes.is_valid() || !planes.engine_handle().same_as(&state.engine_handle) {
			return Err(Error::invalid_argument(
				"UI image_planar requires valid planes from its engine",
			));
		}
		let mut slots = [u32::MAX; crate::ui::IMAGE_MAX_PLANES];
		for (channel, slot) in slots.iter_mut().enumerate() {
			*slot = planes.plane_index(channel)?;
		}
		self.push_planar_cmd(
			planes,
			slots,
			(0..planes.channel_count()).fold(0_u32, |packed, channel| {
				packed | ((planes.plane_dtype(channel) as u32) & 0x3) << (channel * 2)
			}),
			u32::try_from(planes.channel_count())
				.map_err(|_| Error::out_of_range("plane count exceeds donor ABI"))?,
			dst_x,
			dst_y,
		)
	}

	/// Blit one channel of a planar image as grayscale.
	///
	/// Donor: `oa::Ui::imagePlane`.
	pub fn image_plane(
		&mut self,
		planes: &crate::ui::ImagePlanes,
		channel: usize,
		dst_x: i32,
		dst_y: i32,
	) -> Result<()> {
		let state = self
			.state
			.as_mut()
			.ok_or_else(|| Error::failed_precondition("Ui is closed"))?;
		if !planes.is_valid()
			|| !planes.engine_handle().same_as(&state.engine_handle)
			|| channel >= planes.channel_count()
		{
			return Err(Error::invalid_argument(
				"UI image_plane requires a valid channel from its engine",
			));
		}
		let index = planes.plane_index(channel)?;
		self.push_planar_cmd(
			planes,
			[index, u32::MAX, u32::MAX, u32::MAX],
			u32::from(planes.plane_dtype(channel) as u8) & 0x3,
			1,
			dst_x,
			dst_y,
		)
	}

	fn push_planar_cmd(
		&mut self,
		planes: &crate::ui::ImagePlanes,
		slots: [u32; 4],
		dtypes: u32,
		channels: u32,
		dst_x: i32,
		dst_y: i32,
	) -> Result<()> {
		let state = self
			.state
			.as_mut()
			.ok_or_else(|| Error::failed_precondition("Ui is closed"))?;
		let native_w =
			u32::try_from(planes.width()).map_err(|_| Error::out_of_range("plane width exceeds u32"))?;
		let native_h = u32::try_from(planes.height())
			.map_err(|_| Error::out_of_range("plane height exceeds u32"))?;
		let (x, y, w, h) = state.layout.top().map_or_else(
			|| (dst_x, dst_y, native_w, native_h),
			|panel| {
				(
					panel.outer.x + dst_x,
					panel.outer.y + dst_y,
					panel.outer.w,
					panel.outer.h,
				)
			},
		);
		let viewport = PixelRect::new(0, 0, state.compose.width, state.compose.height);
		let destination = PixelRect::new(x, y, w, h);
		let clip = state
			.layout
			.top()
			.map_or(viewport, |panel| viewport.clip(&panel.clip()))
			.clip(&destination);
		if clip.w == 0 || clip.h == 0 {
			return Ok(());
		}
		state.cmds.push(UiCmd::BlitPlanar {
			r_idx: slots[0],
			g_idx: slots[1],
			b_idx: slots[2],
			a_idx: slots[3],
			src_w: u32::try_from(planes.width())
				.map_err(|_| Error::out_of_range("plane width exceeds donor ABI"))?,
			src_h: u32::try_from(planes.height())
				.map_err(|_| Error::out_of_range("plane height exceeds donor ABI"))?,
			dst_x: x,
			dst_y: y,
			dtypes,
			channels,
			dst_w: w,
			dst_h: h,
			clip,
		});
		planes.retain_planes(&mut state.frame_matrices);
		Ok(())
	}

	// ── Tier 4: Plot additions ────────────────────────────────────────────────

	/// Draw an explicit X/Y line series into an explicit pixel rectangle.
	///
	/// `x` and `y` must have the same length.
	///
	/// Donor: `oa::Ui::plotLineXY`.
	#[allow(clippy::too_many_arguments)]
	pub fn plot_line_xy_at(
		&mut self,
		x: &[f32],
		y: &[f32],
		rect: PixelRect,
		y_min: f32,
		y_max: f32,
		color: Color,
		line_width: f32,
	) -> Result<()> {
		if x.len() != y.len() {
			return Err(Error::invalid_argument(
				"plot_line_xy: x and y must have equal length",
			));
		}
		if x.is_empty() {
			return Ok(());
		}
		if rect.w == 0 || rect.h == 0 || !y_min.is_finite() || !y_max.is_finite() || y_min >= y_max {
			return Err(Error::invalid_argument(
				"UI XY line plot requires a positive rectangle and finite increasing Y limits",
			));
		}
		// Interleave x, y into a packed buffer [x0, y0, x1, y1, ...].
		let mut packed: Vec<f32> = Vec::with_capacity(x.len() * 2);
		for (xi, yi) in x.iter().zip(y.iter()) {
			packed.push(*xi);
			packed.push(*yi);
		}
		let x_min = x.iter().cloned().fold(f32::INFINITY, f32::min);
		let x_max = x.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
		let matrix = crate::Matrix::new_from_f32(self.engine, [packed.len()], &packed)?;
		let descriptor = matrix
			.engine_handle()
			.storage_descriptor_index(matrix.storage())?;
		let state = self
			.state
			.as_mut()
			.ok_or_else(|| Error::failed_precondition("Ui is closed"))?;
		state.cmds.push(UiCmd::PlotLine {
			values_idx: descriptor,
			count: u32::try_from(x.len()).map_err(|_| Error::out_of_range("sample count exceeds u32"))?,
			rect,
			clip: PixelRect {
				x: 0,
				y: 0,
				w: state.compose.width,
				h: state.compose.height,
			},
			x_min,
			x_max,
			y_min,
			y_max,
			color,
			fill: false,
			explicit_xy: true,
			antialias_samples: 4,
			line_width,
		});
		state.frame_matrices.push(matrix);
		Ok(())
	}

	/// Draw a ring-buffer line series (circular / most-recent-N view).
	///
	/// `offset` is the index of the oldest sample in `data`.
	///
	/// Donor: `oa::Ui::plotLineRing`.
	#[allow(clippy::too_many_arguments)]
	pub fn plot_line_ring_at(
		&mut self,
		data: &[f32],
		offset: usize,
		rect: PixelRect,
		y_min: f32,
		y_max: f32,
		color: Color,
		line_width: f32,
	) -> Result<()> {
		if data.is_empty() {
			return Ok(());
		}
		// Rotate the ring buffer so the oldest sample comes first.
		let mut ordered: Vec<f32> = Vec::with_capacity(data.len());
		ordered.extend_from_slice(&data[offset..]);
		ordered.extend_from_slice(&data[..offset]);
		self.plot_line_at(&ordered, rect, y_min, y_max, color, line_width)
	}

	// ── Tier 4: Accessibility ─────────────────────────────────────────────────

	/// Returns a lightweight snapshot of the current widget tree for
	/// accessibility tooling.
	///
	/// This implementation returns the node count only; full AT-SPI integration
	/// remains Planned.
	///
	/// Donor: `oa::Ui::accessibilitySnapshot`.
	pub fn accessibility_snapshot(&self) -> UiAccessibilitySnapshot {
		let count = self.state.as_ref().map_or(0, |s| s.cmds.cmds.len());
		UiAccessibilitySnapshot { node_count: count }
	}

	/// Returns `true` when a text input widget currently has focus and the
	/// platform IME should be enabled.
	///
	/// Donor: `oa::Ui::wantsTextInput`.
	pub fn wants_text_input(&self) -> bool {
		self.state.as_ref().is_some_and(|s| s.focused != 0)
	}

	/// Returns the pixel rect of the focused text input widget, if any.
	/// Returns `None` when no text field is focused.
	///
	/// Donor: `oa::Ui::textInputRect`.
	pub fn text_input_rect(&self) -> Option<PixelRect> {
		let state = self.state.as_ref()?;
		if state.focused == 0 {
			return None;
		}
		// Return the innermost panel's current cursor position as a best-effort
		// approximation; exact per-widget rect tracking requires a widget registry.
		state.layout.top().map(|s| PixelRect {
			x: s.cursor[0],
			y: s.cursor[1],
			w: s.inner.w,
			h: default_item_height(0.0),
		})
	}

	/// Emit positioned glyph quads from a pre-shaped buffer.
	///
	/// `glyphs` is a slice of `(x, y, codepoint, color_rgba)` values built by
	/// the caller using the text shaping API.  This is the low-level path used
	/// by custom text renderers.
	///
	/// Donor: `oa::Ui::glyphs`.
	pub fn glyphs_at(
		&mut self,
		text: &str,
		origin: [f32; 2],
		config: TextLayoutConfig,
		color: Color,
		clip: PixelRect,
	) -> Result<()> {
		self.text_at(text, origin, config, color, clip)
	}

	/// Vertically center text in `rect` and emit a `Glyphs` command.
	///
	/// Private static helper used by widget implementations that already hold
	/// `&mut UiState`.
	fn emit_text(
		state: &mut UiState,
		text: &str,
		rect: PixelRect,
		color: Color,
		config: TextLayoutConfig,
	) -> Result<()> {
		if text.is_empty() || rect.w == 0 || rect.h == 0 {
			return Ok(());
		}
		let baseline_y = rect.y as f32 + rect.h as f32 * 0.5 + config.size * 0.35;
		let origin = [rect.x as f32 + 4.0, baseline_y];
		let positioned = TextLayout::shape(&state.text_atlas, text, origin, config, color)?;
		if positioned.is_empty() {
			return Ok(());
		}
		let mut bytes = Vec::with_capacity(positioned.len() * 48);
		for glyph in positioned {
			let Some(info) = state
				.text_atlas
				.find_glyph(glyph.font, glyph.codepoint, config.size)
			else {
				continue;
			};
			let scale = config.size / info.raster_size;
			for value in [
				0.0_f32,
				0.0,
				glyph.x + info.bearing_x * scale,
				glyph.y - info.bearing_y * scale,
				info.atlas_w * scale,
				info.atlas_h * scale,
			] {
				bytes.extend_from_slice(&value.to_le_bytes());
			}
			for value in [
				info.atlas_x as u32,
				info.atlas_y as u32,
				info.atlas_w as u32,
				info.atlas_h as u32,
				glyph.color.to_u32(),
				0,
			] {
				bytes.extend_from_slice(&value.to_le_bytes());
			}
		}
		if bytes.is_empty() {
			return Ok(());
		}
		let clip = rect;
		let storage = state.engine_handle.create_storage(&bytes)?;
		let glyph_idx = state.engine_handle.storage_descriptor_index(&storage)?;
		let count = u32::try_from(bytes.len() / 48)
			.map_err(|_| Error::out_of_range("glyph count exceeds u32"))?;
		state.cmds.push(UiCmd::Glyphs {
			glyph_idx,
			atlas_idx: state.text_atlas.descriptor(),
			count,
			dst: None,
			clip,
			atlas_w: state.text_atlas.width(),
			atlas_h: state.text_atlas.height(),
		});
		state.frame_storages.push(storage);
		Ok(())
	}

	fn emit_text_or_record(
		state: &mut UiState,
		text: &str,
		rect: PixelRect,
		color: Color,
		config: TextLayoutConfig,
	) {
		if let Err(error) = Self::emit_text(state, text, rect, color, config)
			&& state.recording_error.is_none()
		{
			state.recording_error = Some(error);
		}
	}

	fn frame_bindings(state: &UiState) -> Result<Option<std::rc::Rc<UiFrameBindings>>> {
		state.engine_handle.prepare_ui_sources()?;
		let Some(layouts) = &state.pipelines.bounded else {
			return Ok(None);
		};
		let engine = &state.engine_handle;
		let storages = state
			.frame_storages
			.iter()
			.chain(std::iter::once(state.text_atlas.storage()))
			.chain(state.frame_matrices.iter().map(crate::Matrix::storage))
			.chain(
				state
					.frame_textures
					.iter()
					.filter_map(crate::Texture::packed_storage),
			);
		let mut buffers = std::collections::HashMap::new();
		for storage in storages {
			buffers.insert(
				engine.storage_descriptor_index(storage)?,
				engine.ui_buffer_info(storage)?,
			);
		}
		Ok(Some(std::rc::Rc::new(UiFrameBindings::new(
			engine,
			layouts,
			engine.ui_compose_view(state.compose.raw_image())?,
			&state.cmds.cmds,
			|index| {
				buffers
					.get(&index)
					.copied()
					.ok_or_else(|| Error::internal("UI command source is not retained"))
			},
		)?)))
	}

	fn record_render_state(
		state: &UiState,
		device: &vk::Device,
		command_buffer: vk::CommandBuffer,
		descriptor_set: vk::DescriptorSet,
		pipeline_layout: vk::PipelineLayout,
		bindings: Option<&UiFrameBindings>,
	) -> Result<()> {
		let dst_idx = if bindings.is_some() {
			0
		} else {
			state.compose.bindless_index
		};

		// Transition compose image to GENERAL for compute writes.
		let range = vk::ImageSubresourceRange::new()
			.set_aspect_mask(vk::ImageAspectFlagBits::COLOR_BIT)
			.set_level_count(1)
			.set_layer_count(1);
		let prior_layout = state.compose.layout();
		let barrier_in = vk::ImageMemoryBarrier2::new()
			.set_src_stage_mask(if prior_layout == vk::ImageLayout::UNDEFINED {
				vk::PipelineStageFlagBits2::NONE
			} else {
				vk::PipelineStageFlagBits2::ALL_TRANSFER_BIT
			})
			.set_src_access_mask(if prior_layout == vk::ImageLayout::UNDEFINED {
				vk::AccessFlagBits2::NONE
			} else {
				vk::AccessFlagBits2::TRANSFER_READ_BIT
			})
			.set_dst_stage_mask(vk::PipelineStageFlagBits2::COMPUTE_SHADER_BIT)
			.set_dst_access_mask(vk::AccessFlagBits2::SHADER_STORAGE_WRITE_BIT)
			.set_old_layout(prior_layout)
			.set_new_layout(vk::ImageLayout::GENERAL)
			.set_image(state.compose.raw_image())
			.set_subresource_range(range);

		// SAFETY: the Engine supplies a recording command buffer and retains the
		// compose image. The barrier preserves its prior layout and compute-write scope.
		unsafe {
			state.sync_commands.pipeline_barrier(
				device,
				command_buffer,
				&vk::DependencyInfo::new()
					.set_image_memory_barriers(&[barrier_in])
					.map_err(Error::from)?
					.set_memory_barriers(&[vk::MemoryBarrier2::new()
						.set_src_stage_mask(
							vk::PipelineStageFlagBits2::ALL_COMMANDS_BIT | vk::PipelineStageFlagBits2::HOST_BIT,
						)
						.set_src_access_mask(vk::AccessFlagBits2::MEMORY_WRITE_BIT)
						.set_dst_stage_mask(vk::PipelineStageFlagBits2::COMPUTE_SHADER_BIT)
						.set_dst_access_mask(
							vk::AccessFlagBits2::SHADER_STORAGE_READ_BIT
								| vk::AccessFlagBits2::SHADER_STORAGE_WRITE_BIT,
						)])
					.map_err(Error::from)?,
			)?;

			// Strict uses the shared heap; bounded sets are bound per command.
			if bindings.is_none() {
				device.cmd_bind_descriptor_sets(
					command_buffer,
					vk::PipelineBindPoint::COMPUTE,
					pipeline_layout,
					0,
					&[descriptor_set],
					&[],
				)?;
			}
		}

		// Replay each command. Alpha composition reads pixels written by the
		// preceding dispatch, so preserve that storage-image RAW dependency.
		let commands = bindings.map_or(state.cmds.cmds.as_slice(), |bindings| {
			bindings.commands.as_slice()
		});
		for (index, cmd) in commands.iter().enumerate() {
			if let Some(bindings) = bindings {
				// SAFETY: each immutable set matches the command's exact layout;
				// its sources and pool remain retained through completion.
				unsafe {
					device.cmd_bind_descriptor_sets(
						command_buffer,
						vk::PipelineBindPoint::COMPUTE,
						bindings.layouts[index],
						0,
						&[bindings.sets[index]],
						&[],
					)?;
				}
			}
			if index != 0 {
				let dependency = vk::MemoryBarrier2::new()
					.set_src_stage_mask(vk::PipelineStageFlagBits2::COMPUTE_SHADER_BIT)
					.set_src_access_mask(vk::AccessFlagBits2::SHADER_STORAGE_WRITE_BIT)
					.set_dst_stage_mask(vk::PipelineStageFlagBits2::COMPUTE_SHADER_BIT)
					.set_dst_access_mask(
						vk::AccessFlagBits2::SHADER_STORAGE_READ_BIT
							| vk::AccessFlagBits2::SHADER_STORAGE_WRITE_BIT,
					);
				// SAFETY: consecutive dispatches share the retained compose image;
				// these scopes make preceding writes visible to alpha reads/writes.
				unsafe {
					state.sync_commands.pipeline_barrier(
						device,
						command_buffer,
						&(vk::DependencyInfo::new()
							.set_memory_barriers(std::slice::from_ref(&dependency))
							.map_err(Error::from)?),
					)?;
				}
			}
			Self::record_cmd(state, device, command_buffer, dst_idx, cmd)?;
		}

		Ok(())
	}

	fn record_cmd(
		state: &UiState,
		device: &vk::Device,
		cmd_buf: vk::CommandBuffer,
		dst_idx: u32,
		cmd: &UiCmd,
	) -> Result<()> {
		let w = state.compose.width;
		let h = state.compose.height;

		match cmd {
			UiCmd::Clear { color } => {
				let pipeline = &state.pipelines.clear_compose;
				let layout = pipeline.layout();
				// push: dst_idx, dst_w, dst_h, rgba
				let push = [dst_idx, w, h, color.to_u32()];
				let push_bytes = bytemuck_push(&push);
				unsafe {
					device.cmd_bind_pipeline(cmd_buf, vk::PipelineBindPoint::COMPUTE, pipeline.raw())?;
					device.cmd_push_constants(
						cmd_buf,
						layout,
						vk::ShaderStageFlagBits::COMPUTE_BIT,
						0,
						push_bytes,
					)?;
					device.cmd_dispatch(cmd_buf, div_ceil(w, 8), div_ceil(h, 8), 1)?;
				}
			}
			UiCmd::Heatmap {
				values_idx,
				rect,
				push,
			} => {
				let pipeline = &state.pipelines.draw_heatmap;
				let mut push = *push;
				push[0] = *values_idx;
				push[3] = dst_idx;
				// SAFETY: retained matrix/image and exact 72-byte donor ABI.
				unsafe {
					device.cmd_bind_pipeline(cmd_buf, vk::PipelineBindPoint::COMPUTE, pipeline.raw())?;
					device.cmd_push_constants(
						cmd_buf,
						pipeline.layout(),
						vk::ShaderStageFlagBits::COMPUTE_BIT,
						0,
						bytemuck_push(&push),
					)?;
					device.cmd_dispatch(cmd_buf, div_ceil(rect.w, 8), div_ceil(rect.h, 8), 1)?;
				}
			}
			UiCmd::Waveform {
				values_idx,
				rect,
				push,
			} => {
				let pipeline = &state.pipelines.draw_waveform;
				let mut push = *push;
				push[0] = *values_idx;
				push[2] = dst_idx;
				// SAFETY: retained matrix/image and exact 56-byte donor ABI; columns own disjoint pixels.
				unsafe {
					device.cmd_bind_pipeline(cmd_buf, vk::PipelineBindPoint::COMPUTE, pipeline.raw())?;
					device.cmd_push_constants(
						cmd_buf,
						pipeline.layout(),
						vk::ShaderStageFlagBits::COMPUTE_BIT,
						0,
						bytemuck_push(&push),
					)?;
					device.cmd_dispatch(cmd_buf, div_ceil(rect.w, 64), 1, 1)?;
				}
			}
			UiCmd::RoundedMask {
				rect,
				radius,
				outside,
				clip,
			} => {
				let pipeline = &state.pipelines.mask_rounded_viewport;
				let push = [
					dst_idx,
					rect.x as u32,
					rect.y as u32,
					rect.w,
					rect.h,
					radius.to_bits(),
					outside.to_u32(),
					clip.x as u32,
					clip.y as u32,
					clip.w,
					clip.h,
				];
				// SAFETY: exact 44-byte donor ABI; retained compose image/layout and bounded dispatch.
				unsafe {
					device.cmd_bind_pipeline(cmd_buf, vk::PipelineBindPoint::COMPUTE, pipeline.raw())?;
					device.cmd_push_constants(
						cmd_buf,
						pipeline.layout(),
						vk::ShaderStageFlagBits::COMPUTE_BIT,
						0,
						bytemuck_push(&push),
					)?;
					device.cmd_dispatch(cmd_buf, div_ceil(rect.w, 8), div_ceil(rect.h, 8), 1)?;
				}
			}
			UiCmd::Triangle {
				points,
				color,
				bounds,
			} => {
				let pipeline = &state.pipelines.draw_triangle;
				let push = [
					dst_idx,
					points[0][0].to_bits(),
					points[0][1].to_bits(),
					points[1][0].to_bits(),
					points[1][1].to_bits(),
					points[2][0].to_bits(),
					points[2][1].to_bits(),
					color.to_u32(),
					bounds.x as u32,
					bounds.y as u32,
					bounds.w,
					bounds.h,
				];
				// SAFETY: exact 48-byte donor ABI; retained image/layout, nonnegative bounds.
				unsafe {
					device.cmd_bind_pipeline(cmd_buf, vk::PipelineBindPoint::COMPUTE, pipeline.raw())?;
					device.cmd_push_constants(
						cmd_buf,
						pipeline.layout(),
						vk::ShaderStageFlagBits::COMPUTE_BIT,
						0,
						bytemuck_push(&push),
					)?;
					device.cmd_dispatch(cmd_buf, div_ceil(bounds.w, 8), div_ceil(bounds.h, 8), 1)?;
				}
			}
			UiCmd::CanvasGrid { rect, push } => {
				let pipeline = &state.pipelines.draw_canvas_grid;
				let mut push = *push;
				push[0] = dst_idx;
				// SAFETY: 120-byte reflected ABI, exact retained layout/image, bounded dispatch.
				unsafe {
					device.cmd_bind_pipeline(cmd_buf, vk::PipelineBindPoint::COMPUTE, pipeline.raw())?;
					device.cmd_push_constants(
						cmd_buf,
						pipeline.layout(),
						vk::ShaderStageFlagBits::COMPUTE_BIT,
						0,
						bytemuck_push(&push),
					)?;
					device.cmd_dispatch(cmd_buf, div_ceil(rect.w, 8), div_ceil(rect.h, 8), 1)?;
				}
			}
			UiCmd::Rect {
				rect,
				color,
				clip,
				corner_radius,
			} => {
				let pipeline = &state.pipelines.draw_rect;
				let layout = pipeline.layout();
				// push: dst_idx, dst_x, dst_y, dst_w, dst_h, rgba, clip_x, clip_y, clip_w, clip_h, corner_radius
				let mut push = [0u8; 44];
				write_u32(&mut push, 0, dst_idx);
				write_i32(&mut push, 4, rect.x);
				write_i32(&mut push, 8, rect.y);
				write_u32(&mut push, 12, rect.w);
				write_u32(&mut push, 16, rect.h);
				write_u32(&mut push, 20, color.to_u32());
				let cw = clip_words(*clip);
				write_i32(&mut push, 24, cw[0]);
				write_i32(&mut push, 28, cw[1]);
				write_i32(&mut push, 32, cw[2]);
				write_i32(&mut push, 36, cw[3]);
				write_f32(&mut push, 40, *corner_radius);
				unsafe {
					device.cmd_bind_pipeline(cmd_buf, vk::PipelineBindPoint::COMPUTE, pipeline.raw())?;
					device.cmd_push_constants(
						cmd_buf,
						layout,
						vk::ShaderStageFlagBits::COMPUTE_BIT,
						0,
						&push,
					)?;
					device.cmd_dispatch(cmd_buf, div_ceil(rect.w, 8), div_ceil(rect.h, 8), 1)?;
				}
			}
			UiCmd::RectOutline {
				rect,
				color,
				thickness,
				clip,
				corner_radius,
			} => {
				let pipeline = &state.pipelines.draw_rect_outline;
				let layout = pipeline.layout();
				// push: dst_idx, dst_x, dst_y, dst_w, dst_h, thickness, rgba, clip_x, clip_y, clip_w, clip_h, corner_radius
				let mut push = [0u8; 48];
				write_u32(&mut push, 0, dst_idx);
				write_i32(&mut push, 4, rect.x);
				write_i32(&mut push, 8, rect.y);
				write_u32(&mut push, 12, rect.w);
				write_u32(&mut push, 16, rect.h);
				write_u32(&mut push, 20, *thickness);
				write_u32(&mut push, 24, color.to_u32());
				let cw = clip_words(*clip);
				write_i32(&mut push, 28, cw[0]);
				write_i32(&mut push, 32, cw[1]);
				write_i32(&mut push, 36, cw[2]);
				write_i32(&mut push, 40, cw[3]);
				write_f32(&mut push, 44, *corner_radius);
				unsafe {
					device.cmd_bind_pipeline(cmd_buf, vk::PipelineBindPoint::COMPUTE, pipeline.raw())?;
					device.cmd_push_constants(
						cmd_buf,
						layout,
						vk::ShaderStageFlagBits::COMPUTE_BIT,
						0,
						&push,
					)?;
					device.cmd_dispatch(cmd_buf, div_ceil(rect.w, 8), div_ceil(rect.h, 8), 1)?;
				}
			}
			UiCmd::Line {
				x0,
				y0,
				x1,
				y1,
				color,
				thickness,
				bounds,
			} => {
				let pipeline = &state.pipelines.draw_line;
				let layout = pipeline.layout();
				// push: dst_idx, x0, y0, x1, y1, thickness, rgba, bounds_x, bounds_y, bounds_w, bounds_h
				let mut push = [0u8; 44];
				write_u32(&mut push, 0, dst_idx);
				write_f32(&mut push, 4, *x0);
				write_f32(&mut push, 8, *y0);
				write_f32(&mut push, 12, *x1);
				write_f32(&mut push, 16, *y1);
				write_f32(&mut push, 20, *thickness);
				write_u32(&mut push, 24, color.to_u32());
				write_u32(&mut push, 28, bounds.x as u32);
				write_u32(&mut push, 32, bounds.y as u32);
				write_u32(&mut push, 36, bounds.w);
				write_u32(&mut push, 40, bounds.h);
				unsafe {
					device.cmd_bind_pipeline(cmd_buf, vk::PipelineBindPoint::COMPUTE, pipeline.raw())?;
					device.cmd_push_constants(
						cmd_buf,
						layout,
						vk::ShaderStageFlagBits::COMPUTE_BIT,
						0,
						&push,
					)?;
					device.cmd_dispatch(cmd_buf, div_ceil(bounds.w, 8), div_ceil(bounds.h, 8), 1)?;
				}
			}
			UiCmd::BlitRgba {
				src_idx,
				src_w,
				src_h,
				dst_x,
				dst_y,
				dst_w,
				dst_h,
				clip,
				channel,
			} => {
				// Dispatch only the clip-visible region: the destination rect
				// grows with zoom, so launching over it exhausts workgroups.
				let (vis_w, vis_h, base_x, base_y) =
					visible_blit_region(*dst_x, *dst_y, *dst_w, *dst_h, *clip);
				if vis_w > 0 && vis_h > 0 {
					let pipeline = &state.pipelines.blit_rgba;
					let layout = pipeline.layout();
					// push: source/destination geometry, clip, channel selector.
					let mut push = [0u8; 60];
					write_u32(&mut push, 0, *src_idx);
					write_u32(&mut push, 4, dst_idx);
					write_u32(&mut push, 8, *src_w);
					write_u32(&mut push, 12, *src_h);
					write_i32(&mut push, 16, *dst_x);
					write_i32(&mut push, 20, *dst_y);
					write_u32(&mut push, 24, *dst_w);
					write_u32(&mut push, 28, *dst_h);
					let cw = clip_words(*clip);
					write_i32(&mut push, 32, cw[0]);
					write_i32(&mut push, 36, cw[1]);
					write_i32(&mut push, 40, cw[2]);
					write_i32(&mut push, 44, cw[3]);
					write_u32(&mut push, 48, *channel);
					write_u32(&mut push, 52, base_x);
					write_u32(&mut push, 56, base_y);
					unsafe {
						device.cmd_bind_pipeline(cmd_buf, vk::PipelineBindPoint::COMPUTE, pipeline.raw())?;
						device.cmd_push_constants(
							cmd_buf,
							layout,
							vk::ShaderStageFlagBits::COMPUTE_BIT,
							0,
							&push,
						)?;
						device.cmd_dispatch(cmd_buf, div_ceil(vis_w, 8), div_ceil(vis_h, 8), 1)?;
					}
				}
			}
			UiCmd::BlitPlanar {
				r_idx,
				g_idx,
				b_idx,
				a_idx,
				src_w,
				src_h,
				dst_x,
				dst_y,
				dtypes,
				channels,
				dst_w,
				dst_h,
				clip,
			} => {
				// Dispatch only the clip-visible region; see BlitRgba.
				let (vis_w, vis_h, base_x, base_y) =
					visible_blit_region(*dst_x, *dst_y, *dst_w, *dst_h, *clip);
				if vis_w > 0 && vis_h > 0 {
					let pipeline = &state.pipelines.blit_planar;
					let layout = pipeline.layout();
					// push: plane/destination indices, geometry, dtype pack, clip.
					let mut push = [0u8; 76];
					write_u32(&mut push, 0, *r_idx);
					write_u32(&mut push, 4, *g_idx);
					write_u32(&mut push, 8, *b_idx);
					write_u32(&mut push, 12, *a_idx);
					write_u32(&mut push, 16, dst_idx);
					write_u32(&mut push, 20, *src_w);
					write_u32(&mut push, 24, *src_h);
					write_i32(&mut push, 28, *dst_x);
					write_i32(&mut push, 32, *dst_y);
					write_u32(&mut push, 36, *dtypes);
					write_u32(&mut push, 40, *channels);
					write_u32(&mut push, 44, *dst_w);
					write_u32(&mut push, 48, *dst_h);
					let cw = clip_words(*clip);
					write_i32(&mut push, 52, cw[0]);
					write_i32(&mut push, 56, cw[1]);
					write_i32(&mut push, 60, cw[2]);
					write_i32(&mut push, 64, cw[3]);
					write_u32(&mut push, 68, base_x);
					write_u32(&mut push, 72, base_y);
					unsafe {
						device.cmd_bind_pipeline(cmd_buf, vk::PipelineBindPoint::COMPUTE, pipeline.raw())?;
						device.cmd_push_constants(
							cmd_buf,
							layout,
							vk::ShaderStageFlagBits::COMPUTE_BIT,
							0,
							&push,
						)?;
						device.cmd_dispatch(cmd_buf, div_ceil(vis_w, 8), div_ceil(vis_h, 8), 1)?;
					}
				}
			}
			UiCmd::PlotLine {
				values_idx,
				count,
				rect,
				clip,
				x_min,
				x_max,
				y_min,
				y_max,
				color,
				fill,
				explicit_xy,
				antialias_samples,
				line_width,
			} => {
				let pipeline = &state.pipelines.draw_plot_line;
				let layout = pipeline.layout();
				// push: values_idx, count, dst_idx, dst_x, dst_y, dst_w, dst_h,
				//       x_min, x_max, y_min, y_max, rgba, flags, antialias_samples,
				//       line_width, clip_x, clip_y, clip_w, clip_h
				let mut push = [0u8; 76];
				write_u32(&mut push, 0, *values_idx);
				write_u32(&mut push, 4, *count);
				write_u32(&mut push, 8, dst_idx);
				write_i32(&mut push, 12, rect.x);
				write_i32(&mut push, 16, rect.y);
				write_u32(&mut push, 20, rect.w);
				write_u32(&mut push, 24, rect.h);
				write_f32(&mut push, 28, *x_min);
				write_f32(&mut push, 32, *x_max);
				write_f32(&mut push, 36, *y_min);
				write_f32(&mut push, 40, *y_max);
				write_u32(&mut push, 44, color.to_u32());
				let flags: u32 = if *fill { 2 } else { 0 } | if *explicit_xy { 4 } else { 0 };
				write_u32(&mut push, 48, flags);
				let aa = match *antialias_samples {
					1 => 1u32,
					8 => 8u32,
					_ => 4u32,
				};
				write_u32(&mut push, 52, aa);
				write_f32(&mut push, 56, *line_width);
				let cw = clip_words(*clip);
				write_i32(&mut push, 60, cw[0]);
				write_i32(&mut push, 64, cw[1]);
				write_i32(&mut push, 68, cw[2]);
				write_i32(&mut push, 72, cw[3]);
				unsafe {
					device.cmd_bind_pipeline(cmd_buf, vk::PipelineBindPoint::COMPUTE, pipeline.raw())?;
					device.cmd_push_constants(
						cmd_buf,
						layout,
						vk::ShaderStageFlagBits::COMPUTE_BIT,
						0,
						&push,
					)?;
					device.cmd_dispatch(cmd_buf, div_ceil(rect.w, 64), 1, 1)?;
				}
			}
			UiCmd::Glyphs {
				glyph_idx,
				atlas_idx,
				count,
				dst,
				clip,
				atlas_w,
				atlas_h,
			} => {
				let pipeline = &state.pipelines.draw_glyphs;
				let layout = pipeline.layout();
				let (dst_x, dst_y, dst_w, dst_h) =
					dst.map_or((0, 0, w, h), |rect| (rect.x, rect.y, rect.w, rect.h));
				let mut push = [0_u8; 72];
				write_u32(&mut push, 0, *glyph_idx);
				write_u32(&mut push, 4, 0);
				write_u32(&mut push, 8, *atlas_idx);
				write_u32(&mut push, 12, *count);
				write_u32(&mut push, 16, 0);
				write_u32(&mut push, 20, 1);
				write_u32(&mut push, 24, dst_idx);
				write_i32(&mut push, 28, dst_x);
				write_i32(&mut push, 32, dst_y);
				write_u32(&mut push, 36, dst_w);
				write_u32(&mut push, 40, dst_h);
				let cw = clip_words(*clip);
				write_i32(&mut push, 44, cw[0]);
				write_i32(&mut push, 48, cw[1]);
				write_i32(&mut push, 52, cw[2]);
				write_i32(&mut push, 56, cw[3]);
				write_u32(&mut push, 60, *atlas_w);
				write_u32(&mut push, 64, *atlas_h);
				write_f32(&mut push, 68, 0.0);
				unsafe {
					device.cmd_bind_pipeline(cmd_buf, vk::PipelineBindPoint::COMPUTE, pipeline.raw())?;
					device.cmd_push_constants(
						cmd_buf,
						layout,
						vk::ShaderStageFlagBits::COMPUTE_BIT,
						0,
						&push,
					)?;
					device.cmd_dispatch(cmd_buf, *count, 1, 1)?;
				}
			}
			UiCmd::RectOutlines {
				rect_idx,
				count,
				dst,
				clip,
				thickness,
				rgba,
			} => {
				let pipeline = &state.pipelines.draw_rect_outlines;
				let layout = pipeline.layout();
				let mut push = [0_u8; 52];
				write_u32(&mut push, 0, *rect_idx);
				write_u32(&mut push, 4, *count);
				write_u32(&mut push, 8, dst_idx);
				write_i32(&mut push, 12, dst.x);
				write_i32(&mut push, 16, dst.y);
				write_u32(&mut push, 20, dst.w);
				write_u32(&mut push, 24, dst.h);
				let cw = clip_words(*clip);
				write_i32(&mut push, 28, cw[0]);
				write_i32(&mut push, 32, cw[1]);
				write_i32(&mut push, 36, cw[2]);
				write_i32(&mut push, 40, cw[3]);
				write_u32(&mut push, 44, *thickness);
				write_u32(&mut push, 48, *rgba);
				unsafe {
					device.cmd_bind_pipeline(cmd_buf, vk::PipelineBindPoint::COMPUTE, pipeline.raw())?;
					device.cmd_push_constants(
						cmd_buf,
						layout,
						vk::ShaderStageFlagBits::COMPUTE_BIT,
						0,
						&push,
					)?;
					device.cmd_dispatch(cmd_buf, *count, 1, 1)?;
				}
			}
		}
		Ok(())
	}

	/// Compose the current command list on the GPU and return exact packed RGBA8.
	///
	/// This is an explicit blocking host-observation boundary intended for
	/// headless image/file sinks and test oracles.
	pub fn render_rgba8(&mut self) -> Result<Vec<u8>> {
		if let Some(error) = self
			.state
			.as_mut()
			.and_then(|state| state.recording_error.take())
		{
			return Err(error);
		}
		let state = self
			.state
			.as_ref()
			.ok_or_else(|| Error::failed_precondition("Ui is closed"))?;
		let handle = state.engine_handle.clone();
		let image = state.compose.raw_image();
		let width = state.compose.width;
		let height = state.compose.height;
		let descriptor_set = state.engine_handle.descriptor_set();
		let pipeline_layout = state.engine_handle.ui_pipeline_layout();
		let bindings = Self::frame_bindings(state)?;
		// Failure cannot prove completion; quarantine exact owners rather than
		// destroy resources that an uncertain submission may still reference.
		let retention = (
			state.pipelines.clone(),
			state.compose.clone(),
			state.text_atlas.clone(),
			state.frame_matrices.clone(),
			state.frame_textures.clone(),
			state.frame_storages.clone(),
			bindings.clone(),
		);
		let submission = handle.submit_ui_readback(image, width, height, |device, command| {
			Self::record_render_state(
				state,
				device,
				command,
				descriptor_set,
				pipeline_layout,
				bindings.as_deref(),
			)
		});
		let (event, readback) = match submission {
			Ok(submission) => submission,
			Err(error) => {
				std::mem::forget(retention);
				return Err(error);
			}
		};
		state
			.compose
			.set_layout(vk::ImageLayout::TRANSFER_SRC_OPTIMAL);
		if let Err(error) = event.wait() {
			std::mem::forget((retention, readback));
			return Err(error);
		}
		let mut rgba = vec![0_u8; width as usize * height as usize * 4];
		readback.read_prefix(&mut rgba)?;
		Ok(rgba)
	}

	/// Return the compose image extent.
	pub fn compose_extent(&self) -> Option<[u32; 2]> {
		self
			.state
			.as_ref()
			.map(|s| [s.compose.width, s.compose.height])
	}

	/// Compose the current command list and blit directly to `presenter`'s
	/// swapchain — no host readback.
	///
	/// Call pattern: build frame inside `begin_frame` / `end_frame`, then call
	/// this instead of `render_rgba8`.  The compose image size drives the
	/// source; the swapchain blit scales to the drawable extent automatically.
	pub fn present_to(&mut self, presenter: &mut crate::Presenter<'_>) -> Result<bool> {
		if let Some(error) = self
			.state
			.as_mut()
			.and_then(|state| state.recording_error.take())
		{
			return Err(error);
		}
		let state = self
			.state
			.as_ref()
			.ok_or_else(|| Error::failed_precondition("Ui is closed"))?;
		presenter.validate_ui_engine(&state.engine_handle)?;
		let image = state.compose.raw_image();
		let width = state.compose.width;
		let height = state.compose.height;
		let descriptor_set = state.engine_handle.descriptor_set();
		let pipeline_layout = state.engine_handle.ui_pipeline_layout();
		let bindings = Self::frame_bindings(state)?;
		// Retain the exact recorded frame, including pipelines and the old compose
		// image when resize replaces it. The presentation fence retires these
		// owners; assembling the next frame never waits or releases live inputs.
		let retention = std::rc::Rc::new((
			state.pipelines.clone(),
			state.compose.clone(),
			state.text_atlas.clone(),
			state.frame_matrices.clone(),
			state.frame_textures.clone(),
			state.frame_storages.clone(),
			bindings.clone(),
		));
		let result = presenter.present_compose_retained(
			image,
			width,
			height,
			descriptor_set,
			pipeline_layout,
			Some(retention),
			|device, cmd| {
				Self::record_render_state(
					state,
					device,
					cmd,
					descriptor_set,
					pipeline_layout,
					bindings.as_deref(),
				)
			},
		)?;
		state
			.compose
			.set_layout(vk::ImageLayout::TRANSFER_SRC_OPTIMAL);
		Ok(result)
	}

	/// Release this UI session's owners. Submitted presentation frames retain
	/// their exact resources until completion; use `Presenter::wait_presentation_idle`
	/// when explicit blocking retirement is required.
	pub fn close(&mut self) -> Result<()> {
		if self.state.is_none() {
			return Ok(());
		}
		self.state = None;
		Ok(())
	}
}

impl Drop for Ui<'_> {
	fn drop(&mut self) {
		self.state.take();
	}
}

// ── Helpers ───────────────────────────────────────────────────────────────────

/// Stable id hash for a widget label + panel context.
fn widget_hash(label: &str, panel_rect: Option<PixelRect>) -> u64 {
	use std::hash::{Hash, Hasher};
	let mut h = std::collections::hash_map::DefaultHasher::new();
	label.hash(&mut h);
	if let Some(r) = panel_rect {
		r.x.hash(&mut h);
		r.y.hash(&mut h);
	}
	h.finish()
}

/// True if `(x, y)` is inside `rect`.
fn point_in_rect(x: f32, y: f32, rect: PixelRect) -> bool {
	x >= rect.x as f32
		&& y >= rect.y as f32
		&& x < (rect.x + rect.w as i32) as f32
		&& y < (rect.y + rect.h as i32) as f32
}

/// True on pointer-up-inside after a prior pointer-down-inside.
fn was_pressed_and_released(state: &mut UiState, id: u64) -> bool {
	if state.pressed == id && !state.input.pointer_down {
		state.pressed = 0;
		return true;
	}
	false
}

/// Allocate a standard widget rect from the active panel.
fn animate_hover(state: &mut UiState, widget_id: u64, hovered: bool) -> f32 {
	let target = u8::from(hovered) as f32;
	let motion = state
		.widget_motion
		.entry(widget_id)
		.or_insert(WidgetMotion {
			start: 0.0,
			current: 0.0,
			target: 0.0,
			elapsed_ms: 0.0,
			last_seen: state.frame_index,
		});
	motion.last_seen = state.frame_index;
	if motion.target != target {
		motion.start = motion.current;
		motion.target = target;
		motion.elapsed_ms = 0.0;
	}
	let duration = state.motion_speed.duration_ms(120.0);
	motion.elapsed_ms = (motion.elapsed_ms + state.frame_delta_ms).min(duration);
	let fraction = if duration <= 0.0 {
		1.0
	} else {
		motion.elapsed_ms / duration
	};
	let inverse = 1.0 - fraction;
	motion.current = motion.start + (target - motion.start) * (1.0 - inverse * inverse * inverse);
	motion.current
}

fn allocate_widget(state: &mut UiState, padding: f32) -> (PixelRect, PixelRect) {
	let clip = state.layout.top().map(|s| s.clip()).unwrap_or_default();
	let Some(scope) = state.layout.top_mut() else {
		return (PixelRect::default(), PixelRect::default());
	};
	let h = default_item_height(padding);
	let w = resolve_width(&scope.layout.width.clone(), scope.content_width());
	let r = scope.allocate(w.max(1), h);
	(r, clip)
}

/// Resolve a chart grid for a plot rect the way the donor's
/// `resolveChartGridConfig` does: centered origin, a minor spacing that puts
/// 4–6 major cells across the shorter side at ~144 logical px per major, and
/// themed opacity.
fn chart_grid_config(
	rect: PixelRect,
	content_scale: f32,
	fill_background: bool,
	draw_grid: bool,
) -> UiGridConfig {
	let short_side = (rect.w.min(rect.h).saturating_sub(1)).max(1) as f32;
	let target_major_pixels = (144.0 * content_scale).max(1.0);
	let short_side_cells = (short_side / target_major_pixels).round() as i32;
	let short_side_cells = short_side_cells.clamp(4, 6);
	let minor_spacing = (short_side / (short_side_cells * 10) as f32).max(1.0);
	UiGridConfig {
		origin: [
			rect.x as f32 + 0.5 * (rect.w - 1) as f32,
			rect.y as f32 + 0.5 * (rect.h - 1) as f32,
		],
		minor_spacing: [minor_spacing, minor_spacing],
		opacity: 0.65,
		fill_background,
		draw_grid,
		draw_axes: draw_grid,
		..UiGridConfig::default()
	}
}

/// Allocate a widget area with an explicit height (used by multi-row widgets).
fn allocate_widget_explicit(state: &mut UiState, _padding: f32, h: u32) -> (PixelRect, PixelRect) {
	let clip = state.layout.top().map(|s| s.clip()).unwrap_or_default();
	let Some(scope) = state.layout.top_mut() else {
		return (PixelRect::default(), PixelRect::default());
	};
	let w = resolve_width(&scope.layout.width.clone(), scope.content_width());
	let r = scope.allocate(w.max(1), h.max(1));
	(r, clip)
}

/// Donor `oa::Ui::virtualRows` arithmetic, kept independent of GPU state so
/// scroll virtualization can be checked without creating a Vulkan device.
fn virtual_row_range(
	item_count: i32,
	row_height: i32,
	row_gap: i32,
	overscan_rows: i32,
	scroll_offset_y: i32,
	viewport_height: u32,
	padding_top: f32,
) -> UiVirtualRange {
	if item_count <= 0
		|| row_height <= 0
		|| row_gap < 0
		|| overscan_rows < 0
		|| !padding_top.is_finite()
	{
		return UiVirtualRange::default();
	}
	let stride = i64::from(row_height) + i64::from(row_gap);
	let padding = padding_top.floor().max(0.0) as i64;
	let offset = i64::from(scroll_offset_y.max(0));
	let begin = (offset - padding).max(0);
	let end = (offset + i64::from(viewport_height) - padding).max(begin);
	let first = (begin / stride - i64::from(overscan_rows)).max(0);
	let last = ((end + stride - 1) / stride + i64::from(overscan_rows)).min(i64::from(item_count));
	UiVirtualRange {
		first: first.min(i64::from(item_count)) as i32,
		one_past_last: last as i32,
	}
}

fn split_rects(
	rect: PixelRect,
	direction: UiDirection,
	handle_width: u32,
	ratio: f32,
) -> (PixelRect, PixelRect, PixelRect) {
	let extent = match direction {
		UiDirection::Row => rect.w,
		UiDirection::Column => rect.h,
	};
	let available = extent.saturating_sub(handle_width);
	let first = (available as f32 * ratio)
		.round()
		.clamp(0.0, available as f32) as u32;
	let second = available - first;
	match direction {
		UiDirection::Row => {
			let handle_x = rect.x.saturating_add(first as i32);
			(
				PixelRect::new(rect.x, rect.y, first, rect.h),
				PixelRect::new(handle_x, rect.y, handle_width, rect.h),
				PixelRect::new(
					handle_x.saturating_add(handle_width as i32),
					rect.y,
					second,
					rect.h,
				),
			)
		}
		UiDirection::Column => {
			let handle_y = rect.y.saturating_add(first as i32);
			(
				PixelRect::new(rect.x, rect.y, rect.w, first),
				PixelRect::new(rect.x, handle_y, rect.w, handle_width),
				PixelRect::new(
					rect.x,
					handle_y.saturating_add(handle_width as i32),
					rect.w,
					second,
				),
			)
		}
	}
}

/// Map a normalised value `t ∈ [0, 1]` to a color using a simple 3-stop
/// colormap.  `colormap` selects the palette: 0 = grayscale, 1 = viridis-like,
/// 2 = hot (blue→orange→white).
fn colormap_sample(t: f32, colormap: i32) -> crate::Color {
	let t = t.clamp(0.0, 1.0);
	match colormap {
		0 => crate::Color::new(t, t, t, 1.0),
		2 => {
			// Hot: dark-blue → orange → white
			if t < 0.5 {
				let s = t * 2.0;
				crate::Color::new(s * 0.85, s * 0.33, 0.85 - s * 0.6, 1.0)
			} else {
				let s = (t - 0.5) * 2.0;
				crate::Color::new(0.85 + s * 0.15, 0.66 + s * 0.34, 0.25 + s * 0.75, 1.0)
			}
		}
		_ => {
			// Viridis-like: purple → blue → green → yellow
			if t < 0.33 {
				let s = t / 0.33;
				crate::Color::new(0.27 - s * 0.15, 0.0 + s * 0.25, 0.33 + s * 0.40, 1.0)
			} else if t < 0.66 {
				let s = (t - 0.33) / 0.33;
				crate::Color::new(0.12 + s * 0.40, 0.25 + s * 0.50, 0.73 - s * 0.40, 1.0)
			} else {
				let s = (t - 0.66) / 0.34;
				crate::Color::new(0.52 + s * 0.48, 0.75 + s * 0.25, 0.33 - s * 0.33, 1.0)
			}
		}
	}
}

/// Map a [`UiIcon`] to a Unicode stand-in character for text rendering.
/// Allocate a taller plot area from the active panel.
fn allocate_plot(state: &mut UiState, padding: f32) -> (PixelRect, PixelRect) {
	let clip = state.layout.top().map(|s| s.clip()).unwrap_or_default();
	let Some(scope) = state.layout.top_mut() else {
		return (PixelRect::default(), PixelRect::default());
	};
	let h = default_item_height(padding) * 6;
	let w = resolve_width(&scope.layout.width.clone(), scope.content_width());
	let r = scope.allocate(w.max(1), h);
	(r, clip)
}

// ── Push-constant byte helpers ────────────────────────────────────────────────

fn bytemuck_push(words: &[u32]) -> &[u8] {
	// SAFETY: u32 has no alignment requirements stricter than u8.
	unsafe { std::slice::from_raw_parts(words.as_ptr().cast(), words.len() * 4) }
}

fn write_u32(buf: &mut [u8], offset: usize, val: u32) {
	buf[offset..offset + 4].copy_from_slice(&val.to_ne_bytes());
}
fn write_i32(buf: &mut [u8], offset: usize, val: i32) {
	buf[offset..offset + 4].copy_from_slice(&val.to_ne_bytes());
}
fn write_f32(buf: &mut [u8], offset: usize, val: f32) {
	buf[offset..offset + 4].copy_from_slice(&val.to_bits().to_ne_bytes());
}
