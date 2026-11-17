//! `oa::plot::FigureView` — interactive `ViewerLiveSource` for a retained
//! `Figure`.
//!
//! The view borrows only the bounded input/UI capabilities every
//! `ViewerLiveSource` receives:
//!
//! - wheel scroll zooms about the pointer's data position;
//! - left-drag pans the axes under the press;
//! - Shift+left-drag box-zooms into the selection;
//! - double-click or `R` restores the limits recorded before interaction;
//! - clicking a legend row toggles that artist's visibility;
//! - the crosshair reports data coordinates and snaps a tooltip to the
//!   nearest recorded sample.
//!
//! View transitions animate over `Ui`'s motion speed.  All view math runs in
//! display space — logarithmic axes pan and zoom on transformed values and
//! write back through the axis scale.

use crate::input::{InputEvent, InputEventKind, Key};
use crate::ui::{PixelRect, TextLayoutConfig, Ui};
use crate::viewer::{ViewerLiveCapabilities, ViewerLiveSource};
use crate::{Color, Engine, Error, Result};

use super::Figure;
use super::axes::{ArtistKind, legend_row_rect, scale_value, unscale_value};

/// Zoom multiplier applied per wheel step.
const SCROLL_ZOOM_STEP: f32 = 1.15;
/// Minimum box-zoom selection edge before the zoom applies.
const BOX_ZOOM_MIN_PX: f32 = 4.0;
/// Nearest-sample tooltip pick radius, in pixels.
const TOOLTIP_PICK_PX: f32 = 24.0;
/// Base view-transition length in milliseconds, scaled by `Ui` motion speed.
const VIEW_ANIM_MS: f32 = 160.0;

/// Left-drag pan state captured at pointer press.
struct DragState {
	/// Flattened `(row * cols + col)` cell index under the press.
	cell: usize,
	press: [f32; 2],
	/// Effective display-space limits when the drag started.
	limits: [f32; 4],
}

/// In-flight animated view transition for one cell.
struct ViewAnim {
	from: [f32; 4],
	to: [f32; 4],
	elapsed_ms: f32,
	duration_ms: f32,
	/// Restore automatic data limits once the transition lands.
	clear_on_finish: bool,
}

/// Shift+drag box-zoom selection in progress.
struct BoxSelect {
	cell: usize,
	start: [f32; 2],
	current: [f32; 2],
}

/// Serializable snapshot of a `FigureView`'s per-axes view overrides.
///
/// Limits live in display space (post-scale); `restore` writes them back
/// through the axes' scale transforms.  The vector is flattened row-major
/// over the figure grid.
#[derive(Clone, Debug, Default)]
pub struct FigureViewState {
	/// Per-cell view override; `None` leaves the recorded limits.
	pub limits: Vec<Option<[f32; 4]>>,
}

/// Interactive `Figure` presentation through `Viewer` live mode.
///
/// Attach with `Viewer::set_live_source`, which switches the viewer to live
/// mode; the figure keeps its recorded theme and artists while the view owns
/// only per-axes limit overrides.
pub struct FigureView {
	figure: Figure,
	/// Display-space limits recorded on each axes before interaction, so `R`
	/// restores the caller's view instead of always auto-fitting.
	base_limits: Vec<Option<[f32; 4]>>,
	/// Active per-axes view overrides in display space; `None` leaves the
	/// recorded limits.
	view_limits: Vec<Option<[f32; 4]>>,
	view_anims: Vec<Option<ViewAnim>>,
	drag: Option<DragState>,
	box_select: Option<BoxSelect>,
	/// Hovered legend entry, as `(cell, drawn row, artist index)`.
	legend_hover: Option<(usize, usize, usize)>,
	pointer: [f32; 2],
	pointer_valid: bool,
	scroll_pos: [f32; 2],
	scroll_delta: f32,
	reset_pending: bool,
	extent: [u32; 2],
	/// UI-space origin of the last rendered frame; `[0, 0]` when rendered
	/// full-window. Hit-testing and overlays offset by this origin so pointer
	/// coordinates stay in UI space.
	origin: [i32; 2],
	/// Pane identity used when hosted as a `WorkspaceItem`.
	name: String,
	show_crosshair: bool,
	/// Cached transition duration resolved from `Ui`'s motion speed.
	motion_ms: f32,
}

impl FigureView {
	/// Wrap `figure` for interactive presentation.
	pub fn new(figure: Figure) -> Self {
		let mut figure = figure;
		let rows = figure.rows().max(1);
		let cols = figure.cols().max(1);
		let mut base_limits = Vec::with_capacity((rows * cols) as usize);
		for row in 0..rows {
			for col in 0..cols {
				let axes = figure.ax(row, col);
				base_limits.push(
					axes
						.explicit_limits()
						.map(|_| axes.scaled_limits())
						.map(|l| [l.x_min, l.x_max, l.y_min, l.y_max]),
				);
			}
		}
		let cells = base_limits.len();
		Self {
			figure,
			base_limits,
			view_limits: vec![None; cells],
			view_anims: (0..cells).map(|_| None).collect(),
			drag: None,
			box_select: None,
			legend_hover: None,
			pointer: [0.0; 2],
			pointer_valid: false,
			scroll_pos: [0.0; 2],
			scroll_delta: 0.0,
			reset_pending: false,
			extent: [0; 2],
			origin: [0; 2],
			name: String::new(),
			show_crosshair: true,
			motion_ms: VIEW_ANIM_MS,
		}
	}

	/// The wrapped figure.
	pub fn figure(&self) -> &Figure {
		&self.figure
	}

	/// Mutable figure access for streaming series updates between frames.
	///
	/// Recorded `limits` only take effect while the axes has no active view
	/// override; `R` restores the limits present at attach time.
	pub fn figure_mut(&mut self) -> &mut Figure {
		&mut self.figure
	}

	/// Toggle the pointer crosshair and data-coordinate readout (default on).
	pub fn crosshair(&mut self, visible: bool) {
		self.show_crosshair = visible;
	}

	/// Set the pane identity shown when the view is hosted as a
	/// `WorkspaceItem` (tab id and label).
	pub fn set_name(&mut self, name: impl Into<String>) {
		self.name = name.into();
	}

	/// The pane identity assigned by `set_name` (empty until set).
	pub fn name(&self) -> &str {
		&self.name
	}

	/// Render the figure into `rect` of the current `Ui` frame instead of
	/// filling the window — the `WorkspaceItem` pane path.
	pub fn render_at(&mut self, ui: &mut Ui<'_>, rect: PixelRect) -> Result<()> {
		if rect.w == 0 || rect.h == 0 {
			return Ok(());
		}
		self.extent = [rect.w, rect.h];
		self.origin = [rect.x, rect.y];
		self.motion_ms = ui.motion_speed().duration_ms(VIEW_ANIM_MS);
		self.apply_interactions();
		self.apply_legend_highlight();
		self.figure.render_frame_at(rect, ui)?;
		self.render_overlay(ui, rect.w, rect.h)?;
		Ok(())
	}

	/// Restore every axes to the limits recorded at attach time.
	pub fn reset_view(&mut self) {
		self.reset_pending = true;
	}

	/// Snapshot the active view overrides for later `restore`.
	pub fn state(&self) -> FigureViewState {
		FigureViewState {
			limits: self.view_limits.clone(),
		}
	}

	/// Restore a snapshot taken by `state`, dropping in-flight transitions.
	pub fn restore(&mut self, state: &FigureViewState) {
		for (cell, limits) in state.limits.iter().enumerate() {
			if cell >= self.cell_count() {
				break;
			}
			self.view_limits[cell] = *limits;
			self.view_anims[cell] = None;
		}
	}

	// ── Interaction helpers ─────────────────────────────────────────────────

	fn cell_count(&self) -> usize {
		self.base_limits.len()
	}

	fn cols(&self) -> usize {
		self.figure.cols().max(1) as usize
	}

	fn cell_index(&self, row: i32, col: i32) -> usize {
		row as usize * self.cols() + col as usize
	}

	/// Flattened cell index under a pointer position, if any.
	fn cell_at(&self, px: f32, py: f32) -> Option<usize> {
		let [w, h] = self.extent;
		if w == 0 || h == 0 {
			return None;
		}
		for row in 0..self.figure.rows().max(1) {
			for col in 0..self.figure.cols().max(1) {
				let rect = self.figure.cell_rect(row, col, w, h);
				if PixelRect::new(
					rect.x + self.origin[0],
					rect.y + self.origin[1],
					rect.w.max(0) as u32,
					rect.h.max(0) as u32,
				)
				.contains(px, py)
				{
					return Some(self.cell_index(row, col));
				}
			}
		}
		None
	}

	fn cell_rect(&self, cell: usize) -> PixelRect {
		let (row, col) = (cell / self.cols(), cell % self.cols());
		let rect = self
			.figure
			.cell_rect(row as i32, col as i32, self.extent[0], self.extent[1]);
		PixelRect::new(
			rect.x + self.origin[0],
			rect.y + self.origin[1],
			rect.w.max(0) as u32,
			rect.h.max(0) as u32,
		)
	}

	/// Data-mapped plot area of a cell (post colorbar reservation).
	fn plot_rect(&self, cell: usize) -> PixelRect {
		self.figure.axes()[cell].plot_area(self.cell_rect(cell))
	}

	/// Display-space limits a cell would show without a view override.
	fn axes_limits(&self, cell: usize) -> [f32; 4] {
		let limits = self.figure.axes()[cell].scaled_limits();
		[limits.x_min, limits.x_max, limits.y_min, limits.y_max]
	}

	/// Effective display-space limits for a cell: override, else recorded.
	fn effective_limits(&self, cell: usize) -> [f32; 4] {
		self
			.view_limits
			.get(cell)
			.copied()
			.flatten()
			.unwrap_or_else(|| self.axes_limits(cell))
	}

	/// Start an animated transition toward `target` display-space limits.
	///
	/// Instant under `MotionSpeed::Instant`; a zero-duration transition just
	/// writes the override (and the optional auto-restore) directly.
	fn animate_to(&mut self, cell: usize, target: [f32; 4], clear_on_finish: bool) {
		if !target.iter().all(|v| v.is_finite()) || target[0] >= target[1] || target[2] >= target[3] {
			return;
		}
		if self.motion_ms <= 0.0 {
			self.view_anims[cell] = None;
			if clear_on_finish {
				self.view_limits[cell] = None;
				let (row, col) = (cell / self.cols(), cell % self.cols());
				self.figure.ax(row as i32, col as i32).auto_limits();
			} else {
				self.view_limits[cell] = Some(target);
			}
			return;
		}
		let from = self.effective_limits(cell);
		self.view_limits[cell] = Some(from);
		self.view_anims[cell] = Some(ViewAnim {
			from,
			to: target,
			elapsed_ms: 0.0,
			duration_ms: self.motion_ms,
			clear_on_finish,
		});
	}

	/// Legend row hit-test — returns `(drawn row, artist index, hidden)`.
	fn legend_hit(&self, cell: usize, px: f32, py: f32) -> Option<(usize, usize, bool)> {
		let axes = &self.figure.axes()[cell];
		if !axes.cmds.show_legend {
			return None;
		}
		let area = axes.plot_area(self.cell_rect(cell));
		let palette = self.figure.config().theme.ui_style().data_palette;
		for entry in axes.legend_entries(&palette) {
			if entry.row == usize::MAX {
				continue;
			}
			if legend_row_rect(area, entry.row).contains(px, py) {
				return Some((entry.row, entry.artist, entry.hidden));
			}
		}
		None
	}

	/// Write the legend-hover series emphasis into every axes. The flag lives
	/// on `Axes`, so each frame clears it before setting the hovered cell's.
	fn apply_legend_highlight(&mut self) {
		for cell in 0..self.cell_count() {
			let (row, col) = (cell / self.cols(), cell % self.cols());
			let artist = match self.legend_hover {
				Some((hover, _, artist)) if hover == cell => Some(artist),
				_ => None,
			};
			self
				.figure
				.ax(row as i32, col as i32)
				.set_series_highlight(artist);
		}
	}

	/// Apply pending scroll/reset input, then write view overrides into the
	/// figure's axes (unscaled back into recorded data limits).
	fn apply_interactions(&mut self) {
		if self.reset_pending {
			self.reset_pending = false;
			for cell in 0..self.cell_count() {
				match self.base_limits[cell] {
					Some(limits) => self.animate_to(cell, limits, false),
					None => {
						let target = self.axes_limits(cell);
						self.animate_to(cell, target, true);
					}
				}
			}
		}
		if self.scroll_delta != 0.0 {
			let steps = self.scroll_delta;
			self.scroll_delta = 0.0;
			if let Some(cell) = self.cell_at(self.scroll_pos[0], self.scroll_pos[1]) {
				let rect = self.plot_rect(cell);
				let limits = self.effective_limits(cell);
				self.animate_to(
					cell,
					zoom_limits(limits, rect, self.scroll_pos, steps),
					false,
				);
			}
		}
		for cell in 0..self.cell_count() {
			if let Some(limits) = self.view_limits[cell] {
				let (row, col) = (cell / self.cols(), cell % self.cols());
				let axes = self.figure.ax(row as i32, col as i32);
				let (x_scale, y_scale) = (axes.cmds.x_scale, axes.cmds.y_scale);
				axes.limits(
					unscale_value(limits[0], x_scale),
					unscale_value(limits[1], x_scale),
					unscale_value(limits[2], y_scale),
					unscale_value(limits[3], y_scale),
				);
			}
		}
	}

	/// Nearest recorded sample to the pointer, in pixel space.
	///
	/// Only host-backed line and scatter artists participate — a
	/// device-resident series has no host samples to inspect.  Returns the
	/// resolved color, raw data value, and screen position of the hit.
	fn nearest_sample(
		&self,
		cell: usize,
		area: PixelRect,
		limits: [f32; 4],
		px: f32,
		py: f32,
	) -> Option<(String, Color, f32, f32, f32, f32)> {
		let axes = &self.figure.axes()[cell];
		let (x_scale, y_scale) = (axes.cmds.x_scale, axes.cmds.y_scale);
		let palette = self.figure.config().theme.ui_style().data_palette;
		let x_span = (limits[1] - limits[0]).max(f32::EPSILON);
		let y_span = (limits[3] - limits[2]).max(f32::EPSILON);
		let map = |x: f32, y: f32| -> Option<[f32; 2]> {
			let sx = scale_value(x, x_scale);
			let sy = scale_value(y, y_scale);
			if !sx.is_finite() || !sy.is_finite() {
				return None;
			}
			Some([
				area.x as f32 + (sx - limits[0]) / x_span * area.w.saturating_sub(1) as f32,
				area.y as f32 + (1.0 - (sy - limits[2]) / y_span) * area.h.saturating_sub(1) as f32,
			])
		};
		let (mut line, mut scatter, mut bar, mut matrix, mut stream, mut band, mut bx) =
			(0usize, 0usize, 0usize, 0usize, 0usize, 0usize, 0usize);
		let mut palette_idx = 0usize;
		let mut best: Option<(String, Color, f32, f32, f32, f32, f32)> = None;
		for kind in &axes.cmds.artists {
			let (samples, label, color, hidden): (&dyn SeriesPoints, &str, Color, bool) = match kind {
				ArtistKind::Line => {
					let Some(cmd) = axes.cmds.lines.get(line) else {
						continue;
					};
					line += 1;
					(cmd, &cmd.style.label, cmd.style.color, cmd.style.hidden)
				}
				ArtistKind::Scatter => {
					let Some(cmd) = axes.cmds.scatters.get(scatter) else {
						continue;
					};
					scatter += 1;
					(cmd, &cmd.style.label, cmd.style.color, cmd.style.hidden)
				}
				ArtistKind::Bar => {
					let Some(cmd) = axes.cmds.bars.get(bar) else {
						continue;
					};
					bar += 1;
					if cmd.style.color.a == 0.0 {
						palette_idx += 1;
					}
					continue;
				}
				ArtistKind::LineMatrix => {
					let Some(cmd) = axes.cmds.line_matrices.get(matrix) else {
						continue;
					};
					matrix += 1;
					if cmd.style.color.a == 0.0 {
						palette_idx += 1;
					}
					continue;
				}
				ArtistKind::Stream => {
					let Some(cmd) = axes.cmds.streams.get(stream) else {
						continue;
					};
					stream += 1;
					(cmd, &cmd.style.label, cmd.style.color, cmd.style.hidden)
				}
				ArtistKind::Band => {
					let Some(cmd) = axes.cmds.bands.get(band) else {
						continue;
					};
					band += 1;
					if cmd.style.color.a == 0.0 {
						palette_idx += 1;
					}
					continue;
				}
				ArtistKind::BoxPlot => {
					let Some(cmd) = axes.cmds.boxes.get(bx) else {
						continue;
					};
					bx += 1;
					if cmd.style.color.a == 0.0 {
						palette_idx += 1;
					}
					continue;
				}
			};
			let color = if color.a == 0.0 {
				let resolved = palette[palette_idx % palette.len()];
				palette_idx += 1;
				resolved
			} else {
				color
			};
			if hidden {
				continue;
			}
			for index in 0..samples.len() {
				let (x, y) = samples.point(index);
				let Some(point) = map(x, y) else {
					continue;
				};
				let dx = point[0] - px;
				let dy = point[1] - py;
				let dist = dx * dx + dy * dy;
				let better = best.as_ref().is_none_or(|hit| dist < hit.6);
				if better {
					best = Some((label.to_owned(), color, x, y, point[0], point[1], dist));
				}
			}
		}
		best
			.filter(|hit| hit.6 <= TOOLTIP_PICK_PX * TOOLTIP_PICK_PX)
			.map(|(label, color, x, y, sx, sy, _)| (label, color, x, y, sx, sy))
	}

	/// Crosshair, nearest-sample tooltip, legend hover, and box-zoom overlay.
	fn render_overlay(&mut self, ui: &mut Ui<'_>, w: u32, h: u32) -> Result<()> {
		let style = self.figure.config().theme.ui_style();
		let full = PixelRect::new(self.origin[0], self.origin[1], w, h);
		let label_cfg = TextLayoutConfig {
			size: 10.0,
			..Default::default()
		};

		// Box-zoom selection rectangle.
		if let Some(select) = &self.box_select {
			let rect = selection_rect(select.start, select.current);
			let accent = Color::new(style.accent.r, style.accent.g, style.accent.b, 0.12);
			ui.rect_raw(rect, accent, full, 0.0);
			ui.rect_outline_raw(rect, style.accent, 1, full);
		}

		// Hovered legend row highlight.
		if let Some((cell, row, _)) = self.legend_hover {
			let area = self.plot_rect(cell);
			let band = legend_row_rect(area, row);
			let hover = Color::new(style.text.r, style.text.g, style.text.b, 0.08);
			ui.rect_raw(band, hover, self.cell_rect(cell), 0.0);
		}

		if !self.show_crosshair || !self.pointer_valid {
			return Ok(());
		}
		let Some(cell) = self.cell_at(self.pointer[0], self.pointer[1]) else {
			return Ok(());
		};
		let area = self.plot_rect(cell);
		let limits = self.effective_limits(cell);
		let axes = &self.figure.axes()[cell];
		let (x_scale, y_scale) = (axes.cmds.x_scale, axes.cmds.y_scale);
		let x_span = (limits[1] - limits[0]).max(f32::EPSILON);
		let y_span = (limits[3] - limits[2]).max(f32::EPSILON);
		let px = self.pointer[0].clamp(
			area.x as f32,
			(area.x + area.w.saturating_sub(1) as i32) as f32,
		);
		let py = self.pointer[1].clamp(
			area.y as f32,
			(area.y + area.h.saturating_sub(1) as i32) as f32,
		);
		let data_x = limits[0] + (px - area.x as f32) / area.w.max(1) as f32 * x_span;
		let data_y = limits[3] - (py - area.y as f32) / area.h.max(1) as f32 * y_span;

		let crosshair = Color::new(style.accent.r, style.accent.g, style.accent.b, 0.65);
		let right = (area.x + area.w as i32) as f32;
		let bottom = (area.y + area.h as i32) as f32;
		ui.line_at([px, area.y as f32], [px, bottom], crosshair, 1.0, area)?;
		ui.line_at([area.x as f32, py], [right, py], crosshair, 1.0, area)?;

		if let Some((label, color, raw_x, raw_y, sx, sy)) =
			self.nearest_sample(cell, area, limits, px, py)
		{
			ui.line_at([sx, sy], [sx, sy], color, 9.0, area)?;
			let name = if label.is_empty() {
				"series".to_owned()
			} else {
				label
			};
			ui.text_at(
				&format!("{name} ({raw_x:.4}, {raw_y:.4})"),
				[px + 8.0, py - 4.0],
				label_cfg,
				style.text,
				full,
			)?;
		} else {
			ui.text_at(
				&format!(
					"{:.4}, {:.4}",
					unscale_value(data_x, x_scale),
					unscale_value(data_y, y_scale),
				),
				[px + 8.0, py - 4.0],
				label_cfg,
				style.text,
				full,
			)?;
		}
		Ok(())
	}
}

/// Point access shared by line and scatter commands during hit-testing.
trait SeriesPoints {
	fn len(&self) -> usize;
	fn point(&self, index: usize) -> (f32, f32);
}

impl SeriesPoints for super::axes::LineCmd {
	fn len(&self) -> usize {
		self.x.len()
	}
	fn point(&self, index: usize) -> (f32, f32) {
		(self.x[index], self.y[index])
	}
}

impl SeriesPoints for super::axes::StreamCmd {
	fn len(&self) -> usize {
		self.len
	}
	fn point(&self, index: usize) -> (f32, f32) {
		let capacity = self.values.len();
		(index as f32, self.values[(self.head + index) % capacity])
	}
}

impl SeriesPoints for super::axes::ScatterCmd {
	fn len(&self) -> usize {
		self.x.len()
	}
	fn point(&self, index: usize) -> (f32, f32) {
		(self.x[index], self.y[index])
	}
}

/// Zoom `limits` about the pointer's data position by `steps` wheel ticks.
fn zoom_limits(limits: [f32; 4], rect: PixelRect, pos: [f32; 2], steps: f32) -> [f32; 4] {
	// Positive scroll (up) shrinks the span — zoom in.
	let factor = SCROLL_ZOOM_STEP.powf(-steps);
	let w = rect.w.max(1) as f32;
	let h = rect.h.max(1) as f32;
	let fx = ((pos[0] - rect.x as f32) / w).clamp(0.0, 1.0);
	// Data y increases upward; screen y increases downward.
	let fy = 1.0 - ((pos[1] - rect.y as f32) / h).clamp(0.0, 1.0);
	let [x_min, x_max, y_min, y_max] = limits;
	let anchor_x = x_min + (x_max - x_min) * fx;
	let anchor_y = y_min + (y_max - y_min) * fy;
	let x_span = (x_max - x_min) * factor;
	let y_span = (y_max - y_min) * factor;
	[
		anchor_x - x_span * fx,
		anchor_x + x_span * (1.0 - fx),
		anchor_y - y_span * fy,
		anchor_y + y_span * (1.0 - fy),
	]
}

/// Pan `limits` by the pixel delta between press and current pointer.
fn pan_limits(limits: [f32; 4], rect: PixelRect, press: [f32; 2], pos: [f32; 2]) -> [f32; 4] {
	let dx = (pos[0] - press[0]) / rect.w.max(1) as f32;
	let dy = (pos[1] - press[1]) / rect.h.max(1) as f32;
	let x_span = limits[1] - limits[0];
	let y_span = limits[3] - limits[2];
	[
		limits[0] - dx * x_span,
		limits[1] - dx * x_span,
		limits[2] + dy * y_span,
		limits[3] + dy * y_span,
	]
}

/// Axis-aligned pixel rect spanned by two box-select corners.
fn selection_rect(a: [f32; 2], b: [f32; 2]) -> PixelRect {
	let x = a[0].min(b[0]);
	let y = a[1].min(b[1]);
	PixelRect::new(
		x as i32,
		y as i32,
		(a[0] - b[0]).abs().max(1.0) as u32,
		(a[1] - b[1]).abs().max(1.0) as u32,
	)
}

/// Limits for the display-space span selected by a box-zoom gesture.
fn box_zoom_limits(limits: [f32; 4], rect: PixelRect, a: [f32; 2], b: [f32; 2]) -> [f32; 4] {
	let w = rect.w.max(1) as f32;
	let h = rect.h.max(1) as f32;
	let x_span = limits[1] - limits[0];
	let y_span = limits[3] - limits[2];
	let fx = |px: f32| limits[0] + ((px - rect.x as f32) / w).clamp(0.0, 1.0) * x_span;
	let fy = |py: f32| limits[3] - ((py - rect.y as f32) / h).clamp(0.0, 1.0) * y_span;
	let (x0, x1) = (fx(a[0]).min(fx(b[0])), fx(a[0]).max(fx(b[0])));
	let (y0, y1) = (fy(a[1]).min(fy(b[1])), fy(a[1]).max(fy(b[1])));
	[x0, x1, y0, y1]
}

/// Ease-out cubic interpolation between two limit boxes.
fn lerp_limits(from: [f32; 4], to: [f32; 4], t: f32) -> [f32; 4] {
	[
		from[0] + (to[0] - from[0]) * t,
		from[1] + (to[1] - from[1]) * t,
		from[2] + (to[2] - from[2]) * t,
		from[3] + (to[3] - from[3]) * t,
	]
}

impl ViewerLiveSource for FigureView {
	fn capabilities(&self) -> ViewerLiveCapabilities {
		ViewerLiveCapabilities {
			receives_events: true,
			..ViewerLiveCapabilities::default()
		}
	}

	fn open(&mut self, _engine: &Engine) -> Result<()> {
		Ok(())
	}

	fn update(&mut self, delta_ms: f32) -> Result<()> {
		if !delta_ms.is_finite() || delta_ms < 0.0 {
			return Err(Error::invalid_argument(
				"FigureView::update requires a finite non-negative delta",
			));
		}
		for cell in 0..self.cell_count() {
			let Some(anim) = &self.view_anims[cell] else {
				continue;
			};
			let elapsed = anim.elapsed_ms + delta_ms;
			let duration = anim.duration_ms.max(1.0);
			let t = (elapsed / duration).min(1.0);
			let eased = 1.0 - (1.0 - t).powi(3);
			let value = lerp_limits(anim.from, anim.to, eased);
			let done = t >= 1.0;
			let clear = anim.clear_on_finish;
			self.view_limits[cell] = Some(value);
			if done {
				self.view_anims[cell] = None;
				if clear {
					self.view_limits[cell] = None;
					let (row, col) = (cell / self.cols(), cell % self.cols());
					self.figure.ax(row as i32, col as i32).auto_limits();
				}
			} else if let Some(anim) = &mut self.view_anims[cell] {
				anim.elapsed_ms = elapsed;
			}
		}
		Ok(())
	}

	fn on_event(&mut self, event: &InputEvent) -> Result<()> {
		match event.kind {
			InputEventKind::MouseScroll => {
				self.scroll_pos = [event.mouse_x, event.mouse_y];
				self.scroll_delta += event.scroll_y;
			}
			InputEventKind::MouseDown if event.button == 1 => {
				let Some(cell) = self.cell_at(event.mouse_x, event.mouse_y) else {
					return Ok(());
				};
				// Legend rows swallow the press — clicking toggles the artist.
				if let Some((_, artist, hidden)) = self.legend_hit(cell, event.mouse_x, event.mouse_y) {
					let (row, col) = (cell / self.cols(), cell % self.cols());
					self
						.figure
						.ax(row as i32, col as i32)
						.set_artist_hidden(artist, !hidden);
					return Ok(());
				}
				if event.click_count >= 2 {
					// Double-click restores this cell's attach-time limits.
					match self.base_limits[cell] {
						Some(limits) => self.animate_to(cell, limits, false),
						None => {
							let target = self.axes_limits(cell);
							self.animate_to(cell, target, true);
						}
					}
					return Ok(());
				}
				if event.shift() {
					self.box_select = Some(BoxSelect {
						cell,
						start: [event.mouse_x, event.mouse_y],
						current: [event.mouse_x, event.mouse_y],
					});
					return Ok(());
				}
				self.view_anims[cell] = None;
				self.drag = Some(DragState {
					cell,
					press: [event.mouse_x, event.mouse_y],
					limits: self.effective_limits(cell),
				});
			}
			InputEventKind::MouseMove => {
				self.pointer = [event.mouse_x, event.mouse_y];
				self.pointer_valid = true;
				self.legend_hover = self.cell_at(event.mouse_x, event.mouse_y).and_then(|cell| {
					self
						.legend_hit(cell, event.mouse_x, event.mouse_y)
						.map(|(row, artist, _)| (cell, row, artist))
				});
				if let Some(select) = &mut self.box_select {
					select.current = [event.mouse_x, event.mouse_y];
				}
				// Pan applies immediately so a press/drag/release inside one
				// frame still lands; render only publishes the overrides.
				if let Some(drag) = &self.drag {
					let pos = [event.mouse_x, event.mouse_y];
					if drag.cell < self.cell_count() {
						let rect = self.plot_rect(drag.cell);
						if rect.w > 0 && rect.h > 0 {
							self.view_limits[drag.cell] = Some(pan_limits(drag.limits, rect, drag.press, pos));
						}
					}
				}
			}
			InputEventKind::MouseUp if event.button == 1 => {
				if let Some(select) = self.box_select.take() {
					let dx = (select.current[0] - select.start[0]).abs();
					let dy = (select.current[1] - select.start[1]).abs();
					if dx >= BOX_ZOOM_MIN_PX && dy >= BOX_ZOOM_MIN_PX {
						let rect = self.plot_rect(select.cell);
						let limits = self.effective_limits(select.cell);
						let target = box_zoom_limits(limits, rect, select.start, select.current);
						self.animate_to(select.cell, target, false);
					}
				}
				self.drag = None;
			}
			InputEventKind::MouseLeave => {
				self.pointer_valid = false;
				self.drag = None;
				self.box_select = None;
				self.legend_hover = None;
			}
			InputEventKind::KeyDown if event.key == Key::R => {
				self.reset_pending = true;
			}
			_ => {}
		}
		Ok(())
	}

	fn render(&mut self, ui: &mut Ui<'_>, width: u32, height: u32) -> Result<()> {
		self.render_at(ui, PixelRect::new(0, 0, width, height))
	}

	fn close(&mut self) -> Result<()> {
		Ok(())
	}
}

impl crate::viewer::WorkspaceItem for FigureView {
	fn id(&self) -> &str {
		&self.name
	}

	fn title(&self) -> &str {
		if self.name.is_empty() {
			"figure"
		} else {
			&self.name
		}
	}

	fn update(&mut self, delta_ms: f32) -> Result<()> {
		ViewerLiveSource::update(self, delta_ms)
	}

	fn render(&mut self, ui: &mut Ui<'_>, rect: PixelRect) -> Result<()> {
		self.render_at(ui, rect)
	}

	fn on_event(&mut self, event: &InputEvent) -> Result<()> {
		ViewerLiveSource::on_event(self, event)
	}
}
