//! `TrainingViewerSource` — non-owning `TrainingSession` dashboard adapter.
//!
//! Donor: `oa/ui/trainingViewer.h`, `oa/ui/trainingViewer.cpp`.
//!
//! Training owns the model, optimizer, graph, and worker thread. The viewer
//! only reads immutable snapshots and enqueues typed commands at the shared
//! safe point, so it can attach to supervised, PPO, DQN, or SAC training
//! without another trainer abstraction.
//!
//! The donor registers its key actions into the `Viewer`-owned `InputSystem`
//! during `init`. The Rust `ViewerLiveSource` seam delivers raw UI events
//! instead, so this source owns one [`InputSystem`] and dispatches the same
//! four actions — Space pause/resume, C checkpoint, E evaluate, S stop.

use std::sync::Mutex;

use crate::input::{InputContext, InputEvent, InputSystem, Key, KeyAction, KeyBinding};
use crate::ml::training::{TrainingSession, TrainingSessionSnapshot, TrainingState};
use crate::ui::{
	PixelRect, Ui, UiDirection, UiDropdownConfig, UiEdge, UiLayout, UiPlotConfig, UiScrollConfig,
	UiSplitConfig, UiTabItem, UiTooltipConfig, UiTreeRowConfig,
};
pub use crate::viewer::types::TrainingViewerConfig;
use crate::{Color, Engine, Error, Event, Result, Texture};

use super::{ViewerLiveCapabilities, ViewerLiveSource};

impl Default for TrainingViewerConfig {
	fn default() -> Self {
		Self {
			title: String::from("OA training"),
			history_capacity: 512,
			max_metric_plots: 6,
			show_gpu_timing: true,
			show_wall_timing: true,
			show_preview: true,
		}
	}
}

/// Immutable generated artifact handed from a training/evaluation producer to
/// the viewer. The texture allocation is shared; the timeline event prevents
/// presentation until the producing GPU batch has completed. Replacing a
/// pending frame drops it without ever blocking the training owner.
///
/// Donor: `oa::TrainingPreviewFrame`.
#[derive(Clone)]
pub struct TrainingPreviewFrame {
	/// Produced preview texture, required for publication.
	pub texture: Option<Texture>,
	/// Producer completion event; the frame is promoted only when complete.
	pub completion: Event,
	/// Optional frame label.
	pub label: String,
	/// Training step that produced the frame.
	pub step: i64,
}

/// Non-owning `TrainingSession` dashboard source for `Viewer` live mode.
///
/// Donor: `oa::TrainingViewerSource`.
pub struct TrainingViewerSource {
	session: TrainingSession,
	config: TrainingViewerConfig,
	input: InputSystem,
	snapshot: Option<TrainingSessionSnapshot>,
	series: Vec<Series>,
	last_result_sequence: u64,
	viewer_frame_ms: f32,
	/// 0=responsive, 1/2=explicit metric-card column count.
	metric_columns: i32,
	selected_metric: i32,
	compact_tabs: crate::ui::UiTabBarState,
	preview_split_ratio: f32,
	preview: Mutex<PreviewState>,
}

struct Series {
	name: String,
	values: Vec<f32>,
	color: Color,
	expanded: bool,
}

#[derive(Default)]
struct PreviewState {
	pending: Option<TrainingPreviewFrame>,
	current: Option<TrainingPreviewFrame>,
}

impl TrainingViewerSource {
	/// Attach to `session`. The handle is shared, not owned: enqueueing and
	/// snapshot reads never block the training thread.
	pub fn new(session: &TrainingSession, mut config: TrainingViewerConfig) -> Self {
		config.history_capacity = config.history_capacity.max(2);
		config.max_metric_plots = config.max_metric_plots.clamp(1, 12);
		let mut input = InputSystem::new();
		let binding = |key| KeyBinding {
			key,
			modifiers: crate::input::INPUT_MOD_NONE,
		};
		let pause_session = session.clone();
		input.register_action(KeyAction {
			name: "training-pause-resume".into(),
			binding: binding(Key::Space),
			context: InputContext::Global,
			allow_repeat: false,
			callback: Some(Box::new(move || match pause_session.state() {
				TrainingState::Running => {
					log_enqueue(pause_session.pause(0), "pause");
				}
				TrainingState::Paused => {
					log_enqueue(pause_session.resume(0), "resume");
				}
				_ => {}
			})),
		});
		let stop_session = session.clone();
		input.register_action(KeyAction {
			name: "training-stop".into(),
			binding: binding(Key::S),
			context: InputContext::Global,
			allow_repeat: false,
			callback: Some(Box::new(move || {
				log_enqueue(stop_session.stop(0), "stop");
			})),
		});
		let checkpoint_session = session.clone();
		input.register_action(KeyAction {
			name: "training-checkpoint".into(),
			binding: binding(Key::C),
			context: InputContext::Global,
			allow_repeat: false,
			callback: Some(Box::new(move || {
				log_enqueue(checkpoint_session.checkpoint(0), "checkpoint");
			})),
		});
		let evaluate_session = session.clone();
		input.register_action(KeyAction {
			name: "training-evaluate".into(),
			binding: binding(Key::E),
			context: InputContext::Global,
			allow_repeat: false,
			callback: Some(Box::new(move || {
				log_enqueue(evaluate_session.evaluate(0), "evaluate");
			})),
		});
		crate::log_info!(
			crate::LogComponent::UI,
			"OaTrainingViewer: Space=pause/resume · C=checkpoint · E=evaluate · S=stop"
		);
		Self {
			session: session.clone(),
			config,
			input,
			snapshot: None,
			series: Vec::new(),
			last_result_sequence: 0,
			viewer_frame_ms: 0.0,
			metric_columns: 0,
			selected_metric: 0,
			compact_tabs: crate::ui::UiTabBarState {
				selected: 0,
				first_visible: 0,
			},
			preview_split_ratio: 0.60,
			preview: Mutex::new(PreviewState::default()),
		}
	}

	/// Publish one preview frame; an incomplete producer event defers
	/// promotion until the next `update` observes completion.
	///
	/// # Errors
	///
	/// Returns [`ErrorKind::InvalidArgument`] when the frame has no texture.
	///
	/// [`ErrorKind::InvalidArgument`]: crate::ErrorKind::InvalidArgument
	pub fn publish_preview(&self, frame: TrainingPreviewFrame) -> Result<()> {
		if frame.texture.is_none() {
			return Err(Error::invalid_argument(
				"OaTrainingViewer preview requires a valid texture",
			));
		}
		self
			.preview
			.lock()
			.unwrap_or_else(std::sync::PoisonError::into_inner)
			.pending = Some(frame);
		Ok(())
	}

	/// Return the normalized dashboard configuration.
	#[must_use]
	pub const fn config(&self) -> &TrainingViewerConfig {
		&self.config
	}

	/// Return the latest session snapshot observed by the dashboard.
	#[must_use]
	pub fn latest_snapshot(&self) -> Option<TrainingSessionSnapshot> {
		self.snapshot.clone()
	}

	/// Return the latest promoted preview frame, if any.
	#[must_use]
	pub fn latest_preview(&self) -> Option<TrainingPreviewFrame> {
		self
			.preview
			.lock()
			.unwrap_or_else(std::sync::PoisonError::into_inner)
			.current
			.clone()
	}

	/// Return the number of retained metric series.
	#[must_use]
	pub fn metric_series_count(&self) -> u32 {
		self.series.len() as u32
	}

	/// Return the retained sample count for the series named `name`.
	#[must_use]
	pub fn metric_sample_count(&self, name: &str) -> u32 {
		self
			.series
			.iter()
			.find(|series| series.name == name)
			.map_or(0, |series| series.values.len() as u32)
	}

	fn find_series(&mut self, name: &str) -> Option<&mut Series> {
		self.series.iter_mut().find(|series| series.name == name)
	}

	fn append_series(&mut self, name: &str, value: f64, color: Color) {
		if !value.is_finite() {
			return;
		}
		if self.find_series(name).is_none() {
			if self.series.len() >= self.config.max_metric_plots as usize {
				return;
			}
			self.series.push(Series {
				name: name.to_owned(),
				values: Vec::new(),
				color,
				expanded: true,
			});
		}
		let capacity = self.config.history_capacity.max(2) as usize;
		if let Some(series) = self.find_series(name) {
			series.values.push(value as f32);
			if series.values.len() > capacity {
				let excess = series.values.len() - capacity;
				series.values.drain(..excess);
			}
		}
	}

	fn append_snapshot(&mut self, snapshot: &TrainingSessionSnapshot) {
		let new_step = self
			.snapshot
			.as_ref()
			.is_none_or(|previous| snapshot.step != previous.step);
		if new_step && snapshot.step > 0 {
			self.append_series("loss", f64::from(snapshot.loss), Color::accent());
			if self.config.show_gpu_timing {
				self.append_series("gpu_ms", snapshot.gpu_ms, Color::new(0.70, 0.46, 0.96, 1.0));
			}
			if self.config.show_wall_timing {
				self.append_series("wall_ms", snapshot.wall_ms, Color::warning());
			}
			let colors = [
				Color::success(),
				Color::new(0.30, 0.72, 0.94, 1.0),
				Color::new(0.95, 0.43, 0.36, 1.0),
				Color::new(0.85, 0.67, 0.24, 1.0),
				Color::new(0.55, 0.78, 0.42, 1.0),
			];
			for (index, metric) in snapshot.metrics.iter().enumerate() {
				self.append_series(&metric.name, metric.value, colors[index % colors.len()]);
			}
		}
		self.snapshot = Some(snapshot.clone());
	}

	fn poll_results(&mut self) {
		for result in self.session.results_after(self.last_result_sequence) {
			self.last_result_sequence = self.last_result_sequence.max(result.sequence);
		}
	}
}

fn state_name(state: TrainingState) -> &'static str {
	match state {
		TrainingState::Running => "RUNNING",
		TrainingState::Paused => "PAUSED",
		TrainingState::Stopping => "STOPPING",
		TrainingState::Completed => "COMPLETED",
		TrainingState::Failed => "FAILED",
	}
}

fn state_color(state: TrainingState) -> Color {
	match state {
		TrainingState::Running | TrainingState::Completed => Color::success(),
		TrainingState::Paused | TrainingState::Stopping => Color::warning(),
		TrainingState::Failed => Color::error(),
	}
}

fn log_enqueue(result: Result<u64>, name: &str) {
	if let Err(error) = result {
		crate::log_warn!(
			crate::LogComponent::UI,
			"OaTrainingViewer {name} rejected: {error}"
		);
	}
}

#[allow(clippy::too_many_arguments)]
fn draw_label(
	ui: &mut Ui<'_>,
	content_scale: f32,
	id: &str,
	rect: PixelRect,
	text: &str,
	size: f32,
	color: Color,
) {
	let mut style = ui.current_style();
	style.font_size = size * content_scale;
	style.text = color;
	let layout = UiLayout {
		padding: UiEdge::default(),
		..UiLayout::default()
	};
	ui.push_style(style);
	ui.begin_panel_with_layout(id, rect, layout);
	ui.label(text);
	ui.end_panel();
	ui.pop_style();
}

impl ViewerLiveSource for TrainingViewerSource {
	fn capabilities(&self) -> ViewerLiveCapabilities {
		ViewerLiveCapabilities {
			receives_events: true,
			..ViewerLiveCapabilities::default()
		}
	}

	fn open(&mut self, _engine: &Engine) -> Result<()> {
		if let Some(snapshot) = self.session.latest_snapshot() {
			self.append_snapshot(&snapshot);
		}
		Ok(())
	}

	fn on_event(&mut self, event: &InputEvent) -> Result<()> {
		self.input.dispatch(event);
		Ok(())
	}

	fn update(&mut self, delta_ms: f32) -> Result<()> {
		if !delta_ms.is_finite() || delta_ms < 0.0 {
			return Err(Error::invalid_argument(
				"TrainingViewerSource::update requires a finite non-negative delta",
			));
		}
		if delta_ms > 0.0 {
			self.viewer_frame_ms = if self.viewer_frame_ms <= 0.0 {
				delta_ms
			} else {
				self.viewer_frame_ms * 0.90 + delta_ms * 0.10
			};
		}
		if let Some(snapshot) = self.session.latest_snapshot() {
			let changed = self.snapshot.as_ref().is_none_or(|previous| {
				snapshot.step != previous.step
					|| snapshot.revision != previous.revision
					|| snapshot.state != previous.state
			});
			if changed {
				self.append_snapshot(&snapshot);
			}
		}
		{
			let mut preview = self
				.preview
				.lock()
				.unwrap_or_else(std::sync::PoisonError::into_inner);
			if preview
				.pending
				.as_ref()
				.is_some_and(|frame| frame.completion.is_complete().unwrap_or(false))
			{
				preview.current = preview.pending.take();
			}
		}
		self.poll_results();
		Ok(())
	}

	#[allow(clippy::too_many_lines)]
	fn render(&mut self, ui: &mut Ui<'_>, width: u32, height: u32) -> Result<()> {
		if width > i32::MAX as u32 || height > i32::MAX as u32 {
			return Err(Error::invalid_argument(
				"TrainingViewerSource::render extent exceeds signed UI coordinates",
			));
		}
		let (width, height) = (width as i32, height as i32);
		let content_scale = ui.content_scale();
		let px = |logical: i32| ((logical as f32 * content_scale).round() as i32).max(1);
		if width < px(320) || height < px(240) {
			return Ok(());
		}
		let margin = px(20);
		let gap = px(14);
		let header_height = px(108);
		let header = PixelRect::new(
			margin,
			margin,
			(width - margin * 2) as u32,
			header_height as u32,
		);
		ui.rect(header, Color::new(0.045, 0.045, 0.045, 1.0), 0.0)?;
		ui.rect_outline(header, Color::new(1.0, 1.0, 1.0, 0.10), 1, 0.0)?;

		let default_snapshot = TrainingSessionSnapshot {
			revision: 0,
			state: TrainingState::Running,
			step: 0,
			epoch: 0,
			learning_rate: 0.0,
			loss: 0.0,
			gpu_ms: 0.0,
			wall_ms: 0.0,
			metrics: Vec::new(),
		};
		let snapshot = self.snapshot.clone().unwrap_or(default_snapshot);
		let status_color = state_color(snapshot.state);
		ui.rect(
			PixelRect::new(header.x, header.y, px(6) as u32, header.h),
			status_color,
			0.0,
		)?;
		draw_label(
			ui,
			content_scale,
			"training-title",
			PixelRect::new(
				header.x + px(22),
				header.y + px(10),
				(header.w as i32 - px(240)).max(1) as u32,
				px(24) as u32,
			),
			&self.config.title,
			18.0,
			Color::new(0.96, 0.96, 0.96, 1.0),
		);
		ui.begin_panel_with_layout(
			"training-metric-layout",
			PixelRect::new(
				header.x + (header.w as i32 - px(194)).max(0),
				header.y + px(8),
				px(174) as u32,
				px(30) as u32,
			),
			UiLayout {
				padding: UiEdge::default(),
				gap: 0.0,
				..UiLayout::default()
			},
		);
		ui.dropdown(
			"metrics",
			&["Auto", "1 column", "2 columns"],
			&mut self.metric_columns,
			UiDropdownConfig {
				max_visible_items: 3,
				..UiDropdownConfig::default()
			},
		);
		ui.tooltip(
			"Choose responsive, one-column, or two-column metric cards.",
			UiTooltipConfig::default(),
		);
		ui.end_panel();
		let summary = format!(
			"{}   step {}   epoch {}   loss {:.6}   lr {:.6}",
			state_name(snapshot.state),
			snapshot.step,
			snapshot.epoch,
			f64::from(snapshot.loss),
			f64::from(snapshot.learning_rate),
		);
		draw_label(
			ui,
			content_scale,
			"training-summary",
			PixelRect::new(
				header.x + px(22),
				header.y + px(40),
				(header.w as i32 - px(44)).max(1) as u32,
				px(20) as u32,
			),
			&summary,
			13.0,
			status_color,
		);
		let fps = if self.viewer_frame_ms > 0.0 {
			1000.0 / self.viewer_frame_ms
		} else {
			0.0
		};
		let timing = format!(
			"GPU {:.3} ms   wall {:.3} ms   viewer {:.1} FPS   Space pause/resume   C checkpoint   E evaluate   S stop",
			snapshot.gpu_ms,
			snapshot.wall_ms,
			f64::from(fps),
		);
		draw_label(
			ui,
			content_scale,
			"training-timing",
			PixelRect::new(
				header.x + px(22),
				header.y + px(70),
				(header.w as i32 - px(44)).max(1) as u32,
				px(18) as u32,
			),
			&timing,
			11.0,
			Color::new(0.68, 0.68, 0.68, 1.0),
		);

		let preview = self
			.preview
			.lock()
			.unwrap_or_else(std::sync::PoisonError::into_inner)
			.current
			.clone();
		let preview_available = self.config.show_preview
			&& preview
				.as_ref()
				.is_some_and(|frame| frame.texture.is_some());
		let show_split_preview = width >= px(720) && preview_available;
		let plot_count = self.series.len().min(self.config.max_metric_plots as usize) as u32;
		let mut plot_top = header.y + header.h as i32 + gap;
		let compact_tabs = width < px(720) && preview_available;
		if compact_tabs {
			let tabs = [
				UiTabItem {
					id: "metrics".into(),
					label: "metrics".into(),
					closable: false,
					..UiTabItem::default()
				},
				UiTabItem {
					id: "preview".into(),
					label: "preview".into(),
					closable: false,
					..UiTabItem::default()
				},
			];
			let mut tab_state = std::mem::take(&mut self.compact_tabs);
			ui.tab_bar(
				"training-compact-tabs",
				PixelRect::new(margin, plot_top, (width - margin * 2) as u32, px(28) as u32),
				&tabs,
				&mut tab_state,
			);
			self.compact_tabs = tab_state;
			plot_top += px(28) + gap;
		}
		let available_height = (height - plot_top - margin).max(px(80));
		let mut plot_viewport = PixelRect::new(
			margin,
			plot_top,
			(width - margin * 2) as u32,
			available_height as u32,
		);
		let mut preview_panel = PixelRect::default();
		if compact_tabs && self.compact_tabs.selected == 1 {
			ui.rect(plot_viewport, Color::new(0.025, 0.025, 0.025, 1.0), 0.0)?;
			ui.begin_panel("training-preview", plot_viewport);
			if let Some(frame) = &preview
				&& let Some(texture) = &frame.texture
			{
				ui.image_at(texture, plot_viewport)?;
			}
			ui.end_panel();
			ui.rect_outline(plot_viewport, Color::new(1.0, 1.0, 1.0, 0.10), 1, 0.0)?;
			return Ok(());
		}
		if show_split_preview {
			let content = ui.split_pane(
				"training-content-split",
				plot_viewport,
				&mut self.preview_split_ratio,
				UiSplitConfig {
					direction: UiDirection::Row,
					handle_size: px(10) as f32,
					minimum_first: px(300) as f32,
					minimum_second: px(280) as f32,
					..UiSplitConfig::default()
				},
			);
			plot_viewport = content.first;
			preview_panel = content.second;
			ui.tooltip(
				"Drag to resize metrics and preview. Use Left/Right for precise adjustment.",
				UiTooltipConfig::default(),
			);
		}
		let plot_area_width = plot_viewport.w as i32;
		let columns = if self.metric_columns == 0 {
			if plot_area_width >= px(760) { 2 } else { 1 }
		} else {
			self.metric_columns
		};
		let rows = ((plot_count as i32 + columns - 1) / columns).max(1);
		let plot_height = if rows <= 1 {
			available_height
		} else {
			(available_height * 3 / 7).clamp(px(128), px(240))
		};
		let content_height = if plot_count == 0 {
			available_height
		} else {
			rows * plot_height + gap * (rows - 1)
		};
		let plots = ui.begin_scroll_panel(
			"training-metrics",
			plot_viewport,
			content_height,
			UiScrollConfig::default(),
		);
		let plot_width = ((plots.content.w as i32 - gap * (columns - 1)) / columns).max(1);
		let visible_rows = ui.virtual_rows(rows, plot_height, gap, 0);
		for row in visible_rows.first..visible_rows.one_past_last {
			for column in 0..columns {
				let index = (row * columns + column) as u32;
				if index >= plot_count {
					break;
				}
				let rect = PixelRect::new(
					plots.content.x + column * (plot_width + gap),
					plots.content.y + row * (plot_height + gap),
					plot_width.max(1) as u32,
					plot_height as u32,
				);
				let (color, latest, samples, name, expanded) = {
					let series = &self.series[index as usize];
					(
						series.color,
						series.values.last().copied().unwrap_or(0.0),
						series.values.len(),
						series.name.clone(),
						series.expanded,
					)
				};
				ui.rect(rect, Color::new(0.045, 0.045, 0.045, 1.0), 0.0)?;
				let tree = ui.tree_row(
					&name,
					PixelRect::new(
						rect.x + px(8),
						rect.y + px(6),
						(rect.w as i32 - px(16)).max(1) as u32,
						px(26) as u32,
					),
					&name,
					UiTreeRowConfig {
						has_children: true,
						open: expanded,
						selected: self.selected_metric == index as i32,
						..UiTreeRowConfig::new()
					},
				);
				if tree.activated {
					self.selected_metric = index as i32;
				}
				if tree.open_changed {
					self.series[index as usize].expanded = tree.open;
				}
				ui.property_row(
					&name,
					PixelRect::new(
						rect.x + px(8),
						rect.y + px(34),
						(rect.w as i32 - px(16)).max(1) as u32,
						px(22) as u32,
					),
					"latest",
					&format!("{latest:.6} · {samples} samples"),
					crate::ui::UiPropertyRowConfig {
						alternate: index & 1 != 0,
						..Default::default()
					},
				);
				let graph = PixelRect::new(
					rect.x + px(8),
					rect.y + px(62),
					(rect.w as i32 - px(16)).max(1) as u32,
					(rect.h as i32 - px(70)).max(1) as u32,
				);
				if expanded && samples > 0 {
					ui.begin_panel(&name, graph);
					ui.plot_line(
						&self.series[index as usize].values,
						graph,
						&UiPlotConfig {
							color,
							auto_scale: true,
							show_grid: true,
							fill: index == 0,
							..UiPlotConfig::default()
						},
					)?;
					ui.end_panel();
				}
			}
		}
		ui.end_scroll_panel();
		if show_split_preview {
			ui.rect(preview_panel, Color::new(0.025, 0.025, 0.025, 1.0), 0.0)?;
			ui.begin_panel("training-preview", preview_panel);
			if let Some(frame) = &preview
				&& let Some(texture) = &frame.texture
			{
				ui.image_at(texture, preview_panel)?;
			}
			ui.end_panel();
			ui.rect_outline(preview_panel, Color::new(1.0, 1.0, 1.0, 0.10), 1, 0.0)?;
		}
		Ok(())
	}

	fn close(&mut self) -> Result<()> {
		Ok(())
	}
}
