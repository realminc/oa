//! Workspace dock model for `ViewerMode::Workspace`.
//!
//! The workspace composes existing `Ui` primitives — `split_pane` for the dock
//! tree and `tab_bar` for leaf pane selection — into an editor-style host.
//! The dock layout is serializable application state: [`Workspace::to_json`]
//! and [`Workspace::from_json`] round-trip the node tree, while pane content
//! stays outside the serialized record. `Ui` supplies the primitives and
//! input routing; the workspace owns no document model, asset browser, or
//! undo stack.

use serde_json::Value;

use super::{ViewerLiveCapabilities, ViewerLiveSource};
use crate::input::{InputEvent, InputEventKind};
use crate::ui::{PixelRect, Ui, UiDirection, UiSplitConfig, UiTabBarState, UiTabItem};
use crate::{Engine, Error, Result};

/// Content provider rendered inside one workspace leaf pane.
///
/// Donor scope note: items are application-owned surfaces (figure views,
/// media views, inspectors); the workspace only routes geometry and events.
pub trait WorkspaceItem {
	/// Stable identifier referenced by `WorkspaceNode::Tabs::tabs`.
	fn id(&self) -> &str;
	/// Visible tab label.
	fn title(&self) -> &str;
	/// Whether the pane's tab shows a close button.
	fn closable(&self) -> bool {
		true
	}
	/// Per-frame update. `delta_ms` is the frame time in milliseconds.
	fn update(&mut self, _delta_ms: f32) -> Result<()> {
		Ok(())
	}
	/// Render the item inside `rect`, clipped to the pane's panel scope.
	fn render(&mut self, ui: &mut Ui<'_>, rect: PixelRect) -> Result<()>;
	/// Pointer events are routed to the leaf under the cursor; key and text
	/// events are routed to the focused leaf (the last pane clicked).
	fn on_event(&mut self, _event: &InputEvent) -> Result<()> {
		Ok(())
	}
}

/// One node of the serializable dock tree.
#[derive(Clone, Debug)]
pub enum WorkspaceNode {
	/// Leaf pane: a tab bar over a set of item ids plus caller-owned tab state.
	Tabs {
		tabs: Vec<String>,
		state: UiTabBarState,
	},
	/// Interior split: two children laid out by `direction` at `ratio`.
	Split {
		direction: UiDirection,
		ratio: f32,
		first: Box<WorkspaceNode>,
		second: Box<WorkspaceNode>,
	},
}

impl WorkspaceNode {
	/// Leaf pane holding `tabs` with the first tab selected.
	pub fn tabs(tabs: &[&str]) -> Self {
		Self::Tabs {
			tabs: tabs.iter().map(|t| (*t).to_owned()).collect(),
			state: UiTabBarState::default(),
		}
	}

	/// Split `first`/`second` at `ratio` along `direction`.
	pub fn split(
		direction: UiDirection,
		ratio: f32,
		first: WorkspaceNode,
		second: WorkspaceNode,
	) -> Self {
		Self::Split {
			direction,
			ratio,
			first: Box::new(first),
			second: Box::new(second),
		}
	}

	fn to_json_value(&self) -> Value {
		match self {
			Self::Tabs { tabs, state } => serde_json::json!({
				"kind": "tabs",
				"tabs": tabs,
				"selected": state.selected,
				"first_visible": state.first_visible,
			}),
			Self::Split {
				direction,
				ratio,
				first,
				second,
			} => serde_json::json!({
				"kind": "split",
				"direction": match direction {
					UiDirection::Row => "row",
					UiDirection::Column => "column",
				},
				"ratio": ratio,
				"first": first.to_json_value(),
				"second": second.to_json_value(),
			}),
		}
	}

	fn from_json_value(value: &Value) -> Result<Self> {
		let kind = value
			.get("kind")
			.and_then(Value::as_str)
			.ok_or_else(|| Error::invalid_argument("workspace node is missing a string `kind`"))?;
		match kind {
			"tabs" => {
				let tabs = value
					.get("tabs")
					.and_then(Value::as_array)
					.ok_or_else(|| Error::invalid_argument("workspace tabs node needs a `tabs` array"))?
					.iter()
					.map(|t| {
						t.as_str()
							.map(str::to_owned)
							.ok_or_else(|| Error::invalid_argument("workspace tab ids must be strings"))
					})
					.collect::<Result<Vec<String>>>()?;
				let selected = value.get("selected").and_then(Value::as_i64).unwrap_or(0) as i32;
				let first_visible = value
					.get("first_visible")
					.and_then(Value::as_i64)
					.unwrap_or(0) as i32;
				Ok(Self::Tabs {
					tabs,
					state: UiTabBarState {
						selected,
						first_visible,
						..UiTabBarState::default()
					},
				})
			}
			"split" => {
				let direction = match value.get("direction").and_then(Value::as_str) {
					Some("row") => UiDirection::Row,
					Some("column") => UiDirection::Column,
					_ => {
						return Err(Error::invalid_argument(
							"workspace split direction must be `row` or `column`",
						));
					}
				};
				let ratio =
					value.get("ratio").and_then(Value::as_f64).ok_or_else(|| {
						Error::invalid_argument("workspace split node needs a numeric `ratio`")
					})? as f32;
				if !ratio.is_finite() {
					return Err(Error::invalid_argument(
						"workspace split ratio must be finite",
					));
				}
				let first = value
					.get("first")
					.ok_or_else(|| Error::invalid_argument("workspace split node is missing `first`"))?;
				let second = value
					.get("second")
					.ok_or_else(|| Error::invalid_argument("workspace split node is missing `second`"))?;
				Ok(Self::Split {
					direction,
					ratio,
					first: Box::new(Self::from_json_value(first)?),
					second: Box::new(Self::from_json_value(second)?),
				})
			}
			_ => Err(Error::invalid_argument(
				"workspace node `kind` must be `tabs` or `split`",
			)),
		}
	}
}

/// `Rc<RefCell<T>>` adapter: registers shared item state — the ownership
/// pattern the Python bindings use for `FigureView` — as a workspace item.
/// `id`/`title`/`closable` are snapshotted at construction because RefCell
/// borrows cannot escape the trait's `&self` signatures; pane identity is
/// treated as stable for the workspace's lifetime.
pub struct SharedItem<T: WorkspaceItem> {
	inner: std::rc::Rc<std::cell::RefCell<T>>,
	id: String,
	title: String,
	closable: bool,
}

impl<T: WorkspaceItem> SharedItem<T> {
	/// Snapshot the item's identity and register the shared state.
	pub fn new(inner: std::rc::Rc<std::cell::RefCell<T>>) -> Self {
		let (id, title, closable) = {
			let borrowed = inner.borrow();
			(
				borrowed.id().to_owned(),
				borrowed.title().to_owned(),
				borrowed.closable(),
			)
		};
		Self {
			inner,
			id,
			title,
			closable,
		}
	}
}

impl<T: WorkspaceItem> WorkspaceItem for SharedItem<T> {
	fn id(&self) -> &str {
		&self.id
	}

	fn title(&self) -> &str {
		&self.title
	}

	fn closable(&self) -> bool {
		self.closable
	}

	fn update(&mut self, delta_ms: f32) -> Result<()> {
		self.inner.borrow_mut().update(delta_ms)
	}

	fn render(&mut self, ui: &mut Ui<'_>, rect: PixelRect) -> Result<()> {
		self.inner.borrow_mut().render(ui, rect)
	}

	fn on_event(&mut self, event: &InputEvent) -> Result<()> {
		self.inner.borrow_mut().on_event(event)
	}
}

/// Editor-style dock host: a serializable node tree over registered items.
///
/// Implements [`ViewerLiveSource`] so a `Viewer` opened with
/// `ViewerMode::Workspace` renders and routes through it unchanged; the
/// dock layout round-trips through [`Workspace::to_json`] /
/// [`Workspace::from_json`] as application-owned state.
pub struct Workspace {
	root: WorkspaceNode,
	items: Vec<Box<dyn WorkspaceItem>>,
	/// (item id, leaf rect) pairs from the last rendered frame; pointer events
	/// are routed to the leaf whose rect contains the cursor.
	hits: Vec<(String, PixelRect)>,
	/// Leaf geometry from the current render pass; tab drops resolve against
	/// these rects once the dock walk completes.
	panes: Vec<PaneHit>,
	/// Tab drops reported during the current render pass: (source leaf path,
	/// dragged tab index, insertion index). Applied after traversal.
	drops: Vec<(String, i32, i32)>,
	/// Item that owns key/text events; set when a pane is clicked.
	focused: Option<String>,
}

/// One rendered leaf pane's geometry.
struct PaneHit {
	/// Dock path of the leaf (`"ws"`, `"ws/0"`, `"ws/1"`, ...).
	path: String,
	/// Full leaf rectangle, tab bar included.
	rect: PixelRect,
	/// Tab-bar strip inside `rect`.
	bar: PixelRect,
}

impl Workspace {
	/// Empty workspace over `root`; register content with [`Workspace::add_item`].
	pub fn new(root: WorkspaceNode) -> Self {
		Self {
			root,
			items: Vec::new(),
			hits: Vec::new(),
			panes: Vec::new(),
			drops: Vec::new(),
			focused: None,
		}
	}

	/// Replace the dock tree with a parsed layout document (see
	/// [`Workspace::from_json`]); registered items are preserved.
	pub fn set_layout(&mut self, json: &str) -> Result<()> {
		self.root = Workspace::from_json(json)?.root;
		Ok(())
	}

	/// Register `item` and append its id to the root tab set. Returns an
	/// error when the root is a split — multi-pane docks place tabs through
	/// [`Workspace::root_mut`] or [`Workspace::set_layout`].
	pub fn open_tab(&mut self, item: Box<dyn WorkspaceItem>) -> Result<()> {
		let id = item.id().to_owned();
		self.add_item(item);
		match &mut self.root {
			WorkspaceNode::Tabs { tabs, state } => {
				tabs.push(id);
				if state.selected < 0 {
					state.selected = 0;
				}
				Ok(())
			}
			WorkspaceNode::Split { .. } => Err(Error::invalid_argument(
				"Workspace::open_tab needs a tabbed root; place tabs on a split dock through root_mut or set_layout",
			)),
		}
	}

	/// Borrow the dock tree.
	pub fn root(&self) -> &WorkspaceNode {
		&self.root
	}

	/// Mutably borrow the dock tree (e.g. to attach or detach panes).
	pub fn root_mut(&mut self) -> &mut WorkspaceNode {
		&mut self.root
	}

	/// Register pane content. Registering a second item under an existing id
	/// replaces the previous entry.
	pub fn add_item(&mut self, item: Box<dyn WorkspaceItem>) {
		self.items.retain(|i| i.id() != item.id());
		self.items.push(item);
	}

	/// Look up registered content by id.
	pub fn item(&self, id: &str) -> Option<&dyn WorkspaceItem> {
		self.items.iter().find(|i| i.id() == id).map(|i| &**i)
	}

	/// Mutably look up registered content by id.
	pub fn item_mut(&mut self, id: &str) -> Option<&mut (dyn WorkspaceItem + '_)> {
		for item in &mut self.items {
			if item.id() == id {
				return Some(&mut **item);
			}
		}
		None
	}

	/// Registered item ids in registration order.
	pub fn item_ids(&self) -> impl Iterator<Item = &str> {
		self.items.iter().map(|i| i.id())
	}

	/// Serialize the dock tree (no item content) to a JSON document.
	pub fn to_json(&self) -> Result<String> {
		let document = serde_json::json!({
			"version": 1,
			"root": self.root.to_json_value(),
		});
		serde_json::to_string(&document)
			.map_err(|e| Error::backend_failure("serde_json", "workspace layout encode", e))
	}

	/// Parse a document produced by [`Workspace::to_json`]. Tab ids are
	/// preserved verbatim; items are registered separately and unresolved ids
	/// render as disabled tabs until content arrives.
	pub fn from_json(json: &str) -> Result<Self> {
		let document: Value = serde_json::from_str(json)
			.map_err(|e| Error::invalid_argument(format!("workspace layout JSON is malformed: {e}")))?;
		let version = document.get("version").and_then(Value::as_u64).unwrap_or(0);
		if version != 1 {
			return Err(Error::invalid_argument(
				"workspace layout version must be 1",
			));
		}
		let root = document
			.get("root")
			.ok_or_else(|| Error::invalid_argument("workspace layout document is missing `root`"))?;
		Ok(Self::new(WorkspaceNode::from_json_value(root)?))
	}

	/// Record one frame: walk the dock tree over `rect`, driving `split_pane`
	/// and `tab_bar`, then render each leaf's selected item.
	pub fn render(&mut self, ui: &mut Ui<'_>, rect: PixelRect) -> Result<()> {
		self.hits.clear();
		self.panes.clear();
		self.drops.clear();
		let pointer = [ui.input().pointer_x, ui.input().pointer_y];
		let tab_drag_active = tab_drag_active(&self.root);
		render_node(
			&mut self.root,
			&mut DockFrame {
				items: &mut self.items,
				hits: &mut self.hits,
				panes: &mut self.panes,
				drops: &mut self.drops,
				tab_drag_active,
				pointer,
			},
			ui,
			rect,
			"ws",
		)?;
		for (source, index, insert) in std::mem::take(&mut self.drops) {
			self.apply_tab_drop(&source, index, insert, pointer);
		}
		Ok(())
	}

	/// Apply one released tab drag: dropping on another leaf's edge quarter
	/// splits that leaf so the tab opens a new pane on that side; dropping on
	/// its tab bar or center merges the tab into the leaf; dropping on the
	/// source bar reorders in place; anywhere else cancels the move. A leaf
	/// emptied by a completed move collapses out of the split tree.
	fn apply_tab_drop(&mut self, source: &str, index: i32, insert: i32, pointer: [f32; 2]) {
		let [px, py] = pointer;
		let in_rect = |r: PixelRect| {
			px >= r.x as f32
				&& py >= r.y as f32
				&& px < (r.x + r.w as i32) as f32
				&& py < (r.y + r.h as i32) as f32
		};
		let Some(target) = self.panes.iter().find(|p| in_rect(p.rect)) else {
			return;
		};
		if target.path == source && !in_rect(target.bar) {
			// Dropped back onto the source leaf's content: no move.
			return;
		}
		let target_path = target.path.clone();
		let over_target_bar = in_rect(target.bar);
		let zone = classify_zone(target.rect, px, py);

		let Some(WorkspaceNode::Tabs { tabs, state }) = node_at_mut(&mut self.root, source) else {
			return;
		};
		if index < 0 || index as usize >= tabs.len() {
			return;
		}
		let id = tabs.remove(index as usize);
		if state.selected >= tabs.len() as i32 {
			state.selected = tabs.len() as i32 - 1;
		}

		if target_path == source {
			// In-bar reorder: `insert` is a pre-removal boundary index.
			let at = if insert > index { insert - 1 } else { insert };
			tabs.insert(at.clamp(0, tabs.len() as i32) as usize, id);
			state.selected = at.clamp(0, tabs.len() as i32 - 1);
			return;
		}
		if !over_target_bar && zone != DropZone::Center {
			// Edge zone: wrap the target leaf in a new split so the dragged
			// tab opens a pane on that side.
			let direction = match zone {
				DropZone::Left | DropZone::Right => UiDirection::Row,
				DropZone::Top | DropZone::Bottom | DropZone::Center => UiDirection::Column,
			};
			let leaf = WorkspaceNode::Tabs {
				tabs: vec![id],
				state: UiTabBarState::default(),
			};
			let Some(target_node) = node_at_mut(&mut self.root, &target_path) else {
				return;
			};
			let existing = std::mem::replace(target_node, WorkspaceNode::tabs(&[]));
			let (first, second) = if matches!(zone, DropZone::Left | DropZone::Top) {
				(leaf, existing)
			} else {
				(existing, leaf)
			};
			*target_node = WorkspaceNode::Split {
				direction,
				ratio: 0.5,
				first: Box::new(first),
				second: Box::new(second),
			};
			collapse_empty(&mut self.root);
			return;
		}
		let Some(WorkspaceNode::Tabs {
			tabs: target_tabs,
			state: target_state,
		}) = node_at_mut(&mut self.root, &target_path)
		else {
			return;
		};
		let at = if over_target_bar {
			// Mirror Ui::tab_bar's slot width so the drop lands where the
			// insertion indicator was drawn.
			let bar = self
				.panes
				.iter()
				.find(|p| p.path == target_path)
				.map(|p| p.bar)
				.unwrap_or_default();
			let tab_w = (bar.w as i32 / target_tabs.len().max(1) as i32).clamp(88, 220);
			(((px - bar.x as f32) / tab_w as f32) + 0.5) as i32
		} else {
			target_tabs.len() as i32
		};
		target_tabs.insert(at.clamp(0, target_tabs.len() as i32) as usize, id);
		target_state.selected = target_tabs.len() as i32 - 1;
		collapse_empty(&mut self.root);
	}

	/// Route a UI-space input event: pointer events go to the leaf under the
	/// cursor (and move focus there on press); key and text events go to the
	/// focused leaf.
	pub fn route_event(&mut self, event: &InputEvent) -> Result<()> {
		let is_pointer = matches!(
			event.kind,
			InputEventKind::MouseDown
				| InputEventKind::MouseUp
				| InputEventKind::MouseMove
				| InputEventKind::MouseScroll
		);
		let target = if is_pointer {
			let hit = self
				.hits
				.iter()
				.find(|(_, rect)| {
					let px = event.mouse_x as i32;
					let py = event.mouse_y as i32;
					px >= rect.x && py >= rect.y && px < rect.x + rect.w as i32 && py < rect.y + rect.h as i32
				})
				.map(|(id, _)| id.clone());
			if matches!(event.kind, InputEventKind::MouseDown)
				&& event.button == 1
				&& let Some(id) = &hit
			{
				self.focused = Some(id.clone());
			}
			hit.or_else(|| self.focused.clone())
		} else {
			self.focused.clone()
		};
		if let Some(id) = target
			&& let Some(item) = self.item_mut(&id)
		{
			item.on_event(event)?;
		}
		Ok(())
	}
}

impl Default for Workspace {
	/// One empty tabbed root — the common "add views and run" dock.
	fn default() -> Self {
		Self::new(WorkspaceNode::tabs(&[]))
	}
}

impl ViewerLiveSource for Workspace {
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
		for item in &mut self.items {
			item.update(delta_ms)?;
		}
		Ok(())
	}

	fn render(&mut self, ui: &mut Ui<'_>, width: u32, height: u32) -> Result<()> {
		Workspace::render(self, ui, PixelRect::new(0, 0, width, height))
	}

	fn on_event(&mut self, event: &InputEvent) -> Result<()> {
		self.route_event(event)
	}

	fn close(&mut self) -> Result<()> {
		Ok(())
	}
}

/// Per-frame out-params threaded through the dock walk.
struct DockFrame<'a> {
	items: &'a mut [Box<dyn WorkspaceItem>],
	hits: &'a mut Vec<(String, PixelRect)>,
	panes: &'a mut Vec<PaneHit>,
	drops: &'a mut Vec<(String, i32, i32)>,
	/// Whether a tab is held anywhere in the dock; drives the drop-zone
	/// preview on leaves other than the drag source.
	tab_drag_active: bool,
	/// UI-space pointer position for preview classification.
	pointer: [f32; 2],
}

/// Where inside a leaf's rect a released tab drop lands.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum DropZone {
	/// Split the leaf along the row axis with the dragged tab left of it.
	Left,
	/// Split the leaf along the row axis with the dragged tab right of it.
	Right,
	/// Split the leaf along the column axis with the dragged tab above it.
	Top,
	/// Split the leaf along the column axis with the dragged tab below it.
	Bottom,
	/// Merge into the leaf's tab set (tab bar or middle region).
	Center,
}

/// Classify `px,py` inside `rect`: the outer quarter on each side is an
/// edge-split zone (corners resolve to the deeper edge); the interior is
/// the merge zone.
fn classify_zone(rect: PixelRect, px: f32, py: f32) -> DropZone {
	let x = px - rect.x as f32;
	let y = py - rect.y as f32;
	let w = rect.w as f32;
	let h = rect.h as f32;
	let edges = [
		(w * 0.25 - x, DropZone::Left),
		(x - w * 0.75, DropZone::Right),
		(h * 0.25 - y, DropZone::Top),
		(y - h * 0.75, DropZone::Bottom),
	];
	let (depth, zone) = edges
		.iter()
		.fold((f32::NEG_INFINITY, DropZone::Center), |best, &(d, z)| {
			if d > best.0 { (d, z) } else { best }
		});
	if depth > 0.0 { zone } else { DropZone::Center }
}

/// Whether any leaf currently holds a pressed tab — the drag may still be
/// inside its movement threshold, but the preview only shows when the
/// pointer has already reached another leaf.
fn tab_drag_active(node: &WorkspaceNode) -> bool {
	match node {
		WorkspaceNode::Tabs { state, .. } => state.drag_index >= 0,
		WorkspaceNode::Split { first, second, .. } => tab_drag_active(first) || tab_drag_active(second),
	}
}

/// Merge leaf panes left empty by a completed move out of the split tree;
/// the root always keeps at least one node.
fn collapse_empty(node: &mut WorkspaceNode) {
	let WorkspaceNode::Split { first, second, .. } = node else {
		return;
	};
	collapse_empty(first);
	collapse_empty(second);
	let first_empty = matches!(&**first, WorkspaceNode::Tabs { tabs, .. } if tabs.is_empty());
	let second_empty = matches!(&**second, WorkspaceNode::Tabs { tabs, .. } if tabs.is_empty());
	if second_empty {
		*node = std::mem::replace(&mut **first, WorkspaceNode::tabs(&[]));
	} else if first_empty {
		*node = std::mem::replace(&mut **second, WorkspaceNode::tabs(&[]));
	}
}

/// Half of `rect` on `zone`'s side — the region a new pane would occupy.
fn zone_preview_rect(rect: PixelRect, zone: DropZone) -> PixelRect {
	let half_w = rect.w / 2;
	let half_h = rect.h / 2;
	match zone {
		DropZone::Left => PixelRect { w: half_w, ..rect },
		DropZone::Right => PixelRect {
			x: rect.x + (rect.w - half_w) as i32,
			w: half_w,
			..rect
		},
		DropZone::Top => PixelRect { h: half_h, ..rect },
		DropZone::Bottom => PixelRect {
			y: rect.y + (rect.h - half_h) as i32,
			h: half_h,
			..rect
		},
		DropZone::Center => rect,
	}
}

/// Render one dock node. `path` is the stable widget-id prefix for this
/// subtree position (`"ws"`, `"ws/0"`, `"ws/1"`, ...).
fn render_node(
	node: &mut WorkspaceNode,
	frame: &mut DockFrame<'_>,
	ui: &mut Ui<'_>,
	rect: PixelRect,
	path: &str,
) -> Result<()> {
	if rect.w == 0 || rect.h == 0 {
		return Ok(());
	}
	match node {
		WorkspaceNode::Tabs { tabs, state } => {
			let bar_h = (28.0 * ui.content_scale()).round().max(1.0) as u32;
			let bar_rect = PixelRect {
				x: rect.x,
				y: rect.y,
				w: rect.w,
				h: bar_h.min(rect.h),
			};
			let content_rect = PixelRect {
				x: rect.x,
				y: rect.y + bar_rect.h as i32,
				w: rect.w,
				h: rect.h.saturating_sub(bar_rect.h),
			};
			let tab_items: Vec<UiTabItem> = tabs
				.iter()
				.map(|id| {
					let registered = frame.items.iter().find(|i| i.id() == id);
					UiTabItem {
						id: id.clone(),
						label: registered
							.map(|i| i.title().to_owned())
							.unwrap_or_else(|| id.clone()),
						dirty: false,
						closable: registered.is_none_or(|i| i.closable()),
						enabled: registered.is_some(),
					}
				})
				.collect();
			frame.panes.push(PaneHit {
				path: path.to_owned(),
				rect,
				bar: bar_rect,
			});
			let result = ui.tab_bar(path, bar_rect, &tab_items, state);
			if result.dropped {
				frame
					.drops
					.push((path.to_owned(), result.dragged_index, result.drop_index));
			}
			if result.close_requested_index >= 0 && (result.close_requested_index as usize) < tabs.len() {
				tabs.remove(result.close_requested_index as usize);
				if state.selected >= tabs.len() as i32 {
					state.selected = tabs.len() as i32 - 1;
				}
				return Ok(());
			}
			let selected = state.selected.clamp(0, tabs.len().saturating_sub(1) as i32);
			if let Some(tab_id) = tabs.get(selected.max(0) as usize).cloned() {
				frame.hits.push((tab_id.clone(), rect));
				if let Some(item) = frame.items.iter_mut().find(|i| i.id() == tab_id) {
					ui.begin_panel(&format!("{path}/pane"), content_rect);
					let result = item.render(ui, content_rect);
					ui.end_panel();
					result?;
				}
			}
			// Drop-zone preview: while a tab is held elsewhere, tint the
			// region a release here would hand to the dragged tab.
			let [px, py] = frame.pointer;
			if frame.tab_drag_active
				&& state.drag_index < 0
				&& px >= rect.x as f32
				&& px < (rect.x + rect.w as i32) as f32
				&& py >= (rect.y + bar_rect.h as i32) as f32
				&& py < (rect.y + rect.h as i32) as f32
			{
				let zone = classify_zone(rect, px, py);
				let preview = zone_preview_rect(rect, zone);
				let accent = ui.current_style().accent;
				ui.rect_raw(preview, accent.with_alpha(0.18), preview, 4.0);
				ui.rect_outline_raw(preview, accent.with_alpha(0.7), 1, preview);
			}
			Ok(())
		}
		WorkspaceNode::Split {
			direction,
			ratio,
			first,
			second,
		} => {
			let region = ui.split_pane(
				path,
				rect,
				ratio,
				UiSplitConfig {
					direction: *direction,
					handle_size: 4.0,
					minimum_first: 96.0,
					minimum_second: 96.0,
					keyboard_step: 16.0,
				},
			);
			render_node(first, frame, ui, region.first, &format!("{path}/0"))?;
			render_node(second, frame, ui, region.second, &format!("{path}/1"))?;
			Ok(())
		}
	}
}

/// Borrow the dock node at `path` (`"ws"`, `"ws/0"`, `"ws/1"`, ...), where
/// each numeric segment selects a split child (`0` = first, `1` = second).
fn node_at_mut<'a>(node: &'a mut WorkspaceNode, path: &str) -> Option<&'a mut WorkspaceNode> {
	let mut cur = node;
	for segment in path.split('/').skip(1) {
		let WorkspaceNode::Split { first, second, .. } = cur else {
			return None;
		};
		cur = match segment {
			"0" => first,
			"1" => second,
			_ => return None,
		};
	}
	Some(cur)
}
