//! `ViewerLiveSource` trait and capabilities. Donor: `oa/ui/viewer.h`.

use crate::ui::{Ui, types::UiEvent};
use crate::{Engine, Result};

/// Which optional callbacks the live source implements.
#[derive(Clone, Copy, Debug, Default)]
pub struct ViewerLiveCapabilities {
	/// Source requests raw platform events before Viewer focus/popup ownership.
	pub receives_events: bool,
	/// Source publishes a GPU-dependency event for its rendered resources.
	pub publishes_render_dependency: bool,
	/// Source wants the exact Viewer submission event after each present.
	pub retains_consumer_completion: bool,
}

/// Non-owning live GPU producer attached to the Viewer lifecycle.
///
/// Donor: `oa::ViewerLiveSource`.
pub trait ViewerLiveSource {
	fn capabilities(&self) -> ViewerLiveCapabilities;

	/// Called once when the Viewer obtains an `Engine`.
	fn open(&mut self, engine: &Engine) -> Result<()>;

	/// Per-frame update. `delta_ms` is the frame time in milliseconds.
	fn update(&mut self, delta_ms: f32) -> Result<()>;

	/// Render domain-specific UI overlays into `ui`. Called every frame.
	fn render(&mut self, ui: &mut Ui<'_>, width: u32, height: u32) -> Result<()>;

	/// Raw platform event (only called when `receives_events` is true).
	fn on_event(&mut self, _event: &UiEvent) -> Result<()> {
		Ok(())
	}

	/// Called once when the Viewer is closing.
	fn close(&mut self) -> Result<()>;
}
