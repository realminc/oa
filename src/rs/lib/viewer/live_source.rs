//! `ViewerLiveSource` trait and capabilities. Donor: `oa/ui/viewer.h`.

use crate::input::InputEvent;
use crate::ui::{PixelRect, Ui};
pub use crate::viewer::types::ViewerLiveCapabilities;
use crate::{Result, Runtime, Texture};

/// Non-owning live GPU producer attached to the Viewer lifecycle.
///
/// Donor: `oa::ViewerLiveSource`.
pub trait ViewerLiveSource {
	fn capabilities(&self) -> ViewerLiveCapabilities;

	/// Called once when the Viewer obtains an `Runtime`.
	fn open(&mut self, runtime: &Runtime) -> Result<()>;

	/// Per-frame update. `delta_ms` is the frame time in milliseconds.
	fn update(&mut self, delta_ms: f32) -> Result<()>;

	/// Optional camera/image surface presented by the Viewer's shared GPU 2D
	/// canvas. The Viewer owns navigation, fit, zoom, pan and channel selection;
	/// `render` draws only the source's overlays when this is provided.
	fn canvas_texture(&self) -> Option<&Texture> {
		None
	}

	/// Source controls that own pointer gesture starts over the canvas. Motion
	/// and release from an existing canvas drag still reach Navigation.
	fn canvas_controls_rect(&self, _width: u32, _height: u32, _scale: f32) -> Option<PixelRect> {
		None
	}

	/// Render domain-specific UI overlays into `ui`. Called every frame.
	fn render(&mut self, ui: &mut Ui<'_>, width: u32, height: u32) -> Result<()>;

	/// Raw platform event (only called when `receives_events` is true).
	fn on_event(&mut self, _event: &InputEvent) -> Result<()> {
		Ok(())
	}

	/// Called once when the Viewer is closing.
	fn close(&mut self) -> Result<()>;
}
