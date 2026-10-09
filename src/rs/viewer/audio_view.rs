//! Audio analysis cache. Donor: `oa::Viewer::openAudio` analysis section.

use super::config::ViewerConfig;
use crate::Matrix;

/// Pre-computed offline audio analysis matrices.
pub struct AudioAnalysis {
	pub envelope: Option<Matrix>,
	pub spectrum: Option<Matrix>,
	pub mel: Option<Matrix>,
}

/// Analyze an audio file and return cached matrices.
///
/// Returns `None` when the path cannot be decoded.
pub fn analyze(path: &str, config: &ViewerConfig) -> crate::Result<Option<AudioAnalysis>> {
	// Build a minimal compute engine for the offline analysis pass.
	let engine = crate::Engine::new()?;
	let decoded = match crate::audio::decode_file(&engine, path) {
		Ok(a) => a,
		Err(_) => return Ok(None),
	};
	let envelope = crate::audio::waveform_envelope(&decoded, config.audio_waveform_bins).ok();
	Ok(Some(AudioAnalysis {
		envelope,
		spectrum: None,
		mel: None,
	}))
}
