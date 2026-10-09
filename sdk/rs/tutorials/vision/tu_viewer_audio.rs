// ═══════════════════════════════════════════════════════════════════════════
// OA Tutorial: oa::Viewer — Audio playback and analysis
// Level-0 API — oa::Viewer, ViewerMode::Audio
// ═══════════════════════════════════════════════════════════════════════════
//
// Opens a WAV/FLAC/MP3 audio file in the Viewer's audio mode, which renders
// an interactive waveform, spectrum, or Mel-spectrogram visualization and
// plays the audio through the default CPAL output device.
//
// Press Space to play/pause; Q or Esc to quit.
//
// usage:
//   cargo run --example tu_viewer_audio -- [file.flac]
//   cargo run --example tu_viewer_audio -- [file.flac] waveform
//   cargo run --example tu_viewer_audio -- [file.flac] spectrum
//   cargo run --example tu_viewer_audio -- [file.flac] mel
//
// Defaults: sdk/asset/audio/oaNarration.flac, waveform view.
//
// Donor: sdk/cpp/tutorials/vision/tuViewerAudio.cpp
// ═══════════════════════════════════════════════════════════════════════════

use oa::{Result, Viewer, ViewerAudioView, ViewerConfig, ViewerMode};

fn main() -> Result<()> {
	let args: Vec<String> = std::env::args().collect();
	let default_path = oa::Path::asset_rel("audio/oaNarration.flac");
	let default_str = default_path.to_string_lossy().into_owned();
	let path = args.get(1).map(String::as_str).unwrap_or(&default_str);

	let audio_view = match args.get(2).map(String::as_str) {
		Some("spectrum") => ViewerAudioView::Spectrum,
		Some("mel") => ViewerAudioView::Mel,
		_ => ViewerAudioView::Waveform,
	};

	let config = ViewerConfig {
		mode: ViewerMode::Audio,
		path: path.to_owned(),
		audio_view,
		title: "oa::Viewer · Audio".to_owned(),
		width: 960,
		height: 360,
		..ViewerConfig::default()
	};

	Viewer::new(config).run_standalone()
}
