//! Real image/audio preview, including media initialization and explicit teardown.

use oa::viewer::{Viewer, ViewerAudioView, ViewerConfig, ViewerMode};

fn run_until_quit(run: impl FnOnce() -> oa::Result<()>) -> Result<(), Box<dyn std::error::Error>> {
	// Retain SDL while the viewer initializes its own window and engine. A queued
	// quit is handled in the loop that still composes and presents its final frame.
	let sdl = sdl3::init()?;
	let events = sdl.event()?;
	let sender = events.event_sender();
	let quit = std::thread::spawn(move || {
		std::thread::sleep(std::time::Duration::from_secs(1));
		sender.push_event(sdl3::event::Event::Quit { timestamp: 0 })
	});
	let result = run();
	quit.join().expect("quit event thread panicked")?;
	result?;
	Ok(())
}

fn preview_until_quit(config: ViewerConfig) -> Result<(), Box<dyn std::error::Error>> {
	run_until_quit(|| Viewer::new(config).run_standalone())
}

#[test]
#[ignore = "requires Vulkan presentation and a live SDL display"]
fn figure_show_opens_image_viewer_and_closes() -> Result<(), Box<dyn std::error::Error>> {
	let probe = oa::SdlWindow::new("Figure presentation probe", 1, 1)?;
	let engine = oa::Engine::builder()
		.requirements(oa::EngineRequirements::default().presentation())
		.instance_extensions(probe.required_instance_extensions()?)
		.build()?;
	oa::Engine::set_default(engine.clone());
	drop(probe);

	let mut figure = oa::plot::Figure::new(oa::plot::FigureConfig {
		title: "Plot image canvas".to_owned(),
		width: 480,
		height: 320,
		..Default::default()
	});
	figure.ax(0, 0).plot(&[5.0, 2.0, 1.0], Default::default());
	run_until_quit(|| figure.show_on(&engine))
}

#[test]
#[ignore = "requires Vulkan presentation and a live SDL display"]
fn viewer_previews_image_and_closes() -> Result<(), Box<dyn std::error::Error>> {
	let path = std::env::temp_dir().join(format!("oa-viewer-{}.png", std::process::id()));
	image_codec::RgbaImage::from_pixel(7, 5, image_codec::Rgba([255, 0, 0, 255])).save(&path)?;
	let result = preview_until_quit(ViewerConfig {
		mode: ViewerMode::Image,
		path: path.to_string_lossy().into_owned(),
		width: 96,
		height: 64,
		..Default::default()
	});
	std::fs::remove_file(path)?;
	result
}

#[test]
#[ignore = "requires Vulkan presentation, a live SDL display and an audio output device"]
fn viewer_previews_all_audio_views_and_closes() -> Result<(), Box<dyn std::error::Error>> {
	let path = std::env::temp_dir().join(format!("oa-viewer-{}.wav", std::process::id()));
	let samples: Vec<f32> = (0..4096)
		.map(|index| (index as f32 * 0.1).sin() * 0.25)
		.collect();
	std::fs::write(
		&path,
		oa::audio::encode_interleaved_wav_f32(&samples, 48_000, 1)?,
	)?;
	let result = [
		ViewerAudioView::Waveform,
		ViewerAudioView::Spectrum,
		ViewerAudioView::Mel,
	]
	.into_iter()
	.try_for_each(|audio_view| {
		preview_until_quit(ViewerConfig {
			mode: ViewerMode::Audio,
			path: path.to_string_lossy().into_owned(),
			width: 96,
			height: 64,
			start_playing: false,
			audio_waveform_bins: 64,
			audio_analysis_frames: 16,
			audio_fft_size: 64,
			audio_hop_size: 64,
			audio_mel_bins: 8,
			audio_view,
			..Default::default()
		})
	});
	std::fs::remove_file(path)?;
	result
}

#[test]
#[ignore = "requires native Vulkan Video, a live SDL display and audio output"]
fn viewer_previews_embedded_movie_and_closes() -> Result<(), Box<dyn std::error::Error>> {
	let path =
		std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("test/fixtures/media/embedded_av.mp4");
	preview_until_quit(ViewerConfig {
		mode: ViewerMode::Auto,
		path: path.to_string_lossy().into_owned(),
		width: 640,
		height: 360,
		start_playing: true,
		auto_hide_controls: false,
		..Default::default()
	})
}
