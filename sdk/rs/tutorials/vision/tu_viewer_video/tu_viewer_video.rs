// ═══════════════════════════════════════════════════════════════════════════
// OA Tutorial: oa::Viewer — Video playback (H.264 / H.265 / VP9 / AV1)
// Level-0 API — oa::Viewer, ViewerMode::Video
// ═══════════════════════════════════════════════════════════════════════════
//
// Opens a video file in the Viewer's video mode, which renders frames using
// the GPU blit path with pan/zoom navigation and a timeline scrub bar.
//
// Controls:
//   Space — play/pause    Left/Right arrow — step frame    Q/Esc — quit
//   +/-   — zoom          0/F — fit to window
//
// usage:
//   cargo run --example tu_viewer_video -- [file.mp4]
//   cargo run --example tu_viewer_video -- [file.mp4] <title>
//   cargo run --example tu_viewer_video -- <file.mp4> --poster out.png [--frame N]
//
// To exercise specific codec fixtures (SDK asset paths):
//   cargo run --example tu_viewer_video -- sdk/asset/video/clip/shibuya_720p_30fps_h264_high_8bit_420.mp4
//   cargo run --example tu_viewer_video -- sdk/asset/video/clip/shibuya_720p_30fps_h265_main_8bit_420.mp4
//   cargo run --example tu_viewer_video -- sdk/asset/video/clip/shibuya_720p_30fps_vp9_profile0_8bit_420.mp4
//   cargo run --example tu_viewer_video -- sdk/asset/video/clip/shibuya_720p_30fps_av1_main_8bit_420.mp4
//
// Donor: sdk/cpp/tutorials/vision/tuViewerVideo{H264,H265,VP9,AV1}.cpp
// ═══════════════════════════════════════════════════════════════════════════

use oa::video::{VideoPlayer, VideoPlayerConfig};
use oa::{Engine, Result, Viewer, ViewerConfig, ViewerMode};

fn main() -> Result<()> {
	let args: Vec<String> = std::env::args().collect();
	let positional = |index: usize| args.get(index).filter(|arg| !arg.starts_with("--"));
	let flag_value = |flag: &str| {
		args
			.iter()
			.position(|arg| arg == flag)
			.and_then(|index| args.get(index + 1))
	};

	// Headless capture mode: decode one frame through the level-1 player and
	// save it — the same native decode + GPU presentation path the Viewer
	// uses, without opening a window. Documentation assets are generated this
	// way (see asset/poster-frame.png).
	if let Some(output) = flag_value("--poster") {
		let frame = flag_value("--frame")
			.and_then(|value| value.parse().ok())
			.unwrap_or(30_u64);
		let default_path = oa::Path::asset_rel("clip/shibuya_720p_30fps_h264_high_8bit_420.mp4");
		let default_str = default_path.to_string_lossy().into_owned();
		let input = positional(1).map(String::as_str).unwrap_or(&default_str);
		return write_poster(input, output, frame);
	}

	// Default to the H.264 fixture when no path is supplied.
	let default_path = oa::Path::asset_rel("clip/shibuya_720p_30fps_h264_high_8bit_420.mp4");
	let default_str = default_path.to_string_lossy().into_owned();
	let path = positional(1).map(String::as_str).unwrap_or(&default_str);
	let title = positional(2)
		.map(String::as_str)
		.unwrap_or("oa::Viewer · Video");

	let config = ViewerConfig {
		mode: ViewerMode::Video,
		path: path.to_owned(),
		title: title.to_owned(),
		loop_media: false,
		..ViewerConfig::default()
	};

	Viewer::new(config).run_standalone()
}

/// Decode `frame` of `input` through the native Vulkan Video path, convert
/// the decoded YUV planes to an RGBA texture on-device, and write a PNG.
fn write_poster(input: &str, output: &str, frame: u64) -> Result<()> {
	let engine = Engine::new()?;
	let mut player = VideoPlayer::new_native(
		&engine,
		std::path::Path::new(input),
		VideoPlayerConfig {
			start_playing: false,
			loop_playback: false,
			..VideoPlayerConfig::default()
		},
	)?;
	player.seek_frame(frame)?;
	let texture = player.current_frame_texture()?;
	let rgba = texture.read_rgba8()?;
	let width = u32::try_from(texture.width())
		.map_err(|_| oa::Error::out_of_range("video frame width exceeds u32"))?;
	let height = u32::try_from(texture.height())
		.map_err(|_| oa::Error::out_of_range("video frame height exceeds u32"))?;
	oa::image::save_rgba_file(&rgba, width, height, output, 90)?;
	player.close();
	Ok(())
}
