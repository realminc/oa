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
//
// To exercise specific codec fixtures (SDK asset paths):
//   cargo run --example tu_viewer_video -- sdk/asset/clip/shibuya_720p_30fps_h264_high_8bit_420.mp4
//   cargo run --example tu_viewer_video -- sdk/asset/clip/shibuya_720p_30fps_h265_main_8bit_420.mp4
//   cargo run --example tu_viewer_video -- sdk/asset/clip/shibuya_720p_30fps_vp9_profile0_8bit_420.webm
//   cargo run --example tu_viewer_video -- sdk/asset/clip/shibuya_720p_30fps_av1_main_8bit_420.mp4
//
// Donor: sdk/cpp/tutorials/vision/tuViewerVideo{H264,H265,VP9,AV1}.cpp
// ═══════════════════════════════════════════════════════════════════════════

use oa::{Result, Viewer, ViewerConfig, ViewerMode};

fn main() -> Result<()> {
	let args: Vec<String> = std::env::args().collect();

	// Default to the H.264 fixture when no path is supplied.
	let default_path = oa::Path::asset_rel("clip/shibuya_720p_30fps_h264_high_8bit_420.mp4");
	let default_str = default_path.to_string_lossy().into_owned();
	let path = args.get(1).map(String::as_str).unwrap_or(&default_str);
	let title = args
		.get(2)
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
