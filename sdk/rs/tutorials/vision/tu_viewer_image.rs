// ═══════════════════════════════════════════════════════════════════════════
// OA Tutorial: oa::Viewer — Universal GPU-Accelerated Image Viewer
// Level-0 API — oa::Viewer (preconfigured application class)
// ═══════════════════════════════════════════════════════════════════════════
//
// Demonstrates the unified viewer application with interactive image
// navigation.
//
// Parallel structure to OpenCV's image-display tutorial:
//
//   OpenCV                           OA Rust
//   ─────────────────────────────    ──────────────────────────────────────
//   cv::imread(path)                 let path = …;
//   cv::imshow("name", mat)          Viewer::preview(path)?;
//   cv::waitKey(0)
//
// Controls (same as oa::Viewer):
//   LMB / MMB / wheel — pan and zoom
//   +/-  — zoom in / out     0 / F — fit to window     9 — 100 %
//   1–4  — R/G/B/A channel   5 — RGB composite         B / G — canvas mode
//   Q / Esc — quit
//
// usage:  cargo run --example tu_viewer_image -- [image.jpg]
//
// Donor: sdk/cpp/tutorials/vision/tuViewerImage.cpp
// ═══════════════════════════════════════════════════════════════════════════

fn main() -> oa::Result<()> {
	let args: Vec<String> = std::env::args().collect();
	let default_path = oa::Path::asset_rel("image/coverMl.jpg");
	let path = args
		.get(1)
		.map(String::as_str)
		.unwrap_or_else(|| default_path.to_str().unwrap_or(""));

	// ── OA_DOC_BEGIN: viewer-intro ──────────────────────────────────────────
	oa::Viewer::preview(path)
	// ── OA_DOC_END: viewer-intro ────────────────────────────────────────────
}
