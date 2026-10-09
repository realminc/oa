// ═══════════════════════════════════════════════════════════════════════════
// OA Tutorial: oa::Viewer — Headless image sink
// Level-0 API — Headless engine + oa::image save/blit/clear
// ═══════════════════════════════════════════════════════════════════════════
//
// The renderer/sink split (architecture/oaArchitecture.md §10).
//
// Same producer (image decode → GPU texture), but no window and no event
// loop.  The output is an oa::Texture; the sink decides what happens to it.
// In the interactive tutorial the sink is a swapchain present. Here the
// sink is save_texture_file().  Identical producer, different terminal node.
//
// This is the batch/CI/render-farm shape: one binary can run on any GPU
// regardless of whether a display server or surface is reachable.
//
// Additionally exercises:
//   - oa::render::blit      — GPU copy of one texture into another
//   - oa::render::clear     — solid-color fill (red)
//   - oa::render::save_texture_file — readback + encode + filesystem write
//
// usage:
//   cargo run --example tu_viewer_image_headless -- [input.jpg] [output.png]
//
// Defaults:
//   input  → sdk/asset/image/coverMl.jpg
//   output → /tmp/oa_viewport_batch.png
//
// Donor: sdk/cpp/tutorials/vision/tuViewerImageHeadless.cpp
// ═══════════════════════════════════════════════════════════════════════════

use oa::image::decode_file as decode_image;
use oa::render::{blit, clear, save_texture_file, texture_from_image};
use oa::{Engine, ImageFormat, Result};
use std::path::PathBuf;

fn main() -> Result<()> {
	let args: Vec<String> = std::env::args().collect();
	let default_input = oa::Path::asset_rel("image/coverMl.jpg");
	let in_path: PathBuf = args
		.get(1)
		.map(PathBuf::from)
		.unwrap_or(default_input.into());
	let out_path: PathBuf = args
		.get(2)
		.map(PathBuf::from)
		.unwrap_or_else(|| PathBuf::from("/tmp/oa_viewport_batch.png"));

	// ── Engine bring-up — headless (pure compute) ─────────────────────────────
	// No surface, no VK_KHR_swapchain.  Any GPU admitted by the driver works.
	let engine = Engine::builder().build()?;

	eprintln!(
		"tu_viewer_image_headless: {} → {}",
		in_path.display(),
		out_path.display()
	);

	// ── Producer: decode a semantic image, then lower it to a Texture ────────
	let image = decode_image(&engine, &in_path, ImageFormat::Rgba)?;
	let tex = texture_from_image(&image)?;

	// ── Sink: save to file (readback + encode + write) ────────────────────────
	save_texture_file(&tex, &out_path, 90)?;
	println!(
		"OK: {}×{}  {} → {}",
		tex.width(),
		tex.height(),
		in_path.display(),
		out_path.display()
	);

	// ── Smoke: blit (GPU copy — should round-trip to original) ───────────────
	let blit_path = PathBuf::from("/tmp/oa_viewport_batch_blit.png");
	let clone = blit(&tex)?;
	save_texture_file(&clone, &blit_path, 90)?;
	println!("OK: Blit → {}", blit_path.display());

	// ── Smoke: clear (solid red) ──────────────────────────────────────────────
	let clear_path = PathBuf::from("/tmp/oa_viewport_batch_clear.png");
	let red = clear(&clone, oa::Color::new(0.95, 0.10, 0.10, 1.0))?;
	save_texture_file(&red, &clear_path, 90)?;
	println!("OK: clear(red) → {}", clear_path.display());

	Ok(())
}
