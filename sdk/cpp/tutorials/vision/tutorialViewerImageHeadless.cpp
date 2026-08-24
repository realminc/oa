// ═══════════════════════════════════════════════════════════════════════════
// OA Tutorial: oa::Viewer — Headless image sink
// level 0 API — Headless engine + oa::FnImage::SaveTextureFile
// ═══════════════════════════════════════════════════════════════════════════
//
// The renderer/sink split in code (architecture/oaArchitecture.md §10).
//
// Same renderer (here: a still-image load — the simplest possible producer),
// no window, no swapchain, no event loop. The renderer's output is an
// oa::Texture; the sink decides what happens to it. in GUI mode (the existing
// TutorialViewerImage) the sink is present(swapchain, target). Here the sink is
// saveFile(target, path). Identical producer, different terminal node.
//
// This is the Maya-style batch / render-farm / CI worker shape: one binary
// can ship in every environment that has a GPU, regardless of whether a
// display server / surface is reachable.
//
// usage:
//   ./TutorialViewerImageHeadless [input.jpg] [output.png]
//
// Defaults: sdk/asset/image/coverMl.jpg → /tmp/oa_viewport_batch.png
// ═══════════════════════════════════════════════════════════════════════════

#include <oa/runtime/engine.h>
#include <oa/ui/image.h>
#include <oa/vision/fnImage.h>
#include <oa/core/log.h>
#include <oa/core/paths.h>

#include <cstdio>
#include <vector>


int main(int argc, char** argv) {
	const oa::String defaultInput =
		oa::Paths::asset("image/coverMl.jpg").string();
	const char* inPath  = (argc > 1) ? argv[1] : defaultInput.cStr();
	const char* outPath = (argc > 2) ? argv[2] : "/tmp/oa_viewport_batch.png";

	// ─── Engine bring-up ─────────────────────────────────────────────────────
	// PresentationMode::None — pure compute. We never instantiate a surface,
	// never request VK_KHR_swapchain, and the runtime is free to pick any
	// admitted vulkan device. This is
	// exactly the engine config a render-farm worker or CI box would use.
	oa::EngineConfig cfg;
	cfg.presentationMode = oa::PresentationMode::None;
	cfg.selectForThread = true;

	auto engineResult = oa::Engine::create(cfg);
	if (not engineResult.isOk()) {
		std::fprintf(stderr, "oa::Engine::create failed: %s\n",
			engineResult.getStatus().toString().cStr());
		return 1;
	}
	oa::Engine& engine = *engineResult.getValue();

	OaLogInfo(oa::LogComponent::App,
		"TutorialViewerImageHeadless: %s → %s", inPath, outPath);

	// ─── producer: decode a semantic image, then lower it to a texture ────────
	auto imageR = oa::FnImage::decodeFile(inPath, oa::ImageFormat::Rgba);
	if (not imageR.isOk()) {
		std::fprintf(stderr, "oa::FnImage::decodeFile(%s) failed: %s\n",
			inPath, imageR.getStatus().toString().cStr());
		return 1;
	}
	auto loadR = oa::FnTexture::fromImage(engine, *imageR);
	if (not loadR.isOk()) {
		std::fprintf(stderr, "oa::FnTexture::fromImage failed: %s\n",
			loadR.getStatus().toString().cStr());
		return 1;
	}
	oa::Texture tex = oa::move(*loadR);

	// ─── Sink: saveImage (architecture/oaArchitecture.md §10) ───────────────────────────────────
	// readback + encode + filesystem write. No swapchain involved.
	if (auto s = oa::FnImage::saveTextureFile(engine, tex, outPath); not s.isOk()) {
		std::fprintf(stderr, "oa::FnImage::saveTextureFile(%s) failed: %s\n",
			outPath, s.toString().cStr());
		return 1;
	}

	std::printf("OK: %dx%d  %s → %s\n", tex.width(), tex.height(), inPath, outPath);

	// ─── texture operation smoke ─────────────────────────────────────────────
	// Both record through the texture domain API (no immediate dispatch), then
	// submit and wait explicitly at the file-output boundary. Produces
	// two extra PNGs:
	//
	//   /tmp/oa_viewport_batch_blit.png   — clone of the source via Blit
	//                                       (round-trip: should byte-equal the
	//                                       original outPath save above)
	//   /tmp/oa_viewport_batch_clear.png  — solid red via clear
	std::vector<oa::U8> zeros(static_cast<size_t>(tex.width()) *
	                        static_cast<size_t>(tex.height()) * 4U, 0);
	auto cloneR = oa::FnTexture::fromPixels(
		engine, oa::Span<const oa::U8>(zeros.data(), zeros.size()),
		tex.width(), tex.height());
	if (not cloneR.isOk()) {
		std::fprintf(stderr, "oa::FnTexture::fromPixels (clone target) failed: %s\n",
			cloneR.getStatus().toString().cStr());
		return 1;
	}
	oa::Texture clone = cloneR.getValue();

	oa::BlitDesc desc;
	desc.src = &tex;
	desc.dst = &clone;
	if (auto status = oa::FnTexture::blit(desc); not status.isOk()) {
		std::fprintf(stderr, "oa::FnTexture::blit failed: %s\n",
			status.toString().cStr());
		return 1;
	}
	auto blitSubmitted = engine.submit();
	if (not blitSubmitted.isOk()
		or not engine.wait(blitSubmitted.getValue()).isOk()) {
		std::fprintf(stderr, "oa::FnTexture::blit submission failed\n");
		return 1;
	}

	if (auto s = oa::FnImage::saveTextureFile(engine, clone, "/tmp/oa_viewport_batch_blit.png");
		not s.isOk()) {
		std::fprintf(stderr, "saveFile(blit) failed: %s\n", s.toString().cStr());
	} else {
		std::printf("OK: Blit → /tmp/oa_viewport_batch_blit.png\n");
	}

	// Solid red, full opacity.
	if (auto status = oa::FnTexture::clear(
			clone, oa::ClearColor{0.95F, 0.10F, 0.10F, 1.0F});
		not status.isOk()) {
		std::fprintf(stderr, "oa::FnTexture::clear failed: %s\n",
			status.toString().cStr());
		return 1;
	}
	auto clearSubmitted = engine.submit();
	if (not clearSubmitted.isOk()
		or not engine.wait(clearSubmitted.getValue()).isOk()) {
		std::fprintf(stderr, "oa::FnTexture::clear submission failed\n");
		return 1;
	}

	if (auto s = oa::FnImage::saveTextureFile(engine, clone, "/tmp/oa_viewport_batch_clear.png");
		not s.isOk()) {
		std::fprintf(stderr, "saveFile(clear) failed: %s\n", s.toString().cStr());
	} else {
		std::printf("OK: clear(red) → /tmp/oa_viewport_batch_clear.png\n");
	}

	return 0;
}
