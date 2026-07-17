// ═══════════════════════════════════════════════════════════════════════════
// OA Tutorial: OaViewer — Headless image sink
// Level 0 API — Headless engine + OaFnImage::SaveFile
// ═══════════════════════════════════════════════════════════════════════════
//
// The renderer/sink split in code (Architecture/OaArchitecture.md §10).
//
// Same renderer (here: a still-image load — the simplest possible producer),
// no window, no swapchain, no event loop. The renderer's output is an
// OaTexture; the sink decides what happens to it. In GUI mode (the existing
// TutorialViewerImage) the sink is Present(swapchain, target). Here the sink is
// SaveFile(target, path). Identical producer, different terminal node.
//
// This is the Maya-style batch / render-farm / CI worker shape: one binary
// can ship in every environment that has a GPU, regardless of whether a
// display server / surface is reachable.
//
// Usage:
//   ./TutorialViewerImageHeadless [input.jpg] [output.png]
//
// Defaults: Asset/Image/Realm1024px.jpg → /tmp/oa_viewport_batch.png
// ═══════════════════════════════════════════════════════════════════════════

#include <Oa/Runtime/Engine.h>
#include <Oa/Runtime/Context.h>
#include <Oa/Ui/Image.h>
#include <Oa/Vision/FnImage.h>
#include <Oa/Core/Log.h>

#include <cstdio>
#include <vector>


int main(int argc, char** argv) {
	const char* inPath  = (argc > 1) ? argv[1] : "Asset/Image/Realm1024px.jpg";
	const char* outPath = (argc > 2) ? argv[2] : "/tmp/oa_viewport_batch.png";

	// ─── Engine bring-up ─────────────────────────────────────────────────────
	// PresentationMode::None — pure compute. We never instantiate a surface,
	// never request VK_KHR_swapchain, and the runtime is free to pick any
	// device (discrete, integrated, or even CPU/llvmpipe fallback). This is
	// exactly the engine config a render-farm worker or CI box would use.
	OaEngineConfig cfg;
	cfg.PresentationMode = OaPresentationMode::None;
	cfg.RegisterAsGlobal = true;

	auto engineResult = OaComputeEngine::Create(cfg);
	if (not engineResult.IsOk()) {
		std::fprintf(stderr, "OaComputeEngine::Create failed: %s\n",
			engineResult.GetStatus().ToString().c_str());
		return 1;
	}
	OaComputeEngine& engine = *engineResult.GetValue();

	OA_LOG_INFO(OaLogComponent::App,
		"TutorialViewerImageHeadless: %s → %s", inPath, outPath);

	// ─── Producer: load the input image into an OaTexture (CPU decode + GPU upload) ─
	auto loadR = OaTexture::LoadFile(engine, inPath);
	if (not loadR.IsOk()) {
		std::fprintf(stderr, "OaTexture::LoadFile(%s) failed: %s\n",
			inPath, loadR.GetStatus().ToString().c_str());
		return 1;
	}
	OaTexture tex = loadR.GetValue();

	// ─── Sink: SaveImage (Architecture/OaArchitecture.md §10) ───────────────────────────────────
	// Readback + encode + filesystem write. No swapchain involved.
	if (auto s = OaFnImage::SaveFile(engine, tex, outPath); not s.IsOk()) {
		std::fprintf(stderr, "OaFnImage::SaveFile(%s) failed: %s\n",
			outPath, s.ToString().c_str());
		tex.Destroy(engine);
		return 1;
	}

	std::printf("OK: %dx%d  %s → %s\n", tex.Width, tex.Height, inPath, outPath);

	// ─── Step 3b.2 smoke: RecordBlit + RecordClear through OaContext ─────────
	// Both go through the unified context graph (no immediate dispatch) and
	// execute alongside any pending compute work in one ctx.Execute(). Produces
	// two extra PNGs:
	//
	//   /tmp/oa_viewport_batch_blit.png   — clone of the source via RecordBlit
	//                                       (round-trip: should byte-equal the
	//                                       original outPath save above)
	//   /tmp/oa_viewport_batch_clear.png  — solid red via RecordClear
	std::vector<OaU8> zeros(static_cast<size_t>(tex.Width) *
	                        static_cast<size_t>(tex.Height) * 4U, 0);
	auto cloneR = OaTexture::FromPixels(engine, OaSpan<const OaU8>(zeros.data(), zeros.size()),
	                                    tex.Width, tex.Height);
	if (not cloneR.IsOk()) {
		std::fprintf(stderr, "OaTexture::FromPixels (clone target) failed: %s\n",
			cloneR.GetStatus().ToString().c_str());
		tex.Destroy(engine);
		return 1;
	}
	OaTexture clone = cloneR.GetValue();

	auto& ctx = OaContext::GetDefault();
	{
		OaContext::Scope scope(ctx);
		OaBlitDesc desc;
		desc.Src = &tex;
		desc.Dst = &clone;
		ctx.RecordBlit(desc);
	}  // scope dtor: ctx.Execute() + ctx.Sync()

	if (auto s = OaFnImage::SaveFile(engine, clone, "/tmp/oa_viewport_batch_blit.png");
		not s.IsOk()) {
		std::fprintf(stderr, "SaveFile(blit) failed: %s\n", s.ToString().c_str());
	} else {
		std::printf("OK: RecordBlit → /tmp/oa_viewport_batch_blit.png\n");
	}

	{
		OaContext::Scope scope(ctx);
		// Solid red, full opacity.
		ctx.RecordClear(clone, OaClearColor{ 0.95F, 0.10F, 0.10F, 1.0F });
	}

	if (auto s = OaFnImage::SaveFile(engine, clone, "/tmp/oa_viewport_batch_clear.png");
		not s.IsOk()) {
		std::fprintf(stderr, "SaveFile(clear) failed: %s\n", s.ToString().c_str());
	} else {
		std::printf("OK: RecordClear(red) → /tmp/oa_viewport_batch_clear.png\n");
	}

	clone.Destroy(engine);
	tex.Destroy(engine);
	return 0;
}
