// OaPlot::Figure — implementation. Layout + replay over OaContext sinks.
//
// UnifiedExecutionArchitecture.md §3.5 / OaUiFinalGlueBridge.md Step 3e. The Show() path drives
// an internal OaDeviceUiApp that, in OnRender, iterates every Axes and
// records its commands through OaContext (Imshow = ctx.RecordBlit) or OaUi
// (Title / Caption / PlotLine). The SaveFig() path does the same layout on
// the CPU — readback each Imshow source via the staging-buffer pattern,
// composite into an in-memory RGBA8 framebuffer, write PNG.

// Engine first — VK_NO_PROTOTYPES before any vulkan.h pull-in.
#include <Oa/Runtime/Engine.h>
#include <Oa/Runtime/Context.h>

#include <Oa/Ui/Plot/Figure.h>

#include <Oa/Core/Log.h>
#include <Oa/Ui/DeviceUi.h>
#include <Oa/Ui/Ui.h>

#include "../../../../ThirdParty/stb/stb_image_write.h"

#include <cstdio>
#include <cstring>


namespace OaPlot {

// ─── Figure::Impl ──────────────────────────────────────────────────────────

struct Figure::Impl {
	OaVec<Axes> Axes_;
	OaString    Title_;
	OaString    XLabel_;
	OaString    YLabel_;

	// Rasterized canvas — fixed size = Config_.Width × Config_.Height.
	// Built by Rasterize() in two passes:
	//   1. CPU paint into Framebuffer_ (background fill + per-axes Imshow tiles)
	//   2. GPU upload as Canvas_ via OaTexture::FromPixels
	// After Rasterize(), Show() displays Canvas_ as a single textured panel
	// letterboxed in the window — cell proportions never depend on window size.
	OaVec<OaU8>          Framebuffer_;
	OaTexture            Canvas_;
	OaComputeEngine*   CanvasRt_ = nullptr;  // engine that owns Canvas_, for ~Figure
};


// ─── ctor / dtor / move ────────────────────────────────────────────────────

Figure::Figure(const FigureConfig& InConfig)
	: Impl_(OaMakeUniquePtr<Impl>())
	, Config_(InConfig)
{
	if (Config_.Rows < 1) Config_.Rows = 1;
	if (Config_.Cols < 1) Config_.Cols = 1;
	Impl_->Axes_.Resize(Config_.Rows * Config_.Cols);
}

Figure::~Figure() = default;
Figure::Figure(Figure&&) noexcept = default;
Figure& Figure::operator=(Figure&&) noexcept = default;


// ─── Layout ────────────────────────────────────────────────────────────────

Axes& Figure::Ax(OaI32 InRow, OaI32 InCol) {
	OaI32 r = InRow < 0 ? 0 : (InRow >= Config_.Rows ? Config_.Rows - 1 : InRow);
	OaI32 c = InCol < 0 ? 0 : (InCol >= Config_.Cols ? Config_.Cols - 1 : InCol);
	return Impl_->Axes_[static_cast<OaUsize>(r) * Config_.Cols + c];
}

void Figure::Title(const char* InText)  { Impl_->Title_  = InText ? InText : ""; }
void Figure::XLabel(const char* InText) { Impl_->XLabel_ = InText ? InText : ""; }
void Figure::YLabel(const char* InText) { Impl_->YLabel_ = InText ? InText : ""; }

Figure::Rect Figure::CellRect(OaI32 InRow, OaI32 InCol,
                              OaU32 InW, OaU32 InH) const noexcept {
	// Letterbox the figure inside (InW, InH) to preserve Config_.Width:Height
	// aspect. Matches matplotlib's "figure has a fixed canvas; window shows
	// it centred" behaviour — cells grow uniformly with the window without
	// the inter-cell gaps shrinking in proportion.
	const OaF32 figAspect = static_cast<OaF32>(Config_.Width) /
	                        static_cast<OaF32>(Config_.Height);
	const OaF32 winAspect = static_cast<OaF32>(InW) / static_cast<OaF32>(InH);
	OaI32 figW;
	OaI32 figH;
	if (winAspect > figAspect) {
		// Window wider than figure → bars on left/right.
		figH = static_cast<OaI32>(InH);
		figW = static_cast<OaI32>(static_cast<OaF32>(InH) * figAspect);
	} else {
		// Window taller than figure → bars on top/bottom.
		figW = static_cast<OaI32>(InW);
		figH = static_cast<OaI32>(static_cast<OaF32>(InW) / figAspect);
	}
	const OaI32 figX = (static_cast<OaI32>(InW) - figW) / 2;
	const OaI32 figY = (static_cast<OaI32>(InH) - figH) / 2;

	// Pad / spacing scale with the figure render size so the proportions
	// the user configured are preserved across resizes.
	const OaF32 scale = static_cast<OaF32>(figW) / static_cast<OaF32>(Config_.Width);
	const OaI32 pad   = static_cast<OaI32>(static_cast<OaF32>(Config_.Padding)  * scale);
	const OaI32 hgap  = static_cast<OaI32>(static_cast<OaF32>(Config_.HSpacing) * scale);
	const OaI32 vgap  = static_cast<OaI32>(static_cast<OaF32>(Config_.VSpacing) * scale);
	const OaI32 gridW = figW - 2 * pad;
	const OaI32 gridH = figH - 2 * pad;
	const OaI32 cellW = (gridW - (Config_.Cols - 1) * hgap) / Config_.Cols;
	const OaI32 cellH = (gridH - (Config_.Rows - 1) * vgap) / Config_.Rows;
	return {
		.X = figX + pad + InCol * (cellW + hgap),
		.Y = figY + pad + InRow * (cellH + vgap),
		.W = cellW,
		.H = cellH,
	};
}


// ─── Layout sub-rects within a cell (title / image / caption) ─────────────

static constexpr OaI32 kTitleH   = 20;
static constexpr OaI32 kCaptionH = 18;
static constexpr OaI32 kTitleGap = 4;


// ─── Canvas accessor ──────────────────────────────────────────────────────

const OaTexture& Figure::Canvas() const noexcept {
	return Impl_->Canvas_;
}


// ─── RenderFrame — display the rasterized canvas as a textured quad ──────
//
// Step 3e v2: the figure is a frozen 2D image at fixed Config_.Width × Height.
// RenderFrame just letterboxes Canvas_ inside the viewport and draws it as
// one OaUi::Image (BlitRgba.slang compute → compose) — window resize only
// scales the quad, never the cells. Call Rasterize() before Show() to
// populate Canvas_.
//
// Legacy fallback (Canvas_ not yet built): per-frame Axes replay through
// OaUi widgets. Cell proportions follow the window in this path.

void Figure::RenderFrame(::OaDeviceUi& InGpui, ::OaUi& InOui) {
	const OaU32 W = InGpui.Width();
	const OaU32 H = InGpui.Height();
	if (W == 0U or H == 0U) { return; }

	// ─── Rasterized path: display Canvas_ as one letterboxed panel ─────
	if (Impl_->Canvas_.IsValid()) {
		// Letterbox the canvas inside the viewport, preserving its aspect.
		const OaF32 canvasAspect = static_cast<OaF32>(Impl_->Canvas_.Width) /
		                           static_cast<OaF32>(Impl_->Canvas_.Height);
		const OaF32 winAspect    = static_cast<OaF32>(W) / static_cast<OaF32>(H);
		OaI32 dW;
		OaI32 dH;
		if (winAspect > canvasAspect) {
			dH = static_cast<OaI32>(H);
			dW = static_cast<OaI32>(static_cast<OaF32>(H) * canvasAspect);
		} else {
			dW = static_cast<OaI32>(W);
			dH = static_cast<OaI32>(static_cast<OaF32>(W) / canvasAspect);
		}
		const OaI32 dX = (static_cast<OaI32>(W) - dW) / 2;
		const OaI32 dY = (static_cast<OaI32>(H) - dH) / 2;

		InOui.BeginPanel("canvas", {dX, dY, dW, dH});
		InOui.Image(Impl_->Canvas_.BindlessIndex(),
		            Impl_->Canvas_.Width, Impl_->Canvas_.Height);
		InOui.EndPanel();
		return;
	}

	// ─── Legacy fallback: per-frame OaUi widget layout ─────────────────
	for (OaI32 r = 0; r < Rows(); ++r) {
		for (OaI32 c = 0; c < Cols(); ++c) {
			const Rect cell = CellRect(r, c, W, H);
			const Axes& ax = Ax(r, c);

			const OaI32 hasTitle   = ax.Title_.Present   ? 1 : 0;
			const OaI32 hasCaption = ax.Caption_.Present ? 1 : 0;
			const OaI32 titleH     = hasTitle   * (kTitleH + kTitleGap);
			const OaI32 captionH   = hasCaption * (kCaptionH + kTitleGap);
			const OaI32 imgRegionH = cell.H - titleH - captionH;
			if (imgRegionH <= 0) { continue; }

			char cellId[32];
			std::snprintf(cellId, sizeof(cellId), "ax_%d_%d", r, c);
			InOui.BeginPanel(cellId, {cell.X, cell.Y, cell.W, cell.H});

			if (hasTitle != 0) {
				InOui.PushStyle({.Text = ax.Title_.Color});
				InOui.Label(ax.Title_.Text.c_str());
				InOui.PopStyle();
			}

			// ── Imshow via OaUi::Image (compute-shader path) ─────────────
			// OaUi panels expect ABSOLUTE compose coordinates (the panel stack
			// doesn't accumulate — see Ui.cpp::BeginPanel + Image), so the
			// image rect is computed in compose space, not cell-local.
			if (ax.Image_.Present and ax.Image_.Tex.IsValid()) {
				const OaF32 a = static_cast<OaF32>(ax.Image_.Tex.Width) /
				                static_cast<OaF32>(ax.Image_.Tex.Height);
				OaI32 dW = cell.W;
				OaI32 dH = static_cast<OaI32>(static_cast<OaF32>(cell.W) / a);
				if (dH > imgRegionH) {
					dH = imgRegionH;
					dW = static_cast<OaI32>(static_cast<OaF32>(imgRegionH) * a);
				}
				const OaI32 dX = cell.X + (cell.W - dW) / 2;
				const OaI32 dY = cell.Y + titleH + (imgRegionH - dH) / 2;

				char imgId[32];
				std::snprintf(imgId, sizeof(imgId), "im_%d_%d", r, c);
				InOui.BeginPanel(imgId, {dX, dY, dW, dH});
				// Pass the SOURCE texture dimensions — BlitRgba.slang uses
				// these to index into the source buffer. Passing dW/dH would
				// make the shader read past the buffer extent (the artifacts
				// at the top of each cell).
				InOui.Image(ax.Image_.Tex.BindlessIndex(),
				            ax.Image_.Tex.Width, ax.Image_.Tex.Height);
				InOui.EndPanel();
			} else if (imgRegionH > 0) {
				InOui.Spacing(static_cast<OaF32>(imgRegionH));
			}

			if (hasCaption != 0) {
				InOui.PushStyle({.Text = ax.Caption_.Color});
				InOui.Label(ax.Caption_.Text.c_str());
				InOui.PopStyle();
			}

			// Line / Bar / Scatter — fallback rendering when no Imshow.
			if (not ax.Image_.Present) {
				if (ax.Line_.Present and not ax.Line_.Y.Empty()) {
					InOui.PlotLine(ax.Title_.Text.c_str(),
					               ax.Line_.Y.Data(),
					               static_cast<OaI32>(ax.Line_.Y.Size()));
				}
				if (ax.Bar_.Present and not ax.Bar_.V.Empty()) {
					InOui.PlotHistogram(ax.Title_.Text.c_str(),
					                    ax.Bar_.V.Data(),
					                    static_cast<OaI32>(ax.Bar_.V.Size()));
				}
				if (ax.Scatter_.Present and not ax.Scatter_.Ys.Empty()) {
					InOui.PlotLine(ax.Title_.Text.c_str(),
					               ax.Scatter_.Ys.Data(),
					               static_cast<OaI32>(ax.Scatter_.Ys.Size()));
				}
			}

			InOui.EndPanel();
		}
	}
}


// ─── Show() — internal OaDeviceUiApp ───────────────────────────────────────

namespace {

class FigureShowApp : public OaDeviceUiApp {
public:
	Figure* Fig = nullptr;

	void OnInit(OaDeviceUi& InGpui) override {
		auto& input = InGpui.Input();
		input.RegisterAction({.Name = "quit",  .Binding = {.Key = OuiKey::Escape}, .Callback = [this] { Quit(); }});
		input.RegisterAction({.Name = "quitq", .Binding = {.Key = OuiKey::Q},      .Callback = [this] { Quit(); }});
	}

	void OnRender(OaUi& InOui) override {
		if (Fig != nullptr) { Fig->RenderFrame(Gpui(), InOui); }
	}

	void OnShutdown(OaDeviceUi& /*InGpui*/) override {}
};

}  // namespace


OaStatus Figure::Show() {
	FigureShowApp app;
	app.Fig = this;

	OaUiStyle style;
	style.Background = Config_.Background;

	const OaString& title = Impl_->Title_.Empty() ? Config_.Title : Impl_->Title_;

	return app.Run({
		.Title  = title,
		.Width  = Config_.Width,
		.Height = Config_.Height,
		.Style  = style,
	});
}


// ─── SaveFig() — CPU-side composition ──────────────────────────────────────
//
// Phase-1 limitation: image-only. Text overlays (Title / Caption / XLabel /
// YLabel) and Plot/Bar/Scatter widgets are skipped because they require the
// ImGui pipeline which only runs on a swapchain context today. The PNG
// shows the image grid with the figure background colour fill — sufficient
// to verify layout + Imshow plumbing. Text rasterization lands as Phase 2.

namespace {

// Read a buffer-backed OaTexture into a host vector via staging. The texture
// must have a valid DeviceBuf (the FromPixels / LoadFile path).
static OaStatus ReadbackTexture(OaComputeEngine& InEngine,
                                const OaTexture& InTex,
                                OaVec<OaU8>& OutBytes) {
	if (not InTex.IsValid() or InTex.DeviceBuf.Buffer == nullptr) {
		return OaStatus::Error("ReadbackTexture: not buffer-backed");
	}
	const OaU64 bytes = static_cast<OaU64>(InTex.Width) *
	                    static_cast<OaU64>(InTex.Height) * 4U;
	if (InTex.DeviceBuf.Size < bytes) {
		return OaStatus::Error("ReadbackTexture: device buffer too small");
	}

	auto stageR = InEngine.Allocator.AllocHostVisible(bytes);
	if (not stageR.IsOk()) return stageR.GetStatus();
	OaVkBuffer staging = stageR.GetValue();

	if (auto s = InEngine.CopyBufferAsync(InTex.DeviceBuf, staging, bytes); not s.IsOk()) {
		InEngine.Allocator.Free(staging);
		return s;
	}
	if (auto s = InEngine.WaitTransfer(); not s.IsOk()) {
		InEngine.Allocator.Free(staging);
		return s;
	}
	if (staging.MappedPtr == nullptr) {
		InEngine.Allocator.Free(staging);
		return OaStatus::Error("ReadbackTexture: null MappedPtr");
	}

	OutBytes.Resize(static_cast<OaI64>(bytes));
	std::memcpy(OutBytes.Data(), staging.MappedPtr, bytes);
	InEngine.Allocator.Free(staging);
	return OaStatus::Ok();
}

// Nearest-neighbour blit srcW×srcH RGBA8 → dstW×dstH region inside fbW×fbH RGBA8.
static void BlitNearest(const OaU8* InSrc, OaI32 InSrcW, OaI32 InSrcH,
                        OaU8* InFb, OaI32 InFbW, OaI32 InFbH,
                        OaI32 InDstX, OaI32 InDstY, OaI32 InDstW, OaI32 InDstH) {
	for (OaI32 y = 0; y < InDstH; ++y) {
		const OaI32 fy = InDstY + y;
		if (fy < 0 or fy >= InFbH) { continue; }
		const OaI32 sy = (y * InSrcH) / InDstH;
		for (OaI32 x = 0; x < InDstW; ++x) {
			const OaI32 fx = InDstX + x;
			if (fx < 0 or fx >= InFbW) { continue; }
			const OaI32 sx = (x * InSrcW) / InDstW;
			const OaI64 srcIdx = (static_cast<OaI64>(sy) * InSrcW + sx) * 4;
			const OaI64 fbIdx  = (static_cast<OaI64>(fy) * InFbW + fx) * 4;
			InFb[fbIdx + 0] = InSrc[srcIdx + 0];
			InFb[fbIdx + 1] = InSrc[srcIdx + 1];
			InFb[fbIdx + 2] = InSrc[srcIdx + 2];
			InFb[fbIdx + 3] = InSrc[srcIdx + 3];
		}
	}
}

static OaU8 ColorByteFromF32(OaF32 InV) {
	if (InV <= 0.0F) return 0U;
	if (InV >= 1.0F) return 255U;
	return static_cast<OaU8>(InV * 255.0F + 0.5F);
}

}  // namespace


// ─── CompositeFramebuffer — shared CPU paint (member, friend-of-Axes) ────
//
// Builds Impl_->Framebuffer_ (Config_.Width × Height RGBA8) from the
// configured background + each Axes' Imshow tile (letterboxed into its
// cell rect at fixed canvas dimensions). Tile pixels are read back from
// the GPU via the staging-buffer pattern. Used by both Rasterize (which
// then uploads to Canvas_) and SaveFig (which writes PNG directly).

OaStatus Figure::CompositeFramebuffer(OaComputeEngine& InRt) {
	const OaI32 W = static_cast<OaI32>(Config_.Width);
	const OaI32 H = static_cast<OaI32>(Config_.Height);
	const OaU64 fbBytes = static_cast<OaU64>(W) * static_cast<OaU64>(H) * 4U;
	OaVec<OaU8>& fb = Impl_->Framebuffer_;
	fb.Resize(static_cast<OaI64>(fbBytes));

	// Background fill.
	const OaU8 bgR = ColorByteFromF32(Config_.Background.R);
	const OaU8 bgG = ColorByteFromF32(Config_.Background.G);
	const OaU8 bgB = ColorByteFromF32(Config_.Background.B);
	const OaU8 bgA = ColorByteFromF32(Config_.Background.A);
	for (OaI64 i = 0; i < static_cast<OaI64>(W) * H; ++i) {
		fb[i * 4 + 0] = bgR;
		fb[i * 4 + 1] = bgG;
		fb[i * 4 + 2] = bgB;
		fb[i * 4 + 3] = bgA;
	}

	// Pre-calculate a fixed image size for all cells to ensure consistency
	// All cells use the same cell dimensions (first cell), and we assume worst-case
	// title+caption height to ensure consistent sizing regardless of actual content
	OaI32 fixedImgSize = 0;
	bool hasAnyImage = false;
	if (Config_.Rows > 0 and Config_.Cols > 0) {
		const Rect firstCell = CellRect(0, 0, Config_.Width, Config_.Height);
		const OaI32 maxTitleH   = (kTitleH + kTitleGap);
		const OaI32 maxCaptionH = (kCaptionH + kTitleGap);
		const OaI32 imgRegionH = firstCell.H - maxTitleH - maxCaptionH;
		if (imgRegionH > 0) {
			fixedImgSize = (firstCell.W < imgRegionH) ? firstCell.W : imgRegionH;
			hasAnyImage = true;
		}
	}

	OaVec<OaU8> tileBytes;
	for (OaI32 r = 0; r < Config_.Rows; ++r) {
		for (OaI32 c = 0; c < Config_.Cols; ++c) {
			const Axes& ax = Ax(r, c);
			if (not ax.Image_.Present or not ax.Image_.Tex.IsValid()) { continue; }

			// Cell rect inside the canvas (canvas is its own coord system =
			// the configured Width × Height, so CellRect against those).
			const Rect cell = CellRect(r, c, Config_.Width, Config_.Height);
			const OaI32 hasTitle   = ax.Title_.Present   ? 1 : 0;
			const OaI32 hasCaption = ax.Caption_.Present ? 1 : 0;
			const OaI32 titleH     = hasTitle   * (kTitleH + kTitleGap);
			const OaI32 captionH   = hasCaption * (kCaptionH + kTitleGap);
			const OaI32 imgRegionY = cell.Y + titleH;
			const OaI32 imgRegionH = cell.H - titleH - captionH;
			if (imgRegionH <= 0) { continue; }

			// Use fixed size for square images, otherwise calculate per-cell
			OaI32 dW, dH;
			if (ax.Image_.Tex.Width == ax.Image_.Tex.Height and hasAnyImage) {
				dW = fixedImgSize;
				dH = fixedImgSize;
			} else {
				const OaF32 a = static_cast<OaF32>(ax.Image_.Tex.Width) /
				                static_cast<OaF32>(ax.Image_.Tex.Height);
				dW = cell.W;
				dH = static_cast<OaI32>(static_cast<OaF32>(cell.W) / a);
				if (dH > imgRegionH) {
					dH = imgRegionH;
					dW = static_cast<OaI32>(static_cast<OaF32>(imgRegionH) * a);
				}
			}
			const OaI32 dX = cell.X + (cell.W - dW) / 2;
			const OaI32 dY = imgRegionY + (imgRegionH - dH) / 2;

			if (auto s = ReadbackTexture(InRt, ax.Image_.Tex, tileBytes); not s.IsOk()) {
				return s;
			}
			BlitNearest(tileBytes.Data(), ax.Image_.Tex.Width, ax.Image_.Tex.Height,
			            fb.Data(), W, H, dX, dY, dW, dH);
		}
	}
	return OaStatus::Ok();
}


// ─── Rasterize — CPU paint + GPU upload ───────────────────────────────────

void Figure::Rasterize(OaComputeEngine& InRt) {
	if (auto s = CompositeFramebuffer(InRt); not s.IsOk()) {
		OA_LOG_ERROR(OaLogComponent::App,
			"OaPlot::Figure::Rasterize: composite failed: %s", s.ToString().c_str());
		return;
	}
	if (Impl_->Canvas_.IsValid()) {
		Impl_->Canvas_.Destroy(InRt);
	}
	auto r = OaTexture::FromPixels(InRt,
		OaSpan<const OaU8>(Impl_->Framebuffer_.Data(), Impl_->Framebuffer_.Size()),
		static_cast<OaI32>(Config_.Width),
		static_cast<OaI32>(Config_.Height));
	if (not r.IsOk()) {
		OA_LOG_ERROR(OaLogComponent::App,
			"OaPlot::Figure::Rasterize: upload failed: %s", r.GetStatus().ToString().c_str());
		return;
	}
	Impl_->Canvas_   = r.GetValue();
	Impl_->CanvasRt_ = &InRt;
	OA_LOG_INFO(OaLogComponent::App,
		"OaPlot::Figure::Rasterize: canvas %ux%u uploaded (bindless=%u)",
		Config_.Width, Config_.Height, Impl_->Canvas_.BindlessIndex());
}


OaStatus Figure::SaveFig(const char* InPath) {
	if (InPath == nullptr or InPath[0] == '\0') {
		return OaStatus::Error("OaPlot::Figure::SaveFig: null/empty path");
	}
	auto* engine = OaComputeEngine::GetGlobal();
	if (engine == nullptr) {
		return OaStatus::Error("OaPlot::Figure::SaveFig: no global engine");
	}

	const OaI32 W = static_cast<OaI32>(Config_.Width);
	const OaI32 H = static_cast<OaI32>(Config_.Height);
	if (auto s = CompositeFramebuffer(*engine); not s.IsOk()) { return s; }

	if (stbi_write_png(InPath, W, H, 4, Impl_->Framebuffer_.Data(), W * 4) == 0) {
		return OaStatus::Error("OaPlot::Figure::SaveFig: stbi_write_png failed");
	}
	OA_LOG_INFO(OaLogComponent::App,
		"OaPlot::Figure::SaveFig: %dx%d → %s", W, H, InPath);
	return OaStatus::Ok();
}

}  // namespace OaPlot
