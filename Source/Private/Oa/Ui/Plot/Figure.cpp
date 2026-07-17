// OaPlot::Figure — implementation. Layout + replay over OaContext sinks.
//
// Architecture/OaArchitecture.md §10. The Show() path drives
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
#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>


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
			} else if (ax.Heatmap_.Present && !ax.Heatmap_.V.Empty()) {
				OaF32 vMin = ax.Heatmap_.Style.VMin;
				OaF32 vMax = ax.Heatmap_.Style.VMax;
				if (ax.Heatmap_.Style.AutoScale) {
					vMin = std::numeric_limits<OaF32>::infinity();
					vMax = -std::numeric_limits<OaF32>::infinity();
					for (const OaF32 value : ax.Heatmap_.V) {
						if (!std::isfinite(value)) continue;
						vMin = std::min(vMin, value);
						vMax = std::max(vMax, value);
					}
				}
				char plotId[32];
				std::snprintf(plotId, sizeof(plotId), "heat_%d_%d", r, c);
				InOui.BeginPanel(plotId,
					{cell.X, cell.Y + titleH, cell.W, imgRegionH});
				InOui.Heatmap(plotId, ax.Heatmap_.V.Data(), ax.Heatmap_.Rows,
					ax.Heatmap_.Cols, {
						.VMin = vMin,
						.VMax = vMax,
						.Colormap = ax.Heatmap_.Style.Colormap,
						.ShowGrid = ax.Heatmap_.Style.ShowGrid,
					});
				InOui.EndPanel();
			} else if (ax.Line_.Present && !ax.Line_.Y.Empty()) {
				char plotId[32];
				std::snprintf(plotId, sizeof(plotId), "line_%d_%d", r, c);
				InOui.BeginPanel(plotId,
					{cell.X, cell.Y + titleH, cell.W, imgRegionH});
				InOui.PlotLine(plotId, ax.Line_.Y.Data(),
					static_cast<OaI32>(ax.Line_.Y.Size()), {
						.Color = ax.Line_.Style.Color,
					});
				InOui.EndPanel();
			} else if (imgRegionH > 0) {
				InOui.Spacing(static_cast<OaF32>(imgRegionH));
			}

			if (hasCaption != 0) {
				InOui.PushStyle({.Text = ax.Caption_.Color});
				InOui.Label(ax.Caption_.Text.c_str());
				InOui.PopStyle();
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
// Images, curves and heatmaps are painted into the fixed-size framebuffer.
// Text overlays remain interactive-only until the SDF atlas gains a CPU/headless
// composition path.

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

static void PutPixel(OaU8* InFb, OaI32 InW, OaI32 InH,
	OaI32 InX, OaI32 InY, OaColor InColor) {
	if (InX < 0 || InY < 0 || InX >= InW || InY >= InH) return;
	auto* pixel = InFb + (static_cast<OaI64>(InY) * InW + InX) * 4;
	const OaF32 alpha = std::clamp(InColor.A, 0.0F, 1.0F);
	const auto blend = [alpha](OaU8 InDst, OaF32 InSrc) {
		return ColorByteFromF32(std::clamp(InSrc, 0.0F, 1.0F) * alpha
			+ static_cast<OaF32>(InDst) / 255.0F * (1.0F - alpha));
	};
	pixel[0] = blend(pixel[0], InColor.R);
	pixel[1] = blend(pixel[1], InColor.G);
	pixel[2] = blend(pixel[2], InColor.B);
	pixel[3] = 255U;
}

static OaColor HeatColor(OaF32 InT, OaU32 InColormap) {
	const OaF32 t = std::clamp(InT, 0.0F, 1.0F);
	if (InColormap == 3U) return {t, t, t, 1.0F};
	if (InColormap == 2U) {
		if (t < 0.5F) {
			const OaF32 u = t * 2.0F;
			return {
				0.230F + (0.865F - 0.230F) * u,
				0.299F + (0.865F - 0.299F) * u,
				0.754F + (0.865F - 0.754F) * u, 1.0F};
		}
		const OaF32 u = (t - 0.5F) * 2.0F;
		return {
			0.865F + (0.706F - 0.865F) * u,
			0.865F + (0.016F - 0.865F) * u,
			0.865F + (0.150F - 0.865F) * u, 1.0F};
	}
	if (InColormap == 0U) {
		if (t < 0.55F) {
			const OaF32 u = t / 0.55F;
			return {
				0.050F + (0.798F - 0.050F) * u,
				0.030F + (0.280F - 0.030F) * u,
				0.528F + (0.470F - 0.528F) * u, 1.0F};
		}
		const OaF32 u = (t - 0.55F) / 0.45F;
		return {
			0.798F + (0.940F - 0.798F) * u,
			0.280F + (0.975F - 0.280F) * u,
			0.470F + (0.131F - 0.470F) * u, 1.0F};
	}
	// Small viridis anchor interpolation for deterministic headless figures.
	if (t < 0.5F) {
		const OaF32 u = t * 2.0F;
		return {
			0.267F + (0.128F - 0.267F) * u,
			0.005F + (0.567F - 0.005F) * u,
			0.329F + (0.551F - 0.329F) * u, 1.0F};
	}
	const OaF32 u = (t - 0.5F) * 2.0F;
	return {
		0.128F + (0.993F - 0.128F) * u,
		0.567F + (0.906F - 0.567F) * u,
		0.551F + (0.144F - 0.551F) * u, 1.0F};
}

static void PaintHeatmap(OaU8* InFb, OaI32 InFbW, OaI32 InFbH,
	OaI32 InX, OaI32 InY, OaI32 InW, OaI32 InH,
	const OaF32* InValues, OaI32 InRows, OaI32 InCols,
	OaF32 InMin, OaF32 InMax, OaU32 InColormap, bool InGrid) {
	const OaF32 range = std::max(std::abs(InMax - InMin), 1.0e-12F);
	for (OaI32 y = 0; y < InH; ++y) {
		const OaI32 row = std::min(InRows - 1, y * InRows / InH);
		for (OaI32 x = 0; x < InW; ++x) {
			const OaI32 col = std::min(InCols - 1, x * InCols / InW);
			OaColor color = HeatColor(
				(InValues[row * InCols + col] - InMin) / range, InColormap);
			if (InGrid && (((x + 1) * InCols / InW) != col
				|| ((y + 1) * InRows / InH) != row)) {
				color.R *= 0.65F;
				color.G *= 0.65F;
				color.B *= 0.65F;
			}
			PutPixel(InFb, InFbW, InFbH, InX + x, InY + y, color);
		}
	}
}

static void PaintLine(OaU8* InFb, OaI32 InFbW, OaI32 InFbH,
	OaI32 InX, OaI32 InY, OaI32 InW, OaI32 InH,
	const OaF32* InValues, OaI32 InCount, OaColor InColor) {
	if (InCount <= 0 || InW <= 0 || InH <= 0) return;
	OaF32 minimum = std::numeric_limits<OaF32>::infinity();
	OaF32 maximum = -std::numeric_limits<OaF32>::infinity();
	for (OaI32 i = 0; i < InCount; ++i) {
		if (!std::isfinite(InValues[i])) continue;
		minimum = std::min(minimum, InValues[i]);
		maximum = std::max(maximum, InValues[i]);
	}
	if (!std::isfinite(minimum) || !std::isfinite(maximum)) return;
	if (maximum <= minimum) {
		const OaF32 margin = std::max(1.0e-4F, std::abs(minimum) * 0.05F);
		minimum -= margin;
		maximum += margin;
	}
	for (OaI32 q = 1; q < 4; ++q) {
		const OaI32 gx = InX + InW * q / 4;
		const OaI32 gy = InY + InH * q / 4;
		for (OaI32 y = 0; y < InH; ++y)
			PutPixel(InFb, InFbW, InFbH, gx, InY + y,
				{1.0F, 1.0F, 1.0F, 0.08F});
		for (OaI32 x = 0; x < InW; ++x)
			PutPixel(InFb, InFbW, InFbH, InX + x, gy,
				{1.0F, 1.0F, 1.0F, 0.08F});
	}
	auto sampleY = [&](OaI32 x) {
		const OaF32 position = InW <= 1 ? 0.0F
			: static_cast<OaF32>(x) / static_cast<OaF32>(InW - 1);
		const OaF32 index = position * static_cast<OaF32>(InCount - 1);
		const OaI32 lo = static_cast<OaI32>(std::floor(index));
		const OaI32 hi = std::min(InCount - 1, lo + 1);
		const OaF32 value = InValues[lo]
			+ (InValues[hi] - InValues[lo]) * (index - static_cast<OaF32>(lo));
		return InY + static_cast<OaI32>(std::round((1.0F
			- (value - minimum) / (maximum - minimum)) * (InH - 1)));
	};
	for (OaI32 x = 0; x < InW; ++x) {
		const OaI32 y0 = sampleY(std::max(0, x - 1));
		const OaI32 y1 = sampleY(x);
		for (OaI32 y = std::min(y0, y1); y <= std::max(y0, y1); ++y)
			PutPixel(InFb, InFbW, InFbH, InX + x, y, InColor);
	}
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
			if ((!ax.Image_.Present || !ax.Image_.Tex.IsValid())
				&& !ax.Heatmap_.Present && !ax.Line_.Present) continue;

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

			if (ax.Heatmap_.Present && !ax.Heatmap_.V.Empty()) {
				OaF32 vMin = ax.Heatmap_.Style.VMin;
				OaF32 vMax = ax.Heatmap_.Style.VMax;
				if (ax.Heatmap_.Style.AutoScale) {
					vMin = std::numeric_limits<OaF32>::infinity();
					vMax = -std::numeric_limits<OaF32>::infinity();
					for (const OaF32 value : ax.Heatmap_.V) {
						if (!std::isfinite(value)) continue;
						vMin = std::min(vMin, value);
						vMax = std::max(vMax, value);
					}
				}
				PaintHeatmap(fb.Data(), W, H, cell.X, imgRegionY,
					cell.W, imgRegionH, ax.Heatmap_.V.Data(),
					ax.Heatmap_.Rows, ax.Heatmap_.Cols, vMin, vMax,
					ax.Heatmap_.Style.Colormap, ax.Heatmap_.Style.ShowGrid);
				continue;
			}
			if (ax.Line_.Present && !ax.Line_.Y.Empty()) {
				PaintLine(fb.Data(), W, H, cell.X, imgRegionY, cell.W,
					imgRegionH, ax.Line_.Y.Data(),
					static_cast<OaI32>(ax.Line_.Y.Size()), ax.Line_.Style.Color);
				continue;
			}

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
