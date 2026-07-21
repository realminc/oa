// OaUi — OUI: immediate-API widget layer with GPU-retained rendering.
//
// Call pattern (every frame):
//   oui.BeginFrame(delta_ms);
//   oui.BeginPanel("Train", {20, 20, 400, 600});
//     if (oui.Button("Run")) { ... }
//     oui.SliderF32("LR", &lr, 1e-5F, 1e-3F);
//     oui.PlotLine("loss", loss_data, count);
//   oui.EndPanel();
//   oui.EndFrame();
//
// Retained rendering: only dirty widgets re-dispatch their compute shader.
// Static frames (no changes) submit zero GPU work beyond re-blit.

#pragma once

#include <Oa/Core/Types.h>
#include <Oa/Core/Status.h>
#include <Oa/Ui/Style.h>
#include <Oa/Ui/Event.h>
#include <Oa/Ui/Canvas.h>
#include <vulkan/vulkan.h>

class OaEngine;
class OaVkTimelineSemaphore;
class OaVkBuffer;
class OaMatrix;
class OaDetectionBuffer;
class OaGlyphBuffer;
class OaTextAtlas;
struct OaImagePlanes;


// ─── Layout primitives ────────────────────────────────────────────────────────

enum class OuiDirection : OaU8 {
	Column = 0,
	Row    = 1,
};

enum class OuiAlign : OaU8 {
	Start   = 0,
	Center  = 1,
	End     = 2,
	Stretch = 3,
};

enum class OuiSizingKind : OaU8 {
	Fill  = 0,  // expand to fill parent
	Fixed = 1,  // fixed pixel size
	Hug   = 2,  // shrink-wrap content
};

struct OaUiSizing {
	OuiSizingKind Kind  = OuiSizingKind::Fill;
	OaF32         Value = 0.0F;  // pixel size when Kind == Fixed

	[[nodiscard]] static constexpr OaUiSizing Fill()          noexcept { return {.Kind = OuiSizingKind::Fill,  .Value = 0.0F}; }
	[[nodiscard]] static constexpr OaUiSizing Fixed(OaF32 Px) noexcept { return {.Kind = OuiSizingKind::Fixed, .Value = Px}; }
	[[nodiscard]] static constexpr OaUiSizing Hug()           noexcept { return {.Kind = OuiSizingKind::Hug,   .Value = 0.0F}; }
};

struct OaUiEdge {
	OaF32 Top    = 0.0F;
	OaF32 Right  = 0.0F;
	OaF32 Bottom = 0.0F;
	OaF32 Left   = 0.0F;

	constexpr OaUiEdge() = default;
	explicit constexpr OaUiEdge(OaF32 InAll) : Top(InAll), Right(InAll), Bottom(InAll), Left(InAll) {}
	constexpr OaUiEdge(OaF32 InV, OaF32 InH) : Top(InV), Right(InH), Bottom(InV), Left(InH) {}
};

struct OaUiLayout {
	OuiDirection Direction = OuiDirection::Column;
	OuiAlign     Align     = OuiAlign::Stretch;
	OuiAlign     Justify   = OuiAlign::Start;
	OaF32        Gap       = 4.0F;
	OaUiEdge      Padding   = OaUiEdge{8.0F};
	OaUiSizing    Width     = OaUiSizing::Fill();
	OaUiSizing    Height    = OaUiSizing::Hug();
};


// ─── Widget config structs ────────────────────────────────────────────────────

struct OaUiPlotConfig {
	OaColor Color     = {0.388F, 0.400F, 0.945F, 1.0F};  // Accent
	OaF32    YMin      = 0.0F;
	OaF32    YMax      = 1.0F;
	bool     AutoScale = true;
	bool     ShowGrid  = true;
	bool     Fill      = false;
};

struct OaUiHeatmapConfig {
	OaI32    Rows    = 0;
	OaI32    Cols    = 0;
	OaF32    VMin    = -1.0F;
	OaF32    VMax    =  1.0F;
	OaU32    Colormap = 0;  // 0=plasma 1=viridis 2=coolwarm 3=grays
	OaU32    ValueType = 0; // 0=Float32 1=UInt32 2=Int32
	OaU32    OffsetElements = 0;
	bool     ShowGrid = false;
};

// ─── OaUi ──────────────────────────────────────────────────────────────────────

class OaUi {
public:
	OaUi() = default;
	OaUi(const OaUi&)            = delete;
	OaUi& operator=(const OaUi&) = delete;
	OaUi(OaUi&&) noexcept;
	OaUi& operator=(OaUi&&) noexcept;
	~OaUi();

	[[nodiscard]] OaStatus Init(OaEngine& InRt, const OaUiStyle& InStyle = {});
	// Called once after Init, before the first frame.
	// InComposeImageView: VkImageView (as void*) of the compose storage image (set=1, slot 0).
	[[nodiscard]] OaStatus InitBlit(void* InComposeImageView);
	// Called after a compose image rebuild (resize) to refresh the image descriptor.
	void UpdateBlitImage(void* InComposeImageView);
	void Destroy();

	// ── Per-frame ─────────────────────────────────────────────────────────────

	void BeginFrame(OaF32 InDeltaMs);

	// Route a platform event.  Returns true if consumed (widget had focus).
	bool RouteEvent(const OaUiEvent& InEvent);

	// Record all widget dispatch commands into InCmd.
	void RecordRender(VkCommandBuffer InCmd, OaU32 InDstBindlessIdx);
	// Marks transient resources sampled by this frame. Plot buffers are recycled
	// only after the graphics timeline reaches this value.
	void MarkFrameSubmitted(
		const OaVkTimelineSemaphore& InSemaphore,
		OaU64 InValue);

	void EndFrame();

	// ── Style stack (O(1), max depth 32) ─────────────────────────────────────

	void PushStyle(const OaUiStyle& InStyle);
	void PopStyle();
	[[nodiscard]] const OaUiStyle& CurrentStyle() const noexcept;

	// ── Layout containers ─────────────────────────────────────────────────────

	void BeginPanel(OaStringView InId, OaPixelRect InRect, const OaUiLayout& InLayout = {});
	void EndPanel();

	void BeginRow(OaStringView InId = {});
	void EndRow();

	void Spacing(OaF32 InPixels);
	void Separator();

	// ── Widgets ───────────────────────────────────────────────────────────────

	// Returns true on click.
	[[nodiscard]] bool Button(OaStringView InLabel);
	// Returns true when state changes.
	[[nodiscard]] bool Checkbox(OaStringView InLabel, bool& InOutValue);
	// Returns true while dragging.
	[[nodiscard]] bool SliderF32(OaStringView InLabel, OaF32* InOutValue, OaF32 InMin, OaF32 InMax, const char* InFmt = "%.3F");
	[[nodiscard]] bool SliderI32(OaStringView InLabel, OaI32* InOutValue, OaI32 InMin, OaI32 InMax);
	[[nodiscard]] bool InputText(OaStringView InLabel, OaString& InOutText);
	[[nodiscard]] bool Dropdown(OaStringView InLabel, OaI32& InOutIndex, OaSpan<const OaStringView> InItems);

	void Label(OaStringView InText);
	void LabelFmt(const char* InFmt, ...);
	void Text(OaStringView InText);
	void ColorSwatch(OaColor InColor, VlmVec2 InSize = {16.0F, 16.0F});
	void ProgressBar(OaF32 InFraction, OaStringView InOverlay = {});
	// Explicit transport timeline. InOutFraction is normalized to [0, 1];
	// returns true when pointer scrubbing changes it.
	[[nodiscard]] bool Timeline(
		OaStringView InId,
		OaPixelRect InRect,
		OaF32& InOutFraction);
	// Full-surface audio scrubber backed by a GPU [Bins, 2] min/max envelope.
	[[nodiscard]] bool WaveformTimeline(
		OaStringView InId,
		OaPixelRect InRect,
		const OaMatrix& InEnvelope,
		OaF32& InOutFraction);
	void Image(OaU32 InBindlessIdx, OaI32 InW, OaI32 InH);
	void ImageVkRgba(
		void* InImage,
		void* InImageView,
		OaI32 InW,
		OaI32 InH,
		VkImageLayout InLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
	// Draw filled or outlined axis-aligned rectangles directly into the GPU
	// compose image. Rectangles must already be clipped to its extent.
	void Rect(OaPixelRect InRect, OaColor InColor);
	void RectOutline(
		OaPixelRect InRect,
		OaColor InColor,
		OaU32 InThickness = 1);
	// Draw an anti-aliased screen-space line in one GPU dispatch.
	void Line(
		VlmVec2 InBegin,
		VlmVec2 InEnd,
		OaColor InColor,
		OaF32 InThickness = 2.0F);
	// Draw normalized rectangle records from a bindless GPU buffer in one
	// dispatch. InDstRect maps source-image coordinates into the compose image.
	void RectOutlines(
		const OaDetectionBuffer& InDetections,
		OaPixelRect InDstRect,
		OaPixelRect InClipRect,
		OaColor InColor,
		OaU32 InThickness = 1);
	// Draw a source-anchored glyph batch from one persistent SDF atlas.
	void Glyphs(
		const OaGlyphBuffer& InGlyphs,
		const OaTextAtlas& InAtlas,
		OaPixelRect InDstRect,
		OaPixelRect InClipRect);
	// Planar path: BlitPlanar.slang handles per-channel dtype conversion + sRGB.
	void ImagePlanar(const OaImagePlanes& InPlanes, OaI32 InDstX = 0, OaI32 InDstY = 0);

	// ── Data visualization ────────────────────────────────────────────────────

	// CPU float array → line chart.
	void PlotLine(OaStringView InLabel, const OaF32* InData, OaI32 InCount, const OaUiPlotConfig& InCfg = {});
	// Ring-buffer variant: reads InCount floats from InData[InOffset % InCount].
	void PlotLineRing(OaStringView InLabel, const OaF32* InData, OaI32 InCount, OaI32 InOffset, const OaUiPlotConfig& InCfg = {});
	// GPU buffer → heatmap (zero-copy — reads directly from InBuffer on GPU).
	void Heatmap(OaStringView InLabel, const OaVkBuffer& InBuffer, const OaUiHeatmapConfig& InCfg);
	// Matrix convenience overload. Rows/Cols and value type are inferred when
	// omitted, and matrix byte offsets are honored for views.
	void Heatmap(OaStringView InLabel, const OaMatrix& InMatrix, const OaUiHeatmapConfig& InCfg = {});
	// Host values → heatmap through the same bounded frame-safe upload ring as
	// PlotLine. Intended for compact metric tables and recorded OaPlot figures.
	void Heatmap(OaStringView InLabel, const OaF32* InData, OaI32 InRows,
		OaI32 InCols, const OaUiHeatmapConfig& InCfg = {});

	// ── Input state ───────────────────────────────────────────────────────────

	[[nodiscard]] const OaUiInputState& Input() const noexcept { return Input_; }

private:
	struct Impl;
	OaUniquePtr<Impl> Impl_;
	OaUiInputState     Input_;
};
