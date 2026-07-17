// OaDetectionOverlay — completion-safe GPU boxes and labels for image/video views.

// Geometry is normalized to the source image. Rectangle and glyph records are
// compact host-upload adapters today; native GPU inference/NMS can write the
// same OaDetection record layout without changing the display consumer.

// Typical use from an OaDeviceUiApp:
//   Overlay_ = *OaDetectionOverlay::Create(rt);
//   Overlay_.Update(items, gpui.TextAtlas());
//   ... draw image into dst ...
//   Overlay_.Draw(oui, gpui.TextAtlas(), dst, clip);
//   // in OnRenderSubmitted:
//   Overlay_.MarkConsumed(semaphore, value);

#pragma once

#include <Oa/Core/Color.h>
#include <Oa/Core/Status.h>
#include <Oa/Core/Types.h>
#include <Oa/Core/Vlm.h>
#include <Oa/Ui/Canvas.h>
#include <Oa/Vision/Detection.h>

class OaCamera;
class OaCanvasRenderer;
class OaComputeEngine;
class OaTextAtlas;
class OaUi;
class OaVkTimelineSemaphore;

struct OaDetectionOverlayItem {
	OaDetection Detection;
	OaString Label;
};

struct OaDetectionOverlayConfig {
	OaU32 MaxDetections = 256;
	OaU32 MaxGlyphs = 8192;
	OaF32 ThicknessPixels = 3.0F;
	OaF32 FontSize = 18.0F;
	OaF32 LabelPaddingX = 4.0F;
	OaF32 LabelPaddingY = 2.0F;
	OaColor BoxColor = OaColor::Success();
	OaColor LabelTextColor = {0.0F, 0.0F, 0.0F, 1.0F};
	bool ShowLabels = true;
};

class OaDetectionOverlay {
public:
	OaDetectionOverlay() = default;
	OaDetectionOverlay(const OaDetectionOverlay&) = delete;
	OaDetectionOverlay& operator=(const OaDetectionOverlay&) = delete;
	OaDetectionOverlay(OaDetectionOverlay&& InOther) noexcept;
	OaDetectionOverlay& operator=(OaDetectionOverlay&& InOther) noexcept;
	~OaDetectionOverlay();

	[[nodiscard]] static OaResult<OaDetectionOverlay> Create(
		OaComputeEngine& InRuntime,
		const OaDetectionOverlayConfig& InConfig = {});

	// Non-blocking update. A three-slot completion-tracked ring prevents the
	// CPU from overwriting records still consumed by a previous GPU frame.
	[[nodiscard]] OaStatus Update(
		OaSpan<const OaDetectionOverlayItem> InItems,
		const OaTextAtlas& InAtlas);

	// Compute-compose path used by the ordinary image/video widgets.
	void Draw(
		OaUi& InUi,
		const OaTextAtlas& InAtlas,
		OaPixelRect InDestination,
		OaPixelRect InClip) const;

	// Camera-bound graphics path used by pan/zoom canvases.
	void Draw(
		OaCanvasRenderer& InRenderer,
		const OaTextAtlas& InAtlas,
		const OaCamera& InCamera,
		const VlmMat4& InModel,
		OaU32 InSourceWidth,
		OaU32 InSourceHeight) const;

	void MarkConsumed(const OaVkTimelineSemaphore& InSemaphore, OaU64 InValue);
	void Destroy();

	[[nodiscard]] bool IsValid() const noexcept;
	[[nodiscard]] OaU32 Count() const noexcept;

private:
	struct Impl;
	Impl* Impl_ = nullptr;
};
