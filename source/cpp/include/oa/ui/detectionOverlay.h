// DetectionOverlay — completion-safe GPU boxes and labels for image/video views.

// Geometry is normalized to the source image. Rectangle and glyph records are
// compact host-upload adapters today; native GPU inference/NMS can write the
// same oa::Detection record layout without changing the display consumer.

// Typical use from an Viewer live source:
//   overlay_ = *DetectionOverlay::create(rt);
//   overlay_.update(items, textAtlas);
//   ... draw image into dst ...
//   overlay_.draw(ui, textAtlas, dst, clip);
//   // in markConsumed:
//   overlay_.markConsumed(completion);

#pragma once

#include <oa/core/color.h>
#include <oa/core/status.h>
#include <oa/core/types.h>
#include <oa/ui/canvas.h>
#include <oa/vision/detection.h>

namespace oa { class Engine; }
namespace oa {

class TextAtlas;
class Ui;

struct DetectionOverlayItem {
	oa::Detection detection;
	oa::String label;
};

struct DetectionOverlayConfig {
	oa::U32 maxDetections = 256;
	oa::U32 maxGlyphs = 8192;
	oa::F32 thicknessPixels = 3.0F;
	oa::F32 fontSize = 18.0F;
	oa::F32 labelPaddingX = 4.0F;
	oa::F32 labelPaddingY = 2.0F;
	oa::Color boxColor = oa::Color::success();
	oa::Color labelTextColor = {0.0F, 0.0F, 0.0F, 1.0F};
	bool showLabels = true;
};

class DetectionOverlay {
public:
	DetectionOverlay() = default;
	DetectionOverlay(const DetectionOverlay&) = delete;
	DetectionOverlay& operator=(const DetectionOverlay&) = delete;
	DetectionOverlay(DetectionOverlay&& inOther) noexcept;
	DetectionOverlay& operator=(DetectionOverlay&& inOther) noexcept;
	~DetectionOverlay();

	[[nodiscard]] static oa::Result<DetectionOverlay> create(
		oa::Engine& inRuntime,
		const DetectionOverlayConfig& inConfig = {});

	// Non-blocking update. A three-slot completion-tracked ring prevents the
	// CPU from overwriting records still consumed by a previous GPU frame.
	[[nodiscard]] oa::Status update(
		oa::Span<const DetectionOverlayItem> inItems,
		const TextAtlas& inAtlas);

	// Compute-compose path used by the ordinary image/video widgets.
	void draw(
		Ui& inUi,
		const TextAtlas& inAtlas,
		PixelRect inDestination,
		PixelRect inClip) const;

	[[nodiscard]] oa::Status markConsumed(const oa::Event& inCompletion);

	[[nodiscard]] bool isValid() const noexcept;
	[[nodiscard]] oa::U32 count() const noexcept;

private:
	void reset_() noexcept;

	struct Impl;
	oa::UniquePtr<Impl> impl_;
};

}  // namespace oa
