#include <oa/ui/detectionOverlay.h>

#include <oa/runtime/engine.h>
#include <oa/ui/text.h>
#include <oa/ui/ui.h>

#include <oa/core/std/algo.h>
#include <oa/core/std/array.h>
#include <oa/core/std/scalarMath.h>
namespace {

constexpr oa::U32 kOverlayRingSize = 3;

struct OverlaySlot {
	oa::DetectionBuffer detections;
	oa::GlyphBuffer glyphs;

	[[nodiscard]] bool isReady() const {
		return detections.isReady() && glyphs.isReady();
	}
};

oa::Detection canonicalizeDetection(
	const oa::Detection& inDetection,
	oa::U32 inFallbackColor) {
	oa::Detection result = inDetection;
	result.centerX = oa::clamp(result.centerX, 0.0F, 1.0F);
	result.centerY = oa::clamp(result.centerY, 0.0F, 1.0F);
	result.width = oa::clamp(result.width, 0.0F, 1.0F);
	result.height = oa::clamp(result.height, 0.0F, 1.0F);
	result.confidence = oa::clamp(result.confidence, 0.0F, 1.0F);
	if (result.colorRgba == 0) result.colorRgba = inFallbackColor;
	return result;
}

bool isFiniteDetection(const oa::Detection& inDetection) {
	return oa::isFinite(inDetection.centerX)
		&& oa::isFinite(inDetection.centerY)
		&& oa::isFinite(inDetection.width)
		&& oa::isFinite(inDetection.height)
		&& oa::isFinite(inDetection.confidence);
}

void appendLabel(
	const oa::TextAtlas& inAtlas,
	const oa::DetectionOverlayConfig& inConfig,
	const oa::Detection& inDetection,
	oa::StringView inLabel,
	oa::Vec<oa::GlyphInstance>& inOutGlyphs) {
	if (!inConfig.showLabels || inLabel.empty()) return;

	const oa::F32 anchorX = inDetection.centerX - inDetection.width * 0.5F;
	const oa::F32 anchorY = inDetection.centerY - inDetection.height * 0.5F;
	oa::TextLayout layout;
	oa::TextLayoutConfig layoutConfig{
		.font = oa::FontId::Sans,
		.size = inConfig.fontSize,
	};
	oa::Vec<oa::PositionedGlyph> positioned;
	layout.shape(
		inAtlas, inLabel, {}, layoutConfig,
		inConfig.labelTextColor.toU32(), positioned);
	if (positioned.empty()) return;
	const oa::F32 textWidth = layout.measure(inAtlas, inLabel, layoutConfig).x;
	oa::F32 textTop = 1.0e9F;
	oa::F32 textBottom = -1.0e9F;
	for (const auto& item : positioned) {
		const oa::GlyphInfo* glyph = inAtlas.findGlyph(
			item.font, item.codepoint, inConfig.fontSize);
		if (!glyph) continue;
		const oa::F32 scale = inConfig.fontSize / glyph->rasterSize;
		const oa::F32 glyphTop = -glyph->bearingY * scale;
		textTop = oa::min(textTop, glyphTop);
		textBottom = oa::max(textBottom, glyphTop + glyph->inkH * scale);
	}
	if (textBottom <= textTop) return;

	const oa::F32 labelWidth = textWidth + inConfig.labelPaddingX * 2.0F;
	const oa::F32 labelHeight = inConfig.fontSize + inConfig.labelPaddingY * 2.0F;
	const oa::F32 textHeight = textBottom - textTop;
	const oa::F32 baseline =
		-labelHeight + (labelHeight - textHeight) * 0.5F - textTop;
	inOutGlyphs.pushBack({
		.anchorX = anchorX,
		.anchorY = anchorY,
		.offsetX = 0.0F,
		.offsetY = -labelHeight,
		.width = labelWidth,
		.height = labelHeight,
		.color = inDetection.colorRgba,
	});

	for (const auto& item : positioned) {
		const oa::GlyphInfo* glyph = inAtlas.findGlyph(
			item.font, item.codepoint, inConfig.fontSize);
		if (!glyph) continue;
		const oa::F32 scale = inConfig.fontSize / glyph->rasterSize;
		inOutGlyphs.pushBack({
			.anchorX = anchorX,
			.anchorY = anchorY,
			.offsetX = inConfig.labelPaddingX + item.x
				+ glyph->bearingX * scale,
			.offsetY = baseline - glyph->bearingY * scale,
			.width = glyph->atlasW * scale,
			.height = glyph->atlasH * scale,
			.atlasX = static_cast<oa::U32>(glyph->atlasX),
			.atlasY = static_cast<oa::U32>(glyph->atlasY),
			.atlasW = static_cast<oa::U32>(glyph->atlasW),
			.atlasH = static_cast<oa::U32>(glyph->atlasH),
			.color = inConfig.labelTextColor.toU32(),
		});
	}
}

} // namespace

struct oa::DetectionOverlay::Impl {
	oa::Engine* runtime = nullptr;
	oa::DetectionOverlayConfig config;
	oa::Array<OverlaySlot, kOverlayRingSize> slots;
	oa::I32 activeSlot = -1;
	oa::U32 nextSlot = 0;
	oa::U32 count = 0;
};

oa::DetectionOverlay::DetectionOverlay(oa::DetectionOverlay&& inOther) noexcept
	: impl_(oa::move(inOther.impl_)) {}

oa::DetectionOverlay& oa::DetectionOverlay::operator=(
	oa::DetectionOverlay&& inOther) noexcept {
	if (this != &inOther) {
		reset_();
		impl_ = oa::move(inOther.impl_);
	}
	return *this;
}

oa::DetectionOverlay::~DetectionOverlay() {
	reset_();
}

oa::Result<oa::DetectionOverlay> oa::DetectionOverlay::create(
	oa::Engine& inRuntime,
	const oa::DetectionOverlayConfig& inConfig) {
	if (inConfig.maxDetections == 0 || inConfig.maxGlyphs == 0) {
		return oa::Status::invalidArgument(
			"oa::DetectionOverlay: capacities must be non-zero");
	}

	auto impl = oa::makeUnique<Impl>();
	impl->runtime = &inRuntime;
	impl->config = inConfig;
	for (auto& slot : impl->slots) {
		auto detections = oa::DetectionBuffer::createHostUpload(
			inRuntime, inConfig.maxDetections);
		if (!detections.isOk()) {
			return detections.getStatus();
		}
		slot.detections = oa::move(*detections);

		auto glyphs = oa::GlyphBuffer::createHostUpload(
			inRuntime, inConfig.maxGlyphs);
		if (!glyphs.isOk()) {
			return glyphs.getStatus();
		}
		slot.glyphs = oa::move(*glyphs);
	}

	oa::DetectionOverlay result;
	result.impl_ = oa::move(impl);
	return result;
}

oa::Status oa::DetectionOverlay::update(
	oa::Span<const oa::DetectionOverlayItem> inItems,
	const oa::TextAtlas& inAtlas) {
	if (!impl_) {
		return oa::Status::error(
			oa::StatusCode::FailedPrecondition,
			"oa::DetectionOverlay: not initialized");
	}
	if (inItems.size() > impl_->config.maxDetections) {
		return oa::Status::error(
			oa::StatusCode::ResourceExhausted,
			"oa::DetectionOverlay: detection capacity exceeded");
	}

	oa::I32 selected = -1;
	for (oa::U32 offset = 0; offset < kOverlayRingSize; ++offset) {
		const oa::U32 index = (impl_->nextSlot + offset) % kOverlayRingSize;
		if (impl_->slots[index].isReady()) {
			selected = static_cast<oa::I32>(index);
			break;
		}
	}
	if (selected < 0) {
		return oa::Status::error(
			oa::StatusCode::Unavailable,
			"oa::DetectionOverlay: all upload slots are in flight");
	}

	oa::Vec<oa::Detection> detections;
	oa::Vec<oa::GlyphInstance> glyphs;
	detections.reserve(inItems.size());
	glyphs.reserve(oa::min<oa::Usize>(
		impl_->config.maxGlyphs, inItems.size() * 24));
	const oa::U32 fallbackColor = impl_->config.boxColor.toU32();
	for (const auto& item : inItems) {
		if (!isFiniteDetection(item.detection)) {
			return oa::Status::invalidArgument(
				"oa::DetectionOverlay: non-finite detection geometry");
		}
		oa::Detection detection = canonicalizeDetection(
			item.detection, fallbackColor);
		if (detection.width <= 0.0F || detection.height <= 0.0F) continue;
		detections.pushBack(detection);
		appendLabel(inAtlas, impl_->config, detection, item.label, glyphs);
	}
	if (glyphs.size() > impl_->config.maxGlyphs) {
		return oa::Status::error(
			oa::StatusCode::ResourceExhausted,
			"oa::DetectionOverlay: glyph capacity exceeded");
	}

	auto& slot = impl_->slots[static_cast<oa::U32>(selected)];
	OA_RETURN_IF_ERROR(slot.detections.upload(
		oa::Span<const oa::Detection>(detections.data(), detections.size())));
	OA_RETURN_IF_ERROR(slot.glyphs.upload(
		oa::Span<const oa::GlyphInstance>(glyphs.data(), glyphs.size())));
	impl_->activeSlot = selected;
	impl_->nextSlot = (static_cast<oa::U32>(selected) + 1) % kOverlayRingSize;
	impl_->count = static_cast<oa::U32>(detections.size());
	return oa::Status::ok();
}

void oa::DetectionOverlay::draw(
	oa::Ui& inUi,
	const oa::TextAtlas& inAtlas,
	oa::PixelRect inDestination,
	oa::PixelRect inClip) const {
	if (!impl_ || impl_->activeSlot < 0) return;
	const auto& slot = impl_->slots[static_cast<oa::U32>(impl_->activeSlot)];
	inUi.rectOutlines(
		slot.detections,
		inDestination,
		inClip,
		impl_->config.boxColor,
		static_cast<oa::U32>(oa::max(1.0F, oa::round(impl_->config.thicknessPixels))));
	inUi.glyphs(slot.glyphs, inAtlas, inDestination, inClip);
}

oa::Status oa::DetectionOverlay::markConsumed(const oa::Event& inCompletion) {
	if (!impl_ || impl_->activeSlot < 0) return oa::Status::ok();
	auto& slot = impl_->slots[static_cast<oa::U32>(impl_->activeSlot)];
	OA_RETURN_IF_ERROR(slot.detections.markConsumed(inCompletion));
	return slot.glyphs.markConsumed(inCompletion);
}

void oa::DetectionOverlay::reset_() noexcept {
	impl_.reset();
}

bool oa::DetectionOverlay::isValid() const noexcept {
	return impl_ != nullptr;
}

oa::U32 oa::DetectionOverlay::count() const noexcept {
	return impl_ ? impl_->count : 0;
}
