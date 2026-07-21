#include <Oa/Ui/DetectionOverlay.h>

#include <Oa/Render/Renderer.h>
#include <Oa/Runtime/Engine.h>
#include <Oa/Ui/Camera.h>
#include <Oa/Ui/Text.h>
#include <Oa/Ui/Ui.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <new>

namespace {

constexpr OaU32 kOverlayRingSize = 3;

struct OverlaySlot {
	OaDetectionBuffer Detections;
	OaGlyphBuffer Glyphs;

	[[nodiscard]] bool IsReady() const {
		return Detections.IsReady() && Glyphs.IsReady();
	}

	void Destroy() {
		Detections.Destroy();
		Glyphs.Destroy();
	}
};

OaDetection CanonicalizeDetection(
	const OaDetection& InDetection,
	OaU32 InFallbackColor) {
	OaDetection result = InDetection;
	result.CenterX = OaStdClamp(result.CenterX, 0.0F, 1.0F);
	result.CenterY = OaStdClamp(result.CenterY, 0.0F, 1.0F);
	result.Width = OaStdClamp(result.Width, 0.0F, 1.0F);
	result.Height = OaStdClamp(result.Height, 0.0F, 1.0F);
	result.Confidence = OaStdClamp(result.Confidence, 0.0F, 1.0F);
	if (result.ColorRgba == 0) result.ColorRgba = InFallbackColor;
	return result;
}

bool IsFinite(const OaDetection& InDetection) {
	return std::isfinite(InDetection.CenterX)
		&& std::isfinite(InDetection.CenterY)
		&& std::isfinite(InDetection.Width)
		&& std::isfinite(InDetection.Height)
		&& std::isfinite(InDetection.Confidence);
}

void AppendLabel(
	const OaTextAtlas& InAtlas,
	const OaDetectionOverlayConfig& InConfig,
	const OaDetection& InDetection,
	OaStringView InLabel,
	OaVec<OaGlyphInstance>& InOutGlyphs) {
	if (!InConfig.ShowLabels || InLabel.Empty()) return;

	const OaF32 scale = InConfig.FontSize / InAtlas.BaseFontSize();
	const OaF32 anchorX = InDetection.CenterX - InDetection.Width * 0.5F;
	const OaF32 anchorY = InDetection.CenterY - InDetection.Height * 0.5F;
	const OaF32 sdfInset = InAtlas.PxRange() + 2.0F;
	OaF32 textWidth = 0.0F;
	OaF32 textTop = 1.0e9F;
	OaF32 textBottom = -1.0e9F;
	for (const char character : InLabel) {
		const OaGlyphInfo* glyph = InAtlas.FindGlyph(
			OaFontId::Sans, static_cast<OaU8>(character));
		if (!glyph) continue;
		const OaF32 glyphTop = (-glyph->BearingY + sdfInset) * scale;
		textTop = std::min(textTop, glyphTop);
		textBottom = std::max(textBottom, glyphTop + glyph->InkH * scale);
		textWidth += glyph->Advance * scale;
	}
	if (textBottom <= textTop) return;

	const OaF32 labelWidth = textWidth + InConfig.LabelPaddingX * 2.0F;
	const OaF32 labelHeight = InConfig.FontSize + InConfig.LabelPaddingY * 2.0F;
	const OaF32 textHeight = textBottom - textTop;
	const OaF32 baseline =
		-labelHeight + (labelHeight - textHeight) * 0.5F - textTop;
	InOutGlyphs.PushBack({
		.AnchorX = anchorX,
		.AnchorY = anchorY,
		.OffsetX = 0.0F,
		.OffsetY = -labelHeight,
		.Width = labelWidth,
		.Height = labelHeight,
		.Color = InDetection.ColorRgba,
	});

	OaF32 penX = InConfig.LabelPaddingX;
	for (const char character : InLabel) {
		const OaU32 codepoint = static_cast<OaU8>(character);
		const OaGlyphInfo* glyph = InAtlas.FindGlyph(OaFontId::Sans, codepoint);
		if (!glyph) continue;
		InOutGlyphs.PushBack({
			.AnchorX = anchorX,
			.AnchorY = anchorY,
			.OffsetX = penX + glyph->BearingX * scale,
			.OffsetY = baseline - glyph->BearingY * scale,
			.Width = glyph->AtlasW * scale,
			.Height = glyph->AtlasH * scale,
			.AtlasX = static_cast<OaU32>(glyph->AtlasX),
			.AtlasY = static_cast<OaU32>(glyph->AtlasY),
			.AtlasW = static_cast<OaU32>(glyph->AtlasW),
			.AtlasH = static_cast<OaU32>(glyph->AtlasH),
			.Color = InConfig.LabelTextColor.ToU32(),
		});
		penX += glyph->Advance * scale;
	}
}

} // namespace

struct OaDetectionOverlay::Impl {
	OaEngine* Runtime = nullptr;
	OaDetectionOverlayConfig Config;
	std::array<OverlaySlot, kOverlayRingSize> Slots;
	OaI32 ActiveSlot = -1;
	OaU32 NextSlot = 0;
	OaU32 Count = 0;
};

OaDetectionOverlay::OaDetectionOverlay(OaDetectionOverlay&& InOther) noexcept
	: Impl_(InOther.Impl_) {
	InOther.Impl_ = nullptr;
}

OaDetectionOverlay& OaDetectionOverlay::operator=(
	OaDetectionOverlay&& InOther) noexcept {
	if (this != &InOther) {
		Destroy();
		Impl_ = InOther.Impl_;
		InOther.Impl_ = nullptr;
	}
	return *this;
}

OaDetectionOverlay::~OaDetectionOverlay() {
	Destroy();
}

OaResult<OaDetectionOverlay> OaDetectionOverlay::Create(
	OaEngine& InRuntime,
	const OaDetectionOverlayConfig& InConfig) {
	if (InConfig.MaxDetections == 0 || InConfig.MaxGlyphs == 0) {
		return OaStatus::InvalidArgument(
			"OaDetectionOverlay: capacities must be non-zero");
	}

	auto* impl = new (std::nothrow) Impl();
	if (!impl) {
		return OaStatus::Error(
			OaStatusCode::OutOfMemory,
			"OaDetectionOverlay: allocation failed");
	}
	impl->Runtime = &InRuntime;
	impl->Config = InConfig;
	for (auto& slot : impl->Slots) {
		auto detections = OaDetectionBuffer::CreateHostUpload(
			InRuntime, InConfig.MaxDetections);
		if (!detections.IsOk()) {
			for (auto& cleanup : impl->Slots) cleanup.Destroy();
			delete impl;
			return detections.GetStatus();
		}
		slot.Detections = OaStdMove(*detections);

		auto glyphs = OaGlyphBuffer::CreateHostUpload(
			InRuntime, InConfig.MaxGlyphs);
		if (!glyphs.IsOk()) {
			for (auto& cleanup : impl->Slots) cleanup.Destroy();
			delete impl;
			return glyphs.GetStatus();
		}
		slot.Glyphs = OaStdMove(*glyphs);
	}

	OaDetectionOverlay result;
	result.Impl_ = impl;
	return result;
}

OaStatus OaDetectionOverlay::Update(
	OaSpan<const OaDetectionOverlayItem> InItems,
	const OaTextAtlas& InAtlas) {
	if (!Impl_) {
		return OaStatus::Error(
			OaStatusCode::FailedPrecondition,
			"OaDetectionOverlay: not initialized");
	}
	if (InItems.Size() > Impl_->Config.MaxDetections) {
		return OaStatus::Error(
			OaStatusCode::ResourceExhausted,
			"OaDetectionOverlay: detection capacity exceeded");
	}

	OaI32 selected = -1;
	for (OaU32 offset = 0; offset < kOverlayRingSize; ++offset) {
		const OaU32 index = (Impl_->NextSlot + offset) % kOverlayRingSize;
		if (Impl_->Slots[index].IsReady()) {
			selected = static_cast<OaI32>(index);
			break;
		}
	}
	if (selected < 0) {
		return OaStatus::Error(
			OaStatusCode::Unavailable,
			"OaDetectionOverlay: all upload slots are in flight");
	}

	OaVec<OaDetection> detections;
	OaVec<OaGlyphInstance> glyphs;
	detections.Reserve(InItems.Size());
	glyphs.Reserve(std::min<OaUsize>(
		Impl_->Config.MaxGlyphs, InItems.Size() * 24));
	const OaU32 fallbackColor = Impl_->Config.BoxColor.ToU32();
	for (const auto& item : InItems) {
		if (!IsFinite(item.Detection)) {
			return OaStatus::InvalidArgument(
				"OaDetectionOverlay: non-finite detection geometry");
		}
		OaDetection detection = CanonicalizeDetection(
			item.Detection, fallbackColor);
		if (detection.Width <= 0.0F || detection.Height <= 0.0F) continue;
		detections.PushBack(detection);
		AppendLabel(InAtlas, Impl_->Config, detection, item.Label, glyphs);
	}
	if (glyphs.Size() > Impl_->Config.MaxGlyphs) {
		return OaStatus::Error(
			OaStatusCode::ResourceExhausted,
			"OaDetectionOverlay: glyph capacity exceeded");
	}

	auto& slot = Impl_->Slots[static_cast<OaU32>(selected)];
	OA_RETURN_IF_ERROR(slot.Detections.Upload(
		OaSpan<const OaDetection>(detections.Data(), detections.Size())));
	OA_RETURN_IF_ERROR(slot.Glyphs.Upload(
		OaSpan<const OaGlyphInstance>(glyphs.Data(), glyphs.Size())));
	Impl_->ActiveSlot = selected;
	Impl_->NextSlot = (static_cast<OaU32>(selected) + 1) % kOverlayRingSize;
	Impl_->Count = static_cast<OaU32>(detections.Size());
	return OaStatus::Ok();
}

void OaDetectionOverlay::Draw(
	OaUi& InUi,
	const OaTextAtlas& InAtlas,
	OaPixelRect InDestination,
	OaPixelRect InClip) const {
	if (!Impl_ || Impl_->ActiveSlot < 0) return;
	const auto& slot = Impl_->Slots[static_cast<OaU32>(Impl_->ActiveSlot)];
	InUi.RectOutlines(
		slot.Detections,
		InDestination,
		InClip,
		Impl_->Config.BoxColor,
		static_cast<OaU32>(std::max(1.0F, std::round(Impl_->Config.ThicknessPixels))));
	InUi.Glyphs(slot.Glyphs, InAtlas, InDestination, InClip);
}

void OaDetectionOverlay::Draw(
	OaCanvasRenderer& InRenderer,
	const OaTextAtlas& InAtlas,
	const OaCamera& InCamera,
	const VlmMat4& InModel,
	OaU32 InSourceWidth,
	OaU32 InSourceHeight) const {
	if (!Impl_ || Impl_->ActiveSlot < 0) return;
	const auto& slot = Impl_->Slots[static_cast<OaU32>(Impl_->ActiveSlot)];
	const OaColor color = Impl_->Config.BoxColor;
	InRenderer.DrawRectInstances({
		.InstanceBufferIndex = slot.Detections.BindlessIndex(),
		.InstanceCount = slot.Detections.Count(),
		.Camera = InCamera,
		.Model = InModel,
		.Color = {color.R, color.G, color.B, color.A},
		.ThicknessPixels = Impl_->Config.ThicknessPixels,
	});
	InRenderer.DrawGlyphInstances({
		.InstanceBufferIndex = slot.Glyphs.BindlessIndex(),
		.InstanceCount = slot.Glyphs.Count(),
		.AtlasBufferIndex = InAtlas.AtlasBindlessIndex(OaFontId::Sans),
		.AtlasWidth = static_cast<OaU32>(InAtlas.AtlasWidth()),
		.AtlasHeight = static_cast<OaU32>(InAtlas.AtlasHeight()),
		.ReferenceWidth = InSourceWidth,
		.ReferenceHeight = InSourceHeight,
		.AtlasPxRange = InAtlas.PxRange(),
		.Camera = InCamera,
		.Model = InModel,
	});
}

void OaDetectionOverlay::MarkConsumed(
	const OaVkTimelineSemaphore& InSemaphore,
	OaU64 InValue) {
	if (!Impl_ || Impl_->ActiveSlot < 0) return;
	auto& slot = Impl_->Slots[static_cast<OaU32>(Impl_->ActiveSlot)];
	slot.Detections.MarkConsumed(InSemaphore, InValue);
	slot.Glyphs.MarkConsumed(InSemaphore, InValue);
}

void OaDetectionOverlay::Destroy() {
	if (!Impl_) return;
	for (auto& slot : Impl_->Slots) slot.Destroy();
	delete Impl_;
	Impl_ = nullptr;
}

bool OaDetectionOverlay::IsValid() const noexcept {
	return Impl_ != nullptr;
}

OaU32 OaDetectionOverlay::Count() const noexcept {
	return Impl_ ? Impl_->Count : 0;
}
