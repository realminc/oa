#include <Oa/Ui/Text.h>

#include <Oa/Core/Memory.h>
#include <Oa/Runtime/Engine.h>

#include <algorithm>
#include <new>

namespace {

struct OaGeneratedGlyph {
	OaU32 Codepoint;
	OaU32 AtlasX;
	OaU32 AtlasY;
	OaU32 AtlasW;
	OaU32 AtlasH;
	OaF32 BearingX;
	OaF32 BearingY;
	OaF32 Advance;
	OaU32 InkW;
	OaU32 InkH;
};

#include "Generated/IBMPlexSansSdf.inc"

struct OaTextAtlasImpl {
	OaEngine* Runtime = nullptr;
	OaVkBuffer Atlas;
	OaGlyphInfo Glyphs[kPlexGlyphCount];
};

} // namespace

OaTextAtlas::OaTextAtlas(OaTextAtlas&& InOther) noexcept
	: Impl_(InOther.Impl_)
	, PxRange_(InOther.PxRange_)
	, AtlasW_(InOther.AtlasW_)
	, AtlasH_(InOther.AtlasH_)
{
	InOther.Impl_    = nullptr;
	InOther.AtlasW_  = 0.0F;
	InOther.AtlasH_  = 0.0F;
}

OaTextAtlas& OaTextAtlas::operator=(OaTextAtlas&& InOther) noexcept {
	if (this != &InOther) {
		Destroy();
		Impl_    = InOther.Impl_;
		PxRange_ = InOther.PxRange_;
		AtlasW_  = InOther.AtlasW_;
		AtlasH_  = InOther.AtlasH_;
		InOther.Impl_   = nullptr;
		InOther.AtlasW_ = 0.0F;
		InOther.AtlasH_ = 0.0F;
	}
	return *this;
}

OaTextAtlas::~OaTextAtlas() { Destroy(); }

OaStatus OaTextAtlas::Init(OaEngine& InRt) {
	Destroy();
	auto allocation = InRt.AllocBuffer(sizeof(kPlexAtlasPixels));
	if (!allocation.IsOk()) return allocation.GetStatus();
	if (!allocation->MappedPtr) {
		InRt.FreeBuffer(*allocation);
		return OaStatus::Error(
			OaStatusCode::FailedPrecondition,
			"OaTextAtlas: atlas allocation is not host-visible");
	}

	auto* impl = new (std::nothrow) OaTextAtlasImpl();
	if (!impl) {
		InRt.FreeBuffer(*allocation);
		return OaStatus::Error(OaStatusCode::OutOfMemory, "OaTextAtlas: metadata allocation failed");
	}
	impl->Runtime = &InRt;
	impl->Atlas = OaStdMove(*allocation);
	OaMemcpy(impl->Atlas.MappedPtr, kPlexAtlasPixels, sizeof(kPlexAtlasPixels));
	for (OaU32 i = 0; i < kPlexGlyphCount; ++i) {
		const auto& source = kPlexGlyphs[i];
		impl->Glyphs[i] = {
			.Codepoint = source.Codepoint,
			.AtlasX = static_cast<OaF32>(source.AtlasX),
			.AtlasY = static_cast<OaF32>(source.AtlasY),
			.AtlasW = static_cast<OaF32>(source.AtlasW),
			.AtlasH = static_cast<OaF32>(source.AtlasH),
			.BearingX = source.BearingX,
			.BearingY = source.BearingY,
			.Advance = source.Advance,
			.InkW = static_cast<OaF32>(source.InkW),
			.InkH = static_cast<OaF32>(source.InkH),
		};
	}
	Impl_ = impl;
	PxRange_ = kPlexAtlasPxRange;
	AtlasW_ = static_cast<OaF32>(kPlexAtlasWidth);
	AtlasH_ = static_cast<OaF32>(kPlexAtlasHeight);
	return OaStatus::Ok();
}

void OaTextAtlas::Destroy() {
	auto* impl = static_cast<OaTextAtlasImpl*>(Impl_);
	if (impl) {
		if (impl->Runtime && impl->Atlas.Buffer) {
			impl->Runtime->FreeBuffer(impl->Atlas);
		}
		delete impl;
	}
	Impl_ = nullptr;
	AtlasW_ = 0.0F;
	AtlasH_ = 0.0F;
}

const OaGlyphInfo* OaTextAtlas::FindGlyph(
	OaFontId InFont,
	OaU32 InCodepoint) const noexcept {
	if (!Impl_ || InFont != OaFontId::Sans
		|| InCodepoint < 32 || InCodepoint > 126) {
		return nullptr;
	}
	auto* impl = static_cast<const OaTextAtlasImpl*>(Impl_);
	const auto& glyph = impl->Glyphs[InCodepoint - 32];
	if (glyph.Codepoint == InCodepoint) return &glyph;
	return nullptr;
}

OaU32 OaTextAtlas::AtlasBindlessIndex(OaFontId InFont) const noexcept {
	if (!Impl_ || InFont != OaFontId::Sans) return OA_BINDLESS_INVALID;
	return static_cast<const OaTextAtlasImpl*>(Impl_)->Atlas.BindlessIndex;
}

OaGlyphBuffer::OaGlyphBuffer(OaGlyphBuffer&& InOther) noexcept
	: Runtime_(InOther.Runtime_)
	, Buffer_(InOther.Buffer_)
	, ConsumerSemaphore_(InOther.ConsumerSemaphore_)
	, ConsumerValue_(InOther.ConsumerValue_)
	, Count_(InOther.Count_)
	, Capacity_(InOther.Capacity_) {
	InOther.Runtime_ = nullptr;
	InOther.Buffer_ = {};
	InOther.ConsumerSemaphore_ = {};
	InOther.ConsumerValue_ = 0;
	InOther.Count_ = 0;
	InOther.Capacity_ = 0;
}

OaGlyphBuffer& OaGlyphBuffer::operator=(OaGlyphBuffer&& InOther) noexcept {
	if (this != &InOther) {
		Destroy();
		Runtime_ = InOther.Runtime_;
		Buffer_ = InOther.Buffer_;
		ConsumerSemaphore_ = InOther.ConsumerSemaphore_;
		ConsumerValue_ = InOther.ConsumerValue_;
		Count_ = InOther.Count_;
		Capacity_ = InOther.Capacity_;
		InOther.Runtime_ = nullptr;
		InOther.Buffer_ = {};
		InOther.ConsumerSemaphore_ = {};
		InOther.ConsumerValue_ = 0;
		InOther.Count_ = 0;
		InOther.Capacity_ = 0;
	}
	return *this;
}

OaGlyphBuffer::~OaGlyphBuffer() { Destroy(); }

OaResult<OaGlyphBuffer> OaGlyphBuffer::CreateHostUpload(
	OaEngine& InRuntime,
	OaU32 InCapacity) {
	if (InCapacity == 0) {
		return OaStatus::InvalidArgument(
			"OaGlyphBuffer::CreateHostUpload: capacity must be non-zero");
	}
	auto allocation = InRuntime.AllocBuffer(
		static_cast<OaU64>(InCapacity) * sizeof(OaGlyphInstance));
	if (!allocation.IsOk()) return allocation.GetStatus();
	OaGlyphBuffer result;
	result.Runtime_ = &InRuntime;
	result.Buffer_ = OaStdMove(*allocation);
	result.Capacity_ = InCapacity;
	return result;
}

void OaGlyphBuffer::Destroy() {
	if (Runtime_ && Buffer_.Buffer) Runtime_->FreeBuffer(Buffer_);
	Runtime_ = nullptr;
	Buffer_ = {};
	ConsumerSemaphore_ = {};
	ConsumerValue_ = 0;
	Count_ = 0;
	Capacity_ = 0;
}

OaStatus OaGlyphBuffer::Upload(OaSpan<const OaGlyphInstance> InGlyphs) {
	if (!Runtime_ || !Buffer_.Buffer || !Buffer_.MappedPtr) {
		return OaStatus::Error(
			OaStatusCode::FailedPrecondition,
			"OaGlyphBuffer::Upload: buffer is not host-visible");
	}
	if (!IsReady()) {
		return OaStatus::Error(
			OaStatusCode::Unavailable,
			"OaGlyphBuffer::Upload: buffer is still consumed by the GPU");
	}
	if (InGlyphs.Size() > Capacity_) {
		return OaStatus::Error(
			OaStatusCode::ResourceExhausted,
			"OaGlyphBuffer::Upload: glyph count exceeds capacity");
	}
	const OaUsize bytes = InGlyphs.Size() * sizeof(OaGlyphInstance);
	if (bytes > 0) OaMemcpy(Buffer_.MappedPtr, InGlyphs.Data(), bytes);
	Count_ = static_cast<OaU32>(InGlyphs.Size());
	ConsumerSemaphore_ = {};
	ConsumerValue_ = 0;
	return OaStatus::Ok();
}

void OaGlyphBuffer::MarkConsumed(
	const OaVkTimelineSemaphore& InSemaphore,
	OaU64 InValue) {
	if (!IsValid() || InSemaphore.Semaphore == nullptr || InValue == 0) return;
	ConsumerSemaphore_ = InSemaphore;
	ConsumerValue_ = InValue;
}

bool OaGlyphBuffer::IsReady() const {
	if (!Runtime_ || !IsValid()) return false;
	return ConsumerSemaphore_.Semaphore == nullptr
		|| ConsumerSemaphore_.GetValue(Runtime_->Device) >= ConsumerValue_;
}

namespace {

OaU32 DecodeAscii(OaStringView InText, OaUsize& InOutIndex) {
	const auto byte = static_cast<OaU8>(InText[InOutIndex++]);
	if (byte < 0x80) return byte;
	while (InOutIndex < InText.size()
		&& (static_cast<OaU8>(InText[InOutIndex]) & 0xC0U) == 0x80U) {
		++InOutIndex;
	}
	return static_cast<OaU32>('?');
}

} // namespace

void OaTextLayout::Shape(
	const OaTextAtlas& InAtlas,
	OaStringView InText,
	VlmVec2 InOrigin,
	const OaTextLayoutConfig& InCfg,
	OaU32 InColor,
	OaVec<OaPositionedGlyph>& InOutGlyphs) const {
	const OaF32 scale = InCfg.Size / InAtlas.BaseFontSize();
	OaF32 x = InOrigin.X;
	OaF32 y = InOrigin.Y;
	for (OaUsize i = 0; i < InText.size();) {
		const OaU32 codepoint = DecodeAscii(InText, i);
		if (codepoint == '\n') {
			x = InOrigin.X;
			y += InCfg.Size * 1.25F;
			continue;
		}
		const OaGlyphInfo* glyph = InAtlas.FindGlyph(InCfg.Font, codepoint);
		if (!glyph) continue;
		if (InCfg.WrapWidth > 0.0F && x > InOrigin.X
			&& x + glyph->Advance * scale > InOrigin.X + InCfg.WrapWidth) {
			x = InOrigin.X;
			y += InCfg.Size * 1.25F;
		}
		InOutGlyphs.PushBack({
			.X = x,
			.Y = y,
			.Codepoint = codepoint,
			.Color = InColor,
			.Font = InCfg.Font,
		});
		x += glyph->Advance * scale;
	}
}

VlmVec2 OaTextLayout::Measure(
	const OaTextAtlas& InAtlas,
	OaStringView InText,
	const OaTextLayoutConfig& InCfg) const {
	const OaF32 scale = InCfg.Size / InAtlas.BaseFontSize();
	OaF32 lineWidth = 0.0F;
	OaF32 maxWidth = 0.0F;
	OaF32 height = InCfg.Size;
	for (OaUsize i = 0; i < InText.size();) {
		const OaU32 codepoint = DecodeAscii(InText, i);
		if (codepoint == '\n') {
			maxWidth = std::max(maxWidth, lineWidth);
			lineWidth = 0.0F;
			height += InCfg.Size * 1.25F;
			continue;
		}
		const OaGlyphInfo* glyph = InAtlas.FindGlyph(InCfg.Font, codepoint);
		if (!glyph) continue;
		const OaF32 advance = glyph->Advance * scale;
		if (InCfg.WrapWidth > 0.0F && lineWidth > 0.0F
			&& lineWidth + advance > InCfg.WrapWidth) {
			maxWidth = std::max(maxWidth, lineWidth);
			lineWidth = 0.0F;
			height += InCfg.Size * 1.25F;
		}
		lineWidth += advance;
	}
	return {std::max(maxWidth, lineWidth), height};
}
