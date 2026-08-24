#include <oa/ui/text.h>

#include <oa/core/memory.h>
#include <oa/runtime/engine.h>
#include <oa/runtime/engine/resourceAccess.h>

#include <hb.h>
#include <hb-ot.h>
#include <utf8proc.h>

#include <algorithm>
#include <cmath>
#include <limits>
#include <new>
#include <vector>

#include "consumedBufferRetirement.h"

namespace {

struct GeneratedGlyph {
	oa::U32 font;
	oa::U32 pixelSize;
	oa::U32 codepoint;
	oa::U32 atlasX;
	oa::U32 atlasY;
	oa::U32 atlasW;
	oa::U32 atlasH;
	oa::F32 bearingX;
	oa::F32 bearingY;
	oa::F32 advance;
};

#include "generated/textCoverageAtlas.inc"

static_assert(kTextGlyphCount
	== kTextFontCount * kTextStrikeCount * kTextCodepointCount);
static_assert(sizeof(kTextAtlasPixels)
	== static_cast<oa::Usize>(kTextAtlasWidth) * kTextAtlasHeight);

} // namespace

struct oa::TextAtlas::Impl {
	oa::Engine* runtime = nullptr;
	oavk::Buffer atlas;
	oa::GlyphInfo glyphs[kTextGlyphCount];
};

namespace {

oa::I32 codepointIndex(oa::U32 inCodepoint) noexcept {
	oa::U32 first = 0U;
	oa::U32 count = kTextCodepointCount;
	while (count > 0U) {
		const oa::U32 step = count / 2U;
		const oa::U32 index = first + step;
		if (kTextCodepoints[index] < inCodepoint) {
			first = index + 1U;
			count -= step + 1U;
		} else {
			count = step;
		}
	}
	return first < kTextCodepointCount && kTextCodepoints[first] == inCodepoint
		? static_cast<oa::I32>(first)
		: -1;
}

bool fontSupportsCodepoint(oa::FontId inFont, oa::U32 inCodepoint) noexcept {
	const oa::U32 font = static_cast<oa::U32>(inFont);
	const oa::I32 codepoint = codepointIndex(inCodepoint);
	return font < kTextFontCount && codepoint >= 0
		&& kTextCodepointSupport[font][static_cast<oa::U32>(codepoint)];
}

bool isValidFont(oa::FontId inFont) noexcept {
	return static_cast<oa::U32>(inFont) < kTextFontCount;
}

oa::I32 nearestStrikeIndex(oa::F32 inPixelSize) noexcept {
	if (!std::isfinite(inPixelSize) || inPixelSize <= 0.0F) return -1;
	oa::U32 nearest = 0U;
	oa::F32 nearestDistance = std::abs(
		inPixelSize - static_cast<oa::F32>(kTextStrikeSizes[0]));
	for (oa::U32 index = 1U; index < kTextStrikeCount; ++index) {
		const oa::F32 distance = std::abs(
			inPixelSize - static_cast<oa::F32>(kTextStrikeSizes[index]));
		if (distance < nearestDistance) {
			nearest = index;
			nearestDistance = distance;
		}
	}
	return static_cast<oa::I32>(nearest);
}

} // namespace

oa::TextAtlas::TextAtlas(oa::TextAtlas&& inOther) noexcept
	: impl_(oa::move(inOther.impl_))
	, atlasW_(inOther.atlasW_)
	, atlasH_(inOther.atlasH_)
{
	inOther.atlasW_  = 0.0F;
	inOther.atlasH_  = 0.0F;
}

oa::TextAtlas& oa::TextAtlas::operator=(oa::TextAtlas&& inOther) noexcept {
	if (this != &inOther) {
		reset_();
		impl_    = oa::move(inOther.impl_);
		atlasW_  = inOther.atlasW_;
		atlasH_  = inOther.atlasH_;
		inOther.atlasW_ = 0.0F;
		inOther.atlasH_ = 0.0F;
	}
	return *this;
}

oa::TextAtlas::~TextAtlas() { reset_(); }

oa::Status oa::TextAtlas::init(oa::Engine& inRt) {
	if (impl_ != nullptr) {
		return oa::Status::error(
			oa::StatusCode::AlreadyExists,
			"oa::TextAtlas::init requires an uninitialized atlas");
	}
	auto allocation = oa::EngineResourceAccess::allocBuffer(inRt, sizeof(kTextAtlasPixels));
	if (!allocation.isOk()) return allocation.getStatus();
	if (!allocation->mappedPtr) {
		oa::EngineResourceAccess::freeBuffer(inRt, *allocation);
		return oa::Status::error(
			oa::StatusCode::FailedPrecondition,
			"oa::TextAtlas: atlas allocation is not host-visible");
	}

	oa::UniquePtr<Impl> impl(new (std::nothrow) Impl());
	if (!impl) {
		oa::EngineResourceAccess::freeBuffer(inRt, *allocation);
		return oa::Status::error(oa::StatusCode::OutOfMemory, "oa::TextAtlas: metadata allocation failed");
	}
	impl->runtime = &inRt;
	impl->atlas = oa::move(*allocation);
	oa::memcpy(impl->atlas.mappedPtr, kTextAtlasPixels, sizeof(kTextAtlasPixels));
	for (oa::U32 i = 0; i < kTextGlyphCount; ++i) {
		const auto& source = kTextGlyphs[i];
		impl->glyphs[i] = {
			.codepoint = source.codepoint,
			.atlasX = static_cast<oa::F32>(source.atlasX),
			.atlasY = static_cast<oa::F32>(source.atlasY),
			.atlasW = static_cast<oa::F32>(source.atlasW),
			.atlasH = static_cast<oa::F32>(source.atlasH),
			.bearingX = source.bearingX,
			.bearingY = source.bearingY,
			.advance = source.advance,
			.inkW = static_cast<oa::F32>(source.atlasW),
			.inkH = static_cast<oa::F32>(source.atlasH),
			.rasterSize = static_cast<oa::F32>(source.pixelSize),
		};
	}
	impl_ = oa::move(impl);
	atlasW_ = static_cast<oa::F32>(kTextAtlasWidth);
	atlasH_ = static_cast<oa::F32>(kTextAtlasHeight);
	return oa::Status::ok();
}

void oa::TextAtlas::reset_() noexcept {
	if (impl_) {
		if (impl_->runtime && impl_->atlas.buffer) {
			oa::EngineResourceAccess::freeBuffer(*impl_->runtime, impl_->atlas);
		}
	}
	impl_.reset();
	atlasW_ = 0.0F;
	atlasH_ = 0.0F;
}

const oa::GlyphInfo* oa::TextAtlas::findGlyph(
	oa::FontId inFont,
	oa::U32 inCodepoint,
	oa::F32 inPixelSize) const noexcept {
	if (!impl_) return nullptr;
	const oa::U32 font = static_cast<oa::U32>(inFont);
	if (font >= kTextFontCount) return nullptr;
	const oa::I32 codepoint = codepointIndex(inCodepoint);
	const oa::I32 strike = nearestStrikeIndex(inPixelSize);
	if (codepoint < 0 || strike < 0
		|| !kTextCodepointSupport[font][static_cast<oa::U32>(codepoint)]) {
		return nullptr;
	}
	const oa::U32 index = (font * kTextStrikeCount
		+ static_cast<oa::U32>(strike)) * kTextCodepointCount
		+ static_cast<oa::U32>(codepoint);
	if (index >= kTextGlyphCount) return nullptr;
	const auto& glyph = impl_->glyphs[index];
	if (glyph.codepoint == inCodepoint) return &glyph;
	return nullptr;
}

oa::U32 oa::TextAtlas::atlasBindlessIndex(oa::FontId inFont) const noexcept {
	if (!impl_ || static_cast<oa::U32>(inFont) >= kTextFontCount) {
		return OA_BINDLESS_INVALID;
	}
	return impl_->atlas.bindlessIndex;
}

oa::GlyphBuffer::GlyphBuffer(oa::GlyphBuffer&& inOther) noexcept
	: runtime_(inOther.runtime_)
	, buffer_(inOther.buffer_)
	, consumerCompletion_(inOther.consumerCompletion_)
	, count_(inOther.count_)
	, capacity_(inOther.capacity_) {
	inOther.runtime_ = nullptr;
	inOther.buffer_ = {};
	inOther.consumerCompletion_ = {};
	inOther.count_ = 0;
	inOther.capacity_ = 0;
}

oa::GlyphBuffer& oa::GlyphBuffer::operator=(oa::GlyphBuffer&& inOther) noexcept {
	if (this != &inOther) {
		reset_();
		runtime_ = inOther.runtime_;
		buffer_ = inOther.buffer_;
		consumerCompletion_ = inOther.consumerCompletion_;
		count_ = inOther.count_;
		capacity_ = inOther.capacity_;
		inOther.runtime_ = nullptr;
		inOther.buffer_ = {};
		inOther.consumerCompletion_ = {};
		inOther.count_ = 0;
		inOther.capacity_ = 0;
	}
	return *this;
}

oa::GlyphBuffer::~GlyphBuffer() { reset_(); }

oa::Result<oa::GlyphBuffer> oa::GlyphBuffer::createHostUpload(
	oa::Engine& inRuntime,
	oa::U32 inCapacity) {
	if (inCapacity == 0) {
		return oa::Status::invalidArgument(
			"oa::GlyphBuffer::createHostUpload: capacity must be non-zero");
	}
	auto allocation = oa::EngineResourceAccess::allocBuffer(inRuntime,
		static_cast<oa::U64>(inCapacity) * sizeof(oa::GlyphInstance));
	if (!allocation.isOk()) return allocation.getStatus();
	oa::GlyphBuffer result;
	result.runtime_ = &inRuntime;
	result.buffer_ = oa::move(*allocation);
	result.capacity_ = inCapacity;
	return result;
}

void oa::GlyphBuffer::reset_() noexcept {
	oa::ConsumedBufferRetirement::releaseOrRetire(
		runtime_, buffer_, consumerCompletion_);
	runtime_ = nullptr;
	buffer_ = {};
	consumerCompletion_ = {};
	count_ = 0;
	capacity_ = 0;
}

oa::Status oa::GlyphBuffer::upload(oa::Span<const oa::GlyphInstance> inGlyphs) {
	if (!runtime_ || !buffer_.buffer || !buffer_.mappedPtr) {
		return oa::Status::error(
			oa::StatusCode::FailedPrecondition,
			"oa::GlyphBuffer::upload: buffer is not host-visible");
	}
	if (!isReady()) {
		return oa::Status::error(
			oa::StatusCode::Unavailable,
			"oa::GlyphBuffer::upload: buffer is still consumed by the GPU");
	}
	if (inGlyphs.size() > capacity_) {
		return oa::Status::error(
			oa::StatusCode::ResourceExhausted,
			"oa::GlyphBuffer::upload: glyph count exceeds capacity");
	}
	const oa::Usize bytes = inGlyphs.size() * sizeof(oa::GlyphInstance);
	if (bytes > 0) oa::memcpy(buffer_.mappedPtr, inGlyphs.data(), bytes);
	count_ = static_cast<oa::U32>(inGlyphs.size());
	consumerCompletion_ = {};
	return oa::Status::ok();
}

oa::Status oa::GlyphBuffer::markConsumed(const oa::Event& inCompletion) {
	if (runtime_ == nullptr or not isValid()) {
		return oa::Status::error(oa::StatusCode::FailedPrecondition,
			"oa::GlyphBuffer::markConsumed requires a valid buffer");
	}
	if (not runtime_->ownsEvent(inCompletion)) {
		return oa::Status::invalidArgument(
			"oa::GlyphBuffer::markConsumed requires an event from its engine");
	}
	consumerCompletion_ = inCompletion.isComplete()
		? oa::Event{}
		: inCompletion;
	return oa::Status::ok();
}

bool oa::GlyphBuffer::isReady() const {
	if (!runtime_ || !isValid()) return false;
	return not consumerCompletion_.isValid()
		or consumerCompletion_.isComplete();
}

namespace {

struct DecodedScalar {
	oa::U32 codepoint = 0xFFFDU;
	oa::U32 cluster = 0U;
};

enum class ShapedItemKind : oa::U8 {
	Glyph,
	LineBreak,
	Tab,
};

struct ShapedTextItem {
	ShapedItemKind kind = ShapedItemKind::Glyph;
	oa::FontId font = oa::FontId::Sans;
	oa::U32 codepoint = 0U;
	oa::U32 cluster = 0U;
	oa::F32 xAdvance = 0.0F;
	oa::F32 xOffset = 0.0F;
	oa::F32 yOffset = 0.0F;
};

struct RunScalar {
	oa::U32 codepoint = 0U;
	oa::U32 cluster = 0U;
};

struct ResolvedScalar {
	oa::FontId font = oa::FontId::Sans;
	oa::U32 codepoint = 0U;
};

struct HbFaceData {
	hb_face_t* face = nullptr;
	std::vector<oa::U32> codepointByGlyph;

	HbFaceData() = default;
	HbFaceData(const HbFaceData&) = delete;
	HbFaceData& operator=(const HbFaceData&) = delete;
	~HbFaceData() {
		if (face != nullptr) hb_face_destroy(face);
	}

	void init(
		const oa::U8* inBytes,
		oa::Usize inByteCount,
		oa::FontId inFont) {
		if (inBytes == nullptr || inByteCount == 0U
			|| inByteCount > std::numeric_limits<unsigned int>::max()) {
			return;
		}
		hb_blob_t* blob = hb_blob_create(
			reinterpret_cast<const char*>(inBytes),
			static_cast<unsigned int>(inByteCount),
			HB_MEMORY_MODE_READONLY,
			nullptr,
			nullptr);
		if (blob == hb_blob_get_empty()) return;
		face = hb_face_create(blob, 0U);
		hb_blob_destroy(blob);
		if (face == hb_face_get_empty()) {
			hb_face_destroy(face);
			face = nullptr;
			return;
		}

		codepointByGlyph.resize(hb_face_get_glyph_count(face), 0U);
		hb_font_t* font = hb_font_create(face);
		if (font == hb_font_get_empty()) {
			hb_font_destroy(font);
			return;
		}
		hb_ot_font_set_funcs(font);
		for (oa::U32 index = 0U; index < kTextCodepointCount; ++index) {
			if (!kTextCodepointSupport[static_cast<oa::U32>(inFont)][index]) {
				continue;
			}
			hb_codepoint_t glyph = 0U;
			if (hb_font_get_nominal_glyph(font, kTextCodepoints[index], &glyph)
				&& glyph < codepointByGlyph.size()
				&& codepointByGlyph[glyph] == 0U) {
				codepointByGlyph[glyph] = kTextCodepoints[index];
			}
		}
		hb_font_destroy(font);
	}
};

struct HbFontRegistry {
	HbFaceData faces[kTextFontCount];

	HbFontRegistry() {
		faces[static_cast<oa::U32>(oa::FontId::Sans)].init(
			kTextSansFontBytes, sizeof(kTextSansFontBytes), oa::FontId::Sans);
		faces[static_cast<oa::U32>(oa::FontId::Mono)].init(
			kTextMonoFontBytes, sizeof(kTextMonoFontBytes), oa::FontId::Mono);
	}
};

const HbFaceData& harfBuzzFace(oa::FontId inFont) {
	static const HbFontRegistry registry;
	return registry.faces[static_cast<oa::U32>(inFont)];
}

bool isCombiningMark(oa::U32 inCodepoint) noexcept {
	const utf8proc_category_t category = utf8proc_category(
		static_cast<utf8proc_int32_t>(inCodepoint));
	return category == UTF8PROC_CATEGORY_MN
		|| category == UTF8PROC_CATEGORY_MC
		|| category == UTF8PROC_CATEGORY_ME;
}

oa::FontId alternateFont(oa::FontId inFont) noexcept {
	return inFont == oa::FontId::Sans ? oa::FontId::Mono : oa::FontId::Sans;
}

ResolvedScalar resolveScalar(
	oa::FontId inPrimary,
	oa::U32 inCodepoint,
	bool inHasPrevious,
	oa::FontId inPrevious) noexcept {
	if (inHasPrevious && isCombiningMark(inCodepoint)
		&& fontSupportsCodepoint(inPrevious, inCodepoint)) {
		return {inPrevious, inCodepoint};
	}
	if (fontSupportsCodepoint(inPrimary, inCodepoint)) {
		return {inPrimary, inCodepoint};
	}
	const oa::FontId alternate = alternateFont(inPrimary);
	if (fontSupportsCodepoint(alternate, inCodepoint)) {
		return {alternate, inCodepoint};
	}
	constexpr oa::U32 replacement = static_cast<oa::U32>('?');
	if (fontSupportsCodepoint(inPrimary, replacement)) {
		return {inPrimary, replacement};
	}
	return {alternate, replacement};
}

bool isContinuation(oa::U8 inByte) noexcept {
	return (inByte & 0xC0U) == 0x80U;
}

DecodedScalar decodeUtf8(oa::StringView inText, oa::Usize& inOutIndex) noexcept {
	const oa::Usize cluster = inOutIndex;
	const oa::U8 lead = static_cast<oa::U8>(inText[inOutIndex++]);
	if (lead < 0x80U) {
		return {lead, static_cast<oa::U32>(cluster)};
	}

	oa::U32 codepoint = 0U;
	oa::U32 continuationCount = 0U;
	oa::U32 minimum = 0U;
	if (lead >= 0xC2U && lead <= 0xDFU) {
		codepoint = lead & 0x1FU;
		continuationCount = 1U;
		minimum = 0x80U;
	} else if (lead >= 0xE0U && lead <= 0xEFU) {
		codepoint = lead & 0x0FU;
		continuationCount = 2U;
		minimum = 0x800U;
	} else if (lead >= 0xF0U && lead <= 0xF4U) {
		codepoint = lead & 0x07U;
		continuationCount = 3U;
		minimum = 0x10000U;
	} else {
		return {0xFFFDU, static_cast<oa::U32>(cluster)};
	}

	if (inOutIndex + continuationCount > inText.size()) {
		return {0xFFFDU, static_cast<oa::U32>(cluster)};
	}
	for (oa::U32 index = 0U; index < continuationCount; ++index) {
		const oa::U8 byte = static_cast<oa::U8>(inText[inOutIndex + index]);
		if (!isContinuation(byte)) {
			return {0xFFFDU, static_cast<oa::U32>(cluster)};
		}
		codepoint = (codepoint << 6U) | (byte & 0x3FU);
	}
	if (codepoint < minimum || codepoint > 0x10FFFFU
		|| (codepoint >= 0xD800U && codepoint <= 0xDFFFU)) {
		return {0xFFFDU, static_cast<oa::U32>(cluster)};
	}
	inOutIndex += continuationCount;
	return {codepoint, static_cast<oa::U32>(cluster)};
}

bool appendShapedRun(
	oa::FontId inFont,
	oa::F32 inPixelSize,
	const std::vector<RunScalar>& inRun,
	std::vector<ShapedTextItem>& inOutItems) {
	if (inRun.empty()) return true;
	const HbFaceData& face = harfBuzzFace(inFont);
	if (face.face == nullptr) return false;

	hb_font_t* font = hb_font_create(face.face);
	if (font == hb_font_get_empty()) {
		hb_font_destroy(font);
		return false;
	}
	hb_ot_font_set_funcs(font);
	const oa::F64 scaled = std::round(static_cast<oa::F64>(inPixelSize) * 64.0);
	if (scaled <= 0.0
		|| scaled > static_cast<oa::F64>(std::numeric_limits<int>::max())) {
		hb_font_destroy(font);
		return false;
	}
	const int scale = static_cast<int>(scaled);
	hb_font_set_scale(font, scale, scale);
	const unsigned int ppem = static_cast<unsigned int>(std::max(
		1.0, std::round(static_cast<oa::F64>(inPixelSize))));
	hb_font_set_ppem(font, ppem, ppem);

	hb_buffer_t* buffer = hb_buffer_create();
	if (buffer == hb_buffer_get_empty()) {
		hb_buffer_destroy(buffer);
		hb_font_destroy(font);
		return false;
	}
	hb_buffer_set_cluster_level(buffer, HB_BUFFER_CLUSTER_LEVEL_MONOTONE_CHARACTERS);
	hb_buffer_set_content_type(buffer, HB_BUFFER_CONTENT_TYPE_UNICODE);
	hb_buffer_set_direction(buffer, HB_DIRECTION_LTR);
	for (const RunScalar& scalar : inRun) {
		hb_buffer_add(buffer, scalar.codepoint, scalar.cluster);
	}
	hb_buffer_guess_segment_properties(buffer);
	if (!hb_buffer_allocation_successful(buffer)) {
		hb_buffer_destroy(buffer);
		hb_font_destroy(font);
		return false;
	}
	hb_shape(font, buffer, nullptr, 0U);
	if (!hb_buffer_allocation_successful(buffer)) {
		hb_buffer_destroy(buffer);
		hb_font_destroy(font);
		return false;
	}

	unsigned int count = 0U;
	const hb_glyph_info_t* infos = hb_buffer_get_glyph_infos(buffer, &count);
	const hb_glyph_position_t* positions = hb_buffer_get_glyph_positions(
		buffer, &count);
	if (infos == nullptr || positions == nullptr) {
		hb_buffer_destroy(buffer);
		hb_font_destroy(font);
		return false;
	}
	inOutItems.reserve(inOutItems.size() + count);
	for (unsigned int index = 0U; index < count; ++index) {
		oa::U32 codepoint = infos[index].codepoint < face.codepointByGlyph.size()
			? face.codepointByGlyph[infos[index].codepoint]
			: 0U;
		if (!fontSupportsCodepoint(inFont, codepoint)) {
			codepoint = static_cast<oa::U32>('?');
		}
		inOutItems.push_back({
			.kind = ShapedItemKind::Glyph,
			.font = inFont,
			.codepoint = codepoint,
			.cluster = infos[index].cluster,
			.xAdvance = static_cast<oa::F32>(positions[index].x_advance) / 64.0F,
			.xOffset = static_cast<oa::F32>(positions[index].x_offset) / 64.0F,
			.yOffset = static_cast<oa::F32>(positions[index].y_offset) / 64.0F,
		});
	}
	hb_buffer_destroy(buffer);
	hb_font_destroy(font);
	return true;
}

bool buildShapedText(
	oa::StringView inText,
	oa::FontId inPrimaryFont,
	oa::F32 inPixelSize,
	std::vector<ShapedTextItem>& outItems) {
	if (!isValidFont(inPrimaryFont)
		|| inText.size() > std::numeric_limits<oa::U32>::max()) {
		return false;
	}
	outItems.clear();
	outItems.reserve(inText.size());
	std::vector<RunScalar> run;
	run.reserve(inText.size());
	oa::FontId runFont = inPrimaryFont;
	bool hasRun = false;
	oa::FontId previousFont = inPrimaryFont;
	bool hasPrevious = false;

	const auto flush = [&]() -> bool {
		if (!hasRun) return true;
		const bool result = appendShapedRun(
			runFont, inPixelSize, run, outItems);
		run.clear();
		hasRun = false;
		return result;
	};

	for (oa::Usize index = 0U; index < inText.size();) {
		const DecodedScalar decoded = decodeUtf8(inText, index);
		if (decoded.codepoint == '\r') continue;
		if (decoded.codepoint == '\n' || decoded.codepoint == '\t') {
			if (!flush()) return false;
			outItems.push_back({
				.kind = decoded.codepoint == '\n'
					? ShapedItemKind::LineBreak
					: ShapedItemKind::Tab,
				.cluster = decoded.cluster,
			});
			hasPrevious = false;
			continue;
		}

		const ResolvedScalar resolved = resolveScalar(
			inPrimaryFont, decoded.codepoint, hasPrevious, previousFont);
		if (hasRun && resolved.font != runFont && !flush()) return false;
		if (!hasRun) {
			runFont = resolved.font;
			hasRun = true;
		}
		run.push_back({resolved.codepoint, decoded.cluster});
		previousFont = resolved.font;
		hasPrevious = true;
	}
	return flush();
}

const GeneratedGlyph* findGeneratedGlyph(
	oa::FontId inFont,
	oa::U32& inOutCodepoint,
	oa::F32 inPixelSize) noexcept {
	const oa::U32 font = static_cast<oa::U32>(inFont);
	const oa::I32 strike = nearestStrikeIndex(inPixelSize);
	auto find = [&](oa::U32 inCodepoint) -> const GeneratedGlyph* {
		const oa::I32 codepoint = codepointIndex(inCodepoint);
		if (font >= kTextFontCount or strike < 0 or codepoint < 0) return nullptr;
		if (!kTextCodepointSupport[font][static_cast<oa::U32>(codepoint)]) {
			return nullptr;
		}
		const oa::U32 index = (font * kTextStrikeCount
			+ static_cast<oa::U32>(strike)) * kTextCodepointCount
			+ static_cast<oa::U32>(codepoint);
		return index < kTextGlyphCount ? &kTextGlyphs[index] : nullptr;
	};
	if (const GeneratedGlyph* glyph = find(inOutCodepoint)) return glyph;
	inOutCodepoint = static_cast<oa::U32>('?');
	return find(inOutCodepoint);
}

oa::F32 tabStopWidth(oa::FontId inPrimaryFont, oa::F32 inPixelSize) noexcept {
	ResolvedScalar space = resolveScalar(
		inPrimaryFont, static_cast<oa::U32>(' '), false, inPrimaryFont);
	oa::U32 codepoint = space.codepoint;
	const GeneratedGlyph* glyph = findGeneratedGlyph(
		space.font, codepoint, inPixelSize);
	if (glyph == nullptr || glyph->pixelSize == 0U) return inPixelSize * 2.0F;
	return std::max(
		1.0F,
		glyph->advance * (inPixelSize / static_cast<oa::F32>(glyph->pixelSize))
			* 4.0F);
}

oa::F32 advanceToTabStop(
	oa::F32 inX,
	oa::F32 inLineOrigin,
	oa::F32 inTabWidth) noexcept {
	const oa::F32 relative = std::max(0.0F, inX - inLineOrigin);
	return inLineOrigin
		+ (std::floor(relative / inTabWidth) + 1.0F) * inTabWidth;
}

} // namespace

void oa::TextLayout::shape(
	const oa::TextAtlas& inAtlas,
	oa::StringView inText,
	oa::vlm::Vec2 inOrigin,
	const oa::TextLayoutConfig& inCfg,
	oa::U32 inColor,
	oa::Vec<oa::PositionedGlyph>& inOutGlyphs) const {
	if (!std::isfinite(inCfg.size) || inCfg.size <= 0.0F) return;
	const oa::FontId font = inCfg.monospace ? oa::FontId::Mono : inCfg.font;
	std::vector<ShapedTextItem> items;
	if (!buildShapedText(inText, font, inCfg.size, items)) return;
	const oa::F32 tabWidth = tabStopWidth(font, inCfg.size);
	oa::F32 x = inOrigin.x;
	oa::F32 y = inOrigin.y;
	for (const ShapedTextItem& item : items) {
		if (item.kind == ShapedItemKind::LineBreak) {
			x = inOrigin.x;
			y += inCfg.size * 1.25F;
			continue;
		}
		if (item.kind == ShapedItemKind::Tab) {
			oa::F32 next = advanceToTabStop(x, inOrigin.x, tabWidth);
			if (inCfg.wrapWidth > 0.0F && x > inOrigin.x
				&& next > inOrigin.x + inCfg.wrapWidth) {
				x = inOrigin.x;
				y += inCfg.size * 1.25F;
				next = advanceToTabStop(x, inOrigin.x, tabWidth);
			}
			x = next;
			continue;
		}
		if (inCfg.wrapWidth > 0.0F && x > inOrigin.x
			&& x + item.xAdvance > inOrigin.x + inCfg.wrapWidth) {
			x = inOrigin.x;
			y += inCfg.size * 1.25F;
		}
		if (inAtlas.findGlyph(item.font, item.codepoint, inCfg.size) == nullptr) {
			x += item.xAdvance;
			continue;
		}
		inOutGlyphs.pushBack({
			.x = x + item.xOffset,
			.y = y - item.yOffset,
			.codepoint = item.codepoint,
			.cluster = item.cluster,
			.color = inColor,
			.font = item.font,
		});
		x += item.xAdvance;
	}
}

oa::vlm::Vec2 oa::TextLayout::measure(
	const oa::TextAtlas& inAtlas,
	oa::StringView inText,
	const oa::TextLayoutConfig& inCfg) const {
	if (!std::isfinite(inCfg.size) || inCfg.size <= 0.0F) return {};
	const oa::FontId font = inCfg.monospace ? oa::FontId::Mono : inCfg.font;
	std::vector<ShapedTextItem> items;
	if (!buildShapedText(inText, font, inCfg.size, items)) return {};
	const oa::F32 tabWidth = tabStopWidth(font, inCfg.size);
	oa::F32 lineWidth = 0.0F;
	oa::F32 maxWidth = 0.0F;
	oa::F32 height = inCfg.size;
	for (const ShapedTextItem& item : items) {
		if (item.kind == ShapedItemKind::LineBreak) {
			maxWidth = std::max(maxWidth, lineWidth);
			lineWidth = 0.0F;
			height += inCfg.size * 1.25F;
			continue;
		}
		if (item.kind == ShapedItemKind::Tab) {
			oa::F32 next = advanceToTabStop(lineWidth, 0.0F, tabWidth);
			if (inCfg.wrapWidth > 0.0F && lineWidth > 0.0F
				&& next > inCfg.wrapWidth) {
				maxWidth = std::max(maxWidth, lineWidth);
				lineWidth = 0.0F;
				height += inCfg.size * 1.25F;
				next = advanceToTabStop(lineWidth, 0.0F, tabWidth);
			}
			lineWidth = next;
			continue;
		}
		const oa::F32 advance = item.xAdvance;
		if (inCfg.wrapWidth > 0.0F && lineWidth > 0.0F
			&& lineWidth + advance > inCfg.wrapWidth) {
			maxWidth = std::max(maxWidth, lineWidth);
			lineWidth = 0.0F;
			height += inCfg.size * 1.25F;
		}
		lineWidth += advance;
	}
	(void)inAtlas;
	return {std::max(maxWidth, lineWidth), height};
}
