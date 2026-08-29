// Engine first — VK_NO_PROTOTYPES before any vulkan.h pull-in.
#include <oa/runtime/engine.h>
#include <oa/runtime/engine/allocatorAccess.h>
#include <oa/runtime/engine/deviceAccess.h>
#include <vkl/vkl.h>
#include <oa/runtime/pipeline.h>
#include <oa/runtime/spirv.h>
#include <oa/runtime/engine/bindlessAccess.h>
#include <oa/ui/ui.h>
#include <oa/ui/image.h>
#include <oa/ui/text.h>
#include <oa/core/log.h>
#include <oa/core/matrix.h>
#include <oa/core/matrixAccess.h>
#include <oa/vision/detection.h>
#include "../runtime/dispatchValidation.h"
#include "../runtime/engine/borrowedServiceRetirement.h"
#include "../runtime/textureAccess.h"

#include <oa/core/std/algo.h>
#include <oa/core/std/array.h>
#include <stdarg.h>
#include <oa/core/std/scalarMath.h>
#include <stdio.h>
#include <oa/core/std/cString.h>
#include <oa/core/std/limits.h>
#include <oa/core/std/vector.h>

// ─── blitRgba push constants (must match blitRgba.slang) ─────────────────────
struct BlitRgbaPc {
	oa::U32 src_idx;
	oa::U32 dst_idx;
	oa::U32 src_w;
	oa::U32 src_h;
	oa::I32 dst_x;
	oa::I32 dst_y;
	oa::U32 dst_w;
	oa::U32 dst_h;
	oa::I32 clip_x;
	oa::I32 clip_y;
	oa::U32 clip_w;
	oa::U32 clip_h;
};
static_assert(sizeof(BlitRgbaPc) == 48U);

// ─── blitPlanar push constants (must match blitPlanar.slang) ─────────────────
struct BlitPlanarPc {
	oa::U32 r_idx;
	oa::U32 g_idx;
	oa::U32 b_idx;
	oa::U32 a_idx;
	oa::U32 dst_idx;
	oa::U32 src_w;
	oa::U32 src_h;
	oa::I32 dst_x;
	oa::I32 dst_y;
	oa::U32 dtypes;
	oa::U32 channels;
	oa::U32 dst_w;
	oa::U32 dst_h;
	oa::I32 clip_x;
	oa::I32 clip_y;
	oa::U32 clip_w;
	oa::U32 clip_h;
};
static_assert(sizeof(BlitPlanarPc) == 68U);

struct BlitImageRgbaPc {
	oa::U32 src_idx;
	oa::U32 dst_idx;
	oa::U32 src_w;
	oa::U32 src_h;
	oa::I32 dst_x;
	oa::I32 dst_y;
	oa::U32 dst_w;
	oa::U32 dst_h;
	oa::I32 clip_x;
	oa::I32 clip_y;
	oa::U32 clip_w;
	oa::U32 clip_h;
};
static_assert(sizeof(BlitImageRgbaPc) == 48U);

// ─── Filled/outlined rectangle push constants ───────────────────────────────
struct DrawRectPc {
	oa::U32 dst_idx;
	oa::I32 dst_x;
	oa::I32 dst_y;
	oa::U32 dst_w;
	oa::U32 dst_h;
	oa::U32 rgba;
	oa::I32 clip_x;
	oa::I32 clip_y;
	oa::U32 clip_w;
	oa::U32 clip_h;
	oa::F32 corner_radius;
};
static_assert(sizeof(DrawRectPc) == 44U);

// Must match drawRectOutline.slang.
struct DrawRectOutlinePc {
	oa::U32 dst_idx;
	oa::I32 dst_x;
	oa::I32 dst_y;
	oa::U32 dst_w;
	oa::U32 dst_h;
	oa::U32 thickness;
	oa::U32 rgba;
	oa::I32 clip_x;
	oa::I32 clip_y;
	oa::U32 clip_w;
	oa::U32 clip_h;
	oa::F32 corner_radius;
};
static_assert(sizeof(DrawRectOutlinePc) == 48U);

struct DrawLinePc {
	oa::U32 dst_idx;
	oa::F32 x0;
	oa::F32 y0;
	oa::F32 x1;
	oa::F32 y1;
	oa::F32 thickness;
	oa::U32 rgba;
	oa::U32 bounds_x;
	oa::U32 bounds_y;
	oa::U32 bounds_w;
	oa::U32 bounds_h;
};

// Must match drawCanvasGrid.slang.
struct DrawCanvasGridPc {
	oa::U32 dst_idx;
	oa::I32 dst_x;
	oa::I32 dst_y;
	oa::U32 dst_w;
	oa::U32 dst_h;
	oa::F32 origin_x;
	oa::F32 origin_y;
	oa::F32 minor_spacing_x;
	oa::F32 minor_spacing_y;
	oa::U32 major_every;
	oa::U32 super_major_every;
	oa::F32 minor_thickness;
	oa::F32 major_thickness;
	oa::F32 super_major_thickness;
	oa::F32 axis_thickness;
	oa::U32 background_top_rgba;
	oa::U32 background_bottom_rgba;
	oa::U32 minor_rgba;
	oa::U32 major_rgba;
	oa::U32 super_major_rgba;
	oa::U32 axis_rgba;
	oa::U32 flags;
	oa::I32 clip_x;
	oa::I32 clip_y;
	oa::U32 clip_w;
	oa::U32 clip_h;
};
static_assert(sizeof(DrawCanvasGridPc) == 104U);

struct DrawWaveformPc {
	oa::U32 envelope_idx;
	oa::U32 bins;
	oa::U32 dst_idx;
	oa::I32 dst_x;
	oa::I32 dst_y;
	oa::U32 dst_w;
	oa::U32 dst_h;
	oa::F32 fraction;
	oa::U32 played_rgba;
	oa::U32 remaining_rgba;
	oa::I32 clip_x;
	oa::I32 clip_y;
	oa::U32 clip_w;
	oa::U32 clip_h;
};
static_assert(sizeof(DrawWaveformPc) == 56U);

struct DrawPlotLinePc {
	oa::U32 values_idx;
	oa::U32 count;
	oa::U32 dst_idx;
	oa::I32 dst_x;
	oa::I32 dst_y;
	oa::U32 dst_w;
	oa::U32 dst_h;
	oa::F32 x_min;
	oa::F32 x_max;
	oa::F32 y_min;
	oa::F32 y_max;
	oa::U32 rgba;
	oa::U32 flags;
	oa::U32 antialias_samples;
	oa::F32 line_width;
	oa::I32 clip_x;
	oa::I32 clip_y;
	oa::U32 clip_w;
	oa::U32 clip_h;
};
static_assert(sizeof(DrawPlotLinePc) == 76U);

struct DrawHeatmapPc {
	oa::U32 values_idx;
	oa::U32 rows;
	oa::U32 cols;
	oa::U32 dst_idx;
	oa::I32 dst_x;
	oa::I32 dst_y;
	oa::U32 dst_w;
	oa::U32 dst_h;
	oa::F32 v_min;
	oa::F32 v_max;
	oa::U32 colormap;
	oa::U32 value_type;
	oa::U32 offset_elements;
	oa::U32 flags;
	oa::I32 clip_x;
	oa::I32 clip_y;
	oa::U32 clip_w;
	oa::U32 clip_h;
};
static_assert(sizeof(DrawHeatmapPc) == 72U);

// ─── drawRectOutlines push constants (must match drawRectOutlines.slang) ─────
struct DrawRectOutlinesPc {
	oa::U32 rect_idx;
	oa::U32 count;
	oa::U32 dst_idx;
	oa::I32 dst_x;
	oa::I32 dst_y;
	oa::U32 dst_w;
	oa::U32 dst_h;
	oa::I32 clip_x;
	oa::I32 clip_y;
	oa::U32 clip_w;
	oa::U32 clip_h;
	oa::U32 thickness;
	oa::U32 rgba;
};

// ─── drawGlyphs push constants (must match drawGlyphs.slang) ────────────────
struct DrawGlyphsPc {
	oa::U32 glyph_idx;
	oa::U32 first;
	oa::U32 atlas_idx;
	oa::U32 count;
	oa::U32 batch;
	oa::U32 batch_count;
	oa::U32 dst_idx;
	oa::I32 dst_x;
	oa::I32 dst_y;
	oa::U32 dst_w;
	oa::U32 dst_h;
	oa::I32 clip_x;
	oa::I32 clip_y;
	oa::U32 clip_w;
	oa::U32 clip_h;
	oa::U32 atlas_w;
	oa::U32 atlas_h;
	oa::F32 px_range;
};
static_assert(sizeof(DrawGlyphsPc) == 72U);


// ─── Deferred blit command ────────────────────────────────────────────────────

enum class BlitKind : oa::U8 {
	Rgba,
	Planar,
	ImageRgba,
	Rect,
	RectOutline,
	Line,
	CanvasGrid,
	RectOutlines,
	Glyphs,
	Waveform,
	PlotLine,
	Heatmap,
};

struct BlitCmd {
	BlitKind kind;
	bool internalText = false;
	VkImage srcImage = VK_NULL_HANDLE;
	VkImageView srcImageView = VK_NULL_HANDLE;
	VkImageLayout srcImageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
	VkPipelineStageFlags2 srcStageMask =
		VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
	VkAccessFlags2 srcAccessMask = VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT;
	union {
		BlitRgbaPc      rgba;
		BlitPlanarPc    planar;
		BlitImageRgbaPc imageRgba;
		DrawRectPc rect;
		DrawRectOutlinePc rectOutline;
		DrawLinePc line;
		DrawCanvasGridPc canvasGrid;
		DrawRectOutlinesPc rectOutlines;
		DrawGlyphsPc glyphs;
		DrawWaveformPc waveform;
		DrawPlotLinePc plotLine;
		DrawHeatmapPc heatmap;
	};
};


namespace {

constexpr oa::U32 kWidgetHashOffset = 2166136261U;

oa::Color mixColor(oa::Color inA, oa::Color inB, oa::F32 inT) noexcept {
	const oa::F32 t = oa::clamp(inT, 0.0F, 1.0F);
	return {
		inA.r + (inB.r - inA.r) * t,
		inA.g + (inB.g - inA.g) * t,
		inA.b + (inB.b - inA.b) * t,
		inA.a + (inB.a - inA.a) * t,
	};
}

oa::Color resolveGridColor(oa::Color inRequested, oa::Color inFallback) noexcept {
	return inRequested.a <= 0.0F ? inFallback : inRequested;
}

oa::U32 resolveLineSampleCount(oa::U32 inRequested) noexcept {
	return inRequested == 1U or inRequested == 8U ? inRequested : 4U;
}

oa::UiGridConfig resolveChartGridConfig(
	oa::PixelRect inRect,
	oa::F32 inContentScale,
	bool inFillBackground,
	bool inDrawGrid) noexcept {
	const oa::F32 shortSide = static_cast<oa::F32>(oa::max(
		1, oa::min(inRect.w - 1, inRect.h - 1)));
	const oa::F32 targetMajorPixels = oa::max(1.0F,
		144.0F * inContentScale);
	const oa::I32 shortSideCells = oa::clamp(
		static_cast<oa::I32>(oa::round(shortSide / targetMajorPixels)), 4, 6);
	const oa::F32 minorSpacing = oa::max(1.0F,
		shortSide / static_cast<oa::F32>(shortSideCells * 10));
	return {
		.origin = {
			static_cast<oa::F32>(inRect.x)
				+ 0.5F * static_cast<oa::F32>(inRect.w - 1),
			static_cast<oa::F32>(inRect.y)
				+ 0.5F * static_cast<oa::F32>(inRect.h - 1),
		},
		.minorSpacing = {minorSpacing, minorSpacing},
		.opacity = 0.65F,
		.fillBackground = inFillBackground,
		.drawGrid = inDrawGrid,
		.drawAxes = inDrawGrid,
	};
}

oa::PixelRect intersectPixelRects(
	oa::PixelRect inA,
	oa::PixelRect inB) noexcept {
	if (inA.w <= 0 || inA.h <= 0 || inB.w <= 0 || inB.h <= 0) return {};
	const oa::I64 left = oa::max<oa::I64>(inA.x, inB.x);
	const oa::I64 top = oa::max<oa::I64>(inA.y, inB.y);
	const oa::I64 right = oa::min<oa::I64>(
		static_cast<oa::I64>(inA.x) + inA.w,
		static_cast<oa::I64>(inB.x) + inB.w);
	const oa::I64 bottom = oa::min<oa::I64>(
		static_cast<oa::I64>(inA.y) + inA.h,
		static_cast<oa::I64>(inB.y) + inB.h);
	if (right <= left || bottom <= top) return {};
	return {
		static_cast<oa::I32>(left),
		static_cast<oa::I32>(top),
		static_cast<oa::I32>(right - left),
		static_cast<oa::I32>(bottom - top),
	};
}

oa::PixelRect clipToNonNegative(oa::PixelRect inRect) noexcept {
	if (inRect.w <= 0 || inRect.h <= 0) return {};
	const oa::I64 right = static_cast<oa::I64>(inRect.x) + inRect.w;
	const oa::I64 bottom = static_cast<oa::I64>(inRect.y) + inRect.h;
	const oa::I64 left = oa::max<oa::I64>(0, inRect.x);
	const oa::I64 top = oa::max<oa::I64>(0, inRect.y);
	if (right <= left || bottom <= top) return {};
	return {
		static_cast<oa::I32>(left),
		static_cast<oa::I32>(top),
		static_cast<oa::I32>(oa::min<oa::I64>(
			right - left, oa::Limits<oa::I32>::max())),
		static_cast<oa::I32>(oa::min<oa::I64>(
			bottom - top, oa::Limits<oa::I32>::max())),
	};
}

oa::U32 hashWidgetId(oa::StringView inId, oa::U32 inSeed = kWidgetHashOffset) noexcept {
	oa::U32 hash = inSeed == 0U ? kWidgetHashOffset : inSeed;
	for (oa::Usize i = 0; i < inId.size(); ++i) {
		hash ^= static_cast<oa::U8>(inId[i]);
		hash *= 16777619U;
	}
	return hash == 0U ? 1U : hash;
}

oa::U32 hashWidgetScope(oa::U32 inScope, oa::StringView inId) noexcept {
	oa::U32 hash = inScope == 0U ? kWidgetHashOffset : inScope;
	hash ^= 0xFFU;
	hash *= 16777619U;
	return hashWidgetId(inId, hash);
}

oa::U32 hashWidgetIndex(oa::U32 inScope, oa::U32 inIndex) noexcept {
	oa::U32 hash = inScope == 0U ? kWidgetHashOffset : inScope;
	for (oa::U32 shift = 0U; shift < 32U; shift += 8U) {
		hash ^= (inIndex >> shift) & 0xFFU;
		hash *= 16777619U;
	}
	return hash == 0U ? 1U : hash;
}

bool isSliderAdjustmentKey(oa::UiKey inKey) noexcept {
	return inKey == oa::UiKey::Left || inKey == oa::UiKey::Down
		|| inKey == oa::UiKey::Right || inKey == oa::UiKey::Up;
}

oa::UiStyle scaleUiStyleGeometry(oa::UiStyle inStyle, oa::F32 inScale) noexcept {
	inStyle.cornerRadius *= inScale;
	inStyle.borderWidth *= inScale;
	inStyle.shadowBlur *= inScale;
	inStyle.shadowOffset *= inScale;
	inStyle.fontSize *= inScale;
	inStyle.itemSpacing *= inScale;
	inStyle.padding *= inScale;
	inStyle.framePaddingX *= inScale;
	inStyle.framePaddingY *= inScale;
	return inStyle;
}

bool isUtf8Continuation(oa::U8 inByte) noexcept {
	return (inByte & 0xC0U) == 0x80U;
}

oa::Usize utf8ScalarEnd(oa::StringView inText, oa::Usize inOffset) noexcept {
	if (inOffset >= inText.size()) return inText.size();
	const oa::U8 lead = static_cast<oa::U8>(inText[inOffset]);
	oa::Usize length = 0U;
	if (lead < 0x80U) length = 1U;
	else if (lead >= 0xC2U && lead <= 0xDFU) length = 2U;
	else if (lead >= 0xE0U && lead <= 0xEFU) length = 3U;
	else if (lead >= 0xF0U && lead <= 0xF4U) length = 4U;
	else return inOffset;
	if (length > inText.size() - inOffset) return inOffset;
	for (oa::Usize index = 1U; index < length; ++index) {
		if (!isUtf8Continuation(static_cast<oa::U8>(inText[inOffset + index]))) {
			return inOffset;
		}
	}
	oa::U32 codepoint = lead & (length == 2U ? 0x1FU : length == 3U ? 0x0FU : 0x07U);
	if (length == 1U) codepoint = lead;
	for (oa::Usize index = 1U; index < length; ++index) {
		codepoint = (codepoint << 6U)
			| (static_cast<oa::U8>(inText[inOffset + index]) & 0x3FU);
	}
	const oa::U32 minimum = length == 1U ? 0U
		: length == 2U ? 0x80U : length == 3U ? 0x800U : 0x10000U;
	if (codepoint < minimum || codepoint > 0x10FFFFU
		|| (codepoint >= 0xD800U && codepoint <= 0xDFFFU)) return inOffset;
	return inOffset + length;
}

bool isValidUtf8(oa::StringView inText) noexcept {
	for (oa::Usize offset = 0U; offset < inText.size();) {
		const oa::Usize next = utf8ScalarEnd(inText, offset);
		if (next == offset) return false;
		offset = next;
	}
	return true;
}

oa::Usize utf8Previous(oa::StringView inText, oa::Usize inOffset) noexcept {
	oa::Usize offset = oa::min(inOffset, inText.size());
	if (offset == 0U) return 0U;
	--offset;
	while (offset > 0U
		&& isUtf8Continuation(static_cast<oa::U8>(inText[offset]))) --offset;
	return offset;
}

oa::Usize utf8ByteOffsetForScalarIndex(
	oa::StringView inText,
	oa::I32 inScalarIndex) noexcept {
	if (inScalarIndex <= 0) return 0U;
	oa::Usize offset = 0U;
	for (oa::I32 index = 0; index < inScalarIndex && offset < inText.size(); ++index) {
		const oa::Usize next = utf8ScalarEnd(inText, offset);
		if (next == offset) return offset;
		offset = next;
	}
	return offset;
}

bool isUnicodeScalar(oa::U32 inCodepoint) noexcept {
	return inCodepoint <= 0x10FFFFU
		&& !(inCodepoint >= 0xD800U && inCodepoint <= 0xDFFFU);
}

void appendUtf8Scalar(oa::String& inOutText, oa::U32 inCodepoint) {
	if (!isUnicodeScalar(inCodepoint)) return;
	if (inCodepoint <= 0x7FU) {
		inOutText.pushBack(static_cast<char>(inCodepoint));
	} else if (inCodepoint <= 0x7FFU) {
		inOutText.pushBack(static_cast<char>(0xC0U | (inCodepoint >> 6U)));
		inOutText.pushBack(static_cast<char>(0x80U | (inCodepoint & 0x3FU)));
	} else if (inCodepoint <= 0xFFFFU) {
		inOutText.pushBack(static_cast<char>(0xE0U | (inCodepoint >> 12U)));
		inOutText.pushBack(static_cast<char>(0x80U | ((inCodepoint >> 6U) & 0x3FU)));
		inOutText.pushBack(static_cast<char>(0x80U | (inCodepoint & 0x3FU)));
	} else {
		inOutText.pushBack(static_cast<char>(0xF0U | (inCodepoint >> 18U)));
		inOutText.pushBack(static_cast<char>(0x80U | ((inCodepoint >> 12U) & 0x3FU)));
		inOutText.pushBack(static_cast<char>(0x80U | ((inCodepoint >> 6U) & 0x3FU)));
		inOutText.pushBack(static_cast<char>(0x80U | (inCodepoint & 0x3FU)));
	}
}

oa::String singleLineUtf8(oa::StringView inText) {
	oa::String text;
	if (!isValidUtf8(inText)) return text;
	for (oa::Usize offset = 0U; offset < inText.size();) {
		const oa::Usize next = utf8ScalarEnd(inText, offset);
		const oa::U8 first = static_cast<oa::U8>(inText[offset]);
		if (first >= 0x20U && first != 0x7FU) {
			text.append(oa::StringView(inText.data() + offset, next - offset));
		}
		offset = next;
	}
	return text;
}

oa::String singleLineCommittedText(const oa::UiEvent& inEvent) {
	oa::String text;
	if (!inEvent.text.empty()) {
		text = singleLineUtf8(inEvent.text.view());
	} else if (inEvent.codepoint >= 0x20U && inEvent.codepoint != 0x7FU) {
		appendUtf8Scalar(text, inEvent.codepoint);
	}
	return text;
}

enum class TextEditKind : oa::U8 {
	Insert,
	UpdateComposition,
	CancelComposition,
	MoveLeft,
	MoveRight,
	MoveHome,
	MoveEnd,
	Backspace,
	Delete,
	SelectAll,
	Copy,
	Cut,
	Undo,
	Redo,
};

struct TextEditEvent {
	TextEditKind kind = TextEditKind::Insert;
	bool extend = false;
	bool byWord = false;
	oa::String text;
	oa::I32 selectionStart = -1;
	oa::I32 selectionLength = -1;
};

bool isTextKey(oa::UiKey inKey) noexcept {
	return (inKey >= oa::UiKey::A && inKey <= oa::UiKey::Num0)
		|| inKey == oa::UiKey::Space || inKey == oa::UiKey::Minus
		|| inKey == oa::UiKey::Equals || inKey == oa::UiKey::Comma
		|| inKey == oa::UiKey::Period || inKey == oa::UiKey::Slash;
}

bool isSafeFloatFormat(const char* inFormat) noexcept {
	if (inFormat == nullptr || *inFormat == '\0') return false;
	oa::U32 conversions = 0U;
	for (const char* cursor = inFormat; *cursor != '\0'; ++cursor) {
		if (*cursor != '%') continue;
		++cursor;
		if (*cursor == '%') continue;
		if (*cursor == '\0' || ++conversions > 1U) return false;
		while (*cursor == '-' || *cursor == '+' || *cursor == ' '
			|| *cursor == '#' || *cursor == '0') ++cursor;
		if (*cursor == '*') return false;
		while (*cursor >= '0' && *cursor <= '9') ++cursor;
		if (*cursor == '.') {
			++cursor;
			if (*cursor == '*') return false;
			while (*cursor >= '0' && *cursor <= '9') ++cursor;
		}
		if (*cursor != 'a' && *cursor != 'A'
			&& *cursor != 'e' && *cursor != 'E'
			&& *cursor != 'f' && *cursor != 'F'
			&& *cursor != 'g' && *cursor != 'G') return false;
	}
	return conversions == 1U;
}

} // namespace


// ─── oa::Ui::Impl ────────────────────────────────────────────────────────────────

struct oa::Ui::Impl {
	oa::Engine* rt = nullptr;

	// Blit pipelines (created in initBlit).
	oa::ComputePipeline blitRgba;
	oa::ComputePipeline blitPlanar;
	oa::ComputePipeline blitImageRgba;
	oa::ComputePipeline drawRect;
	oa::ComputePipeline drawRectOutline;
	oa::ComputePipeline drawLine;
	oa::ComputePipeline drawCanvasGrid;
	oa::ComputePipeline drawRectOutlines;
	oa::ComputePipeline drawGlyphs;
	oa::ComputePipeline drawWaveform;
	oa::ComputePipeline drawPlotLine;
	oa::ComputePipeline drawHeatmap;

	static constexpr oa::U32 kPlotSlotCount = 4;
	static constexpr oa::U32 kPlotCapacity = 4096;
	struct PlotSlot {
		oavk::Buffer buffer;
		oa::Event completion;
		oa::U32 count = 0;
	};
	struct PlotCache {
		oa::U32 id = 0;
		oa::Array<PlotSlot, kPlotSlotCount> slots;
		oa::U32 lastSlot = 0;
		oa::U32 nextSlot = 0;
	};
	struct UsedPlotSlot {
		oa::U32 cache = 0;
		oa::U32 slot = 0;
	};
	oa::Vector<PlotCache> plots;
	oa::Vector<UsedPlotSlot> usedPlots;
	oa::Vector<oa::ImagePlanes*> usedImagePlanes;
	struct TextureRetention {
		oa::Event completion;
		oa::Vector<oa::SharedPtr<oavk::Buffer>> owners;
	};
	oa::Vector<oa::SharedPtr<oavk::Buffer>> pendingTextureOwners;
	oa::Vector<TextureRetention> textureRetentions;
	// exact engine-owned completions for every accepted frame submission. These
	// cover pipelines and descriptor slots in addition to per-buffer reuse.
	oa::Vector<oa::Event> frameCompletions;

	PlotSlot* uploadPlotValues(oa::U32 inId, const oa::F32* inData,
		oa::U32 inCount, oa::U32& outCacheIndex, oa::U32& outSlotIndex) {
		if (rt == nullptr || inData == nullptr || inCount == 0U) return nullptr;
		outCacheIndex = 0U;
		for (; outCacheIndex < plots.size(); ++outCacheIndex) {
			if (plots[outCacheIndex].id == inId) break;
		}
		if (outCacheIndex == plots.size()) {
			PlotCache cache;
			cache.id = inId;
			const auto releaseCache = [&] {
				for (auto& slot : cache.slots) {
					if (slot.buffer.buffer == nullptr) continue;
					oa::EngineBindlessAccess::deregisterBuffer(*rt, slot.buffer);
					oa::EngineAllocatorAccess::get(*rt).free(slot.buffer);
				}
			};
			const oa::U64 bytes = static_cast<oa::U64>(kPlotCapacity)
				* sizeof(oa::F32);
			for (auto& slot : cache.slots) {
				auto buffer = oa::EngineAllocatorAccess::get(*rt).allocHostVisible(bytes);
				if (!buffer.isOk()) {
					releaseCache();
					return nullptr;
				}
				slot.buffer = oa::move(*buffer);
				if (oa::EngineBindlessAccess::registerBuffer(*rt, slot.buffer) == OA_BINDLESS_INVALID) {
					oa::EngineAllocatorAccess::get(*rt).free(slot.buffer);
					releaseCache();
					return nullptr;
				}
			}
			plots.pushBack(oa::move(cache));
		}

		auto& cache = plots[outCacheIndex];
		outSlotIndex = kPlotSlotCount;
		for (oa::U32 offset = 0; offset < kPlotSlotCount; ++offset) {
			const oa::U32 slotIndex = (cache.nextSlot + offset) % kPlotSlotCount;
			const auto& completion = cache.slots[slotIndex].completion;
			if (!completion.isValid() || completion.isComplete()) {
				outSlotIndex = slotIndex;
				break;
			}
		}
		if (outSlotIndex < kPlotSlotCount) {
			auto& slot = cache.slots[outSlotIndex];
			slot.count = oa::min(inCount, kPlotCapacity);
			oa::memcpy(slot.buffer.mappedPtr, inData,
				static_cast<oa::Usize>(slot.count) * sizeof(oa::F32));
			if (!oa::EngineAllocatorAccess::get(*rt).flushHostBuffer(slot.buffer, 0,
				static_cast<oa::U64>(slot.count) * sizeof(oa::F32))) return nullptr;
			cache.lastSlot = outSlotIndex;
			cache.nextSlot = (outSlotIndex + 1U) % kPlotSlotCount;
		} else if (cache.slots[cache.lastSlot].count == 0U) {
			return nullptr;
		}
		outSlotIndex = cache.lastSlot;
		return &cache.slots[cache.lastSlot];
	}

	struct SampledImageSlot {
		VkImageView view = VK_NULL_HANDLE;
		VkImageLayout layout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
		oa::U32 slot = OA_BINDLESS_INVALID;
	};
	oa::Vector<SampledImageSlot> sampledImageSlots;

	// Deferred commands share one renderer. Overlay commands are collected
	// separately only to guarantee top-layer ordering, then merged immediately
	// before recording.
	oa::Vector<BlitCmd> blits;
	oa::Vector<BlitCmd> overlayBlits;
	bool recordingOverlay = false;

	// Panel stack for layout cursor.
	struct PanelState {
		oa::PixelRect rect;
		oa::PixelRect clip;
		oa::UiLayout layout;
		oa::F32 cursor = 0.0F;  // current local Y offset within panel
		oa::F32 rowX = 0.0F;
		oa::F32 rowY = 0.0F;
		oa::F32 rowHeight = 0.0F;
		oa::U32 scope = kWidgetHashOffset;
		oa::U32 rowScope = kWidgetHashOffset;
		oa::U32 nextAnonymousRow = 0U;
		bool explicitRow = false;
		bool scrollPanel = false;
		oa::PixelRect scrollViewport;
		oa::I32 scrollOffsetY = 0;
	};
	oa::Vector<PanelState> panelStack;
	struct ScrollState {
		oa::U32 id = 0U;
		oa::I32 offsetY = 0;
		oa::I32 dragGrabY = 0;
		oa::U64 lastSeenFrame = 0U;
	};
	struct ScrollRecord {
		oa::U32 id = 0U;
		oa::PixelRect viewport;
		oa::I32 maxOffsetY = 0;
		oa::I32 wheelStep = 48;
		bool overlay = false;
	};
	oa::Vector<ScrollState> scrollStates;
	oa::Vector<ScrollRecord> scrollRecords;
	oa::Vector<ScrollRecord> priorScrollRecords;

	oa::Vector<oa::U32> focusOrder;
	oa::Vector<oa::U32> priorFocusOrder;
	oa::Vector<oa::U32> popupFocusOrder;
	oa::Vector<oa::U32> priorPopupFocusOrder;
	oa::Vector<oa::U32> adjustableOrder;
	oa::Vector<oa::U32> priorAdjustableOrder;
	oa::Vector<oa::U32> treeOrder;
	oa::Vector<oa::U32> priorTreeOrder;
	oa::Vector<oa::U32> tabOrder;
	oa::Vector<oa::U32> priorTabOrder;
	oa::Vector<oa::U32> textInputOrder;
	oa::Vector<oa::U32> priorTextInputOrder;
	struct TextEditSnapshot {
		oa::String text;
		oa::Usize cursor = 0U;
		oa::Usize anchor = 0U;
	};
	struct TextEditState {
		oa::U32 id = 0U;
		oa::Usize cursor = 0U;
		oa::Usize anchor = 0U;
		oa::String lastValue;
		oa::String preedit;
		oa::Usize preeditSelectionBegin = 0U;
		oa::Usize preeditSelectionEnd = 0U;
		oa::Vector<TextEditSnapshot> undo;
		oa::Vector<TextEditSnapshot> redo;
		oa::F32 scrollX = 0.0F;
		oa::F32 blinkMs = 0.0F;
		oa::U64 lastSeenFrame = 0U;
	};
	oa::Vector<TextEditState> textEditStates;
	oa::Vector<TextEditEvent> textEditEvents;
	oa::String clipboardWrite;
	oa::PixelRect focusedTextInputRect;
	oa::Vector<oa::UiAccessibilityNode> accessibilityNodes;
	oa::U64 frameIndex = 0U;
	oa::F32 frameDeltaMs = 0.0F;
	oa::F32 contentScale = 1.0F;
	oa::PixelRect frameViewport;
	bool keyboardActivate = false;
	oa::F32 keyboardAdjust = 0.0F;
	oa::U32 keyboardTreeActivateId = 0U;
	oa::I32 keyboardTreeOpenDirection = 0;
	oa::U32 keyboardTabActivateId = 0U;
	oa::U32 keyboardTabCloseId = 0U;
	oa::U32 tabDragBarId = 0U;
	oa::U32 tabDragItemId = 0U;
	bool pointerPressClaimed = false;

	// One retained popup owner bridges event routing (which happens before
	// widget dispatch) and current-frame overlay rendering.
	oa::U32 openPopupId = 0U;
	oa::U32 popupAnchorId = 0U;
	oa::PixelRect popupAnchorRect;
	oa::PixelRect popupRect;
	oa::U32 renderingPopupId = 0U;
	oa::Usize popupPanelDepth = 0U;
	bool popupRenderedThisFrame = false;
	bool popupOpenedThisFrame = false;
	bool popupDismissedThisFrame = false;
	bool popupFocusMovedThisFrame = false;

	// The immediately preceding interactive item is the anchor for convenience
	// popups and tooltips.
	oa::U32 lastItemId = 0U;
	oa::PixelRect lastItemRect;
	bool lastItemHovered = false;
	oa::U32 tooltipOwnerId = 0U;
	oa::U64 tooltipLastSeenFrame = 0U;
	oa::F32 tooltipHoverMs = 0.0F;

	// dynamic widget text uses one frame-wide CPU batch and four host-visible
	// GPU slots. Per-label draw ranges preserve command order without one buffer
	// allocation per label. slots are recycled only after timeline completion.
	static constexpr oa::U32 kTextSlotCount = 4;
	static constexpr oa::U32 kTextGlyphCapacity = 16384;
	const oa::TextAtlas* textAtlas = nullptr;
	oa::Array<oa::GlyphBuffer, kTextSlotCount> textSlots;
	oa::Vector<oa::GlyphInstance> textGlyphs;
	oa::Vector<oa::PositionedGlyph> textScratch;
	oa::U32 nextTextSlot = 0U;
	oa::U32 usedTextSlot = kTextSlotCount;
	bool textPrepared = false;
	bool frameSealed = false;
	oa::Status frameStatus;

	// Cached style stack.
	static constexpr oa::U32 kStyleDepth = 32;
	oa::UiStyle styleStack[kStyleDepth];
	oa::U32    styleDepth = 0;
	oa::UiStyle baseStyle;
	oa::UiStyle defaultStyle;

	void setFrameError(oa::Status inStatus) {
		if (frameStatus.isOk()) frameStatus = oa::move(inStatus);
	}

	void appendBlit(BlitCmd inCommand) {
		if (recordingOverlay) overlayBlits.pushBack(oa::move(inCommand));
		else blits.pushBack(oa::move(inCommand));
	}

	void mergeOverlayBlits() {
		for (BlitCmd& command : overlayBlits) {
			blits.pushBack(oa::move(command));
		}
		overlayBlits.clear();
	}

	[[nodiscard]] bool canInteract(oa::U32 inId) const noexcept {
		if (popupDismissedThisFrame) return false;
		return openPopupId == 0U || recordingOverlay || inId == popupAnchorId;
	}

	void openPopup(
		oa::U32 inPopupId,
		oa::U32 inAnchorId,
		oa::PixelRect inAnchorRect) noexcept {
		openPopupId = inPopupId;
		popupAnchorId = inAnchorId;
		popupAnchorRect = inAnchorRect;
		popupRect = {};
		popupRenderedThisFrame = false;
		popupOpenedThisFrame = true;
		popupDismissedThisFrame = false;
		popupFocusMovedThisFrame = false;
	}

	void closePopup(oa::UiInputState& inInput, bool inRestoreAnchor) noexcept {
		if (openPopupId == 0U) return;
		const oa::U32 anchorId = popupAnchorId;
		openPopupId = 0U;
		popupAnchorId = 0U;
		popupAnchorRect = {};
		popupRect = {};
		popupOpenedThisFrame = false;
		inInput.activeId = 0U;
		if (inRestoreAnchor && anchorId != 0U) inInput.focusId = anchorId;
	}

	[[nodiscard]] oa::PixelRect placePopup(
		oa::PixelRect inAnchor,
		const oa::UiPopupConfig& inConfig) const noexcept {
		if (frameViewport.w <= 0 || frameViewport.h <= 0) return {};
		const oa::I64 margin = oa::min<oa::I32>(4,
			oa::min(frameViewport.w, frameViewport.h) / 2);
		const oa::I64 left = static_cast<oa::I64>(frameViewport.x) + margin;
		const oa::I64 top = static_cast<oa::I64>(frameViewport.y) + margin;
		const oa::I64 right = static_cast<oa::I64>(frameViewport.x)
			+ frameViewport.w - margin;
		const oa::I64 bottom = static_cast<oa::I64>(frameViewport.y)
			+ frameViewport.h - margin;
		const oa::I64 width = oa::clamp<oa::I64>(
			oa::max(inAnchor.w, inConfig.width), 1, oa::max<oa::I64>(1, right - left));
		const oa::I64 height = oa::clamp<oa::I64>(
			inConfig.height, 1, oa::max<oa::I64>(1, bottom - top));
		const oa::I64 x = oa::clamp<oa::I64>(
			inAnchor.x, left, oa::max(left, right - width));
		const oa::I64 below = static_cast<oa::I64>(inAnchor.y) + inAnchor.h
			+ inConfig.gap;
		const oa::I64 above = static_cast<oa::I64>(inAnchor.y) - inConfig.gap - height;
		oa::I64 y = 0;
		if (below + height <= bottom) {
			y = below;
		} else if (above >= top) {
			y = above;
		} else {
			y = oa::clamp(below, top, oa::max(top, bottom - height));
		}
		return {
			static_cast<oa::I32>(oa::clamp<oa::I64>(
				x, oa::Limits<oa::I32>::min(),
				oa::Limits<oa::I32>::max())),
			static_cast<oa::I32>(oa::clamp<oa::I64>(
				y, oa::Limits<oa::I32>::min(),
				oa::Limits<oa::I32>::max())),
			static_cast<oa::I32>(width),
			static_cast<oa::I32>(height),
		};
	}

	[[nodiscard]] oa::PixelRect clipFor(oa::PixelRect inRect) const noexcept {
		oa::PixelRect clip = clipToNonNegative(inRect);
		if (!panelStack.empty()) {
			clip = intersectPixelRects(clip, panelStack.back().clip);
		}
		return clip;
	}

	ScrollState& getScrollState(oa::U32 inId) {
		for (ScrollState& state : scrollStates) {
			if (state.id == inId) {
				state.lastSeenFrame = frameIndex;
				return state;
			}
		}
		scrollStates.pushBack({
			.id = inId,
			.lastSeenFrame = frameIndex,
		});
		return scrollStates.back();
	}

	ScrollState* findScrollState(oa::U32 inId) noexcept {
		for (ScrollState& state : scrollStates) {
			if (state.id == inId) return &state;
		}
		return nullptr;
	}

	[[nodiscard]] oa::U32 currentScope() const noexcept {
		if (panelStack.empty()) return kWidgetHashOffset;
		const auto& panel = panelStack.back();
		return panel.explicitRow ? panel.rowScope : panel.scope;
	}

	void addAccessibilityNode(
		oa::U32 inId,
		oa::UiAccessibilityRole inRole,
		oa::PixelRect inBounds,
		oa::StringView inLabel,
		oa::StringView inValue,
		oa::UiAccessibilityState inState,
		oa::UiAccessibilityAction inActions,
		bool inFocused,
		bool inHasNumericValue = false,
		oa::F64 inMinimum = 0.0,
		oa::F64 inMaximum = 0.0,
		oa::F64 inCurrent = 0.0) {
		const oa::PixelRect bounds = clipFor(inBounds);
		if (inId == 0U || bounds.w <= 0 || bounds.h <= 0) return;
		if (inFocused) {
			inState = inState | oa::UiAccessibilityState::Focused;
		}
		accessibilityNodes.pushBack({
			.id = inId,
			.scope = currentScope(),
			.role = inRole,
			.state = inState,
			.actions = inActions,
			.bounds = bounds,
			.label = oa::String(inLabel),
			.value = oa::String(inValue),
			.minimum = inMinimum,
			.maximum = inMaximum,
			.current = inCurrent,
			.hasNumericValue = inHasNumericValue,
		});
	}

	[[nodiscard]] bool isRow(const PanelState& inPanel) const noexcept {
		return inPanel.layout.direction == oa::UiDirection::Row || inPanel.explicitRow;
	}

	[[nodiscard]] oa::vlm::Vec2 measureText(
		oa::StringView inText,
		const oa::UiStyle& inStyle) {
		if (!oa::isFinite(inStyle.fontSize) || inStyle.fontSize <= 0.0F) {
			setFrameError(oa::Status::invalidArgument(
				"oa::Ui text requires a finite positive font size"));
			return {};
		}
		if (textAtlas == nullptr) {
			setFrameError(oa::Status::error(
				oa::StatusCode::FailedPrecondition,
				"oa::Ui text requires bindTextAtlas before the first frame"));
			return {};
		}
		oa::TextLayout layout;
		return layout.measure(
			*textAtlas,
			inText,
			{.font = oa::FontId::Sans, .size = inStyle.fontSize});
	}

	[[nodiscard]] oa::PixelRect placeItem(oa::F32 inHugWidth, oa::F32 inHeight) {
		if (frameSealed) {
			setFrameError(oa::Status::error(
				oa::StatusCode::FailedPrecondition,
				"oa::Ui widgets cannot be appended after recordRender"));
			return {};
		}
		if (panelStack.empty()) {
			setFrameError(oa::Status::error(
				oa::StatusCode::FailedPrecondition,
				"oa::Ui widgets require an active panel"));
			return {};
		}
		if (!oa::isFinite(inHugWidth) || !oa::isFinite(inHeight)
			|| inHugWidth <= 0.0F || inHeight <= 0.0F) {
			setFrameError(oa::Status::invalidArgument(
				"oa::Ui widget dimensions must be finite and positive"));
			return {};
		}

		auto& panel = panelStack.back();
		const bool row = isRow(panel);
		const oa::F32 left = panel.layout.padding.left;
		const oa::F32 right = oa::max(
			left,
			static_cast<oa::F32>(panel.rect.w) - panel.layout.padding.right);
		const oa::F32 bottom = oa::max(
			panel.layout.padding.top,
			static_cast<oa::F32>(panel.rect.h) - panel.layout.padding.bottom);
		const oa::F32 localX = row ? panel.rowX : left;
		const oa::F32 localY = row ? panel.rowY : panel.cursor;
		const oa::F32 desiredWidth = row ? inHugWidth : right - left;
		const oa::F32 width = oa::max(0.0F, oa::min(desiredWidth, right - localX));
		const oa::F32 height = oa::max(0.0F, oa::min(inHeight, bottom - localY));

		if (row) {
			panel.rowX = localX + inHugWidth + panel.layout.gap;
			panel.rowHeight = oa::max(panel.rowHeight, inHeight);
		} else {
			panel.cursor = localY + inHeight + panel.layout.gap;
		}

		return {
			panel.rect.x + static_cast<oa::I32>(oa::floor(localX + 0.5F)),
			panel.rect.y + static_cast<oa::I32>(oa::floor(localY + 0.5F)),
			static_cast<oa::I32>(oa::floor(width + 0.5F)),
			static_cast<oa::I32>(oa::floor(height + 0.5F)),
		};
	}

	void registerFocusable(oa::U32 inId) {
		for (const oa::U32 existing : focusOrder) {
			if (existing == inId) return;
		}
		focusOrder.pushBack(inId);
		if (recordingOverlay && renderingPopupId != 0U) {
			for (const oa::U32 existing : popupFocusOrder) {
				if (existing == inId) return;
			}
			popupFocusOrder.pushBack(inId);
		}
	}

	void registerAdjustable(oa::U32 inId) {
		registerFocusable(inId);
		for (const oa::U32 existing : adjustableOrder) {
			if (existing == inId) return;
		}
		adjustableOrder.pushBack(inId);
	}

	void registerTree(oa::U32 inId) {
		registerFocusable(inId);
		for (const oa::U32 existing : treeOrder) {
			if (existing == inId) return;
		}
		treeOrder.pushBack(inId);
	}

	void registerTab(oa::U32 inId, bool inFocusable) {
		if (inFocusable) registerFocusable(inId);
		for (const oa::U32 existing : tabOrder) {
			if (existing == inId) return;
		}
		tabOrder.pushBack(inId);
	}

	void registerTextInput(oa::U32 inId) {
		registerFocusable(inId);
		for (const oa::U32 existing : textInputOrder) {
			if (existing == inId) return;
		}
		textInputOrder.pushBack(inId);
	}

	[[nodiscard]] bool wasAdjustable(oa::U32 inId) const noexcept {
		if (inId == 0U) return false;
		for (const oa::U32 id : priorAdjustableOrder) {
			if (id == inId) return true;
		}
		return false;
	}

	[[nodiscard]] bool wasTree(oa::U32 inId) const noexcept {
		if (inId == 0U) return false;
		for (const oa::U32 id : priorTreeOrder) {
			if (id == inId) return true;
		}
		return false;
	}

	[[nodiscard]] bool wasTab(oa::U32 inId) const noexcept {
		if (inId == 0U) return false;
		for (const oa::U32 id : priorTabOrder) {
			if (id == inId) return true;
		}
		return false;
	}

	[[nodiscard]] bool wasTextInput(oa::U32 inId) const noexcept {
		if (inId == 0U) return false;
		for (const oa::U32 id : priorTextInputOrder) {
			if (id == inId) return true;
		}
		return false;
	}

	[[nodiscard]] bool isCurrentTextInput(oa::U32 inId) const noexcept {
		if (inId == 0U) return false;
		for (const oa::U32 id : textInputOrder) {
			if (id == inId) return true;
		}
		return false;
	}

	TextEditState* findTextEditState(oa::U32 inId) noexcept {
		for (TextEditState& state : textEditStates) {
			if (state.id == inId) return &state;
		}
		return nullptr;
	}

	TextEditState& getTextEditState(oa::U32 inId, oa::StringView inInitialText) {
		for (TextEditState& state : textEditStates) {
			if (state.id == inId) {
				state.lastSeenFrame = frameIndex;
				return state;
			}
		}
		TextEditState state;
		state.id = inId;
		state.cursor = inInitialText.size();
		state.anchor = inInitialText.size();
		state.lastValue = oa::String(inInitialText);
		state.lastSeenFrame = frameIndex;
		textEditStates.pushBack(oa::move(state));
		return textEditStates.back();
	}

	struct ControlInteraction {
		bool hovered = false;
		bool held = false;
		bool activated = false;
	};

	[[nodiscard]] ControlInteraction interact(
		oa::UiInputState& inInput,
		oa::U32 inId,
		oa::PixelRect inRect) {
		registerFocusable(inId);
		ControlInteraction result;
		if (!canInteract(inId)) {
			lastItemId = inId;
			lastItemRect = inRect;
			lastItemHovered = false;
			return result;
		}
		const oa::PixelRect hitRect = clipFor(inRect);
		result.hovered = hitRect.contains(inInput.mouseX, inInput.mouseY);
		if (result.hovered) inInput.hoverId = inId;
		if (result.hovered && inInput.lPressed) {
			inInput.activeId = inId;
			inInput.focusId = inId;
			pointerPressClaimed = true;
		}
		if (inInput.activeId == inId && inInput.lReleased) {
			result.activated = result.hovered;
			inInput.activeId = 0U;
		}
		if (inInput.focusId == inId && keyboardActivate) {
			result.activated = true;
			keyboardActivate = false;
		}
		result.held = inInput.activeId == inId && inInput.lButton;
		lastItemId = inId;
		lastItemRect = inRect;
		lastItemHovered = result.hovered;
		return result;
	}

	void advanceFocus(oa::UiInputState& inInput, bool inReverse) const {
		const oa::Vector<oa::U32>& order = openPopupId != 0U
			? priorPopupFocusOrder : priorFocusOrder;
		if (order.empty()) return;
		oa::Usize index = order.size();
		for (oa::Usize i = 0U; i < order.size(); ++i) {
			if (order[i] == inInput.focusId) {
				index = i;
				break;
			}
		}
		if (index == order.size()) {
			index = inReverse ? order.size() - 1U : 0U;
		} else if (inReverse) {
			index = index == 0U ? order.size() - 1U : index - 1U;
		} else {
			index = (index + 1U) % order.size();
		}
		inInput.focusId = order[index];
	}

	[[nodiscard]] oa::U32 advanceTreeFocus(
		oa::UiInputState& inInput,
		bool inReverse,
		bool inToEdge = false) const {
		if (priorTreeOrder.empty()) return 0U;
		oa::Usize index = priorTreeOrder.size();
		for (oa::Usize i = 0U; i < priorTreeOrder.size(); ++i) {
			if (priorTreeOrder[i] == inInput.focusId) {
				index = i;
				break;
			}
		}
		if (inToEdge) {
			index = inReverse ? 0U : priorTreeOrder.size() - 1U;
		} else if (index == priorTreeOrder.size()) {
			index = inReverse ? priorTreeOrder.size() - 1U : 0U;
		} else if (inReverse) {
			index = index == 0U ? 0U : index - 1U;
		} else {
			index = oa::min(index + 1U, priorTreeOrder.size() - 1U);
		}
		inInput.focusId = priorTreeOrder[index];
		return inInput.focusId;
	}

	[[nodiscard]] oa::U32 advanceTabFocus(
		oa::UiInputState& inInput,
		bool inReverse,
		bool inToEdge = false) const {
		if (priorTabOrder.empty()) return 0U;
		oa::Usize index = priorTabOrder.size();
		for (oa::Usize i = 0U; i < priorTabOrder.size(); ++i) {
			if (priorTabOrder[i] == inInput.focusId) {
				index = i;
				break;
			}
		}
		if (inToEdge) {
			index = inReverse ? 0U : priorTabOrder.size() - 1U;
		} else if (index == priorTabOrder.size()) {
			index = inReverse ? priorTabOrder.size() - 1U : 0U;
		} else if (inReverse) {
			index = index == 0U ? priorTabOrder.size() - 1U : index - 1U;
		} else {
			index = (index + 1U) % priorTabOrder.size();
		}
		inInput.focusId = priorTabOrder[index];
		return inInput.focusId;
	}

	void appendText(oa::StringView inText, const oa::UiStyle& inStyle, bool inWrap) {
		if (inText.empty()) return;
		if (frameSealed) {
			setFrameError(oa::Status::error(
				oa::StatusCode::FailedPrecondition,
				"oa::Ui text cannot be appended after recordRender"));
			return;
		}
		if (!oa::isFinite(inStyle.fontSize) || inStyle.fontSize <= 0.0F) {
			setFrameError(oa::Status::invalidArgument(
				"oa::Ui text requires a finite positive font size"));
			return;
		}
		if (textAtlas == nullptr) {
			setFrameError(oa::Status::error(
				oa::StatusCode::FailedPrecondition,
				"oa::Ui text requires bindTextAtlas before the first frame"));
			return;
		}
		if (panelStack.empty()) {
			setFrameError(oa::Status::error(
				oa::StatusCode::FailedPrecondition,
				"oa::Ui text requires an active panel"));
			return;
		}

		auto& panel = panelStack.back();
		if (panel.rect.w <= 0 || panel.rect.h <= 0
			|| panel.clip.w <= 0 || panel.clip.h <= 0) return;
		const oa::F32 innerWidth = oa::max(
			0.0F,
			static_cast<oa::F32>(panel.rect.w)
				- panel.layout.padding.left - panel.layout.padding.right);
		if (innerWidth <= 0.0F) return;
		const bool row = isRow(panel);
		const oa::F32 localX = row ? panel.rowX : panel.layout.padding.left;
		const oa::F32 localY = row ? panel.rowY : panel.cursor;
		const oa::F32 availableWidth = oa::max(
			0.0F,
			static_cast<oa::F32>(panel.rect.w)
				- panel.layout.padding.right - localX);
		if (availableWidth <= 0.0F) return;

		const oa::TextLayoutConfig config{
			.font = oa::FontId::Sans,
			.size = inStyle.fontSize,
			.wrapWidth = inWrap ? availableWidth : 0.0F,
		};
		oa::TextLayout layout;
		textScratch.clear();
		layout.shape(
			*textAtlas,
			inText,
			{localX, localY + inStyle.fontSize},
			config,
			inStyle.text.toU32(),
			textScratch);
		const oa::vlm::Vec2 extent = layout.measure(*textAtlas, inText, config);
		const auto advance = [&] {
			if (row) {
				panel.rowX = localX + extent.x + panel.layout.gap;
				panel.rowHeight = oa::max(panel.rowHeight, extent.y);
			} else {
				panel.cursor = localY + extent.y + inStyle.itemSpacing;
			}
		};
		if (textScratch.empty()) {
			advance();
			return;
		}
		if (textGlyphs.size() + textScratch.size() > kTextGlyphCapacity) {
			setFrameError(oa::Status::error(
				oa::StatusCode::ResourceExhausted,
				"oa::Ui frame text exceeds the 16384-glyph capacity"));
			return;
		}

		const oa::U32 first = static_cast<oa::U32>(textGlyphs.size());
		for (const oa::PositionedGlyph& item : textScratch) {
			const oa::GlyphInfo* glyph = textAtlas->findGlyph(
				item.font, item.codepoint, inStyle.fontSize);
			if (glyph == nullptr || glyph->rasterSize <= 0.0F) continue;
			const oa::F32 scale = inStyle.fontSize / glyph->rasterSize;
			textGlyphs.pushBack({
				.anchorX = 0.0F,
				.anchorY = 0.0F,
				.offsetX = item.x + glyph->bearingX * scale,
				.offsetY = item.y - glyph->bearingY * scale,
				.width = glyph->atlasW * scale,
				.height = glyph->atlasH * scale,
				.atlasX = static_cast<oa::U32>(glyph->atlasX),
				.atlasY = static_cast<oa::U32>(glyph->atlasY),
				.atlasW = static_cast<oa::U32>(glyph->atlasW),
				.atlasH = static_cast<oa::U32>(glyph->atlasH),
				.color = item.color,
			});
		}
		const oa::U32 count = static_cast<oa::U32>(textGlyphs.size()) - first;
		if (count == 0U) {
			advance();
			return;
		}

		BlitCmd command{};
		command.kind = BlitKind::Glyphs;
		command.internalText = true;
		command.glyphs.glyph_idx = OA_BINDLESS_INVALID;
		command.glyphs.first = first;
		command.glyphs.atlas_idx = textAtlas->atlasBindlessIndex(oa::FontId::Sans);
		command.glyphs.count = count;
		command.glyphs.batch = 0U;
		command.glyphs.batch_count = 1U;
		command.glyphs.dst_idx = 0U;
		command.glyphs.dst_x = panel.rect.x;
		command.glyphs.dst_y = panel.rect.y;
		command.glyphs.dst_w = static_cast<oa::U32>(panel.rect.w);
		command.glyphs.dst_h = static_cast<oa::U32>(panel.rect.h);
		command.glyphs.clip_x = panel.clip.x;
		command.glyphs.clip_y = panel.clip.y;
		command.glyphs.clip_w = static_cast<oa::U32>(panel.clip.w);
		command.glyphs.clip_h = static_cast<oa::U32>(panel.clip.h);
		command.glyphs.atlas_w = static_cast<oa::U32>(textAtlas->atlasWidth());
		command.glyphs.atlas_h = static_cast<oa::U32>(textAtlas->atlasHeight());
		command.glyphs.px_range = textAtlas->pxRange();
		appendBlit(oa::move(command));

		advance();
	}

	void appendTextAt(
		oa::StringView inText,
		oa::PixelRect inRect,
		const oa::UiTextConfig& inConfig) {
		if (inText.empty() or inRect.w <= 0 or inRect.h <= 0) return;
		if (frameSealed) {
			setFrameError(oa::Status::error(
				oa::StatusCode::FailedPrecondition,
				"oa::Ui positioned text cannot be appended after recordRender"));
			return;
		}
		const auto validAlign = [](oa::UiAlign inAlign) {
			return inAlign == oa::UiAlign::Start
				or inAlign == oa::UiAlign::Center
				or inAlign == oa::UiAlign::End;
		};
		const bool validDirection =
			inConfig.direction == oa::UiTextDirection::LeftToRight
			or inConfig.direction == oa::UiTextDirection::BottomToTop;
		if (not oa::isFinite(inConfig.fontSize)
			or inConfig.fontSize <= 0.0F
			or not oa::isFinite(inConfig.color.r)
			or not oa::isFinite(inConfig.color.g)
			or not oa::isFinite(inConfig.color.b)
			or not oa::isFinite(inConfig.color.a)
			or not validAlign(inConfig.horizontalAlign)
			or not validAlign(inConfig.verticalAlign)
			or not validDirection) {
			setFrameError(oa::Status::invalidArgument(
				"oa::Ui positioned text requires finite style, explicit alignment, and a supported direction"));
			return;
		}
		if (textAtlas == nullptr) {
			setFrameError(oa::Status::error(
				oa::StatusCode::FailedPrecondition,
				"oa::Ui positioned text requires bindTextAtlas before the first frame"));
			return;
		}
		const oa::PixelRect clip = intersectPixelRects(
			clipToNonNegative(inRect), frameViewport);
		if (clip.w <= 0 or clip.h <= 0) return;

		const oa::TextLayoutConfig textConfig{
			.font = oa::FontId::Sans,
			.size = inConfig.fontSize,
		};
		oa::TextLayout layout;
		const oa::vlm::Vec2 extent = layout.measure(*textAtlas, inText, textConfig);
		if (not oa::isFinite(extent.x) or not oa::isFinite(extent.y)
			or extent.x <= 0.0F or extent.y <= 0.0F) {
			setFrameError(oa::Status::invalidArgument(
				"oa::Ui positioned text produced an invalid measured extent"));
			return;
		}
		auto align = [](oa::UiAlign inAlign, oa::F32 inBegin,
			oa::F32 inAvailable, oa::F32 inExtent) {
			switch (inAlign) {
				case oa::UiAlign::Start: return inBegin;
				case oa::UiAlign::Center:
					return inBegin + (inAvailable - inExtent) * 0.5F;
				case oa::UiAlign::End:
					return inBegin + inAvailable - inExtent;
				case oa::UiAlign::Stretch: break;
			}
			return inBegin;
		};

		textScratch.clear();
		oa::F32 verticalBaselineX = 0.0F;
		oa::F32 verticalBaselineY = 0.0F;
		if (inConfig.direction == oa::UiTextDirection::LeftToRight) {
			const oa::F32 originX = align(
				inConfig.horizontalAlign,
				static_cast<oa::F32>(inRect.x),
				static_cast<oa::F32>(inRect.w), extent.x);
			const oa::F32 top = align(
				inConfig.verticalAlign,
				static_cast<oa::F32>(inRect.y),
				static_cast<oa::F32>(inRect.h), extent.y);
			layout.shape(
				*textAtlas, inText,
				{originX, top + inConfig.fontSize * 0.80F},
				textConfig, inConfig.color.toU32(), textScratch);
		} else {
			const oa::F32 cross = align(
				inConfig.horizontalAlign,
				static_cast<oa::F32>(inRect.x),
				static_cast<oa::F32>(inRect.w), extent.y);
			const oa::F32 along = align(
				inConfig.verticalAlign,
				static_cast<oa::F32>(inRect.y),
				static_cast<oa::F32>(inRect.h), extent.x);
			verticalBaselineX = cross + inConfig.fontSize;
			verticalBaselineY = along + extent.x;
			layout.shape(
				*textAtlas, inText, {}, textConfig,
				inConfig.color.toU32(), textScratch);
		}
		if (textScratch.empty()) return;
		if (textGlyphs.size() > kTextGlyphCapacity
			or textScratch.size() > kTextGlyphCapacity - textGlyphs.size()) {
			setFrameError(oa::Status::error(
				oa::StatusCode::ResourceExhausted,
				"oa::Ui frame text exceeds the 16384-glyph capacity"));
			return;
		}

		const oa::U32 first = static_cast<oa::U32>(textGlyphs.size());
		for (const oa::PositionedGlyph& item : textScratch) {
			const oa::GlyphInfo* glyph = textAtlas->findGlyph(
				item.font, item.codepoint, inConfig.fontSize);
			if (glyph == nullptr or glyph->rasterSize <= 0.0F) continue;
			const oa::F32 scale = inConfig.fontSize / glyph->rasterSize;
			const oa::F32 width = glyph->atlasW * scale;
			const oa::F32 height = glyph->atlasH * scale;
			oa::F32 offsetX = item.x + glyph->bearingX * scale;
			oa::F32 offsetY = item.y - glyph->bearingY * scale;
			oa::U32 flags = 0U;
			if (inConfig.direction == oa::UiTextDirection::BottomToTop) {
				const oa::F32 originX = offsetX;
				const oa::F32 originY = offsetY;
				offsetX = verticalBaselineX + originY;
				offsetY = verticalBaselineY - originX - width;
				flags = 1U;
			}
			textGlyphs.pushBack({
				.anchorX = 0.0F,
				.anchorY = 0.0F,
				.offsetX = offsetX,
				.offsetY = offsetY,
				.width = width,
				.height = height,
				.atlasX = static_cast<oa::U32>(glyph->atlasX),
				.atlasY = static_cast<oa::U32>(glyph->atlasY),
				.atlasW = static_cast<oa::U32>(glyph->atlasW),
				.atlasH = static_cast<oa::U32>(glyph->atlasH),
				.color = item.color,
				.flags = flags,
			});
		}
		const oa::U32 count = static_cast<oa::U32>(textGlyphs.size()) - first;
		if (count == 0U) return;
		BlitCmd command{};
		command.kind = BlitKind::Glyphs;
		command.internalText = true;
		command.glyphs.glyph_idx = OA_BINDLESS_INVALID;
		command.glyphs.first = first;
		command.glyphs.atlas_idx =
			textAtlas->atlasBindlessIndex(oa::FontId::Sans);
		command.glyphs.count = count;
		command.glyphs.batch = 0U;
		command.glyphs.batch_count = 1U;
		command.glyphs.dst_idx = 0U;
		command.glyphs.dst_x = 0;
		command.glyphs.dst_y = 0;
		command.glyphs.dst_w = 1U;
		command.glyphs.dst_h = 1U;
		command.glyphs.clip_x = clip.x;
		command.glyphs.clip_y = clip.y;
		command.glyphs.clip_w = static_cast<oa::U32>(clip.w);
		command.glyphs.clip_h = static_cast<oa::U32>(clip.h);
		command.glyphs.atlas_w = static_cast<oa::U32>(textAtlas->atlasWidth());
		command.glyphs.atlas_h = static_cast<oa::U32>(textAtlas->atlasHeight());
		command.glyphs.px_range = textAtlas->pxRange();
		appendBlit(oa::move(command));
	}

	oa::Status prepareText() {
		if (textPrepared || textGlyphs.empty()) return oa::Status::ok();
		// Glyph quads in one label can overlap at hinted cell boundaries. One
		// workgroup per glyph would then race on the storage-image read/modify/write.
		// Greedily color each label's overlap graph and submit its non-overlapping
		// batches in order. Every glyph in one command has the same color, so the
		// source-over blend is commutative across batches while remaining exact.
		for (BlitCmd& command : blits) {
			if (command.kind != BlitKind::Glyphs || !command.internalText
				|| command.glyphs.count == 0U) continue;
			oa::Vector<oa::PixelRect> bounds(command.glyphs.count);
			oa::Vector<oa::U32> batches(command.glyphs.count);
			oa::U32 batchCount = 1U;
			for (oa::U32 local = 0U; local < command.glyphs.count; ++local) {
				oa::GlyphInstance& glyph = textGlyphs[
					static_cast<oa::Usize>(command.glyphs.first + local)];
				const bool bottomToTop = (glyph.flags & 1U) != 0U;
				const oa::F32 width = bottomToTop ? glyph.height : glyph.width;
				const oa::F32 height = bottomToTop ? glyph.width : glyph.height;
				const oa::I32 x = static_cast<oa::I32>(oa::floor(
					static_cast<oa::F32>(command.glyphs.dst_x)
					+ glyph.anchorX * static_cast<oa::F32>(command.glyphs.dst_w)
					+ glyph.offsetX));
				const oa::I32 y = static_cast<oa::I32>(oa::floor(
					static_cast<oa::F32>(command.glyphs.dst_y)
					+ glyph.anchorY * static_cast<oa::F32>(command.glyphs.dst_h)
					+ glyph.offsetY));
				bounds[local] = {
					x,
					y,
					oa::max(1, static_cast<oa::I32>(oa::ceil(width))),
					oa::max(1, static_cast<oa::I32>(oa::ceil(height))),
				};
				oa::U32 batch = 0U;
				for (; batch < local + 1U; ++batch) {
					bool overlaps = false;
					for (oa::U32 previous = 0U; previous < local; ++previous) {
						if (batches[previous] == batch
							&& bounds[local].intersects(bounds[previous])) {
							overlaps = true;
							break;
						}
					}
					if (!overlaps) break;
				}
				batches[local] = batch;
				batchCount = oa::max(batchCount, batch + 1U);
				glyph.flags = (glyph.flags & 1U) | (batch << 1U);
			}
			command.glyphs.batch = 0U;
			command.glyphs.batch_count = batchCount;
		}
		oa::U32 slotIndex = kTextSlotCount;
		for (oa::U32 offset = 0U; offset < kTextSlotCount; ++offset) {
			const oa::U32 candidate = (nextTextSlot + offset) % kTextSlotCount;
			if (textSlots[candidate].isReady()) {
				slotIndex = candidate;
				break;
			}
		}
		if (slotIndex >= kTextSlotCount) {
			return oa::Status::error(
				oa::StatusCode::Unavailable,
				"oa::Ui text upload ring is still consumed by the GPU");
		}
		OA_RETURN_IF_ERROR(textSlots[slotIndex].upload(
			oa::Span<const oa::GlyphInstance>(textGlyphs.data(), textGlyphs.size())));
		const oa::U32 bindlessIndex = textSlots[slotIndex].bindlessIndex();
		for (BlitCmd& command : blits) {
			if (command.kind == BlitKind::Glyphs && command.internalText) {
				command.glyphs.glyph_idx = bindlessIndex;
			}
		}
		usedTextSlot = slotIndex;
		nextTextSlot = (slotIndex + 1U) % kTextSlotCount;
		textPrepared = true;
		return oa::Status::ok();
	}
};


// ─── oa::Ui move/dtor ───────────────────────────────────────────────────────────

oa::Ui::Ui() = default;

oa::Ui::Ui(oa::Ui&& inOther) noexcept
	: impl_(oa::move(inOther.impl_))
	, input_(inOther.input_)
{}

oa::Ui& oa::Ui::operator=(oa::Ui&& inOther) noexcept {
	if (this != &inOther) {
		abandon_();
		impl_  = oa::move(inOther.impl_);
		input_ = inOther.input_;
	}
	return *this;
}

oa::Ui::~Ui() { abandon_(); }


// ─── init / close ─────────────────────────────────────────────────────────────

oa::Status oa::Ui::init(oa::Engine& inRt, const oa::UiStyle& inStyle) {
	if (impl_) {
		return oa::Status::error(
			oa::StatusCode::AlreadyExists,
			"oa::Ui::init requires an uninitialized or closed renderer");
	}
	OA_RETURN_IF_ERROR(inStyle.validate());
	impl_ = oa::makeUnique<Impl>();
	impl_->rt = &inRt;
	impl_->baseStyle = inStyle;
	impl_->defaultStyle = inStyle;
	return oa::Status::ok();
}

oa::Status oa::Ui::bindTextAtlas(const oa::TextAtlas& inAtlas) {
	if (!impl_ || impl_->rt == nullptr) {
		return oa::Status::error(
			oa::StatusCode::FailedPrecondition,
			"oa::Ui::bindTextAtlas requires an initialized oa::Ui");
	}
	if (impl_->textAtlas != nullptr) {
		return oa::Status::error(
			oa::StatusCode::AlreadyExists,
			"oa::Ui::bindTextAtlas may only be called once");
	}
	if (inAtlas.atlasBindlessIndex(oa::FontId::Sans) == OA_BINDLESS_INVALID) {
		return oa::Status::error(
			oa::StatusCode::FailedPrecondition,
			"oa::Ui::bindTextAtlas requires an initialized atlas");
	}

	oa::Array<oa::GlyphBuffer, Impl::kTextSlotCount> slots;
	for (auto& slot : slots) {
		auto buffer = oa::GlyphBuffer::createHostUpload(
			*impl_->rt, Impl::kTextGlyphCapacity);
		if (!buffer.isOk()) return buffer.getStatus();
		slot = oa::move(*buffer);
	}
	impl_->textSlots = oa::move(slots);
	impl_->textAtlas = &inAtlas;
	return oa::Status::ok();
}

void oa::Ui::release_() noexcept {
	if (!impl_) return;
	if (impl_->rt) {
		impl_->textAtlas = nullptr;
		for (auto& plot : impl_->plots) {
			for (auto& slot : plot.slots) {
				if (slot.buffer.buffer != nullptr) {
					oa::EngineBindlessAccess::deregisterBuffer(*impl_->rt, slot.buffer);
					oa::EngineAllocatorAccess::get(*impl_->rt).free(slot.buffer);
				}
			}
		}
		impl_->plots.clear();
		impl_->usedPlots.clear();
		impl_->pendingTextureOwners.clear();
		impl_->textureRetentions.clear();
		for (const auto& slot : impl_->sampledImageSlots) {
			if (slot.slot != OA_BINDLESS_INVALID) {
				oa::EngineBindlessAccess::get(*impl_->rt)
					.deregisterSampledImage(slot.slot);
			}
		}
		impl_->sampledImageSlots.clear();
		impl_->blitRgba.destroy(oa::EngineDeviceAccess::get(*impl_->rt));
		impl_->blitPlanar.destroy(oa::EngineDeviceAccess::get(*impl_->rt));
		impl_->blitImageRgba.destroy(oa::EngineDeviceAccess::get(*impl_->rt));
		impl_->drawRect.destroy(oa::EngineDeviceAccess::get(*impl_->rt));
		impl_->drawRectOutline.destroy(oa::EngineDeviceAccess::get(*impl_->rt));
		impl_->drawLine.destroy(oa::EngineDeviceAccess::get(*impl_->rt));
		impl_->drawCanvasGrid.destroy(oa::EngineDeviceAccess::get(*impl_->rt));
		impl_->drawRectOutlines.destroy(oa::EngineDeviceAccess::get(*impl_->rt));
		impl_->drawGlyphs.destroy(oa::EngineDeviceAccess::get(*impl_->rt));
		impl_->drawWaveform.destroy(oa::EngineDeviceAccess::get(*impl_->rt));
		impl_->drawPlotLine.destroy(oa::EngineDeviceAccess::get(*impl_->rt));
		impl_->drawHeatmap.destroy(oa::EngineDeviceAccess::get(*impl_->rt));
	}
	impl_.reset();
	input_ = {};
}

oa::Status oa::Ui::close() {
	if (!impl_) return oa::Status::ok();
	for (const oa::Event& completion : impl_->frameCompletions) {
		OA_RETURN_IF_ERROR(completion.wait());
	}
	release_();
	return oa::Status::ok();
}

void oa::Ui::abandon_() noexcept {
	if (!impl_) return;
	oa::Engine* engine = impl_->rt;
	if (engine == nullptr) {
		impl_.reset();
		return;
	}
	auto retired = oa::makeUnique<oa::Ui>(oa::move(*this));
	oa::BorrowedServiceRetirement::retire(
		*engine,
		retired.release(),
		&oa::Ui::completeRetired_,
		&oa::Ui::releaseRetired_);
}

oa::Status oa::Ui::completeRetired_(void* inPayload) {
	auto* ui = static_cast<oa::Ui*>(inPayload);
	return ui ? ui->close() : oa::Status::ok();
}

void oa::Ui::releaseRetired_(void* inPayload) {
	oa::UniquePtr<oa::Ui> ui(static_cast<oa::Ui*>(inPayload));
}


// ─── initBlit ─────────────────────────────────────────────────────────────────
// Creates UI blit pipelines against the engine's unified bindless layout.

oa::Status oa::Ui::initBlit(void* /*inComposeImageView*/) {
	if (!impl_ || !impl_->rt) return oa::Status::error("oa::Ui: not initialized");
	auto& bindless = oa::EngineBindlessAccess::get(*impl_->rt);
	if (!bindless.pipelineLayout) {
		return oa::Status::error("oa::Ui: bindless layout not initialized");
	}

	oa::PipelineSpec spec;
	spec.pushConstantBytes = 128;

	auto createBlitPipeline = [&](const char* inName, oa::ComputePipeline& outPipeline) -> oa::Status {
		const oavk::SpirvEntry* spv = oavk::findSpirv(inName);
		if (!spv) return oa::Status::error(oa::StatusCode::NotFound, "oa::Ui: blit SPIR-V not found");
		auto res = oa::ComputePipeline::create(
			oa::EngineDeviceAccess::get(*impl_->rt),
			oa::Span<const oa::U8>(spv->data, spv->size),
			spec,
			nullptr,
			bindless.pipelineLayout);
		if (!res.isOk()) return res.getStatus();
		outPipeline = oa::move(*res);
		return oa::Status::ok();
	};

	OA_RETURN_IF_ERROR(createBlitPipeline("BlitRgba", impl_->blitRgba));
	OA_RETURN_IF_ERROR(createBlitPipeline("BlitPlanar", impl_->blitPlanar));
	OA_RETURN_IF_ERROR(createBlitPipeline("BlitImageRgba", impl_->blitImageRgba));
	OA_RETURN_IF_ERROR(createBlitPipeline("DrawRect", impl_->drawRect));
	OA_RETURN_IF_ERROR(createBlitPipeline("DrawRectOutline", impl_->drawRectOutline));
	OA_RETURN_IF_ERROR(createBlitPipeline("DrawLine", impl_->drawLine));
	OA_RETURN_IF_ERROR(createBlitPipeline("DrawCanvasGrid", impl_->drawCanvasGrid));
	OA_RETURN_IF_ERROR(createBlitPipeline("DrawRectOutlines", impl_->drawRectOutlines));
	OA_RETURN_IF_ERROR(createBlitPipeline("DrawGlyphs", impl_->drawGlyphs));
	OA_RETURN_IF_ERROR(createBlitPipeline("DrawWaveform", impl_->drawWaveform));
	OA_RETURN_IF_ERROR(createBlitPipeline("DrawPlotLine", impl_->drawPlotLine));
	OA_RETURN_IF_ERROR(createBlitPipeline("DrawHeatmap", impl_->drawHeatmap));
	return oa::Status::ok();
}


// ─── updateBlitImage ──────────────────────────────────────────────────────────

void oa::Ui::updateBlitImage(void* /*inComposeImageView*/) {
	// The viewer registers its compose target directly in the engine's
	// bindless storage-image heap.
}


// ─── Per-frame ────────────────────────────────────────────────────────────────

void oa::Ui::beginFrame(
	oa::F32 inDeltaMs,
	oa::PixelRect inViewport,
	oa::F32 inContentScale) {
	if (!impl_) return;
	++impl_->frameIndex;
	const bool validScale = oa::isFinite(inContentScale)
		&& inContentScale > 0.0F;
	impl_->contentScale = validScale ? inContentScale : 1.0F;
	impl_->defaultStyle = scaleUiStyleGeometry(
		impl_->baseStyle, impl_->contentScale);
	const oa::Status scaledStyleStatus = validScale
		? impl_->defaultStyle.validate()
		: oa::Status::invalidArgument(
			"oa::Ui::beginFrame requires a finite positive content scale");
	const oa::F32 deltaMs = oa::isFinite(inDeltaMs)
		? oa::clamp(inDeltaMs, 0.0F, 1000.0F) : 0.0F;
	impl_->frameDeltaMs = deltaMs;
	impl_->frameViewport = inViewport.w > 0 && inViewport.h > 0
		? clipToNonNegative(inViewport) : oa::PixelRect{};
	for (auto& state : impl_->textEditStates) {
		state.blinkMs = oa::fmod(state.blinkMs + deltaMs, 1200.0F);
	}
	for (oa::Usize index = impl_->textEditStates.size(); index > 0U; --index) {
		const auto& state = impl_->textEditStates[index - 1U];
		if (impl_->frameIndex - state.lastSeenFrame > 600U
			&& input_.focusId != state.id) {
			impl_->textEditStates.erase(impl_->textEditStates.begin() + (index - 1U));
		}
	}
	for (oa::Usize index = impl_->scrollStates.size(); index > 0U; --index) {
		const auto& state = impl_->scrollStates[index - 1U];
		if (impl_->frameIndex - state.lastSeenFrame > 600U) {
			impl_->scrollStates.erase(impl_->scrollStates.begin() + (index - 1U));
		}
	}
	for (oa::Usize index = impl_->textureRetentions.size(); index > 0U; --index) {
		if (impl_->textureRetentions[index - 1U].completion.isComplete()) {
			impl_->textureRetentions.erase(
				impl_->textureRetentions.begin() + (index - 1U));
		}
	}
	impl_->priorFocusOrder.swap(impl_->focusOrder);
	impl_->focusOrder.clear();
	impl_->priorPopupFocusOrder.swap(impl_->popupFocusOrder);
	impl_->popupFocusOrder.clear();
	impl_->priorAdjustableOrder.swap(impl_->adjustableOrder);
	impl_->adjustableOrder.clear();
	impl_->priorTreeOrder.swap(impl_->treeOrder);
	impl_->treeOrder.clear();
	impl_->priorTabOrder.swap(impl_->tabOrder);
	impl_->tabOrder.clear();
	impl_->priorTextInputOrder.swap(impl_->textInputOrder);
	impl_->textInputOrder.clear();
	impl_->priorScrollRecords.swap(impl_->scrollRecords);
	impl_->scrollRecords.clear();
	impl_->textEditEvents.clear();
	impl_->focusedTextInputRect = {};
	impl_->accessibilityNodes.clear();
	impl_->blits.clear();
	impl_->overlayBlits.clear();
	impl_->usedPlots.clear();
	impl_->usedImagePlanes.clear();
	impl_->panelStack.clear();
	impl_->styleDepth = 0;
	impl_->textGlyphs.clear();
	impl_->textScratch.clear();
	impl_->usedTextSlot = Impl::kTextSlotCount;
	impl_->textPrepared = false;
	impl_->frameSealed = false;
	impl_->frameStatus = scaledStyleStatus;
	impl_->keyboardActivate = false;
	impl_->keyboardAdjust = 0.0F;
	impl_->keyboardTreeActivateId = 0U;
	impl_->keyboardTreeOpenDirection = 0;
	impl_->keyboardTabActivateId = 0U;
	impl_->keyboardTabCloseId = 0U;
	impl_->pointerPressClaimed = false;
	impl_->recordingOverlay = false;
	impl_->renderingPopupId = 0U;
	impl_->popupPanelDepth = 0U;
	impl_->popupRenderedThisFrame = false;
	impl_->popupOpenedThisFrame = false;
	impl_->popupDismissedThisFrame = false;
	impl_->popupFocusMovedThisFrame = false;
	impl_->lastItemId = 0U;
	impl_->lastItemRect = {};
	impl_->lastItemHovered = false;
	input_.mouseDX = 0.0F;
	input_.mouseDY = 0.0F;
	input_.scrollX = 0.0F;
	input_.scrollY = 0.0F;
	input_.lPressed = false;
	input_.lReleased = false;
	input_.lClickCount = 0;
	input_.hoverId = 0U;
}

oa::F32 oa::Ui::contentScale() const noexcept {
	return impl_ ? impl_->contentScale : 1.0F;
}

oa::Span<const oa::UiAccessibilityNode>
oa::Ui::accessibilitySnapshot() const noexcept {
	if (!impl_) return {};
	return oa::Span<const oa::UiAccessibilityNode>(
		impl_->accessibilityNodes.data(), impl_->accessibilityNodes.size());
}

bool oa::Ui::routeEvent(const oa::UiEvent& inEvent) {
	input_.modifiers = inEvent.modifiers;
	bool consumed = false;
	bool pointerEvent = false;
	const auto textInputFocused = [&] {
		return impl_ && impl_->wasTextInput(input_.focusId);
	};
	const auto queueTextEdit = [&](TextEditKind inKind, bool inExtend = false,
		bool inByWord = false) {
		TextEditEvent event;
		event.kind = inKind;
		event.extend = inExtend;
		event.byWord = inByWord;
		impl_->textEditEvents.pushBack(oa::move(event));
	};
	switch (inEvent.type) {
		case oa::UiEventType::MouseMove:
			pointerEvent = true;
			input_.mouseX = inEvent.mouseX;
			input_.mouseY = inEvent.mouseY;
			input_.mouseDX += inEvent.mouseDX;
			input_.mouseDY += inEvent.mouseDY;
			if (impl_ && impl_->openPopupId != 0U
				&& (impl_->popupRect.contains(inEvent.mouseX, inEvent.mouseY)
					|| impl_->popupAnchorRect.contains(
						inEvent.mouseX, inEvent.mouseY))) consumed = true;
			break;
		case oa::UiEventType::MouseDown:
			pointerEvent = true;
			input_.mouseX = inEvent.mouseX;
			input_.mouseY = inEvent.mouseY;
			if (impl_ && impl_->openPopupId != 0U) {
				const bool inside = impl_->popupRect.contains(
					inEvent.mouseX, inEvent.mouseY)
					|| impl_->popupAnchorRect.contains(
						inEvent.mouseX, inEvent.mouseY);
				if (!inside) {
					impl_->closePopup(input_, false);
					impl_->popupDismissedThisFrame = true;
				}
				consumed = true;
			}
			if (inEvent.button == 1) {
				input_.lButton = true;
				input_.lPressed = true;
				input_.lClickCount = oa::max(1, inEvent.clickCount);
			} else if (inEvent.button == 2) {
				input_.mButton = true;
			} else if (inEvent.button == 3) {
				input_.rButton = true;
			}
			break;
		case oa::UiEventType::MouseUp:
			pointerEvent = true;
			input_.mouseX = inEvent.mouseX;
			input_.mouseY = inEvent.mouseY;
			if (inEvent.button == 1) {
				input_.lButton = false;
				input_.lReleased = true;
			} else if (inEvent.button == 2) {
				input_.mButton = false;
			} else if (inEvent.button == 3) {
				input_.rButton = false;
			}
			if (impl_ && impl_->openPopupId != 0U
				&& (impl_->popupRect.contains(inEvent.mouseX, inEvent.mouseY)
					|| impl_->popupAnchorRect.contains(
						inEvent.mouseX, inEvent.mouseY))) consumed = true;
			break;
		case oa::UiEventType::MouseScroll:
			pointerEvent = true;
			input_.mouseX = inEvent.mouseX;
			input_.mouseY = inEvent.mouseY;
			if (oa::isFinite(inEvent.scrollX)) input_.scrollX += inEvent.scrollX;
			if (oa::isFinite(inEvent.scrollY)) input_.scrollY += inEvent.scrollY;
			if (impl_ && oa::isFinite(inEvent.scrollY)
				&& inEvent.scrollY != 0.0F) {
				for (oa::Usize index = impl_->priorScrollRecords.size();
					index > 0U; --index) {
					const auto& record = impl_->priorScrollRecords[index - 1U];
					if (impl_->openPopupId != 0U && !record.overlay) continue;
					if (impl_->openPopupId == 0U && record.overlay) continue;
					if (record.maxOffsetY <= 0
						|| !record.viewport.contains(inEvent.mouseX, inEvent.mouseY)) {
						continue;
					}
					auto* state = impl_->findScrollState(record.id);
					if (state == nullptr) continue;
					const oa::F64 scaled = oa::clamp<oa::F64>(
						-static_cast<oa::F64>(inEvent.scrollY) * record.wheelStep,
						oa::Limits<oa::I32>::min(),
						oa::Limits<oa::I32>::max());
					oa::I64 delta = static_cast<oa::I64>(oa::round(scaled));
					if (delta == 0) delta = inEvent.scrollY > 0.0F ? -1 : 1;
					const oa::I32 next = static_cast<oa::I32>(oa::clamp<oa::I64>(
						static_cast<oa::I64>(state->offsetY) + delta,
						0,
						record.maxOffsetY));
					if (next == state->offsetY) continue;
					state->offsetY = next;
					consumed = true;
					break;
				}
			}
			if (impl_ && impl_->openPopupId != 0U
				&& impl_->popupRect.contains(inEvent.mouseX, inEvent.mouseY)) {
				consumed = true;
			}
			break;
		case oa::UiEventType::KeyDown:
			if (impl_ && impl_->openPopupId != 0U
				&& inEvent.key == oa::UiKey::Escape) {
				if (!inEvent.keyRepeat) impl_->closePopup(input_, true);
				consumed = true;
			} else if (impl_ && impl_->openPopupId != 0U
				&& (inEvent.key == oa::UiKey::Up || inEvent.key == oa::UiKey::Down)) {
				if (!inEvent.keyRepeat) {
					impl_->advanceFocus(input_, inEvent.key == oa::UiKey::Up);
					impl_->popupFocusMovedThisFrame = true;
				}
				consumed = true;
			} else if (impl_ && impl_->openPopupId != 0U
				&& inEvent.key == oa::UiKey::Tab) {
				if (!inEvent.keyRepeat) {
					impl_->advanceFocus(input_, inEvent.shift());
					impl_->popupFocusMovedThisFrame = true;
				}
				consumed = true;
			} else if (impl_ && inEvent.key == oa::UiKey::Tab
				&& (!impl_->priorFocusOrder.empty() || input_.focusId != 0U)) {
				if (!inEvent.keyRepeat) {
					impl_->advanceFocus(input_, inEvent.shift());
				}
				consumed = true;
			} else if (textInputFocused()) {
				const bool primary = inEvent.ctrl()
					|| (inEvent.modifiers & oa::UiModifierSuper) != 0U;
				switch (inEvent.key) {
					case oa::UiKey::Left:
						queueTextEdit(TextEditKind::MoveLeft, inEvent.shift(), primary);
						break;
					case oa::UiKey::Right:
						queueTextEdit(TextEditKind::MoveRight, inEvent.shift(), primary);
						break;
					case oa::UiKey::Home:
						queueTextEdit(TextEditKind::MoveHome, inEvent.shift());
						break;
					case oa::UiKey::End:
						queueTextEdit(TextEditKind::MoveEnd, inEvent.shift());
						break;
					case oa::UiKey::Backspace:
						queueTextEdit(TextEditKind::Backspace, false, primary);
						break;
					case oa::UiKey::Delete:
						queueTextEdit(TextEditKind::Delete, false, primary);
						break;
					case oa::UiKey::A:
						if (primary && !inEvent.keyRepeat) {
							queueTextEdit(TextEditKind::SelectAll);
						}
						break;
					case oa::UiKey::C:
						if (primary && !inEvent.keyRepeat) {
							queueTextEdit(TextEditKind::Copy);
						}
						break;
					case oa::UiKey::X:
						if (primary && !inEvent.keyRepeat) {
							queueTextEdit(TextEditKind::Cut);
						}
						break;
					case oa::UiKey::Z:
						if (primary && !inEvent.keyRepeat) {
							queueTextEdit(inEvent.shift()
								? TextEditKind::Redo : TextEditKind::Undo);
						}
						break;
					case oa::UiKey::Y:
						if (primary && !inEvent.keyRepeat) {
							queueTextEdit(TextEditKind::Redo);
						}
						break;
					case oa::UiKey::Return:
					case oa::UiKey::KpEnter:
						if (!inEvent.keyRepeat) input_.focusId = 0U;
						break;
					case oa::UiKey::Escape:
						if (!inEvent.keyRepeat) {
							auto* state = impl_->findTextEditState(input_.focusId);
							if (state != nullptr && !state->preedit.empty()) {
								queueTextEdit(TextEditKind::CancelComposition);
							} else {
								input_.focusId = 0U;
							}
						}
						break;
					default:
						break;
				}
				consumed = isTextKey(inEvent.key) || inEvent.key == oa::UiKey::Left
					|| inEvent.key == oa::UiKey::Right || inEvent.key == oa::UiKey::Home
					|| inEvent.key == oa::UiKey::End || inEvent.key == oa::UiKey::Backspace
					|| inEvent.key == oa::UiKey::Delete || inEvent.key == oa::UiKey::Return
					|| inEvent.key == oa::UiKey::KpEnter || inEvent.key == oa::UiKey::Escape;
			} else if (impl_ && impl_->wasTab(input_.focusId)
				&& (inEvent.key == oa::UiKey::Left || inEvent.key == oa::UiKey::Right
					|| inEvent.key == oa::UiKey::Home || inEvent.key == oa::UiKey::End)) {
				const bool toEdge = inEvent.key == oa::UiKey::Home
					|| inEvent.key == oa::UiKey::End;
				const bool reverse = inEvent.key == oa::UiKey::Left
					|| inEvent.key == oa::UiKey::Home;
				impl_->keyboardTabActivateId = impl_->advanceTabFocus(
					input_, reverse, toEdge);
				consumed = true;
			} else if (impl_ && impl_->wasTab(input_.focusId)
				&& inEvent.key == oa::UiKey::W
				&& (inEvent.ctrl()
					|| (inEvent.modifiers & oa::UiModifierSuper) != 0U)) {
				if (!inEvent.keyRepeat) {
					impl_->keyboardTabCloseId = input_.focusId;
				}
				consumed = true;
			} else if (impl_ && impl_->wasTree(input_.focusId)
				&& (inEvent.key == oa::UiKey::Up || inEvent.key == oa::UiKey::Down
					|| inEvent.key == oa::UiKey::Home || inEvent.key == oa::UiKey::End)) {
				const bool toEdge = inEvent.key == oa::UiKey::Home
					|| inEvent.key == oa::UiKey::End;
				const bool reverse = inEvent.key == oa::UiKey::Up
					|| inEvent.key == oa::UiKey::Home;
				impl_->keyboardTreeActivateId = impl_->advanceTreeFocus(
					input_, reverse, toEdge);
				consumed = true;
			} else if (impl_ && impl_->wasTree(input_.focusId)
				&& (inEvent.key == oa::UiKey::Left || inEvent.key == oa::UiKey::Right)) {
				impl_->keyboardTreeOpenDirection =
					inEvent.key == oa::UiKey::Right ? 1 : -1;
				consumed = true;
			} else if (impl_ && input_.focusId != 0U
				&& (inEvent.key == oa::UiKey::Return
					|| inEvent.key == oa::UiKey::KpEnter
					|| inEvent.key == oa::UiKey::Space)) {
				if (!inEvent.keyRepeat) impl_->keyboardActivate = true;
				consumed = true;
			} else if (impl_ && impl_->wasAdjustable(input_.focusId)
				&& isSliderAdjustmentKey(inEvent.key)) {
				oa::F32 scale = 1.0F;
				if (inEvent.shift()) scale *= 10.0F;
				if (inEvent.ctrl()) scale *= 0.1F;
				const oa::F32 direction =
					(inEvent.key == oa::UiKey::Right || inEvent.key == oa::UiKey::Up)
						? 1.0F : -1.0F;
				impl_->keyboardAdjust += direction * scale;
				consumed = true;
			} else if (impl_ && impl_->openPopupId != 0U) {
				consumed = true;
			}
			break;
		case oa::UiEventType::KeyChar:
			if (textInputFocused()) {
				oa::String text = singleLineCommittedText(inEvent);
				if (!text.empty()) {
					TextEditEvent event;
					event.text = oa::move(text);
					impl_->textEditEvents.pushBack(oa::move(event));
				}
				consumed = true;
			} else if (impl_ && impl_->openPopupId != 0U) consumed = true;
			break;
		case oa::UiEventType::TextEditing:
			if (textInputFocused()) {
				TextEditEvent event;
				event.kind = TextEditKind::UpdateComposition;
				event.text = inEvent.text;
				event.selectionStart = inEvent.textSelectionStart;
				event.selectionLength = inEvent.textSelectionLength;
				impl_->textEditEvents.pushBack(oa::move(event));
				consumed = true;
			} else if (impl_ && impl_->openPopupId != 0U) consumed = true;
			break;
		case oa::UiEventType::KeyUp:
			consumed = impl_ && impl_->openPopupId != 0U
				? true : textInputFocused()
				? (isTextKey(inEvent.key) || inEvent.key == oa::UiKey::Left
					|| inEvent.key == oa::UiKey::Right || inEvent.key == oa::UiKey::Home
					|| inEvent.key == oa::UiKey::End || inEvent.key == oa::UiKey::Backspace
					|| inEvent.key == oa::UiKey::Delete || inEvent.key == oa::UiKey::Return
					|| inEvent.key == oa::UiKey::KpEnter || inEvent.key == oa::UiKey::Escape)
				: input_.focusId != 0U
				&& (inEvent.key == oa::UiKey::Tab
					|| inEvent.key == oa::UiKey::Return
					|| inEvent.key == oa::UiKey::KpEnter
					|| inEvent.key == oa::UiKey::Space
					|| (impl_ && impl_->wasTree(input_.focusId)
						&& (inEvent.key == oa::UiKey::Left
							|| inEvent.key == oa::UiKey::Right
							|| inEvent.key == oa::UiKey::Up
							|| inEvent.key == oa::UiKey::Down
							|| inEvent.key == oa::UiKey::Home
							|| inEvent.key == oa::UiKey::End))
					|| (impl_ && impl_->wasTab(input_.focusId)
						&& (inEvent.key == oa::UiKey::Left
							|| inEvent.key == oa::UiKey::Right
							|| inEvent.key == oa::UiKey::Home
							|| inEvent.key == oa::UiKey::End
							|| (inEvent.key == oa::UiKey::W
								&& (inEvent.ctrl()
									|| (inEvent.modifiers & oa::UiModifierSuper) != 0U))))
					|| (impl_ && impl_->wasAdjustable(input_.focusId)
						&& isSliderAdjustmentKey(inEvent.key)));
			break;
		case oa::UiEventType::WindowBlur:
			consumed = input_.activeId != 0U || input_.focusId != 0U
				|| (impl_ && impl_->openPopupId != 0U);
			input_.lButton = false;
			input_.mButton = false;
			input_.rButton = false;
			input_.lReleased = true;
			input_.activeId = 0U;
			if (impl_) {
				impl_->closePopup(input_, false);
				impl_->keyboardActivate = false;
				impl_->keyboardAdjust = 0.0F;
				impl_->keyboardTreeActivateId = 0U;
				impl_->keyboardTreeOpenDirection = 0;
				impl_->keyboardTabActivateId = 0U;
				impl_->keyboardTabCloseId = 0U;
				impl_->tabDragBarId = 0U;
				impl_->tabDragItemId = 0U;
				impl_->textEditEvents.clear();
				for (auto& state : impl_->textEditStates) state.preedit.clear();
			}
			break;
		default:
			break;
	}
	return consumed || (pointerEvent && (input_.activeId != 0U
		|| (impl_ && impl_->popupDismissedThisFrame)));
}

bool oa::Ui::wantsTextInput() const noexcept {
	return impl_ && impl_->isCurrentTextInput(input_.focusId);
}

oa::PixelRect oa::Ui::textInputRect() const noexcept {
	return wantsTextInput() ? impl_->focusedTextInputRect : oa::PixelRect{};
}

bool oa::Ui::takeClipboardWrite(oa::String& outText) {
	if (!impl_ || impl_->clipboardWrite.empty()) return false;
	outText = oa::move(impl_->clipboardWrite);
	impl_->clipboardWrite.clear();
	return true;
}

oa::Status oa::Ui::recordRender(
	VkCommandBuffer inCmd,
	oa::U32 inDstBindlessIdx)
{
	if (!impl_) return oa::Status::ok();
	if (impl_->renderingPopupId != 0U) {
		impl_->setFrameError(oa::Status::error(
			oa::StatusCode::FailedPrecondition,
			"oa::Ui::recordRender requires the active popup to be ended first"));
	}
	impl_->frameSealed = true;
	impl_->mergeOverlayBlits();
	if (!impl_->frameStatus.isOk()) return impl_->frameStatus;
	OA_RETURN_IF_ERROR(impl_->prepareText());
	if (impl_->blits.empty()) return oa::Status::ok();
	if (!impl_->blitRgba.pipeline && !impl_->blitPlanar.pipeline
		&& !impl_->blitImageRgba.pipeline && !impl_->drawRect.pipeline
		&& !impl_->drawRectOutline.pipeline && !impl_->drawLine.pipeline
		&& !impl_->drawCanvasGrid.pipeline
		&& !impl_->drawRectOutlines.pipeline && !impl_->drawGlyphs.pipeline
		&& !impl_->drawWaveform.pipeline && !impl_->drawPlotLine.pipeline
		&& !impl_->drawHeatmap.pipeline) {
		return oa::Status::ok();
	}

	// Admit the complete frame before recording any command. This preserves a
	// valid, resettable graphics batch if one later widget would exceed a live
	// device limit.
	for (const BlitCmd& bc : impl_->blits) {
		oa::U32 groupsX = 0U;
		oa::U32 groupsY = 1U;
		oa::Bool hasDispatch = true;
		switch (bc.kind) {
			case BlitKind::Rgba:
				hasDispatch = impl_->blitRgba.pipeline != nullptr;
				groupsX = (bc.rgba.dst_w + 7U) / 8U;
				groupsY = (bc.rgba.dst_h + 7U) / 8U;
				break;
			case BlitKind::Planar:
				hasDispatch = impl_->blitPlanar.pipeline != nullptr;
				groupsX = (bc.planar.dst_w + 7U) / 8U;
				groupsY = (bc.planar.dst_h + 7U) / 8U;
				break;
			case BlitKind::ImageRgba:
				hasDispatch = impl_->blitImageRgba.pipeline != nullptr
					and bc.srcImageView != VK_NULL_HANDLE;
				groupsX = (bc.imageRgba.dst_w + 7U) / 8U;
				groupsY = (bc.imageRgba.dst_h + 7U) / 8U;
				break;
			case BlitKind::Rect:
				hasDispatch = impl_->drawRect.pipeline != nullptr;
				groupsX = (bc.rect.dst_w + 7U) / 8U;
				groupsY = (bc.rect.dst_h + 7U) / 8U;
				break;
			case BlitKind::RectOutline:
				hasDispatch = impl_->drawRectOutline.pipeline != nullptr;
				groupsX = (bc.rectOutline.dst_w + 7U) / 8U;
				groupsY = (bc.rectOutline.dst_h + 7U) / 8U;
				break;
			case BlitKind::RectOutlines:
				hasDispatch = impl_->drawRectOutlines.pipeline != nullptr;
				groupsX = bc.rectOutlines.count;
				break;
			case BlitKind::Line:
				hasDispatch = impl_->drawLine.pipeline != nullptr;
				groupsX = (bc.line.bounds_w + 7U) / 8U;
				groupsY = (bc.line.bounds_h + 7U) / 8U;
				break;
			case BlitKind::CanvasGrid:
				hasDispatch = impl_->drawCanvasGrid.pipeline != nullptr;
				groupsX = (bc.canvasGrid.dst_w + 7U) / 8U;
				groupsY = (bc.canvasGrid.dst_h + 7U) / 8U;
				break;
			case BlitKind::Glyphs:
				hasDispatch = impl_->drawGlyphs.pipeline != nullptr;
				groupsX = bc.glyphs.count;
				break;
			case BlitKind::Waveform:
				hasDispatch = impl_->drawWaveform.pipeline != nullptr;
				groupsX = (bc.waveform.dst_w + 63U) / 64U;
				break;
			case BlitKind::PlotLine:
				hasDispatch = impl_->drawPlotLine.pipeline != nullptr;
				groupsX = (bc.plotLine.dst_w + 63U) / 64U;
				break;
			case BlitKind::Heatmap:
				hasDispatch = impl_->drawHeatmap.pipeline != nullptr;
				groupsX = (bc.heatmap.dst_w + 7U) / 8U;
				groupsY = (bc.heatmap.dst_h + 7U) / 8U;
				break;
		}
		if (hasDispatch) {
			OA_RETURN_IF_ERROR(oavk::validateDirectComputeDispatch(
				oa::EngineDeviceAccess::get(*impl_->rt), groupsX, groupsY, 1U));
		}
	}

	VkCommandBuffer cmd = inCmd;
	auto& bindless = oa::EngineBindlessAccess::get(*impl_->rt);
	VkPipelineLayout layout =
		static_cast<VkPipelineLayout>(bindless.pipelineLayout);

	auto getSampledImageSlot = [this, &bindless](
		VkImageView inView,
		VkImageLayout inLayout) -> oa::U32 {
		for (auto& slot : impl_->sampledImageSlots) {
			if (slot.view == inView) {
				if (slot.layout != inLayout) {
					bindless.updateSampledImage(
						oa::EngineDeviceAccess::get(*impl_->rt), slot.slot, inView, inLayout);
					slot.layout = inLayout;
				}
				return slot.slot;
			}
		}
		oa::U32 idx = bindless.registerSampledImage(
			oa::EngineDeviceAccess::get(*impl_->rt), inView, inLayout);
		if (idx != OA_BINDLESS_INVALID) {
			Impl::SampledImageSlot slot;
			slot.view = inView;
			slot.layout = inLayout;
			slot.slot = idx;
			impl_->sampledImageSlots.pushBack(slot);
		}
		return idx;
	};

	for (BlitCmd& bc : impl_->blits) {
		if (bc.kind == BlitKind::ImageRgba && bc.srcImageView) {
			bc.imageRgba.src_idx = getSampledImageSlot(bc.srcImageView, bc.srcImageLayout);
		}
	}

	// images supplied by external producers (video conversion, inference,
	// uploads) may have been written by an earlier compute submission. queue
	// order alone does not make shader writes visible to this sampled-image
	// read; establish the explicit vulkan memory dependency before blitting.
	for (const BlitCmd& bc : impl_->blits) {
		if (bc.kind != BlitKind::ImageRgba || bc.srcImage == VK_NULL_HANDLE) continue;
		VkImageMemoryBarrier2 barrier{};
		barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2;
		barrier.srcStageMask = bc.srcStageMask;
		barrier.srcAccessMask = bc.srcAccessMask;
		barrier.dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
		barrier.dstAccessMask = VK_ACCESS_2_SHADER_SAMPLED_READ_BIT;
		barrier.oldLayout = bc.srcImageLayout;
		barrier.newLayout = bc.srcImageLayout;
		barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
		barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
		barrier.image = bc.srcImage;
		barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
		barrier.subresourceRange.levelCount = 1;
		barrier.subresourceRange.layerCount = 1;
		VkDependencyInfo dependency{};
		dependency.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
		dependency.imageMemoryBarrierCount = 1;
		dependency.pImageMemoryBarriers = &barrier;
		oa::EngineDeviceAccess::get(*impl_->rt).deviceDispatch.vkCmdPipelineBarrier2(cmd, &dependency);
	}
	// A heatmap may consume an evaluation matrix written by an earlier compute
	// submission. Establish shader-write -> shader-read visibility without a
	// host wait; queue order supplies execution order, this supplies memory order.
	for (const BlitCmd& bc : impl_->blits) {
		if (bc.kind != BlitKind::Heatmap) continue;
		VkMemoryBarrier2 barrier{};
		barrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2;
		barrier.srcStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
		barrier.srcAccessMask = VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT;
		barrier.dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
		barrier.dstAccessMask = VK_ACCESS_2_SHADER_STORAGE_READ_BIT;
		VkDependencyInfo dependency{};
		dependency.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
		dependency.memoryBarrierCount = 1;
		dependency.pMemoryBarriers = &barrier;
		oa::EngineDeviceAccess::get(*impl_->rt).deviceDispatch.vkCmdPipelineBarrier2(cmd, &dependency);
		break;
	}

	VkDescriptorSet set = static_cast<VkDescriptorSet>(bindless.descriptorSet);
	oa::EngineDeviceAccess::get(*impl_->rt).deviceDispatch.vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, layout, 0, 1, &set, 0, nullptr);
	const auto dispatch = [this, cmd](
		oa::U32 inGroupsX, oa::U32 inGroupsY, oa::U32 inGroupsZ)
		-> oa::Status
	{
		OA_RETURN_IF_ERROR(oavk::validateDirectComputeDispatch(
			oa::EngineDeviceAccess::get(*impl_->rt), inGroupsX, inGroupsY, inGroupsZ));
		oa::EngineDeviceAccess::get(*impl_->rt).deviceDispatch.vkCmdDispatch(cmd, inGroupsX, inGroupsY, inGroupsZ);
		return oa::Status::ok();
	};
	const auto memoryBarrier = [this, cmd] {
		VkMemoryBarrier barrier{};
		barrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
		barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
		barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT
			| VK_ACCESS_SHADER_WRITE_BIT;
		oa::EngineDeviceAccess::get(*impl_->rt).deviceDispatch.vkCmdPipelineBarrier(cmd,
			VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
			VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
			0, 1, &barrier, 0, nullptr, 0, nullptr);
	};

	for (const BlitCmd& bc : impl_->blits) {
		if (bc.kind == BlitKind::Rgba) {
			if (!impl_->blitRgba.pipeline) continue;
			BlitRgbaPc pc = bc.rgba;
			pc.dst_idx = inDstBindlessIdx;
			oa::EngineDeviceAccess::get(*impl_->rt).deviceDispatch.vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
				static_cast<VkPipeline>(impl_->blitRgba.pipeline));
			oa::EngineDeviceAccess::get(*impl_->rt).deviceDispatch.vkCmdPushConstants(cmd, layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
			oa::U32 gx = (pc.dst_w + 7u) / 8u;
			oa::U32 gy = (pc.dst_h + 7u) / 8u;
			OA_RETURN_IF_ERROR(dispatch(gx, gy, 1));
		} else if (bc.kind == BlitKind::Planar) {
			if (!impl_->blitPlanar.pipeline) continue;
			BlitPlanarPc pc = bc.planar;
			pc.dst_idx = inDstBindlessIdx;
			oa::EngineDeviceAccess::get(*impl_->rt).deviceDispatch.vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
				static_cast<VkPipeline>(impl_->blitPlanar.pipeline));
			oa::EngineDeviceAccess::get(*impl_->rt).deviceDispatch.vkCmdPushConstants(cmd, layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
			oa::U32 gx = (pc.dst_w + 7u) / 8u;
			oa::U32 gy = (pc.dst_h + 7u) / 8u;
			OA_RETURN_IF_ERROR(dispatch(gx, gy, 1));
		} else if (bc.kind == BlitKind::ImageRgba) {
			if (!impl_->blitImageRgba.pipeline || !bc.srcImageView) continue;
			BlitImageRgbaPc pc = bc.imageRgba;
			if (pc.src_idx == OA_BINDLESS_INVALID) continue;
			pc.dst_idx = inDstBindlessIdx;
			oa::EngineDeviceAccess::get(*impl_->rt).deviceDispatch.vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
				static_cast<VkPipeline>(impl_->blitImageRgba.pipeline));
			oa::EngineDeviceAccess::get(*impl_->rt).deviceDispatch.vkCmdPushConstants(cmd, layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
			oa::U32 gx = (pc.dst_w + 7u) / 8u;
			oa::U32 gy = (pc.dst_h + 7u) / 8u;
			OA_RETURN_IF_ERROR(dispatch(gx, gy, 1));
		} else if (bc.kind == BlitKind::Rect) {
			if (!impl_->drawRect.pipeline) continue;
			DrawRectPc pc = bc.rect;
			pc.dst_idx = inDstBindlessIdx;
			oa::EngineDeviceAccess::get(*impl_->rt).deviceDispatch.vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
				static_cast<VkPipeline>(impl_->drawRect.pipeline));
			oa::EngineDeviceAccess::get(*impl_->rt).deviceDispatch.vkCmdPushConstants(cmd, layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
			oa::U32 gx = (pc.dst_w + 7u) / 8u;
			oa::U32 gy = (pc.dst_h + 7u) / 8u;
			OA_RETURN_IF_ERROR(dispatch(gx, gy, 1));
		} else if (bc.kind == BlitKind::RectOutline) {
			if (!impl_->drawRectOutline.pipeline) continue;
			DrawRectOutlinePc pc = bc.rectOutline;
			pc.dst_idx = inDstBindlessIdx;
			oa::EngineDeviceAccess::get(*impl_->rt).deviceDispatch.vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
				static_cast<VkPipeline>(impl_->drawRectOutline.pipeline));
			oa::EngineDeviceAccess::get(*impl_->rt).deviceDispatch.vkCmdPushConstants(cmd, layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
			oa::U32 gx = (pc.dst_w + 7u) / 8u;
			oa::U32 gy = (pc.dst_h + 7u) / 8u;
			OA_RETURN_IF_ERROR(dispatch(gx, gy, 1));
		} else if (bc.kind == BlitKind::RectOutlines) {
			if (!impl_->drawRectOutlines.pipeline) continue;
			DrawRectOutlinesPc pc = bc.rectOutlines;
			pc.dst_idx = inDstBindlessIdx;
			oa::EngineDeviceAccess::get(*impl_->rt).deviceDispatch.vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
				static_cast<VkPipeline>(impl_->drawRectOutlines.pipeline));
			oa::EngineDeviceAccess::get(*impl_->rt).deviceDispatch.vkCmdPushConstants(cmd, layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
			OA_RETURN_IF_ERROR(dispatch(pc.count, 1, 1));
		} else if (bc.kind == BlitKind::Line) {
			if (!impl_->drawLine.pipeline) continue;
			DrawLinePc pc = bc.line;
			pc.dst_idx = inDstBindlessIdx;
			oa::EngineDeviceAccess::get(*impl_->rt).deviceDispatch.vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
				static_cast<VkPipeline>(impl_->drawLine.pipeline));
			oa::EngineDeviceAccess::get(*impl_->rt).deviceDispatch.vkCmdPushConstants(cmd, layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
			OA_RETURN_IF_ERROR(dispatch(
				(pc.bounds_w + 7U) / 8U,
				(pc.bounds_h + 7U) / 8U, 1));
		} else if (bc.kind == BlitKind::CanvasGrid) {
			if (!impl_->drawCanvasGrid.pipeline) continue;
			DrawCanvasGridPc pc = bc.canvasGrid;
			pc.dst_idx = inDstBindlessIdx;
			oa::EngineDeviceAccess::get(*impl_->rt).deviceDispatch.vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
				static_cast<VkPipeline>(impl_->drawCanvasGrid.pipeline));
			oa::EngineDeviceAccess::get(*impl_->rt).deviceDispatch.vkCmdPushConstants(cmd, layout, VK_SHADER_STAGE_COMPUTE_BIT, 0,
				sizeof(pc), &pc);
			OA_RETURN_IF_ERROR(dispatch(
				(pc.dst_w + 7U) / 8U,
				(pc.dst_h + 7U) / 8U, 1));
		} else if (bc.kind == BlitKind::Glyphs) {
			if (!impl_->drawGlyphs.pipeline) continue;
			DrawGlyphsPc pc = bc.glyphs;
			pc.dst_idx = inDstBindlessIdx;
			oa::EngineDeviceAccess::get(*impl_->rt).deviceDispatch.vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
				static_cast<VkPipeline>(impl_->drawGlyphs.pipeline));
			const oa::U32 batchCount = oa::max(1U, pc.batch_count);
			for (oa::U32 batch = 0U; batch < batchCount; ++batch) {
				pc.batch = batch;
				oa::EngineDeviceAccess::get(*impl_->rt).deviceDispatch.vkCmdPushConstants(cmd, layout, VK_SHADER_STAGE_COMPUTE_BIT, 0,
					sizeof(pc), &pc);
				OA_RETURN_IF_ERROR(dispatch(pc.count, 1, 1));
				if (batch + 1U < batchCount) memoryBarrier();
			}
		} else if (bc.kind == BlitKind::Waveform) {
			if (!impl_->drawWaveform.pipeline) continue;
			DrawWaveformPc pc = bc.waveform;
			pc.dst_idx = inDstBindlessIdx;
			oa::EngineDeviceAccess::get(*impl_->rt).deviceDispatch.vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
				static_cast<VkPipeline>(impl_->drawWaveform.pipeline));
			oa::EngineDeviceAccess::get(*impl_->rt).deviceDispatch.vkCmdPushConstants(cmd, layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
			OA_RETURN_IF_ERROR(dispatch((pc.dst_w + 63U) / 64U, 1, 1));
		} else if (bc.kind == BlitKind::PlotLine) {
			if (!impl_->drawPlotLine.pipeline) continue;
			DrawPlotLinePc pc = bc.plotLine;
			pc.dst_idx = inDstBindlessIdx;
			oa::EngineDeviceAccess::get(*impl_->rt).deviceDispatch.vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
				static_cast<VkPipeline>(impl_->drawPlotLine.pipeline));
			oa::EngineDeviceAccess::get(*impl_->rt).deviceDispatch.vkCmdPushConstants(cmd, layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
			OA_RETURN_IF_ERROR(dispatch((pc.dst_w + 63U) / 64U, 1, 1));
		} else if (bc.kind == BlitKind::Heatmap) {
			if (!impl_->drawHeatmap.pipeline) continue;
			DrawHeatmapPc pc = bc.heatmap;
			pc.dst_idx = inDstBindlessIdx;
			oa::EngineDeviceAccess::get(*impl_->rt).deviceDispatch.vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
				static_cast<VkPipeline>(impl_->drawHeatmap.pipeline));
			oa::EngineDeviceAccess::get(*impl_->rt).deviceDispatch.vkCmdPushConstants(cmd, layout, VK_SHADER_STAGE_COMPUTE_BIT, 0,
				sizeof(pc), &pc);
			OA_RETURN_IF_ERROR(dispatch(
				(pc.dst_w + 7U) / 8U,
				(pc.dst_h + 7U) / 8U, 1));
		}

		// memory barrier between dispatches.
		memoryBarrier();
	}
	return oa::Status::ok();
}

void oa::Ui::endFrame() {
	if (!impl_) return;
	if (impl_->renderingPopupId != 0U) {
		impl_->setFrameError(oa::Status::error(
			oa::StatusCode::FailedPrecondition,
			"oa::Ui::endFrame requires the active popup to be ended first"));
	}
	if (impl_->openPopupId != 0U && !impl_->popupRenderedThisFrame) {
		impl_->closePopup(input_, true);
	}
	if (input_.lPressed && !impl_->pointerPressClaimed) input_.focusId = 0U;
	if (input_.focusId != 0U) {
		bool present = false;
		for (const oa::U32 id : impl_->focusOrder) {
			if (id == input_.focusId) {
				present = true;
				break;
			}
		}
		if (!present) input_.focusId = 0U;
	}
	impl_->keyboardActivate = false;
	impl_->keyboardAdjust = 0.0F;
	if (input_.lReleased) {
		impl_->tabDragBarId = 0U;
		impl_->tabDragItemId = 0U;
	}
	impl_->popupOpenedThisFrame = false;
}

oa::Status oa::Ui::markFrameSubmitted(const oa::Event& inCompletion) {
	if (!impl_ || impl_->rt == nullptr) {
		return oa::Status::error(oa::StatusCode::FailedPrecondition,
			"oa::Ui::markFrameSubmitted requires an initialized UI");
	}
	if (not impl_->rt->ownsEvent(inCompletion)) {
		return oa::Status::invalidArgument(
			"oa::Ui::markFrameSubmitted requires an event from its engine");
	}
	for (oa::Usize index = impl_->frameCompletions.size(); index > 0U; --index) {
		if (impl_->frameCompletions[index - 1U].isComplete()) {
			impl_->frameCompletions.erase(
				impl_->frameCompletions.begin() + (index - 1U));
		}
	}
	if (not inCompletion.isComplete()) {
		bool alreadyTracked = false;
		for (const oa::Event& completion : impl_->frameCompletions) {
			if (completion.isSameCompletion(inCompletion)) {
				alreadyTracked = true;
				break;
			}
		}
		if (not alreadyTracked) impl_->frameCompletions.pushBack(inCompletion);
	}
	for (const auto& used : impl_->usedPlots) {
		if (used.cache >= impl_->plots.size()
			|| used.slot >= Impl::kPlotSlotCount) continue;
		impl_->plots[used.cache].slots[used.slot].completion =
			inCompletion;
	}
	if (impl_->usedTextSlot < Impl::kTextSlotCount) {
		OA_RETURN_IF_ERROR(
			impl_->textSlots[impl_->usedTextSlot].markConsumed(inCompletion));
	}
	for (oa::ImagePlanes* planes : impl_->usedImagePlanes) {
		if (planes != nullptr) {
			OA_RETURN_IF_ERROR(planes->markConsumed(inCompletion));
		}
	}
	if (not impl_->pendingTextureOwners.empty()) {
		if (inCompletion.isComplete()) {
			impl_->pendingTextureOwners.clear();
		} else {
			Impl::TextureRetention retention;
			retention.completion = inCompletion;
			retention.owners = oa::move(impl_->pendingTextureOwners);
			impl_->textureRetentions.pushBack(oa::move(retention));
		}
	}
	return oa::Status::ok();
}


// ─── style stack ─────────────────────────────────────────────────────────────

void oa::Ui::pushStyle(const oa::UiStyle& inStyle) {
	if (!impl_) return;
	if (auto status = inStyle.validate(); !status.isOk()) {
		impl_->setFrameError(oa::move(status));
		return;
	}
	if (impl_->styleDepth >= Impl::kStyleDepth) {
		impl_->setFrameError(oa::Status::error(
			oa::StatusCode::ResourceExhausted,
			"oa::Ui style stack exceeds its fixed depth"));
		return;
	}
	impl_->styleStack[impl_->styleDepth++] = inStyle;
}

void oa::Ui::popStyle() {
	if (!impl_) return;
	if (impl_->styleDepth == 0) {
		impl_->setFrameError(oa::Status::error(
			oa::StatusCode::FailedPrecondition,
			"oa::Ui::popStyle requires a matching pushStyle"));
		return;
	}
	--impl_->styleDepth;
}

const oa::UiStyle& oa::Ui::currentStyle() const noexcept {
	if (!impl_) {
		static const oa::UiStyle sDefault;
		return sDefault;
	}
	if (impl_->styleDepth == 0) return impl_->defaultStyle;
	return impl_->styleStack[impl_->styleDepth - 1];
}


// ─── layout containers ───────────────────────────────────────────────────────

void oa::Ui::beginPanel(oa::StringView inId, oa::PixelRect inRect, const oa::UiLayout& inLayout) {
	if (!impl_) return;
	Impl::PanelState ps;
	ps.rect   = inRect;
	ps.clip = clipToNonNegative(inRect);
	if (!impl_->panelStack.empty()) {
		ps.clip = intersectPixelRects(ps.clip, impl_->panelStack.back().clip);
	}
	ps.layout = inLayout;
	ps.cursor = inLayout.padding.top;
	ps.rowX = inLayout.padding.left;
	ps.rowY = inLayout.padding.top;
	ps.scope = hashWidgetScope(impl_->currentScope(), inId);
	ps.rowScope = ps.scope;
	impl_->panelStack.pushBack(ps);
}

void oa::Ui::endPanel() {
	if (!impl_ || impl_->panelStack.empty()) return;
	if (impl_->panelStack.back().scrollPanel) {
		impl_->setFrameError(oa::Status::error(
			oa::StatusCode::FailedPrecondition,
			"oa::Ui::endPanel cannot close a scroll panel; use endScrollPanel"));
		return;
	}
	if (impl_->panelStack.back().explicitRow) {
		impl_->setFrameError(oa::Status::error(
			oa::StatusCode::FailedPrecondition,
			"oa::Ui::endPanel requires the active row to be ended first"));
	}
	impl_->panelStack.popBack();
}

oa::UiScrollRegion oa::Ui::beginScrollPanel(
	oa::StringView inId,
	oa::PixelRect inViewport,
	oa::I32 inContentHeight,
	const oa::UiLayout& inLayout,
	const oa::UiScrollConfig& inConfig) {
	if (!impl_) return {};
	if (impl_->frameSealed) {
		impl_->setFrameError(oa::Status::error(
			oa::StatusCode::FailedPrecondition,
			"oa::Ui scroll panels cannot begin after recordRender"));
		return {};
	}
	if (inViewport.w <= 0 || inViewport.h <= 0 || inContentHeight < 0
		|| inConfig.wheelStep <= 0
		|| (inConfig.showScrollbar && inConfig.scrollbarWidth <= 0)
		|| inConfig.scrollbarGap < 0) {
		impl_->setFrameError(oa::Status::invalidArgument(
			"oa::Ui::beginScrollPanel requires positive viewport/wheel metrics, non-negative content/gap, and positive visible scrollbar width"));
		return {};
	}

	const oa::U32 id = hashWidgetScope(impl_->currentScope(), inId);
	auto& state = impl_->getScrollState(id);
	const oa::I32 contentHeight = oa::max(inViewport.h, inContentHeight);
	const oa::I32 maxOffset = oa::max(0, contentHeight - inViewport.h);
	state.offsetY = oa::clamp(state.offsetY, 0, maxOffset);

	oa::PixelRect inheritedViewport = clipToNonNegative(inViewport);
	if (!impl_->panelStack.empty()) {
		inheritedViewport = intersectPixelRects(
			inheritedViewport, impl_->panelStack.back().clip);
	}
	impl_->scrollRecords.pushBack({
		.id = id,
		.viewport = inheritedViewport,
		.maxOffsetY = maxOffset,
		.wheelStep = inConfig.wheelStep,
		.overlay = impl_->recordingOverlay,
	});

	oa::I32 reservedWidth = 0;
	const bool showScrollbar = inConfig.showScrollbar && maxOffset > 0
		&& inViewport.w > inConfig.scrollbarWidth + inConfig.scrollbarGap;
	if (showScrollbar) {
		reservedWidth = inConfig.scrollbarWidth + inConfig.scrollbarGap;
		const oa::PixelRect track{
			inViewport.x + inViewport.w - inConfig.scrollbarWidth,
			inViewport.y,
			inConfig.scrollbarWidth,
			inViewport.h,
		};
		const oa::I32 thumbHeight = oa::clamp<oa::I32>(
			static_cast<oa::I32>(oa::round(
				static_cast<oa::F64>(track.h)
					* static_cast<oa::F64>(inViewport.h)
					/ static_cast<oa::F64>(contentHeight))),
			oa::min(24, track.h),
			track.h);
		const oa::I32 travel = oa::max(0, track.h - thumbHeight);
		auto thumbY = [&] {
			return track.y + (maxOffset > 0
				? static_cast<oa::I32>(oa::round(
					static_cast<oa::F64>(travel)
						* static_cast<oa::F64>(state.offsetY)
						/ static_cast<oa::F64>(maxOffset)))
				: 0);
		};
		const oa::U32 thumbId = hashWidgetScope(id, "__scrollbar");
		oa::PixelRect thumb{track.x, thumbY(), track.w, thumbHeight};
		const oa::PixelRect trackHit = impl_->clipFor(track);
		const bool hovered = impl_->canInteract(thumbId)
			&& trackHit.contains(input_.mouseX, input_.mouseY);
		if (hovered) input_.hoverId = thumbId;
		if (hovered && input_.lPressed) {
			input_.activeId = thumbId;
			impl_->pointerPressClaimed = true;
			state.dragGrabY = thumb.contains(input_.mouseX, input_.mouseY)
				? static_cast<oa::I32>(oa::floor(input_.mouseY)) - thumb.y
				: thumb.h / 2;
		}
		if (input_.activeId == thumbId
			&& (input_.lButton || input_.lReleased)) {
			const oa::I32 desired = static_cast<oa::I32>(oa::floor(input_.mouseY))
				- state.dragGrabY - track.y;
			state.offsetY = travel > 0
				? oa::clamp<oa::I32>(static_cast<oa::I32>(oa::round(
					static_cast<oa::F64>(oa::clamp(desired, 0, travel))
						* static_cast<oa::F64>(maxOffset)
						/ static_cast<oa::F64>(travel))), 0, maxOffset)
				: 0;
			thumb.y = thumbY();
		}
		if (input_.activeId == thumbId && input_.lReleased) {
			input_.activeId = 0U;
		}
		const oa::UiStyle& style = currentStyle();
		this->rect(track, style.surface.withAlpha(0.72F));
		this->rect(thumb,
			input_.activeId == thumbId ? style.accentActive
				: hovered ? style.accentHover : style.borderStrong);
	}

	const oa::PixelRect contentViewport{
		inViewport.x,
		inViewport.y,
		oa::max(1, inViewport.w - reservedWidth),
		inViewport.h,
	};
	Impl::PanelState panel;
	panel.rect = {
		contentViewport.x,
		contentViewport.y - state.offsetY,
		contentViewport.w,
		contentHeight,
	};
	panel.clip = clipToNonNegative(contentViewport);
	if (!impl_->panelStack.empty()) {
		panel.clip = intersectPixelRects(
			panel.clip, impl_->panelStack.back().clip);
	}
	panel.layout = inLayout;
	panel.cursor = inLayout.padding.top;
	panel.rowX = inLayout.padding.left;
	panel.rowY = inLayout.padding.top;
	panel.scope = id;
	panel.rowScope = id;
	panel.scrollPanel = true;
	panel.scrollViewport = contentViewport;
	panel.scrollOffsetY = state.offsetY;
	impl_->panelStack.pushBack(panel);
	return {
		.viewport = contentViewport,
		.content = panel.rect,
		.offsetY = state.offsetY,
		.maxOffsetY = maxOffset,
	};
}

void oa::Ui::endScrollPanel() {
	if (!impl_ || impl_->panelStack.empty()
		|| !impl_->panelStack.back().scrollPanel) {
		if (impl_) {
			impl_->setFrameError(oa::Status::error(
				oa::StatusCode::FailedPrecondition,
				"oa::Ui::endScrollPanel requires an active scroll panel"));
		}
		return;
	}
	if (impl_->panelStack.back().explicitRow) {
		impl_->setFrameError(oa::Status::error(
			oa::StatusCode::FailedPrecondition,
			"oa::Ui::endScrollPanel requires the active row to be ended first"));
	}
	impl_->panelStack.popBack();
}

oa::UiVirtualRange oa::Ui::virtualRows(
	oa::I32 inItemCount,
	oa::I32 inRowHeight,
	oa::I32 inRowGap,
	oa::I32 inOverscanRows) const {
	if (!impl_ || impl_->panelStack.empty()
		|| !impl_->panelStack.back().scrollPanel
		|| inItemCount < 0 || inRowHeight <= 0 || inRowGap < 0
		|| inOverscanRows < 0) {
		if (impl_) {
			impl_->setFrameError(oa::Status::invalidArgument(
				"oa::Ui::virtualRows requires an active scroll panel and valid row metrics"));
		}
		return {};
	}
	if (inItemCount == 0) return {};
	const auto& panel = impl_->panelStack.back();
	if (!oa::isFinite(panel.layout.padding.top)) {
		impl_->setFrameError(oa::Status::invalidArgument(
			"oa::Ui::virtualRows requires finite scroll-panel padding"));
		return {};
	}
	const oa::I64 stride = static_cast<oa::I64>(inRowHeight) + inRowGap;
	const oa::I64 paddingTop = static_cast<oa::I64>(oa::clamp<oa::F64>(
		oa::floor(panel.layout.padding.top),
		0.0,
		oa::Limits<oa::I32>::max()));
	const oa::I64 visibleBegin = oa::max<oa::I64>(
		0, static_cast<oa::I64>(panel.scrollOffsetY) - paddingTop);
	const oa::I64 visibleEnd = oa::max(
		visibleBegin,
		static_cast<oa::I64>(panel.scrollOffsetY)
			+ panel.scrollViewport.h - paddingTop);
	const oa::I64 first = oa::max<oa::I64>(
		0, visibleBegin / stride - inOverscanRows);
	const oa::I64 onePastLast = oa::min<oa::I64>(
		inItemCount,
		(visibleEnd + stride - 1) / stride + inOverscanRows);
	return {
		.first = static_cast<oa::I32>(first),
		.onePastLast = static_cast<oa::I32>(onePastLast),
	};
}

oa::UiSplitRegion oa::Ui::splitPane(
	oa::StringView inId,
	oa::PixelRect inRect,
	oa::F32& inOutRatio,
	const oa::UiSplitConfig& inConfig) {
	oa::UiSplitRegion result;
	if (!impl_) return result;
	const bool row = inConfig.direction == oa::UiDirection::Row;
	const bool column = inConfig.direction == oa::UiDirection::Column;
	const oa::I32 axisExtent = row ? inRect.w : inRect.h;
	const oa::I64 minimumExtent = static_cast<oa::I64>(inConfig.minimumFirst)
		+ inConfig.handleSize + inConfig.minimumSecond;
	if (impl_->frameSealed) {
		impl_->setFrameError(oa::Status::error(
			oa::StatusCode::FailedPrecondition,
			"oa::Ui split panes cannot be appended after recordRender"));
		return result;
	}
	if ((!row && !column) || inRect.w <= 0 || inRect.h <= 0
		|| inConfig.handleSize <= 0
		|| inConfig.minimumFirst < 0 || inConfig.minimumSecond < 0
		|| static_cast<oa::I64>(axisExtent) < minimumExtent
		|| !oa::isFinite(inOutRatio)
		|| !oa::isFinite(inConfig.keyboardStep)
		|| inConfig.keyboardStep <= 0.0F) {
		impl_->setFrameError(oa::Status::invalidArgument(
			"oa::Ui::splitPane requires a positive rectangle, valid direction, finite ratio/keyboard step, positive handle, and satisfiable non-negative minima"));
		return result;
	}

	const oa::I32 available = axisExtent - inConfig.handleSize;
	const oa::F32 minimumRatio = static_cast<oa::F32>(inConfig.minimumFirst)
		/ static_cast<oa::F32>(available);
	const oa::F32 maximumRatio = 1.0F
		- static_cast<oa::F32>(inConfig.minimumSecond)
			/ static_cast<oa::F32>(available);
	const auto clampRatio = [&](oa::F32 inRatio) {
		return oa::clamp(inRatio, minimumRatio, maximumRatio);
	};
	const oa::F32 clamped = clampRatio(inOutRatio);
	if (clamped != inOutRatio) {
		inOutRatio = clamped;
		result.changed = true;
	}

	const auto resolveRegions = [&] {
		const oa::I32 firstExtent = oa::clamp<oa::I32>(
			static_cast<oa::I32>(oa::round(
				static_cast<oa::F64>(available) * inOutRatio)),
			inConfig.minimumFirst,
			available - inConfig.minimumSecond);
		if (row) {
			result.first = {inRect.x, inRect.y, firstExtent, inRect.h};
			result.handle = {
				inRect.x + firstExtent,
				inRect.y,
				inConfig.handleSize,
				inRect.h,
			};
			result.second = {
				result.handle.x + result.handle.w,
				inRect.y,
				available - firstExtent,
				inRect.h,
			};
		} else {
			result.first = {inRect.x, inRect.y, inRect.w, firstExtent};
			result.handle = {
				inRect.x,
				inRect.y + firstExtent,
				inRect.w,
				inConfig.handleSize,
			};
			result.second = {
				inRect.x,
				result.handle.y + result.handle.h,
				inRect.w,
				available - firstExtent,
			};
		}
	};
	resolveRegions();

	const oa::U32 id = hashWidgetScope(impl_->currentScope(), inId);
	impl_->registerAdjustable(id);
	const bool wasActive = input_.activeId == id;
	const auto interaction = impl_->interact(input_, id, result.handle);
	if (interaction.held || (wasActive && input_.lReleased)) {
		const oa::F32 pointer = row ? input_.mouseX : input_.mouseY;
		const oa::F32 origin = static_cast<oa::F32>(row ? inRect.x : inRect.y);
		const oa::F32 desiredFirst = pointer - origin
			- static_cast<oa::F32>(inConfig.handleSize) * 0.5F;
		const oa::F32 next = clampRatio(
			desiredFirst / static_cast<oa::F32>(available));
		if (next != inOutRatio) {
			inOutRatio = next;
			result.changed = true;
		}
	}
	if (input_.focusId == id && impl_->keyboardAdjust != 0.0F) {
		// The shared adjustable route defines Up as positive. For a vertical
		// stack, Down grows the first/top region, so invert that axis here.
		const oa::F32 axisDirection = row ? 1.0F : -1.0F;
		const oa::F32 next = clampRatio(
			inOutRatio + impl_->keyboardAdjust
				* axisDirection * inConfig.keyboardStep);
		if (next != inOutRatio) {
			inOutRatio = next;
			result.changed = true;
		}
		impl_->keyboardAdjust = 0.0F;
	}
	resolveRegions();
	impl_->lastItemRect = result.handle;

	const oa::UiStyle& style = currentStyle();
	const bool held = input_.activeId == id;
	this->rect(result.handle,
		held ? style.surfaceActive
			: interaction.hovered ? style.surfaceHover : style.surface);
	const oa::I32 guideSize = oa::min<oa::I32>(
		row ? result.handle.w : result.handle.h,
		held || interaction.hovered || input_.focusId == id ? 2 : 1);
	const oa::Color guideColor = held ? style.accentActive
		: interaction.hovered || input_.focusId == id
			? style.accentHover : style.borderStrong;
	if (row) {
		this->rect({
			result.handle.x + (result.handle.w - guideSize) / 2,
			result.handle.y,
			guideSize,
			result.handle.h,
		}, guideColor);
	} else {
		this->rect({
			result.handle.x,
			result.handle.y + (result.handle.h - guideSize) / 2,
			result.handle.w,
			guideSize,
		}, guideColor);
	}
	impl_->addAccessibilityNode(
		id,
		oa::UiAccessibilityRole::Splitter,
		result.handle,
		inId,
		{},
		oa::UiAccessibilityState::None,
		oa::UiAccessibilityAction::Focus
			| oa::UiAccessibilityAction::Increment
			| oa::UiAccessibilityAction::Decrement
			| oa::UiAccessibilityAction::SetValue,
		input_.focusId == id,
		true,
		minimumRatio,
		maximumRatio,
		inOutRatio);
	return result;
}

oa::UiTabBarResult oa::Ui::tabBar(
	oa::StringView inId,
	oa::PixelRect inRect,
	oa::Span<const oa::UiTabItem> inItems,
	oa::UiTabBarState& inOutState,
	const oa::UiTabBarConfig& inConfig) {
	oa::UiTabBarResult result;
	if (!impl_) return result;
	if (impl_->frameSealed) {
		impl_->setFrameError(oa::Status::error(
			oa::StatusCode::FailedPrecondition,
			"oa::Ui tab bars cannot be appended after recordRender"));
		return result;
	}
	if (inRect.w <= 0 || inRect.h <= 0
		|| inItems.size() > static_cast<oa::Usize>(
			oa::Limits<oa::I32>::max())
		|| inConfig.minimumTabWidth <= 0
		|| inConfig.maximumTabWidth < inConfig.minimumTabWidth
		|| inConfig.overflowButtonWidth <= 0
		|| inConfig.closeWidth <= 0) {
		impl_->setFrameError(oa::Status::invalidArgument(
			"oa::Ui::tabBar requires a positive rectangle, bounded item count, positive metrics, and maximum width not smaller than minimum width"));
		return result;
	}
	const oa::I32 count = static_cast<oa::I32>(inItems.size());
	if (inOutState.selected < -1 || inOutState.selected >= count) {
		impl_->setFrameError(oa::Status::invalidArgument(
			"oa::Ui::tabBar selected index is outside the caller-owned item sequence"));
		return result;
	}
	for (oa::I32 index = 0; index < count; ++index) {
		if (inItems[static_cast<oa::Usize>(index)].id.empty()) {
			impl_->setFrameError(oa::Status::invalidArgument(
				"oa::Ui::tabBar item IDs must be non-empty"));
			return result;
		}
		for (oa::I32 prior = 0; prior < index; ++prior) {
			if (inItems[static_cast<oa::Usize>(index)].id
				== inItems[static_cast<oa::Usize>(prior)].id) {
				impl_->setFrameError(oa::Status::invalidArgument(
					"oa::Ui::tabBar item IDs must be unique within the bar"));
				return result;
			}
		}
	}

	result.bar = inRect;
	const oa::UiStyle& style = currentStyle();
	this->rect(inRect, style.background);
	oa::UiLayout barLayout;
	barLayout.padding = oa::UiEdge{};
	beginPanel(inId, inRect, barLayout);
	const oa::U32 barId = impl_->currentScope();

	if (count == 0) {
		inOutState.firstVisible = 0;
		result.tabs = inRect;
		endPanel();
		return result;
	}

	oa::Vector<oa::U32> itemIds(static_cast<oa::Usize>(count));
	for (oa::I32 index = 0; index < count; ++index) {
		const oa::UiTabItem& item = inItems[static_cast<oa::Usize>(index)];
		const oa::U32 itemId = hashWidgetScope(barId, item.id);
		itemIds[static_cast<oa::Usize>(index)] = itemId;
		if (item.enabled) impl_->registerTab(itemId, false);
		if (impl_->keyboardTabActivateId == itemId) {
			if (inOutState.selected != index) {
				inOutState.selected = index;
				result.selectionChanged = true;
			}
			result.activatedIndex = index;
			impl_->keyboardTabActivateId = 0U;
		}
		if (impl_->keyboardTabCloseId == itemId && item.closable) {
			result.closeRequestedIndex = index;
			impl_->keyboardTabCloseId = 0U;
		}
	}

	const oa::I64 minimumTotal = static_cast<oa::I64>(count)
		* inConfig.minimumTabWidth;
	const bool overflow = minimumTotal > inRect.w;
	oa::PixelRect previousRect;
	oa::PixelRect nextRect;
	result.tabs = inRect;
	if (overflow) {
		const oa::I32 buttonWidth = oa::min<oa::I32>(
			inConfig.overflowButtonWidth,
			oa::max(1, inRect.w / 3));
		previousRect = {inRect.x, inRect.y, buttonWidth, inRect.h};
		nextRect = {
			inRect.x + inRect.w - buttonWidth,
			inRect.y,
			buttonWidth,
			inRect.h,
		};
		result.tabs = {
			previousRect.x + previousRect.w,
			inRect.y,
			oa::max(1, inRect.w - previousRect.w - nextRect.w),
			inRect.h,
		};
	}

	const oa::I32 tabWidth = overflow
		? oa::min(inConfig.minimumTabWidth, result.tabs.w)
		: oa::clamp(
			inRect.w / count,
			inConfig.minimumTabWidth,
			inConfig.maximumTabWidth);
	const oa::I32 visibleCapacity = overflow
		? oa::max(1, result.tabs.w / oa::max(1, tabWidth))
		: count;
	const oa::I32 maximumFirst = oa::max(0, count - visibleCapacity);
	inOutState.firstVisible = oa::clamp(
		inOutState.firstVisible, 0, maximumFirst);
	if (inOutState.selected >= 0) {
		if (inOutState.selected < inOutState.firstVisible) {
			inOutState.firstVisible = inOutState.selected;
		} else if (inOutState.selected
			>= inOutState.firstVisible + visibleCapacity) {
			inOutState.firstVisible = oa::min(
				maximumFirst,
				inOutState.selected - visibleCapacity + 1);
		}
	}

	if (overflow) {
		const auto drawOverflowButton = [&](oa::StringView inButtonId,
			oa::PixelRect inButtonRect, bool inPrevious) {
			const oa::U32 id = hashWidgetScope(barId, inButtonId);
			const auto interaction = impl_->interact(input_, id, inButtonRect);
			this->rect(inButtonRect,
				interaction.held ? style.surfaceActive
					: interaction.hovered ? style.surfaceHover : style.surface);
			const oa::I32 centerX = inButtonRect.x + inButtonRect.w / 2;
			const oa::I32 centerY = inButtonRect.y + inButtonRect.h / 2;
			const oa::Color color = interaction.hovered
				? style.text : style.textMuted;
			const oa::I32 direction = inPrevious ? -1 : 1;
			for (oa::I32 step = 0; step < 4; ++step) {
				this->rect({centerX + direction * (step - 2), centerY - step, 1, 1}, color);
				this->rect({centerX + direction * (step - 2), centerY + step, 1, 1}, color);
			}
			return interaction.activated;
		};
		if (drawOverflowButton("__tab_previous", previousRect, true)) {
			inOutState.firstVisible = oa::max(
				0, inOutState.firstVisible - 1);
		}
		if (drawOverflowButton("__tab_next", nextRect, false)) {
			inOutState.firstVisible = oa::min(
				maximumFirst, inOutState.firstVisible + 1);
		}
	}

	result.firstVisible = inOutState.firstVisible;
	result.onePastLast = oa::min(
		count, result.firstVisible + visibleCapacity);
	oa::Vector<oa::PixelRect> tabRects;
	tabRects.reserve(static_cast<oa::Usize>(
		result.onePastLast - result.firstVisible));
	for (oa::I32 index = result.firstVisible;
		index < result.onePastLast; ++index) {
		const oa::I32 localIndex = index - result.firstVisible;
		const oa::I32 x = result.tabs.x + localIndex * tabWidth;
		const oa::I32 width = oa::min(
			tabWidth, result.tabs.x + result.tabs.w - x);
		tabRects.pushBack({x, result.tabs.y, oa::max(0, width), result.tabs.h});
	}

	oa::I32 dragSource = -1;
	if (impl_->tabDragBarId == barId) {
		for (oa::I32 index = 0; index < count; ++index) {
			if (itemIds[static_cast<oa::Usize>(index)] == impl_->tabDragItemId) {
				dragSource = index;
				break;
			}
		}
	}
	oa::I32 dragTarget = -1;
	if (input_.lReleased && impl_->tabDragBarId == barId) {
		for (oa::I32 index = result.firstVisible;
			index < result.onePastLast; ++index) {
			const oa::PixelRect tabRect = tabRects[static_cast<oa::Usize>(
				index - result.firstVisible)];
			if (impl_->clipFor(tabRect).contains(input_.mouseX, input_.mouseY)) {
				dragTarget = index;
				break;
			}
		}
	}
	const bool reorder = inConfig.reorderable && dragTarget >= 0
		&& dragSource >= 0 && dragTarget != dragSource;

	for (oa::I32 index = result.firstVisible;
		index < result.onePastLast; ++index) {
		const oa::UiTabItem& item = inItems[static_cast<oa::Usize>(index)];
		const oa::U32 itemId = itemIds[static_cast<oa::Usize>(index)];
		const oa::PixelRect tabRect = tabRects[static_cast<oa::Usize>(
			index - result.firstVisible)];
		Impl::ControlInteraction interaction;
		if (item.enabled) {
			impl_->registerTab(itemId, true);
			interaction = impl_->interact(input_, itemId, tabRect);
		} else {
			impl_->lastItemId = itemId;
			impl_->lastItemRect = tabRect;
			impl_->lastItemHovered = false;
		}
		if (item.enabled && interaction.held && input_.lPressed) {
			impl_->tabDragBarId = barId;
			impl_->tabDragItemId = itemId;
		}

		const bool selected = inOutState.selected == index;
		this->rect(tabRect,
			selected ? style.surfaceActive
				: interaction.held ? style.surfaceActive
				: interaction.hovered ? style.surfaceHover : style.surface);
		this->rect({tabRect.x + tabRect.w - 1, tabRect.y, 1, tabRect.h},
			style.borderSubtle);
		if (selected) {
			this->rect({tabRect.x, tabRect.y + oa::max(0, tabRect.h - 2),
				tabRect.w, oa::min(2, tabRect.h)}, style.accent);
		}
		if (item.enabled && input_.focusId == itemId) {
			rectOutline(tabRect, style.accentHover, 1U);
		}

		const oa::I32 closeWidth = item.closable
			? oa::min(inConfig.closeWidth, oa::max(0, tabRect.w / 2)) : 0;
		const oa::PixelRect closeRect{
			tabRect.x + tabRect.w - closeWidth,
			tabRect.y,
			closeWidth,
			tabRect.h,
		};
		const oa::I32 dirtyWidth = item.dirty ? 10 : 0;
		const oa::PixelRect labelRect{
			tabRect.x,
			tabRect.y,
			oa::max(0, tabRect.w - closeWidth - dirtyWidth),
			tabRect.h,
		};
		if (item.dirty && tabRect.w > closeWidth + 4) {
			const oa::I32 centerX = tabRect.x + tabRect.w - closeWidth - 5;
			const oa::I32 centerY = tabRect.y + tabRect.h / 2;
			this->rect({centerX - 2, centerY - 2, 4, 4}, style.warning);
		}
		if (item.closable && closeRect.w >= 8) {
			const oa::I32 centerX = closeRect.x + closeRect.w / 2;
			const oa::I32 centerY = closeRect.y + closeRect.h / 2;
			const oa::Color closeColor = interaction.hovered
				&& closeRect.contains(input_.mouseX, input_.mouseY)
					? style.text : style.textMuted;
			for (oa::I32 step = -2; step <= 2; ++step) {
				this->rect({centerX + step, centerY + step, 1, 1}, closeColor);
				this->rect({centerX + step, centerY - step, 1, 1}, closeColor);
			}
		}
		oa::UiAccessibilityState accessibilityState =
			oa::UiAccessibilityState::None;
		if (!item.enabled) {
			accessibilityState = accessibilityState
				| oa::UiAccessibilityState::Disabled;
		}
		const bool activatesTab = interaction.activated && !reorder
			&& !(item.closable
				&& closeRect.contains(input_.mouseX, input_.mouseY));
		if (selected || activatesTab) {
			accessibilityState = accessibilityState
				| oa::UiAccessibilityState::Selected;
		}
		oa::UiAccessibilityAction accessibilityActions = item.enabled
			? oa::UiAccessibilityAction::Focus
				| oa::UiAccessibilityAction::Activate
			: oa::UiAccessibilityAction::None;
		if (item.enabled && item.closable) {
			accessibilityActions = accessibilityActions
				| oa::UiAccessibilityAction::Close;
		}
		impl_->addAccessibilityNode(
			itemId,
			oa::UiAccessibilityRole::Tab,
			tabRect,
			item.label.empty() ? item.id : item.label,
			{},
			accessibilityState,
			accessibilityActions,
			input_.focusId == itemId);
		if (!item.label.empty() && labelRect.w > 0) {
			oa::UiStyle textStyle = style;
			textStyle.text = item.enabled
				? (selected ? style.text : style.textSecondary)
				: style.textDisabled;
			oa::UiLayout textLayout;
			textLayout.padding = oa::UiEdge{};
			textLayout.padding.left = oa::min<oa::F32>(
				6.0F, static_cast<oa::F32>(labelRect.w));
			textLayout.padding.top = oa::max(
				0.0F,
				(static_cast<oa::F32>(labelRect.h) - style.fontSize) * 0.5F);
			beginPanel(item.id, labelRect, textLayout);
			impl_->appendText(item.label, textStyle, false);
			endPanel();
		}

		if (interaction.activated && !reorder) {
			if (item.closable
				&& closeRect.contains(input_.mouseX, input_.mouseY)) {
				result.closeRequestedIndex = index;
			} else {
				if (inOutState.selected != index) {
					inOutState.selected = index;
					result.selectionChanged = true;
				}
				result.activatedIndex = index;
			}
		}
	}

	if (reorder) {
		result.moveFromIndex = dragSource;
		result.moveToIndex = dragTarget;
	}
	if (input_.lReleased && impl_->tabDragBarId == barId) {
		impl_->tabDragBarId = 0U;
		impl_->tabDragItemId = 0U;
	}
	endPanel();
	return result;
}

oa::UiTreeRowResult oa::Ui::treeRow(
	oa::StringView inId,
	oa::PixelRect inRect,
	oa::StringView inLabel,
	const oa::UiTreeRowConfig& inConfig) {
	oa::UiTreeRowResult result;
	result.open = inConfig.open;
	if (!impl_) return result;
	if (impl_->frameSealed) {
		impl_->setFrameError(oa::Status::error(
			oa::StatusCode::FailedPrecondition,
			"oa::Ui tree rows cannot be appended after recordRender"));
		return result;
	}
	const oa::I64 indentation = static_cast<oa::I64>(inConfig.depth)
		* inConfig.indent;
	if (inRect.w <= 0 || inRect.h <= 0 || inConfig.depth < 0
		|| inConfig.indent <= 0 || inConfig.disclosureWidth <= 0
		|| indentation > oa::Limits<oa::I32>::max()) {
		impl_->setFrameError(oa::Status::invalidArgument(
			"oa::Ui::treeRow requires a positive rectangle, non-negative depth, and positive bounded indent/disclosure metrics"));
		return result;
	}

	result.row = inRect;
	const oa::UiStyle& style = currentStyle();
	const oa::F32 finitePadding = oa::isFinite(style.framePaddingX)
		? oa::max(0.0F, style.framePaddingX) : 0.0F;
	const oa::F64 boundedPadding = oa::min(
		static_cast<oa::F64>(finitePadding),
		static_cast<oa::F64>(inRect.w));
	const oa::I32 padding = static_cast<oa::I32>(
		oa::floor(boundedPadding + 0.5F));
	const oa::I32 disclosureOffset = static_cast<oa::I32>(oa::min<oa::I64>(
		indentation + static_cast<oa::I64>(padding),
		static_cast<oa::I64>(inRect.w)));
	result.disclosure = {
		inRect.x + disclosureOffset,
		inRect.y,
		oa::min(inConfig.disclosureWidth, inRect.w - disclosureOffset),
		inRect.h,
	};
	const oa::I32 labelOffset = disclosureOffset + result.disclosure.w;
	result.label = {
		inRect.x + labelOffset,
		inRect.y,
		oa::max(0, inRect.w - labelOffset - padding),
		inRect.h,
	};

	const oa::U32 id = hashWidgetScope(impl_->currentScope(), inId);
	Impl::ControlInteraction interaction;
	const bool keyboardActivated = inConfig.enabled
		&& input_.focusId == id && impl_->keyboardActivate;
	if (inConfig.enabled) {
		impl_->registerTree(id);
		interaction = impl_->interact(input_, id, inRect);
	} else {
		impl_->lastItemId = id;
		impl_->lastItemRect = inRect;
		impl_->lastItemHovered = false;
	}

	const bool disclosureActivated = inConfig.hasChildren
		&& interaction.activated && !keyboardActivated
		&& result.disclosure.contains(input_.mouseX, input_.mouseY);
	if (disclosureActivated) {
		result.open = !inConfig.open;
		result.openChanged = true;
	} else {
		result.activated = interaction.activated;
	}
	if (inConfig.enabled && impl_->keyboardTreeActivateId == id) {
		result.activated = true;
		impl_->keyboardTreeActivateId = 0U;
	}
	if (inConfig.enabled && input_.focusId == id
		&& impl_->keyboardTreeOpenDirection != 0) {
		if (inConfig.hasChildren) {
			const bool desired = impl_->keyboardTreeOpenDirection > 0;
			if (desired != inConfig.open) {
				result.open = desired;
				result.openChanged = true;
			}
		}
		impl_->keyboardTreeOpenDirection = 0;
	}

	if (inConfig.selected) {
		this->rect(inRect, style.accent.withAlpha(
			interaction.held ? 0.48F : interaction.hovered ? 0.38F : 0.28F));
		this->rect({inRect.x, inRect.y, oa::min(3, inRect.w), inRect.h}, style.accent);
	} else if (interaction.held || interaction.hovered) {
		this->rect(inRect,
			interaction.held ? style.surfaceActive : style.surfaceHover);
	}
	if (inConfig.enabled && input_.focusId == id) {
		rectOutline(inRect, style.accentHover, 1U);
	}

	if (inConfig.hasChildren && result.disclosure.w > 4
		&& result.disclosure.h > 4) {
		const oa::I32 marker = oa::max<oa::I32>(3, oa::min<oa::I32>(
			7, oa::min(result.disclosure.w - 4, result.disclosure.h - 4)));
		const oa::I32 centerX = result.disclosure.x + result.disclosure.w / 2;
		const oa::I32 centerY = result.disclosure.y + result.disclosure.h / 2;
		const oa::Color markerColor = inConfig.enabled
			? style.textSecondary : style.textDisabled;
		this->rect({centerX - marker / 2, centerY, marker, 1}, markerColor);
		if (!inConfig.open) {
			this->rect({centerX, centerY - marker / 2, 1, marker}, markerColor);
		}
	}
	oa::UiAccessibilityState accessibilityState =
		oa::UiAccessibilityState::None;
	if (!inConfig.enabled) {
		accessibilityState = accessibilityState
			| oa::UiAccessibilityState::Disabled;
	}
	if (inConfig.selected) {
		accessibilityState = accessibilityState
			| oa::UiAccessibilityState::Selected;
	}
	if (inConfig.hasChildren && result.open) {
		accessibilityState = accessibilityState
			| oa::UiAccessibilityState::Expanded;
	}
	oa::UiAccessibilityAction accessibilityActions = inConfig.enabled
		? oa::UiAccessibilityAction::Focus | oa::UiAccessibilityAction::Activate
		: oa::UiAccessibilityAction::None;
	if (inConfig.enabled && inConfig.hasChildren) {
		accessibilityActions = accessibilityActions
			| oa::UiAccessibilityAction::Toggle;
	}
	impl_->addAccessibilityNode(
		id,
		oa::UiAccessibilityRole::TreeItem,
		inRect,
		inLabel,
		{},
		accessibilityState,
		accessibilityActions,
		input_.focusId == id);

	if (!inLabel.empty() && result.label.w > 0) {
		oa::UiStyle textStyle = style;
		textStyle.text = inConfig.enabled
			? (inConfig.selected ? style.text : style.textSecondary)
			: style.textDisabled;
		oa::UiLayout textLayout;
		textLayout.padding = oa::UiEdge{};
		textLayout.padding.top = oa::max(
			0.0F,
			(static_cast<oa::F32>(result.label.h) - style.fontSize) * 0.5F);
		beginPanel(inId, result.label, textLayout);
		impl_->appendText(inLabel, textStyle, false);
		endPanel();
	}
	return result;
}

oa::UiPropertyRegion oa::Ui::propertyRow(
	oa::StringView inId,
	oa::PixelRect inRect,
	oa::StringView inLabel,
	oa::StringView inValue,
	const oa::UiPropertyRowConfig& inConfig) {
	oa::UiPropertyRegion result;
	if (!impl_) return result;
	if (impl_->frameSealed) {
		impl_->setFrameError(oa::Status::error(
			oa::StatusCode::FailedPrecondition,
			"oa::Ui property rows cannot be appended after recordRender"));
		return result;
	}
	const oa::I64 distributableWide = static_cast<oa::I64>(inRect.w)
		- static_cast<oa::I64>(inConfig.gap);
	if (inRect.w <= 0 || inRect.h <= 0
		|| !oa::isFinite(inConfig.labelFraction)
		|| inConfig.labelFraction <= 0.0F || inConfig.labelFraction >= 1.0F
		|| inConfig.gap < 0 || inConfig.paddingX < 0
		|| distributableWide < 2
		|| distributableWide > oa::Limits<oa::I32>::max()) {
		impl_->setFrameError(oa::Status::invalidArgument(
			"oa::Ui::propertyRow requires a positive rectangle, label fraction inside (0,1), non-negative gap/padding, and room for both columns"));
		return result;
	}
	const oa::I32 distributable = static_cast<oa::I32>(distributableWide);

	result.row = inRect;
	const oa::I32 labelWidth = oa::clamp<oa::I32>(
		static_cast<oa::I32>(oa::round(
			static_cast<oa::F64>(distributable) * inConfig.labelFraction)),
		1,
		distributable - 1);
	result.label = {inRect.x, inRect.y, labelWidth, inRect.h};
	result.value = {
		inRect.x + labelWidth + inConfig.gap,
		inRect.y,
		distributable - labelWidth,
		inRect.h,
	};

	const oa::UiStyle& style = currentStyle();
	if (inConfig.alternate) {
		this->rect(inRect, style.surface.withAlpha(0.58F));
	}
	const oa::I32 separatorX = result.label.x + result.label.w
		+ inConfig.gap / 2;
	this->rect({separatorX, inRect.y, 1, inRect.h}, style.borderSubtle);

	oa::UiLayout rootLayout;
	rootLayout.padding = oa::UiEdge{};
	beginPanel(inId, inRect, rootLayout);
	const auto drawCell = [&](oa::StringView inCellId, oa::PixelRect inCell,
		oa::StringView inText, oa::Color inColor) {
		if (inText.empty() || inCell.w <= 0 || inCell.h <= 0) return;
		oa::UiStyle textStyle = style;
		textStyle.text = inColor;
		oa::UiLayout textLayout;
		textLayout.padding = oa::UiEdge{};
		textLayout.padding.left = static_cast<oa::F32>(oa::min(
			inConfig.paddingX, inCell.w));
		textLayout.padding.right = textLayout.padding.left;
		textLayout.padding.top = oa::max(
			0.0F,
			(static_cast<oa::F32>(inCell.h) - style.fontSize) * 0.5F);
		beginPanel(inCellId, inCell, textLayout);
		impl_->appendText(inText, textStyle, false);
		endPanel();
	};
	drawCell("__property_label", result.label, inLabel, style.textMuted);
	drawCell("__property_value", result.value, inValue, style.textSecondary);
	endPanel();
	return result;
}

void oa::Ui::beginRow(oa::StringView inId) {
	if (!impl_) return;
	if (impl_->frameSealed) {
		impl_->setFrameError(oa::Status::error(
			oa::StatusCode::FailedPrecondition,
			"oa::Ui rows cannot begin after recordRender"));
		return;
	}
	if (impl_->panelStack.empty()) {
		impl_->setFrameError(oa::Status::error(
			oa::StatusCode::FailedPrecondition,
			"oa::Ui::beginRow requires an active panel"));
		return;
	}
	auto& panel = impl_->panelStack.back();
	if (panel.layout.direction == oa::UiDirection::Row || panel.explicitRow) {
		impl_->setFrameError(oa::Status::error(
			oa::StatusCode::FailedPrecondition,
			"oa::Ui rows cannot be nested"));
		return;
	}
	panel.explicitRow = true;
	panel.rowX = panel.layout.padding.left;
	panel.rowY = panel.cursor;
	panel.rowHeight = 0.0F;
	panel.rowScope = inId.empty()
		? hashWidgetIndex(panel.scope, panel.nextAnonymousRow++)
		: hashWidgetScope(panel.scope, inId);
}

void oa::Ui::endRow() {
	if (!impl_) return;
	if (impl_->panelStack.empty() || !impl_->panelStack.back().explicitRow) {
		impl_->setFrameError(oa::Status::error(
			oa::StatusCode::FailedPrecondition,
			"oa::Ui::endRow requires an active explicit row"));
		return;
	}
	auto& panel = impl_->panelStack.back();
	panel.cursor = panel.rowY + panel.rowHeight + panel.layout.gap;
	panel.explicitRow = false;
	panel.rowHeight = 0.0F;
}
void oa::Ui::spacing(oa::F32 inPixels) {
	if (!impl_ || impl_->panelStack.empty()) return;
	auto& panel = impl_->panelStack.back();
	if (impl_->isRow(panel)) {
		panel.rowX += oa::max(0.0F, inPixels);
	} else {
		panel.cursor += oa::max(0.0F, inPixels);
	}
}
void oa::Ui::separator() {
	if (!impl_) return;
	const oa::UiStyle& style = currentStyle();
	const bool row = !impl_->panelStack.empty()
		&& impl_->isRow(impl_->panelStack.back());
	const oa::F32 rowHeight = oa::max(style.fontSize, 1.0F)
		+ oa::max(0.0F, style.framePaddingY) * 2.0F;
	const oa::PixelRect rect = impl_->placeItem(1.0F, row ? rowHeight : 1.0F);
	if (rect.w > 0 && rect.h > 0) this->rect(rect, style.borderStrong);
}


// ─── top-layer overlays ──────────────────────────────────────────────────────

void oa::Ui::openPopup(oa::StringView inId) {
	if (!impl_) return;
	if (impl_->frameSealed || impl_->recordingOverlay || impl_->lastItemId == 0U
		|| impl_->lastItemRect.w <= 0 || impl_->lastItemRect.h <= 0) {
		impl_->setFrameError(oa::Status::error(
			oa::StatusCode::FailedPrecondition,
			"oa::Ui::openPopup requires the immediately preceding interactive item"));
		return;
	}
	impl_->openPopup(
		hashWidgetScope(impl_->currentScope(), inId),
		impl_->lastItemId,
		impl_->lastItemRect);
}

void oa::Ui::openPopup(oa::StringView inId, oa::PixelRect inAnchor) {
	if (!impl_) return;
	if (impl_->frameSealed || impl_->recordingOverlay
		|| inAnchor.w <= 0 || inAnchor.h <= 0) {
		impl_->setFrameError(oa::Status::invalidArgument(
			"oa::Ui::openPopup requires a positive anchor before recordRender"));
		return;
	}
	impl_->openPopup(
		hashWidgetScope(impl_->currentScope(), inId),
		0U,
		inAnchor);
}

void oa::Ui::closePopup() {
	if (!impl_) return;
	impl_->closePopup(input_, true);
}

bool oa::Ui::isPopupOpen(oa::StringView inId) const noexcept {
	return impl_ && impl_->openPopupId
		== hashWidgetScope(impl_->currentScope(), inId);
}

bool oa::Ui::beginPopup(
	oa::StringView inId,
	const oa::UiPopupConfig& inConfig) {
	if (!impl_) return false;
	const oa::U32 id = hashWidgetScope(impl_->currentScope(), inId);
	if (impl_->openPopupId != id) return false;
	const auto finiteEdge = [](const oa::UiEdge& inEdge) noexcept {
		return oa::isFinite(inEdge.top) && oa::isFinite(inEdge.right)
			&& oa::isFinite(inEdge.bottom) && oa::isFinite(inEdge.left)
			&& inEdge.top >= 0.0F && inEdge.right >= 0.0F
			&& inEdge.bottom >= 0.0F && inEdge.left >= 0.0F;
	};
	if (impl_->frameSealed || impl_->recordingOverlay
		|| inConfig.width <= 0 || inConfig.height <= 0 || inConfig.gap < 0
		|| !finiteEdge(inConfig.padding)) {
		impl_->setFrameError(oa::Status::invalidArgument(
			"oa::Ui::beginPopup requires positive dimensions, a non-negative gap/padding, and no active overlay"));
		return false;
	}
	if (impl_->frameViewport.w <= 0 || impl_->frameViewport.h <= 0) {
		impl_->setFrameError(oa::Status::error(
			oa::StatusCode::FailedPrecondition,
			"oa::Ui popups require beginFrame to receive the render viewport"));
		return false;
	}

	const oa::PixelRect rect = impl_->placePopup(impl_->popupAnchorRect, inConfig);
	if (rect.w <= 0 || rect.h <= 0) return false;
	impl_->popupRect = rect;
	impl_->popupRenderedThisFrame = true;
	impl_->recordingOverlay = true;
	impl_->renderingPopupId = id;
	impl_->popupPanelDepth = impl_->panelStack.size();

	Impl::PanelState panel;
	panel.rect = rect;
	panel.clip = intersectPixelRects(
		clipToNonNegative(rect), impl_->frameViewport);
	panel.layout.padding = inConfig.padding;
	panel.layout.gap = oa::max(0.0F, currentStyle().itemSpacing);
	panel.cursor = panel.layout.padding.top;
	panel.rowX = panel.layout.padding.left;
	panel.rowY = panel.layout.padding.top;
	panel.scope = id;
	panel.rowScope = id;
	impl_->panelStack.pushBack(panel);

	const oa::UiStyle& style = currentStyle();
	this->rect(rect, style.background.withAlpha(0.98F));
	if (rect.w > 4 && rect.h > 4) {
		this->rect({rect.x + 2, rect.y + rect.h - 3, rect.w - 4, 2},
			oa::Color{0.0F, 0.0F, 0.0F, 0.40F});
	}
	rectOutline(rect, style.borderStrong, 1U);
	return true;
}

void oa::Ui::endPopup() {
	if (!impl_ || impl_->renderingPopupId == 0U) {
		if (impl_) {
			impl_->setFrameError(oa::Status::error(
				oa::StatusCode::FailedPrecondition,
				"oa::Ui::endPopup requires an active popup"));
		}
		return;
	}
	if (impl_->panelStack.size() != impl_->popupPanelDepth + 1U) {
		impl_->setFrameError(oa::Status::error(
			oa::StatusCode::FailedPrecondition,
			"oa::Ui::endPopup requires every nested panel or row to be ended first"));
	}
	while (impl_->panelStack.size() > impl_->popupPanelDepth) {
		impl_->panelStack.popBack();
	}
	const oa::U32 popupId = impl_->renderingPopupId;
	impl_->renderingPopupId = 0U;
	impl_->recordingOverlay = false;
	if (impl_->popupOpenedThisFrame && impl_->openPopupId == popupId
		&& !impl_->popupFocusOrder.empty()) {
		input_.focusId = impl_->popupFocusOrder[0];
		impl_->popupOpenedThisFrame = false;
	}
}

bool oa::Ui::menuItem(
	oa::StringView inLabel,
	bool inSelected,
	bool inEnabled) {
	if (!impl_) return false;
	if (!impl_->recordingOverlay || impl_->renderingPopupId == 0U) {
		impl_->setFrameError(oa::Status::error(
			oa::StatusCode::FailedPrecondition,
			"oa::Ui::menuItem requires an active popup"));
		return false;
	}
	const oa::UiStyle& style = currentStyle();
	const oa::vlm::Vec2 textExtent = impl_->measureText(inLabel, style);
	const oa::F32 padX = oa::max(0.0F, style.framePaddingX);
	const oa::F32 padY = oa::max(0.0F, style.framePaddingY);
	const oa::F32 height = oa::max(style.fontSize, textExtent.y) + padY * 2.0F;
	const oa::PixelRect rect = impl_->placeItem(
		oa::max(height, textExtent.x + padX * 2.0F), height);
	if (rect.w <= 0 || rect.h <= 0) return false;

	const oa::U32 id = hashWidgetScope(impl_->currentScope(), inLabel);
	Impl::ControlInteraction interaction;
	if (inEnabled) interaction = impl_->interact(input_, id, rect);
	else {
		impl_->lastItemId = id;
		impl_->lastItemRect = rect;
		impl_->lastItemHovered = false;
	}
	if (inSelected) {
		this->rect(rect, style.accent.withAlpha(interaction.hovered ? 0.35F : 0.22F));
		this->rect({rect.x, rect.y, oa::min(3, rect.w), rect.h}, style.accent);
	} else if (interaction.held || interaction.hovered) {
		this->rect(rect, interaction.held ? style.surfaceActive : style.surfaceHover);
	}
	oa::UiAccessibilityState accessibilityState = inSelected
		? oa::UiAccessibilityState::Selected
		: oa::UiAccessibilityState::None;
	if (!inEnabled) {
		accessibilityState = accessibilityState
			| oa::UiAccessibilityState::Disabled;
	}
	impl_->addAccessibilityNode(
		id,
		oa::UiAccessibilityRole::MenuItem,
		rect,
		inLabel,
		{},
		accessibilityState,
		inEnabled
			? oa::UiAccessibilityAction::Focus
				| oa::UiAccessibilityAction::Activate
			: oa::UiAccessibilityAction::None,
		input_.focusId == id);

	oa::UiStyle textStyle = style;
	textStyle.text = inEnabled ? style.text : style.textDisabled;
	oa::UiLayout textLayout;
	textLayout.padding = oa::UiEdge{};
	textLayout.padding.top = oa::max(
		0.0F, (static_cast<oa::F32>(rect.h) - textExtent.y) * 0.5F);
	textLayout.padding.left = padX + (inSelected ? 4.0F : 0.0F);
	beginPanel("__menu_item_text", rect, textLayout);
	impl_->appendText(inLabel, textStyle, false);
	endPanel();
	if (interaction.activated) impl_->closePopup(input_, true);
	return interaction.activated;
}


// ─── Widgets ──────────────────────────────────────────────────────────────────

namespace {

void drawSliderVisual(
	oa::Ui& inUi,
	oa::StringView inLabel,
	oa::StringView inValueText,
	oa::PixelRect inRect,
	oa::F32 inFraction,
	oa::vlm::Vec2 inLabelExtent,
	oa::vlm::Vec2 inValueExtent,
	bool inHovered,
	bool inHeld,
	bool inFocused) {
	if (inRect.w <= 0 || inRect.h <= 0) return;
	const oa::UiStyle& style = inUi.currentStyle();
	const oa::F32 fraction = oa::clamp(inFraction, 0.0F, 1.0F);
	inUi.rect(
		inRect,
		inHeld ? style.surfaceActive
			: inHovered ? style.surfaceHover : style.surface);
	const oa::I32 fillWidth = oa::clamp(
		static_cast<oa::I32>(oa::floor(
			static_cast<oa::F32>(inRect.w) * fraction + 0.5F)),
		0,
		inRect.w);
	if (fillWidth > 0) {
		inUi.rect(
			{inRect.x, inRect.y, fillWidth, inRect.h},
			style.accent.withAlpha(inHeld ? 0.82F : 0.62F));
	}
	const oa::I32 grabWidth = oa::min<oa::I32>(3, inRect.w);
	const oa::I32 grabTravel = oa::max<oa::I32>(0, inRect.w - grabWidth);
	const oa::I32 grabX = inRect.x + static_cast<oa::I32>(oa::floor(
		static_cast<oa::F32>(grabTravel) * fraction + 0.5F));
	inUi.rect(
		{grabX, inRect.y, grabWidth, inRect.h},
		inHeld ? style.text : style.accentHover);
	const oa::U32 borderWidth = static_cast<oa::U32>(oa::clamp(
		oa::isFinite(style.borderWidth)
			? oa::floor(style.borderWidth + 0.5F) : 1.0F,
		1.0F,
		8.0F));
	inUi.rectOutline(
		inRect,
		inFocused ? style.accentHover : style.border,
		borderWidth);

	const oa::F32 padX = oa::max(0.0F, style.framePaddingX);
	const oa::F32 innerGap = oa::max(4.0F, style.itemSpacing);
	const oa::F32 valueStart = oa::max(
		padX,
		static_cast<oa::F32>(inRect.w) - padX - inValueExtent.x);
	const oa::F32 labelStart = padX;
	const oa::F32 labelWidth = oa::max(
		0.0F,
		valueStart - innerGap - labelStart);
	const oa::F32 textTop = oa::max(
		0.0F,
		(static_cast<oa::F32>(inRect.h)
			- oa::max(inLabelExtent.y, inValueExtent.y)) * 0.5F);
	oa::UiLayout textLayout;
	textLayout.padding = oa::UiEdge{};
	textLayout.padding.top = textTop;
	const oa::PixelRect labelRect{
		inRect.x + static_cast<oa::I32>(oa::floor(labelStart + 0.5F)),
		inRect.y,
		oa::max<oa::I32>(0, static_cast<oa::I32>(oa::floor(labelWidth))),
		inRect.h,
	};
	if (!inLabel.empty() && labelRect.w > 0) {
		inUi.beginPanel("__slider_label", labelRect, textLayout);
		inUi.label(inLabel);
		inUi.endPanel();
	}
	const oa::I32 valueOffset = oa::clamp(
		static_cast<oa::I32>(oa::floor(valueStart + 0.5F)),
		0,
		inRect.w);
	const oa::PixelRect valueRect{
		inRect.x + valueOffset,
		inRect.y,
		inRect.w - valueOffset,
		inRect.h,
	};
	if (!inValueText.empty() && valueRect.w > 0) {
		inUi.beginPanel("__slider_value", valueRect, textLayout);
		inUi.label(inValueText);
		inUi.endPanel();
	}
}

} // namespace

bool oa::Ui::button(oa::StringView inLabel) {
	if (!impl_) return false;
	const oa::UiStyle& style = currentStyle();
	const oa::vlm::Vec2 textExtent = impl_->measureText(inLabel, style);
	const oa::F32 padX = oa::max(0.0F, style.framePaddingX);
	const oa::F32 padY = oa::max(0.0F, style.framePaddingY);
	const oa::F32 height = oa::max(style.fontSize, textExtent.y) + padY * 2.0F;
	const oa::F32 hugWidth = oa::max(height, textExtent.x + padX * 2.0F);
	const oa::U32 id = hashWidgetScope(impl_->currentScope(), inLabel);
	const oa::PixelRect rect = impl_->placeItem(hugWidth, height);
	if (rect.w <= 0 || rect.h <= 0) return false;

	const auto interaction = impl_->interact(input_, id, rect);
	const oa::Color fill = interaction.held || interaction.activated
		? style.accentActive
		: interaction.hovered ? style.surfaceActive : style.surfaceHover;
	this->rect(rect, fill);
	const oa::U32 borderWidth = static_cast<oa::U32>(oa::clamp(
		oa::isFinite(style.borderWidth) ? oa::floor(style.borderWidth + 0.5F) : 1.0F,
		1.0F,
		8.0F));
	rectOutline(
		rect,
		input_.focusId == id ? style.accentHover : style.border,
		borderWidth);
	impl_->addAccessibilityNode(
		id,
		oa::UiAccessibilityRole::Button,
		rect,
		inLabel,
		{},
		oa::UiAccessibilityState::None,
		oa::UiAccessibilityAction::Focus | oa::UiAccessibilityAction::Activate,
		input_.focusId == id);

	oa::UiLayout textLayout;
	textLayout.padding = oa::UiEdge{};
	textLayout.padding.top = oa::max(
		0.0F,
		(static_cast<oa::F32>(rect.h) - textExtent.y) * 0.5F);
	textLayout.padding.left = oa::max(
		0.0F,
		(static_cast<oa::F32>(rect.w) - textExtent.x) * 0.5F);
	beginPanel("__button_text", rect, textLayout);
	impl_->appendText(inLabel, style, false);
	endPanel();
	return interaction.activated;
}

bool oa::Ui::chevronButton(
	oa::StringView inLabel,
	oa::PixelRect inRect,
	oa::UiChevronDirection inDirection,
	bool inEnabled) {
	if (!impl_) return false;
	if (!inRect.isValid()
		|| (inDirection != oa::UiChevronDirection::Previous
			&& inDirection != oa::UiChevronDirection::Next)) {
		impl_->setFrameError(oa::Status::invalidArgument(
			"oa::Ui::chevronButton requires a valid rectangle and direction"));
		return false;
	}

	const oa::U32 id = hashWidgetScope(impl_->currentScope(), inLabel);
	Impl::ControlInteraction interaction;
	if (inEnabled) {
		interaction = impl_->interact(input_, id, inRect);
	} else {
		impl_->lastItemId = id;
		impl_->lastItemRect = inRect;
		impl_->lastItemHovered = false;
	}
	const oa::F32 radius = 0.5F
		* static_cast<oa::F32>(oa::min(inRect.w, inRect.h));
	const oa::Color fill = !inEnabled
		? oa::Color{0.04F, 0.04F, 0.04F, 0.42F}
		: interaction.held || interaction.activated
			? oa::Color{0.04F, 0.04F, 0.04F, 0.90F}
			: interaction.hovered
				? oa::Color{0.04F, 0.04F, 0.04F, 0.80F}
				: oa::Color{0.04F, 0.04F, 0.04F, 0.68F};
	this->rect(inRect, fill, radius);
	rectOutline(
		inRect,
		input_.focusId == id
			? currentStyle().accentHover
			: oa::Color{1.0F, 1.0F, 1.0F, interaction.hovered ? 0.28F : 0.16F},
		1U,
		radius);

	const oa::F32 centerX = static_cast<oa::F32>(inRect.x)
		+ 0.5F * static_cast<oa::F32>(inRect.w);
	const oa::F32 centerY = static_cast<oa::F32>(inRect.y)
		+ 0.5F * static_cast<oa::F32>(inRect.h);
	const oa::F32 span = oa::max(
		2.0F,
		0.18F * static_cast<oa::F32>(oa::min(inRect.w, inRect.h)));
	const oa::F32 direction = inDirection == oa::UiChevronDirection::Previous
		? -1.0F : 1.0F;
	const oa::vlm::Vec2 tip{centerX + direction * span, centerY};
	const oa::F32 tailX = centerX - direction * span;
	const oa::Color icon = inEnabled
		? oa::Color{1.0F, 1.0F, 1.0F, 0.94F}
		: oa::Color{1.0F, 1.0F, 1.0F, 0.38F};
	const oa::F32 thickness = oa::max(
		1.5F,
		0.055F * static_cast<oa::F32>(oa::min(inRect.w, inRect.h)));
	line({tailX, centerY - span}, tip, icon, thickness);
	line(tip, {tailX, centerY + span}, icon, thickness);

	const oa::UiAccessibilityState state = inEnabled
		? oa::UiAccessibilityState::None
		: oa::UiAccessibilityState::Disabled;
	impl_->addAccessibilityNode(
		id,
		oa::UiAccessibilityRole::Button,
		inRect,
		inLabel,
		{},
		state,
		inEnabled
			? oa::UiAccessibilityAction::Focus
				| oa::UiAccessibilityAction::Activate
			: oa::UiAccessibilityAction::None,
		input_.focusId == id);
	return inEnabled && interaction.activated;
}

bool oa::Ui::checkbox(oa::StringView inLabel, bool& inOutValue) {
	if (!impl_) return false;
	const oa::UiStyle& style = currentStyle();
	const oa::vlm::Vec2 textExtent = impl_->measureText(inLabel, style);
	const oa::F32 padX = oa::max(0.0F, style.framePaddingX);
	const oa::F32 padY = oa::max(0.0F, style.framePaddingY);
	const oa::F32 boxSize = oa::max(12.0F, oa::ceil(style.fontSize));
	const oa::F32 innerGap = oa::max(4.0F, style.itemSpacing);
	const oa::F32 height = oa::max(boxSize, textExtent.y) + padY * 2.0F;
	const oa::F32 hugWidth = padX * 2.0F + boxSize + innerGap + textExtent.x;
	const oa::U32 id = hashWidgetScope(impl_->currentScope(), inLabel);
	const oa::PixelRect rect = impl_->placeItem(hugWidth, height);
	if (rect.w <= 0 || rect.h <= 0) return false;

	const auto interaction = impl_->interact(input_, id, rect);
	if (interaction.activated) inOutValue = !inOutValue;
	if (interaction.hovered || interaction.held) {
		this->rect(rect, style.surfaceHover.withAlpha(interaction.held ? 0.90F : 0.55F));
	}
	if (input_.focusId == id) rectOutline(rect, style.accentHover, 1U);

	const oa::I32 box = oa::max<oa::I32>(1, static_cast<oa::I32>(boxSize + 0.5F));
	const oa::PixelRect boxRect{
		rect.x + static_cast<oa::I32>(padX + 0.5F),
		rect.y + oa::max<oa::I32>(0, (rect.h - box) / 2),
		oa::min(box, rect.w),
		oa::min(box, rect.h),
	};
	this->rect(boxRect, inOutValue ? style.accent : style.surfaceActive);
	rectOutline(boxRect, interaction.hovered ? style.borderStrong : style.border, 1U);
	if (inOutValue && boxRect.w >= 8 && boxRect.h >= 8) {
		const oa::F32 x = static_cast<oa::F32>(boxRect.x);
		const oa::F32 y = static_cast<oa::F32>(boxRect.y);
		const oa::F32 w = static_cast<oa::F32>(boxRect.w);
		const oa::F32 h = static_cast<oa::F32>(boxRect.h);
		line({x + w * 0.20F, y + h * 0.52F},
			{x + w * 0.43F, y + h * 0.75F}, style.text, 1.5F);
		line({x + w * 0.43F, y + h * 0.75F},
			{x + w * 0.82F, y + h * 0.27F}, style.text, 1.5F);
	}
	impl_->addAccessibilityNode(
		id,
		oa::UiAccessibilityRole::Checkbox,
		rect,
		inLabel,
		inOutValue ? oa::StringView("true") : oa::StringView("false"),
		inOutValue
			? oa::UiAccessibilityState::Checked
			: oa::UiAccessibilityState::None,
		oa::UiAccessibilityAction::Focus | oa::UiAccessibilityAction::Toggle,
		input_.focusId == id);

	oa::UiLayout textLayout;
	textLayout.padding = oa::UiEdge{};
	textLayout.padding.top = oa::max(
		0.0F,
		(static_cast<oa::F32>(rect.h) - textExtent.y) * 0.5F);
	textLayout.padding.left = oa::min(
		static_cast<oa::F32>(rect.w),
		padX + boxSize + innerGap);
	beginPanel("__checkbox_text", rect, textLayout);
	impl_->appendText(inLabel, style, false);
	endPanel();
	return interaction.activated;
}
bool oa::Ui::sliderF32(
	oa::StringView inLabel,
	oa::F32* inOutValue,
	oa::F32 inMin,
	oa::F32 inMax,
	const char* inFmt) {
	if (!impl_) return false;
	if (inOutValue == nullptr) {
		impl_->setFrameError(oa::Status::invalidArgument(
			"oa::Ui::sliderF32 requires a value pointer"));
		return false;
	}
	if (!oa::isFinite(inMin) || !oa::isFinite(inMax)
		|| !oa::isFinite(*inOutValue) || inMin > inMax) {
		impl_->setFrameError(oa::Status::invalidArgument(
			"oa::Ui::sliderF32 requires finite values and min <= max"));
		return false;
	}
	if (!isSafeFloatFormat(inFmt)) {
		impl_->setFrameError(oa::Status::invalidArgument(
			"oa::Ui::sliderF32 requires one safe floating-point format conversion"));
		return false;
	}

	bool changed = false;
	const oa::F32 clamped = oa::clamp(*inOutValue, inMin, inMax);
	if (*inOutValue != clamped) {
		*inOutValue = clamped;
		changed = true;
	}
	oa::Array<char, 128> valueBuffer{};
	auto formatValue = [&](oa::F32 inValue) -> oa::StringView {
		const int written = ::snprintf(
			valueBuffer.data(), valueBuffer.size(), inFmt,
			static_cast<double>(inValue));
		if (written < 0) {
			impl_->setFrameError(oa::Status::invalidArgument(
				"oa::Ui::sliderF32 could not format its value"));
			valueBuffer[0] = '?';
			valueBuffer[1] = '\0';
			return oa::StringView(valueBuffer.data(), 1U);
		}
		if (static_cast<oa::Usize>(written) >= valueBuffer.size()) {
			impl_->setFrameError(oa::Status::error(
				oa::StatusCode::ResourceExhausted,
				"oa::Ui::sliderF32 formatted value exceeds 127 bytes"));
			valueBuffer[0] = '?';
			valueBuffer[1] = '\0';
			return oa::StringView(valueBuffer.data(), 1U);
		}
		return oa::StringView(
			valueBuffer.data(), static_cast<oa::Usize>(written));
	};

	const oa::UiStyle& style = currentStyle();
	const oa::vlm::Vec2 labelExtent = impl_->measureText(inLabel, style);
	oa::StringView valueText = formatValue(*inOutValue);
	oa::vlm::Vec2 valueExtent = impl_->measureText(valueText, style);
	const oa::F32 padX = oa::max(0.0F, style.framePaddingX);
	const oa::F32 padY = oa::max(0.0F, style.framePaddingY);
	const oa::F32 height = oa::max(
		style.fontSize,
		oa::max(labelExtent.y, valueExtent.y)) + padY * 2.0F;
	const oa::F32 hugWidth = oa::max(
		180.0F,
		labelExtent.x + valueExtent.x + padX * 2.0F
			+ oa::max(4.0F, style.itemSpacing));
	const oa::U32 id = hashWidgetScope(impl_->currentScope(), inLabel);
	const oa::PixelRect rect = impl_->placeItem(hugWidth, height);
	if (rect.w <= 0 || rect.h <= 0) return changed;
	impl_->registerAdjustable(id);
	const auto interaction = impl_->interact(input_, id, rect);
	const double range = static_cast<double>(inMax) - static_cast<double>(inMin);
	const bool pointerAdjust = interaction.held
		|| (interaction.hovered && input_.lPressed);
	if (pointerAdjust && range > 0.0) {
		const double denominator = static_cast<double>(oa::max(1, rect.w - 1));
		const double fraction = oa::clamp(
			(static_cast<double>(input_.mouseX) - static_cast<double>(rect.x))
				/ denominator,
			0.0,
			1.0);
		const oa::F32 next = static_cast<oa::F32>(
			static_cast<double>(inMin) + range * fraction);
		if (*inOutValue != next) {
			*inOutValue = next;
			changed = true;
		}
	}
	if (input_.focusId == id && impl_->keyboardAdjust != 0.0F) {
		const double next = oa::clamp(
			static_cast<double>(*inOutValue)
				+ range * 0.01 * static_cast<double>(impl_->keyboardAdjust),
			static_cast<double>(inMin),
			static_cast<double>(inMax));
		const oa::F32 adjusted = static_cast<oa::F32>(next);
		if (*inOutValue != adjusted) {
			*inOutValue = adjusted;
			changed = true;
		}
		impl_->keyboardAdjust = 0.0F;
	}
	valueText = formatValue(*inOutValue);
	valueExtent = impl_->measureText(valueText, style);
	const oa::F32 fraction = range > 0.0
		? static_cast<oa::F32>(
			(static_cast<double>(*inOutValue) - static_cast<double>(inMin)) / range)
		: 0.0F;
	drawSliderVisual(
		*this, inLabel, valueText, rect, fraction,
		labelExtent, valueExtent,
		interaction.hovered, interaction.held, input_.focusId == id);
	impl_->addAccessibilityNode(
		id,
		oa::UiAccessibilityRole::Slider,
		rect,
		inLabel,
		valueText,
		oa::UiAccessibilityState::None,
		oa::UiAccessibilityAction::Focus
			| oa::UiAccessibilityAction::Increment
			| oa::UiAccessibilityAction::Decrement
			| oa::UiAccessibilityAction::SetValue,
		input_.focusId == id,
		true,
		inMin,
		inMax,
		*inOutValue);
	return changed;
}

bool oa::Ui::sliderI32(
	oa::StringView inLabel,
	oa::I32* inOutValue,
	oa::I32 inMin,
	oa::I32 inMax) {
	if (!impl_) return false;
	if (inOutValue == nullptr) {
		impl_->setFrameError(oa::Status::invalidArgument(
			"oa::Ui::sliderI32 requires a value pointer"));
		return false;
	}
	if (inMin > inMax) {
		impl_->setFrameError(oa::Status::invalidArgument(
			"oa::Ui::sliderI32 requires min <= max"));
		return false;
	}
	bool changed = false;
	const oa::I32 clamped = oa::clamp(*inOutValue, inMin, inMax);
	if (*inOutValue != clamped) {
		*inOutValue = clamped;
		changed = true;
	}
	oa::Array<char, 32> valueBuffer{};
	auto formatValue = [&]() -> oa::StringView {
		const int written = ::snprintf(
			valueBuffer.data(), valueBuffer.size(), "%d",
			static_cast<int>(*inOutValue));
		if (written < 0 || static_cast<oa::Usize>(written) >= valueBuffer.size()) {
			impl_->setFrameError(oa::Status::error(
				oa::StatusCode::Internal,
				"oa::Ui::sliderI32 could not format its value"));
			valueBuffer[0] = '?';
			valueBuffer[1] = '\0';
			return oa::StringView(valueBuffer.data(), 1U);
		}
		return oa::StringView(
			valueBuffer.data(), static_cast<oa::Usize>(written));
	};

	const oa::UiStyle& style = currentStyle();
	const oa::vlm::Vec2 labelExtent = impl_->measureText(inLabel, style);
	oa::StringView valueText = formatValue();
	oa::vlm::Vec2 valueExtent = impl_->measureText(valueText, style);
	const oa::F32 padX = oa::max(0.0F, style.framePaddingX);
	const oa::F32 padY = oa::max(0.0F, style.framePaddingY);
	const oa::F32 height = oa::max(
		style.fontSize,
		oa::max(labelExtent.y, valueExtent.y)) + padY * 2.0F;
	const oa::F32 hugWidth = oa::max(
		180.0F,
		labelExtent.x + valueExtent.x + padX * 2.0F
			+ oa::max(4.0F, style.itemSpacing));
	const oa::U32 id = hashWidgetScope(impl_->currentScope(), inLabel);
	const oa::PixelRect rect = impl_->placeItem(hugWidth, height);
	if (rect.w <= 0 || rect.h <= 0) return changed;
	impl_->registerAdjustable(id);
	const auto interaction = impl_->interact(input_, id, rect);
	const oa::I64 range = static_cast<oa::I64>(inMax) - static_cast<oa::I64>(inMin);
	const bool pointerAdjust = interaction.held
		|| (interaction.hovered && input_.lPressed);
	if (pointerAdjust && range > 0) {
		const double denominator = static_cast<double>(oa::max(1, rect.w - 1));
		const double fraction = oa::clamp(
			(static_cast<double>(input_.mouseX) - static_cast<double>(rect.x))
				/ denominator,
			0.0,
			1.0);
		const oa::I64 next = oa::clamp<oa::I64>(
			static_cast<oa::I64>(inMin)
				+ static_cast<oa::I64>(oa::round(static_cast<double>(range) * fraction)),
			static_cast<oa::I64>(inMin),
			static_cast<oa::I64>(inMax));
		const oa::I32 adjusted = static_cast<oa::I32>(next);
		if (*inOutValue != adjusted) {
			*inOutValue = adjusted;
			changed = true;
		}
	}
	if (input_.focusId == id && impl_->keyboardAdjust != 0.0F) {
		const oa::I64 magnitude = oa::max<oa::I64>(
			1,
			static_cast<oa::I64>(oa::round(
				static_cast<double>(range) * 0.01
					* oa::abs(static_cast<double>(impl_->keyboardAdjust)))));
		const oa::I64 direction = impl_->keyboardAdjust > 0.0F ? 1 : -1;
		const oa::I64 next = oa::clamp<oa::I64>(
			static_cast<oa::I64>(*inOutValue) + direction * magnitude,
			static_cast<oa::I64>(inMin),
			static_cast<oa::I64>(inMax));
		const oa::I32 adjusted = static_cast<oa::I32>(next);
		if (*inOutValue != adjusted) {
			*inOutValue = adjusted;
			changed = true;
		}
		impl_->keyboardAdjust = 0.0F;
	}
	valueText = formatValue();
	valueExtent = impl_->measureText(valueText, style);
	const oa::F32 fraction = range > 0
		? static_cast<oa::F32>(
			static_cast<double>(static_cast<oa::I64>(*inOutValue)
				- static_cast<oa::I64>(inMin)) / static_cast<double>(range))
		: 0.0F;
	drawSliderVisual(
		*this, inLabel, valueText, rect, fraction,
		labelExtent, valueExtent,
		interaction.hovered, interaction.held, input_.focusId == id);
	impl_->addAccessibilityNode(
		id,
		oa::UiAccessibilityRole::Slider,
		rect,
		inLabel,
		valueText,
		oa::UiAccessibilityState::None,
		oa::UiAccessibilityAction::Focus
			| oa::UiAccessibilityAction::Increment
			| oa::UiAccessibilityAction::Decrement
			| oa::UiAccessibilityAction::SetValue,
		input_.focusId == id,
		true,
		inMin,
		inMax,
		*inOutValue);
	return changed;
}
bool oa::Ui::inputText(oa::StringView inLabel, oa::String& inOutText) {
	if (!impl_) return false;
	if (!isValidUtf8(inOutText.view())) {
		impl_->setFrameError(oa::Status::invalidArgument(
			"oa::Ui::inputText requires valid UTF-8"));
		return false;
	}

	const oa::UiStyle& style = currentStyle();
	const oa::vlm::Vec2 labelExtent = impl_->measureText(inLabel, style);
	oa::vlm::Vec2 valueExtent = impl_->measureText(inOutText.view(), style);
	const oa::F32 padX = oa::max(0.0F, style.framePaddingX);
	const oa::F32 padY = oa::max(0.0F, style.framePaddingY);
	const oa::F32 innerGap = oa::max(4.0F, style.itemSpacing);
	const oa::F32 height = oa::max(
		style.fontSize,
		oa::max(labelExtent.y, valueExtent.y)) + padY * 2.0F;
	const oa::F32 hugWidth = oa::max(
		220.0F,
		labelExtent.x + 128.0F + padX * 3.0F + innerGap);
	const oa::U32 id = hashWidgetScope(impl_->currentScope(), inLabel);
	const oa::PixelRect rect = impl_->placeItem(hugWidth, height);
	if (rect.w <= 0 || rect.h <= 0) return false;

	impl_->registerTextInput(id);
	auto& state = impl_->getTextEditState(id, inOutText.view());
	if (state.lastValue != inOutText) {
		// The caller remains the value owner. A replacement made outside this
		// field establishes a new local history root instead of making undo
		// resurrect stale application state.
		state.cursor = inOutText.size();
		state.anchor = state.cursor;
		state.scrollX = 0.0F;
		state.preedit.clear();
		state.undo.clear();
		state.redo.clear();
		state.lastValue = inOutText;
	}
	state.cursor = oa::min(state.cursor, inOutText.size());
	state.anchor = oa::min(state.anchor, inOutText.size());
	while (state.cursor > 0U && state.cursor < inOutText.size()
		&& isUtf8Continuation(static_cast<oa::U8>(inOutText[state.cursor]))) {
		--state.cursor;
	}
	while (state.anchor > 0U && state.anchor < inOutText.size()
		&& isUtf8Continuation(static_cast<oa::U8>(inOutText[state.anchor]))) {
		--state.anchor;
	}

	const auto interaction = impl_->interact(input_, id, rect);
	const bool focused = input_.focusId == id;
	if (!focused) state.preedit.clear();
	const oa::I32 minimumEditorWidth = oa::min<oa::I32>(48, rect.w);
	const oa::I32 labelPixels = static_cast<oa::I32>(oa::ceil(
		padX + labelExtent.x + innerGap));
	const oa::I32 proportional = static_cast<oa::I32>(oa::floor(
		static_cast<oa::F32>(rect.w) * 0.38F));
	const oa::I32 valueOffset = oa::clamp(
		oa::max(labelPixels, proportional),
		0,
		oa::max(0, rect.w - minimumEditorWidth));
	const oa::PixelRect editRect{
		rect.x + valueOffset,
		rect.y,
		rect.w - valueOffset,
		rect.h,
	};
	const oa::I32 textPad = oa::max<oa::I32>(
		2, static_cast<oa::I32>(oa::floor(padX + 0.5F)));
	const oa::PixelRect textRect{
		editRect.x + textPad,
		editRect.y,
		oa::max(0, editRect.w - textPad * 2),
		editRect.h,
	};

	const auto selection = [&] {
		return oa::Pair<oa::Usize, oa::Usize>{
			oa::min(state.cursor, state.anchor),
			oa::max(state.cursor, state.anchor),
		};
	};
	const auto replaceRange = [&](oa::Usize inBegin, oa::Usize inEnd,
		oa::StringView inReplacement) {
		oa::String next;
		next.reserve(inOutText.size() - (inEnd - inBegin) + inReplacement.size());
		next.append(oa::StringView(inOutText.data(), inBegin));
		next.append(inReplacement);
		next.append(oa::StringView(
			inOutText.data() + inEnd, inOutText.size() - inEnd));
		inOutText = oa::move(next);
		state.cursor = inBegin + inReplacement.size();
		state.anchor = state.cursor;
	};
	const auto eraseSelection = [&] {
		const auto [begin, end] = selection();
		if (begin == end) return false;
		replaceRange(begin, end, {});
		return true;
	};
	const auto isWordAt = [&](oa::Usize inOffset) {
		if (inOffset >= inOutText.size()) return false;
		const oa::U8 byte = static_cast<oa::U8>(inOutText[inOffset]);
		return byte >= 0x80U || (byte >= 'a' && byte <= 'z')
			|| (byte >= 'A' && byte <= 'Z') || (byte >= '0' && byte <= '9')
			|| byte == '_';
	};
	const auto previousWord = [&](oa::Usize inOffset) {
		oa::Usize offset = oa::min(inOffset, inOutText.size());
		while (offset > 0U) {
			const oa::Usize previous = utf8Previous(inOutText.view(), offset);
			if (isWordAt(previous)) break;
			offset = previous;
		}
		while (offset > 0U) {
			const oa::Usize previous = utf8Previous(inOutText.view(), offset);
			if (!isWordAt(previous)) break;
			offset = previous;
		}
		return offset;
	};
	const auto nextWord = [&](oa::Usize inOffset) {
		oa::Usize offset = oa::min(inOffset, inOutText.size());
		while (offset < inOutText.size() && !isWordAt(offset)) {
			offset = utf8ScalarEnd(inOutText.view(), offset);
		}
		while (offset < inOutText.size() && isWordAt(offset)) {
			offset = utf8ScalarEnd(inOutText.view(), offset);
		}
		return offset;
	};
	const auto wordBounds = [&](oa::Usize inOffset) {
		if (inOutText.empty()) {
			return oa::Pair<oa::Usize, oa::Usize>{0U, 0U};
		}
		oa::Usize probe = oa::min(inOffset, inOutText.size());
		if (probe == inOutText.size()) probe = utf8Previous(inOutText.view(), probe);
		if (!isWordAt(probe) && probe > 0U) {
			const oa::Usize prior = utf8Previous(inOutText.view(), probe);
			if (isWordAt(prior)) probe = prior;
		}
		if (isWordAt(probe)) {
			return oa::Pair<oa::Usize, oa::Usize>{
				previousWord(utf8ScalarEnd(inOutText.view(), probe)),
				nextWord(probe),
			};
		}
		return oa::Pair<oa::Usize, oa::Usize>{
			probe, utf8ScalarEnd(inOutText.view(), probe)};
	};
	const auto moveCursor = [&](oa::Usize inOffset, bool inExtend) {
		state.cursor = oa::min(inOffset, inOutText.size());
		if (!inExtend) state.anchor = state.cursor;
	};

	bool changed = false;
	bool caretActivity = false;
	static constexpr oa::Usize kTextHistoryCapacity = 64U;
	const auto snapshot = [&] {
		return Impl::TextEditSnapshot{
			.text = inOutText,
			.cursor = state.cursor,
			.anchor = state.anchor,
		};
	};
	const auto pushHistory = [&](oa::Vector<Impl::TextEditSnapshot>& inHistory,
		Impl::TextEditSnapshot inSnapshot) {
		if (inHistory.size() >= kTextHistoryCapacity) {
			inHistory.erase(inHistory.begin());
		}
		inHistory.pushBack(oa::move(inSnapshot));
	};
	const auto recordMutation = [&](Impl::TextEditSnapshot inBefore) {
		if (inBefore.text == inOutText) return false;
		pushHistory(state.undo, oa::move(inBefore));
		state.redo.clear();
		changed = true;
		caretActivity = true;
		return true;
	};
	const auto restoreHistory = [&](oa::Vector<Impl::TextEditSnapshot>& inFrom,
		oa::Vector<Impl::TextEditSnapshot>& inTo) {
		if (inFrom.empty()) return false;
		Impl::TextEditSnapshot current = snapshot();
		Impl::TextEditSnapshot restore = oa::move(inFrom.back());
		inFrom.popBack();
		pushHistory(inTo, oa::move(current));
		inOutText = oa::move(restore.text);
		state.cursor = oa::min(restore.cursor, inOutText.size());
		state.anchor = oa::min(restore.anchor, inOutText.size());
		state.preedit.clear();
		changed = true;
		caretActivity = true;
		return true;
	};
	if (focused) {
		for (const TextEditEvent& event : impl_->textEditEvents) {
			const bool compositionOwnsKey = !state.preedit.empty()
				&& (event.kind == TextEditKind::MoveLeft
					|| event.kind == TextEditKind::MoveRight
					|| event.kind == TextEditKind::MoveHome
					|| event.kind == TextEditKind::MoveEnd
					|| event.kind == TextEditKind::Backspace
					|| event.kind == TextEditKind::Delete
					|| event.kind == TextEditKind::SelectAll
					|| event.kind == TextEditKind::Cut
					|| event.kind == TextEditKind::Undo
					|| event.kind == TextEditKind::Redo);
			if (compositionOwnsKey) {
				caretActivity = true;
				continue;
			}
			switch (event.kind) {
				case TextEditKind::Insert: {
					if (event.text.empty()) break;
					state.preedit.clear();
					auto before = snapshot();
					const auto [begin, end] = selection();
					replaceRange(begin, end, event.text.view());
					(void)recordMutation(oa::move(before));
					break;
				}
				case TextEditKind::UpdateComposition: {
					if (!isValidUtf8(event.text.view())) {
						state.preedit.clear();
						impl_->setFrameError(oa::Status::invalidArgument(
							"oa::Ui IME pre-edit requires valid UTF-8"));
						break;
					}
					state.preedit = singleLineUtf8(event.text.view());
					if (state.preedit.empty()) {
						state.preeditSelectionBegin = 0U;
						state.preeditSelectionEnd = 0U;
					} else if (event.selectionStart < 0) {
						state.preeditSelectionBegin = state.preedit.size();
						state.preeditSelectionEnd = state.preedit.size();
					} else {
						const oa::I32 start = event.selectionStart;
						const oa::I64 end64 = static_cast<oa::I64>(start)
							+ oa::max<oa::I64>(0, event.selectionLength);
						const oa::I32 end = static_cast<oa::I32>(oa::min<oa::I64>(
							end64, oa::Limits<oa::I32>::max()));
						state.preeditSelectionBegin = utf8ByteOffsetForScalarIndex(
							state.preedit.view(), start);
						state.preeditSelectionEnd = utf8ByteOffsetForScalarIndex(
							state.preedit.view(), end);
					}
					caretActivity = true;
					break;
				}
				case TextEditKind::CancelComposition:
					state.preedit.clear();
					caretActivity = true;
					break;
				case TextEditKind::MoveLeft: {
					const auto [begin, end] = selection();
					const oa::Usize target = !event.extend && begin != end
						? begin : event.byWord ? previousWord(state.cursor)
						: utf8Previous(inOutText.view(), state.cursor);
					moveCursor(target, event.extend);
					caretActivity = true;
					break;
				}
				case TextEditKind::MoveRight: {
					const auto [begin, end] = selection();
					const oa::Usize target = !event.extend && begin != end
						? end : event.byWord ? nextWord(state.cursor)
						: utf8ScalarEnd(inOutText.view(), state.cursor);
					moveCursor(target, event.extend);
					caretActivity = true;
					break;
				}
				case TextEditKind::MoveHome:
					moveCursor(0U, event.extend);
					caretActivity = true;
					break;
				case TextEditKind::MoveEnd:
					moveCursor(inOutText.size(), event.extend);
					caretActivity = true;
					break;
				case TextEditKind::Backspace: {
					auto before = snapshot();
					const bool erasedSelection = eraseSelection();
					if (!erasedSelection && state.cursor > 0U) {
						const oa::Usize begin = event.byWord
							? previousWord(state.cursor)
							: utf8Previous(inOutText.view(), state.cursor);
						replaceRange(begin, state.cursor, {});
					} else if (!erasedSelection) {
						break;
					}
					(void)recordMutation(oa::move(before));
					break;
				}
				case TextEditKind::Delete: {
					auto before = snapshot();
					const bool erasedSelection = eraseSelection();
					if (!erasedSelection && state.cursor < inOutText.size()) {
						const oa::Usize end = event.byWord
							? nextWord(state.cursor)
							: utf8ScalarEnd(inOutText.view(), state.cursor);
						replaceRange(state.cursor, end, {});
					} else if (!erasedSelection) {
						break;
					}
					(void)recordMutation(oa::move(before));
					break;
				}
				case TextEditKind::SelectAll:
					state.anchor = 0U;
					state.cursor = inOutText.size();
					caretActivity = true;
					break;
				case TextEditKind::Copy:
				case TextEditKind::Cut: {
					const auto [begin, end] = selection();
					if (begin == end) break;
					impl_->clipboardWrite = oa::String(
						inOutText.data() + begin, end - begin);
					if (event.kind == TextEditKind::Cut) {
						auto before = snapshot();
						replaceRange(begin, end, {});
						(void)recordMutation(oa::move(before));
					}
					break;
				}
				case TextEditKind::Undo:
					state.preedit.clear();
					(void)restoreHistory(state.undo, state.redo);
					break;
				case TextEditKind::Redo:
					state.preedit.clear();
					(void)restoreHistory(state.redo, state.undo);
					break;
			}
		}
		impl_->textEditEvents.clear();
	}

	const auto measureCommittedPrefix = [&](oa::Usize inEnd) {
		return impl_->measureText(
			oa::StringView(inOutText.data(), oa::min(inEnd, inOutText.size())),
			style).x;
	};
	const auto cursorFromPointer = [&] {
		const oa::F32 target = input_.mouseX
			- static_cast<oa::F32>(textRect.x) + state.scrollX;
		oa::Usize previous = 0U;
		oa::F32 previousX = 0.0F;
		for (oa::Usize next = utf8ScalarEnd(inOutText.view(), 0U);
			previous < inOutText.size();
			previous = next, next = utf8ScalarEnd(inOutText.view(), next)) {
			const oa::F32 nextX = measureCommittedPrefix(next);
			if (target < (previousX + nextX) * 0.5F) return previous;
			previousX = nextX;
		}
		return inOutText.size();
	};
	if (interaction.held && (input_.lPressed
		|| input_.mouseDX != 0.0F || input_.mouseDY != 0.0F)) {
		if (input_.lPressed) state.preedit.clear();
		const oa::Usize cursor = cursorFromPointer();
		if (input_.lPressed && input_.lClickCount >= 3) {
			state.anchor = 0U;
			state.cursor = inOutText.size();
		} else if (input_.lPressed && input_.lClickCount == 2) {
			const auto [begin, end] = wordBounds(cursor);
			state.anchor = begin;
			state.cursor = end;
		} else {
			if (input_.lPressed && (input_.modifiers & oa::UiModifierShift) == 0U) {
				state.anchor = cursor;
			}
			state.cursor = cursor;
		}
		caretActivity = true;
	}
	if (caretActivity) state.blinkMs = 0.0F;

	const auto [selectionBegin, selectionEnd] = selection();
	const bool composing = focused && !state.preedit.empty();
	oa::String displayText;
	oa::Usize compositionBegin = 0U;
	oa::Usize compositionEnd = 0U;
	oa::Usize displayCaret = state.cursor;
	if (composing) {
		displayText.reserve(inOutText.size()
			- (selectionEnd - selectionBegin) + state.preedit.size());
		displayText.append(oa::StringView(inOutText.data(), selectionBegin));
		compositionBegin = displayText.size();
		displayText.append(state.preedit.view());
		compositionEnd = displayText.size();
		displayText.append(oa::StringView(
			inOutText.data() + selectionEnd, inOutText.size() - selectionEnd));
		displayCaret = compositionBegin + state.preeditSelectionEnd;
	} else {
		displayText = inOutText;
	}
	valueExtent = impl_->measureText(displayText.view(), style);
	const auto measureDisplayPrefix = [&](oa::Usize inEnd) {
		return impl_->measureText(
			oa::StringView(displayText.data(), oa::min(inEnd, displayText.size())),
			style).x;
	};

	const oa::F32 caretX = measureDisplayPrefix(displayCaret);
	const oa::F32 availableWidth = static_cast<oa::F32>(oa::max(1, textRect.w));
	if (caretX < state.scrollX) state.scrollX = caretX;
	if (caretX - state.scrollX > availableWidth - 1.0F) {
		state.scrollX = caretX - availableWidth + 1.0F;
	}
	const oa::F32 textWidth = valueExtent.x;
	state.scrollX = oa::clamp(
		state.scrollX, 0.0F, oa::max(0.0F, textWidth - availableWidth));
	const oa::I32 caretPixel = textRect.w > 0
		? oa::clamp(
			textRect.x + static_cast<oa::I32>(oa::floor(
				caretX - state.scrollX + 0.5F)),
			textRect.x,
			textRect.x + textRect.w - 1)
		: textRect.x;

	this->rect(rect, interaction.held ? style.surfaceActive
		: interaction.hovered ? style.surfaceHover : style.surface);
	if (editRect.w > 0) {
		this->rect(editRect, style.background.withAlpha(0.72F));
	}
	if (focused) {
		impl_->focusedTextInputRect = {
			caretPixel, editRect.y, 1, editRect.h,
		};
	}
	if (focused && !composing && selectionBegin != selectionEnd
		&& textRect.w > 0) {
		const oa::F32 beginX = measureDisplayPrefix(selectionBegin) - state.scrollX;
		const oa::F32 endX = measureDisplayPrefix(selectionEnd) - state.scrollX;
		const oa::I32 left = oa::clamp(
			textRect.x + static_cast<oa::I32>(oa::floor(beginX)),
			textRect.x,
			textRect.x + textRect.w);
		const oa::I32 right = oa::clamp(
			textRect.x + static_cast<oa::I32>(oa::ceil(endX)),
			textRect.x,
			textRect.x + textRect.w);
		if (right > left) {
			this->rect({left, textRect.y + 2, right - left,
				oa::max(1, textRect.h - 4)}, style.accent.withAlpha(0.45F));
		}
	}
	if (composing && textRect.w > 0) {
		const oa::Usize selectedBegin = compositionBegin
			+ oa::min(state.preeditSelectionBegin, state.preedit.size());
		const oa::Usize selectedEnd = compositionBegin
			+ oa::min(state.preeditSelectionEnd, state.preedit.size());
		const oa::F32 compositionX = measureDisplayPrefix(compositionBegin)
			- state.scrollX;
		const oa::F32 compositionRight = measureDisplayPrefix(compositionEnd)
			- state.scrollX;
		const oa::I32 underlineLeft = oa::clamp(
			textRect.x + static_cast<oa::I32>(oa::floor(compositionX)),
			textRect.x, textRect.x + textRect.w);
		const oa::I32 underlineRight = oa::clamp(
			textRect.x + static_cast<oa::I32>(oa::ceil(compositionRight)),
			textRect.x, textRect.x + textRect.w);
		if (selectedEnd > selectedBegin) {
			const oa::I32 left = oa::clamp(
				textRect.x + static_cast<oa::I32>(oa::floor(
					measureDisplayPrefix(selectedBegin) - state.scrollX)),
				textRect.x, textRect.x + textRect.w);
			const oa::I32 right = oa::clamp(
				textRect.x + static_cast<oa::I32>(oa::ceil(
					measureDisplayPrefix(selectedEnd) - state.scrollX)),
				textRect.x, textRect.x + textRect.w);
			if (right > left) {
				this->rect({left, textRect.y + 2, right - left,
					oa::max(1, textRect.h - 4)}, style.accent.withAlpha(0.38F));
			}
		}
		if (underlineRight > underlineLeft) {
			this->rect({underlineLeft, textRect.y + textRect.h - 3,
				underlineRight - underlineLeft, 2}, style.accentHover);
		}
	}

	oa::UiStyle labelStyle = style;
	labelStyle.text = style.textSecondary;
	oa::UiLayout labelLayout;
	labelLayout.padding = oa::UiEdge{};
	labelLayout.padding.left = padX;
	labelLayout.padding.top = oa::max(
		0.0F, (static_cast<oa::F32>(rect.h) - labelExtent.y) * 0.5F);
	const oa::PixelRect labelRect{rect.x, rect.y, valueOffset, rect.h};
	if (!inLabel.empty() && labelRect.w > 0) {
		beginPanel("__input_label", labelRect, labelLayout);
		impl_->appendText(inLabel, labelStyle, false);
		endPanel();
	}
	if (!displayText.empty() && textRect.w > 0) {
		oa::UiLayout textLayout;
		textLayout.padding = oa::UiEdge{};
		textLayout.padding.left = -state.scrollX;
		textLayout.padding.top = oa::max(
			0.0F, (static_cast<oa::F32>(textRect.h) - valueExtent.y) * 0.5F);
		beginPanel("__input_value", textRect, textLayout);
		impl_->appendText(displayText.view(), style, false);
		endPanel();
	}
	if (focused && state.blinkMs < 600.0F && textRect.w > 0) {
		const oa::I32 caretHeight = oa::max(
			1, oa::min(textRect.h - 4,
				static_cast<oa::I32>(oa::ceil(style.fontSize))));
		this->rect({caretPixel, textRect.y + oa::max(2, (textRect.h - caretHeight) / 2),
			1, caretHeight}, style.text);
	}
	const oa::U32 borderWidth = static_cast<oa::U32>(oa::clamp(
		oa::isFinite(style.borderWidth)
			? oa::floor(style.borderWidth + 0.5F) : 1.0F,
		1.0F,
		8.0F));
	rectOutline(rect, focused ? style.accentHover : style.border, borderWidth);
	impl_->addAccessibilityNode(
		id,
		oa::UiAccessibilityRole::TextField,
		rect,
		inLabel,
		inOutText.view(),
		oa::UiAccessibilityState::Editable,
		oa::UiAccessibilityAction::Focus | oa::UiAccessibilityAction::SetValue,
		focused);
	state.lastValue = inOutText;
	return changed;
}

bool oa::Ui::dropdown(
	oa::StringView inLabel,
	oa::Span<const oa::StringView> inItems,
	oa::I32& inOutSelected,
	const oa::UiDropdownConfig& inConfig) {
	if (!impl_) return false;
	if (inItems.empty()
		|| inItems.size() > static_cast<oa::Usize>(oa::Limits<oa::I32>::max())
		|| inOutSelected < 0
		|| static_cast<oa::Usize>(inOutSelected) >= inItems.size()
		|| inConfig.maxVisibleItems <= 0 || inConfig.popupWidth < 0
		|| inConfig.popupGap < 0) {
		impl_->setFrameError(oa::Status::invalidArgument(
			"oa::Ui::dropdown requires items, a valid selection, positive visibility, and non-negative popup metrics"));
		return false;
	}

	const oa::UiStyle& style = currentStyle();
	oa::String display;
	display.append(inLabel);
	if (!inLabel.empty()) display.append(oa::StringView("  "));
	display.append(inItems[static_cast<oa::Usize>(inOutSelected)]);
	const oa::vlm::Vec2 textExtent = impl_->measureText(display.view(), style);
	const oa::F32 padX = oa::max(0.0F, style.framePaddingX);
	const oa::F32 padY = oa::max(0.0F, style.framePaddingY);
	const oa::F32 height = oa::max(style.fontSize, textExtent.y) + padY * 2.0F;
	const oa::F32 arrowWidth = oa::max(10.0F, style.fontSize * 0.75F);
	const oa::F32 hugWidth = oa::max(
		height, textExtent.x + padX * 3.0F + arrowWidth);
	const oa::U32 id = hashWidgetScope(impl_->currentScope(), inLabel);
	const oa::PixelRect rect = impl_->placeItem(hugWidth, height);
	if (rect.w <= 0 || rect.h <= 0) return false;

	const auto interaction = impl_->interact(input_, id, rect);
	const oa::U32 anchorLastItemId = impl_->lastItemId;
	const oa::PixelRect anchorLastItemRect = impl_->lastItemRect;
	const bool anchorLastItemHovered = impl_->lastItemHovered;
	oa::String popupName(inLabel);
	popupName.append(oa::StringView("##dropdown_popup"));
	const oa::U32 popupId = hashWidgetScope(impl_->currentScope(), popupName.view());
	if (interaction.activated) {
		if (impl_->openPopupId == popupId) impl_->closePopup(input_, true);
		else impl_->openPopup(popupId, id, rect);
	}
	const bool open = impl_->openPopupId == popupId;
	this->rect(rect,
		interaction.held ? style.surfaceActive
			: interaction.hovered || open ? style.surfaceHover : style.surface);
	rectOutline(rect,
		input_.focusId == id || open ? style.accentHover : style.border,
		1U);

	oa::UiLayout textLayout;
	textLayout.padding = oa::UiEdge{};
	textLayout.padding.left = padX;
	textLayout.padding.right = padX * 2.0F + arrowWidth;
	textLayout.padding.top = oa::max(
		0.0F, (static_cast<oa::F32>(rect.h) - textExtent.y) * 0.5F);
	beginPanel("__dropdown_text", rect, textLayout);
	impl_->appendText(display.view(), style, false);
	endPanel();
	const oa::F32 centerX = static_cast<oa::F32>(rect.x + rect.w)
		- padX - arrowWidth * 0.5F;
	const oa::F32 centerY = static_cast<oa::F32>(rect.y)
		+ static_cast<oa::F32>(rect.h) * 0.5F;
	const oa::F32 half = oa::max(2.0F, arrowWidth * 0.24F);
	const oa::F32 direction = open ? -1.0F : 1.0F;
	line({centerX - half, centerY - direction * half * 0.5F},
		{centerX, centerY + direction * half * 0.5F}, style.textSecondary, 1.25F);
	line({centerX, centerY + direction * half * 0.5F},
		{centerX + half, centerY - direction * half * 0.5F},
		style.textSecondary, 1.25F);

	bool changed = false;
	const oa::I32 itemHeight = oa::max<oa::I32>(1,
		static_cast<oa::I32>(oa::ceil(height)));
	if (inItems.size() > static_cast<oa::Usize>(
		(oa::Limits<oa::I32>::max() - 8) / itemHeight)) {
		impl_->setFrameError(oa::Status::error(
			oa::StatusCode::OutOfRange,
			"oa::Ui::dropdown item geometry exceeds signed layout capacity"));
		return false;
	}
	const oa::I32 visibleItems = oa::min<oa::I32>(
		inConfig.maxVisibleItems, static_cast<oa::I32>(inItems.size()));
	const oa::I32 popupPadding = 4;
	const oa::UiPopupConfig popupConfig{
		.width = oa::max(rect.w, inConfig.popupWidth),
		.height = visibleItems * itemHeight + popupPadding * 2,
		.gap = inConfig.popupGap,
		.padding = oa::UiEdge{},
	};
	if (beginPopup(popupName.view(), popupConfig)) {
		const oa::PixelRect popup = impl_->popupRect;
		const oa::PixelRect scrollViewport{
			popup.x + popupPadding,
			popup.y + popupPadding,
			oa::max(1, popup.w - popupPadding * 2),
			oa::max(1, popup.h - popupPadding * 2),
		};
		const oa::U32 scrollId = hashWidgetScope(
			impl_->currentScope(), "__dropdown_scroll");
		auto& scrollState = impl_->getScrollState(scrollId);
		oa::I32 focusedIndex = inOutSelected;
		if (impl_->popupFocusMovedThisFrame) {
			for (oa::Usize index = 0U;
				index < impl_->priorPopupFocusOrder.size(); ++index) {
				if (impl_->priorPopupFocusOrder[index] == input_.focusId) {
					focusedIndex = static_cast<oa::I32>(index);
					break;
				}
			}
		}
		if (impl_->popupOpenedThisFrame || impl_->popupFocusMovedThisFrame) {
			const oa::I32 focusedTop = focusedIndex * itemHeight;
			const oa::I32 focusedBottom = focusedTop + itemHeight;
			if (focusedTop < scrollState.offsetY) {
				scrollState.offsetY = focusedTop;
			} else if (static_cast<oa::I64>(focusedBottom)
				> static_cast<oa::I64>(scrollState.offsetY) + scrollViewport.h) {
				scrollState.offsetY = focusedBottom - scrollViewport.h;
			}
		}
		oa::UiLayout scrollLayout;
		scrollLayout.padding = oa::UiEdge{};
		scrollLayout.gap = 0.0F;
		(void)beginScrollPanel(
			"__dropdown_scroll",
			scrollViewport,
			static_cast<oa::I32>(inItems.size()) * itemHeight,
			scrollLayout,
			{.wheelStep = itemHeight, .scrollbarWidth = 8,
			 .scrollbarGap = 3, .showScrollbar = true});
		auto& panel = impl_->panelStack.back();
		const oa::U32 itemScope = panel.scope;
		for (oa::Usize index = 0U; index < inItems.size(); ++index) {
			impl_->registerFocusable(hashWidgetScope(
				hashWidgetIndex(itemScope, static_cast<oa::U32>(index)),
				inItems[index]));
		}
		const oa::UiVirtualRange visible = virtualRows(
			static_cast<oa::I32>(inItems.size()), itemHeight, 0, 1);
		panel.cursor = static_cast<oa::F32>(visible.first * itemHeight);
		for (oa::I32 index = visible.first;
			index < visible.onePastLast; ++index) {
			panel.scope = hashWidgetIndex(itemScope, static_cast<oa::U32>(index));
			if (menuItem(inItems[static_cast<oa::Usize>(index)],
				index == inOutSelected)) {
				inOutSelected = index;
				changed = true;
			}
		}
		panel.scope = itemScope;
		endScrollPanel();
		if (impl_->popupOpenedThisFrame && impl_->openPopupId == popupId
			&& static_cast<oa::Usize>(inOutSelected)
				< impl_->popupFocusOrder.size()) {
			input_.focusId = impl_->popupFocusOrder[
				static_cast<oa::Usize>(inOutSelected)];
			impl_->popupOpenedThisFrame = false;
		}
		endPopup();
	}
	impl_->addAccessibilityNode(
		id,
		oa::UiAccessibilityRole::ComboBox,
		rect,
		inLabel,
		inItems[static_cast<oa::Usize>(inOutSelected)],
		impl_->openPopupId == popupId
			? oa::UiAccessibilityState::HasPopup
				| oa::UiAccessibilityState::Expanded
			: oa::UiAccessibilityState::HasPopup,
		oa::UiAccessibilityAction::Focus | oa::UiAccessibilityAction::Activate,
		input_.focusId == id);
	// The dropdown itself, not the last option it rendered in the overlay, is
	// the attachment point for a following tooltip call.
	impl_->lastItemId = anchorLastItemId;
	impl_->lastItemRect = anchorLastItemRect;
	impl_->lastItemHovered = anchorLastItemHovered;
	return changed;
}

void oa::Ui::tooltip(
	oa::StringView inText,
	const oa::UiTooltipConfig& inConfig) {
	if (!impl_ || inText.empty()) return;
	const auto finiteEdge = [](const oa::UiEdge& inEdge) noexcept {
		return oa::isFinite(inEdge.top) && oa::isFinite(inEdge.right)
			&& oa::isFinite(inEdge.bottom) && oa::isFinite(inEdge.left)
			&& inEdge.top >= 0.0F && inEdge.right >= 0.0F
			&& inEdge.bottom >= 0.0F && inEdge.left >= 0.0F;
	};
	if (!oa::isFinite(inConfig.delayMs) || inConfig.delayMs < 0.0F
		|| inConfig.maxWidth <= 0 || inConfig.gap < 0
		|| !finiteEdge(inConfig.padding)) {
		impl_->setFrameError(oa::Status::invalidArgument(
			"oa::Ui::tooltip requires a finite delay, positive width, and non-negative gap/padding"));
		return;
	}
	if (impl_->lastItemId == 0U) {
		impl_->setFrameError(oa::Status::error(
			oa::StatusCode::FailedPrecondition,
			"oa::Ui::tooltip requires an immediately preceding interactive item"));
		return;
	}
	if (!impl_->lastItemHovered) {
		if (impl_->tooltipOwnerId == impl_->lastItemId) {
			impl_->tooltipOwnerId = 0U;
			impl_->tooltipHoverMs = 0.0F;
		}
		return;
	}
	if (impl_->tooltipOwnerId != impl_->lastItemId
		|| impl_->tooltipLastSeenFrame + 1U != impl_->frameIndex) {
		impl_->tooltipOwnerId = impl_->lastItemId;
		impl_->tooltipHoverMs = 0.0F;
	} else {
		impl_->tooltipHoverMs += impl_->frameDeltaMs;
	}
	impl_->tooltipLastSeenFrame = impl_->frameIndex;
	if (impl_->tooltipHoverMs < inConfig.delayMs
		|| (impl_->openPopupId != 0U
			&& impl_->lastItemId == impl_->popupAnchorId)) return;
	if (impl_->frameViewport.w <= 0 || impl_->frameViewport.h <= 0) {
		impl_->setFrameError(oa::Status::error(
			oa::StatusCode::FailedPrecondition,
			"oa::Ui tooltips require beginFrame to receive the render viewport"));
		return;
	}
	if (impl_->textAtlas == nullptr) {
		impl_->setFrameError(oa::Status::error(
			oa::StatusCode::FailedPrecondition,
			"oa::Ui tooltips require bindTextAtlas before the first frame"));
		return;
	}

	const oa::UiStyle& style = currentStyle();
	if (!oa::isFinite(style.fontSize) || style.fontSize <= 0.0F) {
		impl_->setFrameError(oa::Status::invalidArgument(
			"oa::Ui tooltips require a finite positive font size"));
		return;
	}
	const oa::F32 contentWidth = oa::max(
		1.0F,
		static_cast<oa::F32>(inConfig.maxWidth)
			- inConfig.padding.left - inConfig.padding.right);
	oa::TextLayout textLayoutEngine;
	const oa::vlm::Vec2 extent = textLayoutEngine.measure(
		*impl_->textAtlas,
		inText,
		{.font = oa::FontId::Sans, .size = style.fontSize,
		 .wrapWidth = contentWidth});
	const oa::I32 width = oa::clamp<oa::I32>(
		static_cast<oa::I32>(oa::ceil(
			extent.x + inConfig.padding.left + inConfig.padding.right)),
		1,
		oa::min(inConfig.maxWidth, impl_->frameViewport.w));
	const oa::I32 height = oa::clamp<oa::I32>(
		static_cast<oa::I32>(oa::ceil(
			extent.y + inConfig.padding.top + inConfig.padding.bottom)),
		1,
		impl_->frameViewport.h);
	const oa::I32 left = impl_->frameViewport.x;
	const oa::I32 top = impl_->frameViewport.y;
	const oa::I32 right = left + impl_->frameViewport.w;
	const oa::I32 bottom = top + impl_->frameViewport.h;
	oa::I32 x = static_cast<oa::I32>(oa::floor(input_.mouseX)) + inConfig.gap;
	if (x + width > right) {
		x = static_cast<oa::I32>(oa::floor(input_.mouseX))
			- inConfig.gap - width;
	}
	x = oa::clamp(x, left, oa::max(left, right - width));
	oa::I32 y = impl_->lastItemRect.y + impl_->lastItemRect.h + inConfig.gap;
	if (y + height > bottom) {
		y = impl_->lastItemRect.y - inConfig.gap - height;
	}
	y = oa::clamp(y, top, oa::max(top, bottom - height));
	const oa::PixelRect rect{x, y, width, height};

	const bool previousOverlay = impl_->recordingOverlay;
	impl_->recordingOverlay = true;
	Impl::PanelState panel;
	panel.rect = rect;
	panel.clip = intersectPixelRects(
		clipToNonNegative(rect), impl_->frameViewport);
	panel.layout.padding = inConfig.padding;
	panel.layout.gap = 0.0F;
	panel.cursor = panel.layout.padding.top;
	panel.rowX = panel.layout.padding.left;
	panel.rowY = panel.layout.padding.top;
	panel.scope = hashWidgetScope(impl_->lastItemId, "__tooltip");
	panel.rowScope = panel.scope;
	impl_->panelStack.pushBack(panel);
	this->rect(rect, style.surfaceActive.withAlpha(0.99F));
	rectOutline(rect, style.borderStrong, 1U);
	impl_->appendText(inText, style, true);
	impl_->panelStack.popBack();
	impl_->recordingOverlay = previousOverlay;
}

void oa::Ui::label(oa::StringView inText) {
	if (!impl_) return;
	impl_->appendText(inText, currentStyle(), false);
}

void oa::Ui::labelFmt(const char* inFmt, ...) {
	if (!impl_) return;
	if (inFmt == nullptr) {
		impl_->setFrameError(oa::Status::invalidArgument(
			"oa::Ui::labelFmt requires a format string"));
		return;
	}
	va_list args;
	va_start(args, inFmt);
	va_list copy;
	va_copy(copy, args);
	const int required = ::vsnprintf(nullptr, 0, inFmt, args);
	va_end(args);
	if (required < 0) {
		va_end(copy);
		impl_->setFrameError(oa::Status::invalidArgument(
			"oa::Ui::labelFmt could not format the label"));
		return;
	}
	oa::Vector<char> text(static_cast<oa::Usize>(required) + 1U);
	(void)::vsnprintf(text.data(), text.size(), inFmt, copy);
	va_end(copy);
	label(oa::StringView(text.data(), static_cast<oa::Usize>(required)));
}

void oa::Ui::text(oa::StringView inText) {
	if (!impl_) return;
	impl_->appendText(inText, currentStyle(), true);
}

void oa::Ui::textAt(
	oa::StringView inText,
	oa::PixelRect inRect,
	const oa::UiTextConfig& inConfig) {
	if (!impl_) return;
	impl_->appendTextAt(inText, inRect, inConfig);
}

void oa::Ui::colorSwatch(oa::Color inColor, oa::vlm::Vec2 inSize) {
	if (!impl_) return;
	if (!oa::isFinite(inSize.x) || !oa::isFinite(inSize.y)
		|| inSize.x <= 0.0F || inSize.y <= 0.0F) {
		impl_->setFrameError(oa::Status::invalidArgument(
			"oa::Ui::colorSwatch requires a finite positive size"));
		return;
	}
	const oa::PixelRect rect = impl_->placeItem(inSize.x, inSize.y);
	if (rect.w <= 0 || rect.h <= 0) return;
	this->rect(rect, inColor);
	rectOutline(rect, currentStyle().borderStrong, 1U);
}
void oa::Ui::progressBar(oa::F32 inFraction, oa::StringView inOverlay) {
	if (!impl_) return;
	if (!oa::isFinite(inFraction)) {
		impl_->setFrameError(oa::Status::invalidArgument(
			"oa::Ui::progressBar requires a finite fraction"));
		return;
	}
	const oa::F32 fraction = oa::clamp(inFraction, 0.0F, 1.0F);
	oa::Array<char, 16> generatedOverlay{};
	oa::StringView overlay = inOverlay;
	if (overlay.empty()) {
		const int written = ::snprintf(
			generatedOverlay.data(), generatedOverlay.size(), "%d%%",
			static_cast<int>(oa::round(fraction * 100.0F)));
		if (written < 0
			|| static_cast<oa::Usize>(written) >= generatedOverlay.size()) {
			impl_->setFrameError(oa::Status::error(
				oa::StatusCode::Internal,
				"oa::Ui::progressBar could not format its overlay"));
			return;
		}
		overlay = oa::StringView(
			generatedOverlay.data(), static_cast<oa::Usize>(written));
	}

	const oa::UiStyle& style = currentStyle();
	const oa::vlm::Vec2 textExtent = impl_->measureText(overlay, style);
	const oa::F32 padX = oa::max(0.0F, style.framePaddingX);
	const oa::F32 padY = oa::max(0.0F, style.framePaddingY);
	const oa::F32 height = oa::max(style.fontSize, textExtent.y) + padY * 2.0F;
	const oa::F32 hugWidth = oa::max(140.0F, textExtent.x + padX * 2.0F);
	const oa::PixelRect rect = impl_->placeItem(hugWidth, height);
	if (rect.w <= 0 || rect.h <= 0) return;
	this->rect(rect, style.surface);
	const oa::I32 fillWidth = oa::clamp(
		static_cast<oa::I32>(oa::floor(
			static_cast<oa::F32>(rect.w) * fraction + 0.5F)),
		0,
		rect.w);
	if (fillWidth > 0) {
		this->rect(
			{rect.x, rect.y, fillWidth, rect.h},
			style.accent.withAlpha(0.75F));
	}
	rectOutline(rect, style.border, 1U);
	oa::UiLayout textLayout;
	textLayout.padding = oa::UiEdge{};
	textLayout.padding.top = oa::max(
		0.0F,
		(static_cast<oa::F32>(rect.h) - textExtent.y) * 0.5F);
	textLayout.padding.left = oa::max(
		0.0F,
		(static_cast<oa::F32>(rect.w) - textExtent.x) * 0.5F);
	beginPanel("__progress_text", rect, textLayout);
	impl_->appendText(overlay, style, false);
	endPanel();
}

bool oa::Ui::timeline(
	oa::StringView inId,
	oa::PixelRect inRect,
	oa::F32& inOutFraction,
	bool* outActive) {
	if (outActive != nullptr) *outActive = false;
	if (!impl_ || inRect.w <= 0 || inRect.h <= 0) return false;

	const oa::U32 id = hashWidgetScope(impl_->currentScope(), inId);
	impl_->registerAdjustable(id);
	const auto interaction = impl_->interact(input_, id, inRect);
	if (outActive != nullptr) *outActive = interaction.held;

	bool changed = false;
	if (interaction.held || interaction.activated) {
		const oa::F32 next = oa::clamp(
			(input_.mouseX - static_cast<oa::F32>(inRect.x))
				/ static_cast<oa::F32>(inRect.w),
			0.0F,
			1.0F);
		changed = next != inOutFraction;
		inOutFraction = next;
	}
	if (input_.focusId == id && impl_->keyboardAdjust != 0.0F) {
		const oa::F32 next = oa::clamp(
			inOutFraction + impl_->keyboardAdjust * 0.01F, 0.0F, 1.0F);
		changed = changed || next != inOutFraction;
		inOutFraction = next;
		impl_->keyboardAdjust = 0.0F;
	}

	inOutFraction = oa::clamp(inOutFraction, 0.0F, 1.0F);
	const oa::UiStyle& style = currentStyle();
	this->rect(inRect, style.surface.withAlpha(0.94F));
	const oa::I32 fillWidth = static_cast<oa::I32>(
		static_cast<oa::F32>(inRect.w) * inOutFraction + 0.5F);
	if (fillWidth > 0) {
		this->rect({inRect.x, inRect.y, fillWidth, inRect.h}, style.accent);
	}
	const oa::I32 maximumHandle = oa::max<oa::I32>(1,
		static_cast<oa::I32>(oa::round(6.0F * impl_->contentScale)));
	const oa::I32 handleWidth = oa::max<oa::I32>(1,
		oa::min<oa::I32>(inRect.w, oa::min<oa::I32>(
			maximumHandle, oa::max<oa::I32>(1, inRect.h / 2))));
	const oa::I32 handleX = oa::clamp(
		inRect.x + fillWidth - handleWidth / 2,
		inRect.x,
		inRect.x + inRect.w - handleWidth);
	const oa::I32 handleOverhang = oa::max<oa::I32>(1,
		static_cast<oa::I32>(oa::round(3.0F * impl_->contentScale)));
	this->rect({handleX, inRect.y - handleOverhang,
		handleWidth, inRect.h + handleOverhang * 2},
		interaction.hovered || interaction.held
			? style.accentHover : style.textSecondary);
	impl_->addAccessibilityNode(
		id,
		oa::UiAccessibilityRole::Timeline,
		inRect,
		inId,
		{},
		oa::UiAccessibilityState::None,
		oa::UiAccessibilityAction::Focus
			| oa::UiAccessibilityAction::Increment
			| oa::UiAccessibilityAction::Decrement
			| oa::UiAccessibilityAction::SetValue,
		input_.focusId == id,
		true,
		0.0,
		1.0,
		inOutFraction);
	return changed;
}

bool oa::Ui::waveformTimeline(
	oa::StringView inId,
	oa::PixelRect inRect,
	const oa::Matrix& inEnvelope,
	oa::F32& inOutFraction) {
	const oa::MatrixShape shape = inEnvelope.getShape();
	if (!impl_ || inRect.w <= 0 || inRect.h <= 0
		|| shape.rank != 2 || shape[0] <= 0 || shape[1] != 2
		|| inEnvelope.getDtype() != oa::ScalarType::Float32
		|| inEnvelope.heapSlot() < 0) {
		return false;
	}

	const oa::U32 id = hashWidgetScope(impl_->currentScope(), inId);
	const oa::PixelRect clip = impl_->clipFor(inRect);
	impl_->registerAdjustable(id);
	const auto interaction = impl_->interact(input_, id, inRect);

	bool changed = false;
	if (interaction.held || interaction.activated) {
		const oa::F32 next = oa::clamp(
			(input_.mouseX - static_cast<oa::F32>(inRect.x))
				/ static_cast<oa::F32>(inRect.w),
			0.0F,
			1.0F);
		changed = next != inOutFraction;
		inOutFraction = next;
	}
	if (input_.focusId == id && impl_->keyboardAdjust != 0.0F) {
		const oa::F32 next = oa::clamp(
			inOutFraction + impl_->keyboardAdjust * 0.01F, 0.0F, 1.0F);
		changed = changed || next != inOutFraction;
		inOutFraction = next;
		impl_->keyboardAdjust = 0.0F;
	}
	inOutFraction = oa::clamp(inOutFraction, 0.0F, 1.0F);

	const oa::UiStyle& style = currentStyle();
	this->rect(inRect, style.surface.withAlpha(0.55F));
	BlitCmd command{};
	command.kind = BlitKind::Waveform;
	command.waveform.envelope_idx = static_cast<oa::U32>(inEnvelope.heapSlot());
	command.waveform.bins = static_cast<oa::U32>(shape[0]);
	command.waveform.dst_idx = 0U;
	command.waveform.dst_x = inRect.x;
	command.waveform.dst_y = inRect.y;
	command.waveform.dst_w = static_cast<oa::U32>(inRect.w);
	command.waveform.dst_h = static_cast<oa::U32>(inRect.h);
	command.waveform.fraction = inOutFraction;
	command.waveform.played_rgba = style.accent.toU32();
	command.waveform.remaining_rgba = style.textSecondary.withAlpha(0.60F).toU32();
	command.waveform.clip_x = clip.x;
	command.waveform.clip_y = clip.y;
	command.waveform.clip_w = static_cast<oa::U32>(clip.w);
	command.waveform.clip_h = static_cast<oa::U32>(clip.h);
	if (clip.w <= 0 || clip.h <= 0) return changed;
	impl_->appendBlit(oa::move(command));

	const oa::I32 playheadX = oa::clamp(
		inRect.x + static_cast<oa::I32>(static_cast<oa::F32>(inRect.w) * inOutFraction),
		inRect.x,
		inRect.x + inRect.w - 1);
	const oa::I32 idlePlayhead = oa::max<oa::I32>(1,
		static_cast<oa::I32>(oa::round(2.0F * impl_->contentScale)));
	const oa::I32 activePlayhead = oa::max<oa::I32>(idlePlayhead,
		static_cast<oa::I32>(oa::round(3.0F * impl_->contentScale)));
	this->rect({playheadX, inRect.y,
		interaction.hovered || interaction.held
			? activePlayhead : idlePlayhead, inRect.h},
		interaction.hovered || interaction.held
			? style.accentHover : style.text);
	impl_->addAccessibilityNode(
		id,
		oa::UiAccessibilityRole::Timeline,
		inRect,
		inId,
		{},
		oa::UiAccessibilityState::None,
		oa::UiAccessibilityAction::Focus
			| oa::UiAccessibilityAction::Increment
			| oa::UiAccessibilityAction::Decrement
			| oa::UiAccessibilityAction::SetValue,
		input_.focusId == id,
		true,
		0.0,
		1.0,
		inOutFraction);
	return changed;
}


// ─── Image widgets ────────────────────────────────────────────────────────────

void oa::Ui::image(
	const oa::Texture& inTexture,
	VkPipelineStageFlags2 inSourceStageMask,
	VkAccessFlags2 inSourceAccessMask)
{
	if (!impl_) return;
	if (not inTexture.isValid()
		or oa::TextureAccess::engine(inTexture) != impl_->rt) {
		impl_->setFrameError(oa::Status::invalidArgument(
			"oa::Ui::image requires a valid texture from its engine"));
		return;
	}
	if (inTexture.isImageBacked()) {
		imageVkRgba(
			oa::TextureAccess::image(inTexture),
			oa::TextureAccess::view(inTexture),
			inTexture.width(),
			inTexture.height(),
			static_cast<VkImageLayout>(oa::TextureAccess::layout(inTexture)),
			inSourceStageMask,
			inSourceAccessMask);
		return;
	}
	const auto& owner = oa::TextureAccess::bufferOwner(inTexture);
	if (not owner) {
		impl_->setFrameError(oa::Status::invalidArgument(
			"oa::Ui::image requires owned buffer storage"));
		return;
	}
	oa::I32 dstX = 0, dstY = 0;
	oa::I32 dstW = inTexture.width(), dstH = inTexture.height();
	if (!impl_->panelStack.empty()) {
		const auto& ps = impl_->panelStack.back();
		dstX = ps.rect.x; dstY = ps.rect.y;
		dstW = ps.rect.w; dstH = ps.rect.h;
	}
	const oa::PixelRect destination{dstX, dstY, dstW, dstH};
	const oa::PixelRect clip = impl_->clipFor(destination);
	if (clip.w <= 0 || clip.h <= 0) return;
	BlitCmd bc{};
	bc.kind         = BlitKind::Rgba;
	bc.rgba.src_idx = inTexture.bindlessIndex();
	bc.rgba.dst_idx = 0;
	bc.rgba.src_w   = static_cast<oa::U32>(inTexture.width());
	bc.rgba.src_h   = static_cast<oa::U32>(inTexture.height());
	bc.rgba.dst_x   = dstX;
	bc.rgba.dst_y   = dstY;
	bc.rgba.dst_w   = static_cast<oa::U32>(dstW);
	bc.rgba.dst_h   = static_cast<oa::U32>(dstH);
	bc.rgba.clip_x  = clip.x;
	bc.rgba.clip_y  = clip.y;
	bc.rgba.clip_w  = static_cast<oa::U32>(clip.w);
	bc.rgba.clip_h  = static_cast<oa::U32>(clip.h);
	impl_->appendBlit(oa::move(bc));
	for (const auto& retained : impl_->pendingTextureOwners) {
		if (retained.get() == owner.get()) return;
	}
	impl_->pendingTextureOwners.pushBack(owner);
}

void oa::Ui::imageVkRgba(
	void* inImage,
	void* inImageView,
	oa::I32 inW,
	oa::I32 inH,
	VkImageLayout inLayout,
	VkPipelineStageFlags2 inSourceStageMask,
	VkAccessFlags2 inSourceAccessMask) {
	if (!impl_ || !inImageView) return;
	oa::I32 dstX = 0, dstY = 0, dstW = inW, dstH = inH;
	if (!impl_->panelStack.empty()) {
		const auto& ps = impl_->panelStack.back();
		dstX = ps.rect.x; dstY = ps.rect.y;
		dstW = ps.rect.w; dstH = ps.rect.h;
	}
	const oa::PixelRect destination{dstX, dstY, dstW, dstH};
	const oa::PixelRect clip = impl_->clipFor(destination);
	if (clip.w <= 0 || clip.h <= 0) return;
	BlitCmd bc{};
	bc.kind = BlitKind::ImageRgba;
	bc.srcImage = static_cast<VkImage>(inImage);
	bc.srcImageView = static_cast<VkImageView>(inImageView);
	bc.srcImageLayout = inLayout;
	bc.srcStageMask = inSourceStageMask;
	bc.srcAccessMask = inSourceAccessMask;
	bc.imageRgba.src_w = static_cast<oa::U32>(inW);
	bc.imageRgba.src_h = static_cast<oa::U32>(inH);
	bc.imageRgba.dst_idx = 0;
	bc.imageRgba.dst_x = dstX;
	bc.imageRgba.dst_y = dstY;
	bc.imageRgba.dst_w = static_cast<oa::U32>(dstW);
	bc.imageRgba.dst_h = static_cast<oa::U32>(dstH);
	bc.imageRgba.clip_x = clip.x;
	bc.imageRgba.clip_y = clip.y;
	bc.imageRgba.clip_w = static_cast<oa::U32>(clip.w);
	bc.imageRgba.clip_h = static_cast<oa::U32>(clip.h);
	impl_->appendBlit(oa::move(bc));
}

void oa::Ui::rect(
	oa::PixelRect inRect,
	oa::Color inColor,
	oa::F32 inCornerRadius) {
	if (!impl_ || inRect.w <= 0 || inRect.h <= 0) {
		return;
	}
	if (!oa::isFinite(inCornerRadius) || inCornerRadius < 0.0F) {
		impl_->setFrameError(oa::Status::invalidArgument(
			"oa::Ui::rect requires a finite non-negative corner radius"));
		return;
	}
	const oa::PixelRect clip = impl_->clipFor(inRect);
	if (clip.w <= 0 || clip.h <= 0) return;
	BlitCmd bc{};
	bc.kind = BlitKind::Rect;
	bc.rect.dst_idx = 0;
	bc.rect.dst_x = inRect.x;
	bc.rect.dst_y = inRect.y;
	bc.rect.dst_w = static_cast<oa::U32>(inRect.w);
	bc.rect.dst_h = static_cast<oa::U32>(inRect.h);
	bc.rect.rgba = inColor.toU32();
	bc.rect.clip_x = clip.x;
	bc.rect.clip_y = clip.y;
	bc.rect.clip_w = static_cast<oa::U32>(clip.w);
	bc.rect.clip_h = static_cast<oa::U32>(clip.h);
	bc.rect.corner_radius = oa::min(
		inCornerRadius,
		0.5F * static_cast<oa::F32>(oa::min(inRect.w, inRect.h)));
	impl_->appendBlit(oa::move(bc));
}

void oa::Ui::rectOutline(
	oa::PixelRect inRect,
	oa::Color inColor,
	oa::U32 inThickness,
	oa::F32 inCornerRadius) {
	if (!impl_ || inRect.w <= 0 || inRect.h <= 0) {
		return;
	}
	if (!oa::isFinite(inCornerRadius) || inCornerRadius < 0.0F) {
		impl_->setFrameError(oa::Status::invalidArgument(
			"oa::Ui::rectOutline requires a finite non-negative corner radius"));
		return;
	}
	const oa::PixelRect clip = impl_->clipFor(inRect);
	if (clip.w <= 0 || clip.h <= 0) return;
	BlitCmd bc{};
	bc.kind = BlitKind::RectOutline;
	bc.rectOutline.dst_idx = 0;
	bc.rectOutline.dst_x = inRect.x;
	bc.rectOutline.dst_y = inRect.y;
	bc.rectOutline.dst_w = static_cast<oa::U32>(inRect.w);
	bc.rectOutline.dst_h = static_cast<oa::U32>(inRect.h);
	bc.rectOutline.thickness = oa::max<oa::U32>(1, inThickness);
	bc.rectOutline.rgba = inColor.toU32();
	bc.rectOutline.clip_x = clip.x;
	bc.rectOutline.clip_y = clip.y;
	bc.rectOutline.clip_w = static_cast<oa::U32>(clip.w);
	bc.rectOutline.clip_h = static_cast<oa::U32>(clip.h);
	bc.rectOutline.corner_radius = oa::min(
		inCornerRadius,
		0.5F * static_cast<oa::F32>(oa::min(inRect.w, inRect.h)));
	impl_->appendBlit(oa::move(bc));
}

void oa::Ui::line(
	oa::vlm::Vec2 inBegin,
	oa::vlm::Vec2 inEnd,
	oa::Color inColor,
	oa::F32 inThickness) {
	if (!impl_ || !oa::isFinite(inBegin.x) || !oa::isFinite(inBegin.y)
		|| !oa::isFinite(inEnd.x) || !oa::isFinite(inEnd.y)
		|| !oa::isFinite(inThickness) || inThickness <= 0.0F) return;
	const oa::F32 padding = inThickness * 0.5F + 1.5F;
	const oa::I32 x = static_cast<oa::I32>(
		oa::floor(oa::min(inBegin.x, inEnd.x) - padding));
	const oa::I32 y = static_cast<oa::I32>(
		oa::floor(oa::min(inBegin.y, inEnd.y) - padding));
	const oa::I32 right = static_cast<oa::I32>(
		oa::ceil(oa::max(inBegin.x, inEnd.x) + padding));
	const oa::I32 bottom = static_cast<oa::I32>(
		oa::ceil(oa::max(inBegin.y, inEnd.y) + padding));
	if (right <= x || bottom <= y) return;
	const oa::PixelRect bounds = impl_->clipFor({x, y, right - x, bottom - y});
	if (bounds.w <= 0 || bounds.h <= 0) return;
	BlitCmd command{};
	command.kind = BlitKind::Line;
	command.line.dst_idx = 0U;
	command.line.x0 = inBegin.x;
	command.line.y0 = inBegin.y;
	command.line.x1 = inEnd.x;
	command.line.y1 = inEnd.y;
	command.line.thickness = inThickness;
	command.line.rgba = inColor.toU32();
	command.line.bounds_x = static_cast<oa::U32>(bounds.x);
	command.line.bounds_y = static_cast<oa::U32>(bounds.y);
	command.line.bounds_w = static_cast<oa::U32>(bounds.w);
	command.line.bounds_h = static_cast<oa::U32>(bounds.h);
	impl_->appendBlit(oa::move(command));
}

void oa::Ui::nodeCanvasGrid(
	const oa::NodeCanvas& inCanvas,
	oa::PixelRect inRect,
	const oa::UiNodeCanvasGridConfig& inConfig) {
	if (!impl_) return;
	auto grid = inCanvas.grid(
		inConfig.minimumScreenSpacing,
		inConfig.baseWorldStep,
		inConfig.majorEvery,
		inConfig.superMajorEvery);
	if (not grid.isOk()) {
		impl_->setFrameError(grid.getStatus());
		return;
	}
	this->grid(inRect, {
		.origin = {
			static_cast<oa::F32>(inRect.x) + grid->originScreen.x,
			static_cast<oa::F32>(inRect.y) + grid->originScreen.y,
		},
		.minorSpacing = {grid->minorScreenStep, grid->minorScreenStep},
		.majorEvery = grid->majorEvery,
		.superMajorEvery = grid->superMajorEvery,
		.minorThickness = inConfig.minorThickness,
		.majorThickness = inConfig.majorThickness,
		.superMajorThickness = inConfig.superMajorThickness,
		.axisThickness = inConfig.axisThickness,
		.fillBackground = inConfig.fillBackground,
		.drawGrid = true,
		.drawAxes = inConfig.drawAxes,
	});
}

void oa::Ui::grid(oa::PixelRect inRect, const oa::UiGridConfig& inConfig) {
	if (!impl_) return;
	const bool validThickness =
		oa::isFinite(inConfig.minorThickness)
		and oa::isFinite(inConfig.majorThickness)
		and oa::isFinite(inConfig.superMajorThickness)
		and oa::isFinite(inConfig.axisThickness)
		and inConfig.minorThickness > 0.0F
		and inConfig.majorThickness > 0.0F
		and inConfig.superMajorThickness > 0.0F
		and inConfig.axisThickness > 0.0F
		and inConfig.minorThickness <= 16.0F
		and inConfig.majorThickness <= 16.0F
		and inConfig.superMajorThickness <= 16.0F
		and inConfig.axisThickness <= 16.0F;
	const bool validSpacing = oa::isFinite(inConfig.origin.x)
		and oa::isFinite(inConfig.origin.y)
		and oa::isFinite(inConfig.minorSpacing.x)
		and oa::isFinite(inConfig.minorSpacing.y)
		and oa::isFinite(inConfig.opacity)
		and inConfig.minorSpacing.x > 0.0F
		and inConfig.minorSpacing.y > 0.0F
		and inConfig.opacity >= 0.0F and inConfig.opacity <= 1.0F
		and inConfig.majorEvery >= 2U and inConfig.majorEvery <= 100U
		and inConfig.superMajorEvery >= inConfig.majorEvery * 2U
		and inConfig.superMajorEvery <= 10000U
		and inConfig.superMajorEvery % inConfig.majorEvery == 0U;
	if (!inRect.isValid() or not validThickness or not validSpacing) {
		impl_->setFrameError(oa::Status::invalidArgument(
			"oa::Ui::grid requires a valid rectangle, finite positive spacing/thickness, opacity in [0,1], and divisible decimal tiers"));
		return;
	}
	const oa::PixelRect clip = impl_->clipFor(inRect);
	if (!clip.isValid()) return;
	const oa::UiStyle& style = currentStyle();
	const bool light = style.background.r + style.background.g
		+ style.background.b > 1.5F;
	const oa::Color topDefault = mixColor(
		style.surface, {1.0F, 1.0F, 1.0F, style.surface.a}, 0.055F);
	const oa::Color bottomDefault = mixColor(
		style.surface, {0.0F, 0.0F, 0.0F, style.surface.a}, 0.045F);
	const oa::Color minorDefault = light
		? oa::Color{0.0F, 0.0F, 0.0F, 0.045F}
		: oa::Color{1.0F, 1.0F, 1.0F, 0.045F};
	const oa::Color majorDefault = light
		? oa::Color{0.0F, 0.0F, 0.0F, 0.14F}
		: oa::Color{0.78F, 0.80F, 0.82F, 0.16F};
	const oa::Color superMajorDefault = light
		? oa::Color{0.12F, 0.12F, 0.12F, 0.24F}
		: oa::Color{0.03F, 0.03F, 0.03F, 0.62F};
	const oa::Color axisDefault{0.0F, 0.0F, 0.0F, light ? 0.60F : 0.82F};
	const auto resolveLineColor = [&](oa::Color inRequested, oa::Color inFallback) {
		const oa::Color color = resolveGridColor(inRequested, inFallback);
		return color.withAlpha(color.a * inConfig.opacity);
	};
	BlitCmd command{};
	command.kind = BlitKind::CanvasGrid;
	command.canvasGrid.dst_idx = 0U;
	command.canvasGrid.dst_x = clip.x;
	command.canvasGrid.dst_y = clip.y;
	command.canvasGrid.dst_w = static_cast<oa::U32>(clip.w);
	command.canvasGrid.dst_h = static_cast<oa::U32>(clip.h);
	command.canvasGrid.origin_x = inConfig.origin.x;
	command.canvasGrid.origin_y = inConfig.origin.y;
	command.canvasGrid.minor_spacing_x = inConfig.minorSpacing.x;
	command.canvasGrid.minor_spacing_y = inConfig.minorSpacing.y;
	command.canvasGrid.major_every = inConfig.majorEvery;
	command.canvasGrid.super_major_every = inConfig.superMajorEvery;
	command.canvasGrid.minor_thickness = inConfig.minorThickness;
	command.canvasGrid.major_thickness = inConfig.majorThickness;
	command.canvasGrid.super_major_thickness = inConfig.superMajorThickness;
	command.canvasGrid.axis_thickness = inConfig.axisThickness;
	command.canvasGrid.background_top_rgba = resolveGridColor(
		inConfig.backgroundTop, topDefault).toU32();
	command.canvasGrid.background_bottom_rgba = resolveGridColor(
		inConfig.backgroundBottom, bottomDefault).toU32();
	command.canvasGrid.minor_rgba = resolveLineColor(
		inConfig.minorColor, minorDefault).toU32();
	command.canvasGrid.major_rgba = resolveLineColor(
		inConfig.majorColor, majorDefault).toU32();
	command.canvasGrid.super_major_rgba = resolveLineColor(
		inConfig.superMajorColor, superMajorDefault).toU32();
	command.canvasGrid.axis_rgba = resolveLineColor(
		inConfig.axisColor, axisDefault).toU32();
	command.canvasGrid.flags = (inConfig.fillBackground ? 1U : 0U)
		| (inConfig.drawAxes ? 2U : 0U)
		| (inConfig.drawGrid ? 4U : 0U);
	command.canvasGrid.clip_x = clip.x;
	command.canvasGrid.clip_y = clip.y;
	command.canvasGrid.clip_w = static_cast<oa::U32>(clip.w);
	command.canvasGrid.clip_h = static_cast<oa::U32>(clip.h);
	impl_->appendBlit(oa::move(command));
}

void oa::Ui::rectOutlines(
	const oa::DetectionBuffer& inDetections,
	oa::PixelRect inDstRect,
	oa::PixelRect inClipRect,
	oa::Color inColor,
	oa::U32 inThickness) {
	if (!impl_ || inDetections.count() == 0
		|| inDetections.bindlessIndex() == OA_BINDLESS_INVALID
		|| inDstRect.w <= 0 || inDstRect.h <= 0
		|| inClipRect.w <= 0 || inClipRect.h <= 0) {
		return;
	}
	const oa::PixelRect clip = impl_->clipFor(
		intersectPixelRects(inDstRect, inClipRect));
	if (clip.w <= 0 || clip.h <= 0) return;
	BlitCmd bc{};
	bc.kind = BlitKind::RectOutlines;
	bc.rectOutlines.rect_idx = inDetections.bindlessIndex();
	bc.rectOutlines.count = inDetections.count();
	bc.rectOutlines.dst_idx = 0;
	bc.rectOutlines.dst_x = inDstRect.x;
	bc.rectOutlines.dst_y = inDstRect.y;
	bc.rectOutlines.dst_w = static_cast<oa::U32>(inDstRect.w);
	bc.rectOutlines.dst_h = static_cast<oa::U32>(inDstRect.h);
	bc.rectOutlines.clip_x = clip.x;
	bc.rectOutlines.clip_y = clip.y;
	bc.rectOutlines.clip_w = static_cast<oa::U32>(clip.w);
	bc.rectOutlines.clip_h = static_cast<oa::U32>(clip.h);
	bc.rectOutlines.thickness = oa::max<oa::U32>(1, inThickness);
	bc.rectOutlines.rgba = inColor.toU32();
	impl_->appendBlit(oa::move(bc));
}

void oa::Ui::glyphs(
	const oa::GlyphBuffer& inGlyphs,
	const oa::TextAtlas& inAtlas,
	oa::PixelRect inDstRect,
	oa::PixelRect inClipRect) {
	const oa::U32 atlasIndex = inAtlas.atlasBindlessIndex(oa::FontId::Sans);
	if (!impl_ || inGlyphs.count() == 0
		|| inGlyphs.bindlessIndex() == OA_BINDLESS_INVALID
		|| atlasIndex == OA_BINDLESS_INVALID
		|| inDstRect.w <= 0 || inDstRect.h <= 0
		|| inClipRect.w <= 0 || inClipRect.h <= 0) {
		return;
	}
	const oa::PixelRect clip = impl_->clipFor(
		intersectPixelRects(inDstRect, inClipRect));
	if (clip.w <= 0 || clip.h <= 0) return;
	BlitCmd bc{};
	bc.kind = BlitKind::Glyphs;
	bc.glyphs.glyph_idx = inGlyphs.bindlessIndex();
	bc.glyphs.first = 0U;
	bc.glyphs.atlas_idx = atlasIndex;
	bc.glyphs.count = inGlyphs.count();
	bc.glyphs.batch = 0U;
	bc.glyphs.batch_count = 1U;
	bc.glyphs.dst_idx = 0;
	bc.glyphs.dst_x = inDstRect.x;
	bc.glyphs.dst_y = inDstRect.y;
	bc.glyphs.dst_w = static_cast<oa::U32>(inDstRect.w);
	bc.glyphs.dst_h = static_cast<oa::U32>(inDstRect.h);
	bc.glyphs.clip_x = clip.x;
	bc.glyphs.clip_y = clip.y;
	bc.glyphs.clip_w = static_cast<oa::U32>(clip.w);
	bc.glyphs.clip_h = static_cast<oa::U32>(clip.h);
	bc.glyphs.atlas_w = static_cast<oa::U32>(inAtlas.atlasWidth());
	bc.glyphs.atlas_h = static_cast<oa::U32>(inAtlas.atlasHeight());
	bc.glyphs.px_range = inAtlas.pxRange();
	impl_->appendBlit(oa::move(bc));
}

void oa::Ui::imagePlanar(oa::ImagePlanes& inPlanes, oa::I32 inDstX, oa::I32 inDstY) {
	if (!impl_) return;
	if (not inPlanes.isValid() or inPlanes.engine_ != impl_->rt) {
		impl_->setFrameError(oa::Status::invalidArgument(
			"oa::Ui::imagePlanar requires valid planes from its engine"));
		return;
	}

	oa::I32 dstX = inDstX, dstY = inDstY;
	oa::I32 dstW = inPlanes.width(), dstH = inPlanes.height();
	if (!impl_->panelStack.empty()) {
		const auto& ps = impl_->panelStack.back();
		dstX = ps.rect.x + inDstX;
		dstY = ps.rect.y + inDstY;
		dstW = ps.rect.w;
		dstH = ps.rect.h;
	}

	oa::U32 dtypes = 0;
	for (oa::U32 c = 0; c < static_cast<oa::U32>(inPlanes.channelCount()); ++c) {
		dtypes |= (static_cast<oa::U32>(inPlanes.planeDtype(c)) & 0x3U) << (c * 2U);
	}

	const oa::PixelRect destination{dstX, dstY, dstW, dstH};
	const oa::PixelRect clip = impl_->clipFor(destination);
	if (clip.w <= 0 || clip.h <= 0) return;
	BlitCmd bc{};
	bc.kind            = BlitKind::Planar;
	bc.planar.r_idx    = inPlanes.planeIndex(0U);
	bc.planar.g_idx    = inPlanes.planeIndex(1U);
	bc.planar.b_idx    = inPlanes.planeIndex(2U);
	bc.planar.a_idx    = inPlanes.planeIndex(3U);
	bc.planar.dst_idx  = 0;
	bc.planar.src_w    = static_cast<oa::U32>(inPlanes.width());
	bc.planar.src_h    = static_cast<oa::U32>(inPlanes.height());
	bc.planar.dst_x    = dstX;
	bc.planar.dst_y    = dstY;
	bc.planar.dtypes   = dtypes;
	bc.planar.channels = static_cast<oa::U32>(inPlanes.channelCount());
	bc.planar.dst_w    = static_cast<oa::U32>(dstW);
	bc.planar.dst_h    = static_cast<oa::U32>(dstH);
	bc.planar.clip_x   = clip.x;
	bc.planar.clip_y   = clip.y;
	bc.planar.clip_w   = static_cast<oa::U32>(clip.w);
	bc.planar.clip_h   = static_cast<oa::U32>(clip.h);
	impl_->appendBlit(oa::move(bc));
	for (oa::ImagePlanes* planes : impl_->usedImagePlanes) {
		if (planes == &inPlanes) return;
	}
	impl_->usedImagePlanes.pushBack(&inPlanes);
}

void oa::Ui::imagePlane(
	oa::ImagePlanes& inPlanes,
	oa::U32 inChannel,
	oa::I32 inDstX,
	oa::I32 inDstY)
{
	if (!impl_) return;
	if (not inPlanes.isValid() or inPlanes.engine_ != impl_->rt
		or inChannel >= inPlanes.channelCount()) {
		impl_->setFrameError(oa::Status::invalidArgument(
			"oa::Ui::imagePlane requires a valid channel from its engine"));
		return;
	}

	oa::I32 dstX = inDstX;
	oa::I32 dstY = inDstY;
	oa::I32 dstW = inPlanes.width();
	oa::I32 dstH = inPlanes.height();
	if (!impl_->panelStack.empty()) {
		const auto& panel = impl_->panelStack.back();
		dstX = panel.rect.x + inDstX;
		dstY = panel.rect.y + inDstY;
		dstW = panel.rect.w;
		dstH = panel.rect.h;
	}

	const oa::PixelRect destination{dstX, dstY, dstW, dstH};
	const oa::PixelRect clip = impl_->clipFor(destination);
	if (clip.w <= 0 || clip.h <= 0) return;
	BlitCmd command{};
	command.kind = BlitKind::Planar;
	command.planar.r_idx = inPlanes.planeIndex(inChannel);
	command.planar.g_idx = OA_BINDLESS_INVALID;
	command.planar.b_idx = OA_BINDLESS_INVALID;
	command.planar.a_idx = OA_BINDLESS_INVALID;
	command.planar.dst_idx = 0U;
	command.planar.src_w = static_cast<oa::U32>(inPlanes.width());
	command.planar.src_h = static_cast<oa::U32>(inPlanes.height());
	command.planar.dst_x = dstX;
	command.planar.dst_y = dstY;
	command.planar.dtypes = static_cast<oa::U32>(
		inPlanes.planeDtype(inChannel));
	command.planar.channels = 1U;
	command.planar.dst_w = static_cast<oa::U32>(dstW);
	command.planar.dst_h = static_cast<oa::U32>(dstH);
	command.planar.clip_x = clip.x;
	command.planar.clip_y = clip.y;
	command.planar.clip_w = static_cast<oa::U32>(clip.w);
	command.planar.clip_h = static_cast<oa::U32>(clip.h);
	impl_->appendBlit(oa::move(command));
	for (oa::ImagePlanes* planes : impl_->usedImagePlanes) {
		if (planes == &inPlanes) return;
	}
	impl_->usedImagePlanes.pushBack(&inPlanes);
}


// ─── Data visualization ──────────────────────────────────────────────────────

void oa::Ui::plotLine(
	oa::StringView inLabel,
	const oa::F32* inData,
	oa::I32 inCount,
	const oa::UiPlotConfig& inCfg) {
	if (!impl_ || !impl_->rt || inData == nullptr || inCount <= 0
		|| impl_->panelStack.empty()) return;
	const oa::PixelRect rect = impl_->panelStack.back().rect;
	const oa::PixelRect clip = impl_->clipFor(rect);
	if (rect.w <= 0 || rect.h <= 0 || clip.w <= 0 || clip.h <= 0) return;

	oa::U32 cacheIndex = 0U;
	oa::U32 slotIndex = 0U;
	auto* slot = impl_->uploadPlotValues(hashWidgetId(inLabel), inData,
		static_cast<oa::U32>(inCount), cacheIndex, slotIndex);
	if (slot == nullptr) return;
	oa::F32 yMin = inCfg.yMin;
	oa::F32 yMax = inCfg.yMax;
	if (inCfg.autoScale) {
		yMin = oa::Limits<oa::F32>::infinity();
		yMax = -oa::Limits<oa::F32>::infinity();
		const auto* values = static_cast<const oa::F32*>(slot->buffer.mappedPtr);
		for (oa::U32 index = 0; index < slot->count; ++index) {
			if (!oa::isFinite(values[index])) continue;
			yMin = oa::min(yMin, values[index]);
			yMax = oa::max(yMax, values[index]);
		}
	}
	if (!oa::isFinite(yMin) || !oa::isFinite(yMax)) return;
	if (yMax <= yMin) {
		const oa::F32 margin = oa::max(1.0e-4F, oa::abs(yMin) * 0.05F);
		yMin -= margin;
		yMax += margin;
	}

	if (inCfg.drawSurface or inCfg.showGrid) {
		grid(rect, resolveChartGridConfig(
			rect, impl_->contentScale, inCfg.drawSurface, inCfg.showGrid));
	}
	BlitCmd command{};
	command.kind = BlitKind::PlotLine;
	command.plotLine.values_idx = slot->buffer.bindlessIndex;
	command.plotLine.count = slot->count;
	command.plotLine.dst_idx = 0U;
	command.plotLine.dst_x = rect.x;
	command.plotLine.dst_y = rect.y;
	command.plotLine.dst_w = static_cast<oa::U32>(rect.w);
	command.plotLine.dst_h = static_cast<oa::U32>(rect.h);
	command.plotLine.x_min = inCfg.xMin;
	command.plotLine.x_max = inCfg.xMax;
	command.plotLine.y_min = yMin;
	command.plotLine.y_max = yMax;
	command.plotLine.rgba = inCfg.color.toU32();
	command.plotLine.flags = inCfg.fill ? 2U : 0U;
	command.plotLine.antialias_samples = resolveLineSampleCount(
		inCfg.antialiasSamples);
	command.plotLine.line_width = oa::clamp(
		oa::isFinite(inCfg.lineWidth) ? inCfg.lineWidth : 1.35F,
		0.5F, 16.0F);
	command.plotLine.clip_x = clip.x;
	command.plotLine.clip_y = clip.y;
	command.plotLine.clip_w = static_cast<oa::U32>(clip.w);
	command.plotLine.clip_h = static_cast<oa::U32>(clip.h);
	impl_->appendBlit(oa::move(command));
	impl_->usedPlots.pushBack({.cache = cacheIndex, .slot = slotIndex});
}

void oa::Ui::plotLineXY(
	oa::StringView inLabel,
	const oa::F32* inX,
	const oa::F32* inY,
	oa::I32 inCount,
	const oa::UiPlotConfig& inCfg) {
	if (!impl_ || !impl_->rt || inX == nullptr || inY == nullptr
		|| inCount <= 0 || inCount > static_cast<oa::I32>(Impl::kPlotCapacity / 2U)
		|| impl_->panelStack.empty()) return;
	const oa::PixelRect rect = impl_->panelStack.back().rect;
	const oa::PixelRect clip = impl_->clipFor(rect);
	if (rect.w <= 0 || rect.h <= 0 || clip.w <= 0 || clip.h <= 0) return;
	if (!oa::isFinite(inCfg.xMin) || !oa::isFinite(inCfg.xMax)
		|| !oa::isFinite(inCfg.yMin) || !oa::isFinite(inCfg.yMax)
		|| inCfg.xMax <= inCfg.xMin || inCfg.yMax <= inCfg.yMin) return;

	oa::Vector<oa::F32> packed(static_cast<oa::Usize>(inCount) * 2U);
	oa::memcpy(packed.data(), inX,
		static_cast<oa::Usize>(inCount) * sizeof(oa::F32));
	oa::memcpy(packed.data() + inCount, inY,
		static_cast<oa::Usize>(inCount) * sizeof(oa::F32));
	oa::U32 cacheIndex = 0U;
	oa::U32 slotIndex = 0U;
	auto* slot = impl_->uploadPlotValues(hashWidgetId(inLabel), packed.data(),
		static_cast<oa::U32>(packed.size()), cacheIndex, slotIndex);
	if (slot == nullptr) return;
	if (inCfg.drawSurface or inCfg.showGrid) {
		grid(rect, resolveChartGridConfig(
			rect, impl_->contentScale, inCfg.drawSurface, inCfg.showGrid));
	}
	BlitCmd command{};
	command.kind = BlitKind::PlotLine;
	command.plotLine.values_idx = slot->buffer.bindlessIndex;
	command.plotLine.count = static_cast<oa::U32>(inCount);
	command.plotLine.dst_idx = 0U;
	command.plotLine.dst_x = rect.x;
	command.plotLine.dst_y = rect.y;
	command.plotLine.dst_w = static_cast<oa::U32>(rect.w);
	command.plotLine.dst_h = static_cast<oa::U32>(rect.h);
	command.plotLine.x_min = inCfg.xMin;
	command.plotLine.x_max = inCfg.xMax;
	command.plotLine.y_min = inCfg.yMin;
	command.plotLine.y_max = inCfg.yMax;
	command.plotLine.rgba = inCfg.color.toU32();
	command.plotLine.flags = 4U;
	command.plotLine.antialias_samples = resolveLineSampleCount(
		inCfg.antialiasSamples);
	command.plotLine.line_width = oa::clamp(
		oa::isFinite(inCfg.lineWidth) ? inCfg.lineWidth : 1.35F,
		0.5F, 16.0F);
	command.plotLine.clip_x = clip.x;
	command.plotLine.clip_y = clip.y;
	command.plotLine.clip_w = static_cast<oa::U32>(clip.w);
	command.plotLine.clip_h = static_cast<oa::U32>(clip.h);
	impl_->appendBlit(oa::move(command));
	impl_->usedPlots.pushBack({.cache = cacheIndex, .slot = slotIndex});
}

void oa::Ui::plotLineRing(
	oa::StringView inLabel,
	const oa::F32* inData,
	oa::I32 inCount,
	oa::I32 inOffset,
	const oa::UiPlotConfig& inCfg) {
	if (inData == nullptr || inCount <= 0) return;
	oa::Vector<oa::F32> ordered(static_cast<oa::Usize>(inCount));
	const oa::I32 offset = ((inOffset % inCount) + inCount) % inCount;
	for (oa::I32 index = 0; index < inCount; ++index) {
		ordered[static_cast<oa::Usize>(index)] =
			inData[static_cast<oa::Usize>((offset + index) % inCount)];
	}
	plotLine(inLabel, ordered.data(), inCount, inCfg);
}
void oa::Ui::heatmap(oa::StringView /*inLabel*/, const oavk::Buffer& inBuffer,
	const oa::UiHeatmapConfig& inCfg) {
	if (!impl_ || inBuffer.bindlessIndex == OA_BINDLESS_INVALID
		|| inCfg.rows <= 0 || inCfg.cols <= 0
		|| impl_->panelStack.empty()) return;
	const oa::U64 elementCount = static_cast<oa::U64>(inCfg.rows)
		* static_cast<oa::U64>(inCfg.cols);
	const oa::U64 firstElement = inCfg.offsetElements;
	if (elementCount > oa::Limits<oa::U64>::max() - firstElement
		|| firstElement + elementCount > inBuffer.size / sizeof(oa::U32)) {
		impl_->setFrameError(oa::Status::invalidArgument(
			"oa::Ui::heatmap source buffer is smaller than its declared matrix range"));
		return;
	}
	const oa::PixelRect rect = impl_->panelStack.back().rect;
	const oa::PixelRect clip = impl_->clipFor(rect);
	if (rect.w <= 0 || rect.h <= 0 || clip.w <= 0 || clip.h <= 0) return;
	oa::F32 vMin = inCfg.vMin;
	oa::F32 vMax = inCfg.vMax;
	if (!oa::isFinite(vMin) || !oa::isFinite(vMax)) return;
	if (vMax <= vMin) {
		const oa::F32 margin = oa::max(1.0e-4F, oa::abs(vMin) * 0.05F);
		vMin -= margin;
		vMax += margin;
	}
	BlitCmd command{};
	command.kind = BlitKind::Heatmap;
	command.heatmap.values_idx = inBuffer.bindlessIndex;
	command.heatmap.rows = static_cast<oa::U32>(inCfg.rows);
	command.heatmap.cols = static_cast<oa::U32>(inCfg.cols);
	command.heatmap.dst_idx = 0U;
	command.heatmap.dst_x = rect.x;
	command.heatmap.dst_y = rect.y;
	command.heatmap.dst_w = static_cast<oa::U32>(rect.w);
	command.heatmap.dst_h = static_cast<oa::U32>(rect.h);
	command.heatmap.v_min = vMin;
	command.heatmap.v_max = vMax;
	command.heatmap.colormap = oa::min(inCfg.colormap, 3U);
	command.heatmap.value_type = oa::min(inCfg.valueType, 2U);
	command.heatmap.offset_elements = inCfg.offsetElements;
	command.heatmap.flags = inCfg.showGrid ? 1U : 0U;
	command.heatmap.clip_x = clip.x;
	command.heatmap.clip_y = clip.y;
	command.heatmap.clip_w = static_cast<oa::U32>(clip.w);
	command.heatmap.clip_h = static_cast<oa::U32>(clip.h);
	impl_->appendBlit(oa::move(command));
}

void oa::Ui::heatmap(oa::StringView inLabel, const oa::Matrix& inMatrix,
	const oa::UiHeatmapConfig& inCfg) {
	if (inMatrix.isEmpty() || inMatrix.rank() != 2) return;
	oa::UiHeatmapConfig config = inCfg;
	if (config.rows <= 0) config.rows = static_cast<oa::I32>(inMatrix.size(0));
	if (config.cols <= 0) config.cols = static_cast<oa::I32>(inMatrix.size(1));
	switch (inMatrix.getDtype()) {
		case oa::ScalarType::Float32: config.valueType = 0U; break;
		case oa::ScalarType::UInt32: config.valueType = 1U; break;
		case oa::ScalarType::Int32: config.valueType = 2U; break;
		default: return;
	}
	const oa::U64 offsetElements = static_cast<oa::U64>(config.offsetElements)
		+ inMatrix.byteOffset() / sizeof(oa::U32);
	if (offsetElements > oa::Limits<oa::U32>::max()) {
		if (impl_) {
			impl_->setFrameError(oa::Status::error(
				oa::StatusCode::OutOfRange,
				"oa::Ui::heatmap matrix offset exceeds the shader ABI"));
		}
		return;
	}
	config.offsetElements = static_cast<oa::U32>(offsetElements);
	const oavk::Buffer buffer = oa::MatrixAccess::descriptor(inMatrix);
	heatmap(inLabel, buffer, config);
}

void oa::Ui::heatmap(oa::StringView inLabel, const oa::F32* inData,
	oa::I32 inRows, oa::I32 inCols, const oa::UiHeatmapConfig& inCfg) {
	if (!impl_ || inData == nullptr || inRows <= 0 || inCols <= 0) return;
	const oa::I64 count64 = static_cast<oa::I64>(inRows) * inCols;
	if (count64 <= 0 || count64 > Impl::kPlotCapacity) return;
	oa::U32 cacheIndex = 0U;
	oa::U32 slotIndex = 0U;
	auto* slot = impl_->uploadPlotValues(hashWidgetId(inLabel), inData,
		static_cast<oa::U32>(count64), cacheIndex, slotIndex);
	if (slot == nullptr) return;
	oa::UiHeatmapConfig config = inCfg;
	config.rows = inRows;
	config.cols = inCols;
	config.valueType = 0U;
	config.offsetElements = 0U;
	heatmap(inLabel, slot->buffer, config);
	impl_->usedPlots.pushBack({.cache = cacheIndex, .slot = slotIndex});
}
