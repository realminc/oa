// Ui - persistent hinted text atlas, layout, and GPU glyph batches.
//
// exact-size grayscale coverage strikes are generated offline and embedded as
// a C++ byte array. No font loading or glyph rasterization occurs at runtime.
//
// Embedded fonts:
//   IBM Plex Sans  - UI labels, panels, and detection overlays
//   Intel One Mono - code, expressions, coordinates, and diagnostics

#pragma once

#include <oa/core/types.h>
#include <oa/core/status.h>
#include <oa/ui/canvas.h>   // oa::vlm::Vec2
#include <oa/runtime/buffer.h>
#include <oa/runtime/event.h>

namespace oa { class Engine; }


// ─── GlyphInfo ──────────────────────────────────────────────────────────────

namespace oa {

struct GlyphInfo {
	oa::U32  codepoint  = 0;
	oa::F32  atlasX     = 0.0F;  // pixel rectangle in the atlas
	oa::F32  atlasY     = 0.0F;
	oa::F32  atlasW     = 0.0F;
	oa::F32  atlasH     = 0.0F;
	oa::F32  bearingX   = 0.0F;  // layout metrics
	oa::F32  bearingY   = 0.0F;
	oa::F32  advance    = 0.0F;
	oa::F32  inkW       = 0.0F;  // visible glyph bounds inside the atlas cell
	oa::F32  inkH       = 0.0F;
	oa::F32  rasterSize = 0.0F;  // physical-pixel strike used for this glyph
};


// ─── FontId ─────────────────────────────────────────────────────────────────

enum class FontId : oa::U8 {
	Sans  = 0,   // IBM Plex Sans
	Mono  = 1,   // Intel One Mono
};


// ─── TextAtlas ──────────────────────────────────────────────────────────────
// Manages the device-side coverage atlas and the glyph metadata table.

class TextAtlas {
public:
	TextAtlas();
	TextAtlas(const TextAtlas&)            = delete;
	TextAtlas& operator=(const TextAtlas&) = delete;
	TextAtlas(TextAtlas&&) noexcept;
	TextAtlas& operator=(TextAtlas&&) noexcept;
	~TextAtlas();

	[[nodiscard]] oa::Status init(oa::Engine& inRt);

	// Selects the nearest generated physical-pixel strike. Returns nullptr for
	// unmapped codepoints or an invalid font/size.
	[[nodiscard]] const GlyphInfo* findGlyph(
		FontId inFont,
		oa::U32 inCodepoint,
		oa::F32 inPixelSize) const noexcept;

	[[nodiscard]] oa::U32  atlasBindlessIndex(FontId inFont) const noexcept;
	// Zero identifies direct grayscale coverage to the shared glyph shader;
	// positive values remain reserved for a future SDF lowering.
	[[nodiscard]] oa::F32  pxRange()          const noexcept { return 0.0F; }
	[[nodiscard]] oa::F32  atlasWidth()       const noexcept { return atlasW_; }
	[[nodiscard]] oa::F32  atlasHeight()      const noexcept { return atlasH_; }

private:
	void reset_() noexcept;

	struct Impl;
	oa::UniquePtr<Impl> impl_;
	oa::F32 atlasW_ = 0.0F;
	oa::F32 atlasH_ = 0.0F;
};

// One source-anchored glyph quad. anchor is normalized to the source image;
// offsets and dimensions are compose-target pixels so labels remain readable
// while the image camera pans and zooms.
struct GlyphInstance {
	oa::F32 anchorX = 0.0F;
	oa::F32 anchorY = 0.0F;
	oa::F32 offsetX = 0.0F;
	oa::F32 offsetY = 0.0F;
	oa::F32 width = 0.0F;
	oa::F32 height = 0.0F;
	oa::U32 atlasX = 0;
	oa::U32 atlasY = 0;
	oa::U32 atlasW = 0;
	oa::U32 atlasH = 0;
	oa::U32 color = 0xF5F5F5FFU;
	// Bit 0 rotates the source coverage counter-clockwise for bottom-to-top text.
	// Higher bits are reserved for Ui's deterministic non-overlap batching.
	oa::U32 flags = 0;
};
static_assert(sizeof(GlyphInstance) == 48);

class GlyphBuffer {
public:
	GlyphBuffer() = default;
	GlyphBuffer(const GlyphBuffer&) = delete;
	GlyphBuffer& operator=(const GlyphBuffer&) = delete;
	GlyphBuffer(GlyphBuffer&& inOther) noexcept;
	GlyphBuffer& operator=(GlyphBuffer&& inOther) noexcept;
	~GlyphBuffer();

	[[nodiscard]] static oa::Result<GlyphBuffer> createHostUpload(
		oa::Engine& inRuntime,
		oa::U32 inCapacity);

	[[nodiscard]] oa::Status upload(oa::Span<const GlyphInstance> inGlyphs);
	[[nodiscard]] oa::Status markConsumed(const oa::Event& inCompletion);

	[[nodiscard]] bool isReady() const;
	[[nodiscard]] bool isValid() const noexcept { return buffer_.buffer != nullptr; }
	[[nodiscard]] oa::U32 count() const noexcept { return count_; }
	[[nodiscard]] oa::U32 capacity() const noexcept { return capacity_; }
	[[nodiscard]] oa::U32 bindlessIndex() const noexcept { return buffer_.bindlessIndex; }

private:
	void reset_() noexcept;

	oa::Engine* runtime_ = nullptr;
	oavk::Buffer buffer_;
	oa::Event consumerCompletion_;
	oa::U32 count_ = 0;
	oa::U32 capacity_ = 0;
};


// ─── TextLayout ─────────────────────────────────────────────────────────────
// CPU-side layout: strictly decodes UTF-8, selects explicit per-codepoint font
// fallback, and uses HarfBuzz for OpenType substitution, positioning, and
// source-byte clusters over OA's embedded left-to-right coverage. Paragraph
// BiDi and right-to-left script coverage are intentionally outside this API.

struct PositionedGlyph {
	oa::F32  x          = 0.0F;
	oa::F32  y          = 0.0F;
	oa::U32  codepoint  = 0;
	oa::U32  cluster    = 0;  // UTF-8 byte offset of the source shaping cluster
	oa::U32  color      = 0xF5F5F5FFU;  // packed RGBA8 — default: text.primary
	FontId font = FontId::Sans;
};

struct TextLayoutConfig {
	FontId font       = FontId::Sans;
	oa::F32    size       = 14.0F;
	oa::F32    wrapWidth  = 0.0F;   // 0 = no wrap
	bool     monospace  = false;
};

class TextLayout {
public:
	// Shape inText at inOrigin.  Appends to inOutGlyphs (does not clear).
	void shape(
		const TextAtlas&        inAtlas,
		oa::StringView              inText,
		oa::vlm::Vec2                    inOrigin,
		const TextLayoutConfig& inCfg,
		oa::U32                     inColor,
		oa::Vector<PositionedGlyph>& inOutGlyphs) const;

	// measure text extent without appending.
	[[nodiscard]] oa::vlm::Vec2 measure(
		const TextAtlas&        inAtlas,
		oa::StringView              inText,
		const TextLayoutConfig& inCfg) const;
};

}  // namespace oa
