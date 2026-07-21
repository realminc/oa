// OaUi - persistent SDF text atlas, layout, and GPU glyph batches.
//
// The atlas is generated offline and embedded as a C++ byte array. No font
// loading or glyph rasterization occurs at runtime.
//
// Current embedded font:
//   IBM Plex Sans - UI labels, panels, and detection overlays

#pragma once

#include <Oa/Core/Types.h>
#include <Oa/Core/Status.h>
#include <Oa/Ui/Canvas.h>   // VlmVec2
#include <Oa/Runtime/Allocator.h>
#include <Oa/Runtime/Sync.h>

class OaEngine;


// ─── OaGlyphInfo ──────────────────────────────────────────────────────────────

struct OaGlyphInfo {
	OaU32  Codepoint  = 0;
	OaF32  AtlasX     = 0.0F;  // pixel rectangle in the atlas
	OaF32  AtlasY     = 0.0F;
	OaF32  AtlasW     = 0.0F;
	OaF32  AtlasH     = 0.0F;
	OaF32  BearingX   = 0.0F;  // layout metrics
	OaF32  BearingY   = 0.0F;
	OaF32  Advance    = 0.0F;
	OaF32  InkW       = 0.0F;  // visible glyph bounds inside the SDF cell
	OaF32  InkH       = 0.0F;
};


// ─── OaFontId ─────────────────────────────────────────────────────────────────

enum class OaFontId : OaU8 {
	Sans  = 0,   // IBM Plex Sans
	Mono  = 1,   // reserved for a future embedded monospace atlas
};


// ─── OaTextAtlas ──────────────────────────────────────────────────────────────
// Manages the device-side MSDF atlas texture and the glyph metadata table.

class OaTextAtlas {
public:
	OaTextAtlas() = default;
	OaTextAtlas(const OaTextAtlas&)            = delete;
	OaTextAtlas& operator=(const OaTextAtlas&) = delete;
	OaTextAtlas(OaTextAtlas&&) noexcept;
	OaTextAtlas& operator=(OaTextAtlas&&) noexcept;
	~OaTextAtlas();

	[[nodiscard]] OaStatus Init(OaEngine& InRt);
	void Destroy();

	// Glyph lookup — returns nullptr for unmapped codepoints.
	[[nodiscard]] const OaGlyphInfo* FindGlyph(OaFontId InFont, OaU32 InCodepoint) const noexcept;

	[[nodiscard]] OaU32  AtlasBindlessIndex(OaFontId InFont) const noexcept;
	[[nodiscard]] OaF32  PxRange()          const noexcept { return PxRange_; }
	[[nodiscard]] OaF32  AtlasWidth()       const noexcept { return AtlasW_; }
	[[nodiscard]] OaF32  AtlasHeight()      const noexcept { return AtlasH_; }
	[[nodiscard]] OaF32  BaseFontSize()     const noexcept { return 48.0F; }

private:
	void*  Impl_    = nullptr;  // opaque FontSlot array (avoids incomplete type in OaVec)
	OaF32  PxRange_ = 4.0F;
	OaF32  AtlasW_  = 0.0F;
	OaF32  AtlasH_  = 0.0F;
};

// One source-anchored glyph quad. Anchor is normalized to the source image;
// offsets and dimensions are compose-target pixels so labels remain readable
// while the image camera pans and zooms.
struct OaGlyphInstance {
	OaF32 AnchorX = 0.0F;
	OaF32 AnchorY = 0.0F;
	OaF32 OffsetX = 0.0F;
	OaF32 OffsetY = 0.0F;
	OaF32 Width = 0.0F;
	OaF32 Height = 0.0F;
	OaU32 AtlasX = 0;
	OaU32 AtlasY = 0;
	OaU32 AtlasW = 0;
	OaU32 AtlasH = 0;
	OaU32 Color = 0xF5F5F5FFU;
	OaU32 Reserved0 = 0;
};
static_assert(sizeof(OaGlyphInstance) == 48);

class OaGlyphBuffer {
public:
	OaGlyphBuffer() = default;
	OaGlyphBuffer(const OaGlyphBuffer&) = delete;
	OaGlyphBuffer& operator=(const OaGlyphBuffer&) = delete;
	OaGlyphBuffer(OaGlyphBuffer&& InOther) noexcept;
	OaGlyphBuffer& operator=(OaGlyphBuffer&& InOther) noexcept;
	~OaGlyphBuffer();

	[[nodiscard]] static OaResult<OaGlyphBuffer> CreateHostUpload(
		OaEngine& InRuntime,
		OaU32 InCapacity);

	void Destroy();
	[[nodiscard]] OaStatus Upload(OaSpan<const OaGlyphInstance> InGlyphs);
	void MarkConsumed(const OaVkTimelineSemaphore& InSemaphore, OaU64 InValue);

	[[nodiscard]] bool IsReady() const;
	[[nodiscard]] bool IsValid() const noexcept { return Buffer_.Buffer != nullptr; }
	[[nodiscard]] OaU32 Count() const noexcept { return Count_; }
	[[nodiscard]] OaU32 Capacity() const noexcept { return Capacity_; }
	[[nodiscard]] OaU32 BindlessIndex() const noexcept { return Buffer_.BindlessIndex; }

private:
	OaEngine* Runtime_ = nullptr;
	OaVkBuffer Buffer_;
	OaVkTimelineSemaphore ConsumerSemaphore_;
	OaU64 ConsumerValue_ = 0;
	OaU32 Count_ = 0;
	OaU32 Capacity_ = 0;
};


// ─── OaTextLayout ─────────────────────────────────────────────────────────────
// CPU-side text shaper: turns a UTF-8 string into a flat array of positioned
// glyph records ready for upload to the text_sdf.slang shader.

struct OaPositionedGlyph {
	OaF32  X          = 0.0F;
	OaF32  Y          = 0.0F;
	OaU32  Codepoint  = 0;
	OaU32  Color      = 0xF5F5F5FFU;  // packed RGBA8 — default: text.primary
	OaFontId Font = OaFontId::Sans;
};

struct OaTextLayoutConfig {
	OaFontId Font       = OaFontId::Sans;
	OaF32    Size       = 14.0F;
	OaF32    WrapWidth  = 0.0F;   // 0 = no wrap
	bool     Monospace  = false;
};

class OaTextLayout {
public:
	// Shape InText at InOrigin.  Appends to InOutGlyphs (does not clear).
	void Shape(
		const OaTextAtlas&        InAtlas,
		OaStringView              InText,
		VlmVec2                    InOrigin,
		const OaTextLayoutConfig& InCfg,
		OaU32                     InColor,
		OaVec<OaPositionedGlyph>& InOutGlyphs) const;

	// Measure text extent without appending.
	[[nodiscard]] VlmVec2 Measure(
		const OaTextAtlas&        InAtlas,
		OaStringView              InText,
		const OaTextLayoutConfig& InCfg) const;
};
