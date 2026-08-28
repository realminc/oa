// Ui — image representations for display via Ui::image() / Ui::imagePlanar().
//
// Two layouts:
//
//   oa::Texture     — Runtime-owned packed/image-backed GPU texture consumed by
//                   the display path. Raw handles stay private.
//
//   ImagePlanes — planar, one oavk::Buffer per channel, independent dtype per
//                   channel (U8 / U16 / F32 / BF16).  Better for GPU compute:
//                   channel-wise ops are coalesced; mixed precision is free.
//                   blitPlanar.slang reads the planes and writes to compose.
//
// Typical flows:
//   file → oa::FnImage::decodeFile() → oa::FnTexture::fromImage()
//   file → ImagePlanes::loadFile()       — same file, planar U8, compute-ready
//   Sensor depth → ImagePlanes::FromPlanes — RGB U8 + depth U16 upload

#pragma once

#include <oa/core/std/array.h>
#include <oa/core/types.h>
#include <oa/core/status.h>
#include <oa/core/matrix.h>
#include <oa/core/image.h>
#include <oa/runtime/texture.h>
#include <oa/runtime/buffer.h>
#include <oa/runtime/event.h>

namespace oa { class Engine; }


// ─── ImageDtype ─────────────────────────────────────────────────────────────
// Per-channel storage type.  Affects how blitPlanar reads and normalizes values.

namespace oa {

enum class ImageDtype : oa::U8 {
	U8   = 0,  // uint8   [0, 255]   → [0.0, 1.0]
	U16  = 1,  // uint16  [0, 65535] → [0.0, 1.0]
	F32  = 2,  // float32  passed through as-is
	BF16 = 3,  // bfloat16 decoded to float32 as-is
};


// ─── ImagePlanes ────────────────────────────────────────────────────────────
// Planar image: one device-local oavk::Buffer per channel, independent dtype.
//
// layout: each plane is a flat array of W*H elements in row-major order.
// channels:  1=gray, 2=gray+alpha, 3=RGB, 4=RGBA or RGB+depth.
// Missing alpha plane (channelCount < 4) defaults to 1.0 in the blit shader.
//
// Pass to Ui::imagePlanar().

static constexpr oa::U32 kImageMaxPlanes = 4;

class ImagePlanes {
public:
	ImagePlanes() = default;
	ImagePlanes(const ImagePlanes&) = delete;
	ImagePlanes& operator=(const ImagePlanes&) = delete;
	ImagePlanes(ImagePlanes&& inOther) noexcept;
	ImagePlanes& operator=(ImagePlanes&& inOther) noexcept;
	~ImagePlanes();

	[[nodiscard]] oa::U32 planeIndex(oa::U32 inChannel) const noexcept {
		return inChannel < channelCount_
			? planes_[inChannel].bindlessIndex
			: UINT32_MAX;
	}
	[[nodiscard]] ImageDtype planeDtype(oa::U32 inChannel) const noexcept {
		return inChannel < channelCount_
			? dtypes_[inChannel]
			: ImageDtype::U8;
	}
	[[nodiscard]] oa::I32 width() const noexcept { return width_; }
	[[nodiscard]] oa::I32 height() const noexcept { return height_; }
	[[nodiscard]] oa::U8 channelCount() const noexcept { return channelCount_; }
	[[nodiscard]] bool isValid() const noexcept {
		if (engine_ == nullptr or channelCount_ == 0
			or width_ <= 0 or height_ <= 0) return false;
		for (oa::U32 channel = 0U; channel < channelCount_; ++channel) {
			if (planes_[channel].buffer == nullptr
				or planes_[channel].bindlessIndex == UINT32_MAX) return false;
		}
		return true;
	}

	// Decode any stb_image-supported file to planar U8.
	// 1-channel files → gray (channelCount=1).
	// 3-channel files → RGB (channelCount=3).
	// 4-channel files → RGBA (channelCount=4).
	// HDR (.hdr) → planar F32. synchronous (transfer completes on return).
	[[nodiscard]] static oa::Result<ImagePlanes> loadFile(
		oa::Engine& inRt,
		oa::StringView       inPath);

	// upload caller-provided planes.  inPlanes and inDtypes must have the same
	// length (1–4).  Each span is W*H * sizeof(dtype) bytes.  Synchronous.
	[[nodiscard]] static oa::Result<ImagePlanes> fromPlanes(
		oa::Engine&                 inRt,
		oa::Span<const oa::Span<const oa::U8>>   inPlanes,
		oa::Span<const ImageDtype>         inDtypes,
		oa::I32                              inW,
		oa::I32                              inH);

	// Records the exact UI consumer completion. Abandoning the value before
	// completion transfers all owned plane buffers to engine retirement.
	[[nodiscard]] oa::Status markConsumed(const oa::Event& inCompletion);

private:
	friend class Ui;

	void abandon_() noexcept;
	void release_() noexcept;
	static oa::Status completeRetired_(void* inPayload);
	static void releaseRetired_(void* inPayload);

	oa::Engine* engine_ = nullptr;
	oa::Array<oavk::Buffer, kImageMaxPlanes> planes_ = {};
	oa::Array<ImageDtype, kImageMaxPlanes> dtypes_ = {};
	oa::Vector<oa::Event> consumerCompletions_;
	oa::I32 width_ = 0;
	oa::I32 height_ = 0;
	oa::U8 channelCount_ = 0;
};

}  // namespace oa
