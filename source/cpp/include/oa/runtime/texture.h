// oa::Texture — semantic GPU texture shared by Render, Vision, and Ui.
//
// A texture may be buffer-backed for compute/display interop or image-backed
// for rasterization and presentation. Runtime owns this device-resource value
// contract; producers and sinks consume it without creating domain-module
// cycles or becoming owners of each other's sessions.

#pragma once

#include <oa/core/image.h>
#include <oa/core/status.h>
#include <oa/core/types.h>

namespace oavk { class Buffer; }

namespace oa {

class Engine;
class TextureAccess;

class Texture {
public:
	Texture() = default;
	Texture(const Texture&) = default;
	Texture& operator=(const Texture&) = default;
	Texture(Texture&&) noexcept = default;
	Texture& operator=(Texture&&) noexcept = default;
	~Texture() = default;

	[[nodiscard]] oa::U32 bindlessIndex() const noexcept;
	[[nodiscard]] bool isImageBacked() const noexcept;
	[[nodiscard]] bool isValid() const noexcept;
	[[nodiscard]] oa::I32 width() const noexcept { return width_; }
	[[nodiscard]] oa::I32 height() const noexcept { return height_; }

private:
	friend class TextureAccess;

	oa::Engine* engine_ = nullptr;
	oa::SharedPtr<oavk::Buffer> bufferOwner_;
	// Image-backed textures are non-owning views. Their producing session owns
	// the image and requires an exact consumer-completion handoff.
	oa::U64 image_ = 0;        // Opaque VkImage bits
	oa::U64 view_ = 0;         // Opaque VkImageView bits
	oa::I32 format_ = 0;       // VkFormat
	oa::I32 layout_ = 0;       // VkImageLayout
	oa::I32 width_ = 0;
	oa::I32 height_ = 0;
};

struct ClearColor {
	oa::F32 r = 0.0F;
	oa::F32 g = 0.0F;
	oa::F32 b = 0.0F;
	oa::F32 a = 1.0F;
};

struct Rect2D {
	oa::I32 x = 0;
	oa::I32 y = 0;
	oa::I32 w = 0;
	oa::I32 h = 0;

	[[nodiscard]] bool isEmpty() const noexcept {
		return w <= 0 or h <= 0;
	}
};

struct BlitDesc {
	const Texture* src = nullptr;
	const Texture* dst = nullptr;
	oa::Filter filter = oa::Filter::Linear;
	Rect2D srcRect = {};
	Rect2D dstRect = {};
};

namespace FnTexture {

// Synchronously upload one exact packed-RGBA8 host image.
[[nodiscard]] oa::Result<Texture> fromPixels(
	oa::Engine& inEngine,
	oa::Span<const oa::U8> inRgba,
	oa::I32 inWidth,
	oa::I32 inHeight);

// Record conversion from a validated single semantic image to packed RGBA8.
// Every supported oa::Image layout and channel format is interpreted explicitly.
[[nodiscard]] oa::Result<Texture> fromImage(
	oa::Engine& inEngine,
	const oa::Image& inImage);

// Synchronous semantic readback for buffer-backed RGBA8 textures. Any work
// recorded on the texture's engine completes before bytes are copied.
[[nodiscard]] oa::Status copyToHost(
	oa::Engine& inEngine,
	const Texture& inTexture,
	void* outHost,
	oa::U64 inBytes);

[[nodiscard]] oa::Status blit(const BlitDesc& inDesc);

[[nodiscard]] oa::Status clear(
	const Texture& inTarget,
	ClearColor inColor);

} // namespace FnTexture

} // namespace oa
