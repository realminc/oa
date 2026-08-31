#pragma once

#include <oa/runtime/engine.h>
#include <oa/runtime/allocator.h>
#include <oa/runtime/texture.h>
#include <oa/core/std/typeTraits.h>

// Sole private bridge for raw buffer/image representation and engine
// provenance of the public semantic texture value.
class oa::TextureAccess {
private:
	template<typename T>
	[[nodiscard]] static oa::U64 encodeHandle_(T inHandle) noexcept {
		if constexpr (oa::isPointerV<T>) {
			return static_cast<oa::U64>(
				reinterpret_cast<oa::Usize>(inHandle));
		} else {
			return static_cast<oa::U64>(inHandle);
		}
	}

	template<typename T>
	[[nodiscard]] static T decodeHandle_(oa::U64 inHandle) noexcept {
		if constexpr (oa::isPointerV<T>) {
			return reinterpret_cast<T>(
				static_cast<oa::Usize>(inHandle));
		} else {
			return static_cast<T>(inHandle);
		}
	}

public:
	[[nodiscard]] static oa::Result<oavk::Buffer> uploadBuffer(
		oa::Engine& inEngine,
		const void* inData,
		oa::U64 inBytes);

	[[nodiscard]] static oa::Texture fromBuffer(
		oa::Engine& inEngine,
		oa::SharedPtr<oavk::Buffer> inOwner,
		oa::I32 inWidth,
		oa::I32 inHeight) noexcept
	{
		oa::Texture texture;
		texture.engine_ = &inEngine;
		texture.bufferOwner_ = oa::move(inOwner);
		texture.width_ = inWidth;
		texture.height_ = inHeight;
		return texture;
	}

	[[nodiscard]] static oa::Texture fromBorrowedImage(
		oa::Engine& inEngine,
		VkImage inImage,
		VkImageView inView,
		VkFormat inFormat,
		VkImageLayout inLayout,
		oa::I32 inWidth,
		oa::I32 inHeight) noexcept
	{
		oa::Texture texture;
		texture.engine_ = &inEngine;
		texture.image_ = encodeHandle_(inImage);
		texture.view_ = encodeHandle_(inView);
		texture.format_ = static_cast<oa::I32>(inFormat);
		texture.layout_ = static_cast<oa::I32>(inLayout);
		texture.width_ = inWidth;
		texture.height_ = inHeight;
		return texture;
	}

	[[nodiscard]] static oa::Engine* engine(const oa::Texture& inTexture) noexcept {
		return inTexture.engine_;
	}
	[[nodiscard]] static const oa::SharedPtr<oavk::Buffer>& bufferOwner(
		const oa::Texture& inTexture) noexcept
	{
		return inTexture.bufferOwner_;
	}
	[[nodiscard]] static const oavk::Buffer* buffer(
		const oa::Texture& inTexture) noexcept
	{
		return inTexture.bufferOwner_ ? inTexture.bufferOwner_.get() : nullptr;
	}
	[[nodiscard]] static VkImage image(const oa::Texture& inTexture) noexcept {
		return decodeHandle_<VkImage>(inTexture.image_);
	}
	[[nodiscard]] static VkImageView view(const oa::Texture& inTexture) noexcept {
		return decodeHandle_<VkImageView>(inTexture.view_);
	}
	[[nodiscard]] static VkFormat format(const oa::Texture& inTexture) noexcept {
		return static_cast<VkFormat>(inTexture.format_);
	}
	[[nodiscard]] static VkImageLayout layout(const oa::Texture& inTexture) noexcept {
		return static_cast<VkImageLayout>(inTexture.layout_);
	}
};
