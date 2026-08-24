// oa::ExternalMemory — Linux DMA-BUF image import
//
// Linux-only. Imports compatible producer-owned DMA-BUF images. Falls back to
// Unimplemented on non-Linux or when the required extensions are unavailable.

#pragma once

#include <oa/core/types.h>
#include <oa/core/status.h>
#include <oa/runtime/oaVk.h>

namespace oa {

class Engine;

// Single-plane Linux DMA-BUF image description. offset and rowPitch come from
// the producer's plane metadata; modifier is the negotiated DRM modifier.
// The importer duplicates Fd, so ownership always remains with the producer.
struct DmaBufImageDesc {
	int fd = -1;
	oa::U32 width = 0;
	oa::U32 height = 0;
	VkFormat format = VK_FORMAT_UNDEFINED;
	oa::U64 modifier = 0;
	oa::U64 offset = 0;
	oa::U64 rowPitch = 0;
};

// RAII wrapper for a producer-owned DMA-BUF imported as a sampled VkImage.
// queue-family ownership starts at VK_QUEUE_FAMILY_FOREIGN_EXT; consumers must
// acquire/release it around GPU use before the producer reuses the buffer.
class ImportedDmaBufImage {
public:
	ImportedDmaBufImage() = default;
	ImportedDmaBufImage(ImportedDmaBufImage&& inOther) noexcept;
	ImportedDmaBufImage& operator=(ImportedDmaBufImage&& inOther) noexcept;
	ImportedDmaBufImage(const ImportedDmaBufImage&) = delete;
	ImportedDmaBufImage& operator=(const ImportedDmaBufImage&) = delete;
	~ImportedDmaBufImage();

	[[nodiscard]] static oa::Result<ImportedDmaBufImage> import(
		oa::Engine& inEngine, const DmaBufImageDesc& inDesc);

	[[nodiscard]] bool isValid() const noexcept { return image_ != VK_NULL_HANDLE; }
	[[nodiscard]] VkImage image() const noexcept { return image_; }
	[[nodiscard]] VkImageView view() const noexcept { return view_; }
	[[nodiscard]] VkFormat format() const noexcept { return format_; }
	[[nodiscard]] oa::U32 width() const noexcept { return width_; }
	[[nodiscard]] oa::U32 height() const noexcept { return height_; }

private:
	void reset_() noexcept;
	oa::Engine* engine_ = nullptr;
	VkImage image_ = VK_NULL_HANDLE;
	VkImageView view_ = VK_NULL_HANDLE;
	VkDeviceMemory memory_ = VK_NULL_HANDLE;
	VkFormat format_ = VK_FORMAT_UNDEFINED;
	oa::U32 width_ = 0;
	oa::U32 height_ = 0;
};

} // namespace oa
