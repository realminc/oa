// OaExternalMemory — opaque-fd buffer sharing + Linux DMA-BUF image import
//
// Linux-only. Requires VK_KHR_external_memory + VK_KHR_external_memory_fd.
// Export Vulkan memory as an opaque fd, or import producer DMA-BUF images.
// Falls back to Unimplemented on non-Linux or when extensions unavailable.

#pragma once

#include <Oa/Core/Types.h>
#include <Oa/Core/Status.h>
#include <Oa/Runtime/Allocator.h>
#include <Oa/Runtime/OaVk.h>

class OaComputeEngine;
class OaVkDevice;

class OaExternalBuffer {
public:
	int Fd = -1;
	OaU64 Size = 0;
	OaU32 SourceNode = 0;

	[[nodiscard]] OaBool IsValid() const { return Fd >= 0; }
	void Close();
};

// Export a buffer's backing memory as an opaque POSIX fd. This is Vulkan-to-
// Vulkan sharing, not a Linux DMA-BUF contract.
// Uses OaVmaGetAllocationInfo to retrieve the underlying VkDeviceMemory.
// Returns Unimplemented if the device lacks VK_KHR_external_memory_fd.
[[nodiscard]] OaResult<OaExternalBuffer> OaExportBufferFd(
	const OaVkDevice& InDevice,
	OaVma& InAllocator,
	const OaVkBuffer& InBuf,
	OaU32 InSourceNode
);

// Import a POSIX fd as a new VkBuffer on InDevice.
// The fd is consumed (closed by Vulkan on successful import).
// The returned buffer has OA_VK_BUFFER_FLAG_IMPORTED set —
// caller must use OaVma::FreeImported(), NOT OaVma::Free().
[[nodiscard]] OaResult<OaVkBuffer> OaImportBufferFd(
	const OaVkDevice& InDevice,
	OaVma& InAllocator,
	const OaExternalBuffer& InExt
);

// Check if a pair of devices can use opaque-fd zero-copy transfer.
// Both must support VK_KHR_external_memory_fd and be the same vendor.
[[nodiscard]] OaBool OaCanUseDmaBuf(
	const OaVkDevice& InSrc,
	const OaVkDevice& InDst
);

// Single-plane Linux DMA-BUF image description. Offset and RowPitch come from
// the producer's plane metadata; Modifier is the negotiated DRM modifier.
// The importer duplicates Fd, so ownership always remains with the producer.
struct OaDmaBufImageDesc {
	int Fd = -1;
	OaU32 Width = 0;
	OaU32 Height = 0;
	VkFormat Format = VK_FORMAT_UNDEFINED;
	OaU64 Modifier = 0;
	OaU64 Offset = 0;
	OaU64 RowPitch = 0;
};

// RAII wrapper for a producer-owned DMA-BUF imported as a sampled VkImage.
// Queue-family ownership starts at VK_QUEUE_FAMILY_FOREIGN_EXT; consumers must
// acquire/release it around GPU use before the producer reuses the buffer.
class OaImportedDmaBufImage {
public:
	OaImportedDmaBufImage() = default;
	OaImportedDmaBufImage(OaImportedDmaBufImage&& InOther) noexcept;
	OaImportedDmaBufImage& operator=(OaImportedDmaBufImage&& InOther) noexcept;
	OaImportedDmaBufImage(const OaImportedDmaBufImage&) = delete;
	OaImportedDmaBufImage& operator=(const OaImportedDmaBufImage&) = delete;
	~OaImportedDmaBufImage();

	[[nodiscard]] static OaResult<OaImportedDmaBufImage> Import(
		OaComputeEngine& InEngine, const OaDmaBufImageDesc& InDesc);
	void Destroy();

	[[nodiscard]] bool IsValid() const noexcept { return Image_ != VK_NULL_HANDLE; }
	[[nodiscard]] VkImage Image() const noexcept { return Image_; }
	[[nodiscard]] VkImageView View() const noexcept { return View_; }
	[[nodiscard]] VkFormat Format() const noexcept { return Format_; }
	[[nodiscard]] OaU32 Width() const noexcept { return Width_; }
	[[nodiscard]] OaU32 Height() const noexcept { return Height_; }

private:
	OaComputeEngine* Engine_ = nullptr;
	VkImage Image_ = VK_NULL_HANDLE;
	VkImageView View_ = VK_NULL_HANDLE;
	VkDeviceMemory Memory_ = VK_NULL_HANDLE;
	VkFormat Format_ = VK_FORMAT_UNDEFINED;
	OaU32 Width_ = 0;
	OaU32 Height_ = 0;
};
