#pragma once

#include <oa/core/types.h>
#include <oa/core/status.h>
#include <oa/runtime/allocator.h>
#include <oa/runtime/engine.h>
#include <vkl/vkl.h>
#include <oa/runtime/sync.h>

namespace oavk {

class Stream;

enum class DescriptorKind : oa::U32 {
	StorageBuffer,
	SampledImage,
	StorageImage,
	Sampler,
};

struct ImageDispatchBinding {
	DescriptorKind kind = DescriptorKind::StorageBuffer;
	oa::U32 binding = 0;
	oavk::Buffer buffer = {};
	VkImageView imageView = VK_NULL_HANDLE;
	VkSampler sampler = VK_NULL_HANDLE;
	VkImageLayout imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

	// Optional image transition owned by this dispatch. ImageLayout is the
	// layout used while the shader executes; initialLayout -> ImageLayout is
	// emitted before dispatch and ImageLayout -> finalLayout afterwards.
	// Leave Image null for descriptor-only bindings or when the caller owns
	// synchronization externally. Only one binding should transition a given
	// image even when several plane views are bound.
	VkImage image = VK_NULL_HANDLE;
	VkImageLayout initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
	VkImageLayout finalLayout = VK_IMAGE_LAYOUT_UNDEFINED;
	VkImageAspectFlags aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
	oa::U32 baseMipLevel = 0U;
	oa::U32 levelCount = 1U;
	oa::U32 baseArrayLayer = 0U;
	oa::U32 layerCount = 1U;
	// Optional external queue-family ownership transfer. The compute family is
	// implied as the local owner between the pre/post barriers.
	oa::U32 initialQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
	oa::U32 finalQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
};

// Exact pipeline bridge for shaders that retain OA's global bindless set 0 but
// require one additional fixed-layout descriptor set. Vulkan sampler YCbCr
// conversion is the first consumer: its combined image sampler must be
// immutable and therefore cannot live in the global update-after-bind heap.
// All handles are borrowed and must remain alive through the returned ticket.
struct ImageDispatchPipeline {
	VkPipeline pipeline = VK_NULL_HANDLE;
	VkPipelineLayout pipelineLayout = VK_NULL_HANDLE;
	VkDescriptorSet auxiliaryDescriptorSet = VK_NULL_HANDLE;
};

class ImageDispatchTicket {
public:
	ImageDispatchTicket() = default;
	ImageDispatchTicket(const ImageDispatchTicket&) = delete;
	ImageDispatchTicket& operator=(const ImageDispatchTicket&) = delete;
	ImageDispatchTicket(ImageDispatchTicket&& inOther) noexcept;
	ImageDispatchTicket& operator=(ImageDispatchTicket&& inOther) noexcept;
	~ImageDispatchTicket();

	// wait for the dispatch timeline value without releasing the stream or its
	// semaphore. Use when another queued job still references this ticket.
	[[nodiscard]] oa::Status waitForSignal(oa::U64 inTimeoutNs = UINT64_MAX) const;
	[[nodiscard]] oa::Status wait(oa::U64 inTimeoutNs = UINT64_MAX);
	[[nodiscard]] oa::Bool isComplete() const;
	[[nodiscard]] bool isValid() const noexcept { return stream_ != nullptr; }
	[[nodiscard]] const oavk::TimelineSemaphore& semaphore() const;
	[[nodiscard]] oa::U64 value() const;
	[[nodiscard]] oa::Event completion() const;

	// Transfer ownership of a temporary view referenced by this dispatch. The
	// view is destroyed only after the ticket's timeline value completes. This
	// keeps callers asynchronous without leaking vulkan object lifetime into a
	// host wait at submission time.
	void adoptImageView(VkImageView inView);

private:
	friend class ImageDispatch;
	void cleanup_();
	void retire_();

	oa::Engine* engine_ = nullptr;
	oavk::Stream* stream_ = nullptr;
	oa::Vector<oa::U32> storageImageSlots_;
	oa::Vector<oa::U32> sampledImageSlots_;
	oa::Vector<oa::U32> samplerSlots_;
	oa::Vector<VkImageView> ownedImageViews_;
};

class ImageDispatch {
public:
	// inStorageDtype describes dtype-following storage-buffer bindings and
	// outputs. Image-only conversions use Float32 unless their shader writes a
	// typed tensor buffer.
	[[nodiscard]] static oa::Status run(
		oa::Engine& inRt,
		oa::StringView inShaderName,
		oa::Span<const ImageDispatchBinding> inBindings,
		const void* inPushData,
		oa::U32 inPushSize,
		oa::ScalarType inStorageDtype,
		oa::U32 inGroupsX,
		oa::U32 inGroupsY = 1,
		oa::U32 inGroupsZ = 1);

	// Record and submit without a host wait. The returned ticket owns the
	// stream and temporary bindless slots until wait(). Destroying an
	// unfinished ticket transfers those resources to engine retirement; the
	// destructor never waits for the GPU.
	[[nodiscard]] static oa::Result<ImageDispatchTicket> runAsync(
		oa::Engine& inRt,
		oa::StringView inShaderName,
		oa::Span<const ImageDispatchBinding> inBindings,
		const void* inPushData,
		oa::U32 inPushSize,
		oa::ScalarType inStorageDtype,
		oa::U32 inGroupsX,
		oa::U32 inGroupsY = 1,
		oa::U32 inGroupsZ = 1);

	// Same as run but the compute dispatch waits on inWaitSem reaching
	// inWaitValue before executing. Used for cross-queue async sync
	// (e.g. compute conversion waits on video decode/transition completion).
	[[nodiscard]] static oa::Status runWithDependency(
		oa::Engine& inRt,
		oa::StringView inShaderName,
		oa::Span<const ImageDispatchBinding> inBindings,
		const void* inPushData,
		oa::U32 inPushSize,
		oa::ScalarType inStorageDtype,
		oa::U32 inGroupsX,
		oa::U32 inGroupsY,
		oa::U32 inGroupsZ,
		const oavk::TimelineSemaphore& inWaitSem,
		oa::U64 inWaitValue);

	[[nodiscard]] static oa::Result<ImageDispatchTicket> runWithDependencyAsync(
		oa::Engine& inRt,
		oa::StringView inShaderName,
		oa::Span<const ImageDispatchBinding> inBindings,
		const void* inPushData,
		oa::U32 inPushSize,
		oa::ScalarType inStorageDtype,
		oa::U32 inGroupsX,
		oa::U32 inGroupsY,
		oa::U32 inGroupsZ,
		const oavk::TimelineSemaphore& inWaitSem,
		oa::U64 inWaitValue);

	[[nodiscard]] static oa::Result<ImageDispatchTicket> runWithPipelineDependencyAsync(
		oa::Engine& inRt,
		const ImageDispatchPipeline& inPipeline,
		oa::Span<const ImageDispatchBinding> inBindings,
		const void* inPushData,
		oa::U32 inPushSize,
		oa::U32 inGroupsX,
		oa::U32 inGroupsY,
		oa::U32 inGroupsZ,
		const oavk::TimelineSemaphore& inWaitSem,
		oa::U64 inWaitValue);
};

} // namespace oavk
