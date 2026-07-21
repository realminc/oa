#pragma once

#include <Oa/Core/Types.h>
#include <Oa/Core/Status.h>
#include <Oa/Runtime/Allocator.h>
#include <Oa/Runtime/Engine.h>
#include <Oa/Runtime/OaVk.h>

class OaVkStream;

enum class OaVkDescriptorKind : OaU32 {
	StorageBuffer,
	SampledImage,
	StorageImage,
	Sampler,
	CombinedImageSampler,
};

struct OaVkImageDispatchBinding {
	OaVkDescriptorKind Kind = OaVkDescriptorKind::StorageBuffer;
	OaU32 Binding = 0;
	OaVkBuffer Buffer = {};
	VkImageView ImageView = VK_NULL_HANDLE;
	VkSampler Sampler = VK_NULL_HANDLE;
	VkImageLayout ImageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

	// Optional image transition owned by this dispatch. ImageLayout is the
	// layout used while the shader executes; InitialLayout -> ImageLayout is
	// emitted before dispatch and ImageLayout -> FinalLayout afterwards.
	// Leave Image null for descriptor-only bindings or when the caller owns
	// synchronization externally. Only one binding should transition a given
	// image even when several plane views are bound.
	VkImage Image = VK_NULL_HANDLE;
	VkImageLayout InitialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
	VkImageLayout FinalLayout = VK_IMAGE_LAYOUT_UNDEFINED;
	VkImageAspectFlags AspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
	OaU32 BaseMipLevel = 0U;
	OaU32 LevelCount = 1U;
	OaU32 BaseArrayLayer = 0U;
	OaU32 LayerCount = 1U;
	// Optional external queue-family ownership transfer. The compute family is
	// implied as the local owner between the pre/post barriers.
	OaU32 InitialQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
	OaU32 FinalQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
};

class OaVkImageDispatchTicket {
public:
	OaVkImageDispatchTicket() = default;
	OaVkImageDispatchTicket(const OaVkImageDispatchTicket&) = delete;
	OaVkImageDispatchTicket& operator=(const OaVkImageDispatchTicket&) = delete;
	OaVkImageDispatchTicket(OaVkImageDispatchTicket&& InOther) noexcept;
	OaVkImageDispatchTicket& operator=(OaVkImageDispatchTicket&& InOther) noexcept;
	~OaVkImageDispatchTicket();

	// Wait for the dispatch timeline value without releasing the stream or its
	// semaphore. Use when another queued job still references this ticket.
	[[nodiscard]] OaStatus WaitForSignal(OaU64 InTimeoutNs = UINT64_MAX) const;
	[[nodiscard]] OaStatus Wait(OaU64 InTimeoutNs = UINT64_MAX);
	[[nodiscard]] OaBool IsComplete() const;
	[[nodiscard]] bool IsValid() const noexcept { return Stream_ != nullptr; }
	[[nodiscard]] const OaVkTimelineSemaphore& Semaphore() const;
	[[nodiscard]] OaU64 Value() const;
	[[nodiscard]] OaCompletionToken Completion() const;

	// Transfer ownership of a temporary view referenced by this dispatch. The
	// view is destroyed only after the ticket's timeline value completes. This
	// keeps callers asynchronous without leaking Vulkan object lifetime into a
	// host wait at submission time.
	void AdoptImageView(VkImageView InView);

private:
	friend class OaVkImageDispatch;
	void Cleanup_();
	void Retire_();

	OaEngine* Engine_ = nullptr;
	OaVkStream* Stream_ = nullptr;
	OaVec<OaU32> StorageImageSlots_;
	OaVec<OaU32> SampledImageSlots_;
	OaVec<OaU32> SamplerSlots_;
	OaVec<VkImageView> OwnedImageViews_;
};

class OaVkImageDispatch {
public:
	[[nodiscard]] static OaStatus Run(
		OaEngine& InRt,
		OaStringView InShaderName,
		OaSpan<const OaVkImageDispatchBinding> InBindings,
		const void* InPushData,
		OaU32 InPushSize,
		OaU32 InGroupsX,
		OaU32 InGroupsY = 1,
		OaU32 InGroupsZ = 1);

	// Record and submit without a host wait. The returned ticket owns the
	// stream and temporary bindless slots until Wait(). Destroying an
	// unfinished ticket transfers those resources to engine retirement; the
	// destructor never waits for the GPU.
	[[nodiscard]] static OaResult<OaVkImageDispatchTicket> RunAsync(
		OaEngine& InRt,
		OaStringView InShaderName,
		OaSpan<const OaVkImageDispatchBinding> InBindings,
		const void* InPushData,
		OaU32 InPushSize,
		OaU32 InGroupsX,
		OaU32 InGroupsY = 1,
		OaU32 InGroupsZ = 1);

	// Same as Run but the compute dispatch waits on InWaitSem reaching
	// InWaitValue before executing. Used for cross-queue async sync
	// (e.g. compute conversion waits on video decode/transition completion).
	[[nodiscard]] static OaStatus RunWithDependency(
		OaEngine& InRt,
		OaStringView InShaderName,
		OaSpan<const OaVkImageDispatchBinding> InBindings,
		const void* InPushData,
		OaU32 InPushSize,
		OaU32 InGroupsX,
		OaU32 InGroupsY,
		OaU32 InGroupsZ,
		const OaVkTimelineSemaphore& InWaitSem,
		OaU64 InWaitValue);

	[[nodiscard]] static OaResult<OaVkImageDispatchTicket> RunWithDependencyAsync(
		OaEngine& InRt,
		OaStringView InShaderName,
		OaSpan<const OaVkImageDispatchBinding> InBindings,
		const void* InPushData,
		OaU32 InPushSize,
		OaU32 InGroupsX,
		OaU32 InGroupsY,
		OaU32 InGroupsZ,
		const OaVkTimelineSemaphore& InWaitSem,
		OaU64 InWaitValue);
};
