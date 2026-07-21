#include <Oa/Runtime/ImageDispatch.h>
#include <Oa/Runtime/ShaderProvider.h>
#include <Oa/Runtime/Stream.h>
#include <Oa/Runtime/Pipeline.h>
#include <Oa/Runtime/Bindless.h>

#include <cstring>

OaVkImageDispatchTicket::OaVkImageDispatchTicket(OaVkImageDispatchTicket&& InOther) noexcept
	: Engine_(InOther.Engine_)
	, Stream_(InOther.Stream_)
	, StorageImageSlots_(OaStdMove(InOther.StorageImageSlots_))
	, SampledImageSlots_(OaStdMove(InOther.SampledImageSlots_))
	, SamplerSlots_(OaStdMove(InOther.SamplerSlots_))
	, OwnedImageViews_(OaStdMove(InOther.OwnedImageViews_))
{
	InOther.Engine_ = nullptr;
	InOther.Stream_ = nullptr;
}

OaVkImageDispatchTicket& OaVkImageDispatchTicket::operator=(OaVkImageDispatchTicket&& InOther) noexcept
{
	if (this != &InOther) {
		Retire_();
		Engine_ = InOther.Engine_;
		Stream_ = InOther.Stream_;
		StorageImageSlots_ = OaStdMove(InOther.StorageImageSlots_);
		SampledImageSlots_ = OaStdMove(InOther.SampledImageSlots_);
		SamplerSlots_ = OaStdMove(InOther.SamplerSlots_);
		OwnedImageViews_ = OaStdMove(InOther.OwnedImageViews_);
		InOther.Engine_ = nullptr;
		InOther.Stream_ = nullptr;
	}
	return *this;
}

OaVkImageDispatchTicket::~OaVkImageDispatchTicket()
{
	Retire_();
}

OaStatus OaVkImageDispatchTicket::WaitForSignal(OaU64 InTimeoutNs) const
{
	if (!Stream_ || !Engine_) {
		return OaStatus::Ok();
	}
	return Stream_->TimelineSem.Wait(
		Engine_->Device,
		Stream_->TimelineValue,
		InTimeoutNs);
}

OaStatus OaVkImageDispatchTicket::Wait(OaU64 InTimeoutNs)
{
	OaStatus status = WaitForSignal(InTimeoutNs);
	if (status.IsOk()) {
		if (!Stream_ || !Engine_) return status;
		Stream_->Submitted = false;
		Cleanup_();
	}
	return status;
}

OaBool OaVkImageDispatchTicket::IsComplete() const
{
	return !Stream_ || !Engine_ || Stream_->IsComplete(Engine_->Device);
}

const OaVkTimelineSemaphore& OaVkImageDispatchTicket::Semaphore() const
{
	static const OaVkTimelineSemaphore empty = {};
	return Stream_ ? Stream_->TimelineSem : empty;
}

OaU64 OaVkImageDispatchTicket::Value() const
{
	return Stream_ ? Stream_->TimelineValue : 0;
}

OaCompletionToken OaVkImageDispatchTicket::Completion() const
{
	return Stream_ && Engine_
		? OaCompletionToken(Engine_->Device, Stream_->TimelineSem, Stream_->TimelineValue)
		: OaCompletionToken();
}

void OaVkImageDispatchTicket::AdoptImageView(VkImageView InView)
{
	if (InView != VK_NULL_HANDLE) {
		OwnedImageViews_.PushBack(InView);
	}
}

void OaVkImageDispatchTicket::Cleanup_()
{
	const VkDevice device = Engine_
		? static_cast<VkDevice>(Engine_->Device.Device)
		: VK_NULL_HANDLE;
	if (device != VK_NULL_HANDLE) {
		for (VkImageView view : OwnedImageViews_) {
			if (view != VK_NULL_HANDLE) vkDestroyImageView(device, view, nullptr);
		}
	}
	OwnedImageViews_.Clear();
	for (OaU32 idx : StorageImageSlots_) { Engine_->Bindless.DeregisterStorageImage(idx); }
	for (OaU32 idx : SampledImageSlots_) { Engine_->Bindless.DeregisterSampledImage(idx); }
	for (OaU32 idx : SamplerSlots_) { Engine_->Bindless.DeregisterSampler(idx); }
	StorageImageSlots_.Clear();
	SampledImageSlots_.Clear();
	SamplerSlots_.Clear();
	Engine_->ReleaseStream(Stream_);
	Stream_ = nullptr;
	Engine_ = nullptr;
}

void OaVkImageDispatchTicket::Retire_()
{
	if (!Stream_ || !Engine_) return;
	if (Stream_->IsComplete(Engine_->Device)) {
		Stream_->Submitted = false;
		Cleanup_();
		return;
	}
	Engine_->RetireImageDispatch(
		Stream_,
		OaStdMove(StorageImageSlots_),
		OaStdMove(SampledImageSlots_),
		OaStdMove(SamplerSlots_),
		OaStdMove(OwnedImageViews_));
	Stream_ = nullptr;
	Engine_ = nullptr;
}

// Shared helper: validates, registers bindless resources, and records dispatch.
// Returns the acquired stream (caller must release) or nullptr on error.
static OaVkStream* ImageDispatchSetupAndRecord(
	OaEngine& InRt,
	OaStringView InShaderName,
	OaSpan<const OaVkImageDispatchBinding> InBindings,
	const void* InPushData,
	OaU32 InPushSize,
	OaU32 InGroupsX,
	OaU32 InGroupsY,
	OaU32 InGroupsZ,
	OaVec<OaU32>& OutStorageImageSlots,
	OaVec<OaU32>& OutSampledImageSlots,
	OaVec<OaU32>& OutSamplerSlots,
	OaStatus& OutStatus)
{
	if (InBindings.Empty()) {
		OutStatus = OaStatus::Error(OaStatusCode::InvalidArgument, "image dispatch requires at least one binding");
		return nullptr;
	}
	if (!OaVkBindlessPushFits(static_cast<OaU32>(InBindings.Size()), InPushSize)) {
		OutStatus = OaStatus::Error(OaStatusCode::InvalidArgument,
			"image dispatch bindless push exceeds OA_VK_MAX_PUSH_CONSTANT_BYTES");
		return nullptr;
	}
	if (!InRt.Bindless.DescriptorSet || !InRt.Bindless.PipelineLayout) {
		OutStatus = OaStatus::Error(OaStatusCode::FailedPrecondition, "image dispatch requires bindless heap");
		return nullptr;
	}
	for (const auto& binding : InBindings) {
		if (binding.Image != VK_NULL_HANDLE
			and binding.ImageLayout == VK_IMAGE_LAYOUT_UNDEFINED) {
			OutStatus = OaStatus::Error(OaStatusCode::InvalidArgument,
				"image dispatch transition requires a shader image layout");
			return nullptr;
		}
	}

	const OaSpvEntry* spirv = OaShaderProviderFind(OaString(InShaderName).c_str());
	if (!spirv) {
		OutStatus = OaStatus::NotFound("image dispatch shader not found");
		return nullptr;
	}

	OaPipelineSpec spec{.WgSize = 256, .NumBindings = 16, .PushConstantBytes = 128,
		.SpecConstants = {{.Id = 0, .Value = InRt.DtypeSpecConstant()}}};
	OutStatus = InRt.EnsurePipeline(
		InShaderName,
		OaSpan<const OaU8>(spirv->Data, spirv->Size),
		spec);
	if (!OutStatus.IsOk()) {
		return nullptr;
	}

	OaVec<OaU32> resourceIndices;
	resourceIndices.Reserve(InBindings.Size());

	for (const auto& binding : InBindings) {
		OaU32 idx = OA_BINDLESS_INVALID;
		switch (binding.Kind) {
			case OaVkDescriptorKind::StorageBuffer:
				idx = binding.Buffer.BindlessIndex;
				if (idx == OA_BINDLESS_INVALID) {
					OutStatus = OaStatus::Error(OaStatusCode::InvalidArgument,
						"image dispatch storage buffer is not registered in bindless heap");
					return nullptr;
				}
				break;
			case OaVkDescriptorKind::StorageImage:
				idx = InRt.Bindless.RegisterStorageImage(
					InRt.Device,
					binding.ImageView,
					binding.ImageLayout == VK_IMAGE_LAYOUT_UNDEFINED ? VK_IMAGE_LAYOUT_GENERAL : binding.ImageLayout);
				if (idx == OA_BINDLESS_INVALID) {
					OutStatus = OaStatus::Error(OaStatusCode::InvalidArgument,
						"image dispatch failed to register storage image");
					return nullptr;
				}
				OutStorageImageSlots.PushBack(idx);
				break;
			case OaVkDescriptorKind::SampledImage:
				idx = InRt.Bindless.RegisterSampledImage(
					InRt.Device,
					binding.ImageView,
					binding.ImageLayout == VK_IMAGE_LAYOUT_UNDEFINED ? VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL : binding.ImageLayout);
				if (idx == OA_BINDLESS_INVALID) {
					OutStatus = OaStatus::Error(OaStatusCode::InvalidArgument,
						"image dispatch failed to register sampled image");
					return nullptr;
				}
				OutSampledImageSlots.PushBack(idx);
				break;
			case OaVkDescriptorKind::Sampler:
				idx = InRt.Bindless.RegisterSampler(InRt.Device, binding.Sampler);
				if (idx == OA_BINDLESS_INVALID) {
					OutStatus = OaStatus::Error(OaStatusCode::InvalidArgument,
						"image dispatch failed to register sampler");
					return nullptr;
				}
				OutSamplerSlots.PushBack(idx);
				break;
			case OaVkDescriptorKind::CombinedImageSampler:
				{
					OaU32 imageIdx = InRt.Bindless.RegisterSampledImage(
						InRt.Device,
						binding.ImageView,
						binding.ImageLayout == VK_IMAGE_LAYOUT_UNDEFINED ? VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL : binding.ImageLayout);
					if (imageIdx == OA_BINDLESS_INVALID) {
						OutStatus = OaStatus::Error(OaStatusCode::InvalidArgument,
							"image dispatch failed to register combined image sampler (image)");
						return nullptr;
					}
					OutSampledImageSlots.PushBack(imageIdx);
					
					OaU32 samplerIdx = InRt.Bindless.RegisterSampler(InRt.Device, binding.Sampler);
					if (samplerIdx == OA_BINDLESS_INVALID) {
						OutStatus = OaStatus::Error(OaStatusCode::InvalidArgument,
							"image dispatch failed to register combined image sampler (sampler)");
						return nullptr;
					}
					OutSamplerSlots.PushBack(samplerIdx);
					
					resourceIndices.PushBack(imageIdx);
					idx = samplerIdx;
				}
				break;
		}
		resourceIndices.PushBack(idx);
	}

	OaVkStream* stream = InRt.AcquireStream();
	if (!stream) {
		OutStatus = OaStatus::Error(OaStatusCode::VulkanError, "image dispatch: failed to acquire stream");
		return nullptr;
	}

	OutStatus = stream->Begin(InRt.Device);
	if (!OutStatus.IsOk()) {
		InRt.ReleaseStream(stream);
		return nullptr;
	}

	OaComputePipeline& pipeline = InRt.GetPipeline(InShaderName);
	if (!pipeline.Pipeline) {
		OutStatus = OaStatus::Error(OaStatusCode::PipelineError, "image dispatch: bindless pipeline not found");
		InRt.ReleaseStream(stream);
		return nullptr;
	}

	VkCommandBuffer cmd = static_cast<VkCommandBuffer>(stream->CommandBuffer);

	// Optional image transitions happen in the same command buffer as the
	// dispatch. This is required when a storage image crosses between compute
	// and Vulkan Video layouts; a descriptor layout alone is not a barrier.
	OaVec<VkImageMemoryBarrier2> preBarriers;
	for (const auto& binding : InBindings) {
		if (binding.Image == VK_NULL_HANDLE) continue;
		const bool externalAcquire =
			binding.InitialQueueFamilyIndex != VK_QUEUE_FAMILY_IGNORED;
		VkImageMemoryBarrier2 barrier = {};
		barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2;
		barrier.srcStageMask = externalAcquire or binding.InitialLayout == VK_IMAGE_LAYOUT_UNDEFINED
			? VK_PIPELINE_STAGE_2_NONE : VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT;
		barrier.srcAccessMask = externalAcquire or binding.InitialLayout == VK_IMAGE_LAYOUT_UNDEFINED
			? VK_ACCESS_2_NONE : VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT;
		barrier.dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
		barrier.dstAccessMask = VK_ACCESS_2_SHADER_READ_BIT | VK_ACCESS_2_SHADER_WRITE_BIT;
		barrier.oldLayout = binding.InitialLayout;
		barrier.newLayout = binding.ImageLayout;
		barrier.srcQueueFamilyIndex = binding.InitialQueueFamilyIndex;
		barrier.dstQueueFamilyIndex = binding.InitialQueueFamilyIndex == VK_QUEUE_FAMILY_IGNORED
			? VK_QUEUE_FAMILY_IGNORED : InRt.Device.Queues.ComputeQueueFamily;
		barrier.image = binding.Image;
		barrier.subresourceRange.aspectMask = binding.AspectMask;
		barrier.subresourceRange.baseMipLevel = binding.BaseMipLevel;
		barrier.subresourceRange.levelCount = binding.LevelCount;
		barrier.subresourceRange.baseArrayLayer = binding.BaseArrayLayer;
		barrier.subresourceRange.layerCount = binding.LayerCount;
		preBarriers.PushBack(barrier);
	}
	if (not preBarriers.Empty()) {
		VkDependencyInfo dependency = {};
		dependency.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
		dependency.imageMemoryBarrierCount = static_cast<OaU32>(preBarriers.Size());
		dependency.pImageMemoryBarriers = preBarriers.Data();
		vkCmdPipelineBarrier2(cmd, &dependency);
	}

	vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, static_cast<VkPipeline>(pipeline.Pipeline));
	VkDescriptorSet descriptorSet = static_cast<VkDescriptorSet>(InRt.Bindless.DescriptorSet);
	vkCmdBindDescriptorSets(
		cmd,
		VK_PIPELINE_BIND_POINT_COMPUTE,
		static_cast<VkPipelineLayout>(pipeline.PipelineLayout),
		0,
		1,
		&descriptorSet,
		0,
		nullptr);

	const OaU32 headerBytes = static_cast<OaU32>(resourceIndices.Size()) * sizeof(OaU32);
	const OaU32 totalPush = headerBytes + InPushSize;
	alignas(16) OaU8 pushBuf[OA_VK_MAX_PUSH_CONSTANT_BYTES] = {};
	if (!resourceIndices.Empty()) {
		std::memcpy(pushBuf, resourceIndices.Data(), headerBytes);
	}
	if (InPushData && InPushSize > 0) {
		std::memcpy(pushBuf + headerBytes, InPushData, InPushSize);
	}
	vkCmdPushConstants(
		cmd,
		static_cast<VkPipelineLayout>(pipeline.PipelineLayout),
		VK_SHADER_STAGE_COMPUTE_BIT,
		0,
		totalPush,
		pushBuf);
	vkCmdDispatch(cmd, InGroupsX, InGroupsY, InGroupsZ);

	OaVec<VkImageMemoryBarrier2> postBarriers;
	for (const auto& binding : InBindings) {
		if (binding.Image == VK_NULL_HANDLE or binding.FinalLayout == VK_IMAGE_LAYOUT_UNDEFINED
			or binding.FinalLayout == binding.ImageLayout) continue;
		VkImageMemoryBarrier2 barrier = {};
		const bool externalRelease =
			binding.FinalQueueFamilyIndex != VK_QUEUE_FAMILY_IGNORED;
		barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2;
		barrier.srcStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
		barrier.srcAccessMask = binding.Kind == OaVkDescriptorKind::SampledImage
			? VK_ACCESS_2_SHADER_READ_BIT
			: VK_ACCESS_2_SHADER_READ_BIT | VK_ACCESS_2_SHADER_WRITE_BIT;
		// Keep this transition valid on a compute-only queue. A cross-queue
		// consumer supplies its actual destination stage/access through the
		// timeline wait and a same-layout barrier on that queue.
		barrier.dstStageMask = externalRelease
			? VK_PIPELINE_STAGE_2_NONE : VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT;
		barrier.dstAccessMask = externalRelease
			? VK_ACCESS_2_NONE
			: binding.FinalLayout == VK_IMAGE_LAYOUT_VIDEO_ENCODE_SRC_KHR
			? VK_ACCESS_2_MEMORY_READ_BIT
			: VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT;
		barrier.oldLayout = binding.ImageLayout;
		barrier.newLayout = binding.FinalLayout;
		barrier.srcQueueFamilyIndex = binding.FinalQueueFamilyIndex == VK_QUEUE_FAMILY_IGNORED
			? VK_QUEUE_FAMILY_IGNORED : InRt.Device.Queues.ComputeQueueFamily;
		barrier.dstQueueFamilyIndex = binding.FinalQueueFamilyIndex;
		barrier.image = binding.Image;
		barrier.subresourceRange.aspectMask = binding.AspectMask;
		barrier.subresourceRange.baseMipLevel = binding.BaseMipLevel;
		barrier.subresourceRange.levelCount = binding.LevelCount;
		barrier.subresourceRange.baseArrayLayer = binding.BaseArrayLayer;
		barrier.subresourceRange.layerCount = binding.LayerCount;
		postBarriers.PushBack(barrier);
	}
	if (not postBarriers.Empty()) {
		VkDependencyInfo dependency = {};
		dependency.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
		dependency.imageMemoryBarrierCount = static_cast<OaU32>(postBarriers.Size());
		dependency.pImageMemoryBarriers = postBarriers.Data();
		vkCmdPipelineBarrier2(cmd, &dependency);
	}

	VkMemoryBarrier barrier = {};
	barrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
	barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
	barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
	vkCmdPipelineBarrier(cmd,
		VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
		VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT,
		0, 1, &barrier, 0, nullptr, 0, nullptr);

	OutStatus = OaStatus::Ok();
	return stream;
}

OaStatus OaVkImageDispatch::Run(
	OaEngine& InRt,
	OaStringView InShaderName,
	OaSpan<const OaVkImageDispatchBinding> InBindings,
	const void* InPushData,
	OaU32 InPushSize,
	OaU32 InGroupsX,
	OaU32 InGroupsY,
	OaU32 InGroupsZ)
{
	OaVec<OaU32> storageImageSlots;
	OaVec<OaU32> sampledImageSlots;
	OaVec<OaU32> samplerSlots;
	OaStatus status;
	OaVkStream* stream = ImageDispatchSetupAndRecord(
		InRt, InShaderName, InBindings, InPushData, InPushSize,
		InGroupsX, InGroupsY, InGroupsZ,
		storageImageSlots, sampledImageSlots, samplerSlots, status);
	if (!stream) {
		return status;
	}
	status = stream->SubmitAndWait(InRt);
	InRt.ReleaseStream(stream);
	for (OaU32 idx : storageImageSlots) { InRt.Bindless.DeregisterStorageImage(idx); }
	for (OaU32 idx : sampledImageSlots) { InRt.Bindless.DeregisterSampledImage(idx); }
	for (OaU32 idx : samplerSlots) { InRt.Bindless.DeregisterSampler(idx); }
	return status;
}

OaResult<OaVkImageDispatchTicket> OaVkImageDispatch::RunAsync(
	OaEngine& InRt,
	OaStringView InShaderName,
	OaSpan<const OaVkImageDispatchBinding> InBindings,
	const void* InPushData,
	OaU32 InPushSize,
	OaU32 InGroupsX,
	OaU32 InGroupsY,
	OaU32 InGroupsZ)
{
	const OaVkTimelineSemaphore noDependency = {};
	return RunWithDependencyAsync(
		InRt, InShaderName, InBindings, InPushData, InPushSize,
		InGroupsX, InGroupsY, InGroupsZ, noDependency, 0U);
}

OaStatus OaVkImageDispatch::RunWithDependency(
	OaEngine& InRt,
	OaStringView InShaderName,
	OaSpan<const OaVkImageDispatchBinding> InBindings,
	const void* InPushData,
	OaU32 InPushSize,
	OaU32 InGroupsX,
	OaU32 InGroupsY,
	OaU32 InGroupsZ,
	const OaVkTimelineSemaphore& InWaitSem,
	OaU64 InWaitValue)
{
	OaVec<OaU32> storageImageSlots;
	OaVec<OaU32> sampledImageSlots;
	OaVec<OaU32> samplerSlots;
	OaStatus status;
	OaVkStream* stream = ImageDispatchSetupAndRecord(
		InRt, InShaderName, InBindings, InPushData, InPushSize,
		InGroupsX, InGroupsY, InGroupsZ,
		storageImageSlots, sampledImageSlots, samplerSlots, status);
	if (!stream) {
		return status;
	}
	status = stream->SubmitWithDependency(InRt, InWaitSem, InWaitValue);
	if (status.IsOk()) {
		status = stream->Synchronize(InRt.Device);
	}
	InRt.ReleaseStream(stream);
	for (OaU32 idx : storageImageSlots) { InRt.Bindless.DeregisterStorageImage(idx); }
	for (OaU32 idx : sampledImageSlots) { InRt.Bindless.DeregisterSampledImage(idx); }
	for (OaU32 idx : samplerSlots) { InRt.Bindless.DeregisterSampler(idx); }
	return status;
}

OaResult<OaVkImageDispatchTicket> OaVkImageDispatch::RunWithDependencyAsync(
	OaEngine& InRt,
	OaStringView InShaderName,
	OaSpan<const OaVkImageDispatchBinding> InBindings,
	const void* InPushData,
	OaU32 InPushSize,
	OaU32 InGroupsX,
	OaU32 InGroupsY,
	OaU32 InGroupsZ,
	const OaVkTimelineSemaphore& InWaitSem,
	OaU64 InWaitValue)
{
	OaVkImageDispatchTicket ticket;
	OaStatus status;
	OaVkStream* stream = ImageDispatchSetupAndRecord(
		InRt, InShaderName, InBindings, InPushData, InPushSize,
		InGroupsX, InGroupsY, InGroupsZ,
		ticket.StorageImageSlots_,
		ticket.SampledImageSlots_,
		ticket.SamplerSlots_,
		status);
	if (!stream) {
		return status;
	}
	status = stream->SubmitWithDependency(InRt, InWaitSem, InWaitValue);
	if (!status.IsOk()) {
		InRt.ReleaseStream(stream);
		for (OaU32 idx : ticket.StorageImageSlots_) { InRt.Bindless.DeregisterStorageImage(idx); }
		for (OaU32 idx : ticket.SampledImageSlots_) { InRt.Bindless.DeregisterSampledImage(idx); }
		for (OaU32 idx : ticket.SamplerSlots_) { InRt.Bindless.DeregisterSampler(idx); }
		return status;
	}
	ticket.Engine_ = &InRt;
	ticket.Stream_ = stream;
	return ticket;
}
