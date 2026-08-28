#include <oa/runtime/imageDispatch.h>
#include <oa/runtime/eventAccess.h>
#include "engine/deviceAccess.h"
#include "engine/engineAccess.h"
#include <oa/runtime/spirv.h>
#include <oa/runtime/stream.h>
#include <oa/runtime/pipeline.h>
#include <oa/runtime/bindless.h>
#include <oa/runtime/engine/bindlessAccess.h>
#include <oa/runtime/engine/pipelineAccess.h>
#include <oa/runtime/engine/submissionAccess.h>
#include <oa/runtime/engine/deviceAccess.h>
#include "dispatchValidation.h"
#include "storageDtype.h"

#include <oa/core/std/memory.h>

oavk::ImageDispatchTicket::ImageDispatchTicket(oavk::ImageDispatchTicket&& inOther) noexcept
	: engine_(inOther.engine_)
	, stream_(inOther.stream_)
	, storageImageSlots_(oa::move(inOther.storageImageSlots_))
	, sampledImageSlots_(oa::move(inOther.sampledImageSlots_))
	, samplerSlots_(oa::move(inOther.samplerSlots_))
	, ownedImageViews_(oa::move(inOther.ownedImageViews_))
{
	inOther.engine_ = nullptr;
	inOther.stream_ = nullptr;
}

oavk::ImageDispatchTicket& oavk::ImageDispatchTicket::operator=(oavk::ImageDispatchTicket&& inOther) noexcept
{
	if (this != &inOther) {
		retire_();
		engine_ = inOther.engine_;
		stream_ = inOther.stream_;
		storageImageSlots_ = oa::move(inOther.storageImageSlots_);
		sampledImageSlots_ = oa::move(inOther.sampledImageSlots_);
		samplerSlots_ = oa::move(inOther.samplerSlots_);
		ownedImageViews_ = oa::move(inOther.ownedImageViews_);
		inOther.engine_ = nullptr;
		inOther.stream_ = nullptr;
	}
	return *this;
}

oavk::ImageDispatchTicket::~ImageDispatchTicket()
{
	retire_();
}

oa::Status oavk::ImageDispatchTicket::waitForSignal(oa::U64 inTimeoutNs) const
{
	if (!stream_ || !engine_) {
		return oa::Status::ok();
	}
	return stream_->timelineSem.wait(
		oa::EngineDeviceAccess::get(*engine_),
		stream_->timelineValue,
		inTimeoutNs);
}

oa::Status oavk::ImageDispatchTicket::wait(oa::U64 inTimeoutNs)
{
	oa::Status status = waitForSignal(inTimeoutNs);
	if (status.isOk()) {
		if (!stream_ || !engine_) return status;
		stream_->submitted = false;
		cleanup_();
	}
	return status;
}

oa::Bool oavk::ImageDispatchTicket::isComplete() const
{
	return !stream_ || !engine_ || stream_->isComplete(oa::EngineDeviceAccess::get(*engine_));
}

const oavk::TimelineSemaphore& oavk::ImageDispatchTicket::semaphore() const
{
	static const oavk::TimelineSemaphore empty = {};
	return stream_ ? stream_->timelineSem : empty;
}

oa::U64 oavk::ImageDispatchTicket::value() const
{
	return stream_ ? stream_->timelineValue : 0;
}

oa::Event oavk::ImageDispatchTicket::completion() const
{
	return stream_ && engine_
		? oa::EventAccess::create(
			oa::EngineDeviceAccess::get(*engine_),
			stream_->timelineSem,
			stream_->timelineValue,
			stream_->queueFamily)
		: oa::Event();
}

void oavk::ImageDispatchTicket::adoptImageView(VkImageView inView)
{
	if (inView != VK_NULL_HANDLE) {
		ownedImageViews_.pushBack(inView);
	}
}

void oavk::ImageDispatchTicket::cleanup_()
{
	const oavk::Device* ownerDevice = engine_
		? &oa::EngineDeviceAccess::get(*engine_)
		: nullptr;
	const VkDevice device = ownerDevice
		? static_cast<VkDevice>(ownerDevice->device)
		: VK_NULL_HANDLE;
	if (device != VK_NULL_HANDLE) {
		for (VkImageView view : ownedImageViews_) {
			if (view != VK_NULL_HANDLE) {
				ownerDevice->deviceDispatch.vkDestroyImageView(device, view, nullptr);
			}
		}
	}
	ownedImageViews_.clear();
	auto& bindless = oa::EngineBindlessAccess::get(*engine_);
	for (oa::U32 idx : storageImageSlots_) { bindless.deregisterStorageImage(idx); }
	for (oa::U32 idx : sampledImageSlots_) { bindless.deregisterSampledImage(idx); }
	for (oa::U32 idx : samplerSlots_) { bindless.deregisterSampler(idx); }
	storageImageSlots_.clear();
	sampledImageSlots_.clear();
	samplerSlots_.clear();
	oa::EngineSubmissionAccess::releaseStream(*engine_, stream_);
	stream_ = nullptr;
	engine_ = nullptr;
}

void oavk::ImageDispatchTicket::retire_()
{
	if (!stream_ || !engine_) return;
	if (stream_->isComplete(oa::EngineDeviceAccess::get(*engine_))) {
		stream_->submitted = false;
		cleanup_();
		return;
	}
	oa::RetiredImageDispatch retired;
	retired.stream = stream_;
	retired.storageImageSlots = oa::move(storageImageSlots_);
	retired.sampledImageSlots = oa::move(sampledImageSlots_);
	retired.samplerSlots = oa::move(samplerSlots_);
	retired.imageViews = oa::move(ownedImageViews_);
	oa::EngineAccess(*engine_).retireImageDispatch(oa::move(retired));
	stream_ = nullptr;
	engine_ = nullptr;
}

// Shared helper: validates, registers bindless resources, and records dispatch.
// Returns the acquired stream (caller must release) or nullptr on error.
static oavk::Stream* imageDispatchSetupAndRecord(
	oa::Engine& inRt,
	oa::StringView inShaderName,
	oa::Span<const oavk::ImageDispatchBinding> inBindings,
	const void* inPushData,
	oa::U32 inPushSize,
	oa::ScalarType inStorageDtype,
	oa::U32 inGroupsX,
	oa::U32 inGroupsY,
	oa::U32 inGroupsZ,
	oa::Vector<oa::U32>& outStorageImageSlots,
	oa::Vector<oa::U32>& outSampledImageSlots,
	oa::Vector<oa::U32>& outSamplerSlots,
	oa::Status& outStatus)
{
	outStatus = oavk::validateDirectComputeDispatch(
		oa::EngineDeviceAccess::get(inRt), inGroupsX, inGroupsY, inGroupsZ);
	if (not outStatus.isOk()) {
		return nullptr;
	}
	if (inBindings.empty()) {
		outStatus = oa::Status::error(oa::StatusCode::InvalidArgument, "image dispatch requires at least one binding");
		return nullptr;
	}
	if (!oavk::bindlessPushFits(static_cast<oa::U32>(inBindings.size()), inPushSize)) {
		outStatus = oa::Status::error(oa::StatusCode::InvalidArgument,
			"image dispatch bindless push exceeds oavk::OA_VK_MAX_PUSH_CONSTANT_BYTES");
		return nullptr;
	}
	auto& bindless = oa::EngineBindlessAccess::get(inRt);
	if (!bindless.descriptorSet || !bindless.pipelineLayout) {
		outStatus = oa::Status::error(oa::StatusCode::FailedPrecondition, "image dispatch requires bindless heap");
		return nullptr;
	}
	auto dtype = oavk::resolveStorageDtypeSpecConstant(inStorageDtype);
	if (not dtype.isOk()) {
		outStatus = dtype.getStatus();
		return nullptr;
	}
	for (const auto& binding : inBindings) {
		if (binding.image != VK_NULL_HANDLE
			and binding.imageLayout == VK_IMAGE_LAYOUT_UNDEFINED) {
			outStatus = oa::Status::error(oa::StatusCode::InvalidArgument,
				"image dispatch transition requires a shader image layout");
			return nullptr;
		}
	}

	const oavk::SpirvEntry* spirv = oavk::findSpirv(oa::String(inShaderName).cStr());
	if (!spirv) {
		outStatus = oa::Status::notFound("image dispatch shader not found");
		return nullptr;
	}

	oa::PipelineSpec spec{.numBindings = 16, .pushConstantBytes = 128,
		.specConstants = oa::Vector<oa::SpecConstant>{
			oa::SpecConstant{.id = 0, .value = dtype.getValue()}}};
	outStatus = oa::EnginePipelineAccess::ensure(
		inRt,
		inShaderName,
		oa::Span<const oa::U8>(spirv->data, spirv->size),
		spec);
	if (!outStatus.isOk()) {
		return nullptr;
	}

	oa::Vector<oa::U32> resourceIndices;
	resourceIndices.reserve(inBindings.size());

	for (const auto& binding : inBindings) {
		oa::U32 idx = OA_BINDLESS_INVALID;
		switch (binding.kind) {
			case oavk::DescriptorKind::StorageBuffer:
				idx = binding.buffer.bindlessIndex;
				if (idx == OA_BINDLESS_INVALID) {
					outStatus = oa::Status::error(oa::StatusCode::InvalidArgument,
						"image dispatch storage buffer is not registered in bindless heap");
					return nullptr;
				}
				break;
			case oavk::DescriptorKind::StorageImage:
				idx = bindless.registerStorageImage(
					oa::EngineDeviceAccess::get(inRt),
					binding.imageView,
					binding.imageLayout == VK_IMAGE_LAYOUT_UNDEFINED ? VK_IMAGE_LAYOUT_GENERAL : binding.imageLayout);
				if (idx == OA_BINDLESS_INVALID) {
					outStatus = oa::Status::error(oa::StatusCode::InvalidArgument,
						"image dispatch failed to register storage image");
					return nullptr;
				}
				outStorageImageSlots.pushBack(idx);
				break;
			case oavk::DescriptorKind::SampledImage:
				idx = bindless.registerSampledImage(
					oa::EngineDeviceAccess::get(inRt),
					binding.imageView,
					binding.imageLayout == VK_IMAGE_LAYOUT_UNDEFINED ? VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL : binding.imageLayout);
				if (idx == OA_BINDLESS_INVALID) {
					outStatus = oa::Status::error(oa::StatusCode::InvalidArgument,
						"image dispatch failed to register sampled image");
					return nullptr;
				}
				outSampledImageSlots.pushBack(idx);
				break;
			case oavk::DescriptorKind::Sampler:
				idx = bindless.registerSampler(oa::EngineDeviceAccess::get(inRt), binding.sampler);
				if (idx == OA_BINDLESS_INVALID) {
					outStatus = oa::Status::error(oa::StatusCode::InvalidArgument,
						"image dispatch failed to register sampler");
					return nullptr;
				}
				outSamplerSlots.pushBack(idx);
				break;
			case oavk::DescriptorKind::CombinedImageSampler:
				{
					oa::U32 imageIdx = bindless.registerSampledImage(
						oa::EngineDeviceAccess::get(inRt),
						binding.imageView,
						binding.imageLayout == VK_IMAGE_LAYOUT_UNDEFINED ? VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL : binding.imageLayout);
					if (imageIdx == OA_BINDLESS_INVALID) {
						outStatus = oa::Status::error(oa::StatusCode::InvalidArgument,
							"image dispatch failed to register combined image sampler (image)");
						return nullptr;
					}
					outSampledImageSlots.pushBack(imageIdx);
					
					oa::U32 samplerIdx = bindless.registerSampler(
						oa::EngineDeviceAccess::get(inRt), binding.sampler);
					if (samplerIdx == OA_BINDLESS_INVALID) {
						outStatus = oa::Status::error(oa::StatusCode::InvalidArgument,
							"image dispatch failed to register combined image sampler (sampler)");
						return nullptr;
					}
					outSamplerSlots.pushBack(samplerIdx);
					
					resourceIndices.pushBack(imageIdx);
					idx = samplerIdx;
				}
				break;
		}
		resourceIndices.pushBack(idx);
	}

	oavk::Stream* stream = oa::EngineSubmissionAccess::acquireStream(inRt);
	if (!stream) {
		outStatus = oa::Status::error(oa::StatusCode::VulkanError, "image dispatch: failed to acquire stream");
		return nullptr;
	}

	outStatus = stream->begin(oa::EngineDeviceAccess::get(inRt));
	if (!outStatus.isOk()) {
		oa::EngineSubmissionAccess::releaseStream(inRt, stream);
		return nullptr;
	}

	oa::ComputePipeline& pipeline = oa::EnginePipelineAccess::get(inRt).getPipeline(
		inShaderName, dtype.getValue());
	if (!pipeline.pipeline) {
		outStatus = oa::Status::error(oa::StatusCode::PipelineError, "image dispatch: bindless pipeline not found");
		oa::EngineSubmissionAccess::releaseStream(inRt, stream);
		return nullptr;
	}

	VkCommandBuffer cmd = static_cast<VkCommandBuffer>(stream->commandBuffer);

	// Optional image transitions happen in the same command buffer as the
	// dispatch. This is required when a storage image crosses between compute
	// and vulkan Video layouts; a descriptor layout alone is not a barrier.
	oa::Vector<VkImageMemoryBarrier2> preBarriers;
	for (const auto& binding : inBindings) {
		if (binding.image == VK_NULL_HANDLE) continue;
		const bool externalAcquire =
			binding.initialQueueFamilyIndex != VK_QUEUE_FAMILY_IGNORED;
		VkImageMemoryBarrier2 barrier = {};
		barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2;
		barrier.srcStageMask = externalAcquire or binding.initialLayout == VK_IMAGE_LAYOUT_UNDEFINED
			? VK_PIPELINE_STAGE_2_NONE : VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT;
		barrier.srcAccessMask = externalAcquire or binding.initialLayout == VK_IMAGE_LAYOUT_UNDEFINED
			? VK_ACCESS_2_NONE : VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT;
		barrier.dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
		barrier.dstAccessMask = VK_ACCESS_2_SHADER_READ_BIT | VK_ACCESS_2_SHADER_WRITE_BIT;
		barrier.oldLayout = binding.initialLayout;
		barrier.newLayout = binding.imageLayout;
		barrier.srcQueueFamilyIndex = binding.initialQueueFamilyIndex;
		barrier.dstQueueFamilyIndex = binding.initialQueueFamilyIndex == VK_QUEUE_FAMILY_IGNORED
			? VK_QUEUE_FAMILY_IGNORED : oa::EngineDeviceAccess::get(inRt).queues.computeQueueFamily;
		barrier.image = binding.image;
		barrier.subresourceRange.aspectMask = binding.aspectMask;
		barrier.subresourceRange.baseMipLevel = binding.baseMipLevel;
		barrier.subresourceRange.levelCount = binding.levelCount;
		barrier.subresourceRange.baseArrayLayer = binding.baseArrayLayer;
		barrier.subresourceRange.layerCount = binding.layerCount;
		preBarriers.pushBack(barrier);
	}
	if (not preBarriers.empty()) {
		VkDependencyInfo dependency = {};
		dependency.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
		dependency.imageMemoryBarrierCount = static_cast<oa::U32>(preBarriers.size());
		dependency.pImageMemoryBarriers = preBarriers.data();
		oa::EngineDeviceAccess::get(inRt).deviceDispatch.vkCmdPipelineBarrier2(cmd, &dependency);
	}

	oa::EngineDeviceAccess::get(inRt).deviceDispatch.vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, static_cast<VkPipeline>(pipeline.pipeline));
	VkDescriptorSet descriptorSet =
		static_cast<VkDescriptorSet>(bindless.descriptorSet);
	oa::EngineDeviceAccess::get(inRt).deviceDispatch.vkCmdBindDescriptorSets(
		cmd,
		VK_PIPELINE_BIND_POINT_COMPUTE,
		static_cast<VkPipelineLayout>(pipeline.pipelineLayout),
		0,
		1,
		&descriptorSet,
		0,
		nullptr);

	const oa::U32 headerBytes = static_cast<oa::U32>(resourceIndices.size()) * sizeof(oa::U32);
	const oa::U32 totalPush = headerBytes + inPushSize;
	alignas(16) oa::U8 pushBuf[oavk::OA_VK_MAX_PUSH_CONSTANT_BYTES] = {};
	if (!resourceIndices.empty()) {
		oa::memcpy(pushBuf, resourceIndices.data(), headerBytes);
	}
	if (inPushData && inPushSize > 0) {
		oa::memcpy(pushBuf + headerBytes, inPushData, inPushSize);
	}
	oa::EngineDeviceAccess::get(inRt).deviceDispatch.vkCmdPushConstants(
		cmd,
		static_cast<VkPipelineLayout>(pipeline.pipelineLayout),
		VK_SHADER_STAGE_COMPUTE_BIT,
		0,
		totalPush,
		pushBuf);
	oa::EngineDeviceAccess::get(inRt).deviceDispatch.vkCmdDispatch(cmd, inGroupsX, inGroupsY, inGroupsZ);

	oa::Vector<VkImageMemoryBarrier2> postBarriers;
	for (const auto& binding : inBindings) {
		if (binding.image == VK_NULL_HANDLE or binding.finalLayout == VK_IMAGE_LAYOUT_UNDEFINED
			or binding.finalLayout == binding.imageLayout) continue;
		VkImageMemoryBarrier2 barrier = {};
		const bool externalRelease =
			binding.finalQueueFamilyIndex != VK_QUEUE_FAMILY_IGNORED;
		barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2;
		barrier.srcStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
		barrier.srcAccessMask = binding.kind == oavk::DescriptorKind::SampledImage
			? VK_ACCESS_2_SHADER_READ_BIT
			: VK_ACCESS_2_SHADER_READ_BIT | VK_ACCESS_2_SHADER_WRITE_BIT;
		// Keep this transition valid on a compute-only queue. A cross-queue
		// consumer supplies its actual destination stage/access through the
		// timeline wait and a same-layout barrier on that queue.
		barrier.dstStageMask = externalRelease
			? VK_PIPELINE_STAGE_2_NONE : VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT;
		barrier.dstAccessMask = externalRelease
			? VK_ACCESS_2_NONE
			: binding.finalLayout == VK_IMAGE_LAYOUT_VIDEO_ENCODE_SRC_KHR
			? VK_ACCESS_2_MEMORY_READ_BIT
			: VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT;
		barrier.oldLayout = binding.imageLayout;
		barrier.newLayout = binding.finalLayout;
		barrier.srcQueueFamilyIndex = binding.finalQueueFamilyIndex == VK_QUEUE_FAMILY_IGNORED
			? VK_QUEUE_FAMILY_IGNORED : oa::EngineDeviceAccess::get(inRt).queues.computeQueueFamily;
		barrier.dstQueueFamilyIndex = binding.finalQueueFamilyIndex;
		barrier.image = binding.image;
		barrier.subresourceRange.aspectMask = binding.aspectMask;
		barrier.subresourceRange.baseMipLevel = binding.baseMipLevel;
		barrier.subresourceRange.levelCount = binding.levelCount;
		barrier.subresourceRange.baseArrayLayer = binding.baseArrayLayer;
		barrier.subresourceRange.layerCount = binding.layerCount;
		postBarriers.pushBack(barrier);
	}
	if (not postBarriers.empty()) {
		VkDependencyInfo dependency = {};
		dependency.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
		dependency.imageMemoryBarrierCount = static_cast<oa::U32>(postBarriers.size());
		dependency.pImageMemoryBarriers = postBarriers.data();
		oa::EngineDeviceAccess::get(inRt).deviceDispatch.vkCmdPipelineBarrier2(cmd, &dependency);
	}

	VkMemoryBarrier barrier = {};
	barrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
	barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
	barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
	oa::EngineDeviceAccess::get(inRt).deviceDispatch.vkCmdPipelineBarrier(cmd,
		VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
		VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT,
		0, 1, &barrier, 0, nullptr, 0, nullptr);

	outStatus = oa::Status::ok();
	return stream;
}

oa::Status oavk::ImageDispatch::run(
	oa::Engine& inRt,
	oa::StringView inShaderName,
	oa::Span<const oavk::ImageDispatchBinding> inBindings,
	const void* inPushData,
	oa::U32 inPushSize,
	oa::ScalarType inStorageDtype,
	oa::U32 inGroupsX,
	oa::U32 inGroupsY,
	oa::U32 inGroupsZ)
{
	oa::Vector<oa::U32> storageImageSlots;
	oa::Vector<oa::U32> sampledImageSlots;
	oa::Vector<oa::U32> samplerSlots;
	oa::Status status;
	oavk::Stream* stream = imageDispatchSetupAndRecord(
		inRt, inShaderName, inBindings, inPushData, inPushSize,
		inStorageDtype,
		inGroupsX, inGroupsY, inGroupsZ,
		storageImageSlots, sampledImageSlots, samplerSlots, status);
	if (!stream) {
		return status;
	}
	status = stream->submitAndWait(inRt);
	oa::EngineSubmissionAccess::releaseStream(inRt, stream);
	auto& bindless = oa::EngineBindlessAccess::get(inRt);
	for (oa::U32 idx : storageImageSlots) { bindless.deregisterStorageImage(idx); }
	for (oa::U32 idx : sampledImageSlots) { bindless.deregisterSampledImage(idx); }
	for (oa::U32 idx : samplerSlots) { bindless.deregisterSampler(idx); }
	return status;
}

oa::Result<oavk::ImageDispatchTicket> oavk::ImageDispatch::runAsync(
	oa::Engine& inRt,
	oa::StringView inShaderName,
	oa::Span<const oavk::ImageDispatchBinding> inBindings,
	const void* inPushData,
	oa::U32 inPushSize,
	oa::ScalarType inStorageDtype,
	oa::U32 inGroupsX,
	oa::U32 inGroupsY,
	oa::U32 inGroupsZ)
{
	const oavk::TimelineSemaphore noDependency = {};
	return runWithDependencyAsync(
		inRt, inShaderName, inBindings, inPushData, inPushSize,
		inStorageDtype,
		inGroupsX, inGroupsY, inGroupsZ, noDependency, 0U);
}

oa::Status oavk::ImageDispatch::runWithDependency(
	oa::Engine& inRt,
	oa::StringView inShaderName,
	oa::Span<const oavk::ImageDispatchBinding> inBindings,
	const void* inPushData,
	oa::U32 inPushSize,
	oa::ScalarType inStorageDtype,
	oa::U32 inGroupsX,
	oa::U32 inGroupsY,
	oa::U32 inGroupsZ,
	const oavk::TimelineSemaphore& inWaitSem,
	oa::U64 inWaitValue)
{
	oa::Vector<oa::U32> storageImageSlots;
	oa::Vector<oa::U32> sampledImageSlots;
	oa::Vector<oa::U32> samplerSlots;
	oa::Status status;
	oavk::Stream* stream = imageDispatchSetupAndRecord(
		inRt, inShaderName, inBindings, inPushData, inPushSize,
		inStorageDtype,
		inGroupsX, inGroupsY, inGroupsZ,
		storageImageSlots, sampledImageSlots, samplerSlots, status);
	if (!stream) {
		return status;
	}
	status = stream->submitWithDependency(inRt, inWaitSem, inWaitValue);
	if (status.isOk()) {
		status = stream->synchronize(oa::EngineDeviceAccess::get(inRt));
	}
	oa::EngineSubmissionAccess::releaseStream(inRt, stream);
	auto& bindless = oa::EngineBindlessAccess::get(inRt);
	for (oa::U32 idx : storageImageSlots) { bindless.deregisterStorageImage(idx); }
	for (oa::U32 idx : sampledImageSlots) { bindless.deregisterSampledImage(idx); }
	for (oa::U32 idx : samplerSlots) { bindless.deregisterSampler(idx); }
	return status;
}

oa::Result<oavk::ImageDispatchTicket> oavk::ImageDispatch::runWithDependencyAsync(
	oa::Engine& inRt,
	oa::StringView inShaderName,
	oa::Span<const oavk::ImageDispatchBinding> inBindings,
	const void* inPushData,
	oa::U32 inPushSize,
	oa::ScalarType inStorageDtype,
	oa::U32 inGroupsX,
	oa::U32 inGroupsY,
	oa::U32 inGroupsZ,
	const oavk::TimelineSemaphore& inWaitSem,
	oa::U64 inWaitValue)
{
	oavk::ImageDispatchTicket ticket;
	oa::Status status;
	oavk::Stream* stream = imageDispatchSetupAndRecord(
		inRt, inShaderName, inBindings, inPushData, inPushSize,
		inStorageDtype,
		inGroupsX, inGroupsY, inGroupsZ,
		ticket.storageImageSlots_,
		ticket.sampledImageSlots_,
		ticket.samplerSlots_,
		status);
	if (!stream) {
		return status;
	}
	status = stream->submitWithDependency(inRt, inWaitSem, inWaitValue);
	if (!status.isOk()) {
		oa::EngineSubmissionAccess::releaseStream(inRt, stream);
		auto& bindless = oa::EngineBindlessAccess::get(inRt);
		for (oa::U32 idx : ticket.storageImageSlots_) { bindless.deregisterStorageImage(idx); }
		for (oa::U32 idx : ticket.sampledImageSlots_) { bindless.deregisterSampledImage(idx); }
		for (oa::U32 idx : ticket.samplerSlots_) { bindless.deregisterSampler(idx); }
		return status;
	}
	ticket.engine_ = &inRt;
	ticket.stream_ = stream;
	return ticket;
}
