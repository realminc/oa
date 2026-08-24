#pragma once

#include <oa/core/status.h>
#include <oa/runtime/allocator.h>
#include <oa/runtime/device.h>

// Direct-dispatch counts are host-known and must be admitted against the
// selected physical device before command recording. Zero is legal vulkan
// input and intentionally remains a no-op dispatch. indirect counts are
// GPU-authored and have a separate execution-time contract.
namespace oavk {

[[nodiscard]] inline oa::Status validateDirectComputeDispatch(
	const oavk::Device& inDevice,
	oa::U32 inGroupsX,
	oa::U32 inGroupsY,
	oa::U32 inGroupsZ)
{
	const auto& hardware = inDevice.info.hardware;
	if (hardware.maxComputeWorkGroupCountX == 0U
		or hardware.maxComputeWorkGroupCountY == 0U
		or hardware.maxComputeWorkGroupCountZ == 0U)
	{
		return oa::Status::error(oa::StatusCode::FailedPrecondition,
			"selected device has no recorded maxComputeWorkGroupCount");
	}
	if (inGroupsX > hardware.maxComputeWorkGroupCountX) {
		return oa::Status::error(oa::StatusCode::OutOfRange,
			"compute dispatch group count X exceeds the selected device "
			"maxComputeWorkGroupCount[0]");
	}
	if (inGroupsY > hardware.maxComputeWorkGroupCountY) {
		return oa::Status::error(oa::StatusCode::OutOfRange,
			"compute dispatch group count Y exceeds the selected device "
			"maxComputeWorkGroupCount[1]");
	}
	if (inGroupsZ > hardware.maxComputeWorkGroupCountZ) {
		return oa::Status::error(oa::StatusCode::OutOfRange,
			"compute dispatch group count Z exceeds the selected device "
			"maxComputeWorkGroupCount[2]");
	}
	return oa::Status::ok();
}

// indirect group counts stay GPU-authored and are not mapped or synchronized
// here. This validates the host-known VkBuffer contract before command
// recording: usage capability, selected-engine allocation provenance,
// four-byte alignment, and an overflow-safe
// VkDispatchIndirectCommand-sized logical range.
[[nodiscard]] inline oa::Status validateIndirectComputeDispatch(
	const oavk::Buffer& inBuffer,
	oa::U64 inOffset,
	const void* inExpectedAllocatorIdentity = nullptr)
{
	if (not inBuffer.buffer) {
		return oa::Status::error(oa::StatusCode::InvalidArgument,
			"indirect compute dispatch requires a buffer");
	}
	if (not inBuffer.supportsIndirectDispatch()) {
		return oa::Status::error(oa::StatusCode::InvalidArgument,
			"indirect compute dispatch buffer lacks indirect-dispatch usage");
	}
	if (inExpectedAllocatorIdentity != nullptr
		and inBuffer.allocatorIdentity != inExpectedAllocatorIdentity)
	{
		return oa::Status::error(oa::StatusCode::InvalidArgument,
			"indirect compute dispatch buffer is not owned by the selected engine allocator");
	}
	if ((inOffset & 3ULL) != 0U) {
		return oa::Status::error(oa::StatusCode::InvalidArgument,
			"indirect compute dispatch offset must be four-byte aligned");
	}
	constexpr oa::U64 commandSize = 3ULL * sizeof(oa::U32);
	if (inBuffer.size < commandSize
		or inOffset > inBuffer.size - commandSize)
	{
		return oa::Status::error(oa::StatusCode::OutOfRange,
			"indirect compute dispatch command exceeds the logical buffer range");
	}
	return oa::Status::ok();
}

} // namespace oavk
