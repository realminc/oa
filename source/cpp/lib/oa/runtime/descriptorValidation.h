#pragma once

#include <oa/core/status.h>
#include <oa/runtime/allocator.h>
#include <oa/runtime/bindless.h>
#include <oa/runtime/device.h>

// One always-on admission contract for every storage-buffer descriptor route.
// Keep this independent of oa::Validation: these are vulkan legality checks, not
// optional diagnostics, and must remain active in release builds.
namespace oavk {

[[nodiscard]] inline oa::Status validateStorageBufferDescriptor(
	const oavk::Device& inDevice,
	const oavk::Buffer& inBuffer,
	oa::Bool inRequireBindlessRegistration = false,
	const void* inExpectedAllocatorIdentity = nullptr)
{
	if (not inBuffer.buffer) {
		return oa::Status::error(oa::StatusCode::InvalidArgument,
			"storage buffer descriptor requires a non-null VkBuffer");
	}
	if (inRequireBindlessRegistration
		and inBuffer.bindlessIndex == OA_BINDLESS_INVALID)
	{
		return oa::Status::error(oa::StatusCode::FailedPrecondition,
			"storage buffer descriptor is not registered in the selected device bindless heap");
	}
	if (inExpectedAllocatorIdentity != nullptr
		and inBuffer.allocatorIdentity != inExpectedAllocatorIdentity)
	{
		return oa::Status::error(oa::StatusCode::InvalidArgument,
			"storage buffer descriptor is not owned by the selected engine allocator");
	}

	const oa::U64 range = inBuffer.descriptorRange();
	if (range == 0U) {
		return oa::Status::error(oa::StatusCode::InvalidArgument,
			"storage buffer descriptor range must be non-zero");
	}
	const oa::U64 maximum =
		inDevice.info.hardware.maxStorageBufferRangeBytes;
	if (maximum == 0U) {
		return oa::Status::error(oa::StatusCode::FailedPrecondition,
			"selected device has no recorded maxStorageBufferRange");
	}
	if (range > maximum) {
		return oa::Status::error(oa::StatusCode::OutOfRange,
			"storage buffer descriptor range exceeds the selected device maxStorageBufferRange");
	}
	return oa::Status::ok();
}

template<typename BufferType>
[[nodiscard]] inline oa::Status validateStorageBufferDescriptors(
	const oavk::Device& inDevice,
	oa::Span<BufferType> inBuffers,
	oa::Bool inRequireBindlessRegistration = false,
	const void* inExpectedAllocatorIdentity = nullptr)
{
	for (const auto& buffer : inBuffers) {
		OA_RETURN_IF_ERROR(validateStorageBufferDescriptor(
			inDevice,
			buffer,
			inRequireBindlessRegistration,
			inExpectedAllocatorIdentity));
	}
	return oa::Status::ok();
}

} // namespace oavk
