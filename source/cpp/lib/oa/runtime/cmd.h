#pragma once

#include <oa/core/types.h>
#include <oa/core/status.h>

namespace oa { class ComputePipeline; }
struct OaVkDeviceTable;

namespace oavk {

class Buffer;
class Device;

class Command {
public:
	void* commandPool = nullptr;
	void* commandBuffer = nullptr;
	// Borrowed from the device that owns this command pool. The device must
	// outlive the command object, just as it must outlive the vulkan handles.
	const OaVkDeviceTable* deviceDispatch = nullptr;

	[[nodiscard]] static oa::Result<Command> create(const Device& inDevice);
	void destroy(const oavk::Device& inDevice);

	[[nodiscard]] oa::Status begin();
	[[nodiscard]] oa::Status end();
	void bindPipeline(const oa::ComputePipeline& inPipeline);
	void bindDescriptors(const oa::ComputePipeline& inPipeline);
	void bindDescriptorSet(void* inPipelineLayout, void* inDescriptorSet);
	void pushConstants(const oa::ComputePipeline& inPipeline, const void* inData, oa::U32 inSize);
	[[nodiscard]] oa::Status dispatch(
		const oavk::Device& inDevice,
		oa::U32 inGroupsX,
		oa::U32 inGroupsY = 1,
		oa::U32 inGroupsZ = 1);
	void bufferBarrier();
	void copyBuffer(const oavk::Buffer& inSrc, const oavk::Buffer& inDst, oa::U64 inSize);
	[[nodiscard]] oa::Status submit(const oavk::Device& inDevice);
	[[nodiscard]] oa::Status submitAndWait(const oavk::Device& inDevice);
};

} // namespace oavk
