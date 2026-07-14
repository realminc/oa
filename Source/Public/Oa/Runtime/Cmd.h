#pragma once

#include <Oa/Core/Types.h>
#include <Oa/Core/Status.h>

class OaVkDevice;
class OaVkBuffer;
class OaComputePipeline;

class OaVkCmd {
public:
	void* CommandPool = nullptr;
	void* CommandBuffer = nullptr;

	[[nodiscard]] static OaResult<OaVkCmd> Create(const OaVkDevice& InDevice);
	void Destroy(const OaVkDevice& InDevice);

	[[nodiscard]] OaStatus Begin();
	[[nodiscard]] OaStatus End();
	void BindPipeline(const OaComputePipeline& InPipeline);
	void BindDescriptors(const OaComputePipeline& InPipeline);
	void BindDescriptorSet(void* InPipelineLayout, void* InDescriptorSet);
	void PushConstants(const OaComputePipeline& InPipeline, const void* InData, OaU32 InSize);
	void Dispatch(OaU32 InGroupsX, OaU32 InGroupsY = 1, OaU32 InGroupsZ = 1);
	void BufferBarrier();
	void CopyBuffer(const OaVkBuffer& InSrc, const OaVkBuffer& InDst, OaU64 InSize);
	[[nodiscard]] OaStatus Submit(const OaVkDevice& InDevice);
	[[nodiscard]] OaStatus SubmitAndWait(const OaVkDevice& InDevice);
};
