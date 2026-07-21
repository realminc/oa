#include "GraphBuilder.h"

#include <Oa/Core/EnvFlag.h>
#include <Oa/Core/Log.h>
#include <Oa/Runtime/Bindless.h>
#include <Oa/Runtime/ComputeGraph.h>
#include <Oa/Runtime/ComputeKernel.h>
#include <Oa/Runtime/Spirv.h>

OaStatus OaGraphBuilder::Record(const OaComputeDispatchDesc& InDesc) {
	if (not Graph_) {
		return OaStatus::Error(OaStatusCode::Internal,
			"graph builder record: no graph attached");
	}
	if (InDesc.Kernel.Empty()) {
		return OaStatus::Error(OaStatusCode::InvalidArgument,
			"graph builder record: empty kernel name");
	}
	if (InDesc.Access.Size() != InDesc.Buffers.Size()) {
		OA_LOG_ERROR(OaLogComponent::Core,
			"OaGraphBuilder::Record '%.*s': access=%zu buffers=%zu",
			static_cast<int>(InDesc.Kernel.Size()), InDesc.Kernel.Data(),
			InDesc.Access.Size(), InDesc.Buffers.Size());
		return OaStatus::Error(OaStatusCode::InvalidArgument,
			"graph builder record: buffer access count mismatch");
	}
	if (not InDesc.BufferOwners.Empty()
		and InDesc.BufferOwners.Size() != InDesc.Buffers.Size())
	{
		return OaStatus::Error(OaStatusCode::InvalidArgument,
			"graph builder record: buffer owner count mismatch");
	}
	if (InDesc.PushSize > OA_VK_MAX_PUSH_CONSTANT_BYTES
		or (InDesc.PushSize != 0U and InDesc.PushData == nullptr))
	{
		return OaStatus::Error(OaStatusCode::InvalidArgument,
			"graph builder record: invalid push payload");
	}
	for (OaU32 operation = 0;
		operation < InDesc.SemanticOperations.Size(); ++operation)
	{
		if (InDesc.SemanticOperations[operation]
			== OaInvalidSemanticOperationId)
		{
			return OaStatus::Error(OaStatusCode::InvalidArgument,
				"graph builder record: invalid semantic provenance");
		}
		for (OaU32 previous = 0; previous < operation; ++previous) {
			if (InDesc.SemanticOperations[previous]
				== InDesc.SemanticOperations[operation])
			{
				return OaStatus::Error(OaStatusCode::AlreadyExists,
					"graph builder record: duplicate semantic provenance");
			}
		}
	}
	if (not OaVkBindlessPushFits(
			static_cast<OaU32>(InDesc.Buffers.Size()), InDesc.PushSize))
	{
		return OaStatus::Error(OaStatusCode::InvalidArgument,
			"graph builder record: bindless index header plus push payload exceeds limit");
	}
	constexpr OaU64 dispatchCommandSize = 3ULL * sizeof(OaU32);
	if (InDesc.Indirect) {
		if (not InDesc.IndirectBuffer.Buffer
			or (InDesc.IndirectOffset & 3ULL) != 0
			or InDesc.IndirectBuffer.Size < dispatchCommandSize
			or InDesc.IndirectOffset > InDesc.IndirectBuffer.Size - dispatchCommandSize)
		{
			return OaStatus::Error(OaStatusCode::InvalidArgument,
				"graph builder record: invalid indirect dispatch arguments");
		}
	} else if (InDesc.IndirectBuffer.Buffer or InDesc.IndirectOffset != 0) {
		return OaStatus::Error(OaStatusCode::InvalidArgument,
			"graph builder record: indirect fields set on a direct dispatch");
	}

#ifndef NDEBUG
	const OaString kernelName(InDesc.Kernel);
	if (OaComputeKernelUsesDefaultBindlessPipeline(kernelName.c_str())) {
		const OaU32 declared = OaSpvPushConstantBlockSizeByName(kernelName.c_str());
		if (declared != 0U) {
			const OaU32 assembled =
				static_cast<OaU32>(InDesc.Buffers.Size()) * sizeof(OaU32)
				+ InDesc.PushSize;
			if (assembled != declared) {
				OA_LOG_ERROR(OaLogComponent::Core,
					"Bindless push mismatch for '%s': %u buffers * 4 + %u push "
					"tail = %u, shader declares %u bytes",
					kernelName.c_str(), static_cast<OaU32>(InDesc.Buffers.Size()),
					InDesc.PushSize, assembled, declared);
				if (not OaEnvFlag::IsSet("OA_DISABLE_PUSH_CHECK")) {
					return OaStatus::Error(OaStatusCode::InvalidArgument,
						"graph builder record: bindless push contract mismatch");
				}
			}
		}
	}
#endif

	Graph_->Add(InDesc);
	return OaStatus::Ok();
}
