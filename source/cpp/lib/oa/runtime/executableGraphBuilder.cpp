#include "executableGraphBuilder.h"

#include <oa/core/envFlag.h>
#include <oa/core/log.h>
#include <oa/runtime/bindless.h>
#include <oa/runtime/executableGraph.h>
#include <oa/runtime/computeKernel.h>
#include <oa/runtime/spirv.h>
#include "dispatchValidation.h"

oa::Status oa::ExecutableGraphBuilder::record(const oa::ComputeDispatchDesc& inDesc) {
	if (not graph_) {
		return oa::Status::error(oa::StatusCode::Internal,
			"graph builder record: no graph attached");
	}
	if (inDesc.kernel.empty()) {
		return oa::Status::error(oa::StatusCode::InvalidArgument,
			"graph builder record: empty kernel name");
	}
	if (inDesc.access.size() != inDesc.buffers.size()) {
		OaLogError(oa::LogComponent::Compute,
			"oa::ExecutableGraphBuilder::record '{}': access={} buffers={}",
			inDesc.kernel,
			inDesc.access.size(), inDesc.buffers.size());
		return oa::Status::error(oa::StatusCode::InvalidArgument,
			"graph builder record: buffer access count mismatch");
	}
	if (not inDesc.bufferOwners.empty()
		and inDesc.bufferOwners.size() != inDesc.buffers.size())
	{
		return oa::Status::error(oa::StatusCode::InvalidArgument,
			"graph builder record: buffer owner count mismatch");
	}
	if (inDesc.pushSize > oavk::OA_VK_MAX_PUSH_CONSTANT_BYTES
		or (inDesc.pushSize != 0U and inDesc.pushData == nullptr))
	{
		return oa::Status::error(oa::StatusCode::InvalidArgument,
			"graph builder record: invalid push payload");
	}
	for (oa::U32 operation = 0;
		operation < inDesc.semanticOps.size(); ++operation)
	{
		if (inDesc.semanticOps[operation]
			== oa::invalidSemanticOpId)
		{
			return oa::Status::error(oa::StatusCode::InvalidArgument,
				"graph builder record: invalid semantic provenance");
		}
		for (oa::U32 previous = 0; previous < operation; ++previous) {
			if (inDesc.semanticOps[previous]
				== inDesc.semanticOps[operation])
			{
				return oa::Status::error(oa::StatusCode::AlreadyExists,
					"graph builder record: duplicate semantic provenance");
			}
		}
	}
	if (not oavk::bindlessPushFits(
			static_cast<oa::U32>(inDesc.buffers.size()), inDesc.pushSize))
	{
		return oa::Status::error(oa::StatusCode::InvalidArgument,
			"graph builder record: bindless index header plus push payload exceeds limit");
	}
	if (inDesc.indirect) {
		OA_RETURN_IF_ERROR(oavk::validateIndirectComputeDispatch(
			inDesc.indirectBuffer, inDesc.indirectOffset));
	} else if (inDesc.indirectBuffer.buffer or inDesc.indirectOffset != 0) {
		return oa::Status::error(oa::StatusCode::InvalidArgument,
			"graph builder record: indirect fields set on a direct dispatch");
	}

#ifndef NDEBUG
	const oa::String kernelName(inDesc.kernel);
	if (oa::computeKernelUsesDefaultBindlessPipeline(kernelName.cStr())) {
		const oa::U32 declared = oavk::spirvPushConstantBlockSizeByName(kernelName.cStr());
		if (declared != 0U) {
			const oa::U32 assembled =
				static_cast<oa::U32>(inDesc.buffers.size()) * sizeof(oa::U32)
				+ inDesc.pushSize;
			if (assembled != declared) {
				OaLogError(oa::LogComponent::Compute,
					"bindless push mismatch for '{}': {} buffers * 4 + {} push "
					"tail = {}, shader declares {} bytes",
					kernelName.cStr(), static_cast<oa::U32>(inDesc.buffers.size()),
					inDesc.pushSize, assembled, declared);
				if (not oa::EnvFlag::isSet("OA_DISABLE_PUSH_CHECK")) {
					return oa::Status::error(oa::StatusCode::InvalidArgument,
						"graph builder record: bindless push contract mismatch");
				}
			}
		}
	}
#endif

	graph_->add(inDesc);
	return oa::Status::ok();
}
