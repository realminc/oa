#pragma once

#include <oa/core/types.h>
#include <oa/core/status.h>
#include <oa/runtime/computeKernel.h>
#include <oa/runtime/engine.h>

namespace oavk {

class Stream;
class Buffer;

class Batch {
public:
	Stream* stream = nullptr;
};

class Dispatch {
public:
	// Raw vulkan buffers do not carry semantic dtype metadata. Callers must name
	// the storage type consumed by storage.slang's dtype-following bindings;
	// Runtime lowers it to the exact cached pipeline specialization.
	[[nodiscard]] static oa::Status run(
		oa::Engine& inRuntime,
		oa::StringView inPipelineName,
		oa::Span<oavk::Buffer> inBuffers,
		const void* inPushData,
		oa::U32 inPushSize,
		oa::ScalarType inStorageDtype,
		oa::U32 inGroupsX,
		oa::U32 inGroupsY = 1,
		oa::U32 inGroupsZ = 1
	);
	[[nodiscard]] static oa::Status run(
		oa::Engine& inRuntime,
		oa::U64 inKernelId,
		oa::Span<oavk::Buffer> inBuffers,
		const void* inPushData,
		oa::U32 inPushSize,
		oa::ScalarType inStorageDtype,
		oa::U32 inGroupsX,
		oa::U32 inGroupsY = 1,
		oa::U32 inGroupsZ = 1
	);

	// GPU-driven dispatch: workgroup counts read from inIndirectBuffer.
	// The buffer must support indirect dispatch, belong to the selected device
	// node, and contain one aligned three-u32 command at inOffset.
	[[nodiscard]] static oa::Status runIndirect(
		oa::Engine& inRuntime,
		oa::StringView inPipelineName,
		oa::Span<oavk::Buffer> inBuffers,
		const void* inPushData,
		oa::U32 inPushSize,
		oa::ScalarType inStorageDtype,
		const oavk::Buffer& inIndirectBuffer,
		oa::U64 inOffset = 0
	);

	[[nodiscard]] static oa::Result<Batch> beginBatch(oa::Engine& inRuntime);

	[[nodiscard]] static oa::Status record(
		Batch& inBatch,
		oa::Engine& inRuntime,
		oa::StringView inPipelineName,
		oa::Span<oavk::Buffer> inBuffers,
		const void* inPushData,
		oa::U32 inPushSize,
		oa::ScalarType inStorageDtype,
		oa::U32 inGroupsX,
		oa::U32 inGroupsY = 1,
		oa::U32 inGroupsZ = 1
	);
	[[nodiscard]] static oa::Status record(
		Batch& inBatch,
		oa::Engine& inRuntime,
		oa::U64 inKernelId,
		oa::Span<oavk::Buffer> inBuffers,
		const void* inPushData,
		oa::U32 inPushSize,
		oa::ScalarType inStorageDtype,
		oa::U32 inGroupsX,
		oa::U32 inGroupsY = 1,
		oa::U32 inGroupsZ = 1
	);

	[[nodiscard]] static oa::Status flush(Batch& inBatch, oa::Engine& inRuntime);

};

} // namespace oavk
