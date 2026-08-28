// Private oa::Engine memory allocator over the forked VMA implementation.

#pragma once

#include <oa/core/status.h>
#include <oa/runtime/buffer.h>

namespace oavk { class Device; }

class RuntimeAllocatorStats {
public:
	oa::U64 usedBytes = 0;
	oa::U64 budgetBytes = 0;
	oa::U64 allocationBytes = 0;
	oa::U64 blockBytes = 0;
	oa::U64 allocationCount = 0;
	oa::U64 blockCount = 0;
};

class RuntimeAllocator {
public:
	void* allocator = nullptr;
	oa::Bool hasSam = false;

	[[nodiscard]] static oa::Result<RuntimeAllocator> create(const oavk::Device& inDevice);
	void destroy();

	[[nodiscard]] oa::Result<oavk::Buffer> allocDevice(oa::U64 inSize);
	[[nodiscard]] oa::Result<oavk::Buffer> allocHostVisible(oa::U64 inSize);
	[[nodiscard]] oa::Result<oavk::Buffer> allocHostReadback(oa::U64 inSize);
	[[nodiscard]] oa::Result<oavk::Buffer> allocBar(oa::U64 inSize);
	[[nodiscard]] oa::Result<oavk::Buffer> allocPreprocessBuffer(oa::U64 inSize);
	// Raw VMA destruction has no descriptor-heap authority. buffers registered
	// with an oa::Engine must be released through that engine's resource-lifetime
	// path, or explicitly deregistered by the same engine before this call.
	void free(oavk::Buffer& inOutBuffer);

	oa::Status uploadWeights(
		oavk::Buffer& inDst, const void* inSrc, oa::U64 inSize);
	oa::Bool flushHostBuffer(
		const oavk::Buffer& inBuf, oa::U64 inOffset, oa::U64 inSize);
	oa::Bool invalidateHostBuffer(
		const oavk::Buffer& inBuf, oa::U64 inOffset, oa::U64 inSize);

	[[nodiscard]] oa::Result<oavk::Buffer> allocAliased(
		oa::U64 inSize,
		oa::MemoryPlacement inPlacement = oa::MemoryPlacement::HostUpload);
	[[nodiscard]] oa::Result<oavk::Buffer> createAliasingBuffer(
		const oavk::Buffer& inExisting, oa::U64 inSize);
	void freeAlias(oavk::Buffer& inOutBuffer);

	[[nodiscard]] RuntimeAllocatorStats getStats() const;
};
