#pragma once

#include "engineAccess.h"

namespace oa {

// Private raw-resource bridge for vulkan lowering and focused white-box tests.
// Public callers allocate and transfer through semantic values and operations;
// they never select raw buffer placement or manage oavk::Buffer lifetime.
class EngineResourceAccess {
public:
	[[nodiscard]] static oa::Result<oavk::Buffer> allocBuffer(
		oa::Engine& inEngine,
		oa::U64 inSize)
	{
		return oa::EngineAccess(inEngine).allocBuffer(inSize);
	}

	[[nodiscard]] static oa::Result<oavk::Buffer> allocBuffer(
		oa::Engine& inEngine,
		oa::U64 inSize,
		oa::MemoryPlacement inPlacement)
	{
		return oa::EngineAccess(inEngine).allocBuffer(inSize, inPlacement);
	}

	[[nodiscard]] static oa::Result<oavk::Buffer> allocBufferDevice(
		oa::Engine& inEngine,
		oa::U64 inSize)
	{
		return oa::EngineAccess(inEngine).allocBufferDevice(inSize);
	}

	[[nodiscard]] static oa::Result<oavk::Buffer> allocBufferBar(
		oa::Engine& inEngine,
		oa::U64 inSize)
	{
		return oa::EngineAccess(inEngine).allocBufferBar(inSize);
	}

	[[nodiscard]] static oa::MemoryPlacement defaultMatrixPlacement(
		const oa::Engine& inEngine) noexcept
	{
		return oa::EngineAccess::get(inEngine).matrixPlacement_;
	}

	[[nodiscard]] static oa::Status uploadBuffer(
		oa::Engine& inEngine,
		const oavk::Buffer& inDst,
		oa::U64 inDstOffset,
		const void* inData,
		oa::U64 inSize)
	{
		return oa::EngineAccess(inEngine).uploadBuffer(
			inDst, inDstOffset, inData, inSize);
	}

	[[nodiscard]] static oa::Status readbackBuffer(
		oa::Engine& inEngine,
		const oavk::Buffer& inSrc,
		oa::U64 inSrcOffset,
		void* outData,
		oa::U64 inSize)
	{
		return oa::EngineAccess(inEngine).readbackBuffer(
			inSrc, inSrcOffset, outData, inSize);
	}

	static void freeBuffer(
		oa::Engine& inEngine,
		oavk::Buffer& inOutBuffer)
	{
		oa::EngineAccess(inEngine).freeBuffer(inOutBuffer);
	}

	[[nodiscard]] static oa::Result<oa::Event> copyBufferAsync(
		oa::Engine& inEngine,
		const oavk::Buffer& inSrc,
		const oavk::Buffer& inDst,
		oa::U64 inSize)
	{
		return oa::EngineAccess(inEngine).copyBufferAsync(inSrc, inDst, inSize);
	}
};

} // namespace oa
