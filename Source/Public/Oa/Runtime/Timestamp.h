// OA VULKAN - GPU Timestamp Profiling
//
// Lightweight per-dispatch GPU timing via VkQueryPool + vkCmdWriteTimestamp2.
// Create once, bracket dispatches with WriteTimestamp, read back after submit.
//
// Usage:
//   OaVkTimestamp ts;
//   OA_RETURN_IF_ERROR(ts.Create(rt, 128));  // 128 dispatch slots
//   stream->Begin(rt.Device);
//   ts.Reset(stream);
//   ts.WriteTimestamp(stream, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT);
//   stream->RecordDispatch(rt, "Silu", ...);
//   ts.WriteTimestamp(stream, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT);
//   stream->SubmitAndWait(rt);
//   ts.Readback(rt.Device);
//   double ms = ts.ElapsedMs(0, 1);

#pragma once

#include <Oa/Core/Types.h>
#include <Oa/Core/Status.h>

class OaEngine;
class OaVkDevice;
class OaVkStream;

class OaVkTimestamp {
public:
	void* Pool = nullptr;
	OaU32 Capacity = 0;
	OaU32 WriteIndex = 0;
	OaF64 NanosPerTick = 0.0;
	OaVec<OaU64> Results;

	OaVkTimestamp() = default;
	~OaVkTimestamp() = default;

	[[nodiscard]] static OaResult<OaVkTimestamp> Create(
		OaEngine &InRt, OaU32 InMaxQueries
	);
	void Destroy(const OaVkDevice &InDevice);

	void Reset(OaVkStream *InStream);

	void WriteTimestamp(OaVkStream *InStream);

	[[nodiscard]] OaStatus Readback(const OaVkDevice &InDevice);

	[[nodiscard]] OaF64 ElapsedMs(OaU32 InStartIdx, OaU32 InEndIdx) const;
	[[nodiscard]] OaF64 ElapsedNs(OaU32 InStartIdx, OaU32 InEndIdx) const;

	[[nodiscard]] OaU32 Count() const noexcept { return WriteIndex; }

	OaVkTimestamp(OaVkTimestamp &&InOther) noexcept;
	OaVkTimestamp &operator=(OaVkTimestamp &&InOther) noexcept;
	OaVkTimestamp(const OaVkTimestamp &) = delete;
	OaVkTimestamp &operator=(const OaVkTimestamp &) = delete;
};
