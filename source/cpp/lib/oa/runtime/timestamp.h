// OA VULKAN - GPU timestamp Profiling
//
// Private query-pool implementation used by engine-owned profiling and graph
// lowering. SDK callers use oa::Timer.

#pragma once

#include <oa/core/types.h>
#include <oa/core/status.h>

namespace oa {
class ExecutableGraph;
class Engine;
class TimerRegistry;
}

namespace oavk {

class Device;
class Stream;

class Timestamp {
public:
	void* pool = nullptr;
	oa::U32 capacity = 0;
	oa::U32 writeIndex = 0;
	oa::F64 nanosPerTick = 0.0;
	oa::U32 validBits = 0;
	oa::Vec<oa::U64> results;

	Timestamp() = default;
	~Timestamp();

	[[nodiscard]] static oa::Result<Timestamp> create(
		oa::Engine &inRt, oa::U32 inMaxQueries
	);

	void reset(oavk::Stream *inStream);
	void writeTimestamp(oavk::Stream *inStream);
	[[nodiscard]] oa::Status readback(const oa::Engine &inEngine);
	[[nodiscard]] oa::F64 elapsedMs(oa::U32 inStartIdx, oa::U32 inEndIdx) const;
	[[nodiscard]] oa::F64 elapsedNs(oa::U32 inStartIdx, oa::U32 inEndIdx) const;

	[[nodiscard]] oa::U32 count() const noexcept { return writeIndex; }
	[[nodiscard]] bool isInitialized() const noexcept { return pool != nullptr; }

	Timestamp(Timestamp &&inOther) noexcept;
	Timestamp &operator=(Timestamp &&inOther) noexcept;
	Timestamp(const Timestamp &) = delete;
	Timestamp &operator=(const Timestamp &) = delete;

private:
	friend class oa::ExecutableGraph;
	friend class oa::TimerRegistry;
	const oa::Engine* owner_ = nullptr;
	void reset_() noexcept;
	void destroyDevice_(const oavk::Device &inDevice);
	[[nodiscard]] oa::Status readbackDevice_(const oavk::Device &inDevice);
};

} // namespace oavk
