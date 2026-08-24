// OA RUNTIME - Universal Host/Device Timer
//
// One public timer represents a measured region. Host regions use the steady
// clock; device regions use the backend owned by the named oa::Engine. Adding a
// future NPU or other executor extends the private device backend rather than
// creating one public timer class per device kind.

#pragma once

#include <oa/core/status.h>
#include <oa/core/types.h>

namespace oa {

class Engine;
class TimerAccess;

enum class TimerDomain : oa::U8 {
	Auto,
	Host,
	Device,
};

class Timer {
public:
	explicit Timer(
		oa::TimerDomain inDomain = oa::TimerDomain::Auto,
		oa::StringView inName = {});
	~Timer();

	// Select and acquire this engine's device timestamp backend. Auto resolves
	// to Device here. Host timers need no engine and init is a no-op for them.
	[[nodiscard]] oa::Status init(
		oa::Engine& inEngine,
		oa::StringView inName = {});

	// Host-region boundaries. Auto resolves to Host when Begin is the first
	// operation. Device timestamps are bracketed by oa::Engine::submit(timer).
	[[nodiscard]] oa::Status begin();
	[[nodiscard]] oa::Status end();

	// commit reports a status instead of converting unavailable/incomplete
	// measurements into a plausible 0 ms sample. Device commit never waits: the
	// exact submission event must already be complete. Host commit has no engine.
	[[nodiscard]] oa::Result<oa::F64> commit(
		const oa::Engine& inEngine,
		oa::F64 inUnits = 1.0);
	[[nodiscard]] oa::Result<oa::F64> commit(oa::F64 inUnits = 1.0);

	[[nodiscard]] oa::F64 elapsedMs() const noexcept;
	[[nodiscard]] oa::F64 throughput() const noexcept;
	[[nodiscard]] oa::TimerDomain domain() const noexcept;
	[[nodiscard]] oa::Bool isHost() const noexcept;
	[[nodiscard]] oa::Bool isDevice() const noexcept;
	[[nodiscard]] oa::Bool isInitialized() const noexcept;
	[[nodiscard]] oa::Bool isPending() const noexcept;
	[[nodiscard]] oa::StringView getName() const noexcept;

	Timer(const Timer&) = delete;
	Timer& operator=(const Timer&) = delete;
	Timer(Timer&&) noexcept;
	Timer& operator=(Timer&&) noexcept;

private:
	friend class TimerAccess;
	class Impl;
	oa::UniquePtr<Impl> impl_;
};

} // namespace oa
