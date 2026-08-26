#pragma once

#include <oa/core/status.h>
#include <oa/core/types.h>
#include <oa/runtime/sync.h>
#include <oa/runtime/timestamp.h>

namespace oa { class Engine; }
namespace oavk { class Device; }
namespace oavk { class Stream; }

namespace oa {

class TimerSlot {
public:
	oa::U32 index = UINT32_MAX;
	oa::U64 generation = 0;

	[[nodiscard]] oa::Bool isValid() const noexcept {
		return index != UINT32_MAX and generation != 0U;
	}
};

// query pools are engine resources. Timers retain only a weak registry and a
// generation-checked slot, so a timer may safely outlive its engine and a
// pending timer destructor never waits or destroys an in-flight query pool.
class TimerRegistry {
public:
	explicit TimerRegistry(oa::Engine& inOwner) noexcept;
	~TimerRegistry();

	[[nodiscard]] oa::Result<TimerSlot> acquire();
	void release(TimerSlot inSlot) noexcept;
	[[nodiscard]] oa::Status begin(TimerSlot inSlot, oavk::Stream* inStream);
	[[nodiscard]] oa::Status end(TimerSlot inSlot, oavk::Stream* inStream);
	[[nodiscard]] oa::Status attach(
		TimerSlot inSlot, const oa::Event& inCompletion);
	[[nodiscard]] oa::Status markSynchronousComplete(TimerSlot inSlot);
	void cancel(TimerSlot inSlot) noexcept;
	[[nodiscard]] oa::Result<oa::F64> commitMilliseconds(
		TimerSlot inSlot, const oa::Engine& inEngine);
	[[nodiscard]] oa::Bool isPending(TimerSlot inSlot) const noexcept;
	[[nodiscard]] oa::Status close(const oavk::Device& inDevice);

	TimerRegistry(const TimerRegistry&) = delete;
	TimerRegistry& operator=(const TimerRegistry&) = delete;

private:
	enum class SlotState : oa::U8 {
		Free,
		Ready,
		Recording,
		Recorded,
		Submitted,
		Retired,
	};
	class Slot {
	public:
		oavk::Timestamp timestamp;
		SlotState state = SlotState::Free;
		oa::U64 generation = 0;
		oa::Event completion;
		oa::Bool synchronousComplete = false;
	};

	[[nodiscard]] Slot* resolve_(TimerSlot inSlot) noexcept;
	[[nodiscard]] const Slot* resolve_(TimerSlot inSlot) const noexcept;

	oa::Engine* owner_ = nullptr;
	oa::Vec<oa::UniquePtr<Slot>> slots_;
	mutable oa::Mutex mutex_;
	oa::Bool open_ = true;
};

} // namespace oa
