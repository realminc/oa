#include "timerRegistry.h"

#include <oa/runtime/engine.h>
#include <oa/runtime/stream.h>

namespace oa {

TimerRegistry::TimerRegistry(oa::Engine& inOwner) noexcept
	: owner_(&inOwner) {}

TimerRegistry::~TimerRegistry() = default;

TimerRegistry::Slot* TimerRegistry::resolve_(
	TimerSlot inSlot) noexcept
{
	if (not inSlot.isValid() or inSlot.index >= slots_.size()) return nullptr;
	auto* slot = slots_[inSlot.index].get();
	return slot and slot->generation == inSlot.generation ? slot : nullptr;
}

const TimerRegistry::Slot* TimerRegistry::resolve_(
	TimerSlot inSlot) const noexcept
{
	if (not inSlot.isValid() or inSlot.index >= slots_.size()) return nullptr;
	const auto* slot = slots_[inSlot.index].get();
	return slot and slot->generation == inSlot.generation ? slot : nullptr;
}

oa::Result<TimerSlot> TimerRegistry::acquire() {
	std::lock_guard<std::mutex> lock(mutex_);
	if (not open_ or owner_ == nullptr or not owner_->isReady()) {
		return oa::Status::error(oa::StatusCode::FailedPrecondition,
			"device timer requires a ready engine timer registry");
	}

	for (oa::U32 index = 0; index < slots_.size(); ++index) {
		auto& slot = *slots_[index];
		if (slot.state == SlotState::Retired
			and slot.completion.isValid() and slot.completion.isComplete()) {
			slot.state = SlotState::Free;
			slot.completion = {};
		}
		if (slot.state != SlotState::Free) continue;
		++slot.generation;
		if (slot.generation == 0U) ++slot.generation;
		slot.state = SlotState::Ready;
		slot.synchronousComplete = false;
		return TimerSlot{.index = index, .generation = slot.generation};
	}

	auto timestamp = oavk::Timestamp::create(*owner_, 2U);
	if (not timestamp.isOk()) return timestamp.getStatus();
	auto slot = oa::makeUnique<Slot>();
	slot->timestamp = oa::move(timestamp).getValue();
	slot->state = SlotState::Ready;
	slot->generation = 1U;
	slots_.pushBack(oa::move(slot));
	return TimerSlot{
		.index = static_cast<oa::U32>(slots_.size() - 1U),
		.generation = 1U,
	};
}

void TimerRegistry::release(TimerSlot inSlot) noexcept {
	std::lock_guard<std::mutex> lock(mutex_);
	auto* slot = resolve_(inSlot);
	if (slot == nullptr) return;
	if (slot->state == SlotState::Submitted
		and not slot->synchronousComplete
		and (not slot->completion.isValid() or not slot->completion.isComplete())) {
		slot->state = SlotState::Retired;
		return;
	}
	if (slot->state == SlotState::Recording
		or slot->state == SlotState::Recorded) {
		// The command buffer may still reference this pool. Keep it quarantined
		// until the explicit engine Close boundary destroys command buffers first.
		slot->state = SlotState::Retired;
		return;
	}
	slot->state = SlotState::Free;
	slot->completion = {};
	slot->synchronousComplete = false;
}

oa::Status TimerRegistry::begin(TimerSlot inSlot, oavk::Stream* inStream) {
	std::lock_guard<std::mutex> lock(mutex_);
	auto* slot = resolve_(inSlot);
	if (not open_ or slot == nullptr) {
		return oa::Status::error(oa::StatusCode::FailedPrecondition,
			"device timer slot is not active");
	}
	if (inStream == nullptr) {
		return oa::Status::invalidArgument("device timer requires a command stream");
	}
	if (slot->state != SlotState::Ready) {
		return oa::Status::error(oa::StatusCode::FailedPrecondition,
			"device timer requires commit before reuse");
	}
	slot->timestamp.reset(inStream);
	slot->timestamp.writeTimestamp(inStream);
	slot->state = SlotState::Recording;
	return oa::Status::ok();
}

oa::Status TimerRegistry::end(TimerSlot inSlot, oavk::Stream* inStream) {
	std::lock_guard<std::mutex> lock(mutex_);
	auto* slot = resolve_(inSlot);
	if (not open_ or slot == nullptr or inStream == nullptr) {
		return oa::Status::error(oa::StatusCode::FailedPrecondition,
			"device timer has no matching Begin");
	}
	if (slot->state != SlotState::Recording) {
		return oa::Status::error(oa::StatusCode::FailedPrecondition,
			"device timer End requires an active region");
	}
	slot->timestamp.writeTimestamp(inStream);
	slot->state = SlotState::Recorded;
	return oa::Status::ok();
}

oa::Status TimerRegistry::attach(
	TimerSlot inSlot, const oa::Event& inCompletion)
{
	std::lock_guard<std::mutex> lock(mutex_);
	auto* slot = resolve_(inSlot);
	if (slot == nullptr or slot->state != SlotState::Recorded) {
		return oa::Status::error(oa::StatusCode::FailedPrecondition,
			"device timer completion requires a recorded region");
	}
	if (not inCompletion.isValid()) {
		return oa::Status::invalidArgument(
			"device timer completion requires a valid event");
	}
	slot->completion = inCompletion;
	slot->synchronousComplete = false;
	slot->state = SlotState::Submitted;
	return oa::Status::ok();
}

oa::Status TimerRegistry::markSynchronousComplete(TimerSlot inSlot) {
	std::lock_guard<std::mutex> lock(mutex_);
	auto* slot = resolve_(inSlot);
	if (slot == nullptr or slot->state != SlotState::Recorded) {
		return oa::Status::error(oa::StatusCode::FailedPrecondition,
			"device timer completion requires a recorded region");
	}
	slot->completion = {};
	slot->synchronousComplete = true;
	slot->state = SlotState::Submitted;
	return oa::Status::ok();
}

void TimerRegistry::cancel(TimerSlot inSlot) noexcept {
	std::lock_guard<std::mutex> lock(mutex_);
	auto* slot = resolve_(inSlot);
	if (slot == nullptr) return;
	if (slot->state == SlotState::Recording
		or slot->state == SlotState::Recorded) {
		slot->state = SlotState::Ready;
	}
	slot->completion = {};
	slot->synchronousComplete = false;
}

oa::Result<oa::F64> TimerRegistry::commitMilliseconds(
	TimerSlot inSlot, const oa::Engine& inEngine)
{
	std::lock_guard<std::mutex> lock(mutex_);
	if (owner_ != &inEngine) {
		return oa::Status::invalidArgument(
			"device timer commit requires its owning engine");
	}
	auto* slot = resolve_(inSlot);
	if (not open_ or slot == nullptr or slot->state != SlotState::Submitted) {
		return oa::Status::error(oa::StatusCode::FailedPrecondition,
			"device timer has no submitted measurement to commit");
	}
	if (not slot->synchronousComplete
		and (not slot->completion.isValid() or not slot->completion.isComplete())) {
		return oa::Status::error(oa::StatusCode::FailedPrecondition,
			"device timer commit requires the submission event to complete");
	}
	const oa::Status readback = slot->timestamp.readback(inEngine);
	if (not readback.isOk()) return readback;
	const oa::F64 elapsed = slot->timestamp.elapsedMs(0U, 1U);
	slot->state = SlotState::Ready;
	slot->completion = {};
	slot->synchronousComplete = false;
	return elapsed;
}

oa::Bool TimerRegistry::isPending(TimerSlot inSlot) const noexcept {
	std::lock_guard<std::mutex> lock(mutex_);
	const auto* slot = resolve_(inSlot);
	return slot != nullptr and (
		slot->state == SlotState::Recording
		or slot->state == SlotState::Recorded
		or slot->state == SlotState::Submitted
		or slot->state == SlotState::Retired);
}

oa::Status TimerRegistry::close(const oavk::Device& inDevice) {
	std::lock_guard<std::mutex> lock(mutex_);
	if (not open_) return oa::Status::ok();
	for (auto& slot : slots_) {
		if (slot and slot->timestamp.isInitialized()) {
			slot->timestamp.destroyDevice_(inDevice);
		}
		if (slot) {
			slot->state = SlotState::Free;
			slot->completion = {};
		}
	}
	open_ = false;
	owner_ = nullptr;
	return oa::Status::ok();
}

} // namespace oa
