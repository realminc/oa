#include <oa/runtime/timer.h>

#include "timerAccess.h"
#include "timerRegistry.h"
#include "engine/engineAccess.h"
#include <oa/core/time.h>
#include <oa/runtime/engine.h>

class oa::Timer::Impl {
public:
	~Impl() { releaseDevice(); }

	oa::TimerDomain domain = oa::TimerDomain::Auto;
	oa::String name;
	oa::WeakPtr<oa::TimerRegistry> registry;
	oa::TimerSlot slot;
	oa::Timestamp hostStart;
	oa::Timestamp hostEnd;
	oa::Bool hostRunning = false;
	oa::Bool hostComplete = false;
	oa::F64 lastMilliseconds = 0.0;
	oa::F64 lastUnits = 1.0;

	void releaseDevice() noexcept {
		if (slot.isValid()) {
			if (auto owner = registry.lock()) owner->release(slot);
		}
		registry.reset();
		slot = {};
	}
};

oa::Timer::Timer(oa::TimerDomain inDomain, oa::StringView inName)
	: impl_(oa::makeUnique<Impl>())
{
	impl_->domain = inDomain;
	impl_->name = oa::String(inName);
}

oa::Timer::~Timer() {
}

oa::Status oa::Timer::init(oa::Engine& inEngine, oa::StringView inName) {
	if (not impl_) impl_ = oa::makeUnique<Impl>();
	if (not inName.empty()) impl_->name = oa::String(inName);
	if (impl_->domain == oa::TimerDomain::Auto) {
		impl_->domain = oa::TimerDomain::Device;
	}
	if (impl_->domain == oa::TimerDomain::Host) return oa::Status::ok();
	if (not inEngine.isReady()) {
		return oa::Status::error(oa::StatusCode::FailedPrecondition,
			"oa::Timer::init requires a ready engine");
	}
	impl_->releaseDevice();
	auto registry = oa::EngineAccess::get(inEngine).timerRegistry_;
	if (not registry) {
		return oa::Status::error(oa::StatusCode::FailedPrecondition,
			"engine timer registry is unavailable");
	}
	auto slot = registry->acquire();
	if (not slot.isOk()) return slot.getStatus();
	impl_->registry = oa::WeakPtr<oa::TimerRegistry>(registry);
	impl_->slot = oa::move(slot).getValue();
	return oa::Status::ok();
}

oa::Status oa::Timer::begin() {
	if (not impl_) {
		return oa::Status::error(oa::StatusCode::FailedPrecondition,
			"oa::Timer has no state");
	}
	if (impl_->domain == oa::TimerDomain::Auto) {
		impl_->domain = oa::TimerDomain::Host;
	}
	if (impl_->domain != oa::TimerDomain::Host) {
		return oa::Status::error(oa::StatusCode::FailedPrecondition,
			"device timer regions are recorded by oa::Engine::submit");
	}
	if (impl_->hostRunning) {
		return oa::Status::error(oa::StatusCode::FailedPrecondition,
			"host timer Begin requires the previous region to end");
	}
	impl_->hostStart = oa::Timestamp::now();
	impl_->hostRunning = true;
	impl_->hostComplete = false;
	return oa::Status::ok();
}

oa::Status oa::Timer::end() {
	if (not impl_ or impl_->domain != oa::TimerDomain::Host
		or not impl_->hostRunning) {
		return oa::Status::error(oa::StatusCode::FailedPrecondition,
			"host timer End requires a matching Begin");
	}
	impl_->hostEnd = oa::Timestamp::now();
	impl_->hostRunning = false;
	impl_->hostComplete = true;
	return oa::Status::ok();
}

oa::Result<oa::F64> oa::Timer::commit(
	const oa::Engine& inEngine, oa::F64 inUnits)
{
	if (not impl_ or impl_->domain != oa::TimerDomain::Device
		or not impl_->slot.isValid()) {
		return oa::Status::error(oa::StatusCode::FailedPrecondition,
			"device timer commit requires init and a submitted region");
	}
	auto registry = impl_->registry.lock();
	if (not registry) {
		return oa::Status::error(oa::StatusCode::FailedPrecondition,
			"device timer's engine is no longer available");
	}
	auto elapsed = registry->commitMilliseconds(impl_->slot, inEngine);
	if (not elapsed.isOk()) return elapsed.getStatus();
	impl_->lastMilliseconds = elapsed.getValue();
	impl_->lastUnits = inUnits;
	return impl_->lastMilliseconds;
}

oa::Result<oa::F64> oa::Timer::commit(oa::F64 inUnits) {
	if (not impl_ or impl_->domain != oa::TimerDomain::Host
		or not impl_->hostComplete) {
		return oa::Status::error(oa::StatusCode::FailedPrecondition,
			"host timer commit requires a completed Begin/End region");
	}
	impl_->lastMilliseconds = (impl_->hostEnd - impl_->hostStart).toMs();
	impl_->lastUnits = inUnits;
	impl_->hostComplete = false;
	return impl_->lastMilliseconds;
}

oa::F64 oa::Timer::elapsedMs() const noexcept {
	return impl_ ? impl_->lastMilliseconds : 0.0;
}

oa::F64 oa::Timer::throughput() const noexcept {
	if (not impl_ or impl_->lastMilliseconds <= 0.0) return 0.0;
	return impl_->lastUnits / (impl_->lastMilliseconds / 1000.0);
}

oa::TimerDomain oa::Timer::domain() const noexcept {
	return impl_ ? impl_->domain : oa::TimerDomain::Auto;
}

oa::Bool oa::Timer::isHost() const noexcept {
	return domain() == oa::TimerDomain::Host;
}

oa::Bool oa::Timer::isDevice() const noexcept {
	return domain() == oa::TimerDomain::Device;
}

oa::Bool oa::Timer::isInitialized() const noexcept {
	if (not impl_) return false;
	if (impl_->domain == oa::TimerDomain::Host) return true;
	return impl_->slot.isValid() and not impl_->registry.expired();
}

oa::Bool oa::Timer::isPending() const noexcept {
	if (not impl_) return false;
	if (impl_->domain == oa::TimerDomain::Host) return impl_->hostRunning;
	if (auto registry = impl_->registry.lock()) {
		return registry->isPending(impl_->slot);
	}
	return false;
}

oa::StringView oa::Timer::getName() const noexcept {
	return impl_ ? oa::StringView(impl_->name) : oa::StringView{};
}

oa::Timer::Timer(oa::Timer&& inOther) noexcept = default;
oa::Timer& oa::Timer::operator=(oa::Timer&& inOther) noexcept = default;

oa::Status oa::TimerAccess::beginDevice(
	oa::Timer& inTimer, oavk::Stream* inStream)
{
	if (not inTimer.impl_ or inTimer.impl_->domain != oa::TimerDomain::Device) {
		return oa::Status::error(oa::StatusCode::FailedPrecondition,
			"oa::Engine::submit timer must be initialized for Device timing");
	}
	auto registry = inTimer.impl_->registry.lock();
	if (not registry) {
		return oa::Status::error(oa::StatusCode::FailedPrecondition,
			"device timer's engine is no longer available");
	}
	return registry->begin(inTimer.impl_->slot, inStream);
}

oa::Status oa::TimerAccess::endDevice(
	oa::Timer& inTimer, oavk::Stream* inStream)
{
	if (not inTimer.impl_) {
		return oa::Status::error(oa::StatusCode::FailedPrecondition,
			"device timer has no state");
	}
	auto registry = inTimer.impl_->registry.lock();
	return registry
		? registry->end(inTimer.impl_->slot, inStream)
		: oa::Status::error(oa::StatusCode::FailedPrecondition,
			"device timer's engine is no longer available");
}

oa::Status oa::TimerAccess::attachCompletion(
	oa::Timer& inTimer, const oa::Event& inCompletion)
{
	if (not inTimer.impl_) {
		return oa::Status::error(oa::StatusCode::FailedPrecondition,
			"device timer has no state");
	}
	auto registry = inTimer.impl_->registry.lock();
	return registry
		? registry->attach(inTimer.impl_->slot, inCompletion)
		: oa::Status::error(oa::StatusCode::FailedPrecondition,
			"device timer's engine is no longer available");
}

oa::Status oa::TimerAccess::completeSynchronously(oa::Timer& inTimer) {
	if (not inTimer.impl_) {
		return oa::Status::error(oa::StatusCode::FailedPrecondition,
			"device timer has no state");
	}
	auto registry = inTimer.impl_->registry.lock();
	return registry
		? registry->markSynchronousComplete(inTimer.impl_->slot)
		: oa::Status::error(oa::StatusCode::FailedPrecondition,
			"device timer's engine is no longer available");
}

void oa::TimerAccess::cancelDevice(oa::Timer& inTimer) noexcept {
	if (not inTimer.impl_) return;
	if (auto registry = inTimer.impl_->registry.lock()) {
		registry->cancel(inTimer.impl_->slot);
	}
}
