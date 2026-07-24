#include "GraphicsStream.h"

#include <Oa/Core/Log.h>
#include <Oa/Runtime/Engine.h>
#include <Oa/Runtime/Stream.h>

OaResult<OaGraphicsStreamLease> OaGraphicsStreamLease::Acquire(
	OaEngine& InEngine)
{
	if (not InEngine.IsReady()) {
		return OaStatus::Error(
			OaStatusCode::FailedPrecondition,
			"graphics stream acquisition requires a ready engine");
	}
	if (InEngine.Device.Queues.GraphicsQueue == nullptr
		or InEngine.Device.Queues.GraphicsQueueFamily
			== OaVkEnumerationIndexUnset) {
		return OaStatus::Error(
			OaStatusCode::Unavailable,
			"engine was not created with graphics capability");
	}

	InEngine.CollectRetiredGraphicsStreams_();
	std::lock_guard<std::mutex> lock(InEngine.GraphicsStreamPoolMutex_);
	OaU32 slotIndex = static_cast<OaU32>(
		InEngine.GraphicsStreamPool_.Size());
	for (OaU32 index = 0;
		index < static_cast<OaU32>(InEngine.GraphicsStreamPool_.Size()); ++index) {
		if (InEngine.GraphicsStreamPool_[index].State
			== OaEngine::OaGraphicsStreamSlotState::Free) {
			slotIndex = index;
			break;
		}
	}

	if (slotIndex == static_cast<OaU32>(
			InEngine.GraphicsStreamPool_.Size())) {
		auto created = OaVkStream::Create(
			InEngine.Device,
			InEngine.Device.Queues.GraphicsQueueFamily,
			InEngine.Device.Queues.GraphicsQueue);
		if (not created.IsOk()) return created.GetStatus();
		OaEngine::OaGraphicsStreamSlot slot;
		slot.Stream = OaMakeUniquePtr<OaVkStream>(OaStdMove(*created));
		InEngine.GraphicsStreamPool_.PushBack(OaStdMove(slot));
	}

	auto& slot = InEngine.GraphicsStreamPool_[slotIndex];
	if (not slot.Stream) {
		slot.State = OaEngine::OaGraphicsStreamSlotState::Quarantined;
		return OaStatus::Error(
			OaStatusCode::Internal,
			"graphics stream pool contains an empty slot");
	}
	const OaStatus beginStatus = slot.Stream->Begin(InEngine.Device);
	if (not beginStatus.IsOk()) {
		slot.State = OaEngine::OaGraphicsStreamSlotState::Quarantined;
		return beginStatus;
	}
	++slot.Generation;
	if (slot.Generation == 0) ++slot.Generation;
	slot.Completion = {};
	slot.State = OaEngine::OaGraphicsStreamSlotState::Recording;
	return OaGraphicsStreamLease(InEngine, slotIndex, slot.Generation);
}

OaGraphicsStreamLease::OaGraphicsStreamLease(
	OaEngine& InEngine,
	OaU32 InSlot,
	OaU64 InGeneration) noexcept
	: Engine_(&InEngine)
	, Slot_(InSlot)
	, Generation_(InGeneration)
{}

OaGraphicsStreamLease::OaGraphicsStreamLease(
	OaGraphicsStreamLease&& InOther) noexcept
	: Engine_(InOther.Engine_)
	, Slot_(InOther.Slot_)
	, Generation_(InOther.Generation_)
{
	InOther.Reset_();
}

OaGraphicsStreamLease& OaGraphicsStreamLease::operator=(
	OaGraphicsStreamLease&& InOther) noexcept
{
	if (this == &InOther) return *this;
	if (Engine_ != nullptr) {
		const OaStatus status = Close();
		if (not status.IsOk()) {
			OA_LOG_ERROR(OaLogComponent::Core,
				"graphics stream lease move-close failed: %s",
				status.ToString().c_str());
		}
	}
	Engine_ = InOther.Engine_;
	Slot_ = InOther.Slot_;
	Generation_ = InOther.Generation_;
	InOther.Reset_();
	return *this;
}

OaGraphicsStreamLease::~OaGraphicsStreamLease()
{
	if (Engine_ == nullptr) return;
	const OaStatus status = Close();
	if (not status.IsOk()) {
		OA_LOG_ERROR(OaLogComponent::Core,
			"graphics stream lease abandonment failed: %s",
			status.ToString().c_str());
	}
}

bool OaGraphicsStreamLease::IsValid() const noexcept
{
	return Engine_ != nullptr and Generation_ != 0;
}

OaVkStream* OaGraphicsStreamLease::GetStream() noexcept
{
	return Engine_ != nullptr
		? Engine_->GraphicsStreamForLease_(Slot_, Generation_)
		: nullptr;
}

OaResult<OaEvent> OaGraphicsStreamLease::Submit()
{
	return Submit(OaSpan<const OaEvent>{});
}

OaResult<OaEvent> OaGraphicsStreamLease::Submit(
	OaSpan<const OaEvent> InDependencies)
{
	if (Engine_ == nullptr) {
		return OaStatus::Error(
			OaStatusCode::FailedPrecondition,
			"graphics stream lease is closed");
	}
	auto result = Engine_->SubmitGraphicsStream_(
		Slot_, Generation_, InDependencies);
	return result;
}

OaStatus OaGraphicsStreamLease::Cancel()
{
	if (Engine_ == nullptr) {
		return OaStatus::Error(
			OaStatusCode::FailedPrecondition,
			"graphics stream lease is closed");
	}
	const OaStatus status = Engine_->CancelGraphicsStream_(Slot_, Generation_);
	if (status.IsOk()) Reset_();
	return status;
}

OaStatus OaGraphicsStreamLease::Recycle(const OaEvent& InCompletion)
{
	if (Engine_ == nullptr) {
		return OaStatus::Error(
			OaStatusCode::FailedPrecondition,
			"graphics stream lease is closed");
	}
	const OaStatus status = Engine_->RecycleGraphicsStream_(
		Slot_, Generation_, InCompletion);
	if (status.IsOk()) Reset_();
	return status;
}

OaStatus OaGraphicsStreamLease::Close()
{
	if (Engine_ == nullptr) return OaStatus::Ok();
	OaEngine* engine = Engine_;
	const OaU32 slot = Slot_;
	const OaU64 generation = Generation_;
	Reset_();
	return engine->AbandonGraphicsStream_(slot, generation);
}

void OaGraphicsStreamLease::Reset_() noexcept
{
	Engine_ = nullptr;
	Slot_ = 0;
	Generation_ = 0;
}
