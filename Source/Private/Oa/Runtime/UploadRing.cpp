#include <Oa/Runtime/UploadRing.h>

#include <Oa/Core/Memory.h>
#include <Oa/Core/Log.h>
#include <Oa/Core/Std.h>
#include <Oa/Runtime/Allocator.h>
#include <Oa/Runtime/Engine.h>
#include <Oa/Runtime/Stream.h>

#include "UploadRingRetirement.h"

#include <algorithm>

namespace {

OaU64 AlignUp(OaU64 InValue, OaU64 InAlignment) {
	if (InAlignment <= 1) return InValue;
	const OaU64 remainder = InValue % InAlignment;
	return remainder == 0 ? InValue : InValue + (InAlignment - remainder);
}

struct OaPendingUploadCopy {
	OaVkBuffer Dst;
	OaBufferCopyRegion Region;
};

} // namespace

struct OaUploadRing::Impl {
	OaEngine* Engine = nullptr;
	OaVkBuffer Staging;
	OaUploadRingConfig Config;
	OaU64 FrameCapacity = 0;
	OaU32 NextFrame = 0;
	OaU32 ActiveFrame = 0;
	OaBool BatchOpen = false;
	OaCompletionToken LastCompletion;
	OaVec<OaUniquePtr<OaUploadFrame>> Frames;
	OaVec<OaPendingUploadCopy> Copies;
};

OaUploadRing::OaUploadRing(OaUploadRing&&) noexcept = default;
OaUploadRing& OaUploadRing::operator=(OaUploadRing&& InOther) noexcept {
	if (this != &InOther) {
		Abandon_();
		Impl_ = OaStdMove(InOther.Impl_);
	}
	return *this;
}
OaUploadRing::~OaUploadRing() { Abandon_(); }

OaResult<OaUploadRing> OaUploadRing::Create(OaEngine& InEngine,	const OaUploadRingConfig& InConfig) {
	if (InConfig.CapacityBytes == 0 || InConfig.FramesInFlight < 2 || InConfig.Alignment == 0) {
		return OaStatus::InvalidArgument("UploadRing: capacity/alignment must be non-zero and frames must be >= 2");
	}

	const OaU64 frameCapacity =	(InConfig.CapacityBytes / InConfig.FramesInFlight / InConfig.Alignment)	* InConfig.Alignment;
	if (frameCapacity == 0) {
		return OaStatus::InvalidArgument("UploadRing: capacity is too small for frame count");
	}

	OaUploadRing result;
	result.Impl_ = OaMakeUniquePtr<Impl>();
	auto& impl = *result.Impl_;
	impl.Engine = &InEngine;
	impl.Config = InConfig;
	impl.FrameCapacity = frameCapacity;
	impl.Config.CapacityBytes = frameCapacity * InConfig.FramesInFlight;

	auto staging = InEngine.AllocBuffer(impl.Config.CapacityBytes);
	if (!staging) return staging.GetStatus();
	impl.Staging = OaStdMove(*staging);
	if (impl.Staging.MappedPtr == nullptr) {
		InEngine.FreeBuffer(impl.Staging);
		return OaStatus::Error(OaStatusCode::FailedPrecondition, "UploadRing: staging allocation is not mapped");
	}

	impl.Frames.Reserve(InConfig.FramesInFlight);
	// OA buffers are currently exclusive-sharing allocations. A dedicated
	// transfer family would therefore require an explicit release/acquire
	// ownership pair in every downstream consumer. Until that contract is part
	// of the graph, use the compute queue when the families differ. This keeps
	// uploads asynchronous without relying on undefined cross-family ownership.
	const bool sharedQueueFamily =
		InEngine.Device.Queues.TransferQueueFamily
			== InEngine.Device.Queues.ComputeQueueFamily;
	const OaU32 queueFamily = sharedQueueFamily
		? InEngine.Device.Queues.TransferQueueFamily
		: InEngine.Device.Queues.ComputeQueueFamily;
	void* queue = sharedQueueFamily
		? InEngine.Device.Queues.TransferQueue
		: InEngine.Device.Queues.ComputeQueue;
	for (OaU32 index = 0; index < InConfig.FramesInFlight; ++index) {
		auto stream = OaVkStream::Create(
			InEngine.Device,
			queueFamily,
			queue);
		if (!stream) {
			result.Destroy();
			return stream.GetStatus();
		}
		auto frame = OaMakeUniquePtr<OaUploadFrame>();
		frame->Stream = OaStdMove(*stream);
		frame->Begin = static_cast<OaU64>(index) * frameCapacity;
		frame->End = frame->Begin + frameCapacity;
		frame->Cursor = frame->Begin;
		impl.Frames.PushBack(OaStdMove(frame));
	}
	return result;
}

OaStatus OaUploadRing::BeginBatch() {
	if (!Impl_ || !Impl_->Engine) {
		return OaStatus::Error(OaStatusCode::FailedPrecondition, "UploadRing is not initialized");
	}
	if (Impl_->BatchOpen) {
		return OaStatus::Error(OaStatusCode::FailedPrecondition, "UploadRing batch is already open");
	}
	Impl_->ActiveFrame = Impl_->NextFrame % Impl_->Config.FramesInFlight;
	auto& frame = *Impl_->Frames[Impl_->ActiveFrame];
	OA_RETURN_IF_ERROR(frame.Stream.Begin(Impl_->Engine->Device));
	frame.Cursor = frame.Begin;
	Impl_->Copies.Clear();
	Impl_->BatchOpen = true;
	return OaStatus::Ok();
}

OaResult<OaUploadSlice> OaUploadRing::Reserve(OaU64 InSize, OaU64 InAlignment) {
	if (!Impl_ || !Impl_->BatchOpen) {
		return OaStatus::Error(OaStatusCode::FailedPrecondition, "UploadRing batch is not open");
	}
	if (InSize == 0) return OaStatus::InvalidArgument("UploadRing: zero-byte reservation");
	const OaU64 alignment = InAlignment == 0 ? Impl_->Config.Alignment : InAlignment;
	auto& frame = *Impl_->Frames[Impl_->ActiveFrame];
	const OaU64 offset = AlignUp(frame.Cursor, std::max<OaU64>(alignment, 4));
	if (offset > frame.End || InSize > frame.End - offset) {
		return OaStatus::Error(OaStatusCode::ResourceExhausted,
			"UploadRing: active frame capacity exhausted; submit or increase capacity");
	}
	frame.Cursor = offset + InSize;
	auto* base = static_cast<OaByte*>(Impl_->Staging.MappedPtr);
	return OaUploadSlice{.Mapped = base + offset, .Offset = offset, .Size = InSize};
}

OaStatus OaUploadRing::EnqueueCopy(
	const OaUploadSlice& InSlice,
	const OaVkBuffer& InDst,
	OaU64 InDstOffset
) {
	if (!Impl_ || !Impl_->BatchOpen) {
		return OaStatus::Error(OaStatusCode::FailedPrecondition, "UploadRing batch is not open");
	}
	const auto& frame = *Impl_->Frames[Impl_->ActiveFrame];
	auto* base = static_cast<OaByte*>(Impl_->Staging.MappedPtr);
	if (!InSlice.IsValid() || InDst.Buffer == nullptr
		|| InSlice.Offset < frame.Begin || InSlice.Offset > frame.Cursor
		|| InSlice.Size > frame.Cursor - InSlice.Offset
		|| InSlice.Mapped != base + InSlice.Offset
		|| InDstOffset > InDst.Size || InSlice.Size > InDst.Size - InDstOffset) {
		return OaStatus::InvalidArgument("UploadRing: invalid copy range");
	}
	if ((InSlice.Offset & 3U) != 0 || (InDstOffset & 3U) != 0) {
		return OaStatus::InvalidArgument("UploadRing: Vulkan buffer-copy offsets must be 4-byte aligned");
	}
	Impl_->Copies.PushBack(OaPendingUploadCopy{
		.Dst = InDst,
		.Region = {
			.SrcOffset = InSlice.Offset,
			.DstOffset = InDstOffset,
			.Size = InSlice.Size,
		},
	});
	return OaStatus::Ok();
}

OaStatus OaUploadRing::Upload(
	const OaVkBuffer& InDst,
	OaU64 InDstOffset,
	const void* InData,
	OaU64 InSize,
	OaU64 InAlignment
) {
	if (InData == nullptr) return OaStatus::InvalidArgument("UploadRing: source is null");
	auto slice = Reserve(InSize, InAlignment);
	if (!slice) return slice.GetStatus();
	OaMemcpy(slice->Mapped, InData, static_cast<OaUsize>(InSize));
	return EnqueueCopy(*slice, InDst, InDstOffset);
}

OaResult<OaCompletionToken> OaUploadRing::Submit() {
	if (!Impl_ || !Impl_->BatchOpen) {
		return OaStatus::Error(OaStatusCode::FailedPrecondition, "UploadRing batch is not open");
	}
	auto& frame = *Impl_->Frames[Impl_->ActiveFrame];
	if (!Impl_->Copies.Empty() && !Impl_->Engine->Allocator.FlushHostBuffer(
		Impl_->Staging, frame.Begin, frame.Cursor - frame.Begin)) {
		return OaStatus::Error(OaStatusCode::VulkanError, "UploadRing: mapped flush failed");
	}

	// Collapse adjacent jobs targeting the same destination into one
	// vkCmdCopyBuffer call with multiple regions. Different destinations still
	// share this command buffer and this single queue submission.
	OaUsize begin = 0;
	while (begin < Impl_->Copies.Size()) {
		const OaVkBuffer& dst = Impl_->Copies[begin].Dst;
		OaVec<OaBufferCopyRegion> regions;
		OaU64 barrierBegin = UINT64_MAX;
		OaU64 barrierEnd = 0U;
		OaUsize end = begin;
		while (end < Impl_->Copies.Size()
			&& Impl_->Copies[end].Dst.Buffer == dst.Buffer) {
			const OaBufferCopyRegion& region = Impl_->Copies[end].Region;
			regions.PushBack(region);
			barrierBegin = std::min(barrierBegin, region.DstOffset);
			barrierEnd = std::max(barrierEnd, region.DstOffset + region.Size);
			++end;
		}
		frame.Stream.RecordCopyBufferRegions(
			Impl_->Staging, dst,
			OaSpan<const OaBufferCopyRegion>(regions.Data(), regions.Size())
		);
		frame.Stream.RecordTransferWriteBarrier(dst, barrierBegin, barrierEnd - barrierBegin);
		begin = end;
	}
	// Ring arenas protect staging reuse, but destination buffers can repeat
	// across adjacent batches. Queue order alone is not a Vulkan memory
	// dependency, so chain submissions through the previous upload completion.
	// This stays GPU-side and preserves host asynchrony.
	if (Impl_->LastCompletion.IsValid()) {
		const OaVkTimelineWait wait = Impl_->LastCompletion.TimelineWait();
		OA_RETURN_IF_ERROR(frame.Stream.SubmitWithDependencies(*Impl_->Engine, OaSpan<const OaVkTimelineWait>(&wait, 1)));
	} else {
		OA_RETURN_IF_ERROR(frame.Stream.Submit(*Impl_->Engine));
	}
	Impl_->LastCompletion = frame.Stream.Completion(Impl_->Engine->Device);
	Impl_->BatchOpen = false;
	Impl_->Copies.Clear();
	++Impl_->NextFrame;
	return Impl_->LastCompletion;
}

OaStatus OaUploadRing::Wait() {
	return Impl_ && Impl_->LastCompletion.IsValid()
		? Impl_->LastCompletion.Wait()
		: OaStatus::Ok();
}

OaStatus OaUploadRing::Close() {
	if (!Impl_) return OaStatus::Ok();
	OaStatus firstError = OaStatus::Ok();
	const auto retainError = [&firstError](const OaStatus& InStatus) {
		if (firstError.IsOk() && !InStatus.IsOk()) firstError = InStatus;
	};
	if (Impl_->Engine && Impl_->Engine->Device.Device) {
		if (Impl_->BatchOpen) {
			auto& active = *Impl_->Frames[Impl_->ActiveFrame];
			retainError(active.Stream.ResetUnsubmitted(Impl_->Engine->Device));
			Impl_->BatchOpen = false;
			Impl_->Copies.Clear();
		}
		for (auto& frame : Impl_->Frames) {
			if (!frame) continue;
			retainError(frame->Stream.Synchronize(Impl_->Engine->Device));
			frame->Stream.Destroy(Impl_->Engine->Device);
		}
		Impl_->Engine->FreeBuffer(Impl_->Staging);
	}
	Impl_.reset();
	return firstError;
}

void OaUploadRing::Destroy() {
	if (const auto status = Close(); !status.IsOk()) {
		OA_LOG_ERROR(OaLogComponent::Core,
			"OaUploadRing::Destroy: close failed: %s",
			status.GetMessage().c_str()
		);
	}
}

void OaUploadRing::Abandon_() noexcept {
	if (!Impl_) return;
	auto* engine = Impl_->Engine;
	if (engine == nullptr || engine->Device.Device == nullptr) {
		Impl_.reset();
		return;
	}
	if (Impl_->BatchOpen) {
		auto& active = *Impl_->Frames[Impl_->ActiveFrame];
		if (const auto status = active.Stream.ResetUnsubmitted(engine->Device);
			not status.IsOk())
		{
			OA_LOG_ERROR(
				OaLogComponent::Core,
				"OaUploadRing abandonment failed to cancel open batch: %s",
				status.GetMessage().c_str()
			);
		}
		Impl_->BatchOpen = false;
		Impl_->Copies.Clear();
	}

	auto retired = OaMakeUniquePtr<OaRetiredUploadRing>();
	retired->Staging = OaStdMove(Impl_->Staging);
	retired->Frames = OaStdMove(Impl_->Frames);
	engine->RetireUploadRing(OaStdMove(retired));
	Impl_.reset();
}

OaU64 OaUploadRing::CapacityBytes() const noexcept {
	return Impl_ ? Impl_->Config.CapacityBytes : 0;
}
OaU64 OaUploadRing::FrameCapacityBytes() const noexcept {
	return Impl_ ? Impl_->FrameCapacity : 0;
}
OaU64 OaUploadRing::BytesUsed() const noexcept {
	if (!Impl_ || !Impl_->BatchOpen) return 0;
	const auto& frame = *Impl_->Frames[Impl_->ActiveFrame];
	return frame.Cursor - frame.Begin;
}
OaU32 OaUploadRing::PendingCopyCount() const noexcept {
	return Impl_ ? static_cast<OaU32>(Impl_->Copies.Size()) : 0;
}
bool OaUploadRing::IsBatchOpen() const noexcept {
	return Impl_ && Impl_->BatchOpen;
}
