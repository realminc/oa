// OaVideo — implementation. See header for usage.

#include <Oa/Vision/Video.h>
#include <Oa/Vision/FnVideo.h>

#include <Oa/Core/Log.h>
#include <Oa/Runtime/Engine.h>
#include "Oa/Runtime/Engine/BorrowedServiceRetirement.h"

#include <algorithm>
#include <limits>

OaResult<OaVideo> OaVideo::Open(OaEngine& InEngine, const OaVideoConfig& InCfg)
{
	const OaString& uri = InCfg.Uri.Empty() ? InCfg.Path : InCfg.Uri;
	if (uri.Empty()) {
		return OaStatus::Error(OaStatusCode::InvalidArgument, "OaVideo: empty URI");
	}

	auto streamResult = OaVideoStream::Open(uri, InCfg.StreamOptions);
	if (not streamResult.IsOk()) {
		return streamResult.GetStatus();
	}

	auto profile = streamResult->GetVideoProfile();
	profile.MaxDpbSlots = InCfg.MaxDpbSlots;
	auto decoderResult = OaVideoDecoder::Create(InEngine, profile);
	if (not decoderResult.IsOk()) {
		return decoderResult.GetStatus();
	}

	OaVideo it;
	it.Cfg_     = InCfg;
	it.Engine_  = &InEngine;
	it.Stream_.Emplace(OaStdMove(*streamResult));
	it.Decoder_.Emplace(OaStdMove(*decoderResult));
	it.Playing_ = InCfg.StartPlaying;
	it.StreamFormatGeneration_ = it.Stream_->FormatGeneration();
	it.StreamReconnectCount_ = it.Stream_->GetStats().ReconnectCount;
	if (InCfg.Audio) {
		OaAudioStreamConfig audioConfig;
		audioConfig.Uri = uri;
		audioConfig.Loop = InCfg.Loop;
		auto audio = OaAudioStream::Open(InEngine, audioConfig);
		if (audio.IsOk()) {
			it.Audio_.Emplace(OaStdMove(*audio));
			if (InCfg.StartPlaying) (void)it.Audio_->Play();
		} else if (audio.GetStatus().GetCode() != OaStatusCode::NotFound) {
			OA_LOG_WARN(OaLogComponent::App,
				"OaVideo audio disabled: %s", audio.GetStatus().GetMessage().c_str());
		}
	}
	it.DisplayPts_.Reserve(it.Stream_->Samples_.Size());
	for (const auto& sample : it.Stream_->Samples_) {
		it.DisplayPts_.PushBack(
			sample.Dts + static_cast<OaU64>(sample.CtsOffset));
	}
	if (not it.DisplayPts_.Empty()) {
		std::sort(it.DisplayPts_.Data(),
			it.DisplayPts_.Data() + it.DisplayPts_.Size());
	}

	OaU32 fps = it.Stream_->GetInfo().FrameRate;
	if (InCfg.FrameRateOverride > 0.0F) {
		it.FrameIntervalMs_ = 1000.0F / InCfg.FrameRateOverride;
	} else if (fps > 0U) {
		it.FrameIntervalMs_ = 1000.0F / static_cast<OaF32>(fps);
	} else {
		it.FrameIntervalMs_ = 1000.0F / 30.0F;
	}

	OaStatus first = it.StepForward();
	if (not first.IsOk()) {
		return first;
	}

	return OaResult<OaVideo>(OaStdMove(it));
}

OaResult<OaVideo> OaVideo::Create(OaEngine& InEngine, const OaVideoConfig& InCfg)
{
	return Open(InEngine, InCfg);
}

OaVideo::OaVideo(OaVideo&& InOther) noexcept
	: Cfg_(OaStdMove(InOther.Cfg_))
	, Engine_(InOther.Engine_)
	, Stream_(OaStdMove(InOther.Stream_))
	, Decoder_(OaStdMove(InOther.Decoder_))
	, Audio_(OaStdMove(InOther.Audio_))
	, Frame_(InOther.Frame_)
	, FrameIntervalMs_(InOther.FrameIntervalMs_)
	, Accumulator_(InOther.Accumulator_)
	, Playing_(InOther.Playing_)
	, ReachedEos_(InOther.ReachedEos_)
	, StreamEosCurrent_(InOther.StreamEosCurrent_)
	, Reorder_(OaStdMove(InOther.Reorder_))
	, DisplayPts_(OaStdMove(InOther.DisplayPts_))
	, RgbaPool_(OaStdMove(InOther.RgbaPool_))
	, RgbaPoolBusy_(OaStdMove(InOther.RgbaPoolBusy_))
	, RgbaPoolConsumerEvents_(OaStdMove(InOther.RgbaPoolConsumerEvents_))
	, Index_(InOther.Index_)
	, StreamFormatGeneration_(InOther.StreamFormatGeneration_)
	, StreamReconnectCount_(InOther.StreamReconnectCount_)
{
	InOther.Engine_      = nullptr;
	InOther.Frame_       = {};
	InOther.Accumulator_ = 0.0F;
	InOther.Playing_     = false;
	InOther.ReachedEos_  = false;
	InOther.StreamEosCurrent_ = false;
}

OaVideo& OaVideo::operator=(OaVideo&& InOther) noexcept {
	if (this != &InOther) {
		Destroy();
		Cfg_             = OaStdMove(InOther.Cfg_);
		Engine_          = InOther.Engine_;
		Stream_          = OaStdMove(InOther.Stream_);
		Decoder_         = OaStdMove(InOther.Decoder_);
		Audio_           = OaStdMove(InOther.Audio_);
		Frame_           = InOther.Frame_;
		FrameIntervalMs_ = InOther.FrameIntervalMs_;
		Accumulator_     = InOther.Accumulator_;
		Playing_         = InOther.Playing_;
		ReachedEos_      = InOther.ReachedEos_;
		StreamEosCurrent_= InOther.StreamEosCurrent_;
		Reorder_         = OaStdMove(InOther.Reorder_);
		DisplayPts_      = OaStdMove(InOther.DisplayPts_);
		RgbaPool_        = OaStdMove(InOther.RgbaPool_);
		RgbaPoolBusy_    = OaStdMove(InOther.RgbaPoolBusy_);
		RgbaPoolConsumerEvents_ = OaStdMove(InOther.RgbaPoolConsumerEvents_);
		Index_           = InOther.Index_;
		StreamFormatGeneration_ = InOther.StreamFormatGeneration_;
		StreamReconnectCount_ = InOther.StreamReconnectCount_;
		InOther.Engine_      = nullptr;
		InOther.Frame_       = {};
		InOther.Accumulator_ = 0.0F;
		InOther.Playing_     = false;
		InOther.ReachedEos_  = false;
		InOther.StreamEosCurrent_ = false;
	}
	return *this;
}

OaVideo::~OaVideo()
{
	Abandon_();
}

void OaVideo::Abandon_() noexcept
{
	if (Engine_ == nullptr) return;
	Pause();
	OaEngine* engine = Engine_;
	auto retired = OaMakeUniquePtr<OaVideo>(OaStdMove(*this));
	OaBorrowedServiceRetirement::Retire(
		*engine,
		retired.Release(),
		&OaVideo::CompleteRetired_,
		&OaVideo::ReleaseRetired_);
}

OaStatus OaVideo::CompleteRetired_(void* InPayload)
{
	auto* video = static_cast<OaVideo*>(InPayload);
	return video ? video->Close() : OaStatus::Ok();
}

void OaVideo::ReleaseRetired_(void* InPayload)
{
	OaUniquePtr<OaVideo> video(static_cast<OaVideo*>(InPayload));
}

OaStatus OaVideo::Close()
{
	if (Engine_ == nullptr) return OaStatus::Ok();
	Playing_ = false;
	if (Audio_.HasValue()) Audio_->Pause();

	OaStatus firstError = OaStatus::Ok();
	auto retainError = [&firstError](const OaStatus& InStatus) {
		if (firstError.IsOk() and not InStatus.IsOk()) firstError = InStatus;
	};
	retainError(WaitForPoolConsumers_());
	retainError(ClearReorder_());
	if (Decoder_.HasValue()) {
		retainError(Decoder_->Close());
		Decoder_.Reset();
	}
	if (Stream_.HasValue()) {
		Stream_.Reset();
	}
	if (Audio_.HasValue()) {
		retainError(Audio_->Close());
		Audio_.Reset();
	}
	// VkImages were owned by Decoder_ via RgbImages_/RgbAllocations_; the
	// decoder's Destroy() above tore them down. We just drop our pool refs.
	RgbaPool_.Clear();
	RgbaPoolBusy_.Clear();
	RgbaPoolConsumerEvents_.Clear();
	Frame_   = {};
	Engine_  = nullptr;
	Playing_ = false;
	return firstError;
}

void OaVideo::Destroy()
{
	if (const auto status = Close(); not status.IsOk()) {
		OA_LOG_ERROR(OaLogComponent::Core,
			"OaVideo::Destroy: shutdown failed: %s",
			status.ToString().c_str());
	}
}

bool OaVideo::IsDone() const
{
	if (Cfg_.Loop) { return false; }
	return ReachedEos_;
}

OaStatus OaVideo::Next()
{
	return StepForward();
}

void OaVideo::Reset()
{
	(void)RestartDecoder_();
	if (Stream_.HasValue()) {
		(void)Stream_->Seek(0);
	}
	Accumulator_      = 0.0F;
	ReachedEos_       = false;
	StreamEosCurrent_ = false;
	Index_            = 0;
	if (Audio_.HasValue()) (void)Audio_->Seek(0U);
}

void OaVideo::Play()
{
	Playing_ = true;
	if (Audio_.HasValue()) (void)Audio_->Play();
}

void OaVideo::Pause()
{
	Playing_ = false;
	if (Audio_.HasValue()) Audio_->Pause();
}

void OaVideo::TogglePlay()
{
	if (Playing_) Pause(); else Play();
}

void OaVideo::SetLoop(bool InLoop)
{
	Cfg_.Loop = InLoop;
	if (Audio_.HasValue()) Audio_->SetLoop(InLoop);
}

OaU64 OaVideo::DurationUs() const
{
	if (not Stream_.HasValue()) return 0U;
	const auto& info = Stream_->GetInfo();
	OaU64 durationTicks = info.Duration;
	if (durationTicks == 0U && not DisplayPts_.Empty()) {
		durationTicks = DisplayPts_.Back();
		if (DisplayPts_.Size() > 1U) {
			durationTicks += DisplayPts_.Back() - DisplayPts_[DisplayPts_.Size() - 2U];
		}
	}
	if (durationTicks == 0U || info.TimebaseNum == 0U || info.TimebaseDen == 0U) {
		return 0U;
	}
	const long double us = static_cast<long double>(durationTicks)
		* static_cast<long double>(info.TimebaseNum) * 1'000'000.0L
		/ static_cast<long double>(info.TimebaseDen);
	return us >= static_cast<long double>(std::numeric_limits<OaU64>::max())
		? std::numeric_limits<OaU64>::max()
		: static_cast<OaU64>(us + 0.5L);
}

OaU64 OaVideo::PositionUs() const
{
	if (not Stream_.HasValue() || Frame_.ImageView == VK_NULL_HANDLE) return 0U;
	const auto& info = Stream_->GetInfo();
	if (info.TimebaseNum == 0U || info.TimebaseDen == 0U) return 0U;
	const long double us = static_cast<long double>(Frame_.PresentationTimestamp)
		* static_cast<long double>(info.TimebaseNum) * 1'000'000.0L
		/ static_cast<long double>(info.TimebaseDen);
	return us >= static_cast<long double>(std::numeric_limits<OaU64>::max())
		? std::numeric_limits<OaU64>::max()
		: static_cast<OaU64>(us + 0.5L);
}

OaStatus OaVideo::SeekUs(OaU64 InTimestampUs)
{
	if (not Stream_.HasValue()) {
		return OaStatus::Error(OaStatusCode::FailedPrecondition,
			"OaVideo::SeekUs called on a closed video");
	}
	const auto& info = Stream_->GetInfo();
	if (info.TimebaseNum == 0U || info.TimebaseDen == 0U) {
		return OaStatus::Error(OaStatusCode::FailedPrecondition,
			"OaVideo::SeekUs requires a declared stream timebase");
	}
	const OaU64 durationUs = DurationUs();
	if (durationUs > 0U) InTimestampUs = std::min(InTimestampUs, durationUs);
	const long double ticks = static_cast<long double>(InTimestampUs)
		* static_cast<long double>(info.TimebaseDen)
		/ (static_cast<long double>(info.TimebaseNum) * 1'000'000.0L);
	const OaU64 timestamp = ticks >= static_cast<long double>(std::numeric_limits<OaU64>::max())
		? std::numeric_limits<OaU64>::max()
		: static_cast<OaU64>(ticks + 0.5L);
	return Seek(timestamp);
}

OaStatus OaVideo::StepForward()
{
	if (not Decoder_.HasValue() or not Stream_.HasValue()) {
		return OaStatus::Error("OaVideo::StepForward: not initialized");
	}
	// Keep the reorder buffer at depth+1 so the lowest PTS in it is
	// guaranteed to be the next display frame (anything that could come
	// out earlier in display order has already been decoded). When the
	// stream EOSes we just stop topping up.
	OaStatus fillStatus = FillReorderBuffer_();
	if (not fillStatus.IsOk()) {
		return fillStatus;
	}
	if (Reorder_.Empty()) {
		// Drained: either real EOS or loop-and-refill.
		if (Cfg_.Loop) {
			OA_RETURN_IF_ERROR(RestartDecoder_());
			OaStatus seekStatus = Stream_->Seek(0);
			if (not seekStatus.IsOk()) {
				return seekStatus;
			}
			StreamEosCurrent_ = false;
			ReachedEos_ = false;
			Index_ = 0;
			OaStatus refill = FillReorderBuffer_();
			if (not refill.IsOk()) {
				return refill;
			}
			if (Reorder_.Empty()) {
				return OaStatus::Error("OaVideo::StepForward: empty stream after loop");
			}
		} else {
			ReachedEos_ = true;
			return OaStatus::Ok();
		}
	}
	OaStatus presentStatus = PopAndPresentLowestPts_();
	if (presentStatus.IsOk()) {
		++Index_;
	}
	return presentStatus;
}

OaStatus OaVideo::StepBackward()
{
	return StepFrames(-1);
}

OaStatus OaVideo::StepFrames(OaI32 InFrameDelta)
{
	if (InFrameDelta == 0) {
		return OaStatus::Ok();
	}
	if (InFrameDelta > 0) {
		for (OaI32 i = 0; i < InFrameDelta; ++i) {
			OA_RETURN_IF_ERROR(StepForward());
		}
		if (Audio_.HasValue()) OA_RETURN_IF_ERROR(Audio_->Seek(PositionUs()));
		return OaStatus::Ok();
	}
	if (not Stream_.HasValue() or not Decoder_.HasValue()
		or DisplayPts_.Empty()) {
		return OaStatus::Error(OaStatusCode::FailedPrecondition,
			"Reverse frame stepping requires an indexed seekable source");
	}

	const OaI64 current = std::max<OaI64>(0, Index_ - 1);
	const OaI64 target = std::max<OaI64>(
		0, current + static_cast<OaI64>(InFrameDelta));
	if (target == current) {
		return OaStatus::Ok();
	}
	OA_RETURN_IF_ERROR(SeekDisplayFrame_(static_cast<OaUsize>(target)));
	if (Audio_.HasValue()) OA_RETURN_IF_ERROR(Audio_->Seek(PositionUs()));
	return OaStatus::Ok();
}

OaStatus OaVideo::Seek(OaU64 InTimestamp)
{
	if (DisplayPts_.Empty()) {
		if (not Stream_.HasValue() or not Stream_->IsSeekable()) {
			return OaStatus::Error(OaStatusCode::FailedPrecondition,
				"OaVideo::Seek: media source is not seekable");
		}
		OA_RETURN_IF_ERROR(RestartDecoder_());
		OA_RETURN_IF_ERROR(Stream_->Seek(InTimestamp));
		StreamEosCurrent_ = false;
		ReachedEos_ = false;
		Index_ = 0;
		OA_RETURN_IF_ERROR(FillReorderBuffer_());
		if (Reorder_.Empty()) {
			return OaStatus::Error(OaStatusCode::OutOfRange,
				"OaVideo::Seek: no frame at requested timestamp");
		}
		OA_RETURN_IF_ERROR(PopAndPresentLowestPts_());
		Accumulator_ = 0.0F;
	} else {
		OaUsize target = 0U;
		while (target + 1U < DisplayPts_.Size() and DisplayPts_[target] < InTimestamp) {
			++target;
		}
		OA_RETURN_IF_ERROR(SeekDisplayFrame_(target));
	}
	if (Audio_.HasValue()) {
		const auto& info = Stream_->GetInfo();
		const OaU64 timestampUs = info.TimebaseDen > 0U
			? InTimestamp * info.TimebaseNum * 1'000'000ULL / info.TimebaseDen
			: 0U;
		OA_RETURN_IF_ERROR(Audio_->Seek(timestampUs));
	}
	return OaStatus::Ok();
}

OaStatus OaVideo::Flush()
{
	OA_RETURN_IF_ERROR(ClearReorder_());
	if (Decoder_.HasValue()) OA_RETURN_IF_ERROR(Decoder_->Flush());
	Accumulator_ = 0.0F;
	return OaStatus::Ok();
}

void OaVideo::Tick(OaF32 InDeltaMs)
{
	if (not Playing_) {
		Accumulator_ = 0.0F;
		return;
	}
	Accumulator_ += InDeltaMs;
	while (Accumulator_ >= FrameIntervalMs_) {
		Accumulator_ -= FrameIntervalMs_;
		OaStatus status = StepForward();
		if (not status.IsOk()) {
			Accumulator_ = 0.0F;
			Playing_     = false;
			break;
		}
		if (IsDone()) {
			Accumulator_ = 0.0F;
			break;
		}
	}
}

OaU32 OaVideo::Width() const
{
	return Stream_.HasValue() ? Stream_->GetInfo().Width : 0U;
}

OaU32 OaVideo::Height() const
{
	return Stream_.HasValue() ? Stream_->GetInfo().Height : 0U;
}

OaU32 OaVideo::FrameRate() const
{
	return Stream_.HasValue() ? Stream_->GetInfo().FrameRate : 0U;
}

OaUsize OaVideo::FrameCount() const
{
	if (not Stream_.HasValue()) return 0U;
	if (not Stream_->Samples_.Empty()) return Stream_->Samples_.Size();
	const auto& info = Stream_->GetInfo();
	if (info.Duration == 0U or info.TimebaseDen == 0U or info.FrameRate == 0U) return 0U;
	const double seconds = static_cast<double>(info.Duration)
		* static_cast<double>(info.TimebaseNum) / static_cast<double>(info.TimebaseDen);
	return static_cast<OaUsize>(seconds * static_cast<double>(info.FrameRate) + 0.5);
}

bool OaVideo::IsEos() const
{
	return Stream_.HasValue() and Stream_->IsEos();
}

const OaContainerInfo& OaVideo::GetContainerInfo() const
{
	static const OaContainerInfo empty = {};
	return Stream_.HasValue() ? Stream_->GetInfo() : empty;
}

const OaVideoStreamStats& OaVideo::GetStreamStats() const
{
	static const OaVideoStreamStats empty = {};
	return Stream_.HasValue() ? Stream_->GetStats() : empty;
}

void OaVideo::MarkCurrentFrameConsumed(const OaEvent& InConsumed)
{
	if (Frame_.Image == VK_NULL_HANDLE || not InConsumed.IsValid()) {
		return;
	}
	for (OaUsize i = 0; i < RgbaPool_.Size(); ++i) {
		if (RgbaPool_[i].Image == Frame_.Image) {
			RgbaPoolConsumerEvents_[i] = InConsumed;
			return;
		}
	}
}

void OaVideo::MarkCurrentFrameConsumed(
	const OaVkTimelineSemaphore& InSemaphore,
	OaU64 InValue)
{
	if (Engine_ == nullptr || InSemaphore.Semaphore == nullptr || InValue == 0) {
		return;
	}
	MarkCurrentFrameConsumed(OaEvent(
		Engine_->Device,
		InSemaphore,
		InValue));
}

OaResult<OaVec<OaU8>> OaVideo::ReadbackCurrentRgba()
{
	if (!Decoder_.HasValue() || Frame_.ImageView == VK_NULL_HANDLE) {
		return OaStatus::Error("OaVideo::ReadbackCurrentRgba: no current frame");
	}
	return OaFnVideo::ReadbackRgba(*Decoder_, Frame_);
}

OaResult<OaMatrix> OaVideo::CurrentFrameToMatrix(bool InNormalizeImageNet)
{
	if (!Decoder_.HasValue() || Frame_.ImageView == VK_NULL_HANDLE) {
		return OaStatus::Error("OaVideo::CurrentFrameToMatrix: no current frame");
	}
	return OaFnVideo::FrameToBf16(*Decoder_, Frame_, InNormalizeImageNet);
}

OaResult<OaImage> OaVideo::CurrentFrameToImage(bool InNormalizeImageNet)
{
	auto matrixResult = CurrentFrameToMatrix(InNormalizeImageNet);
	if (matrixResult.IsError()) {
		return matrixResult.GetStatus();
	}
	return OaImage(
		OaStdMove(matrixResult).GetValue(),
		OaImageLayout::Nchw,
		OaImageFormat::Rgb);
}

OaResult<OaVideoFrame> OaVideo::AcquireRgbaFromPool_()
{
	for (OaUsize i = 0; i < RgbaPool_.Size(); ++i) {
		const bool consumed = RgbaPoolConsumerEvents_[i].IsComplete();
		if (not RgbaPoolBusy_[i] and consumed) {
			RgbaPoolBusy_[i] = true;
			RgbaPoolConsumerEvents_[i] = {};
			return RgbaPool_[i];
		}
	}
	const OaUsize maxPoolSize = static_cast<OaUsize>(Cfg_.ReorderDepth) + 3U;
	if (RgbaPool_.Size() >= maxPoolSize) {
		for (OaUsize i = 0; i < RgbaPool_.Size(); ++i) {
			if (RgbaPoolBusy_[i]) {
				continue;
			}
			OA_RETURN_IF_ERROR(RgbaPoolConsumerEvents_[i].Wait());
			RgbaPoolBusy_[i] = true;
			RgbaPoolConsumerEvents_[i] = {};
			return RgbaPool_[i];
		}
		return OaStatus::Error(
			OaStatusCode::ResourceExhausted,
			"OaVideo RGBA pool exhausted while every slot is owned");
	}
	auto allocResult = OaFnVideo::AllocateRgbaFrame(
		*Decoder_,
		Stream_->GetInfo().Width,
		Stream_->GetInfo().Height);
	if (not allocResult.IsOk()) {
		return allocResult.GetStatus();
	}
	RgbaPool_.PushBack(*allocResult);
	RgbaPoolBusy_.PushBack(true);
	RgbaPoolConsumerEvents_.PushBack({});
	return *allocResult;
}

void OaVideo::ReleaseRgbaToPool_(const OaVideoFrame& InFrame) {
	if (InFrame.Image == VK_NULL_HANDLE) {
		return;
	}
	for (OaUsize i = 0; i < RgbaPool_.Size(); ++i) {
		if (RgbaPool_[i].Image == InFrame.Image) {
			RgbaPoolBusy_[i] = false;
			return;
		}
	}
}

OaStatus OaVideo::DecodeOneIntoReorder_() {
	// Keep reading packets until we get a picture (skip parameter-set-only packets)
	while (true) {
		OaVideoPacket packet;
		OaStatus readStatus = Stream_->ReadNextPacket(packet);
		if (not readStatus.IsOk()) {
			StreamEosCurrent_ = true;
			return Stream_->IsEos() ? OaStatus::Ok() : readStatus;
		}
		const auto& streamStats = Stream_->GetStats();
		if (streamStats.FormatGeneration != StreamFormatGeneration_
			or streamStats.ReconnectCount != StreamReconnectCount_) {
			StreamFormatGeneration_ = streamStats.FormatGeneration;
			StreamReconnectCount_ = streamStats.ReconnectCount;
			OA_RETURN_IF_ERROR(RestartDecoder_());
			StreamEosCurrent_ = false;
		}

		OaVideoFrame nv12 = {};
		OaStatus decodeStatus = Decoder_->DecodeFrame(
			OaSpan<const OaU8>(packet.Data.Data(), packet.Data.Size()),
			nv12);
		if (not decodeStatus.IsOk()) {
			return decodeStatus;
		}

		// Skip parameter-set-only packets (no frame output)
		if (nv12.ImageView == VK_NULL_HANDLE) {
			continue;
		}

		// Skip hidden AV1/VP9 frames (show_frame=0). They decode into the DPB
		// above (so later frames can reference them) but are never displayed;
		// enqueueing them would present duplicated / out-of-order frames.
		if (not nv12.Shown) {
			continue;
		}

		// Convert NV12 → RGBA *now* into our own pool slot so the DPB layer is
		// free for the next decode. Holding NV12 across decodes is unsafe because
		// the H.264 sliding window evicts oldest slots and the allocator picks
		// them up for the next frame, trashing any data we'd be holding.
	auto rgbaResult = AcquireRgbaFromPool_();
	if (not rgbaResult.IsOk()) {
		return rgbaResult.GetStatus();
	}
	OaVideoFrame rgba = *rgbaResult;
	rgba.PresentationTimestamp = packet.PresentationTimestamp;

	OaVideoConversionOptions options;
	options.ConvertToRgb        = true;
	options.PreferHardwareYCbCr = Cfg_.PreferHardwareYCbCr;
	options.Filter               = Cfg_.Filter;
	OaStatus convertStatus = OaFnVideo::CvtNv12ToRgbInto(
		*Decoder_,
		nv12,
		options,
		rgba);
	if (not convertStatus.IsOk()) {
		ReleaseRgbaToPool_(rgba);
		return convertStatus;
	}

	Reorder_.EmplaceBack(rgba, packet.PresentationTimestamp);
	return OaStatus::Ok();
	} // end while
}

OaStatus OaVideo::FillReorderBuffer_() {
	// Top up to depth+1 entries so the smallest-PTS entry is guaranteed
	// to be the next displayable frame (no late arrival can undercut it).
	while (Reorder_.Size() <= Cfg_.ReorderDepth and not StreamEosCurrent_) {
		OaStatus s = DecodeOneIntoReorder_();
		if (not s.IsOk()) {
			return s;
		}
	}
	return OaStatus::Ok();
}

OaStatus OaVideo::PopAndPresentLowestPts_() {
	if (Reorder_.Empty()) {
		return OaStatus::Error("OaVideo: reorder buffer empty");
	}
	OaUsize minIdx = 0;
	OaU64 minPts = Reorder_[0].Pts;
	for (OaUsize i = 1; i < Reorder_.Size(); ++i) {
		if (Reorder_[i].Pts < minPts) {
			minPts = Reorder_[i].Pts;
			minIdx = i;
		}
	}
	ReorderEntry entry = OaStdMove(Reorder_[minIdx]);
	Reorder_.Erase(Reorder_.Data() + minIdx);

	// Return the previously-displayed RGBA to the pool. Each entry owns its
	// own RGBA target so no decoded frame can corrupt one we haven't shown.
	ReleaseRgbaToPool_(Frame_);
	Frame_ = entry.Rgba;
	return OaStatus::Ok();
}

OaStatus OaVideo::ClearReorder_() {
	for (ReorderEntry& entry : Reorder_) {
		ReleaseRgbaToPool_(entry.Rgba);
	}
	Reorder_.Clear();
	return OaStatus::Ok();
}

OaStatus OaVideo::WaitForPoolConsumers_() {
	for (OaUsize i = 0; i < RgbaPoolConsumerEvents_.Size(); ++i) {
		OA_RETURN_IF_ERROR(RgbaPoolConsumerEvents_[i].Wait());
		RgbaPoolConsumerEvents_[i] = {};
	}
	return OaStatus::Ok();
}

OaStatus OaVideo::RestartDecoder_() {
	if (Engine_ == nullptr or not Stream_.HasValue()) {
		return OaStatus::Error("OaVideo::RestartDecoder_: not initialized");
	}

	OA_RETURN_IF_ERROR(ClearReorder_());
	OA_RETURN_IF_ERROR(WaitForPoolConsumers_());
	if (Decoder_.HasValue()) {
		// Conversion completion precedes the asynchronous DPB restore submit.
		// Flush waits the decoder's latest timeline value, covering both,
		// before RestartDecoder_ destroys the old session and image pool.
		OA_RETURN_IF_ERROR(Decoder_->Flush());
	}

	auto profile = Stream_->GetVideoProfile();
	profile.MaxDpbSlots = Cfg_.MaxDpbSlots;
	auto decoderResult = OaVideoDecoder::Create(*Engine_, profile);
	if (not decoderResult.IsOk()) {
		return decoderResult.GetStatus();
	}

	if (Decoder_.HasValue()) {
		Decoder_->Destroy();
		Decoder_.Reset();
	}
	RgbaPool_.Clear();
	RgbaPoolBusy_.Clear();
	RgbaPoolConsumerEvents_.Clear();
	Frame_ = {};
	Decoder_.Emplace(OaStdMove(*decoderResult));
	return OaStatus::Ok();
}

OaStatus OaVideo::SeekDisplayFrame_(OaUsize InTargetFrameIndex) {
	if (InTargetFrameIndex >= DisplayPts_.Size()) {
		return OaStatus::Error(OaStatusCode::InvalidArgument,
			"OaVideo::SeekDisplayFrame_: target out of range");
	}

	OA_RETURN_IF_ERROR(RestartDecoder_());
	OA_RETURN_IF_ERROR(Stream_->Seek(DisplayPts_[InTargetFrameIndex]));
	StreamEosCurrent_ = false;
	ReachedEos_ = false;

	const OaU64 targetPts = DisplayPts_[InTargetFrameIndex];
	while (true) {
		OA_RETURN_IF_ERROR(FillReorderBuffer_());
		if (Reorder_.Empty()) {
			return OaStatus::Error(
				"OaVideo::SeekDisplayFrame_: target was not decoded");
		}
		OA_RETURN_IF_ERROR(PopAndPresentLowestPts_());
		if (Frame_.PresentationTimestamp >= targetPts) {
			break;
		}
	}
	Index_ = static_cast<OaI64>(InTargetFrameIndex) + 1;
	Accumulator_ = 0.0F;
	return OaStatus::Ok();
}
