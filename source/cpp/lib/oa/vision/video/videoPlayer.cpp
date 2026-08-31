// oa::VideoPlayer — implementation. See header for usage.

#include <oa/vision/videoPlayer.h>
#include <oa/vision/fnVideo.h>

#include <oa/core/log.h>
#include <oa/runtime/engine.h>
#include "oa/runtime/engine/borrowedServiceRetirement.h"

#include <oa/core/std/algo.h>
#include <oa/core/std/limits.h>

oa::Result<oa::VideoPlayer> oa::VideoPlayer::open(oa::Engine& inEngine, const oa::VideoPlayerConfig& inCfg)
{
	const oa::String& uri = inCfg.uri;
	if (uri.empty()) {
		return oa::Status::error(oa::StatusCode::InvalidArgument, "oa::VideoPlayer: empty URI");
	}
	if (inCfg.reorderDepth > 64U || inCfg.presentationCacheFrames > 4096U) {
		return oa::Status::invalidArgument(
			"oa::VideoPlayer reorder/cache frame limits are out of range");
	}

	auto demuxerResult = oa::VideoDemuxer::open(uri, inCfg.demuxerConfig);
	if (not demuxerResult.isOk()) {
		return demuxerResult.getStatus();
	}

	auto profile = demuxerResult->getVideoProfile();
	profile.maxDpbSlots = inCfg.maxDpbSlots;
	auto decoderResult = oa::VideoDecoder::create(inEngine, profile);
	if (not decoderResult.isOk()) {
		return decoderResult.getStatus();
	}

	oa::VideoPlayer it;
	it.cfg_     = inCfg;
	it.engine_  = &inEngine;
	it.demuxer_.emplace(oa::move(*demuxerResult));
	it.decoder_.emplace(oa::move(*decoderResult));
	it.playing_ = inCfg.startPlaying;
	it.demuxerFormatGeneration_ = it.demuxer_->formatGeneration();
	it.demuxerReconnectCount_ = it.demuxer_->getStats().reconnectCount;
	if (inCfg.audio) {
		oa::AudioPlayerConfig audioConfig;
		audioConfig.uri = uri;
		audioConfig.loop = inCfg.loop;
		auto audio = oa::AudioPlayer::open(inEngine, audioConfig);
		if (audio.isOk()) {
			it.audio_.emplace(oa::move(*audio));
		} else if (audio.getStatus().getCode() != oa::StatusCode::NotFound) {
			OaLogWarn(oa::LogComponent::Video,
				"oa::VideoPlayer audio disabled: {}", audio.getStatus().getMessage().cStr());
		}
	}
	it.displayPts_.reserve(it.demuxer_->samples_.size());
	for (oa::Usize sampleIndex = 0U;
		sampleIndex < it.demuxer_->samples_.size(); ++sampleIndex) {
		auto pts = it.demuxer_->samplePresentationTimestamp(sampleIndex);
		if (not pts.isOk()) return pts.getStatus();
		it.displayPts_.pushBack(*pts);
	}
	if (not it.displayPts_.empty()) {
		oa::sort(it.displayPts_.data(),
			it.displayPts_.data() + it.displayPts_.size());
	}

	oa::U32 fps = it.demuxer_->getInfo().frameRate;
	if (inCfg.frameRateOverride > 0.0F) {
		it.frameIntervalMs_ = 1000.0F / inCfg.frameRateOverride;
	} else if (fps > 0U) {
		it.frameIntervalMs_ = 1000.0F / static_cast<oa::F32>(fps);
	} else {
		it.frameIntervalMs_ = 1000.0F / 30.0F;
	}
	if (inCfg.presentationCacheFrames > 0U
		&& inCfg.presentationCacheBytes > 0U) {
		const oa::U64 width = it.demuxer_->getInfo().width;
		const oa::U64 height = it.demuxer_->getInfo().height;
		if (width > 0U && height > 0U
			&& width <= oa::Limits<oa::U64>::max() / height
			&& width * height <= oa::Limits<oa::U64>::max() / 4U) {
			const oa::U64 frameBytes = width * height * 4U;
			const oa::U64 byBudget = inCfg.presentationCacheBytes / frameBytes;
			it.presentationCacheCapacity_ = static_cast<oa::Usize>(oa::min<oa::U64>(
				inCfg.presentationCacheFrames, byBudget));
		}
	}
	it.stats_.presentationCacheCapacity = static_cast<oa::U32>(
		oa::min<oa::Usize>(it.presentationCacheCapacity_,
			oa::Limits<oa::U32>::max()));

	oa::Status first = it.next();
	if (not first.isOk()) {
		return first;
	}
	if (inCfg.startPlaying && it.audio_.hasValue()) {
		OA_RETURN_IF_ERROR(it.audio_->seek(it.positionUs()));
		OA_RETURN_IF_ERROR(it.audio_->play());
	}

	return oa::Result<oa::VideoPlayer>(oa::move(it));
}

oa::VideoPlayer::VideoPlayer(oa::VideoPlayer&& inOther) noexcept
	: cfg_(oa::move(inOther.cfg_))
	, engine_(inOther.engine_)
	, demuxer_(oa::move(inOther.demuxer_))
	, decoder_(oa::move(inOther.decoder_))
	, audio_(oa::move(inOther.audio_))
	, frame_(inOther.frame_)
	, frameIntervalMs_(inOther.frameIntervalMs_)
	, accumulator_(inOther.accumulator_)
	, playing_(inOther.playing_)
	, reachedEos_(inOther.reachedEos_)
	, demuxerEosCurrent_(inOther.demuxerEosCurrent_)
	, reorder_(oa::move(inOther.reorder_))
	, presentationCache_(oa::move(inOther.presentationCache_))
	, displayPts_(oa::move(inOther.displayPts_))
	, rgbaPool_(oa::move(inOther.rgbaPool_))
	, rgbaPoolBusy_(oa::move(inOther.rgbaPoolBusy_))
	, rgbaPoolConsumerEvents_(oa::move(inOther.rgbaPoolConsumerEvents_))
	, index_(inOther.index_)
	, decodeCursor_(inOther.decodeCursor_)
	, presentationCacheCapacity_(inOther.presentationCacheCapacity_)
	, stats_(inOther.stats_)
	, demuxerFormatGeneration_(inOther.demuxerFormatGeneration_)
	, demuxerReconnectCount_(inOther.demuxerReconnectCount_)
{
	inOther.engine_      = nullptr;
	inOther.frame_       = {};
	inOther.accumulator_ = 0.0F;
	inOther.playing_     = false;
	inOther.reachedEos_  = false;
	inOther.demuxerEosCurrent_ = false;
	inOther.decodeCursor_ = 0U;
	inOther.presentationCacheCapacity_ = 0U;
	inOther.stats_ = {};
}

oa::VideoPlayer& oa::VideoPlayer::operator=(oa::VideoPlayer&& inOther) noexcept {
	if (this != &inOther) {
		abandon_();
		cfg_             = oa::move(inOther.cfg_);
		engine_          = inOther.engine_;
		demuxer_          = oa::move(inOther.demuxer_);
		decoder_         = oa::move(inOther.decoder_);
		audio_           = oa::move(inOther.audio_);
		frame_           = inOther.frame_;
		frameIntervalMs_ = inOther.frameIntervalMs_;
		accumulator_     = inOther.accumulator_;
		playing_         = inOther.playing_;
		reachedEos_      = inOther.reachedEos_;
		demuxerEosCurrent_= inOther.demuxerEosCurrent_;
		reorder_         = oa::move(inOther.reorder_);
		presentationCache_ = oa::move(inOther.presentationCache_);
		displayPts_      = oa::move(inOther.displayPts_);
		rgbaPool_        = oa::move(inOther.rgbaPool_);
		rgbaPoolBusy_    = oa::move(inOther.rgbaPoolBusy_);
		rgbaPoolConsumerEvents_ = oa::move(inOther.rgbaPoolConsumerEvents_);
		index_           = inOther.index_;
		decodeCursor_    = inOther.decodeCursor_;
		presentationCacheCapacity_ = inOther.presentationCacheCapacity_;
		stats_           = inOther.stats_;
		demuxerFormatGeneration_ = inOther.demuxerFormatGeneration_;
		demuxerReconnectCount_ = inOther.demuxerReconnectCount_;
		inOther.engine_      = nullptr;
		inOther.frame_       = {};
		inOther.accumulator_ = 0.0F;
		inOther.playing_     = false;
		inOther.reachedEos_  = false;
		inOther.demuxerEosCurrent_ = false;
		inOther.decodeCursor_ = 0U;
		inOther.presentationCacheCapacity_ = 0U;
		inOther.stats_ = {};
	}
	return *this;
}

oa::VideoPlayer::~VideoPlayer()
{
	abandon_();
}

void oa::VideoPlayer::abandon_() noexcept
{
	if (engine_ == nullptr) return;
	pause();
	oa::Engine* engine = engine_;
	auto retired = oa::makeUnique<oa::VideoPlayer>(oa::move(*this));
	oa::BorrowedServiceRetirement::retire(
		*engine,
		retired.release(),
		&oa::VideoPlayer::completeRetired_,
		&oa::VideoPlayer::releaseRetired_);
}

oa::Status oa::VideoPlayer::completeRetired_(void* inPayload)
{
	auto* video = static_cast<oa::VideoPlayer*>(inPayload);
	return video ? video->close() : oa::Status::ok();
}

void oa::VideoPlayer::releaseRetired_(void* inPayload)
{
	oa::UniquePtr<oa::VideoPlayer> video(static_cast<oa::VideoPlayer*>(inPayload));
}

oa::Status oa::VideoPlayer::close()
{
	if (engine_ == nullptr) return oa::Status::ok();
	playing_ = false;
	if (audio_.hasValue()) audio_->pause();

	oa::Status firstError = oa::Status::ok();
	auto retainError = [&firstError](const oa::Status& inStatus) {
		if (firstError.isOk() and not inStatus.isOk()) firstError = inStatus;
	};
	retainError(waitForPoolConsumers_());
	retainError(clearReorder_());
	retainError(clearPresentationCache_());
	if (decoder_.hasValue()) {
		retainError(decoder_->close());
		decoder_.reset();
	}
	if (demuxer_.hasValue()) {
		retainError(demuxer_->close());
		demuxer_.reset();
	}
	if (audio_.hasValue()) {
		retainError(audio_->close());
		audio_.reset();
	}
	// VkImages were owned by decoder_ via rgbImages/rgbAllocations; the
	// decoder close() above tore them down. We just drop our pool refs.
	rgbaPool_.clear();
	rgbaPoolBusy_.clear();
	rgbaPoolConsumerEvents_.clear();
	frame_   = {};
	engine_  = nullptr;
	playing_ = false;
	decodeCursor_ = 0U;
	return firstError;
}

bool oa::VideoPlayer::isDone() const
{
	if (cfg_.loop) { return false; }
	return reachedEos_;
}

void oa::VideoPlayer::reset()
{
	(void)clearPresentationCache_();
	(void)resetDecoderForSeek_();
	if (demuxer_.hasValue()) {
		(void)demuxer_->seek(0);
	}
	accumulator_      = 0.0F;
	reachedEos_       = false;
	demuxerEosCurrent_ = false;
	index_            = 0;
	decodeCursor_     = 0U;
	if (audio_.hasValue()) (void)audio_->seek(0U);
}

void oa::VideoPlayer::play()
{
	playing_ = true;
	if (audio_.hasValue()) {
		(void)audio_->seek(positionUs());
		(void)audio_->play();
	}
}

void oa::VideoPlayer::pause()
{
	playing_ = false;
	if (audio_.hasValue()) audio_->pause();
}

void oa::VideoPlayer::togglePlay()
{
	if (playing_) pause(); else play();
}

oa::Status oa::VideoPlayer::setMuted(bool inMuted)
{
	if (not audio_.hasValue()) {
		return oa::Status::error(
			oa::StatusCode::FailedPrecondition,
			"oa::VideoPlayer::setMuted requires an active audio stream");
	}
	return audio_->setMuted(inMuted);
}

bool oa::VideoPlayer::isMuted() const noexcept
{
	return audio_.hasValue() and audio_->isMuted();
}

void oa::VideoPlayer::setLoop(bool inLoop)
{
	cfg_.loop = inLoop;
	if (audio_.hasValue()) audio_->setLoop(inLoop);
}

oa::U64 oa::VideoPlayer::durationUs() const
{
	if (not demuxer_.hasValue()) return 0U;
	const auto& info = demuxer_->getInfo();
	oa::U64 durationTicks = info.duration;
	if (durationTicks == 0U && not displayPts_.empty()) {
		durationTicks = displayPts_.back();
		if (displayPts_.size() > 1U) {
			durationTicks += displayPts_.back() - displayPts_[displayPts_.size() - 2U];
		}
	}
	if (durationTicks == 0U || info.timebaseNum == 0U || info.timebaseDen == 0U) {
		return 0U;
	}
	const long double us = static_cast<long double>(durationTicks)
		* static_cast<long double>(info.timebaseNum) * 1'000'000.0L
		/ static_cast<long double>(info.timebaseDen);
	return us >= static_cast<long double>(oa::Limits<oa::U64>::max())
		? oa::Limits<oa::U64>::max()
		: static_cast<oa::U64>(us + 0.5L);
}

oa::U64 oa::VideoPlayer::positionUs() const
{
	if (not demuxer_.hasValue() || frame_.imageView == VK_NULL_HANDLE) return 0U;
	const auto& info = demuxer_->getInfo();
	if (info.timebaseNum == 0U || info.timebaseDen == 0U) return 0U;
	const long double us = static_cast<long double>(frame_.presentationTimestamp)
		* static_cast<long double>(info.timebaseNum) * 1'000'000.0L
		/ static_cast<long double>(info.timebaseDen);
	return us >= static_cast<long double>(oa::Limits<oa::U64>::max())
		? oa::Limits<oa::U64>::max()
		: static_cast<oa::U64>(us + 0.5L);
}

oa::Status oa::VideoPlayer::seekUs(oa::U64 inTimestampUs)
{
	if (not demuxer_.hasValue()) {
		return oa::Status::error(oa::StatusCode::FailedPrecondition,
			"oa::VideoPlayer::seekUs called on a closed video");
	}
	const auto& info = demuxer_->getInfo();
	if (info.timebaseNum == 0U || info.timebaseDen == 0U) {
		return oa::Status::error(oa::StatusCode::FailedPrecondition,
			"oa::VideoPlayer::seekUs requires a declared stream timebase");
	}
	const oa::U64 totalDurationUs = durationUs();
	if (totalDurationUs > 0U) {
		inTimestampUs = oa::min(inTimestampUs, totalDurationUs);
	}
	const long double ticks = static_cast<long double>(inTimestampUs)
		* static_cast<long double>(info.timebaseDen)
		/ (static_cast<long double>(info.timebaseNum) * 1'000'000.0L);
	const oa::U64 timestamp = ticks >= static_cast<long double>(oa::Limits<oa::U64>::max())
		? oa::Limits<oa::U64>::max()
		: static_cast<oa::U64>(ticks + 0.5L);
	return seek(timestamp);
}

oa::Status oa::VideoPlayer::seekFrame(oa::Usize inFrameIndex) {
	if (displayPts_.empty()) {
		return oa::Status::error(
			oa::StatusCode::FailedPrecondition,
			"oa::VideoPlayer::seekFrame requires an indexed seekable source");
	}
	if (inFrameIndex >= displayPts_.size()) {
		return oa::Status::error(
			oa::StatusCode::OutOfRange,
			"oa::VideoPlayer::seekFrame index is outside the display sequence");
	}
	OA_RETURN_IF_ERROR(seekDisplayFrame_(inFrameIndex));
	if (audio_.hasValue()) {
		OA_RETURN_IF_ERROR(audio_->seek(positionUs()));
	}
	return oa::Status::ok();
}

oa::Status oa::VideoPlayer::next()
{
	if (not decoder_.hasValue() or not demuxer_.hasValue()) {
		return oa::Status::error("oa::VideoPlayer::next: not initialized");
	}
	if (not displayPts_.empty()
		&& index_ >= 0
		&& static_cast<oa::Usize>(index_) < displayPts_.size()) {
		const oa::Usize target = static_cast<oa::Usize>(index_);
		if (presentCached_(target)) return oa::Status::ok();
		if (target != decodeCursor_) return seekDisplayFrame_(target);
	}
	// Keep the reorder buffer at depth+1 so the lowest PTS in it is
	// guaranteed to be the next display frame (anything that could come
	// out earlier in display order has already been decoded). When the
	// stream EOSes we just stop topping up.
	oa::Status fillStatus = fillReorderBuffer_();
	if (not fillStatus.isOk()) {
		return fillStatus;
	}
	if (reorder_.empty()) {
		// Drained: either real EOS or loop-and-refill.
		if (cfg_.loop) {
			OA_RETURN_IF_ERROR(clearPresentationCache_());
			OA_RETURN_IF_ERROR(resetDecoderForSeek_());
			oa::Status seekStatus = demuxer_->seek(0);
			if (not seekStatus.isOk()) {
				return seekStatus;
			}
			demuxerEosCurrent_ = false;
			reachedEos_ = false;
			index_ = 0;
			decodeCursor_ = 0U;
			oa::Status refill = fillReorderBuffer_();
			if (not refill.isOk()) {
				return refill;
			}
			if (reorder_.empty()) {
				return oa::Status::error("oa::VideoPlayer::next: empty stream after loop");
			}
		} else {
			reachedEos_ = true;
			return oa::Status::ok();
		}
	}
	ReorderEntry entry;
	OA_RETURN_IF_ERROR(popLowestPts_(entry));
	const oa::Usize displayIndex = displayPts_.empty()
		? static_cast<oa::Usize>(oa::max<oa::I64>(0, index_))
		: decodeCursor_;
	OA_RETURN_IF_ERROR(presentDecoded_(oa::move(entry), displayIndex));
	decodeCursor_ = displayIndex + 1U;
	return oa::Status::ok();
}

oa::Status oa::VideoPlayer::stepBackward()
{
	return stepFrames(-1);
}

oa::Status oa::VideoPlayer::stepFrames(oa::I32 inFrameDelta)
{
	if (inFrameDelta == 0) {
		return oa::Status::ok();
	}
	if (inFrameDelta > 0) {
		for (oa::I32 i = 0; i < inFrameDelta; ++i) {
			OA_RETURN_IF_ERROR(next());
		}
		if (audio_.hasValue()) {
			OA_RETURN_IF_ERROR(audio_->seek(positionUs()));
		}
		return oa::Status::ok();
	}
	if (not demuxer_.hasValue() or not decoder_.hasValue()
		or displayPts_.empty()) {
		return oa::Status::error(oa::StatusCode::FailedPrecondition,
			"Reverse frame stepping requires an indexed seekable source");
	}

	const oa::I64 current = oa::max<oa::I64>(0, index_ - 1);
	const oa::I64 target = oa::clamp<oa::I64>(
		current + static_cast<oa::I64>(inFrameDelta),
		0,
		static_cast<oa::I64>(displayPts_.size() - 1U));
	if (target == current) {
		return oa::Status::ok();
	}
	if (not presentCached_(static_cast<oa::Usize>(target))) {
		++stats_.presentationCacheMisses;
		OA_RETURN_IF_ERROR(seekDisplayFrame_(static_cast<oa::Usize>(target)));
	}
	if (audio_.hasValue()) {
		OA_RETURN_IF_ERROR(audio_->seek(positionUs()));
	}
	return oa::Status::ok();
}

oa::Status oa::VideoPlayer::seek(oa::U64 inTimestamp)
{
	if (displayPts_.empty()) {
		if (not demuxer_.hasValue() or not demuxer_->isSeekable()) {
			return oa::Status::error(oa::StatusCode::FailedPrecondition,
				"oa::VideoPlayer::seek: media source is not seekable");
		}
		OA_RETURN_IF_ERROR(clearPresentationCache_());
		OA_RETURN_IF_ERROR(resetDecoderForSeek_());
		OA_RETURN_IF_ERROR(demuxer_->seek(inTimestamp));
		demuxerEosCurrent_ = false;
		reachedEos_ = false;
		index_ = 0;
		decodeCursor_ = 0U;
		OA_RETURN_IF_ERROR(fillReorderBuffer_());
		if (reorder_.empty()) {
			return oa::Status::error(oa::StatusCode::OutOfRange,
				"oa::VideoPlayer::seek: no frame at requested timestamp");
		}
		ReorderEntry entry;
		OA_RETURN_IF_ERROR(popLowestPts_(entry));
		OA_RETURN_IF_ERROR(presentDecoded_(oa::move(entry), 0U));
		decodeCursor_ = 1U;
		accumulator_ = 0.0F;
	} else {
		oa::Usize target = 0U;
		while (target + 1U < displayPts_.size() and displayPts_[target] < inTimestamp) {
			++target;
		}
		OA_RETURN_IF_ERROR(seekDisplayFrame_(target));
	}
	if (audio_.hasValue()) {
		OA_RETURN_IF_ERROR(audio_->seek(positionUs()));
	}
	return oa::Status::ok();
}

oa::Status oa::VideoPlayer::flush()
{
	OA_RETURN_IF_ERROR(clearReorder_());
	OA_RETURN_IF_ERROR(clearPresentationCache_());
	if (decoder_.hasValue()) OA_RETURN_IF_ERROR(decoder_->flush());
	accumulator_ = 0.0F;
	return oa::Status::ok();
}

oa::F32 oa::VideoPlayer::currentFrameIntervalMs_() const {
	if (cfg_.frameRateOverride > 0.0F || not demuxer_.hasValue()
		or index_ <= 0) {
		return frameIntervalMs_;
	}
	const oa::Usize current = static_cast<oa::Usize>(index_ - 1);
	if (current + 1U >= displayPts_.size()) return frameIntervalMs_;
	const oa::U64 currentPts = displayPts_[current];
	const oa::U64 nextPts = displayPts_[current + 1U];
	const auto& info = demuxer_->getInfo();
	if (nextPts <= currentPts || info.timebaseNum == 0U
		or info.timebaseDen == 0U) {
		return frameIntervalMs_;
	}
	const long double milliseconds = static_cast<long double>(nextPts - currentPts)
		* static_cast<long double>(info.timebaseNum) * 1000.0L
		/ static_cast<long double>(info.timebaseDen);
	if (milliseconds <= 0.0L
		or milliseconds > static_cast<long double>(oa::Limits<oa::F32>::max())) {
		return frameIntervalMs_;
	}
	return static_cast<oa::F32>(milliseconds);
}

void oa::VideoPlayer::tick(oa::F32 inDeltaMs)
{
	if (not playing_) {
		accumulator_ = 0.0F;
		return;
	}
	// A frame callback is not authorization for unbounded catch-up decode.
	// Invalid time pauses the session; one update is capped to one second.
	if (not oa::isFinite(inDeltaMs) || inDeltaMs < 0.0F) {
		pause();
		accumulator_ = 0.0F;
		return;
	}
	inDeltaMs = oa::min(inDeltaMs, 1000.0F);
	accumulator_ += inDeltaMs;
	while (true) {
		const oa::F32 intervalMs = currentFrameIntervalMs_();
		if (accumulator_ < intervalMs) break;
		accumulator_ -= intervalMs;
		oa::Status status = next();
		if (not status.isOk()) {
			accumulator_ = 0.0F;
			playing_     = false;
			break;
		}
		if (isDone()) {
			accumulator_ = 0.0F;
			playing_ = false;
			if (audio_.hasValue()) audio_->pause();
			break;
		}
	}
}

oa::U32 oa::VideoPlayer::width() const
{
	return demuxer_.hasValue() ? demuxer_->getInfo().width : 0U;
}

oa::U32 oa::VideoPlayer::height() const
{
	return demuxer_.hasValue() ? demuxer_->getInfo().height : 0U;
}

oa::U32 oa::VideoPlayer::frameRate() const
{
	return demuxer_.hasValue() ? demuxer_->getInfo().frameRate : 0U;
}

oa::Usize oa::VideoPlayer::frameCount() const
{
	if (not demuxer_.hasValue()) return 0U;
	if (not demuxer_->samples_.empty()) return demuxer_->samples_.size();
	const auto& info = demuxer_->getInfo();
	if (info.duration == 0U or info.timebaseDen == 0U or info.frameRate == 0U) return 0U;
	const double seconds = static_cast<double>(info.duration)
		* static_cast<double>(info.timebaseNum) / static_cast<double>(info.timebaseDen);
	return static_cast<oa::Usize>(seconds * static_cast<double>(info.frameRate) + 0.5);
}

bool oa::VideoPlayer::isEos() const
{
	return demuxer_.hasValue() and demuxer_->isEos();
}

const oa::VideoContainerInfo& oa::VideoPlayer::getContainerInfo() const
{
	static const oa::VideoContainerInfo empty = {};
	return demuxer_.hasValue() ? demuxer_->getInfo() : empty;
}

const oa::VideoDemuxerStats& oa::VideoPlayer::getDemuxerStats() const
{
	static const oa::VideoDemuxerStats empty = {};
	return demuxer_.hasValue() ? demuxer_->getStats() : empty;
}

void oa::VideoPlayer::markCurrentFrameConsumed(const oa::Event& inConsumed)
{
	if (frame_.image == VK_NULL_HANDLE || not inConsumed.isValid()) {
		return;
	}
	for (oa::Usize i = 0; i < rgbaPool_.size(); ++i) {
		if (rgbaPool_[i].image == frame_.image) {
			rgbaPoolConsumerEvents_[i] = inConsumed;
			return;
		}
	}
}

oa::Result<oa::Vector<oa::U8>> oa::VideoPlayer::readbackCurrentRgba()
{
	if (!decoder_.hasValue() || frame_.imageView == VK_NULL_HANDLE) {
		return oa::Status::error("oa::VideoPlayer::readbackCurrentRgba: no current frame");
	}
	return decoder_->readbackRgba(frame_);
}

oa::Result<oa::Matrix> oa::VideoPlayer::currentFrameToMatrix(bool inNormalizeImageNet)
{
	if (!decoder_.hasValue() || frame_.imageView == VK_NULL_HANDLE) {
		return oa::Status::error("oa::VideoPlayer::currentFrameToMatrix: no current frame");
	}
	return decoder_->convertFrameToBf16(frame_, inNormalizeImageNet);
}

oa::Result<oa::Image> oa::VideoPlayer::currentFrameToImage(bool inNormalizeImageNet)
{
	auto matrixResult = currentFrameToMatrix(inNormalizeImageNet);
	if (matrixResult.isError()) {
		return matrixResult.getStatus();
	}
	return oa::Image(
		oa::move(matrixResult).getValue(),
		oa::ImageLayout::Nchw,
		oa::ImageFormat::Rgb);
}

oa::Result<oa::VideoFrame> oa::VideoPlayer::acquireRgbaFromPool_()
{
	for (oa::Usize i = 0; i < rgbaPool_.size(); ++i) {
		const bool consumed = rgbaPoolConsumerEvents_[i].isComplete();
		if (not rgbaPoolBusy_[i] and consumed) {
			rgbaPoolBusy_[i] = true;
			rgbaPoolConsumerEvents_[i] = {};
			return rgbaPool_[i];
		}
	}
	const oa::Usize maxPoolSize = static_cast<oa::Usize>(cfg_.reorderDepth)
		+ presentationCacheCapacity_ + 3U;
	if (rgbaPool_.size() >= maxPoolSize) {
		for (oa::Usize i = 0; i < rgbaPool_.size(); ++i) {
			if (rgbaPoolBusy_[i]) {
				continue;
			}
			OA_RETURN_IF_ERROR(rgbaPoolConsumerEvents_[i].wait());
			rgbaPoolBusy_[i] = true;
			rgbaPoolConsumerEvents_[i] = {};
			return rgbaPool_[i];
		}
		return oa::Status::error(
			oa::StatusCode::ResourceExhausted,
			"oa::VideoPlayer RGBA pool exhausted while every slot is owned");
	}
	auto allocResult = decoder_->allocateRgbaFrame(
		demuxer_->getInfo().width,
		demuxer_->getInfo().height);
	if (not allocResult.isOk()) {
		return allocResult.getStatus();
	}
	rgbaPool_.pushBack(*allocResult);
	rgbaPoolBusy_.pushBack(true);
	rgbaPoolConsumerEvents_.pushBack({});
	return *allocResult;
}

void oa::VideoPlayer::releaseRgbaToPool_(const oa::VideoFrame& inFrame) {
	if (inFrame.image == VK_NULL_HANDLE) {
		return;
	}
	for (oa::Usize i = 0; i < rgbaPool_.size(); ++i) {
		if (rgbaPool_[i].image == inFrame.image) {
			// A presenter/encoder consumer event is already ordered after the
			// frame's conversion-ready dependency and is therefore the stronger
			// reuse edge. Frames released before reaching a consumer still retain
			// their conversion completion so seek/flush cannot recycle the target
			// while compute is writing it.
			if (not rgbaPoolConsumerEvents_[i].isValid()
				and inFrame.ready.isValid()) {
				rgbaPoolConsumerEvents_[i] = inFrame.ready;
			}
			rgbaPoolBusy_[i] = false;
			return;
		}
	}
}

oa::Status oa::VideoPlayer::decodeOneIntoReorder_(
	oa::U64 inMinimumRetainedPts
) {
	// Keep reading packets until we get a picture (skip parameter-set-only packets)
	while (true) {
		oa::VideoPacket packet;
		oa::Status readStatus = demuxer_->readNextPacket(packet);
		if (not readStatus.isOk()) {
			demuxerEosCurrent_ = true;
			return demuxer_->isEos() ? oa::Status::ok() : readStatus;
		}
		const auto& streamStats = demuxer_->getStats();
		if (streamStats.formatGeneration != demuxerFormatGeneration_
			or streamStats.reconnectCount != demuxerReconnectCount_) {
			demuxerFormatGeneration_ = streamStats.formatGeneration;
			demuxerReconnectCount_ = streamStats.reconnectCount;
			OA_RETURN_IF_ERROR(recreateDecoder_());
			demuxerEosCurrent_ = false;
		}

		auto decoded = decoder_->decode(
			oa::Span<const oa::U8>(packet.data.data(), packet.data.size()),
			packet.presentationTimestamp);
		if (not decoded.isOk()) return decoded.getStatus();
		oa::VideoFrame nv12 = oa::move(*decoded);

		// Skip parameter-set-only packets (no frame output)
		if (nv12.imageView == VK_NULL_HANDLE) {
			continue;
		}

		// Skip hidden AV1/VP9 frames (show_frame=0). They decode into the DPB
		// above (so later frames can reference them) but are never displayed;
		// enqueueing them would present duplicated / out-of-order frames.
		if (not nv12.shown) {
			continue;
		}
		++stats_.decodedFrames;
		const oa::U64 presentationPts = nv12.presentationTimestamp;
		if (presentationPts < inMinimumRetainedPts) {
			++stats_.seekReplayFrames;
			++stats_.seekReplayConversionsSkipped;
			reorder_.emplaceBack(VideoFrame{}, presentationPts);
			return oa::Status::ok();
		}
		if (inMinimumRetainedPts > 0U) ++stats_.seekReplayFrames;

		// convert NV12 → RGBA *now* into our own pool slot so the DPB layer is
		// free for the next decode. Holding NV12 across decodes is unsafe because
		// the H.264 sliding window evicts oldest slots and the allocator picks
		// them up for the next frame, trashing any data we'd be holding.
		auto rgbaResult = acquireRgbaFromPool_();
		if (not rgbaResult.isOk()) {
			return rgbaResult.getStatus();
		}
		oa::VideoFrame rgba = *rgbaResult;
		rgba.presentationTimestamp = presentationPts;

		oa::VideoConversionOptions options;
		options.convertToRgb = true;
		options.preferHardwareYCbCr = cfg_.preferHardwareYCbCr;
		options.filter = cfg_.filter;
		auto readyResult = decoder_->convertIntoAsync(nv12, options, rgba);
		if (not readyResult.isOk()) {
			releaseRgbaToPool_(rgba);
			return readyResult.getStatus();
		}
		stats_.hardwareYcbcrConversions =
			decoder_->getHardwareYcbcrDispatchCount();
		rgba.ready = *readyResult;

		reorder_.emplaceBack(rgba, presentationPts);
		return oa::Status::ok();
	} // end while
}

oa::Status oa::VideoPlayer::fillReorderBuffer_(
	oa::U64 inMinimumRetainedPts
) {
	// top up to depth+1 entries so the smallest-PTS entry is guaranteed
	// to be the next displayable frame (no late arrival can undercut it).
	while (reorder_.size() <= cfg_.reorderDepth and not demuxerEosCurrent_) {
		oa::Status s = decodeOneIntoReorder_(inMinimumRetainedPts);
		if (not s.isOk()) {
			return s;
		}
	}
	return oa::Status::ok();
}

oa::Status oa::VideoPlayer::popLowestPts_(ReorderEntry& outEntry) {
	if (reorder_.empty()) {
		return oa::Status::error("oa::VideoPlayer: reorder buffer empty");
	}
	oa::Usize minIdx = 0;
	oa::U64 minPts = reorder_[0].pts;
	for (oa::Usize i = 1; i < reorder_.size(); ++i) {
		if (reorder_[i].pts < minPts) {
			minPts = reorder_[i].pts;
			minIdx = i;
		}
	}
	outEntry = oa::move(reorder_[minIdx]);
	reorder_.erase(reorder_.data() + minIdx);
	return oa::Status::ok();
}

oa::Status oa::VideoPlayer::presentDecoded_(
	ReorderEntry&& inEntry,
	oa::Usize inDisplayIndex
) {
	if (presentationCacheCapacity_ == 0U) {
		releaseRgbaToPool_(frame_);
		frame_ = inEntry.rgba;
	} else {
		cachePresented_(inEntry.rgba, inEntry.pts, inDisplayIndex);
		frame_ = inEntry.rgba;
	}
	index_ = static_cast<oa::I64>(inDisplayIndex) + 1;
	reachedEos_ = false;
	accumulator_ = 0.0F;
	return oa::Status::ok();
}

bool oa::VideoPlayer::presentCached_(oa::Usize inDisplayIndex) {
	for (const PresentationEntry& entry : presentationCache_) {
		if (entry.displayIndex != inDisplayIndex) continue;
		frame_ = entry.rgba;
		index_ = static_cast<oa::I64>(inDisplayIndex) + 1;
		reachedEos_ = false;
		accumulator_ = 0.0F;
		++stats_.presentationCacheHits;
		return true;
	}
	return false;
}

void oa::VideoPlayer::cachePresented_(
	const VideoFrame& inFrame,
	oa::U64 inPts,
	oa::Usize inDisplayIndex
) {
	for (PresentationEntry& existing : presentationCache_) {
		if (existing.displayIndex != inDisplayIndex) continue;
		releaseRgbaToPool_(existing.rgba);
		existing = PresentationEntry{
			.rgba = inFrame,
			.pts = inPts,
			.displayIndex = inDisplayIndex,
		};
		return;
	}
	while (presentationCache_.size() >= presentationCacheCapacity_) {
		releaseRgbaToPool_(presentationCache_.front().rgba);
		presentationCache_.erase(presentationCache_.begin());
	}
	presentationCache_.pushBack(PresentationEntry{
		.rgba = inFrame,
		.pts = inPts,
		.displayIndex = inDisplayIndex,
	});
	stats_.presentationCacheResident = static_cast<oa::U32>(
		presentationCache_.size());
}

oa::Status oa::VideoPlayer::clearPresentationCache_() {
	frame_ = {};
	for (PresentationEntry& entry : presentationCache_) {
		releaseRgbaToPool_(entry.rgba);
	}
	presentationCache_.clear();
	stats_.presentationCacheResident = 0U;
	return oa::Status::ok();
}

oa::Usize oa::VideoPlayer::displayIndexForPts_(
	oa::U64 inPts,
	oa::Usize inMinimum
) const {
	oa::Usize low = oa::min(inMinimum, displayPts_.size());
	oa::Usize high = displayPts_.size();
	while (low < high) {
		const oa::Usize middle = low + (high - low) / 2U;
		if (displayPts_[middle] < inPts) low = middle + 1U;
		else high = middle;
	}
	return low;
}

oa::Status oa::VideoPlayer::clearReorder_() {
	for (ReorderEntry& entry : reorder_) {
		releaseRgbaToPool_(entry.rgba);
	}
	reorder_.clear();
	return oa::Status::ok();
}

oa::Status oa::VideoPlayer::waitForPoolConsumers_() {
	for (oa::Usize i = 0; i < rgbaPoolConsumerEvents_.size(); ++i) {
		OA_RETURN_IF_ERROR(rgbaPoolConsumerEvents_[i].wait());
		rgbaPoolConsumerEvents_[i] = {};
	}
	return oa::Status::ok();
}

oa::Status oa::VideoPlayer::resetDecoderForSeek_() {
	if (not decoder_.hasValue()) {
		return oa::Status::error(
			oa::StatusCode::FailedPrecondition,
			"oa::VideoPlayer seek reset requires an initialized decoder");
	}
	OA_RETURN_IF_ERROR(clearReorder_());
	OA_RETURN_IF_ERROR(decoder_->flush());
	++stats_.seekResets;
	return oa::Status::ok();
}

oa::Status oa::VideoPlayer::recreateDecoder_() {
	if (engine_ == nullptr or not demuxer_.hasValue()) {
		return oa::Status::error("oa::VideoPlayer::recreateDecoder_: not initialized");
	}

	OA_RETURN_IF_ERROR(clearReorder_());
	OA_RETURN_IF_ERROR(clearPresentationCache_());
	OA_RETURN_IF_ERROR(waitForPoolConsumers_());
	if (decoder_.hasValue()) {
		// conversion completion precedes the asynchronous DPB restore submit.
		// flush waits the decoder's latest timeline value, covering both,
		// before recreateDecoder_ closes the old session and image pool.
		OA_RETURN_IF_ERROR(decoder_->flush());
	}

	auto profile = demuxer_->getVideoProfile();
	profile.maxDpbSlots = cfg_.maxDpbSlots;
	auto decoderResult = oa::VideoDecoder::create(*engine_, profile);
	if (not decoderResult.isOk()) {
		return decoderResult.getStatus();
	}

	if (decoder_.hasValue()) {
		OA_RETURN_IF_ERROR(decoder_->close());
		decoder_.reset();
	}
	rgbaPool_.clear();
	rgbaPoolBusy_.clear();
	rgbaPoolConsumerEvents_.clear();
	frame_ = {};
	decoder_.emplace(oa::move(*decoderResult));
	decodeCursor_ = 0U;
	++stats_.decoderRecreations;
	return oa::Status::ok();
}

oa::Status oa::VideoPlayer::seekDisplayFrame_(oa::Usize inTargetFrameIndex) {
	if (inTargetFrameIndex >= displayPts_.size()) {
		return oa::Status::error(oa::StatusCode::InvalidArgument,
			"oa::VideoPlayer::seekDisplayFrame_: target out of range");
	}

	if (presentCached_(inTargetFrameIndex)) {
		reachedEos_ = false;
		return oa::Status::ok();
	}
	const oa::Usize history = oa::max<oa::Usize>(
		1U, presentationCacheCapacity_);
	const oa::Usize seekStart = inTargetFrameIndex + 1U > history
		? inTargetFrameIndex + 1U - history
		: 0U;
	OA_RETURN_IF_ERROR(clearPresentationCache_());
	OA_RETURN_IF_ERROR(resetDecoderForSeek_());
	OA_RETURN_IF_ERROR(demuxer_->seek(displayPts_[seekStart]));
	demuxerEosCurrent_ = false;
	reachedEos_ = false;

	oa::Usize minimumIndex = 0U;
	const oa::U64 minimumRetainedPts = displayPts_[seekStart];
	while (true) {
		OA_RETURN_IF_ERROR(fillReorderBuffer_(minimumRetainedPts));
		if (reorder_.empty()) {
			return oa::Status::error(
				"oa::VideoPlayer::seekDisplayFrame_: target was not decoded");
		}
		ReorderEntry entry;
		OA_RETURN_IF_ERROR(popLowestPts_(entry));
		const oa::Usize decodedIndex = displayIndexForPts_(
			entry.pts, minimumIndex);
		if (decodedIndex >= displayPts_.size()
			or decodedIndex > inTargetFrameIndex) {
			releaseRgbaToPool_(entry.rgba);
			return oa::Status::error(
				oa::StatusCode::DataLoss,
				"oa::VideoPlayer seek skipped the requested display frame");
		}
		if (decodedIndex >= seekStart) {
			if (entry.rgba.imageView == VK_NULL_HANDLE) {
				return oa::Status::error(
					oa::StatusCode::DataLoss,
					"oa::VideoPlayer seek discarded a retained display frame");
			}
			OA_RETURN_IF_ERROR(presentDecoded_(
				oa::move(entry), decodedIndex));
		}
		minimumIndex = decodedIndex + 1U;
		decodeCursor_ = minimumIndex;
		if (decodedIndex == inTargetFrameIndex) break;
	}
	return oa::Status::ok();
}
