#include <oa/vision/videoRecorder.h>

#include <oa/runtime/allocator.h>
#include <oa/runtime/executionSession.h>
#include <oa/runtime/engine.h>
#include <oa/runtime/engine/allocatorAccess.h>
#include <oa/runtime/texture.h>
#include <oa/vision/fnVideo.h>

#include "../../runtime/textureAccess.h"
#include "encoder/videoEncoderInternal.h"

oa::VideoRecorder::VideoRecorder(oa::VideoRecorder&& inOther) noexcept {
	moveFrom_(oa::move(inOther));
}

oa::VideoRecorder& oa::VideoRecorder::operator=(oa::VideoRecorder&& inOther) noexcept {
	if (this != &inOther) {
		moveFrom_(oa::move(inOther));
	}
	return *this;
}

oa::VideoRecorder::~VideoRecorder() {
	// GPU work belongs to encoder_. Its destructor transfers an unfinished
	// session to engine retirement; the remaining members release host-only
	// state without finalizing or manufacturing a partial container.
}

void oa::VideoRecorder::moveFrom_(oa::VideoRecorder&& inOther) noexcept {
	engine_ = inOther.engine_;
	config_ = oa::move(inOther.config_);
	encoder_ = oa::move(inOther.encoder_);
	muxer_ = oa::move(inOther.muxer_);
	audioEncoder_ = oa::move(inOther.audioEncoder_);
	pendingAudio_ = oa::move(inOther.pendingAudio_);
	audioScratch_ = oa::move(inOther.audioScratch_);
	firstVideoPts_ = inOther.firstVideoPts_;
	nextAudioFrame_ = inOther.nextAudioFrame_;
	hasFirstVideoPts_ = inOther.hasFirstVideoPts_;
	submittedFrameCount_ = inOther.submittedFrameCount_;
	muxedFrameCount_ = inOther.muxedFrameCount_;
	codecConfigWritten_ = inOther.codecConfigWritten_;
	finalized_ = inOther.finalized_;

	inOther.engine_ = nullptr;
	inOther.submittedFrameCount_ = 0;
	inOther.muxedFrameCount_ = 0;
	inOther.codecConfigWritten_ = false;
	inOther.firstVideoPts_ = 0U;
	inOther.nextAudioFrame_ = 0U;
	inOther.hasFirstVideoPts_ = false;
	inOther.finalized_ = true;
}

oa::Result<oa::VideoRecorder> oa::VideoRecorder::create(
	oa::Engine& inEngine,
	const oa::VideoRecorderConfig& inConfig)
{
	if (inConfig.outputPath.empty()) {
		return oa::Status::error(oa::StatusCode::InvalidArgument,
			"oa::VideoRecorder output path must not be empty");
	}
	if (inConfig.encode.width == 0U or inConfig.encode.height == 0U) {
		return oa::Status::error(oa::StatusCode::InvalidArgument,
			"oa::VideoRecorder requires a non-zero encode extent");
	}

	auto encoderResult = oa::VideoEncoder::create(inEngine, inConfig.encode);
	if (not encoderResult.isOk()) return encoderResult.getStatus();

	oa::VideoMuxerConfig muxInfo = {};
	muxInfo.outputPath = inConfig.outputPath;
	muxInfo.codec = inConfig.encode.codec;
	muxInfo.width = inConfig.encode.width;
	muxInfo.height = inConfig.encode.height;
	muxInfo.frameRate = inConfig.encode.frameRate;
	muxInfo.audioEnabled = inConfig.audioEnabled;
	muxInfo.audioCodec = inConfig.audio.codec;
	muxInfo.audioSampleRate = inConfig.audio.sampleRate;
	muxInfo.audioChannelCount = inConfig.audio.channelCount;
	oa::AudioEncoder audioEncoder;
	if (inConfig.audioEnabled) {
		auto audioResult = oa::AudioEncoder::create(inConfig.audio);
		if (not audioResult.isOk()) return audioResult.getStatus();
		audioEncoder = oa::move(*audioResult);
		muxInfo.audioPrimingFrames = audioEncoder.getPrimingFrames();
	}
	auto muxerResult = oa::VideoMuxer::create(muxInfo);
	if (not muxerResult.isOk()) return muxerResult.getStatus();

	oa::VideoRecorder recorder;
	recorder.engine_ = &inEngine;
	recorder.config_ = inConfig;
	recorder.encoder_ = oa::move(*encoderResult);
	recorder.muxer_ = oa::move(*muxerResult);
	recorder.audioEncoder_ = oa::move(audioEncoder);
	if (inConfig.audioEnabled) {
		recorder.muxer_.setAudioCodecConfig(recorder.audioEncoder_.getCodecConfig());
	}
	return oa::Result<oa::VideoRecorder>(oa::move(recorder));
}

oa::Status oa::VideoRecorder::writeEncoded_(const oa::EncodedVideoPacket& inFrame) {
	if (not codecConfigWritten_ and inFrame.isKeyframe) {
		auto bytes = oa::Span<const oa::U8>(inFrame.bitstream.data(), inFrame.bitstream.size());
		if (config_.encode.codec == oa::VideoCodec::H264) {
			auto sps = oa::FnVideo::extractSps(bytes);
			auto pps = oa::FnVideo::extractPps(bytes);
			if (sps.empty() or pps.empty()) {
				return oa::Status::error(oa::StatusCode::DataLoss,
					"Encoded H.264 keyframe does not contain SPS/PPS");
			}
			muxer_.setCodecConfig(sps, pps);
		} else if (config_.encode.codec == oa::VideoCodec::H265) {
			auto vps = oa::FnVideo::extractVpsH265(bytes);
			auto sps = oa::FnVideo::extractSpsH265(bytes);
			auto pps = oa::FnVideo::extractPpsH265(bytes);
			if (vps.empty() or sps.empty() or pps.empty()) {
				return oa::Status::error(oa::StatusCode::DataLoss,
					"Encoded H.265 keyframe does not contain VPS/SPS/PPS");
			}
			muxer_.setCodecConfig(vps, sps, pps);
		} else {
			return oa::Status::error(oa::StatusCode::Unimplemented,
				"oa::VideoRecorder container configuration supports H.264/H.265 only");
		}
		codecConfigWritten_ = true;
	}
	OA_RETURN_IF_ERROR(muxer_.writePacket(inFrame));
	++muxedFrameCount_;
	return oa::Status::ok();
}

oa::Status oa::VideoRecorder::writeRgba(
	const oavk::Buffer& inRgba,
	oa::U32 inWidth,
	oa::U32 inHeight,
	oa::U64 inPts)
{
	if (not isOpen()) {
		return oa::Status::error(oa::StatusCode::FailedPrecondition,
			"oa::VideoRecorder is not open");
	}
	if (inWidth != config_.encode.width or inHeight != config_.encode.height) {
		return oa::Status::error(oa::StatusCode::InvalidArgument,
			"oa::VideoRecorder frame extent does not match its fixed encode profile");
	}
	OA_RETURN_IF_ERROR(setFirstVideoPts_(inPts));
	oa::Vector<oa::EncodedVideoPacket> ready;
	OA_RETURN_IF_ERROR(oa::VideoEncoderAccess::submitRgba(encoder_,
		inRgba, inWidth, inHeight, inPts, ready,
		config_.colorSpace, config_.fullRange));
	++submittedFrameCount_;
	for (const auto& encoded : ready) OA_RETURN_IF_ERROR(writeEncoded_(encoded));
	return oa::Status::ok();
}

oa::Status oa::VideoRecorder::write(const oa::VideoFrame& inFrame) {
	oa::Event consumed;
	const oa::Status status = writeAsync(inFrame, consumed);
	const oa::Status waitStatus = consumed.wait();
	return status.isOk() ? waitStatus : status;
}

oa::Status oa::VideoRecorder::writeAsync(
	const oa::VideoFrame& inFrame,
	oa::Event& outInputConsumed)
{
	outInputConsumed = {};
	if (not isOpen()) {
		return oa::Status::error(oa::StatusCode::FailedPrecondition,
			"oa::VideoRecorder is not open");
	}
	if (inFrame.width != config_.encode.width or inFrame.height != config_.encode.height) {
		return oa::Status::error(oa::StatusCode::InvalidArgument,
			"oa::VideoRecorder frame extent does not match its fixed encode profile");
	}
	OA_RETURN_IF_ERROR(setFirstVideoPts_(inFrame.presentationTimestamp));
	const oa::YCbCrModel colorSpace = inFrame.colorSpace == oa::YCbCrModel::Auto
		? config_.colorSpace : inFrame.colorSpace;
	oa::Vector<oa::EncodedVideoPacket> ready;
	if (inFrame.resource == oa::VideoFrameResource::Buffer and inFrame.buffer != nullptr
		and inFrame.isRgb and inFrame.format == VK_FORMAT_R8G8B8A8_UNORM) {
		OA_RETURN_IF_ERROR(oa::VideoEncoderAccess::submitRgba(encoder_,
			*inFrame.buffer, inFrame.width, inFrame.height,
			inFrame.presentationTimestamp, ready, colorSpace, inFrame.fullRange));
	} else if (inFrame.resource == oa::VideoFrameResource::Image
		and inFrame.image != VK_NULL_HANDLE and inFrame.imageView != VK_NULL_HANDLE
		and inFrame.isRgb) {
		OA_RETURN_IF_ERROR(oa::VideoEncoderAccess::submitRgbaImage(encoder_,
			inFrame.image, inFrame.imageView, inFrame.format, inFrame.layout,
			inFrame.width, inFrame.height, inFrame.presentationTimestamp,
			ready, colorSpace, inFrame.fullRange, inFrame.arrayLayer,
			inFrame.ready,
			inFrame.externalQueueFamilyIndex, &outInputConsumed));
	} else {
		return oa::Status::error(oa::StatusCode::InvalidArgument,
			"oa::VideoRecorder requires a buffer- or image-backed RGBA8/BGRA8 frame");
	}
	++submittedFrameCount_;
	for (const auto& encoded : ready) OA_RETURN_IF_ERROR(writeEncoded_(encoded));
	return oa::Status::ok();
}

oa::Status oa::VideoRecorder::write(const oa::Texture& inTexture,	oa::U64 inPts) {
	if (engine_ == nullptr) {
		return oa::Status::error(oa::StatusCode::FailedPrecondition,
			"oa::VideoRecorder is not open");
	}
	auto& context = oa::ExecutionSession::forEngine(*engine_);
	auto frame = oa::FnVideo::fromTexture(inTexture, inPts);
	if (not frame.isOk()) return frame.getStatus();
	if (frame->resource == oa::VideoFrameResource::Buffer) {
		const oavk::Buffer* buffer = oa::TextureAccess::buffer(inTexture);
		const oa::U64 bytes = static_cast<oa::U64>(inTexture.width())
			* static_cast<oa::U64>(inTexture.height()) * 4U;
		if (buffer == nullptr or buffer->buffer == nullptr or buffer->size < bytes
			or buffer->bindlessIndex == OA_BINDLESS_INVALID
			or buffer->allocation == nullptr or buffer->aliasIdentity != nullptr
			or oa::TextureAccess::engine(inTexture) != engine_
			or buffer->allocatorIdentity != oa::EngineAllocatorAccess::get(*engine_).allocator) {
			return oa::Status::error(oa::StatusCode::InvalidArgument,
				"oa::VideoRecorder texture must be a non-aliased buffer owned by its context engine");
		}
		OA_RETURN_IF_ERROR(context.submitAndWait());
	}
	return write(*frame);
}

oa::Status oa::VideoRecorder::writeAudioPackets_(oa::Vector<oa::EncodedAudioPacket>& inPackets)
{
	for (const auto& packet : inPackets) {
		OA_RETURN_IF_ERROR(muxer_.writeAudioPacket(packet));
	}
	inPackets.clear();
	return oa::Status::ok();
}

oa::Status oa::VideoRecorder::writeAudioAligned_(
	oa::Span<const oa::F32> inInterleaved,
	oa::U64 inPts)
{
	const oa::U64 channels = config_.audio.channelCount;
	const oa::U64 inputFrames = inInterleaved.size() / channels;
	const oa::I64 deltaUs = inPts >= firstVideoPts_
		? static_cast<oa::I64>(inPts - firstVideoPts_)
		: -static_cast<oa::I64>(firstVideoPts_ - inPts);
	const oa::I64 desiredStart = deltaUs * static_cast<oa::I64>(config_.audio.sampleRate)
		/ 1'000'000LL;
	const oa::I64 desiredEnd = desiredStart + static_cast<oa::I64>(inputFrames);
	if (desiredEnd <= static_cast<oa::I64>(nextAudioFrame_)) return oa::Status::ok();

	oa::U64 trimFrames = 0U;
	if (desiredStart < static_cast<oa::I64>(nextAudioFrame_)) {
		trimFrames = static_cast<oa::U64>(static_cast<oa::I64>(nextAudioFrame_) - desiredStart);
	}
	if (desiredStart > static_cast<oa::I64>(nextAudioFrame_)) {
		const oa::U64 gapFrames = static_cast<oa::U64>(desiredStart) - nextAudioFrame_;
		audioScratch_.resize(static_cast<oa::Usize>(gapFrames * channels), 0.0F);
		oa::Vector<oa::EncodedAudioPacket> packets;
		OA_RETURN_IF_ERROR(audioEncoder_.encode(
			oa::Span<const oa::F32>(audioScratch_.data(), audioScratch_.size()), packets));
		OA_RETURN_IF_ERROR(writeAudioPackets_(packets));
		nextAudioFrame_ += gapFrames;
	}
	if (trimFrames >= inputFrames) return oa::Status::ok();
	const oa::U64 acceptedFrames = inputFrames - trimFrames;
	const oa::Usize sampleOffset = static_cast<oa::Usize>(trimFrames * channels);
	const oa::Usize sampleCount = static_cast<oa::Usize>(acceptedFrames * channels);
	oa::Vector<oa::EncodedAudioPacket> packets;
	OA_RETURN_IF_ERROR(audioEncoder_.encode(
		oa::Span<const oa::F32>(inInterleaved.data() + sampleOffset, sampleCount), packets));
	OA_RETURN_IF_ERROR(writeAudioPackets_(packets));
	nextAudioFrame_ += acceptedFrames;
	return oa::Status::ok();
}

oa::Status oa::VideoRecorder::setFirstVideoPts_(oa::U64 inPts)
{
	if (hasFirstVideoPts_) return oa::Status::ok();
	firstVideoPts_ = inPts;
	hasFirstVideoPts_ = true;
	for (auto& chunk : pendingAudio_) {
		OA_RETURN_IF_ERROR(writeAudioAligned_(
			oa::Span<const oa::F32>(chunk.samples.data(), chunk.samples.size()), chunk.pts));
	}
	pendingAudio_.clear();
	return oa::Status::ok();
}

oa::Status oa::VideoRecorder::writeAudio(
	oa::Span<const oa::F32> inInterleaved,
	oa::U32 inSampleRate,
	oa::U32 inChannelCount,
	oa::U64 inPts)
{
	if (!isOpen() || !config_.audioEnabled || !audioEncoder_.isOpen()) {
		return oa::Status::error(oa::StatusCode::FailedPrecondition,
			"oa::VideoRecorder audio was not enabled");
	}
	if (inSampleRate != config_.audio.sampleRate
		|| inChannelCount != config_.audio.channelCount) {
		return oa::Status::invalidArgument(
			"Audio chunk format does not match the recorder profile");
	}
	if (inInterleaved.empty() || inInterleaved.size() % inChannelCount != 0U) {
		return oa::Status::invalidArgument("Audio chunk contains incomplete interleaved frames");
	}
	if (!hasFirstVideoPts_) {
		PendingAudioChunk chunk;
		chunk.samples.resize(inInterleaved.size());
		oa::memcpy(chunk.samples.data(), inInterleaved.data(),
			inInterleaved.size() * sizeof(oa::F32));
		chunk.pts = inPts;
		pendingAudio_.pushBack(oa::move(chunk));
		return oa::Status::ok();
	}
	return writeAudioAligned_(inInterleaved, inPts);
}

oa::Status oa::VideoRecorder::writeAudio(const oa::AudioCaptureChunk& inChunk)
{
	return writeAudio(
		oa::Span<const oa::F32>(inChunk.interleaved.data(), inChunk.interleaved.size()),
		inChunk.sampleRate,
		inChunk.channelCount,
		inChunk.presentationTimestamp);
}

oa::Status oa::VideoRecorder::finalize() {
	if (finalized_) return oa::Status::ok();
	if (engine_ == nullptr) {
		return oa::Status::error(oa::StatusCode::FailedPrecondition,
			"oa::VideoRecorder was not created");
	}
	oa::Vector<oa::EncodedVideoPacket> remaining;
	OA_RETURN_IF_ERROR(encoder_.flush(remaining));
	for (const auto& frame : remaining) {
		OA_RETURN_IF_ERROR(writeEncoded_(frame));
	}
	if (config_.audioEnabled && audioEncoder_.isOpen()) {
		if (!hasFirstVideoPts_) pendingAudio_.clear();
		oa::Vector<oa::EncodedAudioPacket> audioPackets;
		OA_RETURN_IF_ERROR(audioEncoder_.flush(audioPackets));
		OA_RETURN_IF_ERROR(writeAudioPackets_(audioPackets));
	}
	if (submittedFrameCount_ == 0U or muxedFrameCount_ == 0U) {
		return oa::Status::error(oa::StatusCode::FailedPrecondition,
			"oa::VideoRecorder cannot finalize an empty recording");
	}
	OA_RETURN_IF_ERROR(muxer_.finalize());
	finalized_ = true;
	return oa::Status::ok();
}

oa::Status oa::VideoRecorder::close() {
	// Finalization remains separate: closing an abandoned recorder must not
	// manufacture a partial file that appears complete.
	oa::Status firstError = encoder_.close();
	const oa::Status audioStatus = audioEncoder_.close();
	if (firstError.isOk() and not audioStatus.isOk()) firstError = audioStatus;
	const oa::Status muxerStatus = muxer_.close();
	if (firstError.isOk() and not muxerStatus.isOk()) firstError = muxerStatus;
	pendingAudio_.clear();
	audioScratch_.clear();
	engine_ = nullptr;
	submittedFrameCount_ = 0;
	muxedFrameCount_ = 0;
	codecConfigWritten_ = false;
	firstVideoPts_ = 0U;
	nextAudioFrame_ = 0U;
	hasFirstVideoPts_ = false;
	finalized_ = true;
	return firstError;
}
