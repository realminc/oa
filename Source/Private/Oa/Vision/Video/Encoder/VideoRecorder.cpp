#include <Oa/Vision/VideoRecorder.h>

#include <Oa/Runtime/Allocator.h>
#include <Oa/Runtime/Context.h>
#include <Oa/Runtime/Engine.h>
#include <Oa/Ui/Image.h>
#include <Oa/Vision/FnVideo.h>

namespace {

OaContext& ContextForEngine(OaEngine& InEngine) {
	OaContext* active = OaContext::GetDefaultPtr();
	return active != nullptr and active->GetEngine() == &InEngine
		? *active : InEngine.GetContext();
}

} // namespace

OaVideoRecorder::OaVideoRecorder(OaVideoRecorder&& InOther) noexcept {
	MoveFrom_(OaStdMove(InOther));
}

OaVideoRecorder& OaVideoRecorder::operator=(OaVideoRecorder&& InOther) noexcept {
	if (this != &InOther) {
		Destroy();
		MoveFrom_(OaStdMove(InOther));
	}
	return *this;
}

OaVideoRecorder::~OaVideoRecorder() {
	// GPU work belongs to Encoder_. Its destructor transfers an unfinished
	// session to engine retirement; the remaining members release host-only
	// state without finalizing or manufacturing a partial container.
}

void OaVideoRecorder::MoveFrom_(OaVideoRecorder&& InOther) noexcept {
	Engine_ = InOther.Engine_;
	Config_ = OaStdMove(InOther.Config_);
	Encoder_ = OaStdMove(InOther.Encoder_);
	Muxer_ = OaStdMove(InOther.Muxer_);
	AudioEncoder_ = OaStdMove(InOther.AudioEncoder_);
	PendingAudio_ = OaStdMove(InOther.PendingAudio_);
	AudioScratch_ = OaStdMove(InOther.AudioScratch_);
	FirstVideoPts_ = InOther.FirstVideoPts_;
	NextAudioFrame_ = InOther.NextAudioFrame_;
	HasFirstVideoPts_ = InOther.HasFirstVideoPts_;
	SubmittedFrameCount_ = InOther.SubmittedFrameCount_;
	MuxedFrameCount_ = InOther.MuxedFrameCount_;
	CodecConfigWritten_ = InOther.CodecConfigWritten_;
	Finalized_ = InOther.Finalized_;

	InOther.Engine_ = nullptr;
	InOther.SubmittedFrameCount_ = 0;
	InOther.MuxedFrameCount_ = 0;
	InOther.CodecConfigWritten_ = false;
	InOther.FirstVideoPts_ = 0U;
	InOther.NextAudioFrame_ = 0U;
	InOther.HasFirstVideoPts_ = false;
	InOther.Finalized_ = true;
}

OaResult<OaVideoRecorder> OaVideoRecorder::Create(
	OaEngine& InEngine,
	const OaVideoRecorderConfig& InConfig)
{
	if (InConfig.OutputPath.empty()) {
		return OaStatus::Error(OaStatusCode::InvalidArgument,
			"OaVideoRecorder output path must not be empty");
	}
	if (InConfig.Encode.Width == 0U or InConfig.Encode.Height == 0U) {
		return OaStatus::Error(OaStatusCode::InvalidArgument,
			"OaVideoRecorder requires a non-zero encode extent");
	}

	auto encoderResult = OaVideoEncoder::Create(InEngine, InConfig.Encode);
	if (not encoderResult.IsOk()) return encoderResult.GetStatus();

	OaVideoMuxer::CreateInfo muxInfo = {};
	muxInfo.Codec = InConfig.Encode.Codec;
	muxInfo.Width = InConfig.Encode.Width;
	muxInfo.Height = InConfig.Encode.Height;
	muxInfo.FrameRate = InConfig.Encode.FrameRate;
	muxInfo.AudioEnabled = InConfig.AudioEnabled;
	muxInfo.AudioCodec = InConfig.Audio.Codec;
	muxInfo.AudioSampleRate = InConfig.Audio.SampleRate;
	muxInfo.AudioChannelCount = InConfig.Audio.ChannelCount;
	OaAudioStreamEncoder audioEncoder;
	if (InConfig.AudioEnabled) {
		auto audioResult = OaAudioStreamEncoder::Create(InConfig.Audio);
		if (not audioResult.IsOk()) return audioResult.GetStatus();
		audioEncoder = OaStdMove(*audioResult);
		muxInfo.AudioPrimingFrames = audioEncoder.GetPrimingFrames();
	}
	auto muxerResult = OaVideoMuxer::Create(InConfig.OutputPath.CStr(), muxInfo);
	if (not muxerResult.IsOk()) return muxerResult.GetStatus();

	OaVideoRecorder recorder;
	recorder.Engine_ = &InEngine;
	recorder.Config_ = InConfig;
	recorder.Encoder_ = OaStdMove(*encoderResult);
	recorder.Muxer_ = OaStdMove(*muxerResult);
	recorder.AudioEncoder_ = OaStdMove(audioEncoder);
	if (InConfig.AudioEnabled) {
		recorder.Muxer_.SetAudioCodecConfig(recorder.AudioEncoder_.GetCodecConfig());
	}
	return OaResult<OaVideoRecorder>(OaStdMove(recorder));
}

OaStatus OaVideoRecorder::WriteEncoded_(const OaEncodedFrame& InFrame) {
	if (not CodecConfigWritten_ and InFrame.IsKeyframe) {
		auto bytes = OaSpan<const OaU8>(InFrame.Bitstream.Data(), InFrame.Bitstream.Size());
		if (Config_.Encode.Codec == OaVideoCodec::H264) {
			auto sps = OaFnVideo::ExtractSps(bytes);
			auto pps = OaFnVideo::ExtractPps(bytes);
			if (sps.Empty() or pps.Empty()) {
				return OaStatus::Error(OaStatusCode::DataLoss,
					"Encoded H.264 keyframe does not contain SPS/PPS");
			}
			Muxer_.SetCodecConfig(sps, pps);
		} else if (Config_.Encode.Codec == OaVideoCodec::H265) {
			auto vps = OaFnVideo::ExtractVpsH265(bytes);
			auto sps = OaFnVideo::ExtractSpsH265(bytes);
			auto pps = OaFnVideo::ExtractPpsH265(bytes);
			if (vps.Empty() or sps.Empty() or pps.Empty()) {
				return OaStatus::Error(OaStatusCode::DataLoss,
					"Encoded H.265 keyframe does not contain VPS/SPS/PPS");
			}
			Muxer_.SetCodecConfig(vps, sps, pps);
		} else {
			return OaStatus::Error(OaStatusCode::Unimplemented,
				"OaVideoRecorder container configuration supports H.264/H.265 only");
		}
		CodecConfigWritten_ = true;
	}
	OA_RETURN_IF_ERROR(Muxer_.WritePacket(InFrame));
	++MuxedFrameCount_;
	return OaStatus::Ok();
}

OaStatus OaVideoRecorder::WriteRgba(
	const OaVkBuffer& InRgba,
	OaU32 InWidth,
	OaU32 InHeight,
	OaU64 InPts)
{
	if (not IsOpen()) {
		return OaStatus::Error(OaStatusCode::FailedPrecondition,
			"OaVideoRecorder is not open");
	}
	if (InWidth != Config_.Encode.Width or InHeight != Config_.Encode.Height) {
		return OaStatus::Error(OaStatusCode::InvalidArgument,
			"OaVideoRecorder frame extent does not match its fixed encode profile");
	}
	OA_RETURN_IF_ERROR(SetFirstVideoPts_(InPts));
	OaVec<OaEncodedFrame> ready;
	OA_RETURN_IF_ERROR(Encoder_.SubmitRgba(
		InRgba, InWidth, InHeight, InPts, ready,
		Config_.ColorSpace, Config_.FullRange));
	++SubmittedFrameCount_;
	for (const auto& encoded : ready) OA_RETURN_IF_ERROR(WriteEncoded_(encoded));
	return OaStatus::Ok();
}

OaStatus OaVideoRecorder::Write(const OaVideoFrame& InFrame) {
	OaCompletionToken consumed;
	const OaStatus status = WriteAsync(InFrame, consumed);
	const OaStatus waitStatus = consumed.Wait();
	return status.IsOk() ? waitStatus : status;
}

OaStatus OaVideoRecorder::WriteAsync(
	const OaVideoFrame& InFrame,
	OaCompletionToken& OutInputConsumed)
{
	OutInputConsumed = {};
	if (not IsOpen()) {
		return OaStatus::Error(OaStatusCode::FailedPrecondition,
			"OaVideoRecorder is not open");
	}
	if (InFrame.Width != Config_.Encode.Width or InFrame.Height != Config_.Encode.Height) {
		return OaStatus::Error(OaStatusCode::InvalidArgument,
			"OaVideoRecorder frame extent does not match its fixed encode profile");
	}
	OA_RETURN_IF_ERROR(SetFirstVideoPts_(InFrame.PresentationTimestamp));
	const OaYCbCrModel colorSpace = InFrame.ColorSpace == OaYCbCrModel::Auto
		? Config_.ColorSpace : InFrame.ColorSpace;
	OaVec<OaEncodedFrame> ready;
	if (InFrame.Resource == OaVideoFrameResource::Buffer and InFrame.Buffer != nullptr
		and InFrame.IsRgb and InFrame.Format == VK_FORMAT_R8G8B8A8_UNORM) {
		OA_RETURN_IF_ERROR(Encoder_.SubmitRgba(
			*InFrame.Buffer, InFrame.Width, InFrame.Height,
			InFrame.PresentationTimestamp, ready, colorSpace, InFrame.FullRange));
	} else if (InFrame.Resource == OaVideoFrameResource::Image
		and InFrame.Image != VK_NULL_HANDLE and InFrame.ImageView != VK_NULL_HANDLE
		and InFrame.IsRgb) {
		OA_RETURN_IF_ERROR(Encoder_.SubmitRgbaImage(
			InFrame.Image, InFrame.ImageView, InFrame.Format, InFrame.Layout,
			InFrame.Width, InFrame.Height, InFrame.PresentationTimestamp,
			ready, colorSpace, InFrame.FullRange, InFrame.ArrayLayer,
			InFrame.Ready,
			InFrame.ExternalQueueFamilyIndex, &OutInputConsumed));
	} else {
		return OaStatus::Error(OaStatusCode::InvalidArgument,
			"OaVideoRecorder requires a buffer- or image-backed RGBA8/BGRA8 frame");
	}
	++SubmittedFrameCount_;
	for (const auto& encoded : ready) OA_RETURN_IF_ERROR(WriteEncoded_(encoded));
	return OaStatus::Ok();
}

OaStatus OaVideoRecorder::Write(
	OaContext& InContext,
	const OaTexture& InTexture,
	OaU64 InPts)
{
	if (Engine_ == nullptr) {
		return OaStatus::Error(OaStatusCode::FailedPrecondition,
			"OaVideoRecorder is not open");
	}
	if (InContext.GetEngine() != Engine_) {
		return OaStatus::Error(OaStatusCode::InvalidArgument,
			"OaVideoRecorder texture context does not own this recorder engine");
	}
	auto frame = OaFnVideo::FromTexture(InTexture, InPts);
	if (not frame.IsOk()) return frame.GetStatus();
	if (frame->Resource == OaVideoFrameResource::Buffer) {
		const OaVkBuffer& buffer = InTexture.DeviceBuf;
		const OaU64 bytes = static_cast<OaU64>(InTexture.Width)
			* static_cast<OaU64>(InTexture.Height) * 4U;
		if (buffer.Buffer == nullptr or buffer.Size < bytes
			or buffer.BindlessIndex == OA_BINDLESS_INVALID
			or buffer.Allocation == nullptr or buffer.AliasIdentity != nullptr
			or buffer.IsImported() or buffer.NodeIndex != 0U
			or buffer.AllocatorIdentity != Engine_->Allocator.Allocator) {
			return OaStatus::Error(OaStatusCode::InvalidArgument,
				"OaVideoRecorder texture must be a non-aliased buffer owned by its context engine");
		}
		OA_RETURN_IF_ERROR(InContext.Execute());
		OA_RETURN_IF_ERROR(InContext.Sync());
	}
	return Write(*frame);
}

OaStatus OaVideoRecorder::Write(const OaTexture& InTexture, OaU64 InPts) {
	if (Engine_ == nullptr) {
		return OaStatus::Error(OaStatusCode::FailedPrecondition,
			"OaVideoRecorder is not open");
	}
	return Write(ContextForEngine(*Engine_), InTexture, InPts);
}

OaStatus OaVideoRecorder::WriteAudioPackets_(OaVec<OaEncodedAudioPacket>& InPackets)
{
	for (const auto& packet : InPackets) {
		OA_RETURN_IF_ERROR(Muxer_.WriteAudioPacket(packet));
	}
	InPackets.Clear();
	return OaStatus::Ok();
}

OaStatus OaVideoRecorder::WriteAudioAligned_(
	OaSpan<const OaF32> InInterleaved,
	OaU64 InPts)
{
	const OaU64 channels = Config_.Audio.ChannelCount;
	const OaU64 inputFrames = InInterleaved.Size() / channels;
	const OaI64 deltaUs = InPts >= FirstVideoPts_
		? static_cast<OaI64>(InPts - FirstVideoPts_)
		: -static_cast<OaI64>(FirstVideoPts_ - InPts);
	const OaI64 desiredStart = deltaUs * static_cast<OaI64>(Config_.Audio.SampleRate)
		/ 1'000'000LL;
	const OaI64 desiredEnd = desiredStart + static_cast<OaI64>(inputFrames);
	if (desiredEnd <= static_cast<OaI64>(NextAudioFrame_)) return OaStatus::Ok();

	OaU64 trimFrames = 0U;
	if (desiredStart < static_cast<OaI64>(NextAudioFrame_)) {
		trimFrames = static_cast<OaU64>(static_cast<OaI64>(NextAudioFrame_) - desiredStart);
	}
	if (desiredStart > static_cast<OaI64>(NextAudioFrame_)) {
		const OaU64 gapFrames = static_cast<OaU64>(desiredStart) - NextAudioFrame_;
		AudioScratch_.Resize(static_cast<OaUsize>(gapFrames * channels), 0.0F);
		OaVec<OaEncodedAudioPacket> packets;
		OA_RETURN_IF_ERROR(AudioEncoder_.Encode(
			OaSpan<const OaF32>(AudioScratch_.Data(), AudioScratch_.Size()), packets));
		OA_RETURN_IF_ERROR(WriteAudioPackets_(packets));
		NextAudioFrame_ += gapFrames;
	}
	if (trimFrames >= inputFrames) return OaStatus::Ok();
	const OaU64 acceptedFrames = inputFrames - trimFrames;
	const OaUsize sampleOffset = static_cast<OaUsize>(trimFrames * channels);
	const OaUsize sampleCount = static_cast<OaUsize>(acceptedFrames * channels);
	OaVec<OaEncodedAudioPacket> packets;
	OA_RETURN_IF_ERROR(AudioEncoder_.Encode(
		OaSpan<const OaF32>(InInterleaved.Data() + sampleOffset, sampleCount), packets));
	OA_RETURN_IF_ERROR(WriteAudioPackets_(packets));
	NextAudioFrame_ += acceptedFrames;
	return OaStatus::Ok();
}

OaStatus OaVideoRecorder::SetFirstVideoPts_(OaU64 InPts)
{
	if (HasFirstVideoPts_) return OaStatus::Ok();
	FirstVideoPts_ = InPts;
	HasFirstVideoPts_ = true;
	for (auto& chunk : PendingAudio_) {
		OA_RETURN_IF_ERROR(WriteAudioAligned_(
			OaSpan<const OaF32>(chunk.Samples.Data(), chunk.Samples.Size()), chunk.Pts));
	}
	PendingAudio_.Clear();
	return OaStatus::Ok();
}

OaStatus OaVideoRecorder::WriteAudio(
	OaSpan<const OaF32> InInterleaved,
	OaU32 InSampleRate,
	OaU32 InChannelCount,
	OaU64 InPts)
{
	if (!IsOpen() || !Config_.AudioEnabled || !AudioEncoder_.IsOpen()) {
		return OaStatus::Error(OaStatusCode::FailedPrecondition,
			"OaVideoRecorder audio was not enabled");
	}
	if (InSampleRate != Config_.Audio.SampleRate
		|| InChannelCount != Config_.Audio.ChannelCount) {
		return OaStatus::InvalidArgument(
			"Audio chunk format does not match the recorder profile");
	}
	if (InInterleaved.Empty() || InInterleaved.Size() % InChannelCount != 0U) {
		return OaStatus::InvalidArgument("Audio chunk contains incomplete interleaved frames");
	}
	if (!HasFirstVideoPts_) {
		PendingAudioChunk chunk;
		chunk.Samples.Resize(InInterleaved.Size());
		OaMemcpy(chunk.Samples.Data(), InInterleaved.Data(),
			InInterleaved.Size() * sizeof(OaF32));
		chunk.Pts = InPts;
		PendingAudio_.PushBack(OaStdMove(chunk));
		return OaStatus::Ok();
	}
	return WriteAudioAligned_(InInterleaved, InPts);
}

OaStatus OaVideoRecorder::WriteAudio(const OaAudioCaptureChunk& InChunk)
{
	return WriteAudio(
		OaSpan<const OaF32>(InChunk.Interleaved.Data(), InChunk.Interleaved.Size()),
		InChunk.SampleRate,
		InChunk.ChannelCount,
		InChunk.PresentationTimestamp);
}

OaStatus OaVideoRecorder::Finalize() {
	if (Finalized_) return OaStatus::Ok();
	if (Engine_ == nullptr) {
		return OaStatus::Error(OaStatusCode::FailedPrecondition,
			"OaVideoRecorder was not created");
	}
	OaVec<OaEncodedFrame> remaining;
	OA_RETURN_IF_ERROR(Encoder_.Flush(remaining));
	for (const auto& frame : remaining) {
		OA_RETURN_IF_ERROR(WriteEncoded_(frame));
	}
	if (Config_.AudioEnabled && AudioEncoder_.IsOpen()) {
		if (!HasFirstVideoPts_) PendingAudio_.Clear();
		OaVec<OaEncodedAudioPacket> audioPackets;
		OA_RETURN_IF_ERROR(AudioEncoder_.Flush(audioPackets));
		OA_RETURN_IF_ERROR(WriteAudioPackets_(audioPackets));
	}
	if (SubmittedFrameCount_ == 0U or MuxedFrameCount_ == 0U) {
		return OaStatus::Error(OaStatusCode::FailedPrecondition,
			"OaVideoRecorder cannot finalize an empty recording");
	}
	OA_RETURN_IF_ERROR(Muxer_.Finalize());
	Finalized_ = true;
	return OaStatus::Ok();
}

void OaVideoRecorder::Destroy() {
	// Destruction is noexcept-shaped: explicit Finalize() is the API that can
	// report disk/container errors. Never silently manufacture an empty file.
	Muxer_.Destroy();
	Encoder_.Destroy();
	AudioEncoder_.Destroy();
	PendingAudio_.Clear();
	AudioScratch_.Clear();
	Engine_ = nullptr;
	SubmittedFrameCount_ = 0;
	MuxedFrameCount_ = 0;
	CodecConfigWritten_ = false;
	FirstVideoPts_ = 0U;
	NextAudioFrame_ = 0U;
	HasFirstVideoPts_ = false;
	Finalized_ = true;
}
