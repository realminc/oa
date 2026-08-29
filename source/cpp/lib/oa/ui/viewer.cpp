// oa::Viewer — one application lifecycle for still images, video and audio.

#include <oa/ui/viewer.h>

#include <oa/core/fnMatrix.h>
#include <oa/core/log.h>
#include <oa/core/std/algo.h>
#include <oa/core/std/limits.h>
#include <oa/core/std/scalarMath.h>
#include <oa/audio/fnAudio.h>
#include <oa/render/renderer.h>
#include <oa/runtime/engine.h>
#include <oa/runtime/eventAccess.h>
#include <oa/runtime/engine/allocatorAccess.h>
#include <oa/ui/ui.h>
#include <oa/vision/fnImage.h>
#include "../runtime/textureAccess.h"


namespace {

const char* viewerModeName(oa::ViewerMode inMode) {
	switch (inMode) {
		case oa::ViewerMode::Image: return "image";
		case oa::ViewerMode::Video: return "video";
		case oa::ViewerMode::Audio: return "audio";
		case oa::ViewerMode::Live: return "live";
		default: return "auto";
	}
}

const char* videoCodecName(oa::VideoCodec inCodec) {
	switch (inCodec) {
		case oa::VideoCodec::H264: return "H.264";
		case oa::VideoCodec::H265: return "H.265";
		case oa::VideoCodec::AV1: return "AV1";
		case oa::VideoCodec::VP9: return "VP9";
		default: return "Unknown";
	}
}

oa::I32 saturatingPixel(oa::F64 inValue) noexcept {
	if (not oa::isFinite(inValue)) return 0;
	return static_cast<oa::I32>(oa::clamp<oa::F64>(
		inValue,
		oa::Limits<oa::I32>::min(),
		oa::Limits<oa::I32>::max()));
}

oa::I32 saturatingExtent(oa::F64 inValue) noexcept {
	if (not oa::isFinite(inValue) or inValue <= 0.0) return 0;
	return static_cast<oa::I32>(oa::clamp<oa::F64>(
		inValue, 1.0, oa::Limits<oa::I32>::max()));
}

bool isPowerOfTwo(oa::U32 inValue) noexcept {
	return inValue != 0U and (inValue & (inValue - 1U)) == 0U;
}

oa::Status validateAudioAnalysisConfig(const oa::ViewerConfig& inConfig) {
	switch (inConfig.audioView) {
		case oa::ViewerAudioView::Waveform:
		case oa::ViewerAudioView::Spectrum:
		case oa::ViewerAudioView::Mel:
			break;
		default:
			return oa::Status::invalidArgument(
				"oa::Viewer audio view is not a declared mode");
	}
	if (inConfig.audioWaveformBins == 0U
		or inConfig.audioWaveformBins > 65'536U) {
		return oa::Status::invalidArgument(
			"oa::Viewer audio waveform bins must be in [1, 65536]");
	}
	if (inConfig.audioAnalysisFrames < 2U
		or inConfig.audioAnalysisFrames > 65'536U) {
		return oa::Status::invalidArgument(
			"oa::Viewer audio analysis frames must be in [2, 65536]");
	}
	if (not isPowerOfTwo(inConfig.audioFftSize)
		or inConfig.audioFftSize < 16U
		or inConfig.audioFftSize > 1024U) {
		return oa::Status::invalidArgument(
			"oa::Viewer audio FFT size must be a power of two in [16, 1024]");
	}
	if (inConfig.audioHopSize == 0U) {
		return oa::Status::invalidArgument(
			"oa::Viewer audio hop size must be non-zero");
	}
	if (inConfig.audioMelBins == 0U or inConfig.audioMelBins > 4096U) {
		return oa::Status::invalidArgument(
			"oa::Viewer audio mel bins must be in [1, 4096]");
	}
	return oa::Status::ok();
}

oa::PixelRect viewerContentRect(
	const oa::Navigation& inNavigation,
	oa::U32 inWidth,
	oa::U32 inHeight,
	oa::U32 inTop) noexcept {
	return {
		.x = saturatingPixel(inNavigation.panX()),
		.y = saturatingPixel(
			static_cast<oa::F64>(inTop) + inNavigation.panY()),
		.w = saturatingExtent(
			static_cast<oa::F64>(inWidth) * inNavigation.zoom()),
		.h = saturatingExtent(
			static_cast<oa::F64>(inHeight) * inNavigation.zoom()),
	};
}

} // namespace

oa::Status oa::Viewer::preview(
	const char* inPath,
	const oa::ViewerConfig& inConfig)
{
	if (inPath == nullptr or inPath[0] == '\0') {
		return oa::Status::invalidArgument("oa::Viewer::preview requires a non-empty path");
	}
	return preview(oa::String(inPath), inConfig);
}

oa::Status oa::Viewer::preview(
	const oa::String& inPath,
	const oa::ViewerConfig& inConfig)
{
	if (inPath.empty()) {
		return oa::Status::invalidArgument("oa::Viewer::preview requires a non-empty path");
	}
	oa::ViewerConfig config = inConfig;
	config.path = inPath;
	oa::Viewer viewer(config);
	return viewer.run();
}

oa::Status oa::Viewer::preview(
	oa::Engine& inEngine,
	const char* inPath,
	const oa::ViewerConfig& inConfig)
{
	if (inPath == nullptr or inPath[0] == '\0') {
		return oa::Status::invalidArgument("oa::Viewer::preview requires a non-empty path");
	}
	return preview(inEngine, oa::String(inPath), inConfig);
}

oa::Status oa::Viewer::preview(
	oa::Engine& inEngine,
	const oa::String& inPath,
	const oa::ViewerConfig& inConfig)
{
	if (inPath.empty()) {
		return oa::Status::invalidArgument("oa::Viewer::preview requires a non-empty path");
	}
	oa::ViewerConfig config = inConfig;
	config.path = inPath;
	oa::Viewer viewer(config);
	return viewer.run(inEngine);
}

oa::Status oa::Viewer::preview(
	oa::Engine& inEngine,
	const oa::Matrix& inMatrix,
	const oa::ViewerConfig& inConfig)
{
	return show(inEngine, inMatrix, inConfig);
}

oa::Status oa::Viewer::preview(
	oa::Engine& inEngine,
	const oa::Image& inImage,
	const oa::ViewerConfig& inConfig)
{
	return show(inEngine, inImage, inConfig);
}

oa::Status oa::Viewer::preview(
	oa::Engine& inEngine,
	const oa::Texture& inTexture,
	const oa::ViewerConfig& inConfig)
{
	return show(inEngine, inTexture, inConfig);
}

oa::Status oa::Viewer::preview(
	oa::Engine& inEngine,
	oa::Renderer& inRenderer,
	const oa::RenderFrame& inFrame,
	const oa::ViewerConfig& inConfig)
{
	return show(inEngine, inRenderer, inFrame, inConfig);
}

oa::Status oa::Viewer::show(
	oa::Engine& inEngine,
	const oa::Matrix& inMatrix,
	const oa::ViewerConfig& inConfig)
{
	if (not inEngine.hasGraphics()) {
		return oa::Status::error(
			oa::StatusCode::FailedPrecondition,
			"oa::Viewer::show requires a presentation-capable oa::Engine");
	}

	const oa::MatrixShape shape = inMatrix.getShape();
	if (shape.rank != 4 or shape[0] != 1
		or (shape[1] != 1 and shape[1] != 2
			and shape[1] != 3 and shape[1] != 4)) {
		return oa::Status::invalidArgument(
			"oa::Viewer::show matrix requires [1,C,H,W] with C in [1,4]");
	}
	const oa::ImageFormat format = shape[1] == 1 ? oa::ImageFormat::Gray
		: shape[1] == 2 ? oa::ImageFormat::GrayAlpha
		: shape[1] == 3 ? oa::ImageFormat::Rgb
		: oa::ImageFormat::Rgba;
	const oa::Image image(inMatrix, oa::ImageLayout::Nchw, format);
	auto textureResult = oa::FnTexture::fromImage(inEngine, image);
	if (not textureResult.isOk()) return textureResult.getStatus();
	oa::Texture texture = oa::move(*textureResult);

	auto submitResult = inEngine.submit();
	if (not submitResult.isOk()) return submitResult.getStatus();
	const oa::Event ready = *submitResult;

	oa::Viewer viewer(inConfig);
	viewer.config_.mode = oa::ViewerMode::Image;
	viewer.config_.path.clear();
	viewer.borrowedImage_ = &texture;
	viewer.borrowedImageReady_ = ready;
	const oa::Status runStatus = viewer.run(inEngine);
	const oa::Status waitStatus = inEngine.wait(ready);
	if (not runStatus.isOk()) return runStatus;
	return waitStatus;
}

oa::Status oa::Viewer::show(
	oa::Engine& inEngine,
	const oa::Image& inImage,
	const oa::ViewerConfig& inConfig)
{
	if (not inEngine.hasGraphics()) {
		return oa::Status::error(
			oa::StatusCode::FailedPrecondition,
			"oa::Viewer::show requires a presentation-capable oa::Engine");
	}

	auto textureResult = oa::FnTexture::fromImage(inEngine, inImage);
	if (not textureResult.isOk()) return textureResult.getStatus();
	oa::Texture texture = oa::move(*textureResult);

	auto submitResult = inEngine.submit();
	if (not submitResult.isOk()) return submitResult.getStatus();
	const oa::Event ready = *submitResult;

	oa::Viewer viewer(inConfig);
	viewer.config_.mode = oa::ViewerMode::Image;
	viewer.config_.path.clear();
	viewer.borrowedImage_ = &texture;
	viewer.borrowedImageReady_ = ready;
	const oa::Status runStatus = viewer.run(inEngine);
	const oa::Status waitStatus = inEngine.wait(ready);
	if (not runStatus.isOk()) return runStatus;
	return waitStatus;
}

oa::Status oa::Viewer::show(
	oa::Engine& inEngine,
	const oa::Texture& inTexture,
	const oa::ViewerConfig& inConfig)
{
	if (not inEngine.hasGraphics()) {
		return oa::Status::error(
			oa::StatusCode::FailedPrecondition,
			"oa::Viewer::show requires a presentation-capable oa::Engine");
	}
	oa::Viewer viewer(inConfig);
	viewer.config_.mode = oa::ViewerMode::Image;
	viewer.config_.path.clear();
	viewer.borrowedImage_ = &inTexture;
	return viewer.run(inEngine);
}

oa::Status oa::Viewer::show(
	oa::Engine& inEngine,
	oa::Renderer& inRenderer,
	const oa::RenderFrame& inFrame,
	const oa::ViewerConfig& inConfig)
{
	if (not inEngine.hasGraphics()) {
		return oa::Status::error(
			oa::StatusCode::FailedPrecondition,
			"oa::Viewer::show requires a presentation-capable oa::Engine");
	}
	oa::Viewer viewer(inConfig);
	viewer.config_.mode = oa::ViewerMode::Image;
	viewer.config_.path.clear();
	viewer.borrowedImage_ = &inFrame.color();
	viewer.borrowedImageReady_ = inFrame.producer();
	if (inFrame.sourceKind_ == oa::RenderFrame::SourceKind::Ui) {
		viewer.borrowedImageSourceStageMask_ =
			VK_PIPELINE_STAGE_2_TRANSFER_BIT
			| VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
		viewer.borrowedImageSourceAccessMask_ =
			VK_ACCESS_2_TRANSFER_WRITE_BIT
			| VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT;
	} else if (inFrame.sourceKind_ == oa::RenderFrame::SourceKind::Mesh) {
		viewer.borrowedImageSourceStageMask_ = VK_PIPELINE_STAGE_2_COPY_BIT;
		viewer.borrowedImageSourceAccessMask_ = VK_ACCESS_2_TRANSFER_READ_BIT;
	} else {
		return oa::Status::invalidArgument(
			"oa::Viewer::show requires a frame produced by oa::Renderer");
	}
	viewer.borrowedFrameMarkConsumed_ =
		[&inRenderer, inFrame](const oa::Event& inCompletion) {
			return inRenderer.markConsumed(inFrame, inCompletion);
		};
	viewer.borrowedFrameAbandon_ = [&inRenderer, inFrame] {
		return inRenderer.abandonFrame(inFrame);
	};
	viewer.borrowedFrameCollect_ = [&inRenderer] {
		return inRenderer.collect();
	};
	const oa::Status runStatus = viewer.run(inEngine);
	const oa::Status frameStatus = viewer.finalizeBorrowedFrame();
	if (not runStatus.isOk()) return runStatus;
	return frameStatus;
}

oa::Status oa::Viewer::save(
	oa::Engine& inEngine,
	const oa::Texture& inTexture,
	const char* inPath) {
	if (not inTexture.isValid()) {
		return oa::Status::invalidArgument("oa::Viewer::save: invalid texture");
	}
	if (inPath == nullptr or inPath[0] == '\0') {
		return oa::Status::invalidArgument("oa::Viewer::save: empty output path");
	}
	return oa::FnImage::saveTextureFile(inEngine, inTexture, inPath);
}

oa::Status oa::Viewer::openImage(oa::Engine& inEngine) {
	if (borrowedImage_ != nullptr) {
		const oa::Texture& image = *borrowedImage_;
		const oavk::Buffer* buffer = oa::TextureAccess::buffer(image);
		if (not image.isValid()
			or oa::TextureAccess::engine(image) != &inEngine) {
			return oa::Status::invalidArgument(
				"oa::Viewer::show requires a valid texture owned by its engine");
		}
		if (image.isImageBacked()) {
			if (not borrowedFrameMarkConsumed_
				or not borrowedFrameAbandon_
				or not borrowedFrameCollect_) {
				return oa::Status::invalidArgument(
					"oa::Viewer image-backed targets require a renderer-frame show overload");
			}
			const VkImageLayout layout = oa::TextureAccess::layout(image);
			if (oa::TextureAccess::format(image) != VK_FORMAT_R8G8B8A8_UNORM
				or (layout != VK_IMAGE_LAYOUT_GENERAL
					and layout != VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL)) {
				return oa::Status::invalidArgument(
					"oa::Viewer renderer frames require RGBA8 UNORM in GENERAL or shader-read layout");
			}
		} else if (buffer == nullptr
			or buffer->allocatorIdentity
				!= oa::EngineAllocatorAccess::get(inEngine).allocator) {
			return oa::Status::invalidArgument(
				"oa::Viewer::show requires a buffer-backed texture owned by its engine");
		}
		if (borrowedImageReady_.isValid()) {
			if (not inEngine.ownsEvent(borrowedImageReady_)
				or not borrowedImageReady_.hasQueueFamily()) {
				return oa::Status::invalidArgument(
					"oa::Viewer::show requires an exact readiness event from its engine");
			}
		} else if (image.isImageBacked()) {
			return oa::Status::invalidArgument(
				"oa::Viewer renderer frames require an exact producer event");
		}
		resolvedMode_ = oa::ViewerMode::Image;
		return oa::Status::ok();
	}

	auto decoded = oa::FnImage::decodeFile(
		oa::Path(config_.path), oa::ImageFormat::Rgba);
	if (not decoded.isOk()) return decoded.getStatus();
	auto image = oa::FnTexture::fromImage(inEngine, *decoded);
	if (not image.isOk()) return image.getStatus();
	auto submitted = inEngine.submit();
	if (not submitted.isOk()) return submitted.getStatus();
	OA_RETURN_IF_ERROR(inEngine.wait(*submitted));
	image_ = oa::move(*image);

	auto planes = oa::ImagePlanes::loadFile(inEngine, config_.path);
	if (planes.isOk()) planes_ = oa::move(*planes);
	resolvedMode_ = oa::ViewerMode::Image;
	return oa::Status::ok();
}

oa::Status oa::Viewer::openVideo(oa::Engine& inEngine) {
	oa::VideoPlayerConfig config;
	config.uri = config_.path;
	config.loop = config_.loop;
	config.frameRateOverride = config_.frameRateOverride;
	config.reorderDepth = config_.reorderDepth;
	config.preferHardwareYCbCr = config_.preferHardwareYCbCr;
	config.filter = config_.filter;
	config.startPlaying = config_.startPlaying;

	auto video = oa::VideoPlayer::open(inEngine, config);
	if (not video.isOk()) return video.getStatus();
	video_.emplace(oa::move(*video));
	resolvedMode_ = oa::ViewerMode::Video;
	return oa::Status::ok();
}

oa::Status oa::Viewer::openAudio(oa::Engine& inEngine) {
	OA_RETURN_IF_ERROR(validateAudioAnalysisConfig(config_));
	oa::AudioPlayerConfig config;
	config.uri = config_.path;
	config.loop = config_.loop;
	config.ringMilliseconds = config_.audioRingMilliseconds;
	auto audio = oa::AudioPlayer::open(inEngine, config);
	if (not audio.isOk()) return audio.getStatus();
	audio_.emplace(oa::move(*audio));
	auto decoded = oa::FnAudio::decodeFile(oa::Path(config.uri));
	if (decoded.isOk()) {
		audioEnvelope_ = oa::FnAudio::waveformEnvelope(
			*decoded,
			config_.audioWaveformBins);

		const oa::I64 samples = decoded->samples();
		if (samples <= 0
			or static_cast<oa::U64>(samples)
				> oa::Limits<oa::U32>::max()) {
			return oa::Status::error(
				oa::StatusCode::OutOfRange,
				"oa::Viewer audio analysis requires a non-empty u32-addressable source");
		}
		const oa::U64 frameIntervals = config_.audioAnalysisFrames - 1U;
		const oa::U64 boundedHop =
			(static_cast<oa::U64>(samples) + frameIntervals - 1U)
				/ frameIntervals;
		const oa::U32 effectiveHop = oa::max(
			config_.audioHopSize,
			static_cast<oa::U32>(boundedHop));

		oa::StftConfig stftConfig;
		stftConfig.fftSize = config_.audioFftSize;
		stftConfig.hopSize = effectiveHop;
		stftConfig.winSize = config_.audioFftSize;
		stftConfig.center = true;
		oa::Matrix spectrum = oa::FnAudio::stft(*decoded, stftConfig);
		if (not spectrum.isEmpty()) {
			spectrum = oa::FnMatrix::mean(spectrum, 0).squeeze(0);
			spectrum = oa::FnMatrix::log(
				oa::FnMatrix::addScalar(spectrum, 1.0e-6F));
			audioSpectrum_ = oa::FnMatrix::transpose(spectrum, 0, 1);
		}

		oa::MelConfig melConfig;
		melConfig.fftSize = config_.audioFftSize;
		melConfig.hopSize = effectiveHop;
		melConfig.numMels = config_.audioMelBins;
		melConfig.logScale = true;
		oa::Matrix mel = oa::FnAudio::melSpectrogram(*decoded, melConfig);
		if (not mel.isEmpty()) {
			audioMel_ = oa::FnMatrix::mean(mel, 0).squeeze(0);
		}

		if (audioEnvelope_.isEmpty() or audioSpectrum_.isEmpty()
			or audioMel_.isEmpty()) {
			return oa::Status::error(
				oa::StatusCode::Internal,
				"oa::Viewer audio analysis failed to produce waveform, spectrum, and mel snapshots");
		}
		auto submitted = inEngine.submit();
		if (not submitted.isOk()) return submitted.getStatus();
		audioAnalysisReady_ = *submitted;
	} else {
		OaLogWarn(oa::LogComponent::Ui,
			"oa::Viewer audio analysis unavailable: {}",
			decoded.getStatus().toString().cStr());
	}
	if (config_.startPlaying) {
		OA_RETURN_IF_ERROR(audio_->play());
	}
	audioView_ = config_.audioView;
	audioViewTabs_.selected = static_cast<oa::I32>(audioView_);
	audioViewTabs_.firstVisible = 0;
	resolvedMode_ = oa::ViewerMode::Audio;
	return oa::Status::ok();
}

oa::Status oa::Viewer::openSource(oa::Engine& inEngine) {
	sourceEngine_ = &inEngine;
	switch (config_.mode) {
		case oa::ViewerMode::Image:
			return openImage(inEngine);
		case oa::ViewerMode::Video:
			return openVideo(inEngine);
		case oa::ViewerMode::Audio:
			return openAudio(inEngine);
		case oa::ViewerMode::Live:
			if (config_.liveSource == nullptr) {
				return oa::Status::invalidArgument(
					"oa::Viewer live mode requires a live source");
			}
			liveCapabilities_ = config_.liveSource->capabilities();
			resolvedMode_ = oa::ViewerMode::Live;
			return config_.liveSource->open(inEngine);
		case oa::ViewerMode::Auto:
			break;
	}

	const oa::Status imageStatus = openImage(inEngine);
	if (imageStatus.isOk()) return imageStatus;

	const oa::Status videoStatus = openVideo(inEngine);
	if (videoStatus.isOk()) return videoStatus;

	const oa::Status audioStatus = openAudio(inEngine);
	if (audioStatus.isOk()) return audioStatus;

	OaLogError(oa::LogComponent::Ui,
		"oa::Viewer could not open '{}' as image ({}), video ({}) or audio ({})",
		config_.path.cStr(),
		imageStatus.toString().cStr(),
		videoStatus.toString().cStr(),
		audioStatus.toString().cStr());
	return oa::Status::error(
		oa::StatusCode::InvalidArgument,
		"oa::Viewer: unsupported or invalid media source");
}

bool oa::Viewer::hasVisualContent() const noexcept {
	return resolvedMode_ == oa::ViewerMode::Image or resolvedMode_ == oa::ViewerMode::Video;
}

bool oa::Viewer::hasTimeline() const noexcept {
	return config_.showTimeline
		and ((resolvedMode_ == oa::ViewerMode::Video and video_.hasValue())
			or (resolvedMode_ == oa::ViewerMode::Audio and audio_.hasValue()))
		and mediaDurationUs() > 0U;
}

bool oa::Viewer::isMediaPlaying() const noexcept {
	if (video_.hasValue()) return video_->isPlaying();
	if (audio_.hasValue()) return audio_->isPlaying();
	return false;
}

bool oa::Viewer::isMediaLooping() const noexcept {
	if (video_.hasValue()) return video_->isLooping();
	return config_.loop;
}

oa::U64 oa::Viewer::mediaDurationUs() const noexcept {
	if (video_.hasValue()) return video_->durationUs();
	if (audio_.hasValue()) return audio_->durationUs();
	return 0U;
}

oa::U64 oa::Viewer::mediaPositionUs() const noexcept {
	if (video_.hasValue()) return video_->positionUs();
	if (audio_.hasValue()) return audio_->positionUs();
	return 0U;
}

oa::F32 oa::Viewer::mediaPositionFraction() const noexcept {
	if (video_.hasValue()) {
		const oa::Usize count = video_->frameCount();
		if (count <= 1U) return 0.0F;
		return static_cast<oa::F32>(
			static_cast<long double>(oa::min(
				video_->currentFrameIndex(), count - 1U))
			/ static_cast<long double>(count - 1U));
	}
	const oa::U64 duration = mediaDurationUs();
	if (duration == 0U) return 0.0F;
	return static_cast<oa::F32>(oa::min(
		1.0L,
		static_cast<long double>(mediaPositionUs())
			/ static_cast<long double>(duration)));
}

oa::PixelRect oa::Viewer::timelineRect() const noexcept {
	const oa::F32 contentScale = oa::max(
		0.01F, (windowPixelScaleX_ + windowPixelScaleY_) * 0.5F);
	const auto px = [contentScale](oa::I32 inLogical) {
		return oa::max<oa::I32>(1, static_cast<oa::I32>(oa::lround(
			static_cast<oa::F32>(inLogical) * contentScale)));
	};
	const oa::I32 contentTop = static_cast<oa::I32>(windowDecorationHeight());
	const oa::I32 contentHeight = oa::max<oa::I32>(
		1, static_cast<oa::I32>(height()) - contentTop);
	const oa::I32 margin = px(24);
	const oa::I32 height = px(20);
	const oa::I32 width = oa::max<oa::I32>(
		1, static_cast<oa::I32>(this->width()) - margin * 2);
	const oa::I32 y = resolvedMode_ == oa::ViewerMode::Audio
		? contentTop + oa::max<oa::I32>(0, contentHeight - px(28))
		: contentTop + oa::max<oa::I32>(0, contentHeight - px(52));
	return {margin, y, width, height};
}

oa::PixelRect oa::Viewer::temporalButtonsRect() const noexcept {
	const oa::F32 contentScale = oa::max(
		0.01F, (windowPixelScaleX_ + windowPixelScaleY_) * 0.5F);
	const auto px = [contentScale](oa::I32 inLogical) {
		return oa::max<oa::I32>(1, static_cast<oa::I32>(oa::lround(
			static_cast<oa::F32>(inLogical) * contentScale)));
	};
	const oa::PixelRect timeline = timelineRect();
	const oa::I32 height = px(34);
	const oa::I32 width = oa::min(timeline.w, px(272));
	const oa::I32 contentTop = static_cast<oa::I32>(windowDecorationHeight());
	return {
		timeline.x,
		oa::max(contentTop + px(8), timeline.y - height - px(8)),
		width,
		height,
	};
}

oa::PixelRect oa::Viewer::audioViewTabRect() const noexcept {
	const oa::F32 scale = oa::max(
		0.01F, (windowPixelScaleX_ + windowPixelScaleY_) * 0.5F);
	const auto px = [scale](oa::I32 inLogical) {
		return oa::max<oa::I32>(1, static_cast<oa::I32>(oa::lround(
			static_cast<oa::F32>(inLogical) * scale)));
	};
	const oa::I32 margin = px(24);
	const oa::I32 available = oa::max<oa::I32>(
		1, static_cast<oa::I32>(width()) - margin * 2);
	return {
		margin,
		static_cast<oa::I32>(windowDecorationHeight()) + px(12),
		oa::min(available, px(360)),
		px(28),
	};
}

oa::PixelRect oa::Viewer::audioVisualizationRect() const noexcept {
	const oa::F32 scale = oa::max(
		0.01F, (windowPixelScaleX_ + windowPixelScaleY_) * 0.5F);
	const auto px = [scale](oa::I32 inLogical) {
		return oa::max<oa::I32>(1, static_cast<oa::I32>(oa::lround(
			static_cast<oa::F32>(inLogical) * scale)));
	};
	const oa::I32 margin = px(24);
	const oa::I32 contentTop = static_cast<oa::I32>(windowDecorationHeight());
	const oa::PixelRect tabs = audioViewTabRect();
	const oa::I32 top = config_.showAudioViewSelector
		? tabs.y + tabs.h + px(12)
		: contentTop + px(24);
	const oa::I32 bottom = config_.showTimeline
		? timelineRect().y - px(12)
		: static_cast<oa::I32>(height()) - px(24);
	return {
		margin,
		top,
		oa::max<oa::I32>(1, static_cast<oa::I32>(width()) - margin * 2),
		oa::max<oa::I32>(1, bottom - top),
	};
}

void oa::Viewer::toggleMediaPlayback() {
	if (video_.hasValue()) {
		video_->togglePlay();
	} else if (audio_.hasValue()) {
		if (audio_->isPlaying()) audio_->pause();
		else if (const oa::Status status = audio_->play(); not status.isOk()) {
			OaLogWarn(oa::LogComponent::Ui,
				"oa::Viewer audio playback failed: {}", status.toString().cStr());
		}
	}
}

void oa::Viewer::toggleMediaLoop() {
	const bool loop = not isMediaLooping();
	config_.loop = loop;
	if (video_.hasValue()) video_->setLoop(loop);
	if (audio_.hasValue()) audio_->setLoop(loop);
}

void oa::Viewer::pauseMedia() {
	if (video_.hasValue()) video_->pause();
	else if (audio_.hasValue()) audio_->pause();
}

void oa::Viewer::seekMediaUs(oa::U64 inTimestampUs) {
	pauseMedia();
	oa::Status status = oa::Status::ok();
	if (video_.hasValue()) status = video_->seekUs(inTimestampUs);
	else if (audio_.hasValue()) status = audio_->seek(inTimestampUs);
	if (not status.isOk()) {
		OaLogWarn(oa::LogComponent::Ui,
			"oa::Viewer media seek failed: {}", status.toString().cStr());
	}
}

void oa::Viewer::seekMediaFraction(oa::F32 inFraction) {
	const long double clamped = static_cast<long double>(
		oa::clamp(inFraction, 0.0F, 1.0F));
	if (video_.hasValue()) {
		pauseMedia();
		const oa::Usize count = video_->frameCount();
		if (count == 0U) return;
		const oa::Usize target = static_cast<oa::Usize>(
			clamped * static_cast<long double>(count - 1U) + 0.5L);
		const oa::Status status = video_->seekFrame(target);
		if (not status.isOk()) {
			OaLogWarn(oa::LogComponent::Ui,
				"oa::Viewer video frame seek failed: {}",
				status.toString().cStr());
		}
		return;
	}
	const oa::U64 duration = mediaDurationUs();
	if (duration == 0U) return;
	seekMediaUs(static_cast<oa::U64>(clamped * static_cast<long double>(duration)));
}

void oa::Viewer::handleTimelineSeek(
	oa::F32 inFraction,
	bool inChanged,
	bool inActive
) {
	if (inChanged) {
		pauseMedia();
		pendingTimelineSeekFraction_.emplace(
			oa::clamp(inFraction, 0.0F, 1.0F));
	}
	if (inActive || not pendingTimelineSeekFraction_.hasValue()) return;
	const oa::F32 committed = *pendingTimelineSeekFraction_;
	pendingTimelineSeekFraction_.reset();
	seekMediaFraction(committed);
}

void oa::Viewer::stepTemporal(oa::I32 inAmount) {
	if (video_.hasValue()) {
		pauseMedia();
		const oa::Status status = video_->stepFrames(inAmount);
		if (not status.isOk()) {
			OaLogWarn(oa::LogComponent::Ui,
				"oa::Viewer video scrub failed: {}", status.toString().cStr());
		}
		return;
	}
	if (not audio_.hasValue()) return;
	pauseMedia();
	const oa::U64 position = audio_->positionUs();
	const oa::U64 step = config_.audioStepUs * static_cast<oa::U64>(oa::abs(inAmount));
	const oa::U64 target = inAmount < 0
		? (position > step ? position - step : 0U)
		: oa::min(mediaDurationUs(), position + step);
	seekMediaUs(target);
}

oa::Status oa::Viewer::configureNavigation() {
	if (not hasVisualContent()) {
		resizeWindow(config_.width, config_.height);
		return oa::Status::ok();
	}
	oa::U32 contentWidth = 0;
	oa::U32 contentHeight = 0;
	if (resolvedMode_ == oa::ViewerMode::Image and imageSource().isValid()) {
		contentWidth = static_cast<oa::U32>(imageSource().width());
		contentHeight = static_cast<oa::U32>(imageSource().height());
	} else if (resolvedMode_ == oa::ViewerMode::Video and video_.hasValue()) {
		contentWidth = video_->width();
		contentHeight = video_->height();
	}

	if (contentWidth == 0 or contentHeight == 0) {
		return oa::Status::error(
			oa::StatusCode::FailedPrecondition,
			"oa::Viewer visual source has zero dimensions");
	}

	const oa::U32 windowWidth = oa::min(
		config_.width, oa::max<oa::U32>(contentWidth, contentWidth * 2U));
	const oa::U32 windowHeight = oa::min(
		config_.height, oa::max<oa::U32>(contentHeight, contentHeight * 2U));
	resizeWindow(windowWidth, windowHeight);

	nav_.setCapturePointer([this](bool inOn) { captureRelativeMouse(inOn); });
	OA_RETURN_IF_ERROR(nav_.setContentSize(
		static_cast<oa::F32>(contentWidth),
		static_cast<oa::F32>(contentHeight)));
	OA_RETURN_IF_ERROR(nav_.setWindowSize(
		static_cast<oa::F32>(width()),
		static_cast<oa::F32>(oa::max<oa::U32>(
			1U, height() - oa::min(height(), windowDecorationHeight())))));
	return nav_.fitToWindow(false);
}

oa::Status oa::Viewer::configureOverlay() {
	if (config_.annotations.empty()) return oa::Status::ok();
	if (engine_ == nullptr) {
		return oa::Status::error(
			"oa::Viewer overlay creation requires an attached engine");
	}
	auto overlay = oa::DetectionOverlay::create(*engine_, config_.annotationStyle);
	if (not overlay.isOk()) {
		OaLogError(oa::LogComponent::Ui,
			"oa::Viewer overlay creation failed: {}",
			overlay.getStatus().toString().cStr());
		return overlay.getStatus();
	}
	detectionOverlay_ = oa::move(*overlay);
	const oa::Status update = detectionOverlay_.update(
		oa::Span<const oa::DetectionOverlayItem>(
			config_.annotations.data(), config_.annotations.size()),
		textAtlas_);
	if (not update.isOk()) {
		OaLogError(oa::LogComponent::Ui,
			"oa::Viewer overlay update failed: {}", update.toString().cStr());
		detectionOverlay_ = {};
		return update;
	}
	return oa::Status::ok();
}

oa::Status oa::Viewer::registerCommonInput() {
	auto& input = input_;
	input.registerAction({.name = "quit", .binding = {.key = config_.keyQuit},
		.callback = [this] { quit(); }});
	input.registerAction({.name = "quitq", .binding = {.key = config_.keyQuitQ},
		.callback = [this] { quit(); }});

	if (hasVisualContent()) {
		oa::NavigationShortcuts keys;
		keys.zoomIn = config_.keyZoomIn;
		keys.zoomOut = config_.keyZoomOut;
		keys.zoomFit = config_.keyZoomFit;
		keys.zoom100 = config_.keyZoom100;
		keys.panUp = config_.keyPanUp;
		keys.panDown = config_.keyPanDown;
		keys.panLeft = config_.keyPanLeft;
		keys.panRight = config_.keyPanRight;
		OA_RETURN_IF_ERROR(registerViewportShortcuts(input, nav_, keys));
	}
	return oa::Status::ok();
}

void oa::Viewer::registerImageInput() {
	auto& input = input_;
	input.registerAction({.name = "red", .binding = {.key = config_.keyRed},
		.callback = [this] { imageMode_ = ImageViewMode::R; }});
	input.registerAction({.name = "green", .binding = {.key = config_.keyGreen},
		.callback = [this] { imageMode_ = ImageViewMode::G; }});
	input.registerAction({.name = "blue", .binding = {.key = config_.keyBlue},
		.callback = [this] { imageMode_ = ImageViewMode::B; }});
	input.registerAction({.name = "alpha", .binding = {.key = config_.keyAlpha},
		.callback = [this] { imageMode_ = ImageViewMode::A; }});
	input.registerAction({.name = "rgb", .binding = {.key = config_.keyRgb},
		.callback = [this] { imageMode_ = ImageViewMode::RGB; }});
}

void oa::Viewer::registerTemporalInput() {
	auto& input = input_;
	input.registerAction({.name = "play", .binding = {.key = oa::UiKey::Space},
		.callback = [this] { toggleMediaPlayback(); }});
	input.registerAction({.name = "loop", .binding = {.key = oa::UiKey::L},
		.callback = [this] { toggleMediaLoop(); }});
	input.registerAction({.name = "stepf", .binding = {.key = oa::UiKey::Right},
		.allowRepeat = true,
		.callback = [this] { stepTemporal(1); }});
	input.registerAction({.name = "stepb", .binding = {.key = oa::UiKey::Left},
		.allowRepeat = true,
		.callback = [this] { stepTemporal(-1); }});
	input.registerAction({.name = "stepf5", .binding = {.key = oa::UiKey::Up},
		.allowRepeat = true,
		.callback = [this] { stepTemporal(5); }});
	input.registerAction({.name = "stepb5", .binding = {.key = oa::UiKey::Down},
		.allowRepeat = true,
		.callback = [this] { stepTemporal(-5); }});
	if (resolvedMode_ == oa::ViewerMode::Audio) {
		input.registerAction({.name = "audio-waveform", .binding = {.key = oa::UiKey::W},
			.callback = [this] {
				audioView_ = oa::ViewerAudioView::Waveform;
				audioViewTabs_.selected = 0;
			}});
		input.registerAction({.name = "audio-spectrum", .binding = {.key = oa::UiKey::S},
			.callback = [this] {
				if (not audioSpectrum_.isEmpty()) {
					audioView_ = oa::ViewerAudioView::Spectrum;
					audioViewTabs_.selected = 1;
				}
			}});
		input.registerAction({.name = "audio-mel", .binding = {.key = oa::UiKey::M},
			.callback = [this] {
				if (not audioMel_.isEmpty()) {
					audioView_ = oa::ViewerAudioView::Mel;
					audioViewTabs_.selected = 2;
				}
			}});
	}
}

oa::Status oa::Viewer::initView() {
	if (resolvedMode_ == oa::ViewerMode::Auto) {
		return oa::Status::error("oa::Viewer has no resolved source");
	}

	OA_RETURN_IF_ERROR(initWindowDecoration());
	OA_RETURN_IF_ERROR(configureNavigation());
	OA_RETURN_IF_ERROR(configureOverlay());
	OA_RETURN_IF_ERROR(registerCommonInput());

	if (resolvedMode_ == oa::ViewerMode::Image) {
		registerImageInput();
	} else if (resolvedMode_ != oa::ViewerMode::Live) {
		registerTemporalInput();
	}
	if (resolvedMode_ == oa::ViewerMode::Live) {
		OA_RETURN_IF_ERROR(config_.liveSource->init(
			input_,
			[this](bool inEnabled) { capturePointer(inEnabled); }));
	}

	if (not config_.showHelp) return oa::Status::ok();
	OaLogInfo(oa::LogComponent::Ui, "═══════════════════════════════════════════════════");
	OaLogInfo(oa::LogComponent::Ui, "oa::Viewer ({})", viewerModeName(resolvedMode_));
	if (resolvedMode_ != oa::ViewerMode::Live and not config_.path.empty()) {
		OaLogInfo(oa::LogComponent::Ui, "  source: {}", config_.path.cStr());
	}
	if (resolvedMode_ == oa::ViewerMode::Image) {
		OaLogInfo(oa::LogComponent::Ui, "  size: {}x{}",
			imageSource().width(), imageSource().height());
		OaLogInfo(oa::LogComponent::Ui,
			"  channels: 1=R  2=G  3=B  4=A  5=RGB");
	} else if (video_.hasValue()) {
		OaLogInfo(oa::LogComponent::Ui, "  codec: {}",
			videoCodecName(video_->getContainerInfo().codec));
		OaLogInfo(oa::LogComponent::Ui, "  size: {}x{} @ {} fps",
			video_->width(), video_->height(), video_->frameRate());
		OaLogInfo(oa::LogComponent::Ui, "  Duration: {:.2f} s",
			static_cast<double>(video_->durationUs()) / 1'000'000.0);
		OaLogInfo(oa::LogComponent::Ui,
			"  Space=play/pause  Arrows=step 1/5 frames  L=loop  timeline=seek");
	} else if (audio_.hasValue()) {
		OaLogInfo(oa::LogComponent::Ui, "  Audio: {} Hz · {} channels · {:.2f} s",
			audio_->sampleRate(), audio_->channelCount(),
			static_cast<double>(audio_->durationUs()) / 1'000'000.0);
		OaLogInfo(oa::LogComponent::Ui,
			"  W/S/M=waveform/spectrum/mel  Space=play/pause  Left/Right=5 s  Up/Down=25 s  L=loop  timeline=seek");
	} else if (resolvedMode_ == oa::ViewerMode::Live) {
		OaLogInfo(oa::LogComponent::Ui, "  source: attached live producer");
	}
	if (hasVisualContent()) OaLogInfo(oa::LogComponent::Ui, "{}", oa::navigationHelpLine());
	OaLogInfo(oa::LogComponent::Ui, "  Q/Esc=Quit");
	OaLogInfo(oa::LogComponent::Ui, "═══════════════════════════════════════════════════");
	return oa::Status::ok();
}

oa::Status oa::Viewer::update(oa::F32 inDeltaMs) {
	if (not oa::isFinite(inDeltaMs) or inDeltaMs < 0.0F) {
		return oa::Status::invalidArgument(
			"oa::Viewer::update requires a finite non-negative delta");
	}
	if (resolvedMode_ == oa::ViewerMode::Live and config_.liveSource != nullptr) {
		OA_RETURN_IF_ERROR(config_.liveSource->update(inDeltaMs));
	}
	if (video_.hasValue()) video_->tick(inDeltaMs);
	if (hasVisualContent()) OA_RETURN_IF_ERROR(nav_.update(inDeltaMs));

	statsAccumMs_ += inDeltaMs;
	++statsFrameCount_;
	if (statsAccumMs_ >= 500.0F) {
		displayFrameMs_ = statsAccumMs_ / static_cast<oa::F32>(statsFrameCount_);
		displayFps_ = static_cast<oa::F32>(statsFrameCount_) * 1000.0F / statsAccumMs_;
		statsAccumMs_ = 0.0F;
		statsFrameCount_ = 0;
	}
	return oa::Status::ok();
}

oa::Status oa::Viewer::routeEvent(const oa::UiEvent& inEvent) {
	if (resolvedMode_ == oa::ViewerMode::Live and config_.liveSource != nullptr) {
		if (liveCapabilities_.receivesEvents) {
			OA_RETURN_IF_ERROR(config_.liveSource->event(inEvent));
		}
		return oa::Status::ok();
	}
	if (not hasVisualContent()) return oa::Status::ok();
	const oa::F32 contentTop = static_cast<oa::F32>(windowDecorationHeight());
	if ((inEvent.type == oa::UiEventType::MouseDown
			or inEvent.type == oa::UiEventType::MouseUp
			or inEvent.type == oa::UiEventType::MouseMove
			or inEvent.type == oa::UiEventType::MouseScroll)
		and inEvent.mouseY < contentTop) {
		return oa::Status::ok();
	}
	const oa::PixelRect controls = timelineRect();
	const oa::F32 contentScale = oa::max(
		0.01F, (windowPixelScaleX_ + windowPixelScaleY_) * 0.5F);
	const oa::F32 controlPadding = 12.0F * contentScale;
	const oa::F64 controlTop = static_cast<oa::F64>(controls.y) - controlPadding;
	const oa::F64 controlBottom = static_cast<oa::F64>(controls.y)
		+ controls.h + controlPadding;
	const bool inControls = static_cast<oa::F64>(inEvent.mouseY) >= controlTop
		and static_cast<oa::F64>(inEvent.mouseY) < controlBottom;
	const bool inTemporalButtons = temporalButtonsRect().contains(
		inEvent.mouseX, inEvent.mouseY);
	const bool timelineActive = ui_.input().activeId != 0U;
	if (not hasTimeline()
		or (not inControls and not inTemporalButtons and not timelineActive)) {
		oa::UiEvent contentEvent = inEvent;
		if (contentEvent.type == oa::UiEventType::WindowResize) {
			contentEvent.windowH = oa::max<oa::I32>(
				1,
				contentEvent.windowH
					- static_cast<oa::I32>(windowDecorationHeight()));
		} else {
			contentEvent.mouseY -= contentTop;
		}
		auto handled = nav_.handleEvent(contentEvent);
		if (not handled.isOk()) return handled.getStatus();
	}
	return oa::Status::ok();
}

void oa::Viewer::drawOverlay(oa::Ui& inUi, oa::PixelRect inDestination) {
	if (not detectionOverlay_.isValid()) return;
	detectionOverlay_.draw(
		inUi,
		textAtlas_,
		inDestination,
		{.x = 0, .y = static_cast<oa::I32>(windowDecorationHeight()),
		 .w = static_cast<oa::I32>(width()),
		 .h = oa::max<oa::I32>(
			1,
			static_cast<oa::I32>(height())
				- static_cast<oa::I32>(windowDecorationHeight()))});
}

void oa::Viewer::renderImage(oa::Ui& inUi) {
	const oa::Texture& image = imageSource();
	if (not image.isValid()) return;
	if (borrowedImageReady_.isValid()) {
		setRenderDependency(borrowedImageReady_);
	}
	const oa::PixelRect destination = viewerContentRect(
		nav_,
		static_cast<oa::U32>(image.width()),
		static_cast<oa::U32>(image.height()),
		windowDecorationHeight());

	if (imageMode_ == ImageViewMode::RGB or not planes_.isValid()) {
		inUi.beginPanel("viewer-image", destination);
		inUi.image(
			image,
			borrowedImage_ != nullptr
				? borrowedImageSourceStageMask_
				: VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
			borrowedImage_ != nullptr
				? borrowedImageSourceAccessMask_
				: VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT);
		inUi.endPanel();
	} else {
		const oa::U32 channel = static_cast<oa::U32>(imageMode_) - 1U;
		if (channel < static_cast<oa::U32>(planes_.channelCount())) {
			inUi.beginPanel("viewer-image", destination);
			inUi.imagePlane(planes_, channel);
			inUi.endPanel();
		}
	}
	drawOverlay(inUi, destination);
}

void oa::Viewer::renderVideo(oa::Ui& inUi) {
	if (not video_.hasValue()) return;
	const auto& frame = video_->currentFrame();
	if (frame.imageView == VK_NULL_HANDLE) return;
	if (frame.ready.isValid()) {
		setRenderDependency(frame.ready);
	}

	const oa::PixelRect destination = viewerContentRect(
		nav_, frame.width, frame.height, windowDecorationHeight());
	inUi.beginPanel("viewer-video", destination);
	inUi.imageVkRgba(
		frame.image,
		frame.imageView,
		static_cast<oa::I32>(frame.width),
		static_cast<oa::I32>(frame.height),
		VK_IMAGE_LAYOUT_GENERAL);
	inUi.endPanel();
	drawOverlay(inUi, destination);

	if (config_.showStats) {
		const oa::F32 contentScale = inUi.contentScale();
		const auto px = [contentScale](oa::I32 inLogical) {
			return oa::max<oa::I32>(1, static_cast<oa::I32>(oa::lround(
				static_cast<oa::F32>(inLogical) * contentScale)));
		};
		inUi.beginPanel("viewer-stats", {
			px(12),
			static_cast<oa::I32>(windowDecorationHeight()) + px(12),
			px(176),
			px(56)});
		inUi.labelFmt("FPS: %.1f", displayFps_);
		inUi.labelFmt("Frame: %.2f ms", displayFrameMs_);
		inUi.endPanel();
	}
}

void oa::Viewer::renderAudio(oa::Ui& inUi) {
	if (not audio_.hasValue()) return;
	if (audioAnalysisReady_.isValid()) {
		setRenderDependency(audioAnalysisReady_);
	}
	renderAudioViewSelector(inUi);
	const oa::U64 duration = mediaDurationUs();
	if (duration == 0U) return;
	oa::F32 fraction = mediaPositionFraction();
	const oa::PixelRect visualization = audioVisualizationRect();
	if (audioView_ == oa::ViewerAudioView::Waveform
		and not audioEnvelope_.isEmpty()) {
		if (inUi.waveformTimeline(
			"viewer-audio-waveform", visualization, audioEnvelope_, fraction)) {
			seekMediaFraction(fraction);
		}
	} else if (audioView_ == oa::ViewerAudioView::Spectrum
		and not audioSpectrum_.isEmpty()) {
		inUi.beginPanel("viewer-audio-spectrum", visualization);
		inUi.heatmap("viewer-audio-spectrum-values", audioSpectrum_, {
			.vMin = -12.0F,
			.vMax = 6.5F,
			.colormap = 1U,
		});
		inUi.endPanel();
	} else if (audioView_ == oa::ViewerAudioView::Mel
		and not audioMel_.isEmpty()) {
		inUi.beginPanel("viewer-audio-mel", visualization);
		inUi.heatmap("viewer-audio-mel-values", audioMel_, {
			.vMin = -12.0F,
			.vMax = 6.5F,
			.colormap = 0U,
		});
		inUi.endPanel();
	}
	bool timelineActive = false;
	const bool timelineChanged = config_.showTimeline and inUi.timeline(
		"viewer-audio-transport", timelineRect(), fraction, &timelineActive);
	if (config_.showTimeline) {
		handleTimelineSeek(fraction, timelineChanged, timelineActive);
	}
}

void oa::Viewer::renderAudioViewSelector(oa::Ui& inUi) {
	if (not config_.showAudioViewSelector) return;
	const oa::UiTabItem items[] = {
		{.id = "waveform", .label = "Waveform", .closable = false,
		 .enabled = not audioEnvelope_.isEmpty()},
		{.id = "spectrum", .label = "Spectrum", .closable = false,
		 .enabled = not audioSpectrum_.isEmpty()},
		{.id = "mel", .label = "Mel", .closable = false,
		 .enabled = not audioMel_.isEmpty()},
	};
	const oa::UiTabBarResult result = inUi.tabBar(
		"viewer-audio-views",
		audioViewTabRect(),
		oa::Span<const oa::UiTabItem>(items, 3U),
		audioViewTabs_,
		{
			.minimumTabWidth = 80,
			.maximumTabWidth = 120,
			.reorderable = false,
		});
	if (result.selectionChanged and audioViewTabs_.selected >= 0
		and audioViewTabs_.selected <= 2) {
		audioView_ = static_cast<oa::ViewerAudioView>(audioViewTabs_.selected);
	}
}

void oa::Viewer::renderTimeline(oa::Ui& inUi) {
	if (not hasTimeline() or resolvedMode_ == oa::ViewerMode::Audio) return;
	const oa::U64 duration = mediaDurationUs();
	if (duration == 0U) return;
	oa::F32 fraction = mediaPositionFraction();
	bool timelineActive = false;
	const bool timelineChanged = inUi.timeline(
		"viewer-transport", timelineRect(), fraction, &timelineActive);
	handleTimelineSeek(fraction, timelineChanged, timelineActive);

	const oa::PixelRect rect = timelineRect();
	const oa::F32 contentScale = inUi.contentScale();
	const auto px = [contentScale](oa::I32 inLogical) {
		return oa::max<oa::I32>(1, static_cast<oa::I32>(oa::lround(
			static_cast<oa::F32>(inLogical) * contentScale)));
	};
	const oa::Color playing = isMediaPlaying()
		? oa::Color::success()
		: oa::Color::warning();
	inUi.rect({px(8), rect.y, px(6), rect.h}, playing);
	if (isMediaLooping()) {
		inUi.rect({rect.x + rect.w + px(10), rect.y, px(6), rect.h},
			oa::Color::accent());
	}
}

void oa::Viewer::renderTemporalButtons(oa::Ui& inUi) {
	if (not hasTimeline()) return;
	const oa::PixelRect group = temporalButtonsRect();
	oa::UiLayout layout;
	layout.padding = oa::UiEdge{};
	layout.gap = 6.0F * inUi.contentScale();
	inUi.beginPanel("viewer-transport-controls", group, layout);
	inUi.beginRow("transport");
	if (inUi.button("|<")) seekMediaFraction(0.0F);
	if (inUi.button("<")) {
		stepTemporal(-1);
	}
	if (inUi.button("Play / Pause")) {
		toggleMediaPlayback();
	}
	if (inUi.button(">")) {
		stepTemporal(1);
	}
	if (inUi.button(">|")) seekMediaFraction(1.0F);
	inUi.endRow();
	inUi.endPanel();
}

oa::Status oa::Viewer::render(oa::Ui& inUi) {
	if (resolvedMode_ == oa::ViewerMode::Image) {
		renderImage(inUi);
	} else if (resolvedMode_ == oa::ViewerMode::Video) {
		renderVideo(inUi);
	} else if (resolvedMode_ == oa::ViewerMode::Audio) {
		renderAudio(inUi);
	} else if (resolvedMode_ == oa::ViewerMode::Live
		and config_.liveSource != nullptr) {
		OA_RETURN_IF_ERROR(config_.liveSource->render(
			inUi, textAtlas_, width(), height()));
		if (liveCapabilities_.publishesRenderDependency) {
			auto ready = config_.liveSource->renderReady();
			if (not ready.isOk()) return ready.getStatus();
			setRenderDependency(*ready);
		}
	}
	renderTimeline(inUi);
	renderTemporalButtons(inUi);
	renderWindowDecoration(inUi);
	return oa::Status::ok();
}

oa::Status oa::Viewer::markRenderSubmitted(const oa::Event& inCompletion) {
	if (not inCompletion.isValid()) {
		return oa::Status::invalidArgument(
			"oa::Viewer render consumption requires a valid completion event");
	}
	if (engine_ == nullptr or not engine_->ownsEvent(inCompletion)) {
		return oa::Status::invalidArgument(
			"oa::Viewer render consumption requires an event from its engine");
	}
	const oavk::TimelineWait wait =
		oa::EventAccess::timelineWait(inCompletion);
	if (wait.semaphore == nullptr or wait.value == 0U) {
		return oa::Status::invalidArgument(
			"oa::Viewer render consumption requires a timeline completion");
	}
	if (resolvedMode_ == oa::ViewerMode::Live and config_.liveSource != nullptr
		and liveCapabilities_.retainsConsumerCompletion) {
		OA_RETURN_IF_ERROR(config_.liveSource->markConsumed(inCompletion));
	}
	if (video_.hasValue()) {
		video_->markCurrentFrameConsumed(inCompletion);
	}
	if (windowTitleGlyphs_.isValid()) {
		OA_RETURN_IF_ERROR(windowTitleGlyphs_.markConsumed(inCompletion));
	}
	return detectionOverlay_.markConsumed(inCompletion);
}

oa::Status oa::Viewer::finalizeBorrowedFrame() {
	if (not borrowedFrameMarkConsumed_
		and not borrowedFrameAbandon_
		and not borrowedFrameCollect_) {
		return oa::Status::ok();
	}

	oa::Status frameStatus = oa::Status::ok();
	if (renderCompletion_.isValid() and borrowedFrameMarkConsumed_) {
		frameStatus = borrowedFrameMarkConsumed_(renderCompletion_);
	} else if (borrowedFrameAbandon_) {
		frameStatus = borrowedFrameAbandon_();
	}
	if (borrowedFrameCollect_) {
		const oa::Status collectStatus = borrowedFrameCollect_();
		if (frameStatus.isOk()) frameStatus = collectStatus;
	}
	borrowedFrameMarkConsumed_ = {};
	borrowedFrameAbandon_ = {};
	borrowedFrameCollect_ = {};
	return frameStatus;
}

const oa::Texture& oa::Viewer::imageSource() const noexcept {
	return borrowedImage_ != nullptr ? *borrowedImage_ : image_;
}

oa::Status oa::Viewer::closeSource() {
	oa::Status sourceStatus = oa::Status::ok();
	if (audioAnalysisReady_.isValid() and sourceEngine_ != nullptr) {
		const oa::Status analysisStatus = sourceEngine_->wait(audioAnalysisReady_);
		if (not analysisStatus.isOk()) sourceStatus = analysisStatus;
	}
	if (resolvedMode_ == oa::ViewerMode::Live and config_.liveSource != nullptr) {
		const oa::Status liveStatus = config_.liveSource->close();
		if (sourceStatus.isOk()) sourceStatus = liveStatus;
	}
	detectionOverlay_ = {};
	if (video_.hasValue()) {
		const oa::Status videoStatus = video_->close();
		if (sourceStatus.isOk() and not videoStatus.isOk()) {
			sourceStatus = videoStatus;
		}
		video_.reset();
	}
	if (audio_.hasValue()) {
		const oa::Status audioStatus = audio_->close();
		if (sourceStatus.isOk() and not audioStatus.isOk()) {
			sourceStatus = audioStatus;
		}
		audio_.reset();
	}
	audioEnvelope_ = {};
	audioSpectrum_ = {};
	audioMel_ = {};
	audioAnalysisReady_ = {};
	audioView_ = oa::ViewerAudioView::Waveform;
	audioViewTabs_ = {};
	image_ = {};
	planes_ = {};
	borrowedImage_ = nullptr;
	borrowedImageReady_ = {};
	borrowedImageSourceStageMask_ = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
	borrowedImageSourceAccessMask_ = VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT;
	liveCapabilities_ = {};
	sourceEngine_ = nullptr;
	resolvedMode_ = oa::ViewerMode::Auto;
	return sourceStatus;
}
