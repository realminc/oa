// OA Tutorial: GPU-resident video with reusable detection overlays.
//
// The annotations are synthetic so the display path can be validated without
// bundling a model checkpoint. Replace buildAnnotations() with inference/NMS
// output; the oa::Detection record and GPU consumer stay unchanged.
//
// pipeline:
//   vulkan Video decode -> RGBA image -> boxes + IBM Plex SDF labels -> present
//
// usage:
//   TutorialDetectionVideo [video.mp4]


#include "../ml/tutorialMl.h"
#include "tutorialVision.h"

#include <oa/core/log.h>
#include <oa/ui/navigation.h>
#include <oa/runtime/engine.h>
#include <oa/ui/detectionOverlay.h>
#include <oa/ui/viewer.h>
#include <oa/vision/videoPlayer.h>

#include <stdlib.h>

namespace {

oa::I32 detectionPixel(oa::F64 inValue) noexcept {
	if (not oa::isFinite(inValue)) return 0;
	return static_cast<oa::I32>(oa::clamp<oa::F64>(
		inValue,
		oa::Limits<oa::I32>::min(),
		oa::Limits<oa::I32>::max()));
}

oa::I32 detectionExtent(oa::F64 inValue) noexcept {
	if (not oa::isFinite(inValue) or inValue <= 0.0) return 0;
	return static_cast<oa::I32>(oa::clamp<oa::F64>(
		inValue, 1.0, oa::Limits<oa::I32>::max()));
}

} // namespace

class DetectionVideoSource final : public oa::ViewerLiveSource {
public:
	oa::String path = tutorialVideoPath("shibuya_crossing_1080p30_av1.mp4");
	[[nodiscard]] oa::ViewerLiveCapabilities capabilities() const noexcept override {
		return {
			.receivesEvents = true,
			.publishesRenderDependency = true,
			.retainsConsumerCompletion = true,
		};
	}

	oa::Status open(oa::Engine& inEngine) override {
		engine_ = &inEngine;
		oa::VideoPlayerConfig config;
		config.uri = path;
		config.loop = true;
		config.filter = oa::Filter::Nearest;
		auto video = oa::VideoPlayer::open(inEngine, config);
		if (!video.isOk()) return video.getStatus();
		video_.emplace(oa::move(*video));
		return oa::Status::ok();
	}

	oa::Status init(
		oa::InputSystem& inInput,
		oa::Fn<void(bool)> inCapturePointer) override {
		if (!video_.hasValue()) {
			return oa::Status::error("detection video source is not open");
		}
		if (engine_ == nullptr) {
			return oa::Status::error(
				"detection video source has no engine owner");
		}
		auto overlay = oa::DetectionOverlay::create(*engine_, {
			.maxDetections = 64,
			.maxGlyphs = 2048,
			.thicknessPixels = 3.0F,
			.fontSize = 18.0F,
		});
		if (!overlay.isOk()) {
			return overlay.getStatus();
		}
		overlay_ = oa::move(*overlay);

		nav_.setCapturePointer(oa::move(inCapturePointer));
		OA_RETURN_IF_ERROR(nav_.setContentSize(
			static_cast<oa::F32>(video_->width()),
			static_cast<oa::F32>(video_->height())));
		OA_RETURN_IF_ERROR(registerViewportShortcuts(inInput, nav_));

		inInput.registerAction({.name = "play", .binding = {.key = oa::UiKey::Space},
			.callback = [this] { video_->togglePlay(); }});
		inInput.registerAction({.name = "stepf", .binding = {.key = oa::UiKey::Right},
			.callback = [this] { scrub(1); }});
		inInput.registerAction({.name = "stepb", .binding = {.key = oa::UiKey::Left},
			.callback = [this] { scrub(-1); }});

		OaLogInfo(oa::LogComponent::App,
			"detection display: {} ({}x{})",
			path.cStr(), video_->width(), video_->height());
		OaLogInfo(oa::LogComponent::App,
			"GPU path: decode -> image compose -> boxes + SDF labels -> present");
		OaLogInfo(oa::LogComponent::App,
			"annotations are synthetic display fixtures; replace with model/NMS output");
		return oa::Status::ok();
	}

	oa::Status update(oa::F32 inDeltaMs) override {
		if (!video_.hasValue()) {
			return oa::Status::error(
				oa::StatusCode::FailedPrecondition,
				"detection video update requires an open source");
		}
		video_->tick(inDeltaMs);
		return nav_.update(inDeltaMs);
	}

	oa::Status event(const oa::UiEvent& inEvent) override {
		auto handled = nav_.handleEvent(inEvent);
		return handled.isOk() ? oa::Status::ok() : handled.getStatus();
	}

	oa::Status render(
		oa::Ui& inUi,
		const oa::TextAtlas& inTextAtlas,
		oa::U32 inWidth,
		oa::U32 inHeight) override {
		if (!video_.hasValue()) {
			return oa::Status::error(
				oa::StatusCode::FailedPrecondition,
				"detection video render requires an open source");
		}
		if (inWidth == 0U || inHeight == 0U
			|| inWidth > static_cast<oa::U32>(oa::Limits<oa::I32>::max())
			|| inHeight > static_cast<oa::U32>(oa::Limits<oa::I32>::max())) {
			return oa::Status::invalidArgument(
				"detection video render requires a signed positive UI extent");
		}
		if (inWidth != viewWidth_ or inHeight != viewHeight_) {
			viewWidth_ = inWidth;
			viewHeight_ = inHeight;
			const oa::Status resizeStatus = nav_.setWindowSize(
				static_cast<oa::F32>(inWidth),
				static_cast<oa::F32>(inHeight));
			if (not resizeStatus.isOk()) return resizeStatus;
			if (not viewInitialized_) {
				const oa::Status fitStatus = nav_.fitToWindow(false);
				if (not fitStatus.isOk()) return fitStatus;
				viewInitialized_ = true;
			}
		}
		if (video_->index() != annotatedFrame_) {
			OA_RETURN_IF_ERROR(updateAnnotations(inTextAtlas));
		}
		const auto& frame = video_->currentFrame();
		if (frame.imageView == VK_NULL_HANDLE) return oa::Status::ok();

		const oa::PixelRect destination = {
			.x = detectionPixel(nav_.panX()),
			.y = detectionPixel(nav_.panY()),
			.w = detectionExtent(static_cast<oa::F64>(frame.width) * nav_.zoom()),
			.h = detectionExtent(static_cast<oa::F64>(frame.height) * nav_.zoom()),
		};
		const oa::PixelRect clip = {
			.x = 0,
			.y = 0,
			.w = static_cast<oa::I32>(inWidth),
			.h = static_cast<oa::I32>(inHeight),
		};

		inUi.beginPanel("detection-video", destination);
		inUi.imageVkRgba(
			frame.image,
			frame.imageView,
			static_cast<oa::I32>(frame.width),
			static_cast<oa::I32>(frame.height),
			VK_IMAGE_LAYOUT_GENERAL);
		inUi.endPanel();
		overlay_.draw(inUi, inTextAtlas, destination, clip);
		return oa::Status::ok();
	}

	[[nodiscard]] oa::Result<oa::Event> renderReady() const override {
		if (!video_.hasValue()) {
			return oa::Status::error(
				oa::StatusCode::FailedPrecondition,
				"detection video readiness requires an open source");
		}
		return video_->currentFrame().ready;
	}

	oa::Status markConsumed(const oa::Event& inCompletion) override {
		if (not inCompletion.isValid()) {
			return oa::Status::invalidArgument(
				"detection viewer consumption requires a valid completion event");
		}
		if (video_.hasValue()) {
			video_->markCurrentFrameConsumed(inCompletion);
		}
		return overlay_.markConsumed(inCompletion);
	}

	oa::Status close() override {
		overlay_ = {};
		engine_ = nullptr;
		if (video_.hasValue()) {
			const oa::Status status = video_->close();
			video_.reset();
			return status;
		}
		return oa::Status::ok();
	}

private:
	void scrub(oa::I32 inFrames) {
		if (!video_.hasValue()) return;
		video_->pause();
		const oa::Status status = video_->stepFrames(inFrames);
		if (!status.isOk()) {
			OaLogWarn(oa::LogComponent::App,
				"Video scrub failed: {}", status.toString().cStr());
		}
	}

	oa::Status updateAnnotations(const oa::TextAtlas& inTextAtlas) {
		if (!video_.hasValue() || !overlay_.isValid()) {
			return oa::Status::error(
				oa::StatusCode::FailedPrecondition,
				"detection annotations require open video and overlay state");
		}
		const oa::F32 phase = static_cast<oa::F32>(video_->index() % 240) / 240.0F;
		const oa::F32 wave = oa::sin(phase * 6.28318530718F);

		oa::Vector<oa::DetectionOverlayItem> items;
		items.pushBack({
			.detection = {
				.centerX = 0.29F + wave * 0.035F,
				.centerY = 0.53F,
				.width = 0.12F,
				.height = 0.42F,
				.confidence = 0.98F,
				.classId = 0,
				.colorRgba = oa::Color::success().toU32(),
				.trackId = 17,
			},
			.label = "person 98% / track 17",
		});
		items.pushBack({
			.detection = {
				.centerX = 0.62F - wave * 0.02F,
				.centerY = 0.67F,
				.width = 0.24F,
				.height = 0.20F,
				.confidence = 0.93F,
				.classId = 2,
				.colorRgba = oa::Color::cyan().toU32(),
				.trackId = 31,
			},
			.label = "car 93% / track 31",
		});
		items.pushBack({
			.detection = {
				.centerX = 0.79F,
				.centerY = 0.48F + wave * 0.025F,
				.width = 0.09F,
				.height = 0.25F,
				.confidence = 0.87F,
				.classId = 1,
				.colorRgba = oa::Color::warning().toU32(),
				.trackId = 44,
			},
			.label = "bicycle 87% / track 44",
		});

		const oa::Status status = overlay_.update(
			oa::Span<const oa::DetectionOverlayItem>(items.data(), items.size()),
			inTextAtlas);
		if (status.isOk()) {
			annotatedFrame_ = video_->index();
			return oa::Status::ok();
		}
		// A full completion-tracked ring deliberately drops this annotation
		// refresh instead of blocking the Viewer frame.
		if (status.getCode() == oa::StatusCode::Unavailable) return oa::Status::ok();
		return status;
	}

	oa::Optional<oa::VideoPlayer> video_;
	oa::DetectionOverlay overlay_;
	oa::Engine* engine_ = nullptr;
	oa::Navigation nav_;
	oa::I64 annotatedFrame_ = -1;
	oa::U32 viewWidth_ = 0;
	oa::U32 viewHeight_ = 0;
	bool viewInitialized_ = false;
};

int main(int argc, char** argv) {
	const oa::I32 deviceIndex = tutorialPreParseDeviceIndex(argc, argv);
	if (deviceIndex >= 0) {
		const oa::String index = oa::toString(static_cast<oa::I64>(deviceIndex));
#if defined(_WIN32)
		_putenv_s("OA_DEVICE", index.cStr());
#else
		::setenv("OA_DEVICE", index.cStr(), 1);
#endif
	}

	DetectionVideoSource source;
	if (argc > 1) source.path = argv[1];
	oa::Viewer viewer({
		.mode = oa::ViewerMode::Live,
		.liveSource = &source,
		.title = "OA detection Video",
		.width = 1280,
		.height = 720,
		.showHelp = false,
		.showStats = false,
		.showTimeline = false,
	});
	return viewer.run().isOk() ? 0 : 1;
}
