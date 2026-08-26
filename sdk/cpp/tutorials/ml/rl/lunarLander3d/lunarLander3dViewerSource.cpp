#include "lunarLander3dViewerSource.h"

#include <ml/rl/lunarLander3d.h>
#include "lunarLander3dRender.h"

#include <oa/core/log.h>
#include <oa/runtime/engine.h>

#include <core/streamText.h>

#include <algorithm>
#include <cmath>
#include <limits>

namespace {

constexpr oa::U32 OA_LUNAR_VIEW_WIDTH = 960U;
constexpr oa::U32 OA_LUNAR_VIEW_HEIGHT = 540U;
constexpr oa::U32 OA_LUNAR_VIEW_TARGET_SLOTS = 3U;
constexpr oa::U64 OA_LUNAR_VIEW_BASE_SEED = 0x4f414c554e415233ULL;
constexpr oa::F32 OA_LUNAR_VIEW_MAX_ACCUMULATED_MS = 250.0F;
constexpr oa::F32 OA_LUNAR_VIEW_TERMINAL_HOLD_MS = 2000.0F;

const char* lunarViewerActionName(oa::LunarAction inAction) noexcept {
	switch (inAction) {
		case oa::LunarAction::Coast: return "coast";
		case oa::LunarAction::MainEngine: return "main engine";
		case oa::LunarAction::PitchPositive: return "pitch +";
		case oa::LunarAction::PitchNegative: return "pitch -";
		case oa::LunarAction::RollPositive: return "roll +";
		case oa::LunarAction::RollNegative: return "roll -";
		case oa::LunarAction::YawPositive: return "yaw +";
		case oa::LunarAction::YawNegative: return "yaw -";
	}
	return "unknown";
}

const char* lunarViewerEndReasonName(oa::LunarEndReason inReason) noexcept {
	switch (inReason) {
		case oa::LunarEndReason::None: return "in flight";
		case oa::LunarEndReason::SafeLanding: return "safe landing";
		case oa::LunarEndReason::BodyImpact: return "body impact";
		case oa::LunarEndReason::HardFootImpact: return "hard foot impact";
		case oa::LunarEndReason::OutOfBounds: return "out of bounds";
		case oa::LunarEndReason::NumericalFailure: return "numerical failure";
		case oa::LunarEndReason::TimeLimit: return "time limit";
		case oa::LunarEndReason::ExternalStop: return "external stop";
		case oa::LunarEndReason::InvalidAction: return "invalid action";
	}
	return "unknown";
}

oa::PixelRect lunarViewerFitRect(
	oa::U32 inSourceWidth,
	oa::U32 inSourceHeight,
	oa::U32 inTargetWidth,
	oa::U32 inTargetHeight) noexcept {
	if (inSourceWidth == 0U or inSourceHeight == 0U
		or inTargetWidth == 0U or inTargetHeight == 0U) {
		return {};
	}
	const oa::F64 sourceAspect = static_cast<oa::F64>(inSourceWidth)
		/ static_cast<oa::F64>(inSourceHeight);
	const oa::F64 targetAspect = static_cast<oa::F64>(inTargetWidth)
		/ static_cast<oa::F64>(inTargetHeight);
	oa::U32 width = inTargetWidth;
	oa::U32 height = inTargetHeight;
	if (targetAspect > sourceAspect) {
		width = static_cast<oa::U32>(std::llround(
			static_cast<oa::F64>(height) * sourceAspect));
	} else {
		height = static_cast<oa::U32>(std::llround(
			static_cast<oa::F64>(width) / sourceAspect));
	}
	return {
		static_cast<oa::I32>((inTargetWidth - width) / 2U),
		static_cast<oa::I32>((inTargetHeight - height) / 2U),
		static_cast<oa::I32>(width),
		static_cast<oa::I32>(height),
	};
}

} // namespace

class LunarLander3dViewerSource::Impl {
public:
	explicit Impl(oa::U32 inSampleCount) noexcept
		: sampleCount_(inSampleCount) {}

	[[nodiscard]] oa::Status open(oa::Engine& inEngine) {
		if (runtime_ != nullptr) {
			return oa::Status::error(
				oa::StatusCode::FailedPrecondition,
				"LunarLander3d viewer source is already open");
		}
		runtime_ = &inEngine;
		failure_ = oa::Status::ok();
		paused_ = false;
		scripted_ = true;
		manualAction_ = oa::LunarAction::Coast;
		episodeNumber_ = 0U;
		stepRequested_ = false;
		resetRequested_ = false;
		frameScheduled_ = false;

		config_.safeDwellSteps_ = 12U;
		config_.maxEpisodeSteps_ = 1200U;
		const oa::LunarEpisodeManifest manifest =
			oa::LunarEpisodeManifest::deriveVersioned(
				OA_LUNAR_VIEW_BASE_SEED, 0U, 0U,
				config_.environmentVersion_, oa::kLunarTerrainVersion,
				config_.physicsVersion_, config_.observationVersion_,
				config_.rewardVersion_, config_.contractFingerprint());
		environment_ = oa::LunarScalarEnvironment::createSeeded(
			config_, manifest);
		if (not environment_.isValid()) {
			runtime_ = nullptr;
			return oa::Status::error(
				oa::StatusCode::FailedPrecondition,
				oa::String("LunarLander3d scalar environment creation failed: ")
					+ oa::sdk::fromStdString(environment_.error()));
		}
		const oa::Status resetStatus = resetSimulation_();
		if (resetStatus.isError()) {
			runtime_ = nullptr;
			return resetStatus;
		}

		LunarLander3dRenderConfig renderConfig;
		renderConfig.width_ = OA_LUNAR_VIEW_WIDTH;
		renderConfig.height_ = OA_LUNAR_VIEW_HEIGHT;
		renderConfig.targetSlotCount_ = OA_LUNAR_VIEW_TARGET_SLOTS;
		renderConfig.sampleCount_ = sampleCount_;
		auto renderer = LunarLander3dRenderSession::create(
			inEngine, config_, environment_.terrain(), renderConfig);
		if (renderer.isError()) {
			runtime_ = nullptr;
			return renderer.getStatus();
		}
		renderer_ = oa::move(*renderer);
		camera_ = LunarLander3dRenderSession::defaultCamera(
			OA_LUNAR_VIEW_WIDTH, OA_LUNAR_VIEW_HEIGHT);
		return produceFrame_();
	}

	[[nodiscard]] oa::Status init(oa::InputSystem& inInput) {
		if (runtime_ == nullptr or not renderer_) {
			return oa::Status::error(
				oa::StatusCode::FailedPrecondition,
				"LunarLander3d viewer source must be open before input setup");
		}
		inInput.registerAction({
			.name = "lunar-pause",
			.binding = {.key = oa::UiKey::Space},
			.callback = [this] { paused_ = not paused_; },
		});
		inInput.registerAction({
			.name = "lunar-step",
			.binding = {.key = oa::UiKey::Right},
			.callback = [this] { stepRequested_ = true; },
		});
		inInput.registerAction({
			.name = "lunar-reset",
			.binding = {.key = oa::UiKey::R},
			.callback = [this] { resetRequested_ = true; },
		});
		inInput.registerAction({
			.name = "lunar-controller-mode",
			.binding = {.key = oa::UiKey::M},
			.callback = [this] { scripted_ = not scripted_; },
		});
		registerManualAction_(inInput, "lunar-coast", oa::UiKey::Num1,
			oa::LunarAction::Coast);
		registerManualAction_(inInput, "lunar-main", oa::UiKey::Num2,
			oa::LunarAction::MainEngine);
		registerManualAction_(inInput, "lunar-pitch-positive", oa::UiKey::Num3,
			oa::LunarAction::PitchPositive);
		registerManualAction_(inInput, "lunar-pitch-negative", oa::UiKey::Num4,
			oa::LunarAction::PitchNegative);
		registerManualAction_(inInput, "lunar-roll-positive", oa::UiKey::Num5,
			oa::LunarAction::RollPositive);
		registerManualAction_(inInput, "lunar-roll-negative", oa::UiKey::Num6,
			oa::LunarAction::RollNegative);
		registerManualAction_(inInput, "lunar-yaw-positive", oa::UiKey::Num7,
			oa::LunarAction::YawPositive);
		registerManualAction_(inInput, "lunar-yaw-negative", oa::UiKey::Num8,
			oa::LunarAction::YawNegative);
		OaLogInfo(
			oa::LogComponent::App,
			"LunarLander3d: Space=pause, Right=single step, R=reset, M=script/manual, 1-8=manual action, Q/Esc=quit");
		return oa::Status::ok();
	}

	void update(oa::F32 inDeltaMs) {
		if (runtime_ == nullptr or failure_.isError()) return;
		if (resetRequested_) {
			resetRequested_ = false;
			const oa::Status status = resetSimulation_();
			if (status.isError()) {
				fail_(status);
				return;
			}
		}

		const oa::F32 finiteDelta = std::isfinite(inDeltaMs) and inDeltaMs > 0.0F
			? std::min(inDeltaMs, OA_LUNAR_VIEW_MAX_ACCUMULATED_MS)
			: 0.0F;
		if (environment_.state().terminated_
			or environment_.state().truncated_) {
			if (not paused_) terminalHoldMs_ += finiteDelta;
			if (terminalHoldMs_ >= OA_LUNAR_VIEW_TERMINAL_HOLD_MS) {
				const oa::Status status = resetSimulation_();
				if (status.isError()) {
					fail_(status);
					return;
				}
			}
		} else if (paused_) {
			if (stepRequested_) {
				stepRequested_ = false;
				const oa::Status status = stepSimulation_();
				if (status.isError()) {
					fail_(status);
					return;
				}
			}
		} else {
			stepRequested_ = false;
			accumulatorMs_ = std::min(
				accumulatorMs_ + finiteDelta,
				OA_LUNAR_VIEW_MAX_ACCUMULATED_MS);
			const oa::F32 fixedStepMs = static_cast<oa::F32>(
				config_.policyTimeStep_ * 1000.0);
			while (accumulatorMs_ >= fixedStepMs
				and not environment_.state().terminated_
				and not environment_.state().truncated_) {
				const oa::Status status = stepSimulation_();
				if (status.isError()) {
					fail_(status);
					return;
				}
				accumulatorMs_ -= fixedStepMs;
			}
		}

		const oa::Status frameStatus = produceFrame_();
		if (frameStatus.isError()) fail_(frameStatus);
	}

	void render(oa::Ui& inUi, oa::U32 inWidth, oa::U32 inHeight) {
		frameScheduled_ = false;
		if (currentFrame_.hasValue()) {
			const auto& frame = *currentFrame_;
			const oa::PixelRect destination = lunarViewerFitRect(
				frame.width(), frame.height(), inWidth, inHeight);
			if (destination.w > 0 and destination.h > 0) {
				inUi.beginPanel("lunar-lander-3d-frame", destination);
				inUi.image(
					frame.color(),
					VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
					VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT);
				inUi.endPanel();
				frameScheduled_ = true;
			}
		}

		const oa::I32 panelWidth = std::min<oa::I32>(
			360, std::max<oa::I32>(0, static_cast<oa::I32>(inWidth) - 24));
		if (panelWidth <= 0 or inHeight < 174U) return;
		const oa::PixelRect panel{12, 12, panelWidth, 150};
		inUi.rect(panel, {0.015F, 0.02F, 0.035F, 0.82F});
		inUi.rectOutline(panel, {0.55F, 0.75F, 1.0F, 0.45F}, 1U);
		inUi.beginPanel("lunar-lander-3d-hud", panel);
		if (failure_.isError()) {
			inUi.label("LunarLander3d failed");
			inUi.labelFmt("%s", failure_.toString().cStr());
			inUi.endPanel();
			return;
		}
		const oa::LunarLander3dState& state = environment_.state();
		const oa::LunarAction displayedAction = scripted_
			? state.lastAction_
			: manualAction_;
		inUi.labelFmt(
			"%s | %s | action: %s",
			paused_ ? "paused" : "running",
			scripted_ ? "scripted" : "manual",
			lunarViewerActionName(displayedAction));
		inUi.labelFmt(
			"episode %llu | step %u | return %.2f",
			static_cast<unsigned long long>(episodeNumber_),
			state.episodeStep_, state.episodeReturn_);
		inUi.labelFmt(
			"altitude %.2f | vertical speed %.2f | fuel %.1f",
			state.position_.y,
			state.linearVelocity_.y, state.fuel_);
		inUi.labelFmt(
			"status: %s",
			lunarViewerEndReasonName(state.endReason_));
		inUi.label("Space pause | Right step | R reset | M mode | 1-8 action");
		inUi.endPanel();
	}

	[[nodiscard]] oa::Event renderReady() const {
		return frameScheduled_ and currentFrame_.hasValue()
			? currentFrame_->producer()
			: oa::Event{};
	}

	[[nodiscard]] const oa::Status& status() const noexcept {
		return failure_;
	}

	[[nodiscard]] oa::Status markConsumed(const oa::Event& inCompletion) {
		oa::Status consumeStatus = oa::Status::ok();
		if (frameScheduled_ and currentFrame_.hasValue()) {
			if (not renderer_) {
				consumeStatus = oa::Status::error(
					oa::StatusCode::Internal,
					"LunarLander3d viewer lost its render session");
			} else {
				consumeStatus = renderer_->markConsumed(
					*currentFrame_, inCompletion);
			}
			// close() owns recovery for both the Submitted and Retired internal
			// states. Never expose this exact frame to a second consumer.
			currentFrame_.reset();
		}
		frameScheduled_ = false;
		if (consumeStatus.isError()) return consumeStatus;
		return failure_;
	}

	[[nodiscard]] oa::Status close() {
		if (runtime_ == nullptr) return oa::Status::ok();
		currentFrame_.reset();
		frameScheduled_ = false;
		oa::Status closeStatus = oa::Status::ok();
		if (renderer_) {
			closeStatus = renderer_->close();
			renderer_.reset();
		}
		runtime_ = nullptr;
		return closeStatus;
	}

private:
	void registerManualAction_(
		oa::InputSystem& inInput,
		oa::StringView inName,
		oa::UiKey inKey,
		oa::LunarAction inAction) {
		inInput.registerAction({
			.name = oa::String(inName),
			.binding = {.key = inKey},
			.callback = [this, inAction] {
				manualAction_ = inAction;
				scripted_ = false;
			},
		});
	}

	[[nodiscard]] oa::Status resetSimulation_() {
		if (not environment_.reset()) {
			return oa::Status::error(
				oa::StatusCode::DataLoss,
				"LunarLander3d deterministic reset produced an invalid state");
		}
		// The fixed vertical start is the scalar oracle used by the headless
		// tutorial. The surrounding terrain remains procedurally seeded.
		oa::LunarLander3dState initialState;
		initialState.position_ = {0.0, 4.0, 0.0};
		initialState.linearVelocity_.y = -0.2;
		initialState.fuel_ = config_.fuelCapacity_;
		if (not environment_.setState(initialState)) {
			return oa::Status::error(
				oa::StatusCode::DataLoss,
				"LunarLander3d viewer initial state was rejected");
		}
		++episodeNumber_;
		accumulatorMs_ = 0.0F;
		terminalHoldMs_ = 0.0F;
		stepRequested_ = false;
		return oa::Status::ok();
	}

	[[nodiscard]] oa::Status stepSimulation_() {
		oa::LunarAction action = manualAction_;
		if (scripted_) {
			action = oa::lunarScriptedLandingAction(
				config_, environment_.state());
		}
		const oa::LunarTransition transition = environment_.step(
			static_cast<oa::U32>(action));
		if (not transition.valid_) {
			return oa::Status::error(
				oa::StatusCode::DataLoss,
				oa::String("LunarLander3d scalar transition failed: ")
					+ oa::sdk::fromStdString(transition.error_));
		}
		return oa::Status::ok();
	}

	[[nodiscard]] oa::Status produceFrame_() {
		if (not renderer_ or currentFrame_.hasValue()) return oa::Status::ok();
		OA_RETURN_IF_ERROR(renderer_->collect());
		const oa::Status beginStatus = renderer_->beginFrame(
			environment_.state(), camera_);
		if (beginStatus.getCode() == oa::StatusCode::ResourceExhausted) {
			return oa::Status::ok();
		}
		if (beginStatus.isError()) return beginStatus;
		auto submitted = renderer_->submitFrame();
		if (submitted.isError()) {
			const oa::Status submitStatus = submitted.getStatus();
			const oa::Status cancelStatus = renderer_->cancelFrame();
			if (cancelStatus.isError()) {
				OaLogError(
					oa::LogComponent::App,
					"LunarLander3d viewer cancellation failed after submit error: %s",
					cancelStatus.toString().cStr());
			}
			return submitStatus;
		}
		currentFrame_.emplace(oa::move(*submitted));
		return oa::Status::ok();
	}

	void fail_(const oa::Status& inStatus) {
		if (failure_.isError()) return;
		failure_ = inStatus;
		paused_ = true;
		OaLogError(
			oa::LogComponent::App,
			"LunarLander3d viewer failed: %s",
			inStatus.toString().cStr());
	}

	oa::Engine* runtime_ = nullptr;
	oa::LunarLander3dConfig config_;
	oa::LunarScalarEnvironment environment_;
	oa::UniquePtr<LunarLander3dRenderSession> renderer_;
	oa::Optional<oa::RenderFrame> currentFrame_;
	oa::CameraState camera_;
	oa::Status failure_ = oa::Status::ok();
	oa::F32 accumulatorMs_ = 0.0F;
	oa::F32 terminalHoldMs_ = 0.0F;
	oa::U64 episodeNumber_ = 0U;
	oa::U32 sampleCount_ = 1U;
	oa::LunarAction manualAction_ = oa::LunarAction::Coast;
	bool paused_ = false;
	bool scripted_ = true;
	bool stepRequested_ = false;
	bool resetRequested_ = false;
	bool frameScheduled_ = false;
};

LunarLander3dViewerSource::LunarLander3dViewerSource(
	oa::U32 inSampleCount)
	: impl_(oa::makeUnique<Impl>(inSampleCount)) {}

LunarLander3dViewerSource::~LunarLander3dViewerSource() = default;

oa::Status LunarLander3dViewerSource::open(oa::Engine& inEngine) {
	return impl_->open(inEngine);
}

oa::Status LunarLander3dViewerSource::init(
	oa::InputSystem& inInput,
	oa::Fn<void(bool)> /*inCapturePointer*/) {
	return impl_->init(inInput);
}

oa::Status LunarLander3dViewerSource::update(oa::F32 inDeltaMs) {
	if (not std::isfinite(inDeltaMs) or inDeltaMs < 0.0F) {
		return oa::Status::invalidArgument(
			"LunarLander3d viewer update requires a finite non-negative delta");
	}
	impl_->update(inDeltaMs);
	return impl_->status();
}

oa::Status LunarLander3dViewerSource::render(
	oa::Ui& inUi,
	const oa::TextAtlas& /*inTextAtlas*/,
	oa::U32 inWidth,
	oa::U32 inHeight) {
	if (inWidth > static_cast<oa::U32>(std::numeric_limits<oa::I32>::max())
		or inHeight > static_cast<oa::U32>(std::numeric_limits<oa::I32>::max())) {
		return oa::Status::invalidArgument(
			"LunarLander3d viewer extent exceeds signed UI coordinates");
	}
	impl_->render(inUi, inWidth, inHeight);
	return impl_->status();
}

oa::Result<oa::Event> LunarLander3dViewerSource::renderReady() const {
	if (impl_->status().isError()) return impl_->status();
	return impl_->renderReady();
}

oa::Status LunarLander3dViewerSource::markConsumed(
	const oa::Event& inCompletion) {
	return impl_->markConsumed(inCompletion);
}

oa::Status LunarLander3dViewerSource::close() {
	return impl_->close();
}
