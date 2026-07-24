#include "LunarLander3dViewerSource.h"

#include "LunarLander3d.h"
#include "LunarLander3dRender.h"

#include <Oa/Core/Log.h>
#include <Oa/Runtime/Engine.h>

#include <algorithm>
#include <cmath>

namespace {

constexpr OaU32 OA_LUNAR_VIEW_WIDTH = 960U;
constexpr OaU32 OA_LUNAR_VIEW_HEIGHT = 540U;
constexpr OaU32 OA_LUNAR_VIEW_TARGET_SLOTS = 3U;
constexpr OaU64 OA_LUNAR_VIEW_BASE_SEED = 0x4f414c554e415233ULL;
constexpr OaF32 OA_LUNAR_VIEW_MAX_ACCUMULATED_MS = 250.0F;
constexpr OaF32 OA_LUNAR_VIEW_TERMINAL_HOLD_MS = 2000.0F;

const char* OaLunarViewerActionName(OaLunarAction InAction) noexcept {
	switch (InAction) {
		case OaLunarAction::Coast: return "coast";
		case OaLunarAction::MainEngine: return "main engine";
		case OaLunarAction::PitchPositive: return "pitch +";
		case OaLunarAction::PitchNegative: return "pitch -";
		case OaLunarAction::RollPositive: return "roll +";
		case OaLunarAction::RollNegative: return "roll -";
		case OaLunarAction::YawPositive: return "yaw +";
		case OaLunarAction::YawNegative: return "yaw -";
	}
	return "unknown";
}

const char* OaLunarViewerEndReasonName(OaLunarEndReason InReason) noexcept {
	switch (InReason) {
		case OaLunarEndReason::None: return "in flight";
		case OaLunarEndReason::SafeLanding: return "safe landing";
		case OaLunarEndReason::BodyImpact: return "body impact";
		case OaLunarEndReason::HardFootImpact: return "hard foot impact";
		case OaLunarEndReason::OutOfBounds: return "out of bounds";
		case OaLunarEndReason::NumericalFailure: return "numerical failure";
		case OaLunarEndReason::TimeLimit: return "time limit";
		case OaLunarEndReason::ExternalStop: return "external stop";
		case OaLunarEndReason::InvalidAction: return "invalid action";
	}
	return "unknown";
}

OaPixelRect OaLunarViewerFitRect(
	OaU32 InSourceWidth,
	OaU32 InSourceHeight,
	OaU32 InTargetWidth,
	OaU32 InTargetHeight) noexcept {
	if (InSourceWidth == 0U or InSourceHeight == 0U
		or InTargetWidth == 0U or InTargetHeight == 0U) {
		return {};
	}
	const OaF64 sourceAspect = static_cast<OaF64>(InSourceWidth)
		/ static_cast<OaF64>(InSourceHeight);
	const OaF64 targetAspect = static_cast<OaF64>(InTargetWidth)
		/ static_cast<OaF64>(InTargetHeight);
	OaU32 width = InTargetWidth;
	OaU32 height = InTargetHeight;
	if (targetAspect > sourceAspect) {
		width = static_cast<OaU32>(std::llround(
			static_cast<OaF64>(height) * sourceAspect));
	} else {
		height = static_cast<OaU32>(std::llround(
			static_cast<OaF64>(width) / sourceAspect));
	}
	return {
		static_cast<OaI32>((InTargetWidth - width) / 2U),
		static_cast<OaI32>((InTargetHeight - height) / 2U),
		static_cast<OaI32>(width),
		static_cast<OaI32>(height),
	};
}

} // namespace

class OaLunarLander3dViewerSource::Impl {
public:
	[[nodiscard]] OaStatus Open(OaEngine& InEngine) {
		if (Runtime_ != nullptr) {
			return OaStatus::Error(
				OaStatusCode::FailedPrecondition,
				"LunarLander3d viewer source is already open");
		}
		Runtime_ = &InEngine;
		Failure_ = OaStatus::Ok();
		Paused_ = false;
		Scripted_ = true;
		ManualAction_ = OaLunarAction::Coast;
		EpisodeNumber_ = 0U;
		StepRequested_ = false;
		ResetRequested_ = false;
		FrameScheduled_ = false;

		Config_.SafeDwellSteps_ = 12U;
		Config_.MaxEpisodeSteps_ = 1200U;
		const OaLunarEpisodeManifest manifest =
			OaLunarEpisodeManifest::DeriveVersioned(
				OA_LUNAR_VIEW_BASE_SEED, 0U, 0U,
				Config_.EnvironmentVersion_, OA_LUNAR_TERRAIN_VERSION,
				Config_.PhysicsVersion_, Config_.ObservationVersion_,
				Config_.RewardVersion_, Config_.ContractFingerprint());
		Environment_ = OaLunarScalarEnvironment::CreateSeeded(
			Config_, manifest);
		if (not Environment_.IsValid()) {
			Runtime_ = nullptr;
			return OaStatus::Error(
				OaStatusCode::FailedPrecondition,
				OaString("LunarLander3d scalar environment creation failed: ")
					+ Environment_.Error());
		}
		const OaStatus resetStatus = ResetSimulation_();
		if (resetStatus.IsError()) {
			Runtime_ = nullptr;
			return resetStatus;
		}

		OaLunarLander3dRenderConfig renderConfig;
		renderConfig.Width_ = OA_LUNAR_VIEW_WIDTH;
		renderConfig.Height_ = OA_LUNAR_VIEW_HEIGHT;
		renderConfig.TargetSlotCount_ = OA_LUNAR_VIEW_TARGET_SLOTS;
		auto renderer = OaLunarLander3dRenderSession::Create(
			InEngine, Config_, Environment_.Terrain(), renderConfig);
		if (renderer.IsError()) {
			Runtime_ = nullptr;
			return renderer.GetStatus();
		}
		Renderer_ = OaStdMove(*renderer);
		Camera_ = OaLunarLander3dRenderSession::DefaultCamera(
			OA_LUNAR_VIEW_WIDTH, OA_LUNAR_VIEW_HEIGHT);
		return ProduceFrame_();
	}

	[[nodiscard]] OaStatus Init(OaInputSystem& InInput) {
		if (Runtime_ == nullptr or not Renderer_) {
			return OaStatus::Error(
				OaStatusCode::FailedPrecondition,
				"LunarLander3d viewer source must be open before input setup");
		}
		InInput.RegisterAction({
			.Name = "lunar-pause",
			.Binding = {.Key = OuiKey::Space},
			.Callback = [this] { Paused_ = not Paused_; },
		});
		InInput.RegisterAction({
			.Name = "lunar-step",
			.Binding = {.Key = OuiKey::Right},
			.Callback = [this] { StepRequested_ = true; },
		});
		InInput.RegisterAction({
			.Name = "lunar-reset",
			.Binding = {.Key = OuiKey::R},
			.Callback = [this] { ResetRequested_ = true; },
		});
		InInput.RegisterAction({
			.Name = "lunar-controller-mode",
			.Binding = {.Key = OuiKey::M},
			.Callback = [this] { Scripted_ = not Scripted_; },
		});
		RegisterManualAction_(InInput, "lunar-coast", OuiKey::Num1,
			OaLunarAction::Coast);
		RegisterManualAction_(InInput, "lunar-main", OuiKey::Num2,
			OaLunarAction::MainEngine);
		RegisterManualAction_(InInput, "lunar-pitch-positive", OuiKey::Num3,
			OaLunarAction::PitchPositive);
		RegisterManualAction_(InInput, "lunar-pitch-negative", OuiKey::Num4,
			OaLunarAction::PitchNegative);
		RegisterManualAction_(InInput, "lunar-roll-positive", OuiKey::Num5,
			OaLunarAction::RollPositive);
		RegisterManualAction_(InInput, "lunar-roll-negative", OuiKey::Num6,
			OaLunarAction::RollNegative);
		RegisterManualAction_(InInput, "lunar-yaw-positive", OuiKey::Num7,
			OaLunarAction::YawPositive);
		RegisterManualAction_(InInput, "lunar-yaw-negative", OuiKey::Num8,
			OaLunarAction::YawNegative);
		OA_LOG_INFO(
			OaLogComponent::App,
			"LunarLander3d: Space=pause, Right=single step, R=reset, M=script/manual, 1-8=manual action, Q/Esc=quit");
		return OaStatus::Ok();
	}

	void Update(OaF32 InDeltaMs) {
		if (Runtime_ == nullptr or Failure_.IsError()) return;
		if (ResetRequested_) {
			ResetRequested_ = false;
			const OaStatus status = ResetSimulation_();
			if (status.IsError()) {
				Fail_(status);
				return;
			}
		}

		const OaF32 finiteDelta = std::isfinite(InDeltaMs) and InDeltaMs > 0.0F
			? std::min(InDeltaMs, OA_LUNAR_VIEW_MAX_ACCUMULATED_MS)
			: 0.0F;
		if (Environment_.State().Terminated_
			or Environment_.State().Truncated_) {
			if (not Paused_) TerminalHoldMs_ += finiteDelta;
			if (TerminalHoldMs_ >= OA_LUNAR_VIEW_TERMINAL_HOLD_MS) {
				const OaStatus status = ResetSimulation_();
				if (status.IsError()) {
					Fail_(status);
					return;
				}
			}
		} else if (Paused_) {
			if (StepRequested_) {
				StepRequested_ = false;
				const OaStatus status = StepSimulation_();
				if (status.IsError()) {
					Fail_(status);
					return;
				}
			}
		} else {
			StepRequested_ = false;
			AccumulatorMs_ = std::min(
				AccumulatorMs_ + finiteDelta,
				OA_LUNAR_VIEW_MAX_ACCUMULATED_MS);
			const OaF32 fixedStepMs = static_cast<OaF32>(
				Config_.PolicyTimeStep_ * 1000.0);
			while (AccumulatorMs_ >= fixedStepMs
				and not Environment_.State().Terminated_
				and not Environment_.State().Truncated_) {
				const OaStatus status = StepSimulation_();
				if (status.IsError()) {
					Fail_(status);
					return;
				}
				AccumulatorMs_ -= fixedStepMs;
			}
		}

		const OaStatus frameStatus = ProduceFrame_();
		if (frameStatus.IsError()) Fail_(frameStatus);
	}

	void Render(OaUi& InUi, OaU32 InWidth, OaU32 InHeight) {
		FrameScheduled_ = false;
		if (CurrentFrame_.HasValue()) {
			const auto& frame = *CurrentFrame_;
			const OaPixelRect destination = OaLunarViewerFitRect(
				frame.Width_, frame.Height_, InWidth, InHeight);
			if (destination.W > 0 and destination.H > 0) {
				InUi.BeginPanel("lunar-lander-3d-frame", destination);
				InUi.ImageVkRgba(
					frame.Image_, frame.ImageView_,
					static_cast<OaI32>(frame.Width_),
					static_cast<OaI32>(frame.Height_),
					frame.ImageLayout_,
					VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
					VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT);
				InUi.EndPanel();
				FrameScheduled_ = true;
			}
		}

		const OaI32 panelWidth = std::min<OaI32>(
			360, std::max<OaI32>(0, static_cast<OaI32>(InWidth) - 24));
		if (panelWidth <= 0 or InHeight < 174U) return;
		const OaPixelRect panel{12, 12, panelWidth, 150};
		InUi.Rect(panel, {0.015F, 0.02F, 0.035F, 0.82F});
		InUi.RectOutline(panel, {0.55F, 0.75F, 1.0F, 0.45F}, 1U);
		InUi.BeginPanel("lunar-lander-3d-hud", panel);
		if (Failure_.IsError()) {
			InUi.Label("LunarLander3d failed");
			InUi.LabelFmt("%s", Failure_.ToString().CStr());
			InUi.EndPanel();
			return;
		}
		const OaLunarLander3dState& state = Environment_.State();
		const OaLunarAction displayedAction = Scripted_
			? state.LastAction_
			: ManualAction_;
		InUi.LabelFmt(
			"%s | %s | action: %s",
			Paused_ ? "paused" : "running",
			Scripted_ ? "scripted" : "manual",
			OaLunarViewerActionName(displayedAction));
		InUi.LabelFmt(
			"episode %llu | step %u | return %.2f",
			static_cast<unsigned long long>(EpisodeNumber_),
			state.EpisodeStep_, state.EpisodeReturn_);
		InUi.LabelFmt(
			"altitude %.2f | vertical speed %.2f | fuel %.1f",
			state.Position_.ComponentY_,
			state.LinearVelocity_.ComponentY_, state.Fuel_);
		InUi.LabelFmt(
			"status: %s",
			OaLunarViewerEndReasonName(state.EndReason_));
		InUi.Label("Space pause | Right step | R reset | M mode | 1-8 action");
		InUi.EndPanel();
	}

	[[nodiscard]] OaEvent RenderReady() const {
		return FrameScheduled_ and CurrentFrame_.HasValue()
			? CurrentFrame_->Producer_
			: OaEvent{};
	}

	[[nodiscard]] OaStatus MarkConsumed(const OaEvent& InCompletion) {
		OaStatus consumeStatus = OaStatus::Ok();
		if (FrameScheduled_ and CurrentFrame_.HasValue()) {
			if (not Renderer_) {
				consumeStatus = OaStatus::Error(
					OaStatusCode::Internal,
					"LunarLander3d viewer lost its render session");
			} else {
				consumeStatus = Renderer_->MarkConsumed(
					*CurrentFrame_, InCompletion);
			}
			// Close() owns recovery for both the Submitted and Retired internal
			// states. Never expose this exact frame to a second consumer.
			CurrentFrame_.Reset();
		}
		FrameScheduled_ = false;
		if (consumeStatus.IsError()) return consumeStatus;
		return Failure_;
	}

	[[nodiscard]] OaStatus Close() {
		if (Runtime_ == nullptr) return OaStatus::Ok();
		CurrentFrame_.Reset();
		FrameScheduled_ = false;
		OaStatus closeStatus = OaStatus::Ok();
		if (Renderer_) {
			closeStatus = Renderer_->Close();
			Renderer_.Reset();
		}
		Runtime_ = nullptr;
		return closeStatus;
	}

private:
	void RegisterManualAction_(
		OaInputSystem& InInput,
		OaStringView InName,
		OuiKey InKey,
		OaLunarAction InAction) {
		InInput.RegisterAction({
			.Name = OaString(InName),
			.Binding = {.Key = InKey},
			.Callback = [this, InAction] {
				ManualAction_ = InAction;
				Scripted_ = false;
			},
		});
	}

	[[nodiscard]] OaStatus ResetSimulation_() {
		if (not Environment_.Reset()) {
			return OaStatus::Error(
				OaStatusCode::DataLoss,
				"LunarLander3d deterministic reset produced an invalid state");
		}
		// The fixed vertical start is the scalar oracle used by the headless
		// tutorial. The surrounding terrain remains procedurally seeded.
		OaLunarLander3dState initialState;
		initialState.Position_ = {0.0, 4.0, 0.0};
		initialState.LinearVelocity_.ComponentY_ = -0.2;
		initialState.Fuel_ = Config_.FuelCapacity_;
		if (not Environment_.SetState(initialState)) {
			return OaStatus::Error(
				OaStatusCode::DataLoss,
				"LunarLander3d viewer initial state was rejected");
		}
		++EpisodeNumber_;
		AccumulatorMs_ = 0.0F;
		TerminalHoldMs_ = 0.0F;
		StepRequested_ = false;
		return OaStatus::Ok();
	}

	[[nodiscard]] OaStatus StepSimulation_() {
		OaLunarAction action = ManualAction_;
		if (Scripted_) {
			action = OaLunarScriptedLandingAction(
				Config_, Environment_.State());
		}
		const OaLunarTransition transition = Environment_.Step(
			static_cast<OaU32>(action));
		if (not transition.Valid_) {
			return OaStatus::Error(
				OaStatusCode::DataLoss,
				OaString("LunarLander3d scalar transition failed: ")
					+ transition.Error_);
		}
		return OaStatus::Ok();
	}

	[[nodiscard]] OaStatus ProduceFrame_() {
		if (not Renderer_ or CurrentFrame_.HasValue()) return OaStatus::Ok();
		OA_RETURN_IF_ERROR(Renderer_->Collect());
		const OaStatus beginStatus = Renderer_->BeginFrame(
			Environment_.State(), Camera_);
		if (beginStatus.GetCode() == OaStatusCode::ResourceExhausted) {
			return OaStatus::Ok();
		}
		if (beginStatus.IsError()) return beginStatus;
		auto submitted = Renderer_->SubmitFrame();
		if (submitted.IsError()) {
			const OaStatus submitStatus = submitted.GetStatus();
			const OaStatus cancelStatus = Renderer_->CancelFrame();
			if (cancelStatus.IsError()) {
				OA_LOG_ERROR(
					OaLogComponent::App,
					"LunarLander3d viewer cancellation failed after submit error: %s",
					cancelStatus.ToString().CStr());
			}
			return submitStatus;
		}
		CurrentFrame_.Emplace(OaStdMove(*submitted));
		return OaStatus::Ok();
	}

	void Fail_(const OaStatus& InStatus) {
		if (Failure_.IsError()) return;
		Failure_ = InStatus;
		Paused_ = true;
		OA_LOG_ERROR(
			OaLogComponent::App,
			"LunarLander3d viewer failed: %s",
			InStatus.ToString().CStr());
	}

	OaEngine* Runtime_ = nullptr;
	OaLunarLander3dConfig Config_;
	OaLunarScalarEnvironment Environment_;
	OaUniquePtr<OaLunarLander3dRenderSession> Renderer_;
	OaOpt<OaLunarLander3dRenderFrame> CurrentFrame_;
	OaCameraState Camera_;
	OaStatus Failure_ = OaStatus::Ok();
	OaF32 AccumulatorMs_ = 0.0F;
	OaF32 TerminalHoldMs_ = 0.0F;
	OaU64 EpisodeNumber_ = 0U;
	OaLunarAction ManualAction_ = OaLunarAction::Coast;
	bool Paused_ = false;
	bool Scripted_ = true;
	bool StepRequested_ = false;
	bool ResetRequested_ = false;
	bool FrameScheduled_ = false;
};

OaLunarLander3dViewerSource::OaLunarLander3dViewerSource()
	: Impl_(OaMakeUniquePtr<Impl>()) {}

OaLunarLander3dViewerSource::~OaLunarLander3dViewerSource() = default;

OaStatus OaLunarLander3dViewerSource::Open(OaEngine& InEngine) {
	return Impl_->Open(InEngine);
}

OaStatus OaLunarLander3dViewerSource::Init(
	OaInputSystem& InInput,
	OaFunc<void(bool)> /*InCapturePointer*/) {
	return Impl_->Init(InInput);
}

void OaLunarLander3dViewerSource::Update(OaF32 InDeltaMs) {
	Impl_->Update(InDeltaMs);
}

void OaLunarLander3dViewerSource::Render(
	OaUi& InUi,
	const OaTextAtlas& /*InTextAtlas*/,
	OaU32 InWidth,
	OaU32 InHeight) {
	Impl_->Render(InUi, InWidth, InHeight);
}

OaEvent OaLunarLander3dViewerSource::RenderReady() const {
	return Impl_->RenderReady();
}

OaStatus OaLunarLander3dViewerSource::MarkConsumed(
	const OaEvent& InCompletion) {
	return Impl_->MarkConsumed(InCompletion);
}

OaStatus OaLunarLander3dViewerSource::Close() {
	return Impl_->Close();
}
