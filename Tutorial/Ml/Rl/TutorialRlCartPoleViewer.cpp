#include "CartPolePpo.h"

#include <Oa/Core/Log.h>
#include <Oa/Ml/TrainingSession.h>
#include <Oa/Ui/Viewer.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdio>

namespace {

class CartPoleLiveSource final : public OaViewerLiveSource {
public:
	OaStatus Open(OaEngine& InEngine) override {
		Runtime_ = &InEngine;
		return ResetSession_();
	}

	OaStatus Init(
		OaInputSystem& InInput,
		OaFunc<void(bool)> /*InCapturePointer*/) override {
		for (auto& labels : LabelSlots_) {
			auto buffer = OaGlyphBuffer::CreateHostUpload(
				*Runtime_, kMaxLabelGlyphs);
			if (buffer.IsError()) {
				Fail_(buffer.GetStatus());
				return buffer.GetStatus();
			}
			labels = OaStdMove(*buffer);
		}
		InInput.RegisterAction({
			.Name = "cartpole-pause",
			.Binding = {.Key = OuiKey::Space},
			.Callback = [this] {
				if (!Session_) return;
				auto& control = Session_->Control();
				if (control.State() == OaTrainingState::Running) {
					(void)control.Pause();
				} else if (control.State() == OaTrainingState::Paused) {
					(void)control.Resume();
				}
			},
		});
		InInput.RegisterAction({
			.Name = "cartpole-step",
			.Binding = {.Key = OuiKey::Right},
			.Callback = [this] { StepRequested_ = true; },
		});
		InInput.RegisterAction({
			.Name = "cartpole-reset",
			.Binding = {.Key = OuiKey::R},
			.Callback = [this] { ResetRequested_ = true; },
		});
		OA_LOG_INFO(OaLogComponent::App,
			"CartPole PPO: Space=pause · Right=one update · R=restart · Q/Esc=quit");
		return OaStatus::Ok();
	}

	void Update(OaF32 InDeltaMs) override {
		if (std::isfinite(InDeltaMs) && InDeltaMs > 0.0F) {
			FrameMsEma_ = FrameMsEma_ <= 0.0F
				? InDeltaMs
				: FrameMsEma_ * 0.90F + InDeltaMs * 0.10F;
			ViewerFps_ = 1000.0F / FrameMsEma_;
		}
		if (ResetRequested_) {
			ResetRequested_ = false;
			if (const OaStatus status = ResetSession_(); status.IsError()) {
				Fail_(status);
			}
		}
		if (!Session_ || Failed_) return;

		const bool stepRequested = StepRequested_;
		StepRequested_ = false;
		if (stepRequested
			&& Session_->Control().State() == OaTrainingState::Paused) {
			(void)Session_->Control().Resume();
			RePauseAfterStep_ = true;
		}
		if (!Session_->IsDone()) {
			const OaU64 stepBefore = Session_->OptimizerStep();
			const auto begin = std::chrono::steady_clock::now();
			if (const OaStatus status = Session_->Advance(); status.IsError()) {
				Fail_(status);
				return;
			}
			const OaF32 elapsed = static_cast<OaF32>(
				std::chrono::duration<OaF64, std::milli>(
					std::chrono::steady_clock::now() - begin).count());
			if (Session_->OptimizerStep() != stepBefore) UpdateMs_.PushBack(elapsed);
			if (RePauseAfterStep_ && Session_->OptimizerStep() != stepBefore) {
				(void)Session_->Control().Pause();
				RePauseAfterStep_ = false;
			}
			const OaU32 rollout = Session_->Metrics().Rollout;
			if (rollout != LastEvaluatedRollout_
				&& (rollout % 5U == 0U || Session_->IsDone())) {
				LastEvaluatedRollout_ = rollout;
				auto evaluation = Session_->Evaluate(kEvaluationSeed);
				if (evaluation.IsError()) {
					Fail_(evaluation.GetStatus());
					return;
				}
				OA_LOG_INFO(OaLogComponent::App,
					"CartPole rollout %u/%u: return %.2f · loss %.5f",
					rollout, Session_->Config().Rollouts,
					evaluation->MeanCompletedReturn,
					Session_->Metrics().TotalLoss);
			}
		} else {
			DemoAccumMs_ += InDeltaMs;
			if (DemoAccumMs_ >= 20.0F) {
				DemoAccumMs_ = std::fmod(DemoAccumMs_, 20.0F);
				if (const OaStatus status = Session_->Demonstrate(); status.IsError()) {
					Fail_(status);
					return;
				}
			}
		}
		auto snapshot = Session_->SnapshotLane(0);
		if (snapshot.IsError()) {
			Fail_(snapshot.GetStatus());
			return;
		}
		Snapshot_ = *snapshot;
	}

	void Render(
		OaUi& InUi,
		const OaTextAtlas& InTextAtlas,
		OaU32 InWidth,
		OaU32 InHeight) override {
		const OaI32 width = static_cast<OaI32>(InWidth);
		const OaI32 height = static_cast<OaI32>(InHeight);
		if (width < 320 || height < 240) return;
		const OaF32 uiScale = std::clamp(
			static_cast<OaF32>(height) / 720.0F, 1.0F, 2.0F);
		const OaI32 plotHeaderHeight = static_cast<OaI32>(
			std::round(42.0F * uiScale));
		const OaI32 margin = 20;
		const OaI32 gap = 16;
		const OaI32 simulationWidth = std::max<OaI32>(200,
			static_cast<OaI32>(static_cast<OaF32>(width) * 0.62F));
		const OaPixelRect simulation{
			margin, margin,
			std::min(simulationWidth, width - margin * 2 - 160),
			height - margin * 2};
		const OaI32 plotX = simulation.X + simulation.W + gap;
		const OaI32 plotWidth = width - plotX - margin;
		const OaI32 plotHeight = std::max<OaI32>(48,
			(simulation.H - gap * 3) / 4);
		const std::array<OaPixelRect, 4> plotRects{{
			{plotX, margin, plotWidth, plotHeight},
			{plotX, margin + (plotHeight + gap), plotWidth, plotHeight},
			{plotX, margin + (plotHeight + gap) * 2, plotWidth, plotHeight},
			{plotX, margin + (plotHeight + gap) * 3, plotWidth, plotHeight},
		}};

		InUi.Rect(simulation, {0.055F, 0.055F, 0.055F, 1.0F});
		InUi.RectOutline(simulation, {1.0F, 1.0F, 1.0F, 0.10F}, 1);
		const OaF32 groundY = static_cast<OaF32>(simulation.Y)
			+ static_cast<OaF32>(simulation.H) * 0.70F;
		InUi.Line(
			{static_cast<OaF32>(simulation.X + 24), groundY + 20.0F},
			{static_cast<OaF32>(simulation.X + simulation.W - 24), groundY + 20.0F},
			{0.55F, 0.55F, 0.55F, 1.0F}, 2.0F);

		const OaF32 travel = static_cast<OaF32>(simulation.W) * 0.34F;
		const OaF32 cartX = static_cast<OaF32>(simulation.X)
			+ static_cast<OaF32>(simulation.W) * 0.5F
			+ std::clamp(Snapshot_.CartPosition / 2.4F, -1.0F, 1.0F) * travel;
		const OaF32 cartY = groundY;
		const OaI32 cartWidth = std::max<OaI32>(52, simulation.W / 10);
		const OaI32 cartHeight = std::max<OaI32>(24, simulation.H / 18);
		const OaPixelRect cart{
			static_cast<OaI32>(cartX) - cartWidth / 2,
			static_cast<OaI32>(cartY) - cartHeight / 2,
			cartWidth, cartHeight};
		const OaColor cartColor = Session_ && Session_->IsDone()
			? OaColor::Success() : OaColor::Accent();
		InUi.Rect(cart, cartColor.WithAlpha(0.88F));
		InUi.RectOutline(cart, {1.0F, 1.0F, 1.0F, 0.55F}, 2);
		const VlmVec2 pivot{cartX, static_cast<OaF32>(cart.Y)};
		const OaF32 poleLength = std::min<OaF32>(
			220.0F, static_cast<OaF32>(simulation.H) * 0.38F);
		const VlmVec2 tip{
			pivot.X + std::sin(Snapshot_.PoleAngle) * poleLength,
			pivot.Y - std::cos(Snapshot_.PoleAngle) * poleLength};
		InUi.Line(pivot, tip, {0.95F, 0.95F, 0.95F, 1.0F}, 8.0F);
		InUi.Rect({static_cast<OaI32>(pivot.X) - 6,
			static_cast<OaI32>(pivot.Y) - 6, 12, 12},
			{0.08F, 0.08F, 0.08F, 1.0F});

		if (!Session_) return;
		const auto& metrics = Session_->Metrics();
		Plot_(InUi, "evaluation-return",
			plotRects[0],
			metrics.EvaluationReturnHistory, OaColor::Success(), true,
			plotHeaderHeight, uiScale);
		Plot_(InUi, "ppo-loss",
			plotRects[1],
			metrics.LossHistory, OaColor::Accent(), false,
			plotHeaderHeight, uiScale);
		Plot_(InUi, "ppo-entropy",
			plotRects[2],
			metrics.EntropyHistory, OaColor::Warning(), false,
			plotHeaderHeight, uiScale);
		Plot_(InUi, "update-ms",
			plotRects[3],
			UpdateMs_, {0.70F, 0.46F, 0.96F, 1.0F}, false,
			plotHeaderHeight, uiScale);
		UpdateAndDrawLabels_(
			InUi, InTextAtlas, width, height,
			plotRects, simulation, uiScale);

		const OaF32 progress = static_cast<OaF32>(metrics.Rollout)
			/ static_cast<OaF32>(Session_->Config().Rollouts);
		InUi.Rect({simulation.X + 20, simulation.Y + simulation.H - 18,
			simulation.W - 40, 6}, {0.18F, 0.18F, 0.18F, 1.0F});
		InUi.Rect({simulation.X + 20, simulation.Y + simulation.H - 18,
			static_cast<OaI32>(static_cast<OaF32>(simulation.W - 40)
				* std::clamp(progress, 0.0F, 1.0F)), 6}, cartColor);
		if (Session_->Control().State() == OaTrainingState::Paused) {
			InUi.Rect({simulation.X + 10, simulation.Y + 10, 6, 38},
				OaColor::Warning());
		}
	}

	void MarkConsumed(
		const OaVkTimelineSemaphore& InSemaphore,
		OaU64 InValue) override {
		if (ActiveLabelSlot_ >= 0) {
			LabelSlots_[static_cast<OaU32>(ActiveLabelSlot_)].MarkConsumed(
				InSemaphore, InValue);
		}
	}

	OaStatus Close() override {
		for (auto& labels : LabelSlots_) labels.Destroy();
		ActiveLabelSlot_ = -1;
		Runtime_ = nullptr;
		Session_.Reset();
		return OaStatus::Ok();
	}

private:
	static constexpr OaU64 kEvaluationSeed = 0x0e7a1ULL;
	static constexpr OaU32 kLabelSlotCount = 4;
	static constexpr OaU32 kMaxLabelGlyphs = 512;

	OaStatus ResetSession_() {
		auto created = OaTutorialCartPolePpo::Create();
		if (created.IsError()) return created.GetStatus();
		Session_ = OaStdMove(*created);
		Failed_ = false;
		StepRequested_ = false;
		RePauseAfterStep_ = false;
		LastEvaluatedRollout_ = 0;
		DemoAccumMs_ = 0.0F;
		UpdateMs_.Clear();
		auto evaluation = Session_->Evaluate(kEvaluationSeed);
		if (evaluation.IsError()) return evaluation.GetStatus();
		auto snapshot = Session_->SnapshotLane(0);
		if (snapshot.IsError()) return snapshot.GetStatus();
		Snapshot_ = *snapshot;
		OA_LOG_INFO(OaLogComponent::App,
			"CartPole initial held-out return: %.2f",
			evaluation->MeanCompletedReturn);
		return OaStatus::Ok();
	}

	void Fail_(const OaStatus& InStatus) {
		Failed_ = true;
		if (Session_) (void)Session_->Control().Stop();
		OA_LOG_ERROR(OaLogComponent::App,
			"CartPole viewer failed: %s", InStatus.ToString().c_str());
	}

	static void Plot_(
		OaUi& InUi,
		OaStringView InId,
		OaPixelRect InRect,
		const OaVec<OaF32>& InValues,
		OaColor InColor,
		bool InFill,
		OaI32 InHeaderHeight,
		OaF32 InUiScale) {
		if (InRect.W <= 0 || InRect.H <= 0 || InValues.Empty()) return;
		InUi.Rect(InRect, {0.055F, 0.055F, 0.055F, 1.0F});
		InUi.Rect({
			InRect.X + static_cast<OaI32>(std::round(8.0F * InUiScale)),
			InRect.Y + static_cast<OaI32>(std::round(9.0F * InUiScale)),
			std::max<OaI32>(3, static_cast<OaI32>(std::round(3.0F * InUiScale))),
			static_cast<OaI32>(std::round(24.0F * InUiScale))}, InColor);
		const OaPixelRect plotRect{
			InRect.X,
			InRect.Y + InHeaderHeight,
			InRect.W,
			std::max<OaI32>(1, InRect.H - InHeaderHeight)};
		InUi.BeginPanel(InId, plotRect);
		InUi.PlotLine(InId, InValues.Data(),
			static_cast<OaI32>(InValues.Size()),
			{.Color = InColor, .AutoScale = true,
			 .ShowGrid = true, .Fill = InFill});
		InUi.EndPanel();
	}

	static void AppendText_(
		const OaTextAtlas& InAtlas,
		OaStringView InText,
		OaF32 InX,
		OaF32 InBaselineY,
		OaF32 InSize,
		OaColor InColor,
		OaVec<OaGlyphInstance>& InOutGlyphs) {
		const OaF32 scale = InSize / InAtlas.BaseFontSize();
		OaF32 penX = InX;
		for (const char character : InText) {
			const OaU32 codepoint = static_cast<OaU8>(character);
			const OaGlyphInfo* glyph = InAtlas.FindGlyph(
				OaFontId::Sans, codepoint);
			if (!glyph) continue;
			InOutGlyphs.PushBack({
				.AnchorX = 0.0F,
				.AnchorY = 0.0F,
				.OffsetX = penX + glyph->BearingX * scale,
				.OffsetY = InBaselineY - glyph->BearingY * scale,
				.Width = glyph->AtlasW * scale,
				.Height = glyph->AtlasH * scale,
				.AtlasX = static_cast<OaU32>(glyph->AtlasX),
				.AtlasY = static_cast<OaU32>(glyph->AtlasY),
				.AtlasW = static_cast<OaU32>(glyph->AtlasW),
				.AtlasH = static_cast<OaU32>(glyph->AtlasH),
				.Color = InColor.ToU32(),
			});
			penX += glyph->Advance * scale;
		}
	}

	void UpdateAndDrawLabels_(
		OaUi& InUi,
		const OaTextAtlas& InTextAtlas,
		OaI32 InWidth,
		OaI32 InHeight,
		const std::array<OaPixelRect, 4>& InRects,
		OaPixelRect InSimulationRect,
		OaF32 InUiScale) {
		OaI32 selected = -1;
		for (OaU32 offset = 0; offset < kLabelSlotCount; ++offset) {
			const OaU32 index = (NextLabelSlot_ + offset) % kLabelSlotCount;
			if (LabelSlots_[index].IsReady()) {
				selected = static_cast<OaI32>(index);
				break;
			}
		}

		if (selected >= 0) {
			const auto& metrics = Session_->Metrics();
			const OaF32 evaluation = metrics.EvaluationReturnHistory.Empty()
				? 0.0F : metrics.EvaluationReturnHistory.Back();
			const OaF32 updateMs = UpdateMs_.Empty() ? 0.0F : UpdateMs_.Back();
			char titles[4][96]{};
			std::snprintf(titles[0], sizeof(titles[0]),
				"Held-out return   %.2f", evaluation);
			std::snprintf(titles[1], sizeof(titles[1]),
				"PPO loss   %.5f", metrics.TotalLoss);
			std::snprintf(titles[2], sizeof(titles[2]),
				"Policy entropy   %.4f", metrics.Entropy);
			std::snprintf(titles[3], sizeof(titles[3]),
				"Update time   %.2f ms", updateMs);
			constexpr const char* descriptions[4] = {
				"Mean completed reward on a fixed evaluation seed",
				"Clipped policy + value loss - entropy bonus",
				"Action uncertainty; lower means more decisive",
				"One incremental PPO optimizer update on this device",
			};

			OaVec<OaGlyphInstance> glyphs;
			glyphs.Reserve(320);
			const OaTextAtlas& atlas = InTextAtlas;
			for (OaU32 index = 0; index < InRects.size(); ++index) {
				const OaF32 x = static_cast<OaF32>(InRects[index].X)
					+ 18.0F * InUiScale;
				const OaF32 y = static_cast<OaF32>(InRects[index].Y);
				AppendText_(atlas, titles[index], x,
					y + 17.0F * InUiScale, 13.0F * InUiScale,
					{0.96F, 0.96F, 0.96F, 1.0F}, glyphs);
				AppendText_(atlas, descriptions[index], x,
					y + 33.0F * InUiScale, 10.0F * InUiScale,
					{0.62F, 0.62F, 0.62F, 1.0F}, glyphs);
			}
			char fps[64]{};
			std::snprintf(fps, sizeof(fps), "Viewer   %.1f FPS", ViewerFps_);
			AppendText_(atlas, fps,
				static_cast<OaF32>(InSimulationRect.X) + 20.0F * InUiScale,
				static_cast<OaF32>(InSimulationRect.Y) + 28.0F * InUiScale,
				14.0F * InUiScale,
				{0.96F, 0.96F, 0.96F, 1.0F}, glyphs);
			if (glyphs.Size() <= kMaxLabelGlyphs) {
				auto& slot = LabelSlots_[static_cast<OaU32>(selected)];
				if (const OaStatus status = slot.Upload(
					OaSpan<const OaGlyphInstance>(glyphs.Data(), glyphs.Size()));
					status.IsOk()) {
					ActiveLabelSlot_ = selected;
					NextLabelSlot_ = (static_cast<OaU32>(selected) + 1U)
						% kLabelSlotCount;
				}
			}
		}

		if (ActiveLabelSlot_ >= 0) {
			const OaPixelRect screen{0, 0, InWidth, InHeight};
			InUi.Glyphs(
				LabelSlots_[static_cast<OaU32>(ActiveLabelSlot_)],
				InTextAtlas, screen, screen);
		}
	}

	OaEngine* Runtime_ = nullptr;
	OaUniquePtr<OaTutorialCartPolePpo> Session_;
	OaTutorialCartPoleSnapshot Snapshot_;
	OaVec<OaF32> UpdateMs_;
	std::array<OaGlyphBuffer, kLabelSlotCount> LabelSlots_;
	OaU32 LastEvaluatedRollout_ = 0;
	OaU32 NextLabelSlot_ = 0;
	OaI32 ActiveLabelSlot_ = -1;
	OaF32 DemoAccumMs_ = 0.0F;
	OaF32 FrameMsEma_ = 0.0F;
	OaF32 ViewerFps_ = 0.0F;
	bool StepRequested_ = false;
	bool RePauseAfterStep_ = false;
	bool ResetRequested_ = false;
	bool Failed_ = false;
};

} // namespace

int main() {
	CartPoleLiveSource source;
	OaViewer viewer({
		.Mode = OaViewerMode::Live,
		.LiveSource = &source,
		.Title = "OA · CartPole PPO",
		.Width = 1280,
		.Height = 720,
		.ShowHelp = true,
		.ShowStats = false,
		.ShowTimeline = false,
	});
	return viewer.Run().IsOk() ? 0 : 1;
}
