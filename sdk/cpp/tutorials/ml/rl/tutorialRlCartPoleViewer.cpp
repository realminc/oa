#include "cartPolePpo.h"

#include <oa/core/log.h>
#include <oa/ml/trainingSession.h>
#include <oa/ui/viewer.h>

namespace {

class CartPoleLiveSource final : public oa::ViewerLiveSource {
public:
	[[nodiscard]] oa::ViewerLiveCapabilities capabilities() const noexcept override {
		return {.retainsConsumerCompletion = true};
	}
	oa::Status open(oa::Engine& inEngine) override {
		runtime_ = &inEngine;
		return resetSession_();
	}

	oa::Status init(
		oa::InputSystem& inInput,
		oa::Fn<void(bool)> /*inCapturePointer*/) override {
		for (auto& labels : labelSlots_) {
			auto buffer = oa::GlyphBuffer::createHostUpload(
				*runtime_, kMaxLabelGlyphs);
			if (buffer.isError()) {
				fail_(buffer.getStatus());
				return buffer.getStatus();
			}
			labels = oa::move(*buffer);
		}
		inInput.registerAction({
			.name = "cartpole-pause",
			.binding = {.key = oa::UiKey::Space},
			.callback = [this] {
				if (!session_) return;
				auto& control = session_->control();
				if (control.state() == oa::TrainingState::Running) {
					(void)control.pause();
				} else if (control.state() == oa::TrainingState::Paused) {
					(void)control.resume();
				}
			},
		});
		inInput.registerAction({
			.name = "cartpole-step",
			.binding = {.key = oa::UiKey::Right},
			.callback = [this] { stepRequested_ = true; },
		});
		inInput.registerAction({
			.name = "cartpole-reset",
			.binding = {.key = oa::UiKey::R},
			.callback = [this] { resetRequested_ = true; },
		});
		OaLogInfo(oa::LogComponent::App,
			"CartPole PPO: Space=pause · Right=one update · R=restart · Q/Esc=quit");
		return oa::Status::ok();
	}

	oa::Status update(oa::F32 inDeltaMs) override {
		if (!oa::isFinite(inDeltaMs) || inDeltaMs < 0.0F) {
			return oa::Status::invalidArgument(
				"CartPole viewer update requires a finite non-negative delta");
		}
		if (inDeltaMs > 0.0F) {
			frameMsEma_ = frameMsEma_ <= 0.0F
				? inDeltaMs
				: frameMsEma_ * 0.90F + inDeltaMs * 0.10F;
			viewerFps_ = 1000.0F / frameMsEma_;
		}
		if (resetRequested_) {
			resetRequested_ = false;
			if (const oa::Status status = resetSession_(); status.isError()) {
				fail_(status);
				return status;
			}
		}
		if (!session_) {
			return oa::Status::error(
				oa::StatusCode::FailedPrecondition,
				"CartPole viewer update requires an open session");
		}
		if (failure_.isError()) return failure_;

		const bool stepRequested = stepRequested_;
		stepRequested_ = false;
		if (stepRequested
			&& session_->control().state() == oa::TrainingState::Paused) {
			(void)session_->control().resume();
			rePauseAfterStep_ = true;
		}
		if (!session_->isDone()) {
			const oa::U64 stepBefore = session_->optimizerStep();
			const auto begin = oa::steadyNow();
			if (const oa::Status status = session_->advance(); status.isError()) {
				fail_(status);
				return status;
			}
			const oa::F32 elapsed = static_cast<oa::F32>(
				(oa::steadyNow() - begin).toMilliseconds());
			if (session_->optimizerStep() != stepBefore) updateMs_.pushBack(elapsed);
			if (rePauseAfterStep_ && session_->optimizerStep() != stepBefore) {
				(void)session_->control().pause();
				rePauseAfterStep_ = false;
			}
			const oa::U32 rollout = session_->metrics().rollout;
			if (rollout != lastEvaluatedRollout_
				&& (rollout % 5U == 0U || session_->isDone())) {
				lastEvaluatedRollout_ = rollout;
				auto evaluation = session_->evaluate(kEvaluationSeed);
				if (evaluation.isError()) {
					fail_(evaluation.getStatus());
					return evaluation.getStatus();
				}
				OaLogInfo(oa::LogComponent::App,
					"CartPole rollout {}/{}: return {:.2f} · loss {:.5f}",
					rollout, session_->config().rollouts,
					evaluation->meanCompletedReturn,
					session_->metrics().totalLoss);
			}
		} else {
			demoAccumMs_ += inDeltaMs;
			if (demoAccumMs_ >= 20.0F) {
				demoAccumMs_ = oa::fmod(demoAccumMs_, 20.0F);
				if (const oa::Status status = session_->demonstrate(); status.isError()) {
					fail_(status);
					return status;
				}
			}
		}
		auto snapshot = session_->snapshotLane(0);
		if (snapshot.isError()) {
			fail_(snapshot.getStatus());
			return snapshot.getStatus();
		}
		snapshot_ = *snapshot;
		return oa::Status::ok();
	}

	oa::Status render(
		oa::Ui& inUi,
		const oa::TextAtlas& inTextAtlas,
		oa::U32 inWidth,
		oa::U32 inHeight) override {
		if (inWidth > static_cast<oa::U32>(oa::Limits<oa::I32>::max())
			|| inHeight > static_cast<oa::U32>(oa::Limits<oa::I32>::max())) {
			return oa::Status::invalidArgument(
				"CartPole viewer extent exceeds signed UI coordinates");
		}
		const oa::I32 width = static_cast<oa::I32>(inWidth);
		const oa::I32 height = static_cast<oa::I32>(inHeight);
		if (width < 320 || height < 240) return oa::Status::ok();
		const oa::F32 uiScale = oa::clamp(
			static_cast<oa::F32>(height) / 720.0F, 1.0F, 2.0F);
		const oa::I32 plotHeaderHeight = static_cast<oa::I32>(
			oa::round(42.0F * uiScale));
		const oa::I32 margin = 20;
		const oa::I32 gap = 16;
		const oa::I32 simulationWidth = oa::max<oa::I32>(200,
			static_cast<oa::I32>(static_cast<oa::F32>(width) * 0.62F));
		const oa::PixelRect simulation{
			margin, margin,
			oa::min(simulationWidth, width - margin * 2 - 160),
			height - margin * 2};
		const oa::I32 plotX = simulation.x + simulation.w + gap;
		const oa::I32 plotWidth = width - plotX - margin;
		const oa::I32 plotHeight = oa::max<oa::I32>(48,
			(simulation.h - gap * 3) / 4);
		const oa::Array<oa::PixelRect, 4> plotRects{
			oa::PixelRect{plotX, margin, plotWidth, plotHeight},
			oa::PixelRect{plotX, margin + (plotHeight + gap), plotWidth, plotHeight},
			oa::PixelRect{plotX, margin + (plotHeight + gap) * 2, plotWidth, plotHeight},
			oa::PixelRect{plotX, margin + (plotHeight + gap) * 3, plotWidth, plotHeight},
		};

		inUi.rect(simulation, {0.055F, 0.055F, 0.055F, 1.0F});
		inUi.rectOutline(simulation, {1.0F, 1.0F, 1.0F, 0.10F}, 1);
		const oa::F32 groundY = static_cast<oa::F32>(simulation.y)
			+ static_cast<oa::F32>(simulation.h) * 0.70F;
		inUi.line(
			{static_cast<oa::F32>(simulation.x + 24), groundY + 20.0F},
			{static_cast<oa::F32>(simulation.x + simulation.w - 24), groundY + 20.0F},
			{0.55F, 0.55F, 0.55F, 1.0F}, 2.0F);

		const oa::F32 travel = static_cast<oa::F32>(simulation.w) * 0.34F;
		const oa::F32 cartX = static_cast<oa::F32>(simulation.x)
			+ static_cast<oa::F32>(simulation.w) * 0.5F
			+ oa::clamp(snapshot_.cartPosition / 2.4F, -1.0F, 1.0F) * travel;
		const oa::F32 cartY = groundY;
		const oa::I32 cartWidth = oa::max<oa::I32>(52, simulation.w / 10);
		const oa::I32 cartHeight = oa::max<oa::I32>(24, simulation.h / 18);
		const oa::PixelRect cart{
			static_cast<oa::I32>(cartX) - cartWidth / 2,
			static_cast<oa::I32>(cartY) - cartHeight / 2,
			cartWidth, cartHeight};
		const oa::Color cartColor = session_ && session_->isDone()
			? oa::Color::success() : oa::Color::accent();
		inUi.rect(cart, cartColor.withAlpha(0.88F));
		inUi.rectOutline(cart, {1.0F, 1.0F, 1.0F, 0.55F}, 2);
		const oa::vlm::Vec2 pivot{cartX, static_cast<oa::F32>(cart.y)};
		const oa::F32 poleLength = oa::min<oa::F32>(
			220.0F, static_cast<oa::F32>(simulation.h) * 0.38F);
		const oa::vlm::Vec2 tip{
			pivot.x + oa::sin(snapshot_.poleAngle) * poleLength,
			pivot.y - oa::cos(snapshot_.poleAngle) * poleLength};
		inUi.line(pivot, tip, {0.95F, 0.95F, 0.95F, 1.0F}, 8.0F);
		inUi.rect({static_cast<oa::I32>(pivot.x) - 6,
			static_cast<oa::I32>(pivot.y) - 6, 12, 12},
			{0.08F, 0.08F, 0.08F, 1.0F});

		if (!session_) {
			return oa::Status::error(
				oa::StatusCode::FailedPrecondition,
				"CartPole viewer render requires an open session");
		}
		const auto& metrics = session_->metrics();
		plot_(inUi, "evaluation-return",
			plotRects[0],
			metrics.evaluationReturnHistory, oa::Color::success(), true,
			plotHeaderHeight, uiScale);
		plot_(inUi, "ppo-loss",
			plotRects[1],
			metrics.lossHistory, oa::Color::accent(), false,
			plotHeaderHeight, uiScale);
		plot_(inUi, "ppo-entropy",
			plotRects[2],
			metrics.entropyHistory, oa::Color::warning(), false,
			plotHeaderHeight, uiScale);
		plot_(inUi, "update-ms",
			plotRects[3],
			updateMs_, {0.70F, 0.46F, 0.96F, 1.0F}, false,
			plotHeaderHeight, uiScale);
		OA_RETURN_IF_ERROR(updateAndDrawLabels_(
			inUi, inTextAtlas, width, height,
			plotRects, simulation, uiScale));

		const oa::F32 progress = static_cast<oa::F32>(metrics.rollout)
			/ static_cast<oa::F32>(session_->config().rollouts);
		inUi.rect({simulation.x + 20, simulation.y + simulation.h - 18,
			simulation.w - 40, 6}, {0.18F, 0.18F, 0.18F, 1.0F});
		inUi.rect({simulation.x + 20, simulation.y + simulation.h - 18,
			static_cast<oa::I32>(static_cast<oa::F32>(simulation.w - 40)
				* oa::clamp(progress, 0.0F, 1.0F)), 6}, cartColor);
		if (session_->control().state() == oa::TrainingState::Paused) {
			inUi.rect({simulation.x + 10, simulation.y + 10, 6, 38},
				oa::Color::warning());
		}
		return oa::Status::ok();
	}

	oa::Status markConsumed(const oa::Event& inCompletion) override {
		if (not inCompletion.isValid()) {
			return oa::Status::invalidArgument(
				"CartPole viewer consumption requires a valid completion event");
		}
		if (activeLabelSlot_ >= 0) {
			return labelSlots_[static_cast<oa::U32>(activeLabelSlot_)]
				.markConsumed(inCompletion);
		}
		return oa::Status::ok();
	}

	oa::Status close() override {
		for (auto& labels : labelSlots_) labels = {};
		activeLabelSlot_ = -1;
		runtime_ = nullptr;
		session_.reset();
		failure_ = oa::Status::ok();
		return oa::Status::ok();
	}

private:
	static constexpr oa::U64 kEvaluationSeed = 0x0e7a1ULL;
	static constexpr oa::U32 kLabelSlotCount = 4;
	static constexpr oa::U32 kMaxLabelGlyphs = 512;

	oa::Status resetSession_() {
		auto created = TutorialCartPolePpo::create(*runtime_);
		if (created.isError()) return created.getStatus();
		session_ = oa::move(*created);
		failure_ = oa::Status::ok();
		stepRequested_ = false;
		rePauseAfterStep_ = false;
		lastEvaluatedRollout_ = 0;
		demoAccumMs_ = 0.0F;
		updateMs_.clear();
		auto evaluation = session_->evaluate(kEvaluationSeed);
		if (evaluation.isError()) return evaluation.getStatus();
		auto snapshot = session_->snapshotLane(0);
		if (snapshot.isError()) return snapshot.getStatus();
		snapshot_ = *snapshot;
		OaLogInfo(oa::LogComponent::App,
			"CartPole initial held-out return: {:.2f}",
			evaluation->meanCompletedReturn);
		return oa::Status::ok();
	}

	void fail_(const oa::Status& inStatus) {
		if (failure_.isOk()) failure_ = inStatus;
		if (session_) (void)session_->control().stop();
		OaLogError(oa::LogComponent::App,
			"CartPole viewer failed: {}", inStatus.toString().cStr());
	}

	static void plot_(
		oa::Ui& inUi,
		oa::StringView inId,
		oa::PixelRect inRect,
		const oa::Vector<oa::F32>& inValues,
		oa::Color inColor,
		bool inFill,
		oa::I32 inHeaderHeight,
		oa::F32 inUiScale) {
		if (inRect.w <= 0 || inRect.h <= 0 || inValues.empty()) return;
		inUi.rect(inRect, {0.055F, 0.055F, 0.055F, 1.0F});
		inUi.rect({
			inRect.x + static_cast<oa::I32>(oa::round(8.0F * inUiScale)),
			inRect.y + static_cast<oa::I32>(oa::round(9.0F * inUiScale)),
			oa::max<oa::I32>(3, static_cast<oa::I32>(oa::round(3.0F * inUiScale))),
			static_cast<oa::I32>(oa::round(24.0F * inUiScale))}, inColor);
		const oa::PixelRect plotRect{
			inRect.x,
			inRect.y + inHeaderHeight,
			inRect.w,
			oa::max<oa::I32>(1, inRect.h - inHeaderHeight)};
		inUi.beginPanel(inId, plotRect);
		inUi.plotLine(inId, inValues.data(),
			static_cast<oa::I32>(inValues.size()),
			{.color = inColor, .autoScale = true,
			 .showGrid = true, .fill = inFill});
		inUi.endPanel();
	}

	static void appendText_(
		const oa::TextAtlas& inAtlas,
		oa::StringView inText,
		oa::F32 inX,
		oa::F32 inBaselineY,
		oa::F32 inSize,
		oa::Color inColor,
		oa::Vector<oa::GlyphInstance>& inOutGlyphs) {
		oa::F32 penX = inX;
		for (const char character : inText) {
			const oa::U32 codepoint = static_cast<oa::U8>(character);
			const oa::GlyphInfo* glyph = inAtlas.findGlyph(
				oa::FontId::Sans, codepoint, inSize);
			if (!glyph) continue;
			const oa::F32 scale = inSize / glyph->rasterSize;
			inOutGlyphs.pushBack({
				.anchorX = 0.0F,
				.anchorY = 0.0F,
				.offsetX = penX + glyph->bearingX * scale,
				.offsetY = inBaselineY - glyph->bearingY * scale,
				.width = glyph->atlasW * scale,
				.height = glyph->atlasH * scale,
				.atlasX = static_cast<oa::U32>(glyph->atlasX),
				.atlasY = static_cast<oa::U32>(glyph->atlasY),
				.atlasW = static_cast<oa::U32>(glyph->atlasW),
				.atlasH = static_cast<oa::U32>(glyph->atlasH),
				.color = inColor.toU32(),
			});
			penX += glyph->advance * scale;
		}
	}

	oa::Status updateAndDrawLabels_(
		oa::Ui& inUi,
		const oa::TextAtlas& inTextAtlas,
		oa::I32 inWidth,
		oa::I32 inHeight,
		const oa::Array<oa::PixelRect, 4>& inRects,
		oa::PixelRect inSimulationRect,
		oa::F32 inUiScale) {
		oa::I32 selected = -1;
		for (oa::U32 offset = 0; offset < kLabelSlotCount; ++offset) {
			const oa::U32 index = (nextLabelSlot_ + offset) % kLabelSlotCount;
			if (labelSlots_[index].isReady()) {
				selected = static_cast<oa::I32>(index);
				break;
			}
		}

		if (selected >= 0) {
			const auto& metrics = session_->metrics();
			const oa::F32 evaluation = metrics.evaluationReturnHistory.empty()
				? 0.0F : metrics.evaluationReturnHistory.back();
			const oa::F32 updateMs = updateMs_.empty() ? 0.0F : updateMs_.back();
			const oa::Array<oa::String, 4> titles{
				oa::format("held-out return   {:.2f}", evaluation),
				oa::format("PPO loss   {:.5f}", metrics.totalLoss),
				oa::format("Policy entropy   {:.4f}", metrics.entropy),
				oa::format("Update time   {:.2f} ms", updateMs),
			};
			constexpr const char* descriptions[4] = {
				"Mean completed reward on a fixed evaluation seed",
				"clipped policy + value loss - entropy bonus",
				"action uncertainty; lower means more decisive",
				"One incremental PPO optimizer update on this device",
			};

			oa::Vector<oa::GlyphInstance> glyphs;
			glyphs.reserve(320);
			const oa::TextAtlas& atlas = inTextAtlas;
			for (oa::U32 index = 0; index < inRects.size(); ++index) {
				const oa::F32 x = static_cast<oa::F32>(inRects[index].x)
					+ 18.0F * inUiScale;
				const oa::F32 y = static_cast<oa::F32>(inRects[index].y);
				appendText_(atlas, titles[index], x,
					y + 17.0F * inUiScale, 13.0F * inUiScale,
					{0.96F, 0.96F, 0.96F, 1.0F}, glyphs);
				appendText_(atlas, descriptions[index], x,
					y + 33.0F * inUiScale, 10.0F * inUiScale,
					{0.62F, 0.62F, 0.62F, 1.0F}, glyphs);
			}
			const oa::String fps = oa::format("Viewer   {:.1f} FPS", viewerFps_);
			appendText_(atlas, fps,
				static_cast<oa::F32>(inSimulationRect.x) + 20.0F * inUiScale,
				static_cast<oa::F32>(inSimulationRect.y) + 28.0F * inUiScale,
				14.0F * inUiScale,
				{0.96F, 0.96F, 0.96F, 1.0F}, glyphs);
			if (glyphs.size() > kMaxLabelGlyphs) {
				return oa::Status::error(
					oa::StatusCode::ResourceExhausted,
					"CartPole viewer label batch exceeds its fixed capacity");
			}
			{
				auto& slot = labelSlots_[static_cast<oa::U32>(selected)];
				OA_RETURN_IF_ERROR(slot.upload(
					oa::Span<const oa::GlyphInstance>(glyphs.data(), glyphs.size())));
				activeLabelSlot_ = selected;
				nextLabelSlot_ = (static_cast<oa::U32>(selected) + 1U)
					% kLabelSlotCount;
			}
		}

		if (activeLabelSlot_ >= 0) {
			const oa::PixelRect screen{0, 0, inWidth, inHeight};
			inUi.glyphs(
				labelSlots_[static_cast<oa::U32>(activeLabelSlot_)],
				inTextAtlas, screen, screen);
		}
		return oa::Status::ok();
	}

	oa::Engine* runtime_ = nullptr;
	oa::UniquePtr<TutorialCartPolePpo> session_;
	TutorialCartPoleSnapshot snapshot_;
	oa::Vector<oa::F32> updateMs_;
	oa::Array<oa::GlyphBuffer, kLabelSlotCount> labelSlots_;
	oa::U32 lastEvaluatedRollout_ = 0;
	oa::U32 nextLabelSlot_ = 0;
	oa::I32 activeLabelSlot_ = -1;
	oa::F32 demoAccumMs_ = 0.0F;
	oa::F32 frameMsEma_ = 0.0F;
	oa::F32 viewerFps_ = 0.0F;
	bool stepRequested_ = false;
	bool rePauseAfterStep_ = false;
	bool resetRequested_ = false;
	oa::Status failure_ = oa::Status::ok();
};

} // namespace

int main() {
	CartPoleLiveSource source;
	oa::Viewer viewer({
		.mode = oa::ViewerMode::Live,
		.liveSource = &source,
		.title = "OA · CartPole PPO",
		.width = 1280,
		.height = 720,
		.showHelp = true,
		.showStats = false,
		.showTimeline = false,
	});
	return viewer.run().isOk() ? 0 : 1;
}
