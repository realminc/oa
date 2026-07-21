#include <Oa/Ui/TrainingViewer.h>

#include <Oa/Core/Log.h>
#include <Oa/Ui/Text.h>
#include <Oa/Ui/Ui.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <mutex>

namespace {

constexpr OaU32 kGlyphSlotCount = 4;
constexpr OaU32 kMaxGlyphs = 4096;

const char* StateName(OaTrainingState InState) {
	switch (InState) {
		case OaTrainingState::Running: return "RUNNING";
		case OaTrainingState::Paused: return "PAUSED";
		case OaTrainingState::Stopping: return "STOPPING";
		case OaTrainingState::Completed: return "COMPLETED";
		case OaTrainingState::Failed: return "FAILED";
	}
	return "UNKNOWN";
}

OaColor StateColor(OaTrainingState InState) {
	switch (InState) {
		case OaTrainingState::Running:
		case OaTrainingState::Completed:
			return OaColor::Success();
		case OaTrainingState::Paused:
		case OaTrainingState::Stopping:
			return OaColor::Warning();
		case OaTrainingState::Failed:
			return OaColor::Error();
	}
	return OaColor::Accent();
}

void AppendText(
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
		const OaGlyphInfo* glyph = InAtlas.FindGlyph(
			OaFontId::Sans, static_cast<OaU8>(character));
		if (glyph == nullptr) continue;
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

} // namespace

struct OaTrainingViewerSource::Impl {
	struct Series {
		OaString Name;
		OaVec<OaF32> Values;
		OaColor Color;
	};

	OaTrainingSession* Session = nullptr;
	OaTrainingViewerConfig Config;
	OaEngine* Runtime = nullptr;
	OaOpt<OaTrainingSnapshot> Snapshot;
	OaOpt<OaTrainingCommandResult> LastCommand;
	OaVec<Series> SeriesList;
	std::array<OaGlyphBuffer, kGlyphSlotCount> LabelSlots;
	OaU32 NextLabelSlot = 0;
	OaI32 ActiveLabelSlot = -1;
	OaU64 LastResultSequence = 0;
	OaF32 ViewerFrameMs = 0.0F;
	mutable std::mutex PreviewMutex;
	OaOpt<OaTrainingPreviewFrame> PendingPreview;
	OaOpt<OaTrainingPreviewFrame> Preview;

	Series* FindSeries(OaStringView InName) {
		for (auto& series : SeriesList) {
			if (series.Name == InName) return &series;
		}
		return nullptr;
	}

	void AppendSeries(OaStringView InName, OaF64 InValue, OaColor InColor) {
		if (!std::isfinite(InValue)) return;
		Series* series = FindSeries(InName);
		if (series == nullptr) {
			if (SeriesList.Size() >= Config.MaxMetricPlots) return;
			SeriesList.PushBack({
				.Name = OaString(InName),
				.Values = {},
				.Color = InColor,
			});
			series = &SeriesList.Back();
		}
		series->Values.PushBack(static_cast<OaF32>(InValue));
		const OaUsize capacity = std::max<OaU32>(Config.HistoryCapacity, 2U);
		if (series->Values.Size() > capacity) {
			const OaUsize excess = series->Values.Size() - capacity;
			series->Values.Erase(
				series->Values.Begin(), series->Values.Begin() + excess);
		}
	}

	void AppendSnapshot(const OaTrainingSnapshot& InSnapshot) {
		const bool newStep = !Snapshot.HasValue()
			|| InSnapshot.Step != Snapshot->Step;
		if (newStep && InSnapshot.Step > 0) {
			AppendSeries("loss", InSnapshot.Loss, OaColor::Accent());
			if (Config.ShowGpuTiming) {
				AppendSeries("gpu_ms", InSnapshot.GpuMs,
					{0.70F, 0.46F, 0.96F, 1.0F});
			}
			if (Config.ShowWallTiming) {
				AppendSeries("wall_ms", InSnapshot.WallMs,
					OaColor::Warning());
			}
			constexpr std::array<OaColor, 5> colors{{
				OaColor::Success(),
				{0.30F, 0.72F, 0.94F, 1.0F},
				{0.95F, 0.43F, 0.36F, 1.0F},
				{0.85F, 0.67F, 0.24F, 1.0F},
				{0.55F, 0.78F, 0.42F, 1.0F},
			}};
			OaU32 colorIndex = 0;
			for (const auto& metric : InSnapshot.Metrics) {
				AppendSeries(metric.Name, metric.Value,
					colors[colorIndex++ % colors.size()]);
			}
		}
		Snapshot = InSnapshot;
	}

	void LogEnqueue(const OaResult<OaU64>& InResult, const char* InName) {
		if (InResult.IsError()) {
			OA_LOG_WARN(OaLogComponent::App,
				"OaTrainingViewer %s rejected: %s", InName,
				InResult.GetStatus().ToString().c_str());
		}
	}

	void TogglePause() {
		const OaTrainingState state = Session->State();
		if (state == OaTrainingState::Running) {
			LogEnqueue(Session->Pause(), "pause");
		} else if (state == OaTrainingState::Paused) {
			LogEnqueue(Session->Resume(), "resume");
		}
	}

	void PollResults() {
		for (const auto& result : Session->ResultsAfter(LastResultSequence)) {
			LastResultSequence = std::max(LastResultSequence, result.Sequence);
			LastCommand = result;
		}
	}

	OaI32 AcquireLabelSlot() {
		for (OaU32 offset = 0; offset < kGlyphSlotCount; ++offset) {
			const OaU32 index = (NextLabelSlot + offset) % kGlyphSlotCount;
			if (LabelSlots[index].IsReady()) return static_cast<OaI32>(index);
		}
		return -1;
	}
};

OaTrainingViewerSource::OaTrainingViewerSource(
	OaTrainingSession& InSession,
	OaTrainingViewerConfig InConfig)
	: Impl_(OaMakeUniquePtr<Impl>()) {
	Impl_->Session = &InSession;
	Impl_->Config = OaStdMove(InConfig);
	Impl_->Config.HistoryCapacity = std::max(Impl_->Config.HistoryCapacity, 2U);
	Impl_->Config.MaxMetricPlots = std::clamp(
		Impl_->Config.MaxMetricPlots, 1U, 12U);
}

OaTrainingViewerSource::~OaTrainingViewerSource() = default;

OaStatus OaTrainingViewerSource::Open(OaEngine& InEngine) {
	Impl_->Runtime = &InEngine;
	if (const auto snapshot = Impl_->Session->LatestSnapshot(); snapshot.HasValue()) {
		Impl_->AppendSnapshot(*snapshot);
	}
	return OaStatus::Ok();
}

OaStatus OaTrainingViewerSource::Init(
	OaInputSystem& InInput,
	OaFunc<void(bool)> /*InCapturePointer*/) {
	for (auto& slot : Impl_->LabelSlots) {
		auto buffer = OaGlyphBuffer::CreateHostUpload(*Impl_->Runtime, kMaxGlyphs);
		if (buffer.IsError()) {
			return buffer.GetStatus();
		}
		slot = OaStdMove(*buffer);
	}
	InInput.RegisterAction({
		.Name = "training-pause-resume",
		.Binding = {.Key = OuiKey::Space},
		.Callback = [this] { Impl_->TogglePause(); },
	});
	InInput.RegisterAction({
		.Name = "training-stop",
		.Binding = {.Key = OuiKey::S},
		.Callback = [this] {
			Impl_->LogEnqueue(Impl_->Session->Stop(), "stop");
		},
	});
	InInput.RegisterAction({
		.Name = "training-checkpoint",
		.Binding = {.Key = OuiKey::C},
		.Callback = [this] {
			Impl_->LogEnqueue(Impl_->Session->Checkpoint(), "checkpoint");
		},
	});
	InInput.RegisterAction({
		.Name = "training-evaluate",
		.Binding = {.Key = OuiKey::E},
		.Callback = [this] {
			Impl_->LogEnqueue(Impl_->Session->Evaluate(), "evaluate");
		},
	});
	OA_LOG_INFO(OaLogComponent::App,
		"OaTrainingViewer: Space=pause/resume · C=checkpoint · E=evaluate · S=stop");
	return OaStatus::Ok();
}

void OaTrainingViewerSource::Update(OaF32 InDeltaMs) {
	if (std::isfinite(InDeltaMs) && InDeltaMs > 0.0F) {
		Impl_->ViewerFrameMs = Impl_->ViewerFrameMs <= 0.0F
			? InDeltaMs
			: Impl_->ViewerFrameMs * 0.90F + InDeltaMs * 0.10F;
	}
	if (const auto snapshot = Impl_->Session->LatestSnapshot(); snapshot.HasValue()) {
		const bool changed = !Impl_->Snapshot.HasValue()
			|| snapshot->Step != Impl_->Snapshot->Step
			|| snapshot->Revision != Impl_->Snapshot->Revision
			|| snapshot->State != Impl_->Snapshot->State;
		if (changed) Impl_->AppendSnapshot(*snapshot);
	}
	{
		std::lock_guard<std::mutex> lock(Impl_->PreviewMutex);
		if (Impl_->PendingPreview.HasValue()
			&& Impl_->PendingPreview->Completion.IsComplete()) {
			Impl_->Preview = OaStdMove(Impl_->PendingPreview);
			Impl_->PendingPreview.Reset();
		}
	}
	Impl_->PollResults();
}

void OaTrainingViewerSource::Render(
	OaUi& InUi,
	const OaTextAtlas& InTextAtlas,
	OaU32 InWidth,
	OaU32 InHeight) {
	const OaI32 width = static_cast<OaI32>(InWidth);
	const OaI32 height = static_cast<OaI32>(InHeight);
	if (width < 320 || height < 240) return;
	const OaI32 margin = 20;
	const OaI32 gap = 14;
	const OaI32 headerHeight = 108;
	const OaPixelRect header{margin, margin, width - margin * 2, headerHeight};
	InUi.Rect(header, {0.045F, 0.045F, 0.045F, 1.0F});
	InUi.RectOutline(header, {1.0F, 1.0F, 1.0F, 0.10F}, 1);

	const OaTrainingSnapshot snapshot = Impl_->Snapshot.HasValue()
		? *Impl_->Snapshot : OaTrainingSnapshot{};
	const OaColor stateColor = StateColor(snapshot.State);
	InUi.Rect({header.X, header.Y, 6, header.H}, stateColor);

	OaOpt<OaTrainingPreviewFrame> preview;
	{
		std::lock_guard<std::mutex> lock(Impl_->PreviewMutex);
		preview = Impl_->Preview;
	}
	const bool showPreview = width >= 720 && Impl_->Config.ShowPreview
		&& preview.HasValue() && preview->Texture
		&& preview->Texture->IsValid();
	const OaU32 plotCount = std::min<OaU32>(
		static_cast<OaU32>(Impl_->SeriesList.Size()),
		Impl_->Config.MaxMetricPlots);
	const OaI32 previewWidth = showPreview
		? std::clamp(width * 2 / 5, 280, 640) : 0;
	const OaI32 plotAreaWidth = width - margin * 2
		- (previewWidth > 0 ? previewWidth + gap : 0);
	const OaI32 columns = plotAreaWidth >= 760 ? 2 : 1;
	const OaI32 rows = std::max<OaI32>(1,
		(static_cast<OaI32>(plotCount) + columns - 1) / columns);
	const OaI32 plotTop = header.Y + header.H + gap;
	const OaI32 availableHeight = std::max<OaI32>(80, height - plotTop - margin);
	const OaI32 plotWidth = (plotAreaWidth - gap * (columns - 1)) / columns;
	const OaI32 plotHeight = std::max<OaI32>(72,
		(availableHeight - gap * (rows - 1)) / rows);

	for (OaU32 index = 0; index < plotCount; ++index) {
		const OaI32 column = static_cast<OaI32>(index) % columns;
		const OaI32 row = static_cast<OaI32>(index) / columns;
		const OaPixelRect rect{
			margin + column * (plotWidth + gap),
			plotTop + row * (plotHeight + gap),
			plotWidth,
			plotHeight,
		};
		const auto& series = Impl_->SeriesList[index];
		InUi.Rect(rect, {0.045F, 0.045F, 0.045F, 1.0F});
		InUi.Rect({rect.X + 10, rect.Y + 10, 4, 20}, series.Color);
		const OaPixelRect graph{
			rect.X + 8, rect.Y + 38,
			std::max<OaI32>(1, rect.W - 16),
			std::max<OaI32>(1, rect.H - 46),
		};
		if (!series.Values.Empty()) {
			InUi.BeginPanel(series.Name, graph);
			InUi.PlotLine(series.Name, series.Values.Data(),
				static_cast<OaI32>(series.Values.Size()),
				{.Color = series.Color, .AutoScale = true,
				 .ShowGrid = true, .Fill = index == 0});
			InUi.EndPanel();
		}
	}
	if (showPreview) {
		const OaI32 panelX = width - margin - previewWidth;
		const OaI32 panelY = plotTop;
		const OaI32 panelW = previewWidth;
		const OaI32 panelH = availableHeight;
		const OaPixelRect panel{panelX, panelY, panelW, panelH};
		InUi.Rect(panel, {0.025F, 0.025F, 0.025F, 1.0F});
		InUi.BeginPanel("training-preview", panel);
		InUi.Image(preview->Texture->BindlessIndex(),
			preview->Texture->Width, preview->Texture->Height);
		InUi.EndPanel();
		InUi.RectOutline(panel, {1.0F, 1.0F, 1.0F, 0.10F}, 1);
	}

	const OaI32 labelSlot = Impl_->AcquireLabelSlot();
	if (labelSlot >= 0) {
		OaVec<OaGlyphInstance> glyphs;
		glyphs.Reserve(512);
		const auto& atlas = InTextAtlas;
		AppendText(atlas, Impl_->Config.Title,
			static_cast<OaF32>(header.X + 22),
			static_cast<OaF32>(header.Y + 30), 18.0F,
			{0.96F, 0.96F, 0.96F, 1.0F}, glyphs);
		char summary[256]{};
		std::snprintf(summary, sizeof(summary),
			"%s   step %lld   epoch %lld   loss %.6f   lr %.6g",
			StateName(snapshot.State),
			static_cast<long long>(snapshot.Step),
			static_cast<long long>(snapshot.Epoch),
			static_cast<double>(snapshot.Loss),
			static_cast<double>(snapshot.LearningRate));
		AppendText(atlas, summary,
			static_cast<OaF32>(header.X + 22),
			static_cast<OaF32>(header.Y + 58), 13.0F,
			stateColor, glyphs);
		char timing[192]{};
		const OaF32 fps = Impl_->ViewerFrameMs > 0.0F
			? 1000.0F / Impl_->ViewerFrameMs : 0.0F;
		std::snprintf(timing, sizeof(timing),
			"GPU %.3f ms   wall %.3f ms   viewer %.1f FPS   Space pause/resume   C checkpoint   E evaluate   S stop",
			snapshot.GpuMs, snapshot.WallMs,
			static_cast<double>(fps));
		AppendText(atlas, timing,
			static_cast<OaF32>(header.X + 22),
			static_cast<OaF32>(header.Y + 84), 11.0F,
			{0.68F, 0.68F, 0.68F, 1.0F}, glyphs);
		for (OaU32 index = 0; index < plotCount; ++index) {
			const OaI32 column = static_cast<OaI32>(index) % columns;
			const OaI32 row = static_cast<OaI32>(index) / columns;
			const OaI32 x = margin + column * (plotWidth + gap);
			const OaI32 y = plotTop + row * (plotHeight + gap);
			const auto& series = Impl_->SeriesList[index];
			char label[160]{};
			const OaF32 latest = series.Values.Empty()
				? 0.0F : series.Values.Back();
			std::snprintf(label, sizeof(label), "%s   %.6g",
				series.Name.c_str(), static_cast<double>(latest));
			AppendText(atlas, label,
				static_cast<OaF32>(x + 22), static_cast<OaF32>(y + 26),
				12.0F, {0.92F, 0.92F, 0.92F, 1.0F}, glyphs);
		}
		if (glyphs.Size() <= kMaxGlyphs) {
			auto& slot = Impl_->LabelSlots[static_cast<OaU32>(labelSlot)];
			if (slot.Upload({glyphs.Data(), glyphs.Size()}).IsOk()) {
				Impl_->ActiveLabelSlot = labelSlot;
				Impl_->NextLabelSlot = (static_cast<OaU32>(labelSlot) + 1U)
					% kGlyphSlotCount;
			}
		}
	}

	if (Impl_->ActiveLabelSlot >= 0) {
		const OaPixelRect screen{0, 0, width, height};
		InUi.Glyphs(
			Impl_->LabelSlots[static_cast<OaU32>(Impl_->ActiveLabelSlot)],
			InTextAtlas, screen, screen);
	}
}

OaStatus OaTrainingViewerSource::PublishPreview(
	OaTrainingPreviewFrame InFrame) {
	if (!InFrame.Texture || !InFrame.Texture->IsValid()) {
		return OaStatus::InvalidArgument(
			"OaTrainingViewer preview requires a valid texture");
	}
	std::lock_guard<std::mutex> lock(Impl_->PreviewMutex);
	Impl_->PendingPreview = OaStdMove(InFrame);
	return OaStatus::Ok();
}

void OaTrainingViewerSource::MarkConsumed(
	const OaVkTimelineSemaphore& InSemaphore,
	OaU64 InValue) {
	if (Impl_->ActiveLabelSlot >= 0) {
		Impl_->LabelSlots[static_cast<OaU32>(Impl_->ActiveLabelSlot)]
			.MarkConsumed(InSemaphore, InValue);
	}
}

OaStatus OaTrainingViewerSource::Close() {
	for (auto& slot : Impl_->LabelSlots) slot.Destroy();
	Impl_->ActiveLabelSlot = -1;
	Impl_->Runtime = nullptr;
	return OaStatus::Ok();
}

OaOpt<OaTrainingSnapshot> OaTrainingViewerSource::LatestSnapshot() const {
	return Impl_->Snapshot;
}

OaOpt<OaTrainingPreviewFrame> OaTrainingViewerSource::LatestPreview() const {
	std::lock_guard<std::mutex> lock(Impl_->PreviewMutex);
	return Impl_->Preview;
}

OaU32 OaTrainingViewerSource::MetricSeriesCount() const {
	return static_cast<OaU32>(Impl_->SeriesList.Size());
}

OaU32 OaTrainingViewerSource::MetricSampleCount(OaStringView InName) const {
	for (const auto& series : Impl_->SeriesList) {
		if (series.Name == InName) {
			return static_cast<OaU32>(series.Values.Size());
		}
	}
	return 0;
}
