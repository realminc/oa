#include <oa/ui/trainingViewer.h>

#include <oa/core/log.h>
#include <oa/core/std/algo.h>
#include <oa/core/std/array.h>
#include <oa/core/std/limits.h>
#include <oa/core/std/scalarMath.h>
#include <oa/core/std/sync.h>
#include <oa/ui/text.h>
#include <oa/ui/ui.h>

#include <stdio.h>

namespace {

const char* stateName(oa::TrainingState inState) {
	switch (inState) {
		case oa::TrainingState::Running: return "RUNNING";
		case oa::TrainingState::Paused: return "PAUSED";
		case oa::TrainingState::Stopping: return "STOPPING";
		case oa::TrainingState::Completed: return "COMPLETED";
		case oa::TrainingState::Failed: return "FAILED";
	}
	return "UNKNOWN";
}

oa::Color stateColor(oa::TrainingState inState) {
	switch (inState) {
		case oa::TrainingState::Running:
		case oa::TrainingState::Completed:
			return oa::Color::success();
		case oa::TrainingState::Paused:
		case oa::TrainingState::Stopping:
			return oa::Color::warning();
		case oa::TrainingState::Failed:
			return oa::Color::error();
	}
	return oa::Color::accent();
}

} // namespace

struct oa::TrainingViewerSource::Impl {
	struct Series {
		oa::String name;
		oa::Vector<oa::F32> values;
		oa::Color color;
		bool expanded = true;
	};

	oa::TrainingSession* session = nullptr;
	oa::TrainingViewerConfig config;
	oa::Optional<oa::TrainingSnapshot> snapshot;
	oa::Optional<oa::TrainingCommandResult> lastCommand;
	oa::Vector<Series> seriesList;
	oa::U64 lastResultSequence = 0;
	oa::F32 viewerFrameMs = 0.0F;
	// 0=responsive, 1/2=explicit metric-card column count.
	oa::I32 metricColumns = 0;
	oa::I32 selectedMetric = 0;
	oa::UiTabBarState compactTabs{.selected = 0};
	oa::F32 previewSplitRatio = 0.60F;
	mutable oa::Mutex previewMutex;
	oa::Optional<oa::TrainingPreviewFrame> pendingPreview;
	oa::Optional<oa::TrainingPreviewFrame> preview;

	Series* findSeries(oa::StringView inName) {
		for (auto& series : seriesList) {
			if (series.name == inName) return &series;
		}
		return nullptr;
	}

	void appendSeries(oa::StringView inName, oa::F64 inValue, oa::Color inColor) {
		if (!oa::isFinite(inValue)) return;
		Series* series = findSeries(inName);
		if (series == nullptr) {
			if (seriesList.size() >= config.maxMetricPlots) return;
			seriesList.pushBack({
				.name = oa::String(inName),
				.values = {},
				.color = inColor,
			});
			series = &seriesList.back();
		}
		series->values.pushBack(static_cast<oa::F32>(inValue));
		const oa::Usize capacity = oa::max<oa::U32>(config.historyCapacity, 2U);
		if (series->values.size() > capacity) {
			const oa::Usize excess = series->values.size() - capacity;
			series->values.erase(
				series->values.begin(), series->values.begin() + excess);
		}
	}

	void appendSnapshot(const oa::TrainingSnapshot& inSnapshot) {
		const bool newStep = !snapshot.hasValue()
			|| inSnapshot.step != snapshot->step;
		if (newStep && inSnapshot.step > 0) {
			appendSeries("loss", inSnapshot.loss, oa::Color::accent());
			if (config.showGpuTiming) {
				appendSeries("gpu_ms", inSnapshot.gpuMs,
					{0.70F, 0.46F, 0.96F, 1.0F});
			}
			if (config.showWallTiming) {
				appendSeries("wall_ms", inSnapshot.wallMs,
					oa::Color::warning());
			}
			constexpr oa::Array<oa::Color, 5> colors{
				oa::Color::success(),
				oa::Color{0.30F, 0.72F, 0.94F, 1.0F},
				oa::Color{0.95F, 0.43F, 0.36F, 1.0F},
				oa::Color{0.85F, 0.67F, 0.24F, 1.0F},
				oa::Color{0.55F, 0.78F, 0.42F, 1.0F},
			};
			oa::U32 colorIndex = 0;
			for (const auto& metric : inSnapshot.metrics) {
				appendSeries(metric.name, metric.value,
					colors[colorIndex++ % colors.size()]);
			}
		}
		snapshot = inSnapshot;
	}

	void logEnqueue(const oa::Result<oa::U64>& inResult, const char* inName) {
		if (inResult.isError()) {
			OaLogWarn(oa::LogComponent::Ui,
				"OaTrainingViewer {} rejected: {}", inName,
				inResult.getStatus().toString().cStr());
		}
	}

	void togglePause() {
		const oa::TrainingState state = session->state();
		if (state == oa::TrainingState::Running) {
			logEnqueue(session->pause(), "pause");
		} else if (state == oa::TrainingState::Paused) {
			logEnqueue(session->resume(), "resume");
		}
	}

	void pollResults() {
		for (const auto& result : session->resultsAfter(lastResultSequence)) {
			lastResultSequence = oa::max(lastResultSequence, result.sequence);
			lastCommand = result;
		}
	}

};

oa::TrainingViewerSource::TrainingViewerSource(
	oa::TrainingSession& inSession,
	oa::TrainingViewerConfig inConfig)
	: impl_(oa::makeUnique<Impl>()) {
	impl_->session = &inSession;
	impl_->config = oa::move(inConfig);
	impl_->config.historyCapacity = oa::max(impl_->config.historyCapacity, 2U);
	impl_->config.maxMetricPlots = oa::clamp(
		impl_->config.maxMetricPlots, 1U, 12U);
}

oa::TrainingViewerSource::~TrainingViewerSource() = default;

oa::Status oa::TrainingViewerSource::open(oa::Engine& inEngine) {
	(void)inEngine;
	if (const auto snapshot = impl_->session->latestSnapshot(); snapshot.hasValue()) {
		impl_->appendSnapshot(*snapshot);
	}
	return oa::Status::ok();
}

oa::Status oa::TrainingViewerSource::init(
	oa::InputSystem& inInput,
	oa::Fn<void(bool)> /*inCapturePointer*/) {
	inInput.registerAction({
		.name = "training-pause-resume",
		.binding = {.key = oa::UiKey::Space},
		.callback = [this] { impl_->togglePause(); },
	});
	inInput.registerAction({
		.name = "training-stop",
		.binding = {.key = oa::UiKey::S},
		.callback = [this] {
			impl_->logEnqueue(impl_->session->stop(), "stop");
		},
	});
	inInput.registerAction({
		.name = "training-checkpoint",
		.binding = {.key = oa::UiKey::C},
		.callback = [this] {
			impl_->logEnqueue(impl_->session->checkpoint(), "checkpoint");
		},
	});
	inInput.registerAction({
		.name = "training-evaluate",
		.binding = {.key = oa::UiKey::E},
		.callback = [this] {
			impl_->logEnqueue(impl_->session->evaluate(), "evaluate");
		},
	});
	OaLogInfo(oa::LogComponent::Ui,
		"OaTrainingViewer: Space=pause/resume · C=checkpoint · E=evaluate · S=stop");
	return oa::Status::ok();
}

oa::Status oa::TrainingViewerSource::update(oa::F32 inDeltaMs) {
	if (!oa::isFinite(inDeltaMs) || inDeltaMs < 0.0F) {
		return oa::Status::invalidArgument(
			"oa::TrainingViewerSource::update requires a finite non-negative delta");
	}
	if (inDeltaMs > 0.0F) {
		impl_->viewerFrameMs = impl_->viewerFrameMs <= 0.0F
			? inDeltaMs
			: impl_->viewerFrameMs * 0.90F + inDeltaMs * 0.10F;
	}
	if (const auto snapshot = impl_->session->latestSnapshot(); snapshot.hasValue()) {
		const bool changed = !impl_->snapshot.hasValue()
			|| snapshot->step != impl_->snapshot->step
			|| snapshot->revision != impl_->snapshot->revision
			|| snapshot->state != impl_->snapshot->state;
		if (changed) impl_->appendSnapshot(*snapshot);
	}
	{
		oa::ScopedLock<oa::Mutex> lock(impl_->previewMutex);
		if (impl_->pendingPreview.hasValue()
			&& impl_->pendingPreview->completion.isComplete()) {
			impl_->preview = oa::move(impl_->pendingPreview);
			impl_->pendingPreview.reset();
		}
	}
	impl_->pollResults();
	return oa::Status::ok();
}

oa::Status oa::TrainingViewerSource::render(
	oa::Ui& inUi,
	const oa::TextAtlas& /*inTextAtlas*/,
	oa::U32 inWidth,
	oa::U32 inHeight) {
	if (inWidth > static_cast<oa::U32>(oa::Limits<oa::I32>::max())
		|| inHeight > static_cast<oa::U32>(oa::Limits<oa::I32>::max())) {
		return oa::Status::invalidArgument(
			"oa::TrainingViewerSource::render extent exceeds signed UI coordinates");
	}
	const oa::I32 width = static_cast<oa::I32>(inWidth);
	const oa::I32 height = static_cast<oa::I32>(inHeight);
	const oa::F32 contentScale = inUi.contentScale();
	const auto px = [contentScale](oa::I32 inLogical) {
		return oa::max<oa::I32>(1, static_cast<oa::I32>(oa::lround(
			static_cast<oa::F32>(inLogical) * contentScale)));
	};
	if (width < px(320) || height < px(240)) return oa::Status::ok();
	const oa::I32 margin = px(20);
	const oa::I32 gap = px(14);
	const oa::I32 headerHeight = px(108);
	const oa::PixelRect header{margin, margin, width - margin * 2, headerHeight};
	inUi.rect(header, {0.045F, 0.045F, 0.045F, 1.0F});
	inUi.rectOutline(header, {1.0F, 1.0F, 1.0F, 0.10F}, 1);

	const oa::TrainingSnapshot snapshot = impl_->snapshot.hasValue()
		? *impl_->snapshot : oa::TrainingSnapshot{};
	const oa::Color statusColor = stateColor(snapshot.state);
	inUi.rect({header.x, header.y, px(6), header.h}, statusColor);
	const auto drawLabel = [&inUi, contentScale](
		oa::StringView inId,
		oa::PixelRect inRect,
		oa::StringView inText,
		oa::F32 inSize,
		oa::Color inColor) {
		oa::UiStyle style = inUi.currentStyle();
		style.fontSize = inSize * contentScale;
		style.text = inColor;
		oa::UiLayout layout;
		layout.padding = oa::UiEdge{};
		inUi.pushStyle(style);
		inUi.beginPanel(inId, inRect, layout);
		inUi.label(inText);
		inUi.endPanel();
		inUi.popStyle();
	};
	drawLabel("training-title", {
		header.x + px(22), header.y + px(10),
		oa::max(1, header.w - px(240)), px(24)},
		impl_->config.title, 18.0F, {0.96F, 0.96F, 0.96F, 1.0F});
	static const oa::Array<oa::StringView, 3> metricLayouts{
		"Auto", "1 column", "2 columns",
	};
	oa::UiLayout metricLayout;
	metricLayout.padding = oa::UiEdge{};
	metricLayout.gap = 0.0F;
	inUi.beginPanel("training-metric-layout", {
		header.x + oa::max(0, header.w - px(194)),
		header.y + px(8), px(174), px(30)},
		metricLayout);
	(void)inUi.dropdown(
		"metrics",
		oa::Span<const oa::StringView>(metricLayouts.data(), metricLayouts.size()),
		impl_->metricColumns,
		{.maxVisibleItems = 3});
	inUi.tooltip("Choose responsive, one-column, or two-column metric cards.");
	inUi.endPanel();
	char summary[256]{};
	::snprintf(summary, sizeof(summary),
		"%s   step %lld   epoch %lld   loss %.6f   lr %.6g",
		stateName(snapshot.state),
		static_cast<long long>(snapshot.step),
		static_cast<long long>(snapshot.epoch),
		static_cast<double>(snapshot.loss),
		static_cast<double>(snapshot.learningRate));
	drawLabel("training-summary", {header.x + px(22), header.y + px(40),
		header.w - px(44), px(20)},
		summary, 13.0F, statusColor);
	char timing[192]{};
	const oa::F32 fps = impl_->viewerFrameMs > 0.0F
		? 1000.0F / impl_->viewerFrameMs : 0.0F;
	::snprintf(timing, sizeof(timing),
		"GPU %.3f ms   wall %.3f ms   viewer %.1f FPS   Space pause/resume   C checkpoint   E evaluate   S stop",
		snapshot.gpuMs, snapshot.wallMs,
		static_cast<double>(fps));
	drawLabel("training-timing", {header.x + px(22), header.y + px(70),
		header.w - px(44), px(18)},
		timing, 11.0F, {0.68F, 0.68F, 0.68F, 1.0F});

	oa::Optional<oa::TrainingPreviewFrame> preview;
	{
		oa::ScopedLock<oa::Mutex> lock(impl_->previewMutex);
		preview = impl_->preview;
	}
	const bool previewAvailable = impl_->config.showPreview
		&& preview.hasValue() && preview->texture
		&& preview->texture->isValid();
	const bool showSplitPreview = width >= px(720) && previewAvailable;
	const oa::U32 plotCount = oa::min<oa::U32>(
		static_cast<oa::U32>(impl_->seriesList.size()),
		impl_->config.maxMetricPlots);
	oa::I32 plotTop = header.y + header.h + gap;
	const bool compactTabs = width < px(720) && previewAvailable;
	if (compactTabs) {
		static const oa::Array<oa::UiTabItem, 2> tabs{
			oa::UiTabItem{.id = "metrics", .label = "metrics", .closable = false},
			oa::UiTabItem{.id = "preview", .label = "preview", .closable = false},
		};
		(void)inUi.tabBar(
			"training-compact-tabs",
			{margin, plotTop, width - margin * 2, px(28)},
			oa::Span<const oa::UiTabItem>(tabs.data(), tabs.size()),
			impl_->compactTabs,
			{.minimumTabWidth = px(96), .maximumTabWidth = px(180)});
		plotTop += px(28) + gap;
	}
	const oa::I32 availableHeight = oa::max<oa::I32>(
		px(80), height - plotTop - margin);
	oa::PixelRect plotViewport{
		margin, plotTop, width - margin * 2, availableHeight};
	oa::PixelRect previewPanel;
	if (compactTabs && impl_->compactTabs.selected == 1) {
		inUi.rect(plotViewport, {0.025F, 0.025F, 0.025F, 1.0F});
		inUi.beginPanel("training-preview", plotViewport);
		inUi.image(*preview->texture);
		inUi.endPanel();
		inUi.rectOutline(plotViewport, {1.0F, 1.0F, 1.0F, 0.10F}, 1);
		return oa::Status::ok();
	}
	if (showSplitPreview) {
		const oa::UiSplitRegion content = inUi.splitPane(
			"training-content-split",
			plotViewport,
			impl_->previewSplitRatio,
			{
				.direction = oa::UiDirection::Row,
				.handleSize = px(10),
				.minimumFirst = px(300),
				.minimumSecond = px(280),
			});
		plotViewport = content.first;
		previewPanel = content.second;
		inUi.tooltip("Drag to resize metrics and preview. Use Left/Right for precise adjustment.");
	}
	const oa::I32 plotAreaWidth = plotViewport.w;
	const oa::I32 columns = impl_->metricColumns == 0
		? (plotAreaWidth >= px(760) ? 2 : 1)
		: impl_->metricColumns;
	const oa::I32 rows = oa::max<oa::I32>(1,
		(static_cast<oa::I32>(plotCount) + columns - 1) / columns);
	const oa::I32 plotHeight = rows <= 1
		? availableHeight
		: oa::clamp(availableHeight * 3 / 7, px(128), px(240));
	const oa::I32 contentHeight = plotCount == 0U
		? availableHeight
		: rows * plotHeight + gap * (rows - 1);
	oa::UiLayout scrollLayout;
	scrollLayout.padding = oa::UiEdge{};
	scrollLayout.gap = 0.0F;
	const oa::UiScrollRegion plots = inUi.beginScrollPanel(
		"training-metrics",
		plotViewport,
		contentHeight,
		scrollLayout);
	const oa::I32 plotWidth = oa::max<oa::I32>(1,
		(plots.content.w - gap * (columns - 1)) / columns);
	const oa::UiVirtualRange visibleRows = inUi.virtualRows(
		rows, plotHeight, gap);
	for (oa::I32 row = visibleRows.first; row < visibleRows.onePastLast; ++row) {
		for (oa::I32 column = 0; column < columns; ++column) {
			const oa::U32 index = static_cast<oa::U32>(row * columns + column);
			if (index >= plotCount) break;
			const oa::PixelRect rect{
				plots.content.x + column * (plotWidth + gap),
				plots.content.y + row * (plotHeight + gap),
				plotWidth,
				plotHeight,
			};
			auto& series = impl_->seriesList[index];
			inUi.rect(rect, {0.045F, 0.045F, 0.045F, 1.0F});
			const oa::UiTreeRowResult tree = inUi.treeRow(
				series.name,
				{rect.x + px(8), rect.y + px(6),
					oa::max(1, rect.w - px(16)), px(26)},
				series.name,
				{
					.hasChildren = true,
					.open = series.expanded,
					.selected = impl_->selectedMetric == static_cast<oa::I32>(index),
				});
			if (tree.activated) {
				impl_->selectedMetric = static_cast<oa::I32>(index);
			}
			if (tree.openChanged) series.expanded = tree.open;

			char latestText[160]{};
			const oa::F32 latest = series.values.empty()
				? 0.0F : series.values.back();
			::snprintf(latestText, sizeof(latestText), "%.6g · %zu samples",
				static_cast<double>(latest),
				static_cast<size_t>(series.values.size()));
			(void)inUi.propertyRow(
				series.name,
				{rect.x + px(8), rect.y + px(34),
					oa::max(1, rect.w - px(16)), px(22)},
				"latest",
				latestText,
				{.alternate = (index & 1U) != 0U});
			const oa::PixelRect graph{
				rect.x + px(8), rect.y + px(62),
				oa::max<oa::I32>(1, rect.w - px(16)),
				oa::max<oa::I32>(1, rect.h - px(70)),
			};
			if (series.expanded && !series.values.empty()) {
				inUi.beginPanel(series.name, graph);
				inUi.plotLine(series.name, series.values.data(),
					static_cast<oa::I32>(series.values.size()),
					{.color = series.color, .autoScale = true,
					 .showGrid = true, .fill = index == 0});
				inUi.endPanel();
			}
		}
	}
	inUi.endScrollPanel();
	if (showSplitPreview) {
		const oa::PixelRect panel = previewPanel;
		inUi.rect(panel, {0.025F, 0.025F, 0.025F, 1.0F});
		inUi.beginPanel("training-preview", panel);
		inUi.image(*preview->texture);
		inUi.endPanel();
		inUi.rectOutline(panel, {1.0F, 1.0F, 1.0F, 0.10F}, 1);
	}
	return oa::Status::ok();
}

oa::Status oa::TrainingViewerSource::publishPreview(
	oa::TrainingPreviewFrame inFrame) {
	if (!inFrame.texture || !inFrame.texture->isValid()) {
		return oa::Status::invalidArgument(
			"OaTrainingViewer preview requires a valid texture");
	}
	oa::ScopedLock<oa::Mutex> lock(impl_->previewMutex);
	impl_->pendingPreview = oa::move(inFrame);
	return oa::Status::ok();
}

oa::Status oa::TrainingViewerSource::close() {
	return oa::Status::ok();
}

oa::Optional<oa::TrainingSnapshot> oa::TrainingViewerSource::latestSnapshot() const {
	return impl_->snapshot;
}

oa::Optional<oa::TrainingPreviewFrame> oa::TrainingViewerSource::latestPreview() const {
	oa::ScopedLock<oa::Mutex> lock(impl_->previewMutex);
	return impl_->preview;
}

oa::U32 oa::TrainingViewerSource::metricSeriesCount() const {
	return static_cast<oa::U32>(impl_->seriesList.size());
}

oa::U32 oa::TrainingViewerSource::metricSampleCount(oa::StringView inName) const {
	for (const auto& series : impl_->seriesList) {
		if (series.name == inName) {
			return static_cast<oa::U32>(series.values.size());
		}
	}
	return 0;
}
