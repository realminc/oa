// ═══════════════════════════════════════════════════════════════════════════
// OA Tutorial: Image-grid Classification — Fashion-MNIST 5×5 grid
// level 1 API — oa::Module + oa::plot::Figure / Axes (FinalGlue step 3e)
// ═══════════════════════════════════════════════════════════════════════════
//
// A compact consumer of the OA plot and viewer contracts:
// Fashion-MNIST classifier predictions rendered as a 5×5 image grid using
// oa::plot::Figure + Axes::imshow / title / caption. Parallel to the TF keras
// classification tutorial — same image grid, OA C++ syntax, GPU all the way
// through the recorder.
//
//   Same renderer body, two sinks (architecture/oaArchitecture.md §10):
//     ./TutorialImageGridClassify                      → show (window)
//     ./TutorialImageGridClassify --save grid.png      → saveTo (batch)
//
// usage:
//   ./Tutorial/Plot/TutorialImageGridClassify [data_dir] [--save path.png]
//
// Default data dir: $OA_DATA_DIR/fashionMnist
// ═══════════════════════════════════════════════════════════════════════════

#include <data/dsMnist.h>
#include <oa/ml.h>
#include <oa/ml/autograd.h>
#include <oa/ui/plot/plot.h>
#include <oa/ui/image.h>
#include <oa/ui/input.h>
#include <oa/ui/viewer.h>
#include <oa/runtime/engine.h>
#include <oa/core/log.h>
#include <oa/core/paths.h>

#include <fstream>
#include <cstring>
#include <cmath>


// ─── Fashion-MNIST class names ─────────────────────────────────────────────

static const char* kClasses[] = {
	"T-shirt", "Trouser",  "Pullover", "Dress",  "Coat",
	"Sandal",  "Shirt",    "Sneaker",  "Bag",    "Boot"
};
static constexpr oa::I32 kNumClasses = 10;

// ─── Raw IDX loader for test images (needed for texture creation) ─────────

struct MnistData {
	oa::Vec<oa::U8> images;
	oa::Vec<oa::U8> labels;
	oa::I32 count = 0;
};

static oa::U32 readBE32(std::ifstream& f) {
	oa::U8 b[4];
	f.read(reinterpret_cast<char*>(b), 4);
	return (oa::U32(b[0]) << 24) | (oa::U32(b[1]) << 16) | (oa::U32(b[2]) << 8) | oa::U32(b[3]);
}

static bool loadMnistIDX(const oa::String& inDir,
                         const oa::String& inImgFile,
                         const oa::String& inLblFile,
                         MnistData& out) {
	std::ifstream imgF((inDir + "/" + inImgFile).cStr(), std::ios::binary);
	std::ifstream lblF((inDir + "/" + inLblFile).cStr(), std::ios::binary);
	if (not imgF or not lblF) { return false; }
	if (readBE32(imgF) != 0x00000803U) { return false; }
	oa::U32 n    = readBE32(imgF);
	oa::U32 rows = readBE32(imgF);
	oa::U32 cols = readBE32(imgF);
	if (rows != 28U or cols != 28U) { return false; }
	if (readBE32(lblF) != 0x00000801U) { return false; }
	if (readBE32(lblF) != n) { return false; }
	out.count = static_cast<oa::I32>(n);
	out.images.resize(static_cast<oa::I64>(n) * 784);
	imgF.read(reinterpret_cast<char*>(out.images.data()), out.images.size());
	out.labels.resize(static_cast<oa::I64>(n));
	lblF.read(reinterpret_cast<char*>(out.labels.data()), out.labels.size());
	return true;
}


// ─── MLP classifier (same shape as TutorialMnistClassifierApi1) ───────────

class MnistClassifier : public oa::Module {
public:
	MnistClassifier() {
		auto wd = oa::FnMatrix::weightDtype();
		fc1_ = oa::makeShared<oa::Linear>(784, 128);
		fc1_->setActivation(oa::Activation::Relu);
		fc1_->parameters()[0].data = oa::FnMatrix::randKaimingUniform(oa::MatrixShape{128, 784}, wd);
		fc2_ = oa::makeShared<oa::Linear>(128, kNumClasses);
		fc2_->parameters()[0].data = oa::FnMatrix::randGlorotUniform(oa::MatrixShape{kNumClasses, 128}, wd);
		registerModule("fc1", fc1_);
		registerModule("fc2", fc2_);
	}

	oa::Matrix forward(const oa::Matrix& inX) override {
		xNorm_  = oa::FnMatrix::scale(inX, 1.0F / 255.0F);
		h1_     = fc1_->forward(xNorm_);
		logits_ = fc2_->forward(h1_);
		return logits_;
	}

	void backward(const oa::Matrix& inDLogits) {
		auto& fc1P = fc1_->parameters();
		auto& fc2P = fc2_->parameters();
		auto gbw2 = oa::FnMatrix::linearWeightBiasBwd(h1_, inDLogits);
		auto dZ1  = oa::FnMatrix::linearDataReluBwd(inDLogits, fc2P[0].data, h1_);
		auto gbw1 = oa::FnMatrix::linearWeightBiasBwd(xNorm_, dZ1);
		fc1P[0].grad() = gbw1.gradWeight;
		fc1P[1].grad() = gbw1.gradBias;
		fc2P[0].grad() = gbw2.gradWeight;
		fc2P[1].grad() = gbw2.gradBias;
		}

private:
	oa::SharedPtr<oa::Linear> fc1_, fc2_;
	oa::Matrix xNorm_, h1_, logits_;
};




// ─── Per-cell result + grid build helper ──────────────────────────────────

static constexpr oa::I32 kGridCols = 5;
static constexpr oa::I32 kGridRows = 5;
static constexpr oa::I32 kGridN    = kGridCols * kGridRows;  // 25

struct GridCell {
	oa::Texture tile;
	oa::I32     actual    = 0;
	oa::I32     predicted = 0;
	bool      correct   = false;
};

// load Fashion-MNIST, train an MLP, fill inOutCells[0..25) with prediction
// tiles + labels. Returns oa::Status on failure.
static oa::Status trainAndPredictGrid(oa::Engine& inRt,
                                    const oa::String&    inDataDir,
                                    oa::I32              inTrainSteps,
                                    oa::Vec<GridCell>&   outCells) {
	// load raw test data for texture creation (oa::DsMnist normalizes data)
	MnistData testData;
	if (not loadMnistIDX(inDataDir, "t10k-images-idx3-ubyte", "t10k-labels-idx1-ubyte", testData)) {
		return oa::Status::error("Fashion-MNIST test set not found");
	}

	oa::DsMnist trainLoader(inDataDir, "train", 64, /*shuffle=*/true);
	oa::DsMnist testLoader(inDataDir, "t10k", 100, /*shuffle=*/false);

	if (trainLoader.numSamples() == 0 || testLoader.numSamples() == 0) {
		return oa::Status::error("Fashion-MNIST not found");
	}
	OaLogInfo(oa::LogComponent::App, "Loaded %d train / %d test images",
	            trainLoader.numSamples(), testLoader.numSamples());

	auto model     = oa::makeShared<MnistClassifier>();
	auto params    = model->allParameterPtrs();
	auto optimizer = oa::makeUnique<oa::AdamW>(params, 0.001F);

	constexpr oa::I32 kBatch = 64;
	const oa::I32 kStepsPerEpoch = trainLoader.numSamples() / kBatch;

	// training loop with oa::ItTraining
	oa::CbProgressBar progressBar;
	oa::CbSummary summary;
	oa::MetricLoss lossMetric;
	progressBar.addMetric(&lossMetric);

	oa::ItTraining loop(inRt, *optimizer, oa::ItTrainingConfig{
		.totalSteps     = inTrainSteps,
		.stepsPerEpoch  = kStepsPerEpoch,
		.batchSize      = kBatch,
		.timerName      = "image_grid_step",
		.metrics        = { &lossMetric },
		.callbacks      = { &progressBar, &summary },
	});

	OaLogInfo(oa::LogComponent::App, "training %d steps ...", inTrainSteps);

	oa::Matrix batchX;
	oa::Matrix batchY;

	while (not loop.isDone()) {
		if (not trainLoader.nextBatch(batchX, batchY)) {
			trainLoader.reset();
			trainLoader.nextBatch(batchX, batchY);
		}

		auto logits     = model->forward(batchX);
		auto loss       = oa::FnLoss::crossEntropy(logits, batchY);
		auto gradLogits = oa::FnLoss::crossEntropyBwd(logits, batchY);
		model->backward(gradLogits);
		loop.next(loss);
	}
	if (not loop.finish().isOk()) {
		return oa::Status::error("training loop finish failed");
	}
	OaLogInfo(oa::LogComponent::App, "training done.");

	// Inference on the first 25 test images.
	auto xTest = oa::FnMatrix::fromBytes(
		oa::Span<const oa::U8>(testData.images.data(), static_cast<oa::I64>(kGridN) * 784),
		oa::MatrixShape{kGridN, 784});
	auto logits = model->forward(xTest);
	auto probs  = oa::FnMatrix::softmax(logits, -1);

	// Copy probs to host for argmax
	oa::I32 batch = static_cast<oa::I32>(probs.size(0));
	oa::I32 nCls  = static_cast<oa::I32>(probs.size(1));
	oa::Vec<oa::F32> host(batch * nCls);
	(void)oa::FnMatrix::copyToHost(probs, host.data(), host.size() * sizeof(oa::F32));

	// Use raw test data for texture creation
	const oa::U8* labels = testData.labels.data();

	outCells.resize(kGridN);

	for (oa::I32 i = 0; i < kGridN; ++i) {
		oa::Vec<oa::U8> rgba(static_cast<oa::I64>(28) * 28 * 4);
		oa::I32 best  = 0;
		oa::F32 bestV = host[static_cast<oa::I64>(i) * kNumClasses];
		for (oa::I32 j = 1; j < kNumClasses; ++j) {
			oa::F32 v = host[static_cast<oa::I64>(i) * kNumClasses + j];
			if (v > bestV) { bestV = v; best = j; }
		}
		outCells[i].actual    = static_cast<oa::I32>(labels[i]);
		outCells[i].predicted = best;
		outCells[i].correct   = (best == outCells[i].actual);

		const oa::U8* src = testData.images.data() + static_cast<oa::I64>(i) * 784;
		for (oa::I32 p = 0; p < 28 * 28; ++p) {
			oa::U8 g = src[p];
			rgba[static_cast<oa::I64>(p) * 4 + 0] = g;
			rgba[static_cast<oa::I64>(p) * 4 + 1] = g;
			rgba[static_cast<oa::I64>(p) * 4 + 2] = g;
			rgba[static_cast<oa::I64>(p) * 4 + 3] = 255U;
		}
		auto r = oa::FnTexture::fromPixels(inRt,
			oa::Span<const oa::U8>(rgba.data(), static_cast<oa::I64>(28) * 28 * 4), 28, 28);
		if (not r.isOk()) {
			return oa::Status::error("oa::FnTexture::fromPixels failed");
		}
		outCells[i].tile = *r;
	}

	oa::I32 correct = 0;
	for (const auto& cell : outCells) {
		if (cell.correct) { ++correct; }
	}
	OaLogInfo(oa::LogComponent::App,
	            "Prediction grid: %d / %d correct on first %d test images",
	            correct, kGridN, kGridN);
	return oa::Status::ok();
}


// ─── PopulateFigure ───────────────────────────────────────────────────────

static const oa::Color kSuccess = {0.188F, 0.820F, 0.345F, 1.0F};  // #30d158
static const oa::Color kError   = {1.000F, 0.271F, 0.227F, 1.0F};  // #ff453a
static const oa::Color kMuted   = {0.565F, 0.565F, 0.565F, 1.0F};  // #909090

static void populateFigure(oa::plot::Figure& inFig, const oa::Vec<GridCell>& inCells) {
	for (oa::I32 i = 0; i < kGridN; ++i) {
		const GridCell& cell = inCells[i];
		auto& ax = inFig.ax(i / kGridCols, i % kGridCols);
		if (cell.tile.isValid()) {
			ax.imshow(cell.tile);
		}
		ax.title(kClasses[cell.predicted], cell.correct ? kSuccess : kError);
		if (not cell.correct) {
			ax.caption(kClasses[cell.actual], kMuted);
		}
	}
}


// ─── show mode — oa::Viewer live source using fig.renderFrame ───────────────

class ImageGridClassifySource final : public oa::ViewerLiveSource {
public:
	ImageGridClassifySource(const oa::String& inDataDir, oa::I32 inTrainSteps)
		: dataDir_(inDataDir), trainSteps_(inTrainSteps),
		  fig_({
				// .title    = "OA — Fashion-MNIST Prediction grid",
				.rows     = kGridRows,
				.cols     = kGridCols,
				.width    = 800U,
				.height   = 800U,
				.hSpacing = 8,
				.vSpacing = 8,
				.padding  = 8
			})
	{}
	[[nodiscard]] oa::ViewerLiveCapabilities capabilities() const noexcept override {
		return {};
	}

	oa::Status open(oa::Engine& inEngine) override {
		engine_ = &inEngine;
		OA_RETURN_IF_ERROR(trainAndPredictGrid(
			inEngine, dataDir_, trainSteps_, cells_));
		populateFigure(fig_, cells_);
		return oa::Status::ok();
	}
	oa::Status init(oa::InputSystem&, oa::Fn<void(bool)>) override {
		return engine_ != nullptr
			? oa::Status::ok()
			: oa::Status::error(
				oa::StatusCode::FailedPrecondition,
				"Image grid source must be open before initialization");
	}
	oa::Status update(oa::F32 inDeltaMs) override {
		return std::isfinite(inDeltaMs) && inDeltaMs >= 0.0F
			? oa::Status::ok()
			: oa::Status::invalidArgument(
				"Image grid update requires a finite non-negative delta");
	}

	oa::Status render(
		oa::Ui& inUi,
		const oa::TextAtlas&,
		oa::U32 inWidth,
		oa::U32 inHeight) override {
		if (cells_.empty()) {
			return oa::Status::error(
				oa::StatusCode::FailedPrecondition,
				"Image grid render requires populated cells");
		}
		fig_.renderFrame(inWidth, inHeight, inUi);
		return oa::Status::ok();
	}

	oa::Status close() override {
		if (engine_ == nullptr) return oa::Status::ok();
		cells_.clear();
		engine_ = nullptr;
		return oa::Status::ok();
	}

private:
	oa::String          dataDir_;
	oa::I32             trainSteps_ = 2000;
	oa::plot::Figure    fig_;
	oa::Vec<GridCell>   cells_;
	oa::Engine*         engine_ = nullptr;
};


// ─── main ──────────────────────────────────────────────────────────────────

int main(int argc, char** argv) {
	const char* dataDir  = nullptr;
	oa::String defaultDataDir;
	const char* savePath = nullptr;
	oa::I32       trainSteps = 2000;

	for (int i = 1; i < argc; ++i) {
		oa::StringView a(argv[i]);
		if (a == "--save" and i + 1 < argc) {
			savePath = argv[++i];
		} else if (a == "--steps" and i + 1 < argc) {
			trainSteps = std::atoi(argv[++i]);
		} else if (dataDir == nullptr) {
			dataDir = argv[i];
		}
	}
	if (dataDir == nullptr) {
		defaultDataDir = oa::Paths::data("fashionMnist").string();
		dataDir = defaultDataDir.cStr();
	}

	std::printf("\n");
	std::printf("╔══════════════════════════════════════════════════════════════════╗\n");
	std::printf("║    OA Tutorial — Fashion-MNIST Prediction grid (oa::plot)          ║\n");
	std::printf("║    Train → Predict → display  (5×5 grid, 25 test images)         ║\n");
	std::printf("║    Green title = correct  ·  Red title = wrong                   ║\n");
	std::printf("║    Mode: %s  ║\n",
		savePath != nullptr
			? "saveTo (headless PNG output)                             "
			: "show (interactive window)                                ");
	std::printf("╚══════════════════════════════════════════════════════════════════╝\n\n");

	// ─── save mode: compute-only engine, no swapchain ──────────────────────
	if (savePath != nullptr) {
		oa::EngineConfig cfg;
		cfg.presentationMode = oa::PresentationMode::None;
		cfg.selectForThread = true;
		auto eR = oa::Engine::create(cfg);
		if (not eR.isOk()) {
			std::fprintf(stderr, "Engine create failed: %s\n",
			             eR.getStatus().toString().cStr());
			return 1;
		}
		oa::Engine& engine = *eR.getValue();

		oa::Vec<GridCell> cells;
		if (auto s = trainAndPredictGrid(engine, dataDir, trainSteps, cells); not s.isOk()) {
			std::fprintf(stderr, "Train/predict: %s\n", s.toString().cStr());
			return 1;
		}

		oa::plot::Figure fig({
			// .title    = "OA — Fashion-MNIST Prediction grid",
			.rows     = kGridRows,
			.cols     = kGridCols,
			.width    = 800U,
			.height   = 800U,
			.hSpacing = 8,
			.vSpacing = 8,
			.padding  = 4
		});
		populateFigure(fig, cells);

		const auto rc = fig.saveTo(engine, savePath);
		cells.clear();
		return rc.isOk() ? 0 : 1;
	}

	// ─── show mode: one oa::Viewer lifecycle with a figure source ───────────
	ImageGridClassifySource source(dataDir, trainSteps);
	oa::ViewerConfig config{
		.mode = oa::ViewerMode::Live,
		.liveSource = &source,
		.width = 800U,
		.height = 800U,
		.showHelp = false,
		.showStats = false,
		.showTimeline = false,
	};
	config.style.background = {0.039F, 0.039F, 0.039F, 1.0F};
	oa::Viewer viewer(config);
	return viewer.run().isOk() ? 0 : 1;
}
