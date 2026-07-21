// ═══════════════════════════════════════════════════════════════════════════
// OA Tutorial: Image-Grid Classification — Fashion-MNIST 5×5 grid
// Level 1 API — OaModule + OaPlot::Figure / Axes (FinalGlue Step 3e)
// ═══════════════════════════════════════════════════════════════════════════
//
// A compact consumer of the OA plot and viewer contracts:
// Fashion-MNIST classifier predictions rendered as a 5×5 image grid using
// OaPlot::Figure + Axes::Imshow / Title / Caption. Parallel to the TF Keras
// classification tutorial — same image grid, OA C++ syntax, GPU all the way
// through the recorder.
//
//   Same renderer body, two sinks (Architecture/OaArchitecture.md §10):
//     ./TutorialImageGridClassify                      → Show (window)
//     ./TutorialImageGridClassify --save grid.png      → SaveFig (batch)
//
// Usage:
//   ./Tutorial/Plot/TutorialImageGridClassify [data_dir] [--save path.png]
//
// Default data dir: $OA_MNIST_DATA, then ../oapy/dataset/FashionMNIST/raw
// ═══════════════════════════════════════════════════════════════════════════

#include <Oa/Ml.h>
#include <Oa/Ml/Autograd.h>
#include <Oa/Ui/Plot/Plot.h>
#include <Oa/Ui/Image.h>
#include <Oa/Ui/Input.h>
#include <Oa/Ui/Viewer.h>
#include <Oa/Runtime/Context.h>
#include <Oa/Runtime/Engine.h>
#include <Oa/Core/Log.h>

#include <fstream>
#include <cstring>
#include <cmath>


// ─── Fashion-MNIST class names ─────────────────────────────────────────────

static const char* kClasses[] = {
	"T-shirt", "Trouser",  "Pullover", "Dress",  "Coat",
	"Sandal",  "Shirt",    "Sneaker",  "Bag",    "Boot"
};
static constexpr OaI32 kNumClasses = 10;

// ─── Raw IDX loader for test images (needed for texture creation) ─────────

struct MnistData {
	OaVec<OaU8> Images;
	OaVec<OaU8> Labels;
	OaI32       Count = 0;
};

static OaU32 ReadBE32(std::ifstream& f) {
	OaU8 b[4];
	f.read(reinterpret_cast<char*>(b), 4);
	return (OaU32(b[0]) << 24) | (OaU32(b[1]) << 16) | (OaU32(b[2]) << 8) | OaU32(b[3]);
}

static bool LoadMnistIDX(const OaString& InDir,
                         const OaString& InImgFile,
                         const OaString& InLblFile,
                         MnistData& Out) {
	std::ifstream imgF((InDir + "/" + InImgFile).c_str(), std::ios::binary);
	std::ifstream lblF((InDir + "/" + InLblFile).c_str(), std::ios::binary);
	if (not imgF or not lblF) { return false; }
	if (ReadBE32(imgF) != 0x00000803U) { return false; }
	OaU32 n    = ReadBE32(imgF);
	OaU32 rows = ReadBE32(imgF);
	OaU32 cols = ReadBE32(imgF);
	if (rows != 28U or cols != 28U) { return false; }
	if (ReadBE32(lblF) != 0x00000801U) { return false; }
	if (ReadBE32(lblF) != n) { return false; }
	Out.Count = static_cast<OaI32>(n);
	Out.Images.Resize(static_cast<OaI64>(n) * 784);
	imgF.read(reinterpret_cast<char*>(Out.Images.Data()), Out.Images.Size());
	Out.Labels.Resize(static_cast<OaI64>(n));
	lblF.read(reinterpret_cast<char*>(Out.Labels.Data()), Out.Labels.Size());
	return true;
}


// ─── MLP classifier (same shape as TutorialMnistClassifierApi1) ───────────

class OaMnistClassifier : public OaModule {
public:
	OaMnistClassifier() {
		auto wd = OaFnMatrix::GetWeightDtype();
		Fc1_ = OaMakeSharedPtr<OaLinear>(784, 128);
		Fc1_->SetActivation(OaActivation::Relu);
		Fc1_->Parameters()[0].Data = OaFnMatrix::RandKaimingUniform(OaMatrixShape{128, 784}, wd);
		Fc2_ = OaMakeSharedPtr<OaLinear>(128, kNumClasses);
		Fc2_->Parameters()[0].Data = OaFnMatrix::RandGlorotUniform(OaMatrixShape{kNumClasses, 128}, wd);
		RegisterModule("fc1", Fc1_);
		RegisterModule("fc2", Fc2_);
	}

	OaMatrix Forward(const OaMatrix& InX) override {
		XNorm_  = OaFnMatrix::Scale(InX, 1.0F / 255.0F);
		H1_     = Fc1_->Forward(XNorm_);
		Logits_ = Fc2_->Forward(H1_);
		return Logits_;
	}

	void Backward(const OaMatrix& InDLogits) {
		auto& fc1P = Fc1_->Parameters();
		auto& fc2P = Fc2_->Parameters();
		auto gbw2 = OaFnMatrix::LinearWeightBiasBwd(H1_, InDLogits);
		auto dZ1  = OaFnMatrix::LinearDataReluBwd(InDLogits, fc2P[0].Data, H1_);
		auto gbw1 = OaFnMatrix::LinearWeightBiasBwd(XNorm_, dZ1);
		fc1P[0].Grad() = gbw1.GradWeight;
		fc1P[1].Grad() = gbw1.GradBias;
		fc2P[0].Grad() = gbw2.GradWeight;
		fc2P[1].Grad() = gbw2.GradBias;
		}

private:
	OaSharedPtr<OaLinear> Fc1_, Fc2_;
	OaMatrix XNorm_, H1_, Logits_;
};




// ─── Per-cell result + grid build helper ──────────────────────────────────

static constexpr OaI32 kGridCols = 5;
static constexpr OaI32 kGridRows = 5;
static constexpr OaI32 kGridN    = kGridCols * kGridRows;  // 25

struct GridCell {
	OaTexture Tile;
	OaI32     Actual    = 0;
	OaI32     Predicted = 0;
	bool      Correct   = false;
};

// Load Fashion-MNIST, train an MLP, fill InOutCells[0..25) with prediction
// tiles + labels. Returns OaStatus on failure.
static OaStatus TrainAndPredictGrid(OaEngine& InRt,
                                    const OaString&    InDataDir,
                                    OaI32              InTrainSteps,
                                    OaVec<GridCell>&   OutCells) {
	// Load raw test data for texture creation (OaDsMnist normalizes data)
	MnistData testData;
	if (not LoadMnistIDX(InDataDir, "t10k-images-idx3-ubyte", "t10k-labels-idx1-ubyte", testData)) {
		return OaStatus::Error("Fashion-MNIST test set not found");
	}

	OaDsMnist trainLoader(InDataDir, "train", 64, /*shuffle=*/true);
	OaDsMnist testLoader(InDataDir, "t10k", 100, /*shuffle=*/false);

	if (trainLoader.NumSamples() == 0 || testLoader.NumSamples() == 0) {
		return OaStatus::Error("Fashion-MNIST not found");
	}
	OA_LOG_INFO(OaLogComponent::App, "Loaded %d train / %d test images",
	            trainLoader.NumSamples(), testLoader.NumSamples());

	auto model     = OaMakeSharedPtr<OaMnistClassifier>();
	auto params    = model->AllParameterPtrs();
	auto optimizer = OaMakeUniquePtr<OaAdamW>(params, 0.001F);
	auto& ctx      = OaContext::GetDefault();

	constexpr OaI32 kBatch = 64;
	const OaI32 kStepsPerEpoch = trainLoader.NumSamples() / kBatch;

	// Training loop with OaItTraining
	OaCbProgressBar progressBar;
	OaCbSummary summary;
	OaMetricLoss lossMetric;
	progressBar.AddMetric(&lossMetric);

	OaItTraining loop(*optimizer, OaItTrainingConfig{
		.TotalSteps     = InTrainSteps,
		.StepsPerEpoch  = kStepsPerEpoch,
		.BatchSize      = kBatch,
		.TimerName      = "image_grid_step",
		.Metrics        = { &lossMetric },
		.Callbacks      = { &progressBar, &summary },
	});

	OA_LOG_INFO(OaLogComponent::App, "Training %d steps ...", InTrainSteps);

	OaVec<OaMatrix> xRing(static_cast<OaI64>(ctx.MaxAsyncSubmissions()));
	OaVec<OaMatrix> yRing(static_cast<OaI64>(ctx.MaxAsyncSubmissions()));

	while (not loop.IsDone()) {
		const OaI64 step = loop.Index();
		OaMatrix& batchX = xRing[(step - 1) % xRing.Size()];
		OaMatrix& batchY = yRing[(step - 1) % yRing.Size()];

		if (not trainLoader.NextBatch(batchX, batchY)) {
			trainLoader.Reset();
			trainLoader.NextBatch(batchX, batchY);
		}

		auto logits     = model->Forward(batchX);
		auto loss       = OaFnLoss::CrossEntropy(logits, batchY);
		auto gradLogits = OaFnLoss::CrossEntropyBwd(logits, batchY);
		model->Backward(gradLogits);
		loop.Next(loss);
	}
	if (not loop.Finish().IsOk()) {
		return OaStatus::Error("Training loop finish failed");
	}
	OA_LOG_INFO(OaLogComponent::App, "Training done.");

	// Inference on the first 25 test images.
	auto xTest = OaFnMatrix::FromBytes(
		OaSpan<const OaU8>(testData.Images.Data(), static_cast<OaI64>(kGridN) * 784),
		OaMatrixShape{kGridN, 784});
	auto logits = model->Forward(xTest);
	auto probs  = OaFnMatrix::Softmax(logits, -1);

	// Copy probs to host for argmax
	OaI32 batch = static_cast<OaI32>(probs.Size(0));
	OaI32 nCls  = static_cast<OaI32>(probs.Size(1));
	OaVec<OaF32> host(batch * nCls);
	(void)OaFnMatrix::CopyToHost(probs, host.Data(), host.Size() * sizeof(OaF32));

	// Use raw test data for texture creation
	const OaU8* labels = testData.Labels.Data();

	OutCells.Resize(kGridN);

	for (OaI32 i = 0; i < kGridN; ++i) {
		OaVec<OaU8> rgba(static_cast<OaI64>(28) * 28 * 4);
		OaI32 best  = 0;
		OaF32 bestV = host[static_cast<OaI64>(i) * kNumClasses];
		for (OaI32 j = 1; j < kNumClasses; ++j) {
			OaF32 v = host[static_cast<OaI64>(i) * kNumClasses + j];
			if (v > bestV) { bestV = v; best = j; }
		}
		OutCells[i].Actual    = static_cast<OaI32>(labels[i]);
		OutCells[i].Predicted = best;
		OutCells[i].Correct   = (best == OutCells[i].Actual);

		const OaU8* src = testData.Images.Data() + static_cast<OaI64>(i) * 784;
		for (OaI32 p = 0; p < 28 * 28; ++p) {
			OaU8 g = src[p];
			rgba[static_cast<OaI64>(p) * 4 + 0] = g;
			rgba[static_cast<OaI64>(p) * 4 + 1] = g;
			rgba[static_cast<OaI64>(p) * 4 + 2] = g;
			rgba[static_cast<OaI64>(p) * 4 + 3] = 255U;
		}
		auto r = OaTexture::FromPixels(InRt,
			OaSpan<const OaU8>(rgba.Data(), static_cast<OaI64>(28) * 28 * 4), 28, 28);
		if (not r.IsOk()) {
			return OaStatus::Error("OaTexture::FromPixels failed");
		}
		OutCells[i].Tile = *r;
	}

	OaI32 correct = 0;
	for (const auto& cell : OutCells) {
		if (cell.Correct) { ++correct; }
	}
	OA_LOG_INFO(OaLogComponent::App,
	            "Prediction grid: %d / %d correct on first %d test images",
	            correct, kGridN, kGridN);
	return OaStatus::Ok();
}


// ─── PopulateFigure ───────────────────────────────────────────────────────

static const OaColor kSuccess = {0.188F, 0.820F, 0.345F, 1.0F};  // #30d158
static const OaColor kError   = {1.000F, 0.271F, 0.227F, 1.0F};  // #ff453a
static const OaColor kMuted   = {0.565F, 0.565F, 0.565F, 1.0F};  // #909090

static void PopulateFigure(OaPlot::Figure& InFig, const OaVec<GridCell>& InCells) {
	for (OaI32 i = 0; i < kGridN; ++i) {
		const GridCell& cell = InCells[i];
		auto& ax = InFig.Ax(i / kGridCols, i % kGridCols);
		if (cell.Tile.IsValid()) {
			ax.Imshow(cell.Tile);
		}
		ax.Title(kClasses[cell.Predicted], cell.Correct ? kSuccess : kError);
		if (not cell.Correct) {
			ax.Caption(kClasses[cell.Actual], kMuted);
		}
	}
}


// ─── Show mode — OaViewer live source using fig.RenderFrame ───────────────

class ImageGridClassifySource final : public OaViewerLiveSource {
public:
	ImageGridClassifySource(const OaString& InDataDir, OaI32 InTrainSteps)
		: DataDir_(InDataDir), TrainSteps_(InTrainSteps),
		  Fig_({
				// .Title    = "OA — Fashion-MNIST Prediction Grid",
				.Rows     = kGridRows,
				.Cols     = kGridCols,
				.Width    = 800U,
				.Height   = 800U,
				.HSpacing = 8,
				.VSpacing = 8,
				.Padding  = 8
			})
	{}

	OaStatus Open(OaEngine& InEngine) override {
		OA_RETURN_IF_ERROR(TrainAndPredictGrid(
			InEngine, DataDir_, TrainSteps_, Cells_));
		PopulateFigure(Fig_, Cells_);
		Fig_.Rasterize(InEngine);
		return OaStatus::Ok();
	}

	void Render(
		OaUi& InOui,
		const OaTextAtlas&,
		OaU32 InWidth,
		OaU32 InHeight) override {
		if (Cells_.Empty()) { return; }
		Fig_.RenderFrame(InWidth, InHeight, InOui);
	}

	OaStatus Close() override {
		auto& rt = *OaEngine::GetGlobal();
		for (auto& cell : Cells_) {
			if (cell.Tile.IsValid()) { cell.Tile.Destroy(rt); }
		}
		return OaStatus::Ok();
	}

private:
	OaString          DataDir_;
	OaI32             TrainSteps_ = 2000;
	OaPlot::Figure    Fig_;
	OaVec<GridCell>   Cells_;
};


// ─── main ──────────────────────────────────────────────────────────────────

int main(int argc, char** argv) {
	const char* dataDir  = nullptr;
	const char* savePath = nullptr;
	OaI32       trainSteps = 2000;

	for (int i = 1; i < argc; ++i) {
		OaStringView a(argv[i]);
		if (a == "--save" and i + 1 < argc) {
			savePath = argv[++i];
		} else if (a == "--steps" and i + 1 < argc) {
			trainSteps = std::atoi(argv[++i]);
		} else if (dataDir == nullptr) {
			dataDir = argv[i];
		}
	}
	if (dataDir == nullptr) { dataDir = std::getenv("OA_MNIST_DATA"); }
	if (dataDir == nullptr) { dataDir = "../oapy/dataset/FashionMNIST/raw"; }

	std::printf("\n");
	std::printf("╔══════════════════════════════════════════════════════════════════╗\n");
	std::printf("║    OA Tutorial — Fashion-MNIST Prediction Grid (OaPlot)          ║\n");
	std::printf("║    Train → Predict → Display  (5×5 grid, 25 test images)         ║\n");
	std::printf("║    Green title = correct  ·  Red title = wrong                   ║\n");
	std::printf("║    Mode: %s  ║\n",
		savePath != nullptr
			? "SaveFig (headless PNG output)                            "
			: "Show (interactive window)                                ");
	std::printf("╚══════════════════════════════════════════════════════════════════╝\n\n");

	// ─── Save mode: compute-only engine, no swapchain ──────────────────────
	if (savePath != nullptr) {
		OaEngineConfig cfg;
		cfg.PresentationMode = OaPresentationMode::None;
		cfg.RegisterAsGlobal = true;
		auto eR = OaEngine::Create(cfg);
		if (not eR.IsOk()) {
			std::fprintf(stderr, "Engine create failed: %s\n",
			             eR.GetStatus().ToString().c_str());
			return 1;
		}
		OaEngine& engine = *eR.GetValue();

		OaVec<GridCell> cells;
		if (auto s = TrainAndPredictGrid(engine, dataDir, trainSteps, cells); not s.IsOk()) {
			std::fprintf(stderr, "Train/predict: %s\n", s.ToString().c_str());
			return 1;
		}

		OaPlot::Figure fig({
			// .Title    = "OA — Fashion-MNIST Prediction Grid",
			.Rows     = kGridRows,
			.Cols     = kGridCols,
			.Width    = 800U,
			.Height   = 800U,
			.HSpacing = 8,
			.VSpacing = 8,
			.Padding  = 4
		});
		PopulateFigure(fig, cells);

		const auto rc = fig.SaveFig(savePath);
		for (auto& cell : cells) {
			if (cell.Tile.IsValid()) { cell.Tile.Destroy(engine); }
		}
		return rc.IsOk() ? 0 : 1;
	}

	// ─── Show mode: one OaViewer lifecycle with a figure source ───────────
	ImageGridClassifySource source(dataDir, trainSteps);
	OaViewerConfig config{
		.Mode = OaViewerMode::Live,
		.LiveSource = &source,
		.Width = 800U,
		.Height = 800U,
		.ShowHelp = false,
		.ShowStats = false,
		.ShowTimeline = false,
	};
	config.Style.Background = {0.039F, 0.039F, 0.039F, 1.0F};
	OaViewer viewer(config);
	return viewer.Run().IsOk() ? 0 : 1;
}
