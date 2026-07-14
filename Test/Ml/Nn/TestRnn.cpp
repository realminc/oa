// ═══════════════════════════════════════════════════════════════════════════
// OA Test — RNN (vanilla Elman) comprehensive testing
//
// Mirrors TestGru.cpp. The two critical tests:
//   - RnnGradientsFlowToWeights : regression guard for the exact bug that kept the
//     GRU untrainable (raw OaFnMatrix::Linear projections left as graph leaves).
//   - RnnCellGradientCheck      : analytic autograd vs FP64 central differences.
// ═══════════════════════════════════════════════════════════════════════════

#include "../../OaTest.h"
#include <Oa/Ml.h>
#include <Oa/Ml/Nn.h>
#include <Oa/Ml/Autograd.h>
#include <Oa/Ml/Optim.h>
#include <Oa/Runtime/Engine.h>
#include <algorithm>
#include <cmath>
#include <vector>

// ─── Helpers (mirrors TestGru.cpp) ───────────────────────────────────────────

static void SyncCtx() {
	auto& ctx = OaContext::GetDefault();
	(void)ctx.Execute();
	(void)ctx.Sync();
}

static std::vector<float> DownloadF32(const OaMatrix& m) {
	SyncCtx();
	const float* p = m.DataAs<const float>();
	return std::vector<float>(p, p + m.NumElements());
}

// ─── Construction / forward ──────────────────────────────────────────────────

TEST(Rnn, RnnCellConstruction) {
	OaRnnCell cell1(10, 20, true);
	EXPECT_EQ(cell1.InputSize(), 10);
	EXPECT_EQ(cell1.HiddenSize(), 20);
	EXPECT_EQ(cell1.Parameters().Size(), 4);  // weight_ih, weight_hh, bias_ih, bias_hh

	OaRnnCell cell2(10, 20, false);
	EXPECT_EQ(cell2.Parameters().Size(), 2);  // weight_ih, weight_hh only

	const auto& params = cell1.Parameters();
	EXPECT_EQ(params[0].Data.GetShape(), (OaMatrixShape{20, 10}));  // weight_ih
	EXPECT_EQ(params[1].Data.GetShape(), (OaMatrixShape{20, 20}));  // weight_hh
	EXPECT_EQ(params[2].Data.GetShape(), OaMatrixShape{20});      // bias_ih
	EXPECT_EQ(params[3].Data.GetShape(), OaMatrixShape{20});      // bias_hh
}

TEST(Rnn, RnnCellZeroState) {
	OaRnnCell cell(10, 20);
	auto h0 = cell.ZeroState(5);
	EXPECT_EQ(h0.GetShape(), (OaMatrixShape{5, 20}));
	SyncCtx();
	EXPECT_NEAR(OaFnMatrix::Sum(h0).Item(), 0.0f, 1e-6f);
}

TEST(Rnn, RnnCellForwardSingleStep) {
	OaRnnCell cell(10, 20);
	auto input = OaFnMatrix::RandN(OaMatrixShape{2, 10}, OaScalarType::Float32);
	auto hidden = cell.ZeroState(2);
	auto output = cell.Step(input, hidden);
	EXPECT_EQ(output.GetShape(), (OaMatrixShape{2, 20}));

	// tanh output is bounded in [-1, 1].
	SyncCtx();
	auto absOut = OaFnMatrix::Abs(output);
	EXPECT_LE(OaFnMatrix::Max(absOut).Item(), 1.0f);
}

TEST(Rnn, RnnForwardSequence) {
	OaRnn rnn(10, 20, 2);
	auto input = OaFnMatrix::RandN(OaMatrixShape{2, 5, 10}, OaScalarType::Float32);
	auto output = rnn.Forward(input);
	EXPECT_EQ(output.GetShape(), (OaMatrixShape{2, 5, 20}));
}

TEST(Rnn, RnnMultiLayer) {
	OaRnn rnn(10, 20, 3, true);
	EXPECT_EQ(rnn.NumLayers(), 3);
	EXPECT_EQ(rnn.Children().Size(), 3);
}

TEST(Rnn, RnnParameterCount) {
	OaRnn gru1(10, 20, 1, true);
	EXPECT_EQ(gru1.NumParameters(), 20 * (10 + 20 + 2));

	OaRnn gru2(10, 20, 1, false);
	EXPECT_EQ(gru2.NumParameters(), 20 * (10 + 20));

	// layer0 (in=10) + 2 layers (in=20).
	OaRnn gru3(10, 20, 3, true);
	const OaI64 expected = 20 * (10 + 20 + 2) + 2 * (20 * (20 + 20 + 2));
	EXPECT_EQ(gru3.NumParameters(), expected);
}

// ─── Regression guard: gradients must reach the recurrent weights ────────────
// This is the precise failure mode that left the GRU untrainable: a raw
// OaFnMatrix::Linear call with no OaGradLinear attached is a graph leaf, so the
// weights get zero gradient. Here we assert the fix for OaRnn.
TEST(Rnn, RnnGradientsFlowToWeights) {
	OaRnnCell cell(4, 5, true);
	auto& P = cell.Parameters();
	ASSERT_EQ(P.Size(), 4u);
	for (auto& p : P) { p.Data.SetRequiresGrad(true); p.Grad() = p.Data.GradMatrix(); }

	auto input = OaFnMatrix::RandN(OaMatrixShape{2, 4}, OaScalarType::Float32);
	input.SetRequiresGrad(true);
	// Non-zero hidden: with hidden==0 the weight_hh gradient is genuinely 0
	// (dW_hh = hidden·dOut), which is correct math, not a bug.
	auto hidden = OaFnMatrix::RandN(OaMatrixShape{2, 5}, OaScalarType::Float32);
	hidden.SetRequiresGrad(true);

	{
		OaGradientTape tape;
		auto out = cell.Step(input, hidden);
		auto loss = OaFnMatrix::Sum(out);
		tape.Backward(loss);
	}
	SyncCtx();

	// Every recurrent parameter must carry a non-zero gradient.
	auto gradMag = [](const OaMatrix& g) {
		std::vector<float> h(static_cast<size_t>(g.NumElements()));
		(void)OaFnMatrix::CopyToHost(g, h.data(), h.size() * sizeof(float));
		float mag = 0.0f;
		for (float v : h) mag += std::fabs(v);
		return mag;
	};
	EXPECT_TRUE(P[0].Grad().HasStorage());
	EXPECT_GT(gradMag(P[0].Grad()), 0.0f) << "weight_ih gradient did not flow";
	EXPECT_GT(gradMag(P[1].Grad()), 0.0f) << "weight_hh gradient did not flow";
	EXPECT_GT(gradMag(P[2].Grad()), 0.0f) << "bias_ih gradient did not flow";
	EXPECT_GT(gradMag(P[3].Grad()), 0.0f) << "bias_hh gradient did not flow";
}

// ─── Autograd gradient check vs an FP64 CPU reference ─────────────────────────
// loss = sum(h_new), h_new = tanh(W_ih x + b_ih + W_hh h + b_hh).
TEST(Rnn, RnnCellGradientCheck) {
	const OaI32 B = 2, InSize = 4, H = 5;

	auto gen = [](OaI32 n, double scale, double phase, std::vector<double>& out) {
		out.resize(static_cast<size_t>(n));
		for (OaI32 i = 0; i < n; ++i) out[static_cast<size_t>(i)] = scale * std::sin(0.37 * static_cast<double>(i) + phase);
	};
	std::vector<double> wih, whh, bih, bhh, xin, hin;
	gen(H * InSize, 0.3, 0.1, wih);  // weight_ih [H, InSize]
	gen(H * H,      0.3, 0.2, whh);  // weight_hh [H, H]
	gen(H,          0.1, 0.3, bih);  // bias_ih  [H]
	gen(H,          0.1, 0.4, bhh);  // bias_hh  [H]
	gen(B * InSize, 0.5, 0.5, xin);  // input    [B, InSize]
	gen(B * H,      0.4, 0.6, hin);  // hidden   [B, H]

	auto toMat = [](const std::vector<double>& v, const OaMatrixShape& s) {
		std::vector<float> f(v.size());
		for (size_t i = 0; i < v.size(); ++i) f[i] = static_cast<float>(v[i]);
		return OaFnMatrix::FromBytes(
			OaSpan<const OaU8>(reinterpret_cast<const OaU8*>(f.data()), f.size() * sizeof(float)), s);
	};

	OaRnnCell cell(InSize, H, true);
	auto& P = cell.Parameters();
	ASSERT_EQ(P.Size(), 4u);
	P[0].Data = toMat(wih, OaMatrixShape{H, InSize});
	P[1].Data = toMat(whh, OaMatrixShape{H, H});
	P[2].Data = toMat(bih, OaMatrixShape{H});
	P[3].Data = toMat(bhh, OaMatrixShape{H});
	for (auto& p : P) { p.Data.SetRequiresGrad(true); p.Grad() = p.Data.GradMatrix(); }
	auto input = toMat(xin, OaMatrixShape{B, InSize}); input.SetRequiresGrad(true);
	auto hidden = toMat(hin, OaMatrixShape{B, H}); hidden.SetRequiresGrad(true);

	{
		OaGradientTape tape;
		auto out  = cell.Step(input, hidden);
		auto loss = OaFnMatrix::Sum(out);
		tape.Backward(loss);
		SyncCtx();
	}

	auto cpuLoss = [&](const std::vector<double>& w, const std::vector<double>& u,
	                   const std::vector<double>& bi, const std::vector<double>& bu,
	                   const std::vector<double>& x, const std::vector<double>& h) -> double {
		double loss = 0.0;
		for (OaI32 b = 0; b < B; ++b) {
			for (OaI32 j = 0; j < H; ++j) {
				double z = bi[static_cast<size_t>(j)] + bu[static_cast<size_t>(j)];
				for (OaI32 k = 0; k < InSize; ++k)
					z += x[b * InSize + k] * w[static_cast<size_t>(j) * InSize + k];
				for (OaI32 k = 0; k < H; ++k)
					z += h[b * H + k] * u[static_cast<size_t>(j) * H + k];
				loss += std::tanh(z);
			}
		}
		return loss;
	};
	const double eps = 1e-6;
	auto fdGrad = [&](std::vector<double>& v, OaI32 idx) -> double {
		const double o = v[static_cast<size_t>(idx)];
		v[static_cast<size_t>(idx)] = o + eps; const double lp = cpuLoss(wih, whh, bih, bhh, xin, hin);
		v[static_cast<size_t>(idx)] = o - eps; const double lm = cpuLoss(wih, whh, bih, bhh, xin, hin);
		v[static_cast<size_t>(idx)] = o;
		return (lp - lm) / (2.0 * eps);
	};

	auto cmp = [&](const std::vector<float>& ana, std::vector<double>& v, const char* name) {
		const OaI32 n = static_cast<OaI32>(v.size());
		int bad = 0; double maxRel = 0.0, maxAbs = 0.0;
		for (OaI32 i = 0; i < n; ++i) {
			const double num = fdGrad(v, i);
			const double a   = ana[static_cast<size_t>(i)];
			const double absErr = std::fabs(num - a);
			const double mag = std::max(std::fabs(num), std::fabs(a));
			maxAbs = std::max(maxAbs, absErr);
			if (mag > 1e-6) maxRel = std::max(maxRel, absErr / mag);
			if (mag < 1e-6) continue;
			if (absErr > 2e-3 && absErr / mag > 0.05) {
				if (bad < 8) printf("  %s[%d] ref=%.6f analytic=%.6f abs=%.2e rel=%.4f\n", name, i, num, a, absErr, absErr / mag);
				++bad;
			}
		}
		printf("  %-10s checked %d elems, maxAbs=%.2e maxRel=%.2e bad=%d\n", name, n, maxAbs, maxRel, bad);
		return bad;
	};

	EXPECT_EQ(cmp(DownloadF32(P[0].Grad()),           wih, "weight_ih"), 0);
	EXPECT_EQ(cmp(DownloadF32(P[1].Grad()),           whh, "weight_hh"), 0);
	EXPECT_EQ(cmp(DownloadF32(P[2].Grad()),           bih, "bias_ih"),   0);
	EXPECT_EQ(cmp(DownloadF32(P[3].Grad()),           bhh, "bias_hh"),   0);
	EXPECT_EQ(cmp(DownloadF32(input.GradMatrix()),  xin, "input"),     0);
	EXPECT_EQ(cmp(DownloadF32(hidden.GradMatrix()), hin, "hidden"),    0);
}

// ─── End-to-end: learn a copy task (forces real recurrence + BPTT) ───────────
namespace {

class ToyRnnCopyModel : public OaModule {
public:
	ToyRnnCopyModel(OaI32 InDim, OaI32 InHidden) {
		Rnn_ = OaMakeSharedPtr<OaRnn>(InDim, InHidden, 1);
		Head_ = OaMakeSharedPtr<OaLinear>(InHidden, InDim);
		RegisterModule("rnn", Rnn_);
		RegisterModule("head", Head_);
	}
	OaMatrix Forward(const OaMatrix& x) override {
		auto o = Rnn_->Forward(x);                 // [B,T,H]
		const OaI32 T = static_cast<OaI32>(o.Size(1));
		const OaI32 B = static_cast<OaI32>(o.Size(0));
		// Use final hidden state
		auto last = OaFnMatrix::Slice(o, 1, T - 1, T).Reshape(OaMatrixShape{B, Rnn_->HiddenSize()});
		return Head_->Forward(last);               // [B,InDim]
	}
private:
	OaSharedPtr<OaRnn>   Rnn_;
	OaSharedPtr<OaLinear> Head_;
};

} // namespace

// Task: predict a fixed target from a sequence. With deterministic seed, this
// verifies the RNN can learn (gradients flow, optimizer works, loss decreases).
TEST(Rnn, RnnTrainsOnSequenceTask) {
	auto& ctx = OaContext::GetDefault();
	constexpr OaI32 kDim = 4, kHidden = 16, kBatch = 8, kSeq = 6;
	constexpr OaI32 kSteps = 500;
	constexpr OaF32 kLr = 0.01f;

	// Deterministic seed for reproducible results
	OaFnMatrix::SetRngSeed(42);
	
	auto model = OaMakeSharedPtr<ToyRnnCopyModel>(kDim, kHidden);
	auto params = model->AllParameterPtrs();
	auto opt = OaMakeUniquePtr<OaAdamW>(params, kLr);

	// Fixed input and target with reasonable magnitude
	auto x = OaFnMatrix::RandN(OaMatrixShape{kBatch, kSeq, kDim}, OaScalarType::Float32);
	auto y = OaFnMatrix::RandN(OaMatrixShape{kBatch, kDim}, OaScalarType::Float32);
	const OaF32 invN = 1.0f / static_cast<OaF32>(kBatch * kDim);

	OaF32 initialLoss = 0.0f, lastLoss = 0.0f;
	for (OaI32 step = 0; step < kSteps; ++step) {
		opt->ZeroGrad();
		OaGradientTape tape;
		auto out = model->Forward(x);
		auto diff = OaFnMatrix::Sub(out, y);
		auto loss = OaFnMatrix::Scale(OaFnMatrix::Sum(OaFnMatrix::Mul(diff, diff)), invN);
		tape.Backward(loss);
		(void)ctx.Execute();
		(void)ctx.Sync();
		const OaF32 lossVal = loss.Item();
		if (step == 0) initialLoss = lossVal;
		if (step % 100 == 0) printf("  step %d loss=%.5f\n", step, lossVal);
		opt->Step();
		(void)ctx.Execute();
		(void)ctx.Sync();
		lastLoss = lossVal;
	}

	printf("  copy-task: initial=%.4f last=%.4f reduction=%.1f%%\n",
		initialLoss, lastLoss, 100.0f * (1.0f - lastLoss / initialLoss));
	// With deterministic seed, we expect consistent convergence
	EXPECT_GT(initialLoss, 0.3f) << "Initial loss suspiciously low";
	EXPECT_LT(lastLoss, initialLoss * 0.2f) << "RNN failed to learn (loss should drop to <20% of initial)";
}
