// ═══════════════════════════════════════════════════════════════════════════
// OA Test — RNN (vanilla Elman) comprehensive testing
//
// Mirrors TestGru.cpp. The two critical tests:
//   - RnnGradientsFlowToWeights : regression guard for the exact bug that kept the
//     GRU untrainable (raw oa::FnMatrix::linear projections left as graph leaves).
//   - RnnCellGradientCheck      : analytic autograd vs FP64 central differences.
// ═══════════════════════════════════════════════════════════════════════════

#include "../../oaTest.h"
#include <oa/ml.h>
#include <oa/ml/nn.h>
#include <oa/ml/autograd.h>
#include <oa/ml/optim.h>
#include <oa/runtime/engine.h>
#include <algorithm>
#include <cmath>
#include <vector>

// ─── helpers (mirrors TestGru.cpp) ───────────────────────────────────────────

static void syncCtx() {
	auto& ctx = oa::ExecutionSession::getActive();
	(void)testSubmitAndWait(ctx);
}

static std::vector<float> downloadF32(const oa::Matrix& m) {
	syncCtx();
	const float* p = m.dataAs<const float>();
	return std::vector<float>(p, p + m.numElements());
}

// ─── Construction / forward ──────────────────────────────────────────────────

TEST(Rnn, RnnCellConstruction) {
	oa::RnnCell cell1(10, 20, true);
	EXPECT_EQ(cell1.inputSize(), 10);
	EXPECT_EQ(cell1.hiddenSize(), 20);
	EXPECT_EQ(cell1.parameters().size(), 4);  // weight_ih, weight_hh, bias_ih, bias_hh

	oa::RnnCell cell2(10, 20, false);
	EXPECT_EQ(cell2.parameters().size(), 2);  // weight_ih, weight_hh only

	const auto& params = cell1.parameters();
	EXPECT_EQ(params[0].data.getShape(), (oa::MatrixShape{20, 10}));  // weight_ih
	EXPECT_EQ(params[1].data.getShape(), (oa::MatrixShape{20, 20}));  // weight_hh
	EXPECT_EQ(params[2].data.getShape(), oa::MatrixShape{20});      // bias_ih
	EXPECT_EQ(params[3].data.getShape(), oa::MatrixShape{20});      // bias_hh
}

TEST(Rnn, RnnCellZeroState) {
	oa::RnnCell cell(10, 20);
	auto h0 = cell.zeroState(5);
	EXPECT_EQ(h0.getShape(), (oa::MatrixShape{5, 20}));
	syncCtx();
	EXPECT_NEAR(oa::FnMatrix::sum(h0).item(), 0.0f, 1e-6f);
}

TEST(Rnn, RnnCellForwardSingleStep) {
	oa::RnnCell cell(10, 20);
	auto input = oa::FnMatrix::randN(oa::MatrixShape{2, 10}, oa::ScalarType::Float32);
	auto hidden = cell.zeroState(2);
	auto output = cell.step(input, hidden);
	EXPECT_EQ(output.getShape(), (oa::MatrixShape{2, 20}));

	// tanh output is bounded in [-1, 1].
	syncCtx();
	auto absOut = oa::FnMatrix::abs(output);
	EXPECT_LE(oa::FnMatrix::max(absOut).item(), 1.0f);
}

TEST(Rnn, RnnForwardSequence) {
	oa::Rnn rnn(10, 20, 2);
	auto input = oa::FnMatrix::randN(oa::MatrixShape{2, 5, 10}, oa::ScalarType::Float32);
	auto output = rnn.forward(input);
	EXPECT_EQ(output.getShape(), (oa::MatrixShape{2, 5, 20}));
}

TEST(Rnn, RnnMultiLayer) {
	oa::Rnn rnn(10, 20, 3, true);
	EXPECT_EQ(rnn.numLayers(), 3);
	EXPECT_EQ(rnn.children().size(), 3);
}

TEST(Rnn, RnnParameterCount) {
	oa::Rnn gru1(10, 20, 1, true);
	EXPECT_EQ(gru1.numParameters(), 20 * (10 + 20 + 2));

	oa::Rnn gru2(10, 20, 1, false);
	EXPECT_EQ(gru2.numParameters(), 20 * (10 + 20));

	// layer0 (in=10) + 2 layers (in=20).
	oa::Rnn gru3(10, 20, 3, true);
	const oa::I64 expected = 20 * (10 + 20 + 2) + 2 * (20 * (20 + 20 + 2));
	EXPECT_EQ(gru3.numParameters(), expected);
}

// ─── Regression guard: gradients must reach the recurrent weights ────────────
// This is the precise failure mode that left the GRU untrainable: a raw
// oa::FnMatrix::linear call with no GradLinear attached is a graph leaf, so the
// weights get zero gradient. Here we assert the fix for oa::Rnn.
TEST(Rnn, RnnGradientsFlowToWeights) {
	oa::RnnCell cell(4, 5, true);
	auto& P = cell.parameters();
	ASSERT_EQ(P.size(), 4u);
	for (auto& p : P) { p.data.setRequiresGrad(true); p.grad() = p.data.gradMatrix(); }

	auto input = oa::FnMatrix::randN(oa::MatrixShape{2, 4}, oa::ScalarType::Float32);
	input.setRequiresGrad(true);
	// Non-zero hidden: with hidden==0 the weight_hh gradient is genuinely 0
	// (dW_hh = hidden·dOut), which is correct math, not a bug.
	auto hidden = oa::FnMatrix::randN(oa::MatrixShape{2, 5}, oa::ScalarType::Float32);
	hidden.setRequiresGrad(true);

	{
		oa::GradientTape tape;
		auto out = cell.step(input, hidden);
		auto loss = oa::FnMatrix::sum(out);
		tape.backward(loss);
	}
	syncCtx();

	// Every recurrent parameter must carry a non-zero gradient.
	auto gradMag = [](const oa::Matrix& g) {
		std::vector<float> h(static_cast<size_t>(g.numElements()));
		(void)oa::FnMatrix::copyToHost(g, h.data(), h.size() * sizeof(float));
		float mag = 0.0f;
		for (float v : h) mag += std::fabs(v);
		return mag;
	};
	EXPECT_TRUE(P[0].grad().hasStorage());
	EXPECT_GT(gradMag(P[0].grad()), 0.0f) << "weight_ih gradient did not flow";
	EXPECT_GT(gradMag(P[1].grad()), 0.0f) << "weight_hh gradient did not flow";
	EXPECT_GT(gradMag(P[2].grad()), 0.0f) << "bias_ih gradient did not flow";
	EXPECT_GT(gradMag(P[3].grad()), 0.0f) << "bias_hh gradient did not flow";
}

// ─── autograd gradient check vs an FP64 CPU reference ─────────────────────────
// loss = sum(h_new), h_new = tanh(W_ih x + b_ih + W_hh h + b_hh).
TEST(Rnn, RnnCellGradientCheck) {
	const oa::I32 B = 2, inSize = 4, H = 5;

	auto gen = [](oa::I32 n, double scale, double phase, std::vector<double>& out) {
		out.resize(static_cast<size_t>(n));
		for (oa::I32 i = 0; i < n; ++i) out[static_cast<size_t>(i)] = scale * std::sin(0.37 * static_cast<double>(i) + phase);
	};
	std::vector<double> wih, whh, bih, bhh, xin, hin;
	gen(H * inSize, 0.3, 0.1, wih);  // weight_ih [H, inSize]
	gen(H * H,      0.3, 0.2, whh);  // weight_hh [H, H]
	gen(H,          0.1, 0.3, bih);  // bias_ih  [H]
	gen(H,          0.1, 0.4, bhh);  // bias_hh  [H]
	gen(B * inSize, 0.5, 0.5, xin);  // input    [B, inSize]
	gen(B * H,      0.4, 0.6, hin);  // hidden   [B, H]

	auto toMat = [](const std::vector<double>& v, const oa::MatrixShape& s) {
		std::vector<float> f(v.size());
		for (size_t i = 0; i < v.size(); ++i) f[i] = static_cast<float>(v[i]);
		return oa::FnMatrix::fromBytes(
			oa::Span<const oa::U8>(reinterpret_cast<const oa::U8*>(f.data()), f.size() * sizeof(float)), s);
	};

	oa::RnnCell cell(inSize, H, true);
	auto& P = cell.parameters();
	ASSERT_EQ(P.size(), 4u);
	P[0].data = toMat(wih, oa::MatrixShape{H, inSize});
	P[1].data = toMat(whh, oa::MatrixShape{H, H});
	P[2].data = toMat(bih, oa::MatrixShape{H});
	P[3].data = toMat(bhh, oa::MatrixShape{H});
	for (auto& p : P) { p.data.setRequiresGrad(true); p.grad() = p.data.gradMatrix(); }
	auto input = toMat(xin, oa::MatrixShape{B, inSize}); input.setRequiresGrad(true);
	auto hidden = toMat(hin, oa::MatrixShape{B, H}); hidden.setRequiresGrad(true);

	{
		oa::GradientTape tape;
		auto out  = cell.step(input, hidden);
		auto loss = oa::FnMatrix::sum(out);
		tape.backward(loss);
		syncCtx();
	}

	auto cpuLoss = [&](const std::vector<double>& w, const std::vector<double>& u,
	                   const std::vector<double>& bi, const std::vector<double>& bu,
	                   const std::vector<double>& x, const std::vector<double>& h) -> double {
		double loss = 0.0;
		for (oa::I32 b = 0; b < B; ++b) {
			for (oa::I32 j = 0; j < H; ++j) {
				double z = bi[static_cast<size_t>(j)] + bu[static_cast<size_t>(j)];
				for (oa::I32 k = 0; k < inSize; ++k)
					z += x[b * inSize + k] * w[static_cast<size_t>(j) * inSize + k];
				for (oa::I32 k = 0; k < H; ++k)
					z += h[b * H + k] * u[static_cast<size_t>(j) * H + k];
				loss += std::tanh(z);
			}
		}
		return loss;
	};
	const double eps = 1e-6;
	auto fdGrad = [&](std::vector<double>& v, oa::I32 idx) -> double {
		const double o = v[static_cast<size_t>(idx)];
		v[static_cast<size_t>(idx)] = o + eps; const double lp = cpuLoss(wih, whh, bih, bhh, xin, hin);
		v[static_cast<size_t>(idx)] = o - eps; const double lm = cpuLoss(wih, whh, bih, bhh, xin, hin);
		v[static_cast<size_t>(idx)] = o;
		return (lp - lm) / (2.0 * eps);
	};

	auto cmp = [&](const std::vector<float>& ana, std::vector<double>& v, const char* name) {
		const oa::I32 n = static_cast<oa::I32>(v.size());
		int bad = 0; double maxRel = 0.0, maxAbs = 0.0;
		for (oa::I32 i = 0; i < n; ++i) {
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

	EXPECT_EQ(cmp(downloadF32(P[0].grad()),           wih, "weight_ih"), 0);
	EXPECT_EQ(cmp(downloadF32(P[1].grad()),           whh, "weight_hh"), 0);
	EXPECT_EQ(cmp(downloadF32(P[2].grad()),           bih, "bias_ih"),   0);
	EXPECT_EQ(cmp(downloadF32(P[3].grad()),           bhh, "bias_hh"),   0);
	EXPECT_EQ(cmp(downloadF32(input.gradMatrix()),  xin, "input"),     0);
	EXPECT_EQ(cmp(downloadF32(hidden.gradMatrix()), hin, "hidden"),    0);
}

// ─── End-to-end: learn a copy task (forces real recurrence + BPTT) ───────────
namespace {

class ToyRnnCopyModel : public oa::Module {
public:
	ToyRnnCopyModel(oa::I32 inDim, oa::I32 inHidden) {
		rnn_ = oa::makeShared<oa::Rnn>(inDim, inHidden, 1);
		head_ = oa::makeShared<oa::Linear>(inHidden, inDim);
		registerModule("rnn", rnn_);
		registerModule("head", head_);
	}
	oa::Matrix forward(const oa::Matrix& x) override {
		auto o = rnn_->forward(x);                 // [B,T,H]
		const oa::I32 T = static_cast<oa::I32>(o.size(1));
		const oa::I32 B = static_cast<oa::I32>(o.size(0));
		// Use final hidden state
		auto last = oa::FnMatrix::slice(o, 1, T - 1, T).reshape(oa::MatrixShape{B, rnn_->hiddenSize()});
		return head_->forward(last);               // [B,inDim]
	}
private:
	oa::SharedPtr<oa::Rnn>   rnn_;
	oa::SharedPtr<oa::Linear> head_;
};

} // namespace

// Task: predict a fixed target from a sequence. With deterministic seed, this
// verifies the RNN can learn (gradients flow, optimizer works, loss decreases).
TEST(Rnn, RnnTrainsOnSequenceTask) {
	auto& ctx = oa::ExecutionSession::getActive();
	constexpr oa::I32 kDim = 4, kHidden = 16, kBatch = 8, kSeq = 6;
	constexpr oa::I32 kSteps = 500;
	constexpr oa::F32 kLr = 0.01f;

	// Deterministic seed for reproducible results
	oa::FnMatrix::setRngSeed(42);
	
	auto model = oa::makeShared<ToyRnnCopyModel>(kDim, kHidden);
	auto params = model->allParameterPtrs();
	auto opt = oa::makeUnique<oa::AdamW>(params, kLr);

	// Fixed input and target with reasonable magnitude
	auto x = oa::FnMatrix::randN(oa::MatrixShape{kBatch, kSeq, kDim}, oa::ScalarType::Float32);
	auto y = oa::FnMatrix::randN(oa::MatrixShape{kBatch, kDim}, oa::ScalarType::Float32);
	const oa::F32 invN = 1.0f / static_cast<oa::F32>(kBatch * kDim);

	oa::F32 initialLoss = 0.0f, lastLoss = 0.0f;
	for (oa::I32 step = 0; step < kSteps; ++step) {
		opt->zeroGrad();
		oa::GradientTape tape;
		auto out = model->forward(x);
		auto diff = oa::FnMatrix::sub(out, y);
		auto loss = oa::FnMatrix::scale(oa::FnMatrix::sum(oa::FnMatrix::mul(diff, diff)), invN);
		tape.backward(loss);
		(void)testSubmitAndWait(ctx);
		const oa::F32 lossVal = loss.item();
		if (step == 0) initialLoss = lossVal;
		if (step % 100 == 0) printf("  step %d loss=%.5f\n", step, lossVal);
		opt->step();
		(void)testSubmitAndWait(ctx);
		lastLoss = lossVal;
	}

	printf("  copy-task: initial=%.4f last=%.4f reduction=%.1f%%\n",
		initialLoss, lastLoss, 100.0f * (1.0f - lastLoss / initialLoss));
	// With deterministic seed, we expect consistent convergence
	EXPECT_GT(initialLoss, 0.3f) << "Initial loss suspiciously low";
	EXPECT_LT(lastLoss, initialLoss * 0.2f) << "RNN failed to learn (loss should drop to <20% of initial)";
}
