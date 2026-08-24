// ─────────────────────────────────────────────────────────────────────────────
// TestAutogradLeak — does a training loop leak GPU memory across steps?
//
// Symptom under investigation (session 9): the NLP tutorials OOM-kill the box and
// crawl (~23 sps) on a tiny model. That is a leak signature, not "model too big".
// This probe runs *tiny* models for many steps and watches OaVma usedBytes. A hard
// VRAM cap fails the test fast so it can NEVER OOM the display server.
//
//   CoreTapeNoLeak — Linear → CrossEntropy. Isolates the core tape/accumulate path.
//   GruTapeNoLeak  — Embedding → GRU → Linear. Same probe, GRU-specific path.
//
// If usedBytes climbs monotonically the lifecycle leaks; if it plateaus it does not.
// ─────────────────────────────────────────────────────────────────────────────

#include "oaTest.h"
#include <oa/ml.h>
#include <oa/ml/autograd.h>
#include <oa/runtime/executionSession.h>
#include <oa/runtime/engine.h>
#include <oa/runtime/engine/allocatorAccess.h>
#include <cstdio>
#include <unistd.h>

namespace {

constexpr oa::U64 kVramCapBytes = 1024ull * 1024ull * 1024ull;  // 1 GiB safety cap
constexpr oa::U64 kRssCapBytes  = 3072ull * 1024ull * 1024ull;  // 3 GiB host-RAM cap

oa::U64 usedBytesNow() {
	return oa::EngineAllocatorAccess::get(testEngine()).getStats().usedBytes;
}

// Resident host memory (RSS) in bytes, read from /proc/self/statm. This is what the
// OS OOM-killer watches; a GPU-only VRAM cap won't catch a host-side autograd leak.
oa::U64 rssBytesNow() {
	FILE* f = std::fopen("/proc/self/statm", "r");
	if (not f) return 0;
	unsigned long sizePages = 0, residentPages = 0;
	const int got = std::fscanf(f, "%lu %lu", &sizePages, &residentPages);
	std::fclose(f);
	if (got < 2) return 0;
	return static_cast<oa::U64>(residentPages) * static_cast<oa::U64>(sysconf(_SC_PAGESIZE));
}

// print the detailed VMA breakdown: live allocation bytes vs reserved block bytes,
// and the live/block counts. A big (Block - alloc) gap = fragmentation/pool slack;
// a climbing allocationCount/allocationBytes = a genuine lifecycle leak.
void printAllocBreakdown(oa::I32 inStep, const char* inTag) {
	const auto s = oa::EngineAllocatorAccess::get(testEngine()).getStats();
	printf("  [step %d] %s VRAM alloc=%.1f miB (count=%llu) block=%.1f MiB  |  host RSS=%.1f MiB\n",
		inStep, inTag,
		static_cast<double>(s.allocationBytes) / (1024.0 * 1024.0),
		static_cast<unsigned long long>(s.allocationCount),
		static_cast<double>(s.blockBytes) / (1024.0 * 1024.0),
		static_cast<double>(rssBytesNow()) / (1024.0 * 1024.0));
}

// abort the probe before host RAM can OOM-kill the desktop. call every step.
void guardRssOrAbort(oa::I32 inStep) {
	const oa::U64 rss = rssBytesNow();
	if (rss > kRssCapBytes) {
		printf("  [step %d] host RSS=%.1f MiB exceeded %.0f MiB cap — aborting probe\n",
			inStep, static_cast<double>(rss) / (1024.0 * 1024.0),
			static_cast<double>(kRssCapBytes) / (1024.0 * 1024.0));
		ADD_FAILURE() << "host RSS cap exceeded at step " << inStep
			<< " (RSS=" << rss << ") — host-side autograd/graph leak";
		std::abort();
	}
}

// Sample usedBytes after a barrier so transient buffers are accounted, then assert
// the cap. Returns the reading so the caller can compare across steps.
oa::U64 sampleVramOrAbort(oa::I32 inStep) {
	auto& ctx = oa::ExecutionSession::getActive();
	(void)testSubmitAndWait(ctx);
	const oa::U64 used = usedBytesNow();
	if (used > kVramCapBytes) {
		printf("  [step %d] usedBytes=%llu exceeded %llu cap — aborting probe\n",
			inStep, static_cast<unsigned long long>(used),
			static_cast<unsigned long long>(kVramCapBytes));
		ADD_FAILURE() << "VRAM cap exceeded at step " << inStep
			<< " (usedBytes=" << used << ") — lifecycle leak";
		std::abort();  // never let it grow toward a desktop-killing OOM
	}
	return used;
}

// ─── Tiny models (mirror the tutorials, miniature dims) ──────────────────────

constexpr oa::I32 kVocab  = 27;
constexpr oa::I32 kEmbed  = 16;
constexpr oa::I32 kHidden = 16;
constexpr oa::I32 kSeq    = 8;
constexpr oa::I32 kBatch  = 8;

class TinyLinearModel : public oa::Module {
public:
	TinyLinearModel() {
		head_ = oa::makeShared<oa::Linear>(kHidden, kVocab);
		for (auto& p : head_->parameters()) {
			p.data.setRequiresGrad(true);
			p.grad() = p.data.gradMatrix();
		}
		registerModule("head", head_);
	}
	oa::Matrix forward(const oa::Matrix& inX) override { return head_->forward(inX); }
private:
	oa::SharedPtr<oa::Linear> head_;
};

class TinyGruModel : public oa::Module {
public:
	TinyGruModel() {
		auto wd = oa::FnMatrix::weightDtype();
		embed_ = oa::makeShared<oa::Embedding>(kVocab, kEmbed);
		embed_->parameters()[0].data = oa::FnMatrix::randN(oa::MatrixShape{kVocab, kEmbed}, wd);
		embed_->parameters()[0].data.setRequiresGrad(true);
		embed_->parameters()[0].grad() = embed_->parameters()[0].data.gradMatrix();

		gru_ = oa::makeShared<oa::Gru>(kEmbed, kHidden, 1);
		for (auto& p : gru_->parameters()) {
			p.data.setRequiresGrad(true);
			p.grad() = p.data.gradMatrix();
		}

		head_ = oa::makeShared<oa::Linear>(kHidden, kVocab);
		for (auto& p : head_->parameters()) {
			p.data.setRequiresGrad(true);
			p.grad() = p.data.gradMatrix();
		}
		registerModule("embed", embed_);
		registerModule("gru", gru_);
		registerModule("head", head_);
	}
	oa::Matrix forward(const oa::Matrix& inTokens) override {
		const oa::I32 batch  = static_cast<oa::I32>(inTokens.size(0));
		const oa::I32 seqLen = static_cast<oa::I32>(inTokens.size(1));
		auto embedded = embed_->forward(inTokens).reshape(oa::MatrixShape{batch, seqLen, kEmbed});
		auto gruOut   = gru_->forward(embedded);
		auto last     = oa::FnMatrix::slice(gruOut, 1, seqLen - 1, seqLen);
		last = last.reshape(oa::MatrixShape{batch, kHidden});
		return head_->forward(last);
	}
private:
	oa::SharedPtr<oa::Embedding> embed_;
	oa::SharedPtr<oa::Gru>       gru_;
	oa::SharedPtr<oa::Linear>    head_;
};

oa::Matrix makeLabels() {
	oa::Vec<oa::U8> y(kBatch);
	for (oa::I32 b = 0; b < kBatch; ++b) y[b] = static_cast<oa::U8>(b % kVocab);
	return oa::FnMatrix::fromBytes(oa::Span<const oa::U8>(y.data(), y.size()),
		oa::MatrixShape{kBatch}, oa::ScalarType::UInt8);
}

oa::Matrix makeTokens() {
	oa::Vec<oa::U8> x(static_cast<oa::I64>(kBatch) * kSeq);
	for (oa::I64 i = 0; i < x.size(); ++i) x[i] = static_cast<oa::U8>(i % kVocab);
	return oa::FnMatrix::fromBytes(oa::Span<const oa::U8>(x.data(), x.size()),
		oa::MatrixShape{kBatch, kSeq}, oa::ScalarType::UInt8);
}

// run a loop, sampling VRAM at a warmup step and at the end. The warmup reading
// (after pools reach steady state) is the baseline; the delta over the back half
// of the loop is what a leak would inflate.
template <typename ModelT, typename ForwardFn>
void runProbe(const char* inName, oa::I32 inSteps, ForwardFn inMakeInput) {
	auto  model  = oa::makeShared<ModelT>();
	auto  params = model->allParameterPtrs();
	auto  opt    = oa::makeUnique<oa::AdamW>(params, 0.01F);
	auto  labels = makeLabels();

	printf("\n[%s] tiny model, %d steps (cap=%llu MiB)\n",
		inName, inSteps, static_cast<unsigned long long>(kVramCapBytes >> 20));

	oa::U64 baseline = 0;
	constexpr oa::I32 kWarmup = 20;

	for (oa::I32 step = 0; step < inSteps; ++step) {
		auto x = inMakeInput();
		opt->zeroGrad();
		oa::GradientTape tape;
		auto logits = model->forward(x);
		auto loss   = oa::FnLoss::crossEntropy(logits, labels);
		tape.backward(loss);
		opt->step();

		if (step == kWarmup) {
			baseline = sampleVramOrAbort(step);
			printf("  [step %d] baseline usedBytes = %llu (%.1f MiB)\n",
				step, static_cast<unsigned long long>(baseline),
				static_cast<double>(baseline) / (1024.0 * 1024.0));
			printAllocBreakdown(step, "baseline");
		} else if (step > kWarmup && (step % 100 == 0 || step == inSteps - 1)) {
			oa::U64 used = sampleVramOrAbort(step);
			oa::I64 delta = static_cast<oa::I64>(used) - static_cast<oa::I64>(baseline);
			printf("  [step %d] usedBytes = %llu (%.1f MiB)  Δ vs baseline = %+lld bytes (%+.2f MiB)\n",
				step, static_cast<unsigned long long>(used),
				static_cast<double>(used) / (1024.0 * 1024.0),
				static_cast<long long>(delta),
				static_cast<double>(delta) / (1024.0 * 1024.0));
			printAllocBreakdown(step, "         ");
		}
	}

	const oa::U64 final = sampleVramOrAbort(inSteps);
	const oa::I64 growth = static_cast<oa::I64>(final) - static_cast<oa::I64>(baseline);
	printf("[%s] growth over %d steps after warmup: %+lld bytes (%+.2f MiB)\n",
		inName, inSteps - kWarmup, static_cast<long long>(growth),
		static_cast<double>(growth) / (1024.0 * 1024.0));

	// A correct lifecycle plateaus after warmup. Allow a small slack for pool
	// rounding; anything above 16 MiB of monotonic growth on a tiny model is a leak.
	EXPECT_LT(growth, 16ll * 1024 * 1024)
		<< "VRAM grew " << growth << " bytes after warmup — autograd/graph lifecycle leak";
}

} // namespace

TEST(AutogradLeak, CoreTapeNoLeak) {
	runProbe<TinyLinearModel>("CoreTape", 400, [] {
		return oa::FnMatrix::randN(oa::MatrixShape{kBatch, kHidden}, oa::FnMatrix::weightDtype());
	});
}

TEST(AutogradLeak, GruTapeNoLeak) {
	runProbe<TinyGruModel>("GruTape", 300, [] { return makeTokens(); });
}

// ─── Real-tutorial dims, but under the same 1 GiB hard cap ───────────────────
// probe the former large GRU tutorial dimensions: embed=128, hidden=256,
// seq=40, batch=64. If the box OOMs
// from genuine activation/working-set volume, this probe will trip the cap and
// std::abort() *before* it can hurt the display server. If it plateaus well under
// 1 GiB, memory is NOT the bottleneck and the 23-sps drag is CPU-side dispatch.
constexpr oa::I32 kRVocab  = 27;
constexpr oa::I32 kREmbed  = 128;
constexpr oa::I32 kRHidden = 256;
constexpr oa::I32 kRSeq    = 40;
constexpr oa::I32 kRBatch  = 64;

class RealGruModel : public oa::Module {
public:
	RealGruModel() {
		auto wd = oa::FnMatrix::weightDtype();
		embed_ = oa::makeShared<oa::Embedding>(kRVocab, kREmbed);
		embed_->parameters()[0].data = oa::FnMatrix::randN(oa::MatrixShape{kRVocab, kREmbed}, wd);
		embed_->parameters()[0].data.setRequiresGrad(true);
		embed_->parameters()[0].grad() = embed_->parameters()[0].data.gradMatrix();

		gru_ = oa::makeShared<oa::Gru>(kREmbed, kRHidden, 1);
		for (auto& p : gru_->parameters()) {
			p.data.setRequiresGrad(true);
			p.grad() = p.data.gradMatrix();
		}
		head_ = oa::makeShared<oa::Linear>(kRHidden, kRVocab);
		for (auto& p : head_->parameters()) {
			p.data.setRequiresGrad(true);
			p.grad() = p.data.gradMatrix();
		}
		registerModule("embed", embed_);
		registerModule("gru", gru_);
		registerModule("head", head_);
	}
	oa::Matrix forward(const oa::Matrix& inTokens) override {
		const oa::I32 batch  = static_cast<oa::I32>(inTokens.size(0));
		const oa::I32 seqLen = static_cast<oa::I32>(inTokens.size(1));
		auto embedded = embed_->forward(inTokens).reshape(oa::MatrixShape{batch, seqLen, kREmbed});
		auto gruOut   = gru_->forward(embedded);
		auto last     = oa::FnMatrix::slice(gruOut, 1, seqLen - 1, seqLen);
		last = last.reshape(oa::MatrixShape{batch, kRHidden});
		return head_->forward(last);
	}
private:
	oa::SharedPtr<oa::Embedding> embed_;
	oa::SharedPtr<oa::Gru>       gru_;
	oa::SharedPtr<oa::Linear>    head_;
};

TEST(AutogradLeak, GruTapeRealDims) {
	auto makeTokens = [] {
		oa::Vec<oa::U8> x(static_cast<oa::I64>(kRBatch) * kRSeq);
		for (oa::I64 i = 0; i < x.size(); ++i) x[i] = static_cast<oa::U8>(i % kRVocab);
		return oa::FnMatrix::fromBytes(oa::Span<const oa::U8>(x.data(), x.size()),
			oa::MatrixShape{kRBatch, kRSeq}, oa::ScalarType::UInt8);
	};
	auto makeLabels = [] {
		oa::Vec<oa::U8> y(kRBatch);
		for (oa::I32 b = 0; b < kRBatch; ++b) y[b] = static_cast<oa::U8>(b % kRVocab);
		return oa::FnMatrix::fromBytes(oa::Span<const oa::U8>(y.data(), y.size()),
			oa::MatrixShape{kRBatch}, oa::ScalarType::UInt8);
	};
	auto  model  = oa::makeShared<RealGruModel>();
	auto  params = model->allParameterPtrs();
	auto  opt    = oa::makeUnique<oa::AdamW>(params, 0.001F);
	auto  labels = makeLabels();
	printf("\n[GruRealDims] embed=%d hidden=%d seq=%d batch=%d, 80 steps (cap=%llu MiB)\n",
		kREmbed, kRHidden, kRSeq, kRBatch,
		static_cast<unsigned long long>(kVramCapBytes >> 20));

	oa::U64 rssBaseline = 0;
	constexpr oa::I32 kWarmup = 10;
	constexpr oa::I32 kSteps  = 80;
	for (oa::I32 step = 0; step < kSteps; ++step) {
		auto x = makeTokens();
		opt->zeroGrad();
		oa::GradientTape tape;
		auto logits = model->forward(x);
		auto loss   = oa::FnLoss::crossEntropy(logits, labels);
		tape.backward(loss);
		opt->step();
		// Force materialization every step, the way the real tutorial does when it
		// reads the loss scalar for its metric — this is what bounds VRAM. The point
		// of the probe is whether HOST RAM stays bounded once VRAM is flushed.
		(void)testSubmitAndWait(oa::ExecutionSession::getActive());
		guardRssOrAbort(step);
		if (step == kWarmup) {
			rssBaseline = rssBytesNow();
			printAllocBreakdown(step, "baseline");
		} else if (step > kWarmup && (step % 10 == 0 || step == kSteps - 1)) {
			const oa::I64 dRss = static_cast<oa::I64>(rssBytesNow()) - static_cast<oa::I64>(rssBaseline);
			printf("  [step %d] host RSS Δ vs baseline = %+.2f MiB\n",
				step, static_cast<double>(dRss) / (1024.0 * 1024.0));
			printAllocBreakdown(step, "         ");
		}
	}
	const oa::I64 rssGrowth = static_cast<oa::I64>(rssBytesNow()) - static_cast<oa::I64>(rssBaseline);
	printf("[GruRealDims] host RSS growth after warmup: %+.2f miB (VRAM live=%.1f MiB)\n",
		static_cast<double>(rssGrowth) / (1024.0 * 1024.0),
		static_cast<double>(oa::EngineAllocatorAccess::get(
			testEngine()).getStats().allocationBytes)
			/ (1024.0 * 1024.0));
	EXPECT_LT(rssGrowth, 128ll * 1024 * 1024)
		<< "real-dims GRU HOST RAM grew " << rssGrowth << " bytes after warmup — host-side leak";
}

// ─── Per-step working-set vs sequence length ─────────────────────────────────
// The real-dims probe showed ~2.8 GiB live *while the tape is alive* for one GRU
// step — ~100x the ~30 MiB of activations BPTT actually needs. This sweep measures
// the with-tape working set as a function of seq length to see how it scales (per
// unrolled timestep) and pin the inflation to the GRU unroll vs a fixed overhead.
TEST(AutogradLeak, GruWorkingSetVsSeq) {
	constexpr oa::I32 kVoc = 27, kEmb = 128, kHid = 256, kBat = 64;
	auto makeModel = [&] {
		auto wd = oa::FnMatrix::weightDtype();
		return std::make_tuple(
			oa::makeShared<oa::Embedding>(kVoc, kEmb),
			oa::makeShared<oa::Gru>(kEmb, kHid, 1),
			oa::makeShared<oa::Linear>(kHid, kVoc),
			wd);
	};
	const oa::I32 seqs[] = {5, 10, 20, 40};
	printf("\n[WorkingSetVsSeq] embed=%d hidden=%d batch=%d — VRAM live while tape held:\n",
		kEmb, kHid, kBat);
	for (oa::I32 seq : seqs) {
		auto [embed, gru, head, wd] = makeModel();
		embed->parameters()[0].data = oa::FnMatrix::randN(oa::MatrixShape{kVoc, kEmb}, wd);
		auto setGrad = [](auto& mod) {
			for (auto& p : mod->parameters()) { p.data.setRequiresGrad(true); p.grad() = p.data.gradMatrix(); }
		};
		setGrad(embed); setGrad(gru); setGrad(head);

		oa::Vec<oa::U8> xb(static_cast<oa::I64>(kBat) * seq);
		for (oa::I64 i = 0; i < xb.size(); ++i) xb[i] = static_cast<oa::U8>(i % kVoc);
		oa::Vec<oa::U8> yb(kBat);
		for (oa::I32 b = 0; b < kBat; ++b) yb[b] = static_cast<oa::U8>(b % kVoc);
		auto labels = oa::FnMatrix::fromBytes(oa::Span<const oa::U8>(yb.data(), yb.size()),
			oa::MatrixShape{kBat}, oa::ScalarType::UInt8);

		oa::U64 peak = 0;
		for (oa::I32 step = 0; step < 4; ++step) {  // warm a couple then measure
			auto tokens = oa::FnMatrix::fromBytes(oa::Span<const oa::U8>(xb.data(), xb.size()),
				oa::MatrixShape{kBat, seq}, oa::ScalarType::UInt8);
			oa::GradientTape tape;
			auto emb = embed->forward(tokens).reshape(oa::MatrixShape{kBat, seq, kEmb});
			auto g   = gru->forward(emb);
			auto last = oa::FnMatrix::slice(g, 1, seq - 1, seq).reshape(oa::MatrixShape{kBat, kHid});
			auto logits = head->forward(last);
			auto loss = oa::FnLoss::crossEntropy(logits, labels);
			tape.backward(loss);
			(void)testSubmitAndWait(oa::ExecutionSession::getActive());
			// measure here: tape (and all saved activations) still alive.
			peak = oa::EngineAllocatorAccess::get(
				testEngine()).getStats().allocationBytes;
			guardRssOrAbort(step);
		}
		// After the loop the last tape is destroyed; read the resting set too.
		(void)testSubmitAndWait(oa::ExecutionSession::getActive());
		const oa::U64 resting = oa::EngineAllocatorAccess::get(
			testEngine()).getStats().allocationBytes;
		printf("  seq=%-3d  with-tape live = %7.1f MiB   resting = %6.1f MiB   per-timestep = %5.1f MiB\n",
			seq, static_cast<double>(peak) / (1024.0 * 1024.0),
			static_cast<double>(resting) / (1024.0 * 1024.0),
			static_cast<double>(peak) / (1024.0 * 1024.0) / static_cast<double>(seq));
	}
}
