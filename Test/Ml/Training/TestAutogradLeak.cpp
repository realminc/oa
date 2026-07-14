// ─────────────────────────────────────────────────────────────────────────────
// TestAutogradLeak — does a training loop leak GPU memory across steps?
//
// Symptom under investigation (Session 9): the NLP tutorials OOM-kill the box and
// crawl (~23 sps) on a tiny model. That is a leak signature, not "model too big".
// This probe runs *tiny* models for many steps and watches OaVma UsedBytes. A hard
// VRAM cap fails the test fast so it can NEVER OOM the display server.
//
//   CoreTapeNoLeak — Linear → CrossEntropy. Isolates the core tape/accumulate path.
//   GruTapeNoLeak  — Embedding → GRU → Linear. Same probe, GRU-specific path.
//
// If UsedBytes climbs monotonically the lifecycle leaks; if it plateaus it does not.
// ─────────────────────────────────────────────────────────────────────────────

#include "../../Test/OaTest.h"
#include <Oa/Ml.h>
#include <Oa/Ml/Autograd.h>
#include <Oa/Runtime/Context.h>
#include <Oa/Runtime/Engine.h>
#include <cstdio>
#include <unistd.h>

namespace {

constexpr OaU64 kVramCapBytes = 1024ull * 1024ull * 1024ull;  // 1 GiB safety cap
constexpr OaU64 kRssCapBytes  = 3072ull * 1024ull * 1024ull;  // 3 GiB host-RAM cap

OaU64 UsedBytesNow() {
	return OaComputeEngine::GetGlobal()->Allocator.GetStats().UsedBytes;
}

// Resident host memory (RSS) in bytes, read from /proc/self/statm. This is what the
// OS OOM-killer watches; a GPU-only VRAM cap won't catch a host-side autograd leak.
OaU64 RssBytesNow() {
	FILE* f = std::fopen("/proc/self/statm", "r");
	if (not f) return 0;
	unsigned long sizePages = 0, residentPages = 0;
	const int got = std::fscanf(f, "%lu %lu", &sizePages, &residentPages);
	std::fclose(f);
	if (got < 2) return 0;
	return static_cast<OaU64>(residentPages) * static_cast<OaU64>(sysconf(_SC_PAGESIZE));
}

// Print the detailed VMA breakdown: live allocation bytes vs reserved block bytes,
// and the live/block counts. A big (Block - Alloc) gap = fragmentation/pool slack;
// a climbing AllocationCount/AllocationBytes = a genuine lifecycle leak.
void PrintAllocBreakdown(OaI32 InStep, const char* InTag) {
	const auto s = OaComputeEngine::GetGlobal()->Allocator.GetStats();
	printf("  [step %d] %s VRAM alloc=%.1f MiB (count=%llu) block=%.1f MiB  |  host RSS=%.1f MiB\n",
		InStep, InTag,
		static_cast<double>(s.AllocationBytes) / (1024.0 * 1024.0),
		static_cast<unsigned long long>(s.AllocationCount),
		static_cast<double>(s.BlockBytes) / (1024.0 * 1024.0),
		static_cast<double>(RssBytesNow()) / (1024.0 * 1024.0));
}

// Abort the probe before host RAM can OOM-kill the desktop. Call every step.
void GuardRssOrAbort(OaI32 InStep) {
	const OaU64 rss = RssBytesNow();
	if (rss > kRssCapBytes) {
		printf("  [step %d] host RSS=%.1f MiB exceeded %.0f MiB cap — aborting probe\n",
			InStep, static_cast<double>(rss) / (1024.0 * 1024.0),
			static_cast<double>(kRssCapBytes) / (1024.0 * 1024.0));
		ADD_FAILURE() << "host RSS cap exceeded at step " << InStep
			<< " (RSS=" << rss << ") — host-side autograd/graph leak";
		std::abort();
	}
}

// Sample UsedBytes after a barrier so transient buffers are accounted, then assert
// the cap. Returns the reading so the caller can compare across steps.
OaU64 SampleVramOrAbort(OaI32 InStep) {
	auto& ctx = OaContext::GetDefault();
	(void)ctx.Execute();
	(void)ctx.Sync();
	const OaU64 used = UsedBytesNow();
	if (used > kVramCapBytes) {
		printf("  [step %d] UsedBytes=%llu exceeded %llu cap — aborting probe\n",
			InStep, static_cast<unsigned long long>(used),
			static_cast<unsigned long long>(kVramCapBytes));
		ADD_FAILURE() << "VRAM cap exceeded at step " << InStep
			<< " (UsedBytes=" << used << ") — lifecycle leak";
		std::abort();  // never let it grow toward a desktop-killing OOM
	}
	return used;
}

// ─── Tiny models (mirror the tutorials, miniature dims) ──────────────────────

constexpr OaI32 kVocab  = 27;
constexpr OaI32 kEmbed  = 16;
constexpr OaI32 kHidden = 16;
constexpr OaI32 kSeq    = 8;
constexpr OaI32 kBatch  = 8;

class TinyLinearModel : public OaModule {
public:
	TinyLinearModel() {
		Head_ = OaMakeSharedPtr<OaLinear>(kHidden, kVocab);
		for (auto& p : Head_->Parameters()) {
			p.Data.SetRequiresGrad(true);
			p.Grad() = p.Data.GradMatrix();
		}
		RegisterModule("head", Head_);
	}
	OaMatrix Forward(const OaMatrix& InX) override { return Head_->Forward(InX); }
private:
	OaSharedPtr<OaLinear> Head_;
};

class TinyGruModel : public OaModule {
public:
	TinyGruModel() {
		auto wd = OaFnMatrix::GetWeightDtype();
		Embed_ = OaMakeSharedPtr<OaEmbedding>(kVocab, kEmbed);
		Embed_->Parameters()[0].Data = OaFnMatrix::RandN(OaMatrixShape{kVocab, kEmbed}, wd);
		Embed_->Parameters()[0].Data.SetRequiresGrad(true);
		Embed_->Parameters()[0].Grad() = Embed_->Parameters()[0].Data.GradMatrix();

		Gru_ = OaMakeSharedPtr<OaGru>(kEmbed, kHidden, 1);
		for (auto& p : Gru_->Parameters()) {
			p.Data.SetRequiresGrad(true);
			p.Grad() = p.Data.GradMatrix();
		}

		Head_ = OaMakeSharedPtr<OaLinear>(kHidden, kVocab);
		for (auto& p : Head_->Parameters()) {
			p.Data.SetRequiresGrad(true);
			p.Grad() = p.Data.GradMatrix();
		}
		RegisterModule("embed", Embed_);
		RegisterModule("gru", Gru_);
		RegisterModule("head", Head_);
	}
	OaMatrix Forward(const OaMatrix& InTokens) override {
		const OaI32 batch  = static_cast<OaI32>(InTokens.Size(0));
		const OaI32 seqLen = static_cast<OaI32>(InTokens.Size(1));
		auto embedded = Embed_->Forward(InTokens).Reshape(OaMatrixShape{batch, seqLen, kEmbed});
		auto gruOut   = Gru_->Forward(embedded);
		auto last     = OaFnMatrix::Slice(gruOut, 1, seqLen - 1, seqLen);
		last = last.Reshape(OaMatrixShape{batch, kHidden});
		return Head_->Forward(last);
	}
private:
	OaSharedPtr<OaEmbedding> Embed_;
	OaSharedPtr<OaGru>       Gru_;
	OaSharedPtr<OaLinear>    Head_;
};

OaMatrix MakeLabels() {
	OaVec<OaU8> y(kBatch);
	for (OaI32 b = 0; b < kBatch; ++b) y[b] = static_cast<OaU8>(b % kVocab);
	return OaFnMatrix::FromBytes(OaSpan<const OaU8>(y.Data(), y.Size()),
		OaMatrixShape{kBatch}, OaScalarType::UInt8);
}

OaMatrix MakeTokens() {
	OaVec<OaU8> x(static_cast<OaI64>(kBatch) * kSeq);
	for (OaI64 i = 0; i < x.Size(); ++i) x[i] = static_cast<OaU8>(i % kVocab);
	return OaFnMatrix::FromBytes(OaSpan<const OaU8>(x.Data(), x.Size()),
		OaMatrixShape{kBatch, kSeq}, OaScalarType::UInt8);
}

// Run a loop, sampling VRAM at a warmup step and at the end. The warmup reading
// (after pools reach steady state) is the baseline; the delta over the back half
// of the loop is what a leak would inflate.
template <typename ModelT, typename ForwardFn>
void RunProbe(const char* InName, OaI32 InSteps, ForwardFn InMakeInput) {
	auto  model  = OaMakeSharedPtr<ModelT>();
	auto  params = model->AllParameterPtrs();
	auto  opt    = OaMakeUniquePtr<OaAdamW>(params, 0.01F);
	auto  labels = MakeLabels();

	printf("\n[%s] tiny model, %d steps (cap=%llu MiB)\n",
		InName, InSteps, static_cast<unsigned long long>(kVramCapBytes >> 20));

	OaU64 baseline = 0;
	constexpr OaI32 kWarmup = 20;

	for (OaI32 step = 0; step < InSteps; ++step) {
		auto x = InMakeInput();
		opt->ZeroGrad();
		OaGradientTape tape;
		auto logits = model->Forward(x);
		auto loss   = OaFnLoss::CrossEntropy(logits, labels);
		tape.Backward(loss);
		opt->Step();

		if (step == kWarmup) {
			baseline = SampleVramOrAbort(step);
			printf("  [step %d] baseline UsedBytes = %llu (%.1f MiB)\n",
				step, static_cast<unsigned long long>(baseline),
				static_cast<double>(baseline) / (1024.0 * 1024.0));
			PrintAllocBreakdown(step, "baseline");
		} else if (step > kWarmup && (step % 100 == 0 || step == InSteps - 1)) {
			OaU64 used = SampleVramOrAbort(step);
			OaI64 delta = static_cast<OaI64>(used) - static_cast<OaI64>(baseline);
			printf("  [step %d] UsedBytes = %llu (%.1f MiB)  Δ vs baseline = %+lld bytes (%+.2f MiB)\n",
				step, static_cast<unsigned long long>(used),
				static_cast<double>(used) / (1024.0 * 1024.0),
				static_cast<long long>(delta),
				static_cast<double>(delta) / (1024.0 * 1024.0));
			PrintAllocBreakdown(step, "         ");
		}
	}

	const OaU64 final = SampleVramOrAbort(InSteps);
	const OaI64 growth = static_cast<OaI64>(final) - static_cast<OaI64>(baseline);
	printf("[%s] growth over %d steps after warmup: %+lld bytes (%+.2f MiB)\n",
		InName, InSteps - kWarmup, static_cast<long long>(growth),
		static_cast<double>(growth) / (1024.0 * 1024.0));

	// A correct lifecycle plateaus after warmup. Allow a small slack for pool
	// rounding; anything above 16 MiB of monotonic growth on a tiny model is a leak.
	EXPECT_LT(growth, 16ll * 1024 * 1024)
		<< "VRAM grew " << growth << " bytes after warmup — autograd/graph lifecycle leak";
}

} // namespace

TEST(AutogradLeak, CoreTapeNoLeak) {
	RunProbe<TinyLinearModel>("CoreTape", 400, [] {
		return OaFnMatrix::RandN(OaMatrixShape{kBatch, kHidden}, OaFnMatrix::GetWeightDtype());
	});
}

TEST(AutogradLeak, GruTapeNoLeak) {
	RunProbe<TinyGruModel>("GruTape", 300, [] { return MakeTokens(); });
}

// ─── Real-tutorial dims, but under the same 1 GiB hard cap ───────────────────
// Probe the former large GRU tutorial dimensions: embed=128, hidden=256,
// seq=40, batch=64. If the box OOMs
// from genuine activation/working-set volume, this probe will trip the cap and
// std::abort() *before* it can hurt the display server. If it plateaus well under
// 1 GiB, memory is NOT the bottleneck and the 23-sps drag is CPU-side dispatch.
constexpr OaI32 kRVocab  = 27;
constexpr OaI32 kREmbed  = 128;
constexpr OaI32 kRHidden = 256;
constexpr OaI32 kRSeq    = 40;
constexpr OaI32 kRBatch  = 64;

class RealGruModel : public OaModule {
public:
	RealGruModel() {
		auto wd = OaFnMatrix::GetWeightDtype();
		Embed_ = OaMakeSharedPtr<OaEmbedding>(kRVocab, kREmbed);
		Embed_->Parameters()[0].Data = OaFnMatrix::RandN(OaMatrixShape{kRVocab, kREmbed}, wd);
		Embed_->Parameters()[0].Data.SetRequiresGrad(true);
		Embed_->Parameters()[0].Grad() = Embed_->Parameters()[0].Data.GradMatrix();

		Gru_ = OaMakeSharedPtr<OaGru>(kREmbed, kRHidden, 1);
		for (auto& p : Gru_->Parameters()) {
			p.Data.SetRequiresGrad(true);
			p.Grad() = p.Data.GradMatrix();
		}
		Head_ = OaMakeSharedPtr<OaLinear>(kRHidden, kRVocab);
		for (auto& p : Head_->Parameters()) {
			p.Data.SetRequiresGrad(true);
			p.Grad() = p.Data.GradMatrix();
		}
		RegisterModule("embed", Embed_);
		RegisterModule("gru", Gru_);
		RegisterModule("head", Head_);
	}
	OaMatrix Forward(const OaMatrix& InTokens) override {
		const OaI32 batch  = static_cast<OaI32>(InTokens.Size(0));
		const OaI32 seqLen = static_cast<OaI32>(InTokens.Size(1));
		auto embedded = Embed_->Forward(InTokens).Reshape(OaMatrixShape{batch, seqLen, kREmbed});
		auto gruOut   = Gru_->Forward(embedded);
		auto last     = OaFnMatrix::Slice(gruOut, 1, seqLen - 1, seqLen);
		last = last.Reshape(OaMatrixShape{batch, kRHidden});
		return Head_->Forward(last);
	}
private:
	OaSharedPtr<OaEmbedding> Embed_;
	OaSharedPtr<OaGru>       Gru_;
	OaSharedPtr<OaLinear>    Head_;
};

TEST(AutogradLeak, GruTapeRealDims) {
	auto makeTokens = [] {
		OaVec<OaU8> x(static_cast<OaI64>(kRBatch) * kRSeq);
		for (OaI64 i = 0; i < x.Size(); ++i) x[i] = static_cast<OaU8>(i % kRVocab);
		return OaFnMatrix::FromBytes(OaSpan<const OaU8>(x.Data(), x.Size()),
			OaMatrixShape{kRBatch, kRSeq}, OaScalarType::UInt8);
	};
	auto makeLabels = [] {
		OaVec<OaU8> y(kRBatch);
		for (OaI32 b = 0; b < kRBatch; ++b) y[b] = static_cast<OaU8>(b % kRVocab);
		return OaFnMatrix::FromBytes(OaSpan<const OaU8>(y.Data(), y.Size()),
			OaMatrixShape{kRBatch}, OaScalarType::UInt8);
	};
	auto  model  = OaMakeSharedPtr<RealGruModel>();
	auto  params = model->AllParameterPtrs();
	auto  opt    = OaMakeUniquePtr<OaAdamW>(params, 0.001F);
	auto  labels = makeLabels();
	printf("\n[GruRealDims] embed=%d hidden=%d seq=%d batch=%d, 80 steps (cap=%llu MiB)\n",
		kREmbed, kRHidden, kRSeq, kRBatch,
		static_cast<unsigned long long>(kVramCapBytes >> 20));

	OaU64 rssBaseline = 0;
	constexpr OaI32 kWarmup = 10;
	constexpr OaI32 kSteps  = 80;
	for (OaI32 step = 0; step < kSteps; ++step) {
		auto x = makeTokens();
		opt->ZeroGrad();
		OaGradientTape tape;
		auto logits = model->Forward(x);
		auto loss   = OaFnLoss::CrossEntropy(logits, labels);
		tape.Backward(loss);
		opt->Step();
		// Force materialization every step, the way the real tutorial does when it
		// reads the loss scalar for its metric — this is what bounds VRAM. The point
		// of the probe is whether HOST RAM stays bounded once VRAM is flushed.
		(void)OaContext::GetDefault().Execute();
		(void)OaContext::GetDefault().Sync();
		GuardRssOrAbort(step);
		if (step == kWarmup) {
			rssBaseline = RssBytesNow();
			PrintAllocBreakdown(step, "baseline");
		} else if (step > kWarmup && (step % 10 == 0 || step == kSteps - 1)) {
			const OaI64 dRss = static_cast<OaI64>(RssBytesNow()) - static_cast<OaI64>(rssBaseline);
			printf("  [step %d] host RSS Δ vs baseline = %+.2f MiB\n",
				step, static_cast<double>(dRss) / (1024.0 * 1024.0));
			PrintAllocBreakdown(step, "         ");
		}
	}
	const OaI64 rssGrowth = static_cast<OaI64>(RssBytesNow()) - static_cast<OaI64>(rssBaseline);
	printf("[GruRealDims] host RSS growth after warmup: %+.2f MiB (VRAM live=%.1f MiB)\n",
		static_cast<double>(rssGrowth) / (1024.0 * 1024.0),
		static_cast<double>(OaComputeEngine::GetGlobal()->Allocator.GetStats().AllocationBytes)
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
	constexpr OaI32 kVoc = 27, kEmb = 128, kHid = 256, kBat = 64;
	auto makeModel = [&] {
		auto wd = OaFnMatrix::GetWeightDtype();
		return std::make_tuple(
			OaMakeSharedPtr<OaEmbedding>(kVoc, kEmb),
			OaMakeSharedPtr<OaGru>(kEmb, kHid, 1),
			OaMakeSharedPtr<OaLinear>(kHid, kVoc),
			wd);
	};
	const OaI32 seqs[] = {5, 10, 20, 40};
	printf("\n[WorkingSetVsSeq] embed=%d hidden=%d batch=%d — VRAM live while tape held:\n",
		kEmb, kHid, kBat);
	for (OaI32 seq : seqs) {
		auto [embed, gru, head, wd] = makeModel();
		embed->Parameters()[0].Data = OaFnMatrix::RandN(OaMatrixShape{kVoc, kEmb}, wd);
		auto setGrad = [](auto& mod) {
			for (auto& p : mod->Parameters()) { p.Data.SetRequiresGrad(true); p.Grad() = p.Data.GradMatrix(); }
		};
		setGrad(embed); setGrad(gru); setGrad(head);

		OaVec<OaU8> xb(static_cast<OaI64>(kBat) * seq);
		for (OaI64 i = 0; i < xb.Size(); ++i) xb[i] = static_cast<OaU8>(i % kVoc);
		OaVec<OaU8> yb(kBat);
		for (OaI32 b = 0; b < kBat; ++b) yb[b] = static_cast<OaU8>(b % kVoc);
		auto labels = OaFnMatrix::FromBytes(OaSpan<const OaU8>(yb.Data(), yb.Size()),
			OaMatrixShape{kBat}, OaScalarType::UInt8);

		OaU64 peak = 0;
		for (OaI32 step = 0; step < 4; ++step) {  // warm a couple then measure
			auto tokens = OaFnMatrix::FromBytes(OaSpan<const OaU8>(xb.Data(), xb.Size()),
				OaMatrixShape{kBat, seq}, OaScalarType::UInt8);
			OaGradientTape tape;
			auto emb = embed->Forward(tokens).Reshape(OaMatrixShape{kBat, seq, kEmb});
			auto g   = gru->Forward(emb);
			auto last = OaFnMatrix::Slice(g, 1, seq - 1, seq).Reshape(OaMatrixShape{kBat, kHid});
			auto logits = head->Forward(last);
			auto loss = OaFnLoss::CrossEntropy(logits, labels);
			tape.Backward(loss);
			(void)OaContext::GetDefault().Execute();
			(void)OaContext::GetDefault().Sync();
			// Measure here: tape (and all saved activations) still alive.
			peak = OaComputeEngine::GetGlobal()->Allocator.GetStats().AllocationBytes;
			GuardRssOrAbort(step);
		}
		// After the loop the last tape is destroyed; read the resting set too.
		(void)OaContext::GetDefault().Execute();
		(void)OaContext::GetDefault().Sync();
		const OaU64 resting = OaComputeEngine::GetGlobal()->Allocator.GetStats().AllocationBytes;
		printf("  seq=%-3d  with-tape live = %7.1f MiB   resting = %6.1f MiB   per-timestep = %5.1f MiB\n",
			seq, static_cast<double>(peak) / (1024.0 * 1024.0),
			static_cast<double>(resting) / (1024.0 * 1024.0),
			static_cast<double>(peak) / (1024.0 * 1024.0) / static_cast<double>(seq));
	}
}
