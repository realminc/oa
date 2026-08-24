// OA Test — VqAssign kernel + detach (stop-gradient) correctness.
//
// VqAssign: on-GPU per-row nearest-code argmin + gather (VQ-VAE codebook lookup).
// detach:   metadata-only stop-gradient — the primitive the straight-through
//           estimator relies on.
//
// These isolate the two primitives added for the GPU-resident VQ path so a
// regression can't hide behind the end-to-end tutorial. All checks run the real
// vulkan/slang kernel on the GPU and validate against a CPU reference / analytic
// autograd identities.

#include "../../oaTest.h"
#include <oa/ml.h>
#include <oa/ml/nn.h>
#include <oa/ml/autograd.h>
#include <oa/ml/modelFile.h>
#include <oa/runtime/executionSession.h>

#include <algorithm>
#include <cfloat>
#include <cmath>
#include <cstring>
#include <vector>

namespace {

struct Lcg {
	oa::U64 s;
	explicit Lcg(oa::U64 seed) : s(seed ? seed : 0x9e3779b97f4a7c15ULL) {}
	float next(float lo, float hi) {
		s = s * 6364136223846793005ULL + 1442695040888963407ULL;
		oa::U32 hi32 = static_cast<oa::U32>(s >> 33);
		float u = static_cast<float>(hi32) / static_cast<float>(0x7fffffffU);
		return lo + (hi - lo) * u;
	}
};

oa::Matrix makeF32(const std::vector<float>& h, const oa::MatrixShape& shape) {
	return oa::FnMatrix::fromBytes(
		oa::Span<const oa::U8>(reinterpret_cast<const oa::U8*>(h.data()), h.size() * sizeof(float)),
		shape, oa::ScalarType::Float32);
}

oa::Matrix makeI32(const std::vector<oa::I32>& h, const oa::MatrixShape& shape) {
	return oa::FnMatrix::fromBytes(
		oa::Span<const oa::U8>(reinterpret_cast<const oa::U8*>(h.data()), h.size() * sizeof(oa::I32)),
		shape, oa::ScalarType::Int32);
}

void flush() {
	auto& ctx = oa::ExecutionSession::getActive();
	(void)testSubmitAndWait(ctx);
}

std::vector<float> rand(Lcg& rng, oa::I64 n, float lo, float hi) {
	std::vector<float> v(static_cast<size_t>(n));
	for (auto& x : v) x = rng.next(lo, hi);
	return v;
}

// CPU reference: per-row nearest code by squared L2 (strict `<` → lowest k on ties,
// matching the kernel).
void cpuVqAssign(const std::vector<float>& ze, const std::vector<float>& cb,
                 oa::I32 N, oa::I32 D, oa::I32 K,
                 std::vector<oa::I32>& outIdx, std::vector<float>& outZq) {
	outIdx.assign(static_cast<size_t>(N), 0);
	outZq.assign(static_cast<size_t>(N) * D, 0.0f);
	for (oa::I32 n = 0; n < N; ++n) {
		const float* z = &ze[static_cast<size_t>(n) * D];
		float best = FLT_MAX; oa::I32 bk = 0;
		for (oa::I32 k = 0; k < K; ++k) {
			const float* e = &cb[static_cast<size_t>(k) * D];
			float d = 0.0f;
			for (oa::I32 c = 0; c < D; ++c) { const float dd = z[c] - e[c]; d += dd * dd; }
			if (d < best) { best = d; bk = k; }
		}
		outIdx[static_cast<size_t>(n)] = bk;
		std::memcpy(&outZq[static_cast<size_t>(n) * D], &cb[static_cast<size_t>(bk) * D],
			static_cast<size_t>(D) * sizeof(float));
	}
}

} // namespace

// ─── VqAssign kernel correctness ─────────────────────────────────────────────

// Construct z_e from a KNOWN target code + tiny noise against a well-separated
// codebook → the argmin is unambiguous, so we can assert the exact expected index
// (and exercise the grid guard with N not a multiple of 256).
TEST(VqAssign, KnownNearestCode) {
	Lcg rng(1234);
	const oa::I32 N = 300;   // not a multiple of 256 → exercises the dispatch guard
	const oa::I32 D = 16;
	const oa::I32 K = 64;

	// Well-separated codes: spread wide so noise can't flip the nearest neighbour.
	std::vector<float> cb = rand(rng, static_cast<oa::I64>(K) * D, -20.0f, 20.0f);

	std::vector<oa::I32> target(static_cast<size_t>(N));
	std::vector<float> ze(static_cast<size_t>(N) * D);
	for (oa::I32 n = 0; n < N; ++n) {
		const oa::I32 t = static_cast<oa::I32>(rng.next(0.0f, static_cast<float>(K) - 1e-3f));
		target[static_cast<size_t>(n)] = t;
		for (oa::I32 c = 0; c < D; ++c) {
			ze[static_cast<size_t>(n) * D + c] = cb[static_cast<size_t>(t) * D + c] + rng.next(-0.05f, 0.05f);
		}
	}

	auto zeM = makeF32(ze, oa::MatrixShape{N, D});
	auto cbM = makeF32(cb, oa::MatrixShape{K, D});
	auto vq = oa::FnMatrix::vqAssign(zeM, cbM);
	flush();

	ASSERT_EQ(vq.idx.numElements(), static_cast<oa::I64>(N));
	ASSERT_EQ(vq.idx.getDtype(), oa::ScalarType::Int32);
	ASSERT_EQ(vq.zq.numElements(), static_cast<oa::I64>(N) * D);

	const oa::I32* idx = vq.idx.dataAs<const oa::I32>();
	const float* zq  = vq.zq.dataAs<const float>();
	for (oa::I32 n = 0; n < N; ++n) {
		EXPECT_EQ(idx[n], target[static_cast<size_t>(n)]) << "row " << n;
		// zq[n] must equal codebook[idx[n]] exactly (same source bytes).
		const oa::I32 k = idx[n];
		for (oa::I32 c = 0; c < D; ++c) {
			EXPECT_FLOAT_EQ(zq[static_cast<size_t>(n) * D + c], cb[static_cast<size_t>(k) * D + c]);
		}
	}
}

// Pure-random cross-check against the CPU reference. To stay robust to fp near-ties
// (which can legitimately flip the index between CPU and GPU), assert the GPU pick
// is a TRUE argmin: its distance is <= every other code's distance (CPU-evaluated).
TEST(VqAssign, MatchesCpuArgmin) {
	Lcg rng(99);
	const oa::I32 N = 128, D = 32, K = 100;
	auto ze = rand(rng, static_cast<oa::I64>(N) * D, -3.0f, 3.0f);
	auto cb = rand(rng, static_cast<oa::I64>(K) * D, -3.0f, 3.0f);

	auto vq = oa::FnMatrix::vqAssign(makeF32(ze, oa::MatrixShape{N, D}), makeF32(cb, oa::MatrixShape{K, D}));
	flush();
	const oa::I32* idx = vq.idx.dataAs<const oa::I32>();

	for (oa::I32 n = 0; n < N; ++n) {
		const float* z = &ze[static_cast<size_t>(n) * D];
		auto dist = [&](oa::I32 k) {
			const float* e = &cb[static_cast<size_t>(k) * D];
			float d = 0.0f; for (oa::I32 c = 0; c < D; ++c) { float dd = z[c] - e[c]; d += dd * dd; }
			return d;
		};
		const oa::I32 gk = idx[n];
		ASSERT_GE(gk, 0); ASSERT_LT(gk, K);
		const float gd = dist(gk);
		float best = FLT_MAX;
		for (oa::I32 k = 0; k < K; ++k) best = std::min(best, dist(k));
		EXPECT_LE(gd, best + 1e-3f) << "row " << n << " GPU pick not a true argmin";
	}
}

// K=1 degenerate case: every row maps to code 0.
TEST(VqAssign, SingleCode) {
	Lcg rng(7);
	const oa::I32 N = 40, D = 8, K = 1;
	auto ze = rand(rng, static_cast<oa::I64>(N) * D, -1.0f, 1.0f);
	auto cb = rand(rng, static_cast<oa::I64>(K) * D, -1.0f, 1.0f);
	auto vq = oa::FnMatrix::vqAssign(makeF32(ze, oa::MatrixShape{N, D}), makeF32(cb, oa::MatrixShape{K, D}));
	flush();
	const oa::I32* idx = vq.idx.dataAs<const oa::I32>();
	const float* zq  = vq.zq.dataAs<const float>();
	for (oa::I32 n = 0; n < N; ++n) {
		EXPECT_EQ(idx[n], 0);
		for (oa::I32 c = 0; c < D; ++c)
			EXPECT_FLOAT_EQ(zq[static_cast<size_t>(n) * D + c], cb[static_cast<size_t>(c)]);
	}
}

// ─── vqEmaUpdate (EMA codebook + dead-code reinit) ───────────────────────────

// EMA blend matches the CPU reference: N_k ← γN_k+(1-γ)cnt, m_k ← γm_k+(1-γ)Σz,
// codebook ← m/max(N,eps). dead_thresh=0 so no code is revived here.
TEST(VqEmaUpdate, MatchesCpuReference) {
	Lcg rng(2024);
	const oa::I32 N = 50, D = 8, K = 12;
	const float decay = 0.9f, eps = 1e-5f;

	auto ze  = rand(rng, static_cast<oa::I64>(N) * D, -2.0f, 2.0f);
	auto m0  = rand(rng, static_cast<oa::I64>(K) * D, -1.0f, 1.0f);   // initial embed_sum
	std::vector<float> n0(static_cast<size_t>(K));
	for (auto& v : n0) v = rng.next(0.5f, 3.0f);                    // initial cluster_size > 0
	std::vector<oa::I32> idx(static_cast<size_t>(N));
	for (auto& v : idx) v = static_cast<oa::I32>(rng.next(0.0f, static_cast<float>(K) - 1e-3f));

	// CPU reference (dead_thresh = 0 ⇒ no reinit; every N_k ≥ γ·0.5 > 0).
	std::vector<float> mRef = m0, cbRef(static_cast<size_t>(K) * D);
	std::vector<float> nRef = n0;
	for (oa::I32 k = 0; k < K; ++k) {
		oa::I32 cnt = 0; for (oa::I32 n = 0; n < N; ++n) if (idx[static_cast<size_t>(n)] == k) ++cnt;
		const float Nk = decay * n0[static_cast<size_t>(k)] + (1.0f - decay) * static_cast<float>(cnt);
		nRef[static_cast<size_t>(k)] = Nk;
		for (oa::I32 d = 0; d < D; ++d) {
			float s = 0.0f;
			for (oa::I32 n = 0; n < N; ++n) if (idx[static_cast<size_t>(n)] == k) s += ze[static_cast<size_t>(n) * D + d];
			const float mk = decay * m0[static_cast<size_t>(k) * D + d] + (1.0f - decay) * s;
			mRef[static_cast<size_t>(k) * D + d]  = mk;
			cbRef[static_cast<size_t>(k) * D + d] = mk / std::max(Nk, eps);
		}
	}

	auto embedSum    = makeF32(m0, oa::MatrixShape{K, D});
	auto clusterSize = makeF32(n0, oa::MatrixShape{K});
	auto codebook    = oa::FnMatrix::zeros(oa::MatrixShape{K, D}, oa::ScalarType::Float32);
	oa::FnMatrix::vqEmaUpdate(makeF32(ze, oa::MatrixShape{N, D}), makeI32(idx, oa::MatrixShape{N}),
		embedSum, clusterSize, codebook, decay, eps, /*deadThresh=*/0.0f, /*seed=*/0u,
		/*normalize=*/false);
	flush();

	const float* gm  = embedSum.dataAs<const float>();
	const float* gn  = clusterSize.dataAs<const float>();
	const float* gcb = codebook.dataAs<const float>();
	for (oa::I32 k = 0; k < K; ++k) {
		EXPECT_NEAR(gn[k], nRef[static_cast<size_t>(k)], 1e-4f) << "N_k " << k;
		for (oa::I32 d = 0; d < D; ++d) {
			const size_t i = static_cast<size_t>(k) * D + d;
			EXPECT_NEAR(gm[i],  mRef[i],  1e-4f) << "m " << k << "," << d;
			EXPECT_NEAR(gcb[i], cbRef[i], 1e-4f) << "cb " << k << "," << d;
		}
	}
}

// A code with no assignments and an EMA count below the threshold is revived from a
// live encoder row chosen by a (code id, step seed) hash, preventing collapse. The seed
// scatters revival across steps so a losing revived code doesn't re-die on a fixed row.
TEST(VqEmaUpdate, RevivesDeadCode) {
	const oa::I32 N = 16, D = 4, K = 6;
	const float decay = 0.9f, eps = 1e-5f, deadThresh = 1.0f;
	Lcg rng(5);
	auto ze = rand(rng, static_cast<oa::I64>(N) * D, -2.0f, 2.0f);

	// assign every row to code 0 → codes 1..k-1 are unused. Start their cluster_size
	// at 0 so their EMA count stays 0 < deadThresh and they must be revived.
	std::vector<oa::I32> idx(static_cast<size_t>(N), 0);
	std::vector<float> m0(static_cast<size_t>(K) * D, 0.0f);
	std::vector<float> n0(static_cast<size_t>(K), 0.0f);

	auto embedSum    = makeF32(m0, oa::MatrixShape{K, D});
	auto clusterSize = makeF32(n0, oa::MatrixShape{K});
	auto codebook    = oa::FnMatrix::zeros(oa::MatrixShape{K, D}, oa::ScalarType::Float32);
	const oa::U32 seed = 7u;
	oa::FnMatrix::vqEmaUpdate(makeF32(ze, oa::MatrixShape{N, D}), makeI32(idx, oa::MatrixShape{N}),
		embedSum, clusterSize, codebook, decay, eps, deadThresh, seed, /*normalize=*/false);
	flush();

	const float* gcb = codebook.dataAs<const float>();
	const float* gn  = clusterSize.dataAs<const float>();
	for (oa::I32 k = 1; k < K; ++k) {
		// Mirror the shader's revival-row hash (must match VqEmaUpdate.slang exactly).
		oa::U32 h = ((static_cast<oa::U32>(k) + 1u) * 2654435761u) ^ (seed * 2246822519u) ^ (seed >> 15);
		h ^= h >> 13; h *= 3266489917u; h ^= h >> 16;
		const oa::U32 r = h % static_cast<oa::U32>(N);
		EXPECT_NEAR(gn[k], 1.0f, 1e-5f) << "revived count " << k;
		for (oa::I32 d = 0; d < D; ++d) {
			EXPECT_FLOAT_EQ(gcb[static_cast<size_t>(k) * D + d], ze[static_cast<size_t>(r) * D + d])
				<< "revived code " << k << " dim " << d;
		}
	}
}

// ─── oa::VectorQuantizer module (productionized VQ layer) ──────────────────────

// End-to-end module check: seed → quantize (STE forward == codebook[idx],
// finite commitment loss) → emaUpdate (runs, codebook stays finite).
TEST(VectorQuantizer, SeedSelectsHighestNormRowsOnGpu) {
	oa::VectorQuantizerConfig cfg;
	cfg.numCodes = 3; cfg.codeDim = 2;
	oa::VectorQuantizer vq(cfg);
	// Squared norms: 1, 25, 13, 25, 0, 16. Equal 25 ties resolve by row index.
	const std::vector<float> host = {1, 0, 3, 4, 2, 3, -3, -4, 0, 0, 4, 0};
	auto z = makeF32(host, oa::MatrixShape{6, 2});
	vq.seed(z);
	const float* cb = vq.codebook().dataAs<const float>();
	const std::vector<float> expected = {3, 4, -3, -4, 4, 0};
	for (size_t i = 0; i < expected.size(); ++i) EXPECT_FLOAT_EQ(cb[i], expected[i]);
}

TEST(VectorQuantizer, QuantizeAndEma) {
	Lcg rng(2025);
	oa::VectorQuantizerConfig cfg;
	cfg.numCodes = 16; cfg.codeDim = 8; cfg.commitBeta = 0.25f;
	oa::VectorQuantizer vq(cfg);

	const oa::I32 N = 40, D = 8, K = 16;
	vq.seed(makeF32(rand(rng, static_cast<oa::I64>(K + 8) * D, -1.0f, 1.0f), oa::MatrixShape{K + 8, D}));

	auto ze = makeF32(rand(rng, static_cast<oa::I64>(N) * D, -1.0f, 1.0f), oa::MatrixShape{N, D});
	auto r  = vq.quantize(ze);
	flush();

	const oa::I32* idx = r.idx.dataAs<const oa::I32>();
	const float* q   = r.quantized.dataAs<const float>();
	const float* cb  = vq.codebook().dataAs<const float>();
	for (oa::I32 n = 0; n < N; ++n) {
		ASSERT_GE(idx[n], 0); ASSERT_LT(idx[n], K);
		// Straight-through forward value == the gathered code.
		for (oa::I32 d = 0; d < D; ++d)
			EXPECT_NEAR(q[static_cast<size_t>(n) * D + d], cb[static_cast<size_t>(idx[n]) * D + d], 1e-4f);
	}
	EXPECT_TRUE(std::isfinite(r.commitLoss.at(0)));
	EXPECT_GE(r.commitLoss.at(0), 0.0f);

	// EMA update runs and keeps the codebook finite.
	vq.emaUpdate(ze, r.idx);
	flush();
	const float* cb2 = vq.codebook().dataAs<const float>();
	for (oa::I32 i = 0; i < K * D; ++i) EXPECT_TRUE(std::isfinite(cb2[i]));
}

// ─── oa::ResidualVectorQuantizer (RVQ) ─────────────────────────────────────────

// The defining RVQ property: more levels → finer reconstruction of z_e. Each level
// quantizes the residual the previous left, so the summed code approaches z_e.
TEST(ResidualVectorQuantizer, FinerWithMoreLevels) {
	Lcg rng(77);
	const oa::I32 N = 64, D = 8, K = 16;
	oa::VectorQuantizerConfig cfg; cfg.numCodes = K; cfg.codeDim = D;
	auto zeH = rand(rng, static_cast<oa::I64>(N) * D, -1.0f, 1.0f);
	auto ze  = makeF32(zeH, oa::MatrixShape{N, D});

	auto reconErr = [&](oa::I32 levels) {
		oa::ResidualVectorQuantizer rvq(cfg, levels);
		rvq.seed(ze);                          // N=64 >= K=16 rows
		auto r = rvq.quantize(ze);
		flush();
		EXPECT_EQ(static_cast<oa::I32>(r.idx.size()), levels);   // one token per level
		const float* q = r.quantized.dataAs<const float>();    // forward == Σzq
		float e = 0.0f;
		for (oa::I32 i = 0; i < N * D; ++i) { float dd = q[i] - zeH[static_cast<size_t>(i)]; e += dd * dd; }
		return e / static_cast<float>(N * D);
	};

	const float e1 = reconErr(1);
	const float e3 = reconErr(3);
	EXPECT_LT(e3, e1) << "RVQ(3) should reconstruct z_e finer than RVQ(1): " << e3 << " vs " << e1;
}

// checkpoint serialization: each level's three buffers must keep DISTINCT dotted
// names. Regression for the leaf-name collision where codebook/embed_sum/cluster_size
// all serialized under the bare module path "level0" — which silently corrupted
// embed_sum and errored cluster_size on load, breaking training-resume and any
// multi-level RVQ (every level's 3 buffers collided to one name).
TEST(ResidualVectorQuantizer, SerializesDistinctBufferNames) {
	oa::VectorQuantizerConfig cfg; cfg.numCodes = 16; cfg.codeDim = 8;
	oa::ResidualVectorQuantizer rvq(cfg, /*levels=*/2);
	flush();

	oa::ModelFile oam;
	ASSERT_TRUE(rvq.saveTo(testEngine(), oam).isOk());

	// 2 levels × {codebook, embed_sum, cluster_size} = 6 uniquely-named state tensors.
	ASSERT_EQ(oam.stateIndex.size(), static_cast<oa::Usize>(6));
	const char* expected[] = {
		"level0.codebook", "level0.embed_sum", "level0.cluster_size",
		"level1.codebook", "level1.embed_sum", "level1.cluster_size",
	};
	for (const char* leaf : expected) {
		oa::I32 cnt = 0;
		for (const auto& e : oam.stateIndex) if (std::strcmp(e.name, leaf) == 0) ++cnt;
		EXPECT_EQ(cnt, 1) << "state name '" << leaf << "' should appear exactly once";
	}
}

// ─── detach (stop-gradient) semantics ────────────────────────────────────────

// forward value is identical to the input (metadata-only view).
TEST(detach, ForwardEqualsInput) {
	Lcg rng(3);
	auto h = rand(rng, 64, -5.0f, 5.0f);
	auto x = makeF32(h, oa::MatrixShape{8, 8});
	auto d = oa::FnMatrix::detach(x);
	flush();
	const float* px = x.dataAs<const float>();
	const float* pd = d.dataAs<const float>();
	for (oa::I64 i = 0; i < 64; ++i) EXPECT_FLOAT_EQ(pd[i], px[i]);
}

// detach severs the tape: a path routed through detach contributes ZERO gradient.
// out = 2*z + detach(z); dOut/dz = 2 (the detached term drops). With loss = sum(out),
// z.grad must be exactly 2 everywhere — not 3.
TEST(detach, BlocksGradientThroughBranch) {
	Lcg rng(5);
	const oa::I64 n = 32;
	auto h = rand(rng, n, -2.0f, 2.0f);

	oa::GradientTape tape;
	auto z = makeF32(h, oa::MatrixShape{n});
	z.setRequiresGrad(true);

	auto out = oa::FnMatrix::scale(z, 2.0f) + oa::FnMatrix::detach(z);
	auto loss = oa::FnMatrix::sum(out, 0);
	tape.backward(loss);
	flush();

	auto g = z.gradMatrix();
	ASSERT_EQ(g.numElements(), n);
	const float* gp = g.dataAs<const float>();
	for (oa::I64 i = 0; i < n; ++i) EXPECT_NEAR(gp[i], 2.0f, 1e-5f) << "elem " << i;
}

// Straight-through estimator identity: q = z + detach(c - z).
//   forward:  q == c          (the codebook value)
//   backward: dq/dz == 1       (gradient flows straight to the encoder)
// This is exactly the VQ STE; loss = sum(q*g) ⇒ z.grad == g.
TEST(detach, StraightThroughIdentity) {
	Lcg rng(11);
	const oa::I64 n = 48;
	auto zh = rand(rng, n, -3.0f, 3.0f);
	auto ch = rand(rng, n, -3.0f, 3.0f);   // stand-in for the quantized code
	auto gh = rand(rng, n, -1.0f, 1.0f);   // cotangent

	oa::GradientTape tape;
	auto z = makeF32(zh, oa::MatrixShape{n});
	z.setRequiresGrad(true);
	auto c = makeF32(ch, oa::MatrixShape{n});    // constant (no grad) — the code
	auto g = makeF32(gh, oa::MatrixShape{n});

	auto q = z + oa::FnMatrix::detach(c - z);
	auto loss = oa::FnMatrix::sum(q * g, 0);
	tape.backward(loss);
	flush();

	// forward: q == c.
	const float* pq = q.dataAs<const float>();
	for (oa::I64 i = 0; i < n; ++i) EXPECT_NEAR(pq[i], ch[static_cast<size_t>(i)], 1e-4f) << "fwd " << i;

	// backward: z.grad == g (pure passthrough).
	auto zg = z.gradMatrix();
	ASSERT_EQ(zg.numElements(), n);
	const float* pzg = zg.dataAs<const float>();
	for (oa::I64 i = 0; i < n; ++i) EXPECT_NEAR(pzg[i], gh[static_cast<size_t>(i)], 1e-4f) << "grad " << i;
}

// ─── Gather auto-attaches the embedding gradient ─────────────────────────────

// oa::FnMatrix::gather now wires its own GradGather node (table lookup → scatter-add),
// so a learned embedding table trains by simply calling Gather — no hand-wired grad.
// Check the scatter-add formula exactly: with loss = sum(gathered * cotangent), the
// table gradient is the per-row sum of cotangents over the positions that selected it.
// Row never selected → zero grad; row selected twice → its two cotangent rows summed.
TEST(Gather, AutoAttachesTableGradient) {
	const oa::I32 V = 5, D = 3;
	// indices select rows: 1, 3, 1, 0  → row 1 chosen twice, rows 2 and 4 never.
	std::vector<oa::I32> idxH = {1, 3, 1, 0};
	const oa::I32 M = static_cast<oa::I32>(idxH.size());
	std::vector<float> wH(static_cast<size_t>(V) * D, 0.0f);
	for (oa::I64 i = 0; i < V * D; ++i) wH[static_cast<size_t>(i)] = 0.1f * static_cast<float>(i);
	std::vector<float> gH(static_cast<size_t>(M) * D);
	{ Lcg rng(91); for (auto& x : gH) x = rng.next(-1.0f, 1.0f); }

	oa::GradientTape tape;
	auto W = makeF32(wH, oa::MatrixShape{V, D});
	W.setRequiresGrad(true);
	auto idx = makeI32(idxH, oa::MatrixShape{M});
	auto g   = makeF32(gH, oa::MatrixShape{M, D});

	auto gathered = oa::FnMatrix::gather(W, idx);          // [M, D]
	// forward: gathered[m] == W[idx[m]].
	flush();
	const float* pg = gathered.dataAs<const float>();
	for (oa::I32 m = 0; m < M; ++m)
		for (oa::I32 d = 0; d < D; ++d)
			EXPECT_NEAR(pg[static_cast<size_t>(m) * D + d],
			            wH[static_cast<size_t>(idxH[static_cast<size_t>(m)]) * D + d], 1e-5f);

	auto loss = oa::FnMatrix::sum(gathered * g);
	tape.backward(loss);
	flush();

	// expected dW: scatter-add g rows into the selected table rows.
	std::vector<float> wantH(static_cast<size_t>(V) * D, 0.0f);
	for (oa::I32 m = 0; m < M; ++m)
		for (oa::I32 d = 0; d < D; ++d)
			wantH[static_cast<size_t>(idxH[static_cast<size_t>(m)]) * D + d] +=
				gH[static_cast<size_t>(m) * D + d];

	auto dW = W.gradMatrix();
	ASSERT_EQ(dW.numElements(), static_cast<oa::I64>(V) * D);
	const float* pdW = dW.dataAs<const float>();
	for (oa::I64 i = 0; i < V * D; ++i)
		EXPECT_NEAR(pdW[i], wantH[static_cast<size_t>(i)], 1e-4f) << "dW elem " << i;
}

// ─── Batched 3D transpose (last two axes) — the channels-first conv primitive ─

// oa::FnMatrix::transpose on a rank-3 [B,R,C] swaps the last two axes → [B,C,R],
// GPU-native materialize (TransposeBatched kernel), with autograd via GradTranspose.
// forward: out[b,c,r] == in[b,r,c]. backward: the cotangent transposes straight back.
TEST(Transpose, Batched3dForwardAndGrad) {
	const oa::I32 B = 2, R = 3, C = 4;
	std::vector<float> xh(static_cast<size_t>(B) * R * C);
	for (oa::I32 b = 0; b < B; ++b) for (oa::I32 r = 0; r < R; ++r) for (oa::I32 c = 0; c < C; ++c)
		xh[(static_cast<size_t>(b) * R + r) * C + c] = static_cast<float>(b * 100 + r * 10 + c);

	oa::GradientTape tape;
	auto x = makeF32(xh, oa::MatrixShape{B, R, C});
	x.setRequiresGrad(true);
	auto y = oa::FnMatrix::transpose(x, 1, 2);    // [B, C, R]
	flush();

	ASSERT_EQ(y.size(0), B); ASSERT_EQ(y.size(1), C); ASSERT_EQ(y.size(2), R);
	const float* py = y.dataAs<const float>();
	for (oa::I32 b = 0; b < B; ++b) for (oa::I32 c = 0; c < C; ++c) for (oa::I32 r = 0; r < R; ++r)
		EXPECT_FLOAT_EQ(py[(static_cast<size_t>(b) * C + c) * R + r],
		                xh[(static_cast<size_t>(b) * R + r) * C + c]) << "fwd " << b << "," << c << "," << r;

	// backward: loss = sum(y * g) ⇒ x.grad[b,r,c] == g[b,c,r].
	std::vector<float> gh(static_cast<size_t>(B) * C * R);
	{ Lcg rng(123); for (auto& v : gh) v = rng.next(-1.0f, 1.0f); }
	auto g = makeF32(gh, oa::MatrixShape{B, C, R});
	auto loss = oa::FnMatrix::sum(oa::FnMatrix::mul(y, g));
	tape.backward(loss);
	flush();

	auto gx = x.gradMatrix();
	ASSERT_EQ(gx.numElements(), static_cast<oa::I64>(B) * R * C);
	const float* pgx = gx.dataAs<const float>();
	for (oa::I32 b = 0; b < B; ++b) for (oa::I32 r = 0; r < R; ++r) for (oa::I32 c = 0; c < C; ++c)
		EXPECT_NEAR(pgx[(static_cast<size_t>(b) * R + r) * C + c],
		            gh[(static_cast<size_t>(b) * C + c) * R + r], 1e-4f) << "grad " << b << "," << r << "," << c;
}

// ─── Batched matmul (Bmm) — the differentiable-FK enabler ────────────────────

// out[n] = A[n] @ B[n]. forward vs CPU reference; backward (dA = dOut @ Bᵀ) checked
// against the analytic per-batch formula.
TEST(Bmm, BatchedForwardAndGrad) {
	const oa::I32 N = 2, M = 2, K = 3, P = 2;
	std::vector<float> ah(static_cast<size_t>(N) * M * K), bh(static_cast<size_t>(N) * K * P);
	{ Lcg rng(7); for (auto& v : ah) v = rng.next(-1.0f, 1.0f); for (auto& v : bh) v = rng.next(-1.0f, 1.0f); }

	oa::GradientTape tape;
	auto A = makeF32(ah, oa::MatrixShape{N, M, K}); A.setRequiresGrad(true);
	auto Bm = makeF32(bh, oa::MatrixShape{N, K, P}); Bm.setRequiresGrad(true);
	auto Y = oa::FnMatrix::bmm(A, Bm);            // [N, M, P]
	flush();

	ASSERT_EQ(Y.size(0), N); ASSERT_EQ(Y.size(1), M); ASSERT_EQ(Y.size(2), P);
	const float* py = Y.dataAs<const float>();
	for (oa::I32 n = 0; n < N; ++n) for (oa::I32 i = 0; i < M; ++i) for (oa::I32 j = 0; j < P; ++j) {
		float acc = 0.0f;
		for (oa::I32 k = 0; k < K; ++k)
			acc += ah[(static_cast<size_t>(n) * M + i) * K + k] * bh[(static_cast<size_t>(n) * K + k) * P + j];
		EXPECT_NEAR(py[(static_cast<size_t>(n) * M + i) * P + j], acc, 1e-4f) << "fwd " << n << "," << i << "," << j;
	}

	// backward: loss = sum(Y * g) ⇒ dA[n] = g[n] @ B[n]ᵀ.
	std::vector<float> gh2(static_cast<size_t>(N) * M * P);
	{ Lcg rng(9); for (auto& v : gh2) v = rng.next(-1.0f, 1.0f); }
	auto G = makeF32(gh2, oa::MatrixShape{N, M, P});
	auto loss = oa::FnMatrix::sum(oa::FnMatrix::mul(Y, G));
	tape.backward(loss);
	flush();

	auto dA = A.gradMatrix();
	const float* pdA = dA.dataAs<const float>();
	for (oa::I32 n = 0; n < N; ++n) for (oa::I32 i = 0; i < M; ++i) for (oa::I32 k = 0; k < K; ++k) {
		float acc = 0.0f;   // (g @ Bᵀ)[i,k] = sum_j g[i,j] * B[k,j]
		for (oa::I32 j = 0; j < P; ++j)
			acc += gh2[(static_cast<size_t>(n) * M + i) * P + j] * bh[(static_cast<size_t>(n) * K + k) * P + j];
		EXPECT_NEAR(pdA[(static_cast<size_t>(n) * M + i) * K + k], acc, 1e-4f) << "dA " << n << "," << i << "," << k;
	}

	auto dB = Bm.gradMatrix();
	const float* pdB = dB.dataAs<const float>();
	for (oa::I32 n = 0; n < N; ++n) for (oa::I32 k = 0; k < K; ++k) for (oa::I32 j = 0; j < P; ++j) {
		float acc = 0.0f;   // (Aᵀ @ g)[k,j] = sum_i A[i,k] * g[i,j]
		for (oa::I32 i = 0; i < M; ++i)
			acc += ah[(static_cast<size_t>(n) * M + i) * K + k] * gh2[(static_cast<size_t>(n) * M + i) * P + j];
		EXPECT_NEAR(pdB[(static_cast<size_t>(n) * K + k) * P + j], acc, 1e-4f) << "dB " << n << "," << k << "," << j;
	}
}

TEST(BmmNt, BatchedForwardAndGrad) {
	const oa::I32 N = 2, M = 2, K = 3, P = 2;
	std::vector<float> ah(static_cast<size_t>(N) * M * K);
	std::vector<float> bh(static_cast<size_t>(N) * P * K);
	{ Lcg rng(17); for (auto& value : ah) value = rng.next(-1.0F, 1.0F);
		for (auto& value : bh) value = rng.next(-1.0F, 1.0F); }

	oa::GradientTape tape;
	auto A = makeF32(ah, oa::MatrixShape{N, M, K}); A.setRequiresGrad(true);
	auto Bm = makeF32(bh, oa::MatrixShape{N, P, K}); Bm.setRequiresGrad(true);
	auto Y = oa::FnMatrix::bmmNt(A, Bm);
	flush();

	ASSERT_EQ(Y.size(0), N); ASSERT_EQ(Y.size(1), M); ASSERT_EQ(Y.size(2), P);
	const float* py = Y.dataAs<const float>();
	for (oa::I32 n = 0; n < N; ++n) for (oa::I32 row = 0; row < M; ++row)
		for (oa::I32 col = 0; col < P; ++col) {
			float expected = 0.0F;
			for (oa::I32 inner = 0; inner < K; ++inner) {
				expected += ah[(static_cast<size_t>(n) * M + row) * K + inner]
					* bh[(static_cast<size_t>(n) * P + col) * K + inner];
			}
			EXPECT_NEAR(py[(static_cast<size_t>(n) * M + row) * P + col],
				expected, 1e-4F);
		}

	std::vector<float> gh(static_cast<size_t>(N) * M * P);
	{ Lcg rng(19); for (auto& value : gh) value = rng.next(-1.0F, 1.0F); }
	auto G = makeF32(gh, oa::MatrixShape{N, M, P});
	auto loss = oa::FnMatrix::sum(oa::FnMatrix::mul(Y, G));
	tape.backward(loss);
	flush();

	const float* pdA = A.gradMatrix().dataAs<const float>();
	for (oa::I32 n = 0; n < N; ++n) for (oa::I32 row = 0; row < M; ++row)
		for (oa::I32 inner = 0; inner < K; ++inner) {
			float expected = 0.0F;
			for (oa::I32 col = 0; col < P; ++col) {
				expected += gh[(static_cast<size_t>(n) * M + row) * P + col]
					* bh[(static_cast<size_t>(n) * P + col) * K + inner];
			}
			EXPECT_NEAR(pdA[(static_cast<size_t>(n) * M + row) * K + inner],
				expected, 1e-4F);
		}

	const float* pdB = Bm.gradMatrix().dataAs<const float>();
	for (oa::I32 n = 0; n < N; ++n) for (oa::I32 col = 0; col < P; ++col)
		for (oa::I32 inner = 0; inner < K; ++inner) {
			float expected = 0.0F;
			for (oa::I32 row = 0; row < M; ++row) {
				expected += gh[(static_cast<size_t>(n) * M + row) * P + col]
					* ah[(static_cast<size_t>(n) * M + row) * K + inner];
			}
			EXPECT_NEAR(pdB[(static_cast<size_t>(n) * P + col) * K + inner],
				expected, 1e-4F);
		}
}
