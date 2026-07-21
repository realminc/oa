// OA Test — VqAssign kernel + Detach (stop-gradient) correctness.
//
// VqAssign: on-GPU per-row nearest-code argmin + gather (VQ-VAE codebook lookup).
// Detach:   metadata-only stop-gradient — the primitive the straight-through
//           estimator relies on.
//
// These isolate the two primitives added for the GPU-resident VQ path so a
// regression can't hide behind the end-to-end tutorial. All checks run the real
// Vulkan/Slang kernel on the GPU and validate against a CPU reference / analytic
// autograd identities.

#include "../../OaTest.h"
#include <Oa/Ml.h>
#include <Oa/Ml/Nn.h>
#include <Oa/Ml/Autograd.h>
#include <Oa/Ml/Oam.h>
#include <Oa/Runtime/Context.h>

#include <algorithm>
#include <cfloat>
#include <cmath>
#include <cstring>
#include <vector>

namespace {

struct Lcg {
	OaU64 s;
	explicit Lcg(OaU64 seed) : s(seed ? seed : 0x9e3779b97f4a7c15ULL) {}
	float Next(float lo, float hi) {
		s = s * 6364136223846793005ULL + 1442695040888963407ULL;
		OaU32 hi32 = static_cast<OaU32>(s >> 33);
		float u = static_cast<float>(hi32) / static_cast<float>(0x7fffffffU);
		return lo + (hi - lo) * u;
	}
};

OaMatrix MakeF32(const std::vector<float>& h, const OaMatrixShape& shape) {
	return OaFnMatrix::FromBytes(
		OaSpan<const OaU8>(reinterpret_cast<const OaU8*>(h.data()), h.size() * sizeof(float)),
		shape, OaScalarType::Float32);
}

OaMatrix MakeI32(const std::vector<OaI32>& h, const OaMatrixShape& shape) {
	return OaFnMatrix::FromBytes(
		OaSpan<const OaU8>(reinterpret_cast<const OaU8*>(h.data()), h.size() * sizeof(OaI32)),
		shape, OaScalarType::Int32);
}

void Flush() {
	auto& ctx = OaContext::GetDefault();
	(void)ctx.Execute();
	(void)ctx.Sync();
}

std::vector<float> Rand(Lcg& rng, OaI64 n, float lo, float hi) {
	std::vector<float> v(static_cast<size_t>(n));
	for (auto& x : v) x = rng.Next(lo, hi);
	return v;
}

// CPU reference: per-row nearest code by squared L2 (strict `<` → lowest k on ties,
// matching the kernel).
void CpuVqAssign(const std::vector<float>& ze, const std::vector<float>& cb,
                 OaI32 N, OaI32 D, OaI32 K,
                 std::vector<OaI32>& outIdx, std::vector<float>& outZq) {
	outIdx.assign(static_cast<size_t>(N), 0);
	outZq.assign(static_cast<size_t>(N) * D, 0.0f);
	for (OaI32 n = 0; n < N; ++n) {
		const float* z = &ze[static_cast<size_t>(n) * D];
		float best = FLT_MAX; OaI32 bk = 0;
		for (OaI32 k = 0; k < K; ++k) {
			const float* e = &cb[static_cast<size_t>(k) * D];
			float d = 0.0f;
			for (OaI32 c = 0; c < D; ++c) { const float dd = z[c] - e[c]; d += dd * dd; }
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
	const OaI32 N = 300;   // not a multiple of 256 → exercises the dispatch guard
	const OaI32 D = 16;
	const OaI32 K = 64;

	// Well-separated codes: spread wide so noise can't flip the nearest neighbour.
	std::vector<float> cb = Rand(rng, static_cast<OaI64>(K) * D, -20.0f, 20.0f);

	std::vector<OaI32> target(static_cast<size_t>(N));
	std::vector<float> ze(static_cast<size_t>(N) * D);
	for (OaI32 n = 0; n < N; ++n) {
		const OaI32 t = static_cast<OaI32>(rng.Next(0.0f, static_cast<float>(K) - 1e-3f));
		target[static_cast<size_t>(n)] = t;
		for (OaI32 c = 0; c < D; ++c) {
			ze[static_cast<size_t>(n) * D + c] = cb[static_cast<size_t>(t) * D + c] + rng.Next(-0.05f, 0.05f);
		}
	}

	auto zeM = MakeF32(ze, OaMatrixShape{N, D});
	auto cbM = MakeF32(cb, OaMatrixShape{K, D});
	auto vq = OaFnMatrix::VqAssign(zeM, cbM);
	Flush();

	ASSERT_EQ(vq.Idx.NumElements(), static_cast<OaI64>(N));
	ASSERT_EQ(vq.Idx.GetDtype(), OaScalarType::Int32);
	ASSERT_EQ(vq.Zq.NumElements(), static_cast<OaI64>(N) * D);

	const OaI32* idx = vq.Idx.DataAs<const OaI32>();
	const float* zq  = vq.Zq.DataAs<const float>();
	for (OaI32 n = 0; n < N; ++n) {
		EXPECT_EQ(idx[n], target[static_cast<size_t>(n)]) << "row " << n;
		// zq[n] must equal codebook[idx[n]] exactly (same source bytes).
		const OaI32 k = idx[n];
		for (OaI32 c = 0; c < D; ++c) {
			EXPECT_FLOAT_EQ(zq[static_cast<size_t>(n) * D + c], cb[static_cast<size_t>(k) * D + c]);
		}
	}
}

// Pure-random cross-check against the CPU reference. To stay robust to fp near-ties
// (which can legitimately flip the index between CPU and GPU), assert the GPU pick
// is a TRUE argmin: its distance is <= every other code's distance (CPU-evaluated).
TEST(VqAssign, MatchesCpuArgmin) {
	Lcg rng(99);
	const OaI32 N = 128, D = 32, K = 100;
	auto ze = Rand(rng, static_cast<OaI64>(N) * D, -3.0f, 3.0f);
	auto cb = Rand(rng, static_cast<OaI64>(K) * D, -3.0f, 3.0f);

	auto vq = OaFnMatrix::VqAssign(MakeF32(ze, OaMatrixShape{N, D}), MakeF32(cb, OaMatrixShape{K, D}));
	Flush();
	const OaI32* idx = vq.Idx.DataAs<const OaI32>();

	for (OaI32 n = 0; n < N; ++n) {
		const float* z = &ze[static_cast<size_t>(n) * D];
		auto dist = [&](OaI32 k) {
			const float* e = &cb[static_cast<size_t>(k) * D];
			float d = 0.0f; for (OaI32 c = 0; c < D; ++c) { float dd = z[c] - e[c]; d += dd * dd; }
			return d;
		};
		const OaI32 gk = idx[n];
		ASSERT_GE(gk, 0); ASSERT_LT(gk, K);
		const float gd = dist(gk);
		float best = FLT_MAX;
		for (OaI32 k = 0; k < K; ++k) best = std::min(best, dist(k));
		EXPECT_LE(gd, best + 1e-3f) << "row " << n << " GPU pick not a true argmin";
	}
}

// K=1 degenerate case: every row maps to code 0.
TEST(VqAssign, SingleCode) {
	Lcg rng(7);
	const OaI32 N = 40, D = 8, K = 1;
	auto ze = Rand(rng, static_cast<OaI64>(N) * D, -1.0f, 1.0f);
	auto cb = Rand(rng, static_cast<OaI64>(K) * D, -1.0f, 1.0f);
	auto vq = OaFnMatrix::VqAssign(MakeF32(ze, OaMatrixShape{N, D}), MakeF32(cb, OaMatrixShape{K, D}));
	Flush();
	const OaI32* idx = vq.Idx.DataAs<const OaI32>();
	const float* zq  = vq.Zq.DataAs<const float>();
	for (OaI32 n = 0; n < N; ++n) {
		EXPECT_EQ(idx[n], 0);
		for (OaI32 c = 0; c < D; ++c)
			EXPECT_FLOAT_EQ(zq[static_cast<size_t>(n) * D + c], cb[static_cast<size_t>(c)]);
	}
}

// ─── VqEmaUpdate (EMA codebook + dead-code reinit) ───────────────────────────

// EMA blend matches the CPU reference: N_k ← γN_k+(1-γ)cnt, m_k ← γm_k+(1-γ)Σz,
// codebook ← m/max(N,eps). dead_thresh=0 so no code is revived here.
TEST(VqEmaUpdate, MatchesCpuReference) {
	Lcg rng(2024);
	const OaI32 N = 50, D = 8, K = 12;
	const float decay = 0.9f, eps = 1e-5f;

	auto ze  = Rand(rng, static_cast<OaI64>(N) * D, -2.0f, 2.0f);
	auto m0  = Rand(rng, static_cast<OaI64>(K) * D, -1.0f, 1.0f);   // initial embed_sum
	std::vector<float> n0(static_cast<size_t>(K));
	for (auto& v : n0) v = rng.Next(0.5f, 3.0f);                    // initial cluster_size > 0
	std::vector<OaI32> idx(static_cast<size_t>(N));
	for (auto& v : idx) v = static_cast<OaI32>(rng.Next(0.0f, static_cast<float>(K) - 1e-3f));

	// CPU reference (dead_thresh = 0 ⇒ no reinit; every N_k ≥ γ·0.5 > 0).
	std::vector<float> mRef = m0, cbRef(static_cast<size_t>(K) * D);
	std::vector<float> nRef = n0;
	for (OaI32 k = 0; k < K; ++k) {
		OaI32 cnt = 0; for (OaI32 n = 0; n < N; ++n) if (idx[static_cast<size_t>(n)] == k) ++cnt;
		const float Nk = decay * n0[static_cast<size_t>(k)] + (1.0f - decay) * static_cast<float>(cnt);
		nRef[static_cast<size_t>(k)] = Nk;
		for (OaI32 d = 0; d < D; ++d) {
			float s = 0.0f;
			for (OaI32 n = 0; n < N; ++n) if (idx[static_cast<size_t>(n)] == k) s += ze[static_cast<size_t>(n) * D + d];
			const float mk = decay * m0[static_cast<size_t>(k) * D + d] + (1.0f - decay) * s;
			mRef[static_cast<size_t>(k) * D + d]  = mk;
			cbRef[static_cast<size_t>(k) * D + d] = mk / std::max(Nk, eps);
		}
	}

	auto embedSum    = MakeF32(m0, OaMatrixShape{K, D});
	auto clusterSize = MakeF32(n0, OaMatrixShape{K});
	auto codebook    = OaFnMatrix::Zeros(OaMatrixShape{K, D}, OaScalarType::Float32);
	OaFnMatrix::VqEmaUpdate(MakeF32(ze, OaMatrixShape{N, D}), MakeI32(idx, OaMatrixShape{N}),
		embedSum, clusterSize, codebook, decay, eps, /*deadThresh=*/0.0f, /*seed=*/0u,
		/*normalize=*/false);
	Flush();

	const float* gm  = embedSum.DataAs<const float>();
	const float* gn  = clusterSize.DataAs<const float>();
	const float* gcb = codebook.DataAs<const float>();
	for (OaI32 k = 0; k < K; ++k) {
		EXPECT_NEAR(gn[k], nRef[static_cast<size_t>(k)], 1e-4f) << "N_k " << k;
		for (OaI32 d = 0; d < D; ++d) {
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
	const OaI32 N = 16, D = 4, K = 6;
	const float decay = 0.9f, eps = 1e-5f, deadThresh = 1.0f;
	Lcg rng(5);
	auto ze = Rand(rng, static_cast<OaI64>(N) * D, -2.0f, 2.0f);

	// Assign every row to code 0 → codes 1..K-1 are unused. Start their cluster_size
	// at 0 so their EMA count stays 0 < deadThresh and they must be revived.
	std::vector<OaI32> idx(static_cast<size_t>(N), 0);
	std::vector<float> m0(static_cast<size_t>(K) * D, 0.0f);
	std::vector<float> n0(static_cast<size_t>(K), 0.0f);

	auto embedSum    = MakeF32(m0, OaMatrixShape{K, D});
	auto clusterSize = MakeF32(n0, OaMatrixShape{K});
	auto codebook    = OaFnMatrix::Zeros(OaMatrixShape{K, D}, OaScalarType::Float32);
	const OaU32 seed = 7u;
	OaFnMatrix::VqEmaUpdate(MakeF32(ze, OaMatrixShape{N, D}), MakeI32(idx, OaMatrixShape{N}),
		embedSum, clusterSize, codebook, decay, eps, deadThresh, seed, /*normalize=*/false);
	Flush();

	const float* gcb = codebook.DataAs<const float>();
	const float* gn  = clusterSize.DataAs<const float>();
	for (OaI32 k = 1; k < K; ++k) {
		// Mirror the shader's revival-row hash (must match VqEmaUpdate.slang exactly).
		OaU32 h = ((static_cast<OaU32>(k) + 1u) * 2654435761u) ^ (seed * 2246822519u) ^ (seed >> 15);
		h ^= h >> 13; h *= 3266489917u; h ^= h >> 16;
		const OaU32 r = h % static_cast<OaU32>(N);
		EXPECT_NEAR(gn[k], 1.0f, 1e-5f) << "revived count " << k;
		for (OaI32 d = 0; d < D; ++d) {
			EXPECT_FLOAT_EQ(gcb[static_cast<size_t>(k) * D + d], ze[static_cast<size_t>(r) * D + d])
				<< "revived code " << k << " dim " << d;
		}
	}
}

// ─── OaVectorQuantizer module (productionized VQ layer) ──────────────────────

// End-to-end module check: Seed → Quantize (STE forward == codebook[idx],
// finite commitment loss) → EmaUpdate (runs, codebook stays finite).
TEST(VectorQuantizer, SeedSelectsHighestNormRowsOnGpu) {
	OaVectorQuantizerConfig cfg;
	cfg.NumCodes = 3; cfg.CodeDim = 2;
	OaVectorQuantizer vq(cfg);
	// Squared norms: 1, 25, 13, 25, 0, 16. Equal 25 ties resolve by row index.
	const std::vector<float> host = {1, 0, 3, 4, 2, 3, -3, -4, 0, 0, 4, 0};
	auto z = MakeF32(host, OaMatrixShape{6, 2});
	vq.Seed(z);
	const float* cb = vq.Codebook().DataAs<const float>();
	const std::vector<float> expected = {3, 4, -3, -4, 4, 0};
	for (size_t i = 0; i < expected.size(); ++i) EXPECT_FLOAT_EQ(cb[i], expected[i]);
}

TEST(VectorQuantizer, QuantizeAndEma) {
	Lcg rng(2025);
	OaVectorQuantizerConfig cfg;
	cfg.NumCodes = 16; cfg.CodeDim = 8; cfg.CommitBeta = 0.25f;
	OaVectorQuantizer vq(cfg);

	const OaI32 N = 40, D = 8, K = 16;
	vq.Seed(MakeF32(Rand(rng, static_cast<OaI64>(K + 8) * D, -1.0f, 1.0f), OaMatrixShape{K + 8, D}));

	auto ze = MakeF32(Rand(rng, static_cast<OaI64>(N) * D, -1.0f, 1.0f), OaMatrixShape{N, D});
	auto r  = vq.Quantize(ze);
	Flush();

	const OaI32* idx = r.Idx.DataAs<const OaI32>();
	const float* q   = r.Quantized.DataAs<const float>();
	const float* cb  = vq.Codebook().DataAs<const float>();
	for (OaI32 n = 0; n < N; ++n) {
		ASSERT_GE(idx[n], 0); ASSERT_LT(idx[n], K);
		// Straight-through forward value == the gathered code.
		for (OaI32 d = 0; d < D; ++d)
			EXPECT_NEAR(q[static_cast<size_t>(n) * D + d], cb[static_cast<size_t>(idx[n]) * D + d], 1e-4f);
	}
	EXPECT_TRUE(std::isfinite(r.CommitLoss.At(0)));
	EXPECT_GE(r.CommitLoss.At(0), 0.0f);

	// EMA update runs and keeps the codebook finite.
	vq.EmaUpdate(ze, r.Idx);
	Flush();
	const float* cb2 = vq.Codebook().DataAs<const float>();
	for (OaI32 i = 0; i < K * D; ++i) EXPECT_TRUE(std::isfinite(cb2[i]));
}

// ─── OaResidualVectorQuantizer (RVQ) ─────────────────────────────────────────

// The defining RVQ property: more levels → finer reconstruction of z_e. Each level
// quantizes the residual the previous left, so the summed code approaches z_e.
TEST(ResidualVectorQuantizer, FinerWithMoreLevels) {
	Lcg rng(77);
	const OaI32 N = 64, D = 8, K = 16;
	OaVectorQuantizerConfig cfg; cfg.NumCodes = K; cfg.CodeDim = D;
	auto zeH = Rand(rng, static_cast<OaI64>(N) * D, -1.0f, 1.0f);
	auto ze  = MakeF32(zeH, OaMatrixShape{N, D});

	auto reconErr = [&](OaI32 levels) {
		OaResidualVectorQuantizer rvq(cfg, levels);
		rvq.Seed(ze);                          // N=64 >= K=16 rows
		auto r = rvq.Quantize(ze);
		Flush();
		EXPECT_EQ(static_cast<OaI32>(r.Idx.Size()), levels);   // one token per level
		const float* q = r.Quantized.DataAs<const float>();    // forward == Σzq
		float e = 0.0f;
		for (OaI32 i = 0; i < N * D; ++i) { float dd = q[i] - zeH[static_cast<size_t>(i)]; e += dd * dd; }
		return e / static_cast<float>(N * D);
	};

	const float e1 = reconErr(1);
	const float e3 = reconErr(3);
	EXPECT_LT(e3, e1) << "RVQ(3) should reconstruct z_e finer than RVQ(1): " << e3 << " vs " << e1;
}

// Checkpoint serialization: each level's three buffers must keep DISTINCT dotted
// names. Regression for the leaf-name collision where codebook/embed_sum/cluster_size
// all serialized under the bare module path "level0" — which silently corrupted
// embed_sum and errored cluster_size on load, breaking training-resume and any
// multi-level RVQ (every level's 3 buffers collided to one name).
TEST(ResidualVectorQuantizer, SerializesDistinctBufferNames) {
	OaVectorQuantizerConfig cfg; cfg.NumCodes = 16; cfg.CodeDim = 8;
	OaResidualVectorQuantizer rvq(cfg, /*levels=*/2);
	Flush();

	OamModel oam;
	ASSERT_TRUE(rvq.SaveTo(oam).IsOk());

	// 2 levels × {codebook, embed_sum, cluster_size} = 6 uniquely-named state tensors.
	ASSERT_EQ(oam.StateIndex.Size(), static_cast<OaUsize>(6));
	const char* expected[] = {
		"level0.codebook", "level0.embed_sum", "level0.cluster_size",
		"level1.codebook", "level1.embed_sum", "level1.cluster_size",
	};
	for (const char* leaf : expected) {
		OaI32 cnt = 0;
		for (const auto& e : oam.StateIndex) if (std::strcmp(e.Name, leaf) == 0) ++cnt;
		EXPECT_EQ(cnt, 1) << "state name '" << leaf << "' should appear exactly once";
	}
}

// ─── Detach (stop-gradient) semantics ────────────────────────────────────────

// Forward value is identical to the input (metadata-only view).
TEST(Detach, ForwardEqualsInput) {
	Lcg rng(3);
	auto h = Rand(rng, 64, -5.0f, 5.0f);
	auto x = MakeF32(h, OaMatrixShape{8, 8});
	auto d = OaFnMatrix::Detach(x);
	Flush();
	const float* px = x.DataAs<const float>();
	const float* pd = d.DataAs<const float>();
	for (OaI64 i = 0; i < 64; ++i) EXPECT_FLOAT_EQ(pd[i], px[i]);
}

// Detach severs the tape: a path routed through Detach contributes ZERO gradient.
// out = 2*z + Detach(z); dOut/dz = 2 (the detached term drops). With loss = sum(out),
// z.grad must be exactly 2 everywhere — not 3.
TEST(Detach, BlocksGradientThroughBranch) {
	Lcg rng(5);
	const OaI64 n = 32;
	auto h = Rand(rng, n, -2.0f, 2.0f);

	OaGradientTape tape;
	auto z = MakeF32(h, OaMatrixShape{n});
	z.SetRequiresGrad(true);

	auto out = OaFnMatrix::Scale(z, 2.0f) + OaFnMatrix::Detach(z);
	auto loss = OaFnMatrix::Sum(out, 0);
	tape.Backward(loss);
	Flush();

	auto g = z.GradMatrix();
	ASSERT_EQ(g.NumElements(), n);
	const float* gp = g.DataAs<const float>();
	for (OaI64 i = 0; i < n; ++i) EXPECT_NEAR(gp[i], 2.0f, 1e-5f) << "elem " << i;
}

// Straight-through estimator identity: q = z + Detach(c - z).
//   forward:  q == c          (the codebook value)
//   backward: dq/dz == 1       (gradient flows straight to the encoder)
// This is exactly the VQ STE; loss = sum(q*g) ⇒ z.grad == g.
TEST(Detach, StraightThroughIdentity) {
	Lcg rng(11);
	const OaI64 n = 48;
	auto zh = Rand(rng, n, -3.0f, 3.0f);
	auto ch = Rand(rng, n, -3.0f, 3.0f);   // stand-in for the quantized code
	auto gh = Rand(rng, n, -1.0f, 1.0f);   // cotangent

	OaGradientTape tape;
	auto z = MakeF32(zh, OaMatrixShape{n});
	z.SetRequiresGrad(true);
	auto c = MakeF32(ch, OaMatrixShape{n});    // constant (no grad) — the code
	auto g = MakeF32(gh, OaMatrixShape{n});

	auto q = z + OaFnMatrix::Detach(c - z);
	auto loss = OaFnMatrix::Sum(q * g, 0);
	tape.Backward(loss);
	Flush();

	// Forward: q == c.
	const float* pq = q.DataAs<const float>();
	for (OaI64 i = 0; i < n; ++i) EXPECT_NEAR(pq[i], ch[static_cast<size_t>(i)], 1e-4f) << "fwd " << i;

	// Backward: z.grad == g (pure passthrough).
	auto zg = z.GradMatrix();
	ASSERT_EQ(zg.NumElements(), n);
	const float* pzg = zg.DataAs<const float>();
	for (OaI64 i = 0; i < n; ++i) EXPECT_NEAR(pzg[i], gh[static_cast<size_t>(i)], 1e-4f) << "grad " << i;
}

// ─── Gather auto-attaches the embedding gradient ─────────────────────────────

// OaFnMatrix::Gather now wires its own OaGradGather node (table lookup → scatter-add),
// so a learned embedding table trains by simply calling Gather — no hand-wired grad.
// Check the scatter-add formula exactly: with loss = sum(gathered * cotangent), the
// table gradient is the per-row sum of cotangents over the positions that selected it.
// Row never selected → zero grad; row selected twice → its two cotangent rows summed.
TEST(Gather, AutoAttachesTableGradient) {
	const OaI32 V = 5, D = 3;
	// indices select rows: 1, 3, 1, 0  → row 1 chosen twice, rows 2 and 4 never.
	std::vector<OaI32> idxH = {1, 3, 1, 0};
	const OaI32 M = static_cast<OaI32>(idxH.size());
	std::vector<float> wH(static_cast<size_t>(V) * D, 0.0f);
	for (OaI64 i = 0; i < V * D; ++i) wH[static_cast<size_t>(i)] = 0.1f * static_cast<float>(i);
	std::vector<float> gH(static_cast<size_t>(M) * D);
	{ Lcg rng(91); for (auto& x : gH) x = rng.Next(-1.0f, 1.0f); }

	OaGradientTape tape;
	auto W = MakeF32(wH, OaMatrixShape{V, D});
	W.SetRequiresGrad(true);
	auto idx = MakeI32(idxH, OaMatrixShape{M});
	auto g   = MakeF32(gH, OaMatrixShape{M, D});

	auto gathered = OaFnMatrix::Gather(W, idx);          // [M, D]
	// Forward: gathered[m] == W[idx[m]].
	Flush();
	const float* pg = gathered.DataAs<const float>();
	for (OaI32 m = 0; m < M; ++m)
		for (OaI32 d = 0; d < D; ++d)
			EXPECT_NEAR(pg[static_cast<size_t>(m) * D + d],
			            wH[static_cast<size_t>(idxH[static_cast<size_t>(m)]) * D + d], 1e-5f);

	auto loss = OaFnMatrix::Sum(gathered * g, 0);
	tape.Backward(loss);
	Flush();

	// Expected dW: scatter-add g rows into the selected table rows.
	std::vector<float> wantH(static_cast<size_t>(V) * D, 0.0f);
	for (OaI32 m = 0; m < M; ++m)
		for (OaI32 d = 0; d < D; ++d)
			wantH[static_cast<size_t>(idxH[static_cast<size_t>(m)]) * D + d] +=
				gH[static_cast<size_t>(m) * D + d];

	auto dW = W.GradMatrix();
	ASSERT_EQ(dW.NumElements(), static_cast<OaI64>(V) * D);
	const float* pdW = dW.DataAs<const float>();
	for (OaI64 i = 0; i < V * D; ++i)
		EXPECT_NEAR(pdW[i], wantH[static_cast<size_t>(i)], 1e-4f) << "dW elem " << i;
}

// ─── Batched 3D transpose (last two axes) — the channels-first conv primitive ─

// OaFnMatrix::Transpose on a rank-3 [B,R,C] swaps the last two axes → [B,C,R],
// GPU-native materialize (TransposeBatched kernel), with autograd via OaGradTranspose.
// Forward: out[b,c,r] == in[b,r,c]. Backward: the cotangent transposes straight back.
TEST(Transpose, Batched3dForwardAndGrad) {
	const OaI32 B = 2, R = 3, C = 4;
	std::vector<float> xh(static_cast<size_t>(B) * R * C);
	for (OaI32 b = 0; b < B; ++b) for (OaI32 r = 0; r < R; ++r) for (OaI32 c = 0; c < C; ++c)
		xh[(static_cast<size_t>(b) * R + r) * C + c] = static_cast<float>(b * 100 + r * 10 + c);

	OaGradientTape tape;
	auto x = MakeF32(xh, OaMatrixShape{B, R, C});
	x.SetRequiresGrad(true);
	auto y = OaFnMatrix::Transpose(x, 1, 2);    // [B, C, R]
	Flush();

	ASSERT_EQ(y.Size(0), B); ASSERT_EQ(y.Size(1), C); ASSERT_EQ(y.Size(2), R);
	const float* py = y.DataAs<const float>();
	for (OaI32 b = 0; b < B; ++b) for (OaI32 c = 0; c < C; ++c) for (OaI32 r = 0; r < R; ++r)
		EXPECT_FLOAT_EQ(py[(static_cast<size_t>(b) * C + c) * R + r],
		                xh[(static_cast<size_t>(b) * R + r) * C + c]) << "fwd " << b << "," << c << "," << r;

	// Backward: loss = sum(y * g) ⇒ x.grad[b,r,c] == g[b,c,r].
	std::vector<float> gh(static_cast<size_t>(B) * C * R);
	{ Lcg rng(123); for (auto& v : gh) v = rng.Next(-1.0f, 1.0f); }
	auto g = MakeF32(gh, OaMatrixShape{B, C, R});
	auto loss = OaFnMatrix::Sum(OaFnMatrix::Mul(y, g));
	tape.Backward(loss);
	Flush();

	auto gx = x.GradMatrix();
	ASSERT_EQ(gx.NumElements(), static_cast<OaI64>(B) * R * C);
	const float* pgx = gx.DataAs<const float>();
	for (OaI32 b = 0; b < B; ++b) for (OaI32 r = 0; r < R; ++r) for (OaI32 c = 0; c < C; ++c)
		EXPECT_NEAR(pgx[(static_cast<size_t>(b) * R + r) * C + c],
		            gh[(static_cast<size_t>(b) * C + c) * R + r], 1e-4f) << "grad " << b << "," << r << "," << c;
}

// ─── Batched matmul (Bmm) — the differentiable-FK enabler ────────────────────

// out[n] = A[n] @ B[n]. Forward vs CPU reference; backward (dA = dOut @ Bᵀ) checked
// against the analytic per-batch formula.
TEST(Bmm, BatchedForwardAndGrad) {
	const OaI32 N = 2, M = 2, K = 3, P = 2;
	std::vector<float> ah(static_cast<size_t>(N) * M * K), bh(static_cast<size_t>(N) * K * P);
	{ Lcg rng(7); for (auto& v : ah) v = rng.Next(-1.0f, 1.0f); for (auto& v : bh) v = rng.Next(-1.0f, 1.0f); }

	OaGradientTape tape;
	auto A = MakeF32(ah, OaMatrixShape{N, M, K}); A.SetRequiresGrad(true);
	auto Bm = MakeF32(bh, OaMatrixShape{N, K, P}); Bm.SetRequiresGrad(true);
	auto Y = OaFnMatrix::Bmm(A, Bm);            // [N, M, P]
	Flush();

	ASSERT_EQ(Y.Size(0), N); ASSERT_EQ(Y.Size(1), M); ASSERT_EQ(Y.Size(2), P);
	const float* py = Y.DataAs<const float>();
	for (OaI32 n = 0; n < N; ++n) for (OaI32 i = 0; i < M; ++i) for (OaI32 j = 0; j < P; ++j) {
		float acc = 0.0f;
		for (OaI32 k = 0; k < K; ++k)
			acc += ah[(static_cast<size_t>(n) * M + i) * K + k] * bh[(static_cast<size_t>(n) * K + k) * P + j];
		EXPECT_NEAR(py[(static_cast<size_t>(n) * M + i) * P + j], acc, 1e-4f) << "fwd " << n << "," << i << "," << j;
	}

	// Backward: loss = sum(Y * g) ⇒ dA[n] = g[n] @ B[n]ᵀ.
	std::vector<float> gh2(static_cast<size_t>(N) * M * P);
	{ Lcg rng(9); for (auto& v : gh2) v = rng.Next(-1.0f, 1.0f); }
	auto G = MakeF32(gh2, OaMatrixShape{N, M, P});
	auto loss = OaFnMatrix::Sum(OaFnMatrix::Mul(Y, G));
	tape.Backward(loss);
	Flush();

	auto dA = A.GradMatrix();
	const float* pdA = dA.DataAs<const float>();
	for (OaI32 n = 0; n < N; ++n) for (OaI32 i = 0; i < M; ++i) for (OaI32 k = 0; k < K; ++k) {
		float acc = 0.0f;   // (g @ Bᵀ)[i,k] = sum_j g[i,j] * B[k,j]
		for (OaI32 j = 0; j < P; ++j)
			acc += gh2[(static_cast<size_t>(n) * M + i) * P + j] * bh[(static_cast<size_t>(n) * K + k) * P + j];
		EXPECT_NEAR(pdA[(static_cast<size_t>(n) * M + i) * K + k], acc, 1e-4f) << "dA " << n << "," << i << "," << k;
	}

	auto dB = Bm.GradMatrix();
	const float* pdB = dB.DataAs<const float>();
	for (OaI32 n = 0; n < N; ++n) for (OaI32 k = 0; k < K; ++k) for (OaI32 j = 0; j < P; ++j) {
		float acc = 0.0f;   // (Aᵀ @ g)[k,j] = sum_i A[i,k] * g[i,j]
		for (OaI32 i = 0; i < M; ++i)
			acc += ah[(static_cast<size_t>(n) * M + i) * K + k] * gh2[(static_cast<size_t>(n) * M + i) * P + j];
		EXPECT_NEAR(pdB[(static_cast<size_t>(n) * K + k) * P + j], acc, 1e-4f) << "dB " << n << "," << k << "," << j;
	}
}
