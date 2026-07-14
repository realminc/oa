// OaVectorQuantizer / OaResidualVectorQuantizer implementation.

#include "Vq.h"

#include <Oa/Core/FnMatrix.h>
#include <Oa/Ml/FnMatrix.h>       // VqAssign / VqEmaUpdate / Detach
#include <Oa/Ml/Autograd.h>
#include <Oa/Runtime/Context.h>

#include <utility>

// ─── OaVectorQuantizer ──────────────────────────────────────────────────────

OaVectorQuantizer::OaVectorQuantizer(const OaVectorQuantizerConfig& InConfig)
	: Config_(InConfig) {
	const OaI64 K = Config_.NumCodes;
	const OaI64 D = Config_.CodeDim;
	const auto wd = OaFnMatrix::GetWeightDtype();
	// Provisional random codebook (replaced by Seed() once the encoder is warm).
	// EmbedSum starts equal to the codes and ClusterSize at 1 so codebook == m/N
	// holds for the first EMA blend. Scale(...,1) makes an INDEPENDENT buffer copy
	// (an aliasing assignment would make the EMA update stomp the codebook).
	Codebook_    = OaFnMatrix::RandGlorotUniform(OaMatrixShape{K, D}, wd);
	EmbedSum_    = OaFnMatrix::Scale(Codebook_, 1.0f);
	ClusterSize_ = OaFnMatrix::Ones(OaMatrixShape{K}, wd);

	// Register as persistent buffers so the trained codebook + EMA stats checkpoint.
	// They are NOT gradient params (the codebook moves by EMA). Seed() copies into
	// these buffers in-place and EmaUpdate writes them in-place, so these handles stay
	// the live state for the module's lifetime.
	RegisterBuffer("codebook",     Codebook_);
	RegisterBuffer("embed_sum",    EmbedSum_);
	RegisterBuffer("cluster_size", ClusterSize_);
}

OaVqResult OaVectorQuantizer::Quantize(const OaMatrix& InZe) {
	auto vq = OaFnMatrix::VqAssign(InZe, Codebook_);   // vq.Idx [N], vq.Zq [N,D] (no grad)

	OaVqResult r;
	r.Idx = vq.Idx;
	// Straight-through: forward value == z_q, gradient flows to z_e (d/dz_e = 1).
	r.Quantized = InZe + OaFnMatrix::Detach(OaFnMatrix::Sub(vq.Zq, InZe));
	// Commitment β·MSE(z_e, sg z_q). vq.Zq carries no grad → only the encoder moves.
	auto d   = OaFnMatrix::Sub(InZe, vq.Zq);
	auto sq  = OaFnMatrix::Mul(d, d);
	auto mse = OaFnMatrix::Scale(OaFnMatrix::Sum(sq), 1.0f / static_cast<float>(sq.NumElements()));
	r.CommitLoss = OaFnMatrix::Scale(mse, Config_.CommitBeta);
	return r;
}

void OaVectorQuantizer::EmaUpdate(const OaMatrix& InZe, const OaMatrix& InIdx) {
	// EmaStep_ varies the dead-code revival row each call (see VqEmaUpdate.slang) so a
	// losing revived code scatters across the batch over steps instead of re-dying.
	OaFnMatrix::VqEmaUpdate(InZe, InIdx, EmbedSum_, ClusterSize_, Codebook_,
		Config_.EmaDecay, Config_.EmaEps, Config_.DeadThresh, EmaStep_++, Config_.NormCode);
}

OaMatrix OaVectorQuantizer::Lookup(const OaMatrix& InIdx) const {
	// Gather the assigned code rows. Codebook_ is [K, D]; InIdx is [N]. Result [N, D].
	// No STE — the pure decode-time inverse of the nearest-code assignment.
	return OaFnMatrix::Gather(Codebook_, InIdx);
}

void OaVectorQuantizer::Seed(const OaMatrix& InLatents) {
	const OaI64 K    = Config_.NumCodes;
	const OaI64 D    = Config_.CodeDim;
	const OaI64 rows = InLatents.Size(0);
	OA_ASSERT(rows >= K && "VQ seed needs at least NumCodes latent rows");
	OA_ASSERT(K <= 512 && "VQ seed exceeds the GPU TopK limit");

	// Preserve the established highest-L2-norm policy entirely on device:
	// [rows,D] -> row squared norms -> top-K row indices -> gathered seed rows.
	// TopK resolves equal norms toward the lower row index, making the seed stable.
	auto norms = OaFnMatrix::Reshape(
		OaFnMatrix::Sum(OaFnMatrix::Mul(InLatents, InLatents), 1),
		OaMatrixShape{rows});
	auto top = OaFnMatrix::TopK(norms, static_cast<OaI32>(K));
	OA_ASSERT(not top.Indices.IsEmpty() && "VQ GPU seed TopK failed");
	auto seeds = OaFnMatrix::Gather(InLatents, top.Indices);
	auto ones = OaFnMatrix::Ones(OaMatrixShape{K}, InLatents.GetDtype());

	// Write IN-PLACE with deferred device copies so the registered Buffers() entries
	// stay live and no pending gathered tensor is observed through a host memcpy.
	OaFnMatrix::CastInto(seeds, Codebook_);
	OaFnMatrix::CastInto(seeds, EmbedSum_);
	OaFnMatrix::CastInto(ones, ClusterSize_);

	// Seed is an explicit initialization boundary. Complete the queued GPU work so
	// a caller cannot discard it with Context::Clear before the first train step.
	auto& ctx = OaContext::GetDefault();
	(void)ctx.Execute(); (void)ctx.Sync();
}

// ─── OaResidualVectorQuantizer ──────────────────────────────────────────────

OaResidualVectorQuantizer::OaResidualVectorQuantizer(
	const OaVectorQuantizerConfig& InConfig, OaI32 InNumLevels)
	: Config_(InConfig)
{
	OA_ASSERT(InNumLevels >= 1 && "RVQ needs at least one level");
	for (OaI32 q = 0; q < InNumLevels; ++q) {
		auto lvl = OaMakeSharedPtr<OaVectorQuantizer>(InConfig);
		char name[16];
		std::snprintf(name, sizeof(name), "level%d", q);
		RegisterModule(name, lvl);
		Levels_.PushBack(lvl);
	}
}

OaResidualVqResult OaResidualVectorQuantizer::Quantize(const OaMatrix& InZe) {
	OaResidualVqResult r;
	OaMatrix residual = InZe;
	OaMatrix total;
	for (OaUsize q = 0; q < Levels_.Size(); ++q) {
		// Nearest code for the running residual (no STE per level; the gathered code
		// carries no gradient). Accumulate the total and peel it off the residual.
		auto a = OaFnMatrix::VqAssign(residual, Levels_[q]->Codebook());
		r.Residuals.PushBack(residual);
		r.Idx.PushBack(a.Idx);
		total    = (q == 0) ? a.Zq : OaFnMatrix::Add(total, a.Zq);
		residual = OaFnMatrix::Sub(residual, a.Zq);
	}
	// Straight-through on the SUM: forward == Σzq, gradient flows to the encoder.
	r.Quantized = InZe + OaFnMatrix::Detach(OaFnMatrix::Sub(total, InZe));
	// Commitment β·MSE(z_e, sg Σzq). total is a sum of no-grad codes → grad to z_e only.
	auto d   = OaFnMatrix::Sub(InZe, total);
	auto sq  = OaFnMatrix::Mul(d, d);
	auto mse = OaFnMatrix::Scale(OaFnMatrix::Sum(sq), 1.0f / static_cast<float>(sq.NumElements()));
	r.CommitLoss = OaFnMatrix::Scale(mse, Config_.CommitBeta);
	return r;
}

void OaResidualVectorQuantizer::EmaUpdate(const OaResidualVqResult& InResult) {
	for (OaUsize q = 0; q < Levels_.Size(); ++q) {
		Levels_[q]->EmaUpdate(InResult.Residuals[q], InResult.Idx[q]);
	}
}

OaMatrix OaResidualVectorQuantizer::Lookup(const OaVec<OaMatrix>& InIdx) const {
	OA_ASSERT(InIdx.Size() >= 1 && "RVQ Lookup needs at least one level of token ids");
	OA_ASSERT(InIdx.Size() <= Levels_.Size() && "RVQ Lookup given more token levels than codebooks");
	// Sum the per-level gathered codes — the inverse of Quantize's Σzq accumulation.
	OaMatrix total = Levels_[0]->Lookup(InIdx[0]);
	for (OaUsize q = 1; q < InIdx.Size(); ++q) {
		total = OaFnMatrix::Add(total, Levels_[q]->Lookup(InIdx[q]));
	}
	return total;
}

void OaResidualVectorQuantizer::Seed(const OaMatrix& InLatents) {
	auto& ctx = OaContext::GetDefault();
	OaMatrix residual = InLatents;
	for (OaUsize q = 0; q < Levels_.Size(); ++q) {
		Levels_[q]->Seed(residual);   // realizes residual (Execute/Sync inside)
		auto a = OaFnMatrix::VqAssign(residual, Levels_[q]->Codebook());
		residual = OaFnMatrix::Sub(residual, a.Zq);
		(void)ctx.Execute(); (void)ctx.Sync();   // realize the residual for the next level
	}
}
