// oa::VectorQuantizer / oa::ResidualVectorQuantizer implementation.

#include <oa/ml/nn/vq/vq.h>

#include <oa/core/fnMatrix.h>
#include <oa/core/std/format.h>
#include <oa/ml/fnMatrix.h>       // VqAssign / VqEmaUpdate / detach
#include <oa/runtime/executionSession.h>

// ─── oa::VectorQuantizer ──────────────────────────────────────────────────────

oa::VectorQuantizer::VectorQuantizer(const oa::VectorQuantizerConfig& inConfig)
	: config_(inConfig) {
	const oa::I64 K = config_.numCodes;
	const oa::I64 D = config_.codeDim;
	const auto wd = oa::FnMatrix::weightDtype();
	// Provisional random codebook (replaced by seed() once the encoder is warm).
	// EmbedSum starts equal to the codes and ClusterSize at 1 so codebook == m/N
	// holds for the first EMA blend. scale(...,1) makes an INDEPENDENT buffer copy
	// (an aliasing assignment would make the EMA update stomp the codebook).
	codebook_    = oa::FnMatrix::randGlorotUniform(oa::MatrixShape{K, D}, wd);
	embedSum_    = oa::FnMatrix::scale(codebook_, 1.0f);
	clusterSize_ = oa::FnMatrix::ones(oa::MatrixShape{K}, wd);

	// register as persistent buffers so the trained codebook + EMA stats checkpoint.
	// They are NOT gradient params (the codebook moves by EMA). seed() copies into
	// these buffers in-place and emaUpdate writes them in-place, so these handles stay
	// the live state for the module's lifetime.
	registerBuffer("codebook",     codebook_);
	registerBuffer("embed_sum",    embedSum_);
	registerBuffer("cluster_size", clusterSize_);
}

oa::VqResult oa::VectorQuantizer::quantize(const oa::Matrix& inZe) {
	auto vq = oa::FnMatrix::vqAssign(inZe, codebook_);   // vq.idx [N], vq.zq [N,D] (no grad)

	oa::VqResult r;
	r.idx = vq.idx;
	// Straight-through: forward value == z_q, gradient flows to z_e (d/dz_e = 1).
	r.quantized = inZe + oa::FnMatrix::detach(oa::FnMatrix::sub(vq.zq, inZe));
	// Commitment β·MSE(z_e, sg z_q). vq.zq carries no grad → only the encoder moves.
	auto d   = oa::FnMatrix::sub(inZe, vq.zq);
	auto sq  = oa::FnMatrix::mul(d, d);
	auto mse = oa::FnMatrix::scale(oa::FnMatrix::sum(sq), 1.0f / static_cast<float>(sq.numElements()));
	r.commitLoss = oa::FnMatrix::scale(mse, config_.commitBeta);
	return r;
}

void oa::VectorQuantizer::emaUpdate(const oa::Matrix& inZe, const oa::Matrix& inIdx) {
	// emaStep_ varies the dead-code revival row each call (see VqEmaUpdate.slang) so a
	// losing revived code scatters across the batch over steps instead of re-dying.
	oa::FnMatrix::vqEmaUpdate(inZe, inIdx, embedSum_, clusterSize_, codebook_,
		config_.emaDecay, config_.emaEps, config_.deadThresh, emaStep_++, config_.normCode);
}

oa::Matrix oa::VectorQuantizer::lookup(const oa::Matrix& inIdx) const {
	// Gather the assigned code rows. codebook_ is [K, D]; inIdx is [N]. result [N, D].
	// No STE — the pure decode-time inverse of the nearest-code assignment.
	return oa::FnMatrix::gather(codebook_, inIdx);
}

void oa::VectorQuantizer::seed(const oa::Matrix& inLatents) {
	const oa::I64 K    = config_.numCodes;
	const oa::I64 rows = inLatents.size(0);
	OA_ASSERT(rows >= K && "VQ seed needs at least numCodes latent rows");
	OA_ASSERT(K <= 512 && "VQ seed exceeds the GPU topK limit");

	// Preserve the established highest-L2-norm policy entirely on device:
	// [rows,D] -> row squared norms -> top-K row indices -> gathered seed rows.
	// topK resolves equal norms toward the lower row index, making the seed stable.
	auto norms = oa::FnMatrix::reshape(
		oa::FnMatrix::sum(oa::FnMatrix::mul(inLatents, inLatents), 1),
		oa::MatrixShape{rows});
	auto top = oa::FnMatrix::topK(norms, static_cast<oa::I32>(K));
	OA_ASSERT(not top.indices.isEmpty() && "VQ GPU seed topK failed");
	auto seeds = oa::FnMatrix::gather(inLatents, top.indices);
	auto ones = oa::FnMatrix::ones(oa::MatrixShape{K}, inLatents.getDtype());

	// Write IN-PLACE with deferred device copies so the registered buffers() entries
	// stay live and no pending gathered tensor is observed through a host memcpy.
	oa::FnMatrix::castInto(seeds, codebook_);
	oa::FnMatrix::castInto(seeds, embedSum_);
	oa::FnMatrix::castInto(ones, clusterSize_);

	// seed is an explicit initialization boundary. Complete the queued GPU work so
	// a caller cannot discard it with context::clear before the first train step.
	auto& ctx = oa::ExecutionSession::getActive();
	(void)ctx.submitAndWait();
}

// ─── oa::ResidualVectorQuantizer ──────────────────────────────────────────────

oa::ResidualVectorQuantizer::ResidualVectorQuantizer(
	const oa::VectorQuantizerConfig& inConfig, oa::I32 inNumLevels)
	: config_(inConfig)
{
	OA_ASSERT(inNumLevels >= 1 && "RVQ needs at least one level");
	for (oa::I32 q = 0; q < inNumLevels; ++q) {
		auto lvl = oa::makeShared<oa::VectorQuantizer>(inConfig);
		const oa::String name = oa::format("level{}", q);
		registerModule(name.cStr(), lvl);
		levels_.pushBack(lvl);
	}
}

oa::ResidualVqResult oa::ResidualVectorQuantizer::quantize(const oa::Matrix& inZe) {
	oa::ResidualVqResult r;
	oa::Matrix residual = inZe;
	oa::Matrix total;
	for (oa::Usize q = 0; q < levels_.size(); ++q) {
		// Nearest code for the running residual (no STE per level; the gathered code
		// carries no gradient). Accumulate the total and peel it off the residual.
		auto a = oa::FnMatrix::vqAssign(residual, levels_[q]->codebook());
		r.residuals.pushBack(residual);
		r.idx.pushBack(a.idx);
		total    = (q == 0) ? a.zq : oa::FnMatrix::add(total, a.zq);
		residual = oa::FnMatrix::sub(residual, a.zq);
	}
	// Straight-through on the SUM: forward == Σzq, gradient flows to the encoder.
	r.quantized = inZe + oa::FnMatrix::detach(oa::FnMatrix::sub(total, inZe));
	// Commitment β·MSE(z_e, sg Σzq). total is a sum of no-grad codes → grad to z_e only.
	auto d   = oa::FnMatrix::sub(inZe, total);
	auto sq  = oa::FnMatrix::mul(d, d);
	auto mse = oa::FnMatrix::scale(oa::FnMatrix::sum(sq), 1.0f / static_cast<float>(sq.numElements()));
	r.commitLoss = oa::FnMatrix::scale(mse, config_.commitBeta);
	return r;
}

void oa::ResidualVectorQuantizer::emaUpdate(const oa::ResidualVqResult& inResult) {
	for (oa::Usize q = 0; q < levels_.size(); ++q) {
		levels_[q]->emaUpdate(inResult.residuals[q], inResult.idx[q]);
	}
}

oa::Matrix oa::ResidualVectorQuantizer::lookup(const oa::Vector<oa::Matrix>& inIdx) const {
	OA_ASSERT(inIdx.size() >= 1 && "RVQ lookup needs at least one level of token ids");
	OA_ASSERT(inIdx.size() <= levels_.size() && "RVQ lookup given more token levels than codebooks");
	// Sum the per-level gathered codes — the inverse of quantize's Σzq accumulation.
	oa::Matrix total = levels_[0]->lookup(inIdx[0]);
	for (oa::Usize q = 1; q < inIdx.size(); ++q) {
		total = oa::FnMatrix::add(total, levels_[q]->lookup(inIdx[q]));
	}
	return total;
}

void oa::ResidualVectorQuantizer::seed(const oa::Matrix& inLatents) {
	auto& ctx = oa::ExecutionSession::getActive();
	oa::Matrix residual = inLatents;
	for (oa::Usize q = 0; q < levels_.size(); ++q) {
		levels_[q]->seed(residual);   // realizes residual at the host-read boundary
		auto a = oa::FnMatrix::vqAssign(residual, levels_[q]->codebook());
		residual = oa::FnMatrix::sub(residual, a.zq);
		(void)ctx.submitAndWait();   // realize the residual for the next level
	}
}
