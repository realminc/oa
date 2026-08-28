// oa::AlmTokenizerAg — faithful temporal-Conv1d VQ-VAE (see AlmTokenizerAg.h).

#include <ml/nn/alm/almTokenizerAg.h>

#include <oa/ml/nn.h>            // oa::Conv1d, oa::ConvTranspose1d
#include <oa/core/fnMatrix.h>
#include <oa/ml/fnMatrix.h>      // oa::FnMatrix::channelNorm, Conv1dRelu
#include <oa/ml/autograd.h>      // oa::FnAutograd::IsEnabled
#include <oa/runtime/executionSession.h>

#include <cmath>
#include <cstdio>
#include <vector>

// oa::Conv1d / oa::ConvTranspose1d init their weight with a DEFERRED uniform Rand
// (all-positive, large) — a poor init that biases ReLU activations positive and is a
// prime suspect for the conv-VQ divergence. Overwrite with Glorot-uniform in-place
// AFTER the deferred Rand has run. weight is [rows, cols, K]; fan_in=cols*K,
// fan_out=rows*K. Deterministic LCG so runs reproduce.
static void glorotInit(oa::Matrix& inWeight, oa::I32 inRows, oa::I32 inCols, oa::I32 inK, oa::U64& inRng) {
	float scale = 1.0F;
	if (const char* e = std::getenv("OA_MG_INITSCALE")) { if (*e) scale = static_cast<float>(std::atof(e)); }
	const float bound = scale * std::sqrt(6.0F / static_cast<float>((inCols * inK) + (inRows * inK)));
	std::vector<float> w(static_cast<size_t>(inRows) * static_cast<size_t>(inCols) * static_cast<size_t>(inK));
	for (auto& v : w) {
		inRng = (inRng * 6364136223846793005ULL) + 1442695040888963407ULL;
		const float u = static_cast<float>(static_cast<oa::U32>(inRng >> 33)) / static_cast<float>(0xFFFFFFFFU >> 1);
		v = ((u * 2.0F) - 1.0F) * bound;
	}
	auto init = oa::FnMatrix::fromBytes(
		oa::Span<const oa::U8>(reinterpret_cast<const oa::U8*>(w.data()), w.size() * sizeof(float)),
		oa::MatrixShape{inRows, inCols, inK}, inWeight.getDtype());
	inWeight.copyFrom(init);
}

// Debug: print RMS of a tensor (forces realize). Gated by OA_MG_DBG so it's a no-op in
// normal runs. Used to localize where a forward-magnitude explosion originates.
static void dbgRms(const char* inTag, const oa::Matrix& inH) {
	if (std::getenv("OA_MG_DBG") == nullptr) { return; }
	auto& ctx = oa::ExecutionSession::getActive();
	auto ss = oa::FnMatrix::sum(oa::FnMatrix::mul(inH, inH));
	(void)ctx.submitAndWait();
	const double rms = std::sqrt(static_cast<double>(ss.at(0)) / static_cast<double>(inH.numElements()));
	std::printf("    [rms] %-10s = %.6f\n", inTag, rms);
}

oa::AlmTokenizerAg::AlmTokenizerAg(const oa::AlmTokenizerConfig& inConfig)
	: config_(inConfig)
{
	const oa::I32 W = config_.width;
	factor_ = 1;
	for (oa::I32 i = 0; i < config_.downT; ++i) factor_ *= 2;

	// track (weight, rows, cols, k) to Glorot-init after the deferred Rand realizes.
	struct InitSpec { oa::Matrix* weight; oa::I32 rows; oa::I32 cols; oa::I32 k; };
	std::vector<InitSpec> toInit;
	auto conv = [&](oa::I32 inC, oa::I32 outC, oa::I32 k, oa::I32 stride, oa::I32 pad, const char* name, oa::I32 dilation = 1) {
		auto c = oa::makeShared<oa::Conv1d>(inC, outC, k, stride, pad, dilation);
		registerModule(name, c);
		toInit.push_back({.weight = &c->parameters()[0].data, .rows = outC, .cols = inC, .k = k});  // [out,in,K]
		return c;
	};
	auto convT = [&](oa::I32 inC, oa::I32 outC, oa::I32 k, oa::I32 stride, oa::I32 pad, const char* name) {
		auto c = oa::makeShared<oa::ConvTranspose1d>(inC, outC, k, stride, pad);
		registerModule(name, c);
		toInit.push_back({.weight = &c->parameters()[0].data, .rows = inC, .cols = outC, .k = k});  // [in,out,K]
		return c;
	};

	// ── encoder: in → (down ; res)×downT → out ──────────────────────────────────
	encIn_ = conv(config_.inputDim, W, 3, 1, 1, "enc_in");
	for (oa::I32 d = 0; d < config_.downT; ++d) {
		char dn[24];
		std::snprintf(dn, sizeof(dn), "enc_down%d", d);
		encDown_.pushBack(conv(W, W, 4, 2, 1, dn));     // stride-2 → halves length
		for (oa::I32 q = 0; q < config_.depth; ++q) {
			char a[28];
			char b[28];
			std::snprintf(a, sizeof(a), "enc_res%d_%d_a", d, q);
			std::snprintf(b, sizeof(b), "enc_res%d_%d_b", d, q);
			encRes_.pushBack(conv(W, W, 3, 1, 3, a, 3));   // dilated-3 (matches Python baseline)
			encRes_.pushBack(conv(W, W, 1, 1, 0, b));   // 1×1 (faithful ResConv1DBlock)
		}
	}
	encOut_ = conv(W, config_.codeDim, 3, 1, 1, "enc_out");

	// ── Quantizer: single-level EMA RVQ (the discrete bottleneck) ────────────────
	oa::VectorQuantizerConfig vqCfg;
	vqCfg.numCodes   = config_.numCodes;
	vqCfg.codeDim    = config_.codeDim;
	vqCfg.commitBeta = config_.commitBeta;
	vqCfg.emaDecay   = config_.emaDecay;
	vqCfg.emaEps     = config_.emaEps;
	vqCfg.deadThresh = config_.deadThresh;
	// Cosine VQ: the encoder emits unit-RMS z_e (see encode()), so renormalize the
	// codebook rows to unit RMS too — L2 assignment becomes cosine and EMA-shrunk codes
	// stop collapsing onto the centroid. This matches the reference tokenizer; the
	// encoder-side RmsNorm was already present, this completes the pairing.
	vqCfg.normCode   = true;
	rvq_ = oa::makeShared<oa::ResidualVectorQuantizer>(vqCfg, /*levels=*/1);
	registerModule("rvq", rvq_);

	// ── Decoder: in → (res ; up)×downT → mid → out ──────────────────────────────
	decIn_ = conv(config_.codeDim, W, 3, 1, 1, "dec_in");
	for (oa::I32 d = 0; d < config_.downT; ++d) {
		for (oa::I32 q = 0; q < config_.depth; ++q) {
			char a[28];
			char b[28];
			std::snprintf(a, sizeof(a), "dec_res%d_%d_a", d, q);
			std::snprintf(b, sizeof(b), "dec_res%d_%d_b", d, q);
			decRes_.pushBack(conv(W, W, 3, 1, 3, a, 3));   // dilated-3 (matches Python baseline)
			decRes_.pushBack(conv(W, W, 1, 1, 0, b));
		}
		char up[24];
		std::snprintf(up, sizeof(up), "dec_up%d", d);
		decUp_.pushBack(convT(W, W, 4, 2, 1, up));       // learnable 2× upsample
	}
	decMid_ = conv(W, W, 3, 1, 1, "dec_mid");
	decOut_ = conv(W, config_.inputDim, 3, 1, 1, "dec_out");

	// ── Learnable-affine channel norms (one per NormC site, in call order) ───────
	// encoder: EncIn + per stage {EncDown + depth res-block pre-norms}.
	// Decoder: DecIn + per stage {depth res-block pre-norms + DecUp} + DecMid.
	auto mkLn = [&](oa::Vector<oa::SharedPtr<oa::LayerNorm>>& inVec, const char* inName) {
		auto ln = oa::makeShared<oa::LayerNorm>(W);
		registerModule(inName, ln);
		inVec.pushBack(ln);
	};
	const oa::I32 nEncLn = 1 + (config_.downT * (1 + config_.depth));
	const oa::I32 nDecLn = 1 + (config_.downT * (config_.depth + 1)) + 1;
	for (oa::I32 i = 0; i < nEncLn; ++i) { char n[20]; std::snprintf(n, sizeof(n), "enc_ln%d", i); mkLn(encLn_, n); }
	for (oa::I32 i = 0; i < nDecLn; ++i) { char n[20]; std::snprintf(n, sizeof(n), "dec_ln%d", i); mkLn(decLn_, n); }

	// Realize the deferred rand (+ RVQ codebook init), THEN overwrite every conv weight
	// with Glorot in-place. A copyFrom before the Rand executes would be clobbered.
	auto& ctx = oa::ExecutionSession::getActive();
	(void)ctx.submitAndWait();
	oa::U64 rng = 0xC0FFEEULL;
	for (auto& s : toInit) glorotInit(*s.weight, s.rows, s.cols, s.k, rng);
	ctx.clear();
}

// Learnable-affine channel norm on [B,C,T]: fused ChannelNorm normalizes over
// the C axis directly — no transpose. Falls back to Transpose+LayerNorm+Transpose
// when autograd is enabled (training) so the tape records the ops for backward.
oa::Matrix oa::AlmTokenizerAg::normC(const oa::SharedPtr<oa::LayerNorm>& inLn, const oa::Matrix& inH) const {
	if (std::getenv("OA_MG_NONORM") != nullptr) { return inH; }    // diagnostic toggle
	const auto& w = inLn->parameters();
	const oa::I32 batch = static_cast<oa::I32>(inH.size(0));
	const oa::I32 channels = static_cast<oa::I32>(inH.size(1));
	const oa::I32 seqLen = static_cast<oa::I32>(inH.size(2));
	return oa::FnMatrix::channelNorm(inH, w[0].data, w[1].data, batch, channels, seqLen, 1e-5F);
}

// Fused ChannelNorm + ReLU. 1 dispatch instead of 2, with full autograd via
// oa::GradChannelNormRelu (fused ChannelNormReluBwd kernel).
oa::Matrix oa::AlmTokenizerAg::normCRelu(const oa::SharedPtr<oa::LayerNorm>& inLn, const oa::Matrix& inH) const {
	if (std::getenv("OA_MG_NONORM") != nullptr) { return inH; }    // diagnostic toggle
	const auto& w = inLn->parameters();
	const oa::I32 batch = static_cast<oa::I32>(inH.size(0));
	const oa::I32 channels = static_cast<oa::I32>(inH.size(1));
	const oa::I32 seqLen = static_cast<oa::I32>(inH.size(2));
	return oa::FnMatrix::channelNormRelu(inH, w[0].data, w[1].data, batch, channels, seqLen, 1e-5F);
}

// Bare Conv1d forward (no activation), im2col + GEMM (Conv1dGemm). Fully
// differentiable — the GEMM path uses oa::FnMatrix::reshape + MatMulNt, which carry
// autograd through the detachForGradAttach fix. The scalar direct-conv kernel was
// retired (oa::Conv1d::forward is itself Conv1dGemm now).
oa::Matrix oa::AlmTokenizerAg::convFwd(const oa::SharedPtr<oa::Conv1d>& inConv, const oa::Matrix& inH) const {
	const auto& p = inConv->parameters();
	return oa::FnMatrix::conv1dGemm(inH, p[0].data, p[1].data,
		inConv->stride(), inConv->padding(), inConv->dilation());
}

// Fused Conv1d + ReLU via im2col + GEMM (Conv1dReluGemm), differentiable. The
// fused scalar Conv1dRelu kernel was retired.
oa::Matrix oa::AlmTokenizerAg::convRelu(const oa::SharedPtr<oa::Conv1d>& inConv, const oa::Matrix& inH) const {
	const auto& p = inConv->parameters();
	return oa::FnMatrix::conv1dReluGemm(inH, p[0].data, p[1].data,
		inConv->stride(), inConv->padding(), inConv->dilation());
}

// Pre-norm residual: h = h + convB(reLU(convA(reLU(LN(h))))). Faithful ResConv1DBlock
// with learnable-affine norm. Consumes depth LNs from inLn starting at inLnCursor.
oa::Matrix oa::AlmTokenizerAg::resStack(const oa::Vector<oa::SharedPtr<oa::Conv1d>>& inConvs,
	const oa::Vector<oa::SharedPtr<oa::LayerNorm>>& inLn, oa::Usize& inLnCursor, const oa::Matrix& inH) const {
	oa::Matrix h = inH;
	for (oa::Usize q = 0; q + 1 < inConvs.size(); q += 2) {
		auto t = convFwd(inConvs[q], normCRelu(inLn[inLnCursor++], h));
		t = convRelu(inConvs[q + 1], t);
		h = oa::FnMatrix::add(h, t);
	}
	return h;
}

oa::Matrix oa::AlmTokenizerAg::encode(const oa::Matrix& inX, oa::I32 inBatch, oa::I32 inSeqLen) {
	// [B,T,inputDim] → channels-first [B,inputDim,T] for Conv1d.
	auto xc = oa::FnMatrix::transpose(inX.reshape(oa::MatrixShape{inBatch, inSeqLen, config_.inputDim}), 1, 2);
	oa::Usize lnc = 0;
	auto h  = normCRelu(encLn_[lnc++], convFwd(encIn_, xc));    // [B, W, T], bounded
	oa::Usize resCursor = 0;
	for (oa::I32 d = 0; d < config_.downT; ++d) {
		h = normCRelu(encLn_[lnc++], convFwd(encDown_[static_cast<oa::Usize>(d)], h));   // [B, W, T/2^(d+1)]
		oa::Vector<oa::SharedPtr<oa::Conv1d>> stage;
		for (oa::I32 q = 0; q < 2 * config_.depth; ++q) stage.pushBack(encRes_[resCursor++]);
		h = resStack(stage, encLn_, lnc, h);
	}
	auto z  = convFwd(encOut_, h);                             // [B, codeDim, T/Factor]
	auto zt = oa::FnMatrix::transpose(z, 1, 2);                  // [B, T/Factor, codeDim]
	const oa::I64 nTok = static_cast<oa::I64>(inBatch) * (inSeqLen / factor_);
	auto zf = zt.reshape(oa::MatrixShape{nTok, config_.codeDim});
	if (std::getenv("OA_MG_NONORM") != nullptr) { return zf; }     // diagnostic: skip latent norm too
	// Unit-RMS latent (bounds the latent/codebook feedback loop).
	auto w = oa::FnMatrix::ones(oa::MatrixShape{config_.codeDim}, oa::FnMatrix::weightDtype());
	return oa::FnMatrix::rmsNorm(zf, w, 1e-5F);
}

oa::Matrix oa::AlmTokenizerAg::decode(const oa::Matrix& inZq, oa::I32 inBatch, oa::I32 inTokLen) {
	auto zt = inZq.reshape(oa::MatrixShape{inBatch, inTokLen, config_.codeDim});
	dbgRms("dec.zq", inZq);
	auto zc = oa::FnMatrix::transpose(zt, 1, 2);                 // [B, codeDim, TokLen]
	oa::Usize lnc = 0;
	auto h  = normCRelu(decLn_[lnc++], convFwd(decIn_, zc));    // [B, W, TokLen], bounded
	dbgRms("dec.in", h);
	oa::Usize resCursor = 0;
	for (oa::I32 d = 0; d < config_.downT; ++d) {
		oa::Vector<oa::SharedPtr<oa::Conv1d>> stage;
		for (oa::I32 q = 0; q < 2 * config_.depth; ++q) stage.pushBack(decRes_[resCursor++]);
		h = resStack(stage, decLn_, lnc, h);
		h = normC(decLn_[lnc++], decUp_[static_cast<oa::Usize>(d)]->forward(h));   // [B, W, ×2], bounded
		dbgRms("dec.up", h);
	}
	h       = normCRelu(decLn_[lnc++], convFwd(decMid_, h));
	dbgRms("dec.mid", h);
	auto o  = convFwd(decOut_, h);                             // [B, inputDim, T]
	dbgRms("dec.out", o);
	auto ot = oa::FnMatrix::transpose(o, 1, 2);                  // [B, T, inputDim]
	const oa::I64 nFrame = static_cast<oa::I64>(inBatch) * static_cast<oa::I64>(inTokLen) * factor_;
	return ot.reshape(oa::MatrixShape{nFrame, config_.inputDim});
}

oa::Vector<oa::Matrix> oa::AlmTokenizerAg::tokenize(const oa::Matrix& inX, oa::I32 inBatch, oa::I32 inSeqLen) {
	auto z = encode(inX, inBatch, inSeqLen);
	return rvq_->quantize(z).idx;
}

oa::Matrix oa::AlmTokenizerAg::detokenize(const oa::Vector<oa::Matrix>& inIdx, oa::I32 inBatch, oa::I32 inTokLen) {
	return decode(rvq_->lookup(inIdx), inBatch, inTokLen);
}
