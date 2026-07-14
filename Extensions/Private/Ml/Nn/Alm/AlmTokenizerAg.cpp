// OaAlmTokenizerAg — faithful temporal-Conv1d VQ-VAE (see AlmTokenizerAg.h).

#include <Ml/Nn/Alm/AlmTokenizerAg.h>

#include <Oa/Ml/Nn.h>            // OaConv1d, OaConvTranspose1d
#include <Oa/Core/FnMatrix.h>
#include <Oa/Ml/FnMatrix.h>      // OaFnMatrix::ChannelNorm, Conv1dRelu
#include <Oa/Ml/Autograd.h>      // OaFnAutograd::IsEnabled
#include <Oa/Runtime/Context.h>

#include <cmath>
#include <cstdio>
#include <vector>

// OaConv1d / OaConvTranspose1d init their weight with a DEFERRED uniform Rand
// (all-positive, large) — a poor init that biases ReLU activations positive and is a
// prime suspect for the conv-VQ divergence. Overwrite with Glorot-uniform in-place
// AFTER the deferred Rand has run. Weight is [Rows, Cols, K]; fan_in=Cols*K,
// fan_out=Rows*K. Deterministic LCG so runs reproduce.
static void GlorotInit(OaMatrix& InWeight, OaI32 InRows, OaI32 InCols, OaI32 InK, OaU64& InRng) {
	float scale = 1.0F;
	if (const char* e = std::getenv("OA_MG_INITSCALE")) { if (*e) scale = static_cast<float>(std::atof(e)); }
	const float bound = scale * std::sqrt(6.0F / static_cast<float>((InCols * InK) + (InRows * InK)));
	std::vector<float> w(static_cast<size_t>(InRows) * static_cast<size_t>(InCols) * static_cast<size_t>(InK));
	for (auto& v : w) {
		InRng = (InRng * 6364136223846793005ULL) + 1442695040888963407ULL;
		const float u = static_cast<float>(static_cast<OaU32>(InRng >> 33)) / static_cast<float>(0xFFFFFFFFU >> 1);
		v = ((u * 2.0F) - 1.0F) * bound;
	}
	auto init = OaFnMatrix::FromBytes(
		OaSpan<const OaU8>(reinterpret_cast<const OaU8*>(w.data()), w.size() * sizeof(float)),
		OaMatrixShape{InRows, InCols, InK}, InWeight.GetDtype());
	InWeight.CopyFrom(init);
}

// Debug: print RMS of a tensor (forces realize). Gated by OA_MG_DBG so it's a no-op in
// normal runs. Used to localize where a forward-magnitude explosion originates.
static void DbgRms(const char* InTag, const OaMatrix& InH) {
	if (std::getenv("OA_MG_DBG") == nullptr) { return; }
	auto& ctx = OaContext::GetDefault();
	auto ss = OaFnMatrix::Sum(OaFnMatrix::Mul(InH, InH));
	(void)ctx.Execute(); (void)ctx.Sync();
	const double rms = std::sqrt(static_cast<double>(ss.At(0)) / static_cast<double>(InH.NumElements()));
	std::printf("    [rms] %-10s = %.6f\n", InTag, rms);
}

OaAlmTokenizerAg::OaAlmTokenizerAg(const OaAlmTokenizerConfig& InConfig)
	: Config_(InConfig)
{
	const OaI32 W = Config_.Width;
	Factor_ = 1;
	for (OaI32 i = 0; i < Config_.DownT; ++i) Factor_ *= 2;

	// Track (weight, rows, cols, k) to Glorot-init after the deferred Rand realizes.
	struct InitSpec { OaMatrix* Weight; OaI32 Rows; OaI32 Cols; OaI32 K; };
	std::vector<InitSpec> toInit;
	auto conv = [&](OaI32 inC, OaI32 outC, OaI32 k, OaI32 stride, OaI32 pad, const char* name, OaI32 dilation = 1) {
		auto c = OaMakeSharedPtr<OaConv1d>(inC, outC, k, stride, pad, dilation);
		RegisterModule(name, c);
		toInit.push_back({.Weight = &c->Parameters()[0].Data, .Rows = outC, .Cols = inC, .K = k});  // [Out,In,K]
		return c;
	};
	auto convT = [&](OaI32 inC, OaI32 outC, OaI32 k, OaI32 stride, OaI32 pad, const char* name) {
		auto c = OaMakeSharedPtr<OaConvTranspose1d>(inC, outC, k, stride, pad);
		RegisterModule(name, c);
		toInit.push_back({.Weight = &c->Parameters()[0].Data, .Rows = inC, .Cols = outC, .K = k});  // [In,Out,K]
		return c;
	};

	// ── Encoder: in → (down ; res)×DownT → out ──────────────────────────────────
	EncIn_ = conv(Config_.InputDim, W, 3, 1, 1, "enc_in");
	for (OaI32 d = 0; d < Config_.DownT; ++d) {
		char dn[24];
		std::snprintf(dn, sizeof(dn), "enc_down%d", d);
		EncDown_.PushBack(conv(W, W, 4, 2, 1, dn));     // stride-2 → halves length
		for (OaI32 q = 0; q < Config_.Depth; ++q) {
			char a[28];
			char b[28];
			std::snprintf(a, sizeof(a), "enc_res%d_%d_a", d, q);
			std::snprintf(b, sizeof(b), "enc_res%d_%d_b", d, q);
			EncRes_.PushBack(conv(W, W, 3, 1, 3, a, 3));   // dilated-3 (matches Python baseline)
			EncRes_.PushBack(conv(W, W, 1, 1, 0, b));   // 1×1 (faithful ResConv1DBlock)
		}
	}
	EncOut_ = conv(W, Config_.CodeDim, 3, 1, 1, "enc_out");

	// ── Quantizer: single-level EMA RVQ (the discrete bottleneck) ────────────────
	OaVectorQuantizerConfig vqCfg;
	vqCfg.NumCodes   = Config_.NumCodes;
	vqCfg.CodeDim    = Config_.CodeDim;
	vqCfg.CommitBeta = Config_.CommitBeta;
	vqCfg.EmaDecay   = Config_.EmaDecay;
	vqCfg.EmaEps     = Config_.EmaEps;
	vqCfg.DeadThresh = Config_.DeadThresh;
	// Cosine VQ: the encoder emits unit-RMS z_e (see Encode()), so renormalize the
	// codebook rows to unit RMS too — L2 assignment becomes cosine and EMA-shrunk codes
	// stop collapsing onto the centroid. This matches the reference tokenizer; the
	// encoder-side RmsNorm was already present, this completes the pairing.
	vqCfg.NormCode   = true;
	Rvq_ = OaMakeSharedPtr<OaResidualVectorQuantizer>(vqCfg, /*levels=*/1);
	RegisterModule("rvq", Rvq_);

	// ── Decoder: in → (res ; up)×DownT → mid → out ──────────────────────────────
	DecIn_ = conv(Config_.CodeDim, W, 3, 1, 1, "dec_in");
	for (OaI32 d = 0; d < Config_.DownT; ++d) {
		for (OaI32 q = 0; q < Config_.Depth; ++q) {
			char a[28];
			char b[28];
			std::snprintf(a, sizeof(a), "dec_res%d_%d_a", d, q);
			std::snprintf(b, sizeof(b), "dec_res%d_%d_b", d, q);
			DecRes_.PushBack(conv(W, W, 3, 1, 3, a, 3));   // dilated-3 (matches Python baseline)
			DecRes_.PushBack(conv(W, W, 1, 1, 0, b));
		}
		char up[24];
		std::snprintf(up, sizeof(up), "dec_up%d", d);
		DecUp_.PushBack(convT(W, W, 4, 2, 1, up));       // learnable 2× upsample
	}
	DecMid_ = conv(W, W, 3, 1, 1, "dec_mid");
	DecOut_ = conv(W, Config_.InputDim, 3, 1, 1, "dec_out");

	// ── Learnable-affine channel norms (one per NormC site, in call order) ───────
	// Encoder: EncIn + per stage {EncDown + Depth res-block pre-norms}.
	// Decoder: DecIn + per stage {Depth res-block pre-norms + DecUp} + DecMid.
	auto mkLn = [&](OaVec<OaSharedPtr<OaLayerNorm>>& InVec, const char* InName) {
		auto ln = OaMakeSharedPtr<OaLayerNorm>(W);
		RegisterModule(InName, ln);
		InVec.PushBack(ln);
	};
	const OaI32 nEncLn = 1 + (Config_.DownT * (1 + Config_.Depth));
	const OaI32 nDecLn = 1 + (Config_.DownT * (Config_.Depth + 1)) + 1;
	for (OaI32 i = 0; i < nEncLn; ++i) { char n[20]; std::snprintf(n, sizeof(n), "enc_ln%d", i); mkLn(EncLn_, n); }
	for (OaI32 i = 0; i < nDecLn; ++i) { char n[20]; std::snprintf(n, sizeof(n), "dec_ln%d", i); mkLn(DecLn_, n); }

	// Realize the deferred Rand (+ RVQ codebook init), THEN overwrite every conv weight
	// with Glorot in-place. A CopyFrom before the Rand executes would be clobbered.
	auto& ctx = OaContext::GetDefault();
	(void)ctx.Execute(); (void)ctx.Sync();
	OaU64 rng = 0xC0FFEEULL;
	for (auto& s : toInit) GlorotInit(*s.Weight, s.Rows, s.Cols, s.K, rng);
	ctx.Clear();
}

// Learnable-affine channel norm on [B,C,T]: fused ChannelNorm normalizes over
// the C axis directly — no transpose. Falls back to Transpose+LayerNorm+Transpose
// when autograd is enabled (training) so the tape records the ops for backward.
OaMatrix OaAlmTokenizerAg::NormC(const OaSharedPtr<OaLayerNorm>& InLn, const OaMatrix& InH) const {
	if (std::getenv("OA_MG_NONORM") != nullptr) { return InH; }    // diagnostic toggle
	const auto& w = InLn->Parameters();
	const OaI32 batch = static_cast<OaI32>(InH.Size(0));
	const OaI32 channels = static_cast<OaI32>(InH.Size(1));
	const OaI32 seqLen = static_cast<OaI32>(InH.Size(2));
	return OaFnMatrix::ChannelNorm(InH, w[0].Data, w[1].Data, batch, channels, seqLen, 1e-5F);
}

// Fused ChannelNorm + ReLU. 1 dispatch instead of 2, with full autograd via
// OaGradChannelNormRelu (fused ChannelNormReluBwd kernel).
OaMatrix OaAlmTokenizerAg::NormCRelu(const OaSharedPtr<OaLayerNorm>& InLn, const OaMatrix& InH) const {
	if (std::getenv("OA_MG_NONORM") != nullptr) { return InH; }    // diagnostic toggle
	const auto& w = InLn->Parameters();
	const OaI32 batch = static_cast<OaI32>(InH.Size(0));
	const OaI32 channels = static_cast<OaI32>(InH.Size(1));
	const OaI32 seqLen = static_cast<OaI32>(InH.Size(2));
	return OaFnMatrix::ChannelNormRelu(InH, w[0].Data, w[1].Data, batch, channels, seqLen, 1e-5F);
}

// Bare Conv1d forward (no activation), im2col + GEMM (Conv1dGemm). Fully
// differentiable — the GEMM path uses OaFnMatrix::Reshape + MatMulNt, which carry
// autograd through the DetachForGradAttach fix. The scalar direct-conv kernel was
// retired (OaConv1d::Forward is itself Conv1dGemm now).
OaMatrix OaAlmTokenizerAg::ConvFwd(const OaSharedPtr<OaConv1d>& InConv, const OaMatrix& InH) const {
	const auto& p = InConv->Parameters();
	return OaFnMatrix::Conv1dGemm(InH, p[0].Data, p[1].Data,
		InConv->Stride(), InConv->Padding(), InConv->Dilation());
}

// Fused Conv1d + ReLU via im2col + GEMM (Conv1dReluGemm), differentiable. The
// fused scalar Conv1dRelu kernel was retired.
OaMatrix OaAlmTokenizerAg::ConvRelu(const OaSharedPtr<OaConv1d>& InConv, const OaMatrix& InH) const {
	const auto& p = InConv->Parameters();
	return OaFnMatrix::Conv1dReluGemm(InH, p[0].Data, p[1].Data,
		InConv->Stride(), InConv->Padding(), InConv->Dilation());
}

// Pre-norm residual: h = h + ConvB(ReLU(ConvA(ReLU(LN(h))))). Faithful ResConv1DBlock
// with learnable-affine norm. Consumes Depth LNs from InLn starting at InLnCursor.
OaMatrix OaAlmTokenizerAg::ResStack(const OaVec<OaSharedPtr<OaConv1d>>& InConvs,
	const OaVec<OaSharedPtr<OaLayerNorm>>& InLn, OaUsize& InLnCursor, const OaMatrix& InH) const {
	OaMatrix h = InH;
	for (OaUsize q = 0; q + 1 < InConvs.Size(); q += 2) {
		auto t = ConvFwd(InConvs[q], NormCRelu(InLn[InLnCursor++], h));
		t = ConvRelu(InConvs[q + 1], t);
		h = OaFnMatrix::Add(h, t);
	}
	return h;
}

OaMatrix OaAlmTokenizerAg::Encode(const OaMatrix& InX, OaI32 InBatch, OaI32 InSeqLen) {
	// [B,T,InputDim] → channels-first [B,InputDim,T] for Conv1d.
	auto xc = OaFnMatrix::Transpose(InX.Reshape(OaMatrixShape{InBatch, InSeqLen, Config_.InputDim}), 1, 2);
	OaUsize lnc = 0;
	auto h  = NormCRelu(EncLn_[lnc++], ConvFwd(EncIn_, xc));    // [B, W, T], bounded
	OaUsize resCursor = 0;
	for (OaI32 d = 0; d < Config_.DownT; ++d) {
		h = NormCRelu(EncLn_[lnc++], ConvFwd(EncDown_[static_cast<OaUsize>(d)], h));   // [B, W, T/2^(d+1)]
		OaVec<OaSharedPtr<OaConv1d>> stage;
		for (OaI32 q = 0; q < 2 * Config_.Depth; ++q) stage.PushBack(EncRes_[resCursor++]);
		h = ResStack(stage, EncLn_, lnc, h);
	}
	auto z  = ConvFwd(EncOut_, h);                             // [B, CodeDim, T/Factor]
	auto zt = OaFnMatrix::Transpose(z, 1, 2);                  // [B, T/Factor, CodeDim]
	const OaI64 nTok = static_cast<OaI64>(InBatch) * (InSeqLen / Factor_);
	auto zf = zt.Reshape(OaMatrixShape{nTok, Config_.CodeDim});
	if (std::getenv("OA_MG_NONORM") != nullptr) { return zf; }     // diagnostic: skip latent norm too
	// Unit-RMS latent (bounds the latent/codebook feedback loop).
	auto w = OaFnMatrix::Ones(OaMatrixShape{Config_.CodeDim}, OaFnMatrix::GetWeightDtype());
	return OaFnMatrix::RmsNorm(zf, w, 1e-5F);
}

OaMatrix OaAlmTokenizerAg::Decode(const OaMatrix& InZq, OaI32 InBatch, OaI32 InTokLen) {
	auto zt = InZq.Reshape(OaMatrixShape{InBatch, InTokLen, Config_.CodeDim});
	DbgRms("dec.zq", InZq);
	auto zc = OaFnMatrix::Transpose(zt, 1, 2);                 // [B, CodeDim, TokLen]
	OaUsize lnc = 0;
	auto h  = NormCRelu(DecLn_[lnc++], ConvFwd(DecIn_, zc));    // [B, W, TokLen], bounded
	DbgRms("dec.in", h);
	OaUsize resCursor = 0;
	for (OaI32 d = 0; d < Config_.DownT; ++d) {
		OaVec<OaSharedPtr<OaConv1d>> stage;
		for (OaI32 q = 0; q < 2 * Config_.Depth; ++q) stage.PushBack(DecRes_[resCursor++]);
		h = ResStack(stage, DecLn_, lnc, h);
		h = NormC(DecLn_[lnc++], DecUp_[static_cast<OaUsize>(d)]->Forward(h));   // [B, W, ×2], bounded
		DbgRms("dec.up", h);
	}
	h       = NormCRelu(DecLn_[lnc++], ConvFwd(DecMid_, h));
	DbgRms("dec.mid", h);
	auto o  = ConvFwd(DecOut_, h);                             // [B, InputDim, T]
	DbgRms("dec.out", o);
	auto ot = OaFnMatrix::Transpose(o, 1, 2);                  // [B, T, InputDim]
	const OaI64 nFrame = static_cast<OaI64>(InBatch) * static_cast<OaI64>(InTokLen) * Factor_;
	return ot.Reshape(OaMatrixShape{nFrame, Config_.InputDim});
}

OaVec<OaMatrix> OaAlmTokenizerAg::Tokenize(const OaMatrix& InX, OaI32 InBatch, OaI32 InSeqLen) {
	auto z = Encode(InX, InBatch, InSeqLen);
	return Rvq_->Quantize(z).Idx;
}

OaMatrix OaAlmTokenizerAg::Detokenize(const OaVec<OaMatrix>& InIdx, OaI32 InBatch, OaI32 InTokLen) {
	return Decode(Rvq_->Lookup(InIdx), InBatch, InTokLen);
}
