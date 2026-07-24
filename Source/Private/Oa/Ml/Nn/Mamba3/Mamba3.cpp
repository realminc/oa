// OaMamba3Module — Mamba-3 selective state space model block
//
// Reference: Mamba-3 paper (https://arxiv.org/abs/2603.15569)
// Based on: https://github.com/state-spaces/mamba (mamba_ssm/modules/mamba3.py)
//
// Forward/Step drive the verified Mamba3Siso kernels (per-token selective A, rotary,
// trapezoidal, full outer-product state). End-to-end LM training: see
// TutorialNlpMamba3Ag (Mamba-3 reference + flat residual + gated out-norm).

#include <Oa/Ml/Nn/Mamba3/Mamba3.h>
#include <Oa/Core/FnMatrix.h>
#include <Oa/Ml/Nn.h>
#include <Oa/Runtime/Context.h>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>

#if __has_include(<nvtx3/nvToolsExt.h>)
#include <nvtx3/nvToolsExt.h>
#define NVTX_RANGE_PUSH(name) nvtxRangePushA(name)
#define NVTX_RANGE_POP() nvtxRangePop()
#else
#define NVTX_RANGE_PUSH(name) ((void)0)
#define NVTX_RANGE_POP() ((void)0)
#endif

OaMamba3Module::OaMamba3Module(
	OaI32 InDModel,
	OaI32 InDState,
	OaI32 InExpand,
	OaI32 InHeadDim,
	OaI32 InNGroups,
	OaF32 InRopeFraction,
	bool InIsMimo,
	OaI32 InMimoRank,
	OaF32 InDtMin,
	OaF32 InDtMax,
	OaF32 InDtInitFloor,
	OaF32 InAFloor,
	bool InIsOutprojNorm)
	: DModel_(InDModel)
	, DState_(InDState)
	, Expand_(InExpand)
	, HeadDim_(InHeadDim)
	, NGroups_(InNGroups)
	, DInner_(InExpand * InDModel)
	, NHeads_(DInner_ / HeadDim_)
	, RopeFraction_(InRopeFraction)
	, RopeDim_(static_cast<OaI32>(DState_ * InRopeFraction) & ~1)  // even split_tensor_size
	, NumRopeAngles_(RopeDim_ / 2)
	, IsMimo_(InIsMimo)
	, IsOutprojNorm_(InIsOutprojNorm)
	, MimoRank_(InIsMimo ? InMimoRank : 1)
	, DtMin_(InDtMin)
	, DtMax_(InDtMax)
	, DtInitFloor_(InDtInitFloor)
	, AFloor_(InAFloor)
{
	auto wd = OaFnMatrix::GetWeightDtype();

	// in_proj weight: [dInProj, d_model]
	// Output order: [z, x, B, C, dd_dt, dd_A, trap, angle]
	OaI32 dInProj = 2 * DInner_ + 2 * DState_ * NGroups_ * MimoRank_ + NumRopeAngles_ + 3 * NHeads_;
	InProj_ = OaFnMatrix::RandGlorotUniform(OaMatrixShape{dInProj, DModel_}, wd);

	// dt_bias: [n_heads]
	auto dtRand = OaFnMatrix::Rand(OaMatrixShape{NHeads_}, OaScalarType::Float32);
	auto dtLogMin = std::log(DtMin_);
	auto dtLogMax = std::log(DtMax_);
	auto dtLog = dtRand * (dtLogMax - dtLogMin) + dtLogMin;
	auto dtExp = OaFnMatrix::Exp(dtLog);
	auto dtClamped = OaFnMatrix::ClampMin(dtExp, DtInitFloor_);
	auto dtExpNeg = OaFnMatrix::Exp(-dtClamped);
	auto dtBias = dtClamped + OaFnMatrix::Log(-dtExpNeg + 1.0f);
	// Pre-reshape to [1, n_heads] before register so the broadcast add in Preprocess
	// (and any bwd) uses a desc with the registered shape. Eliminates another
	// rank-changing view on a leaf param (dt_bias). Tape normalizes handle any
	// residual, but this is cleaner and matches the B/C bias fix.
	DtBias_ = dtBias.Reshape(OaMatrixShape{1, NHeads_});

	// B_bias, C_bias: use 2D [n_heads, d_state] for SISO (!mimo) to avoid rank-changing
	// Reshape views of registered parameters (which break autograd leaf grad delivery
	// due to shape mismatch between captured OaParameter.Grad and graph inputs).
	// MIMO keeps the rank dim; the per-rank slices still go through Slice (grad supported).
	OaMatrixShape biasShape = IsMimo_
		? OaMatrixShape{NHeads_, MimoRank_, DState_}
		: OaMatrixShape{NHeads_, DState_};
	BBias_ = OaFnMatrix::Ones(biasShape, OaScalarType::Float32);
	CBias_ = OaFnMatrix::Ones(biasShape, OaScalarType::Float32);

	// MIMO projections: [nheads, mimo_rank, headdim]
	if (IsMimo_) {
		MimoX_ = OaFnMatrix::Ones(OaMatrixShape{NHeads_, MimoRank_, HeadDim_}, OaScalarType::Float32) / static_cast<OaF32>(MimoRank_);
		MimoZ_ = OaFnMatrix::Ones(OaMatrixShape{NHeads_, MimoRank_, HeadDim_}, OaScalarType::Float32);
		MimoO_ = OaFnMatrix::Ones(OaMatrixShape{NHeads_, MimoRank_, HeadDim_}, OaScalarType::Float32) / static_cast<OaF32>(MimoRank_);
	}

	// D skip: [n_heads]. Start at 0 (instead of 1) so the SSM selective path must carry
	// the sequence context from the beginning. With D=1 the direct x residual can dominate
	// the output (as noted in OaMamb3.md), starving the state/rotary/angle params of
	// gradient signal on easy overfit tasks. The param remains learnable and can grow if
	// the data benefits from a skip; for the byte LM tutorial this makes Mamba3 behave
	// more like the RNN/Transformer (which have no such strong fixed skip).
	D_ = OaFnMatrix::Zeros(OaMatrixShape{NHeads_}, OaScalarType::Float32);

	// out_proj: [d_model, d_inner]
	OutProj_ = OaFnMatrix::RandGlorotUniform(OaMatrixShape{DModel_, DInner_}, wd);

	// Gated output RMSNorm weight: [n_heads, headdim] = [d_inner], per-channel, group=headdim.
	if (IsOutprojNorm_) {
		NormWeight_ = OaFnMatrix::Ones(OaMatrixShape{NHeads_, HeadDim_}, OaScalarType::Float32);
	}

	RegisterParameter("in_proj", InProj_);
	RegisterParameter("dt_bias", DtBias_);
	RegisterParameter("B_bias", BBias_);
	RegisterParameter("C_bias", CBias_);
	if (IsMimo_) {
		RegisterParameter("mimo_x", MimoX_);
		RegisterParameter("mimo_z", MimoZ_);
		RegisterParameter("mimo_o", MimoO_);
	}
	RegisterParameter("D", D_);
	RegisterParameter("out_proj", OutProj_);
	if (IsOutprojNorm_) RegisterParameter("norm_weight", NormWeight_);

	// Re-alias all members to the *exact* OaMatrix descriptors stored inside the
	// OaParameter list (after RegisterParameter has done SetRequiresGrad + captured
	// the Grad alias). This guarantees that every MatMul / Mamba3Siso / Add etc.
	// that receives e.g. InProj_ or BBias_ records the *canonical* desc as graph
	// input. Subsequent AccumulateGrad (in tape) therefore writes into the grad
	// buffer visible via AllParameterPtrs() / p->Data.GradMatrix() / the optimizer.
	// Eliminates the last possible source of "member vs registered handle aliasing"
	// that was suspected in the original OaMamb3.md diagnosis.
	InProj_ = Parameters()[0].Data;
	DtBias_ = Parameters()[1].Data;
	BBias_ = Parameters()[2].Data;
	CBias_ = Parameters()[3].Data;
	OaI32 idx = 4;
	if (IsMimo_) {
		MimoX_ = Parameters()[idx++].Data;
		MimoZ_ = Parameters()[idx++].Data;
		MimoO_ = Parameters()[idx++].Data;
	}
	D_       = Parameters()[idx++].Data;
	OutProj_ = Parameters()[idx++].Data;
	if (IsOutprojNorm_) {
		NormWeight_ = Parameters()[idx++].Data;
	}
}

OaMatrix OaMamba3Module::Forward(const OaMatrix& InInput) {
	NVTX_RANGE_PUSH("Mamba3::Forward");
	// The SSM scan needs an explicit [batch, seqLen, d_model] layout: batch and
	// seqLen are distinct scan axes (state resets per sequence). A flat [N, D]
	// embedding (OaEmbedding/Gather return [B*S, D]) silently misreads D as seqLen
	// and produces internally inconsistent shapes that only blow up later in
	// backward. Reject it loudly here instead — callers must reshape to 3D first
	// (e.g. emb.Reshape([B, S, D]), as EmpyrealmCore::ForwardEmbedded does).
	if (InInput.Rank() != 3 || InInput.Size(2) != static_cast<OaI64>(DModel_)) {
		throw std::invalid_argument(
			"OaMamba3Module::Forward expects a 3D [batch, seqLen, d_model] input; "
			"reshape a flat [B*S, D] embedding to [B, S, D] before calling.");
	}
	OaI32 batch = static_cast<OaI32>(InInput.Size(0));
	OaI32 seqLen = static_cast<OaI32>(InInput.Size(1));

	NVTX_RANGE_PUSH("Mamba3::Preprocess");
	auto pp = Preprocess(InInput, batch, seqLen);
	NVTX_RANGE_POP();

	// Locals for intermediates. Avoids mutating the legacy X_/Z_/Y_ members
	// (kept only for "step reuse" comments) on every Forward. This makes the
	// Mamba3 primitive safe when multiple Forwards are in flight under async
	// ItTraining / pipelined submissions (the previous manual-sync tutorials
	// hid the aliasing because they forced Sync after every step).
	OaMatrix X = pp.X;
	OaMatrix Z = pp.Z;
	OaMatrix Y;

	// Fused Mamba-3 SISO scan (rotary + trapezoidal + selective A + D skip + silu(z) gate).
	OaFnMatrix::OaSsmConfig config{
		.Batch = static_cast<OaU32>(batch),
		.SeqLen = static_cast<OaU32>(seqLen),
		.NHeads = static_cast<OaU32>(NHeads_),
		.HeadDim = static_cast<OaU32>(HeadDim_),
		.StateSize = static_cast<OaU32>(DState_),
		.NumRopeAngles = static_cast<OaU32>(NumRopeAngles_),
		.HasZ = IsOutprojNorm_ ? 0u : 1u,   // gated out-norm replaces the in-kernel silu gate
		.HasD = 1u
	};

	// FP32-island: the D skip param feeds the scan alongside the (now fp32) preprocess
	// outputs — cast it up so the scan dispatch stays uniformly fp32.
	const OaMatrix dF = OaFnMatrix::Cast(D_, OaScalarType::Float32);

	if (IsMimo_) {
		// MIMO: R independent SISO states. Each rank r runs a SISO scan on its own B,C
		// slice with x·mimo_x[r], z·mimo_z[r]; outputs recombined as Σ_r y_r·mimo_o[r].
		// (ngroups == 1; B,C arrive as [B,L,R,N].)
		auto headOnes = OaFnMatrix::Ones(OaMatrixShape{NHeads_, 1}, OaScalarType::Float32);
		OaMatrix yTotal;
		for (OaI32 r = 0; r < MimoRank_; r++) {
			auto Br = OaFnMatrix::Slice(pp.Bh, 2, r, r + 1);  // [B,L,1,N]
			auto Cr = OaFnMatrix::Slice(pp.Ch, 2, r, r + 1);
			auto Brh = (Br * headOnes).Reshape(OaMatrixShape{batch, seqLen, NHeads_, DState_});
			auto Crh = (Cr * headOnes).Reshape(OaMatrixShape{batch, seqLen, NHeads_, DState_});
			auto cbr = OaFnMatrix::Slice(CBias_, 1, r, r + 1).Reshape(OaMatrixShape{NHeads_, DState_});
			auto bbr = OaFnMatrix::Slice(BBias_, 1, r, r + 1).Reshape(OaMatrixShape{NHeads_, DState_});
			auto mxr = OaFnMatrix::Slice(MimoX_, 1, r, r + 1).Reshape(OaMatrixShape{1, 1, NHeads_, HeadDim_});
			auto mzr = OaFnMatrix::Slice(MimoZ_, 1, r, r + 1).Reshape(OaMatrixShape{1, 1, NHeads_, HeadDim_});
			auto mor = OaFnMatrix::Slice(MimoO_, 1, r, r + 1).Reshape(OaMatrixShape{1, 1, NHeads_, HeadDim_});
			auto xr = X * mxr;
			auto zr = Z * mzr;
			auto yr = OaFnMatrix::Mamba3Siso(Crh, Brh, xr, zr, pp.ADT3, pp.DT3, pp.Trap3, pp.Angle3,
				cbr, bbr, dF, config);
			auto contrib = yr * mor;
			yTotal = (r == 0) ? contrib : (yTotal + contrib);
		}
		Y = yTotal;  // [B, L, H, P]
	} else {
		NVTX_RANGE_PUSH("Mamba3Siso");
		// Verified full-sequence Mamba-3 recurrence.
		Y = OaFnMatrix::Mamba3Siso(pp.Ch, pp.Bh, X, Z, pp.ADT3, pp.DT3, pp.Trap3, pp.Angle3,
			pp.CBias2, pp.BBias2, dF, config);  // [B, L, H, P]
		NVTX_RANGE_POP();
	}

	OaMatrix yFlat;
	if (IsOutprojNorm_) {
		// Per-headdim-group gated RMSNorm with per-channel weight [n_heads, headdim].
		// (x·r·w)·silu(z) = [RmsNormGated(x, w=1, z)]·w  (w and silu(z) commute, both per-channel).
		OaI64 rows = static_cast<OaI64>(batch) * seqLen * NHeads_;
		auto yr = Y.Reshape(OaMatrixShape{rows, HeadDim_});
		auto zr = Z.Reshape(OaMatrixShape{rows, HeadDim_});
		auto ones = OaFnMatrix::Ones(OaMatrixShape{HeadDim_}, OaScalarType::Float32);
		auto noBias = OaFnMatrix::Zeros(OaMatrixShape{HeadDim_}, OaScalarType::Float32);
		auto normed = OaFnMatrix::RmsNormGated(yr, ones, noBias, zr, 1e-5f, true);
		auto normed4 = normed.Reshape(OaMatrixShape{batch, seqLen, NHeads_, HeadDim_});
		// NormWeight is a bf16 param; cast up so the FP32-island multiply stays uniform.
		auto wB = OaFnMatrix::Cast(NormWeight_, OaScalarType::Float32).Reshape(OaMatrixShape{1, 1, NHeads_, HeadDim_});
		yFlat = (normed4 * wB).Reshape(OaMatrixShape{batch, seqLen, DInner_});
	} else {
		yFlat = Y.Reshape(OaMatrixShape{batch, seqLen, DInner_});
	}
	// Close the FP32 island: cast back to the out-proj weight dtype (bf16) so the
	// out-projection runs as a plain bf16 GEMM and the block output matches the graph.
	auto yFlat2d = OaFnMatrix::Cast(yFlat.Reshape(OaMatrixShape{batch * seqLen, DInner_}), OutProj_.GetDtype());
	NVTX_RANGE_PUSH("Mamba3::OutProjMatMul");
	auto out2d = OaFnMatrix::MatMulNt(yFlat2d, OutProj_);
	NVTX_RANGE_POP();
	auto out = out2d.Reshape(OaMatrixShape{batch, seqLen, DModel_});

	// Env-gated scan-internals probe (OA_MAMBA_PROBE=1). Logs max|.| of scan output,
	// post-norm yFlat, out_proj output, and the unbounded learnable params — to localise
	// the d_model=512 forward-magnitude blowup empirically (which tensor grows first).
	static int sMProbe = 0;
	if (sMProbe < 45 && std::getenv("OA_MAMBA_PROBE") != nullptr) {
		++sMProbe;
		auto mx = [](const OaMatrix& m) {
			auto f = m.Reshape(OaMatrixShape{m.NumElements()});
			return OaFnMatrix::Max(OaFnMatrix::Abs(f), 0);
		};
		auto sY = mx(Y), sYf = mx(yFlat), sOut = mx(out);
		auto sNw = IsOutprojNorm_ ? mx(NormWeight_) : OaMatrix();
		auto sD = mx(D_), sDt = mx(DtBias_), sOp = mx(OutProj_), sIp = mx(InProj_);
		auto& ctx = OaContext::GetDefault();
		(void)ctx.Execute(); (void)ctx.Sync();
		auto v = [](const OaMatrix& m){ return m.NumElements() > 0 ? m.At(0) : -1.0f; };
		std::printf("[mamba %2d] Y=%.2f yflat=%.2f out=%.2f | normW=%.2f D=%.3f dtb=%.2f outP=%.3f inP=%.3f\n",
			sMProbe, v(sY), v(sYf), v(sOut), v(sNw), v(sD), v(sDt), v(sOp), v(sIp));
		std::fflush(stdout);
	}

	NVTX_RANGE_POP(); // Mamba3::Forward
	return out;
}

OaMamba3Module::PreprocOut OaMamba3Module::Preprocess(
	const OaMatrix& InInput, OaI32 batch, OaI32 seqLen) {
	NVTX_RANGE_PUSH("Mamba3::Preprocess");
	auto in2d = InInput.Reshape(OaMatrixShape{batch * seqLen, DModel_});
	NVTX_RANGE_PUSH("Mamba3::InProjMatMul");
	auto projected2d = OaFnMatrix::MatMulNt(in2d, InProj_);
	NVTX_RANGE_POP();

	// Mixed precision: the SSM math (preprocess split/RMSNorm/discretization + the
	// selective scan) is precision-sensitive and its kernels compute in FP32. Run
	// the whole SSM core as an FP32 island — cast the in-proj output and the SSM
	// bias params up here; the scan output is cast back to bf16 before out-proj in
	// Forward. The in/out projections stay bf16 (plain GEMMs are fine). OaGradCast
	// threads gradients; all casts are no-ops in fp32 mode. See [[oa-bf16-dtype-mess]].
	projected2d = OaFnMatrix::Cast(projected2d, OaScalarType::Float32);
	const OaMatrix dtBiasF = OaFnMatrix::Cast(DtBias_, OaScalarType::Float32);

	// Fused preprocess: split + RMSNorm + dt + adt in one dispatch
	OaFnMatrix::OaMamba3PreprocessConfig ppCfg{
		.DInner = DInner_,
		.DState = DState_,
		.NHeads = NHeads_,
		.NumRopeAngles = NumRopeAngles_,
		.NGroups = NGroups_,
		.MimoRank = MimoRank_,
		.Eps = 1e-5f,
		.DtMin = DtMin_,
		.DtMax = DtMax_,
		.AFloor = AFloor_
	};
	auto pp = OaFnMatrix::Mamba3Preprocess(projected2d, dtBiasF, ppCfg);

	PreprocOut o;
	o.X = pp.X.Reshape(OaMatrixShape{batch, seqLen, NHeads_, HeadDim_});
	o.Z = pp.Z.Reshape(OaMatrixShape{batch, seqLen, NHeads_, HeadDim_});

	if (IsMimo_) {
		o.Bh = pp.Bh.Reshape(OaMatrixShape{batch, seqLen, NGroups_ * MimoRank_, DState_});
		o.Ch = pp.Ch.Reshape(OaMatrixShape{batch, seqLen, NGroups_ * MimoRank_, DState_});
	} else if (NGroups_ == NHeads_) {
		o.Bh = pp.Bh.Reshape(OaMatrixShape{batch, seqLen, NHeads_, DState_});
		o.Ch = pp.Ch.Reshape(OaMatrixShape{batch, seqLen, NHeads_, DState_});
	} else {
		auto b4 = pp.Bh.Reshape(OaMatrixShape{batch, seqLen, NGroups_ * MimoRank_, DState_});
		auto c4 = pp.Ch.Reshape(OaMatrixShape{batch, seqLen, NGroups_ * MimoRank_, DState_});
		auto headOnes = OaFnMatrix::Ones(OaMatrixShape{NHeads_, 1}, OaScalarType::Float32);
		o.Bh = (b4 * headOnes).Reshape(OaMatrixShape{batch, seqLen, NHeads_, DState_});
		o.Ch = (c4 * headOnes).Reshape(OaMatrixShape{batch, seqLen, NHeads_, DState_});
	}

	o.DT3   = pp.DT.Reshape(OaMatrixShape{batch, seqLen, NHeads_});
	o.ADT3  = pp.ADT.Reshape(OaMatrixShape{batch, seqLen, NHeads_});
	o.Trap3 = pp.Trap.Reshape(OaMatrixShape{batch, seqLen, NHeads_});
	o.Angle3 = pp.Angle.Reshape(OaMatrixShape{batch, seqLen, NumRopeAngles_});
	o.CBias2 = OaFnMatrix::Cast(CBias_, OaScalarType::Float32);
	o.BBias2 = OaFnMatrix::Cast(BBias_, OaScalarType::Float32);
	NVTX_RANGE_POP(); // Preprocess
	return o;
}

void OaMamba3Module::ResetState(OaI32 InBatch) {
	StepSsm_   = OaFnMatrix::Zeros(OaMatrixShape{InBatch, NHeads_, HeadDim_, DState_}, OaScalarType::Float32);
	StepAngle_ = OaFnMatrix::Zeros(OaMatrixShape{InBatch, NHeads_, NumRopeAngles_}, OaScalarType::Float32);
	StepK_     = OaFnMatrix::Zeros(OaMatrixShape{InBatch, NHeads_, DState_}, OaScalarType::Float32);
	StepV_     = OaFnMatrix::Zeros(OaMatrixShape{InBatch, NHeads_, HeadDim_}, OaScalarType::Float32);
}

OaMatrix OaMamba3Module::Step(const OaMatrix& InInput) {
	// Single-token autoregressive step (InInput: [B, 1, d_model]).
	OaI32 batch = static_cast<OaI32>(InInput.Size(0));
	if (StepSsm_.NumElements() == 0 or static_cast<OaI32>(StepSsm_.Size(0)) != batch) {
		ResetState(batch);
	}

	auto pp = Preprocess(InInput, batch, 1);

	OaFnMatrix::OaSsmConfig config{
		.Batch = static_cast<OaU32>(batch),
		.SeqLen = 1u,
		.NHeads = static_cast<OaU32>(NHeads_),
		.HeadDim = static_cast<OaU32>(HeadDim_),
		.StateSize = static_cast<OaU32>(DState_),
		.NumRopeAngles = static_cast<OaU32>(NumRopeAngles_),
		.HasZ = IsOutprojNorm_ ? 0u : 1u,
		.HasD = 1u
	};

	auto y = OaFnMatrix::Mamba3SisoStep(pp.Ch, pp.Bh, pp.X, pp.Z, pp.ADT3, pp.DT3, pp.Trap3,
		pp.Angle3, pp.CBias2, pp.BBias2, D_, StepSsm_, StepAngle_, StepK_, StepV_, config);

	OaMatrix yFlat;
	if (IsOutprojNorm_) {
		OaI64 rows = static_cast<OaI64>(batch) * NHeads_;
		auto yr = y.Reshape(OaMatrixShape{rows, HeadDim_});
		auto zr = pp.Z.Reshape(OaMatrixShape{rows, HeadDim_});
		auto ones = OaFnMatrix::Ones(OaMatrixShape{HeadDim_}, OaScalarType::Float32);
		auto noBias = OaFnMatrix::Zeros(OaMatrixShape{HeadDim_}, OaScalarType::Float32);
		auto normed = OaFnMatrix::RmsNormGated(yr, ones, noBias, zr, 1e-5f, true);
		auto wB = NormWeight_.Reshape(OaMatrixShape{1, 1, NHeads_, HeadDim_});
		yFlat = (normed.Reshape(OaMatrixShape{batch, 1, NHeads_, HeadDim_}) * wB).Reshape(OaMatrixShape{batch, 1, DInner_});
	} else {
		yFlat = y.Reshape(OaMatrixShape{batch, 1, DInner_});
	}
	auto yFlat2d = yFlat.Reshape(OaMatrixShape{batch, DInner_});
	auto out2d = OaFnMatrix::MatMulNt(yFlat2d, OutProj_);
	auto out = out2d.Reshape(OaMatrixShape{batch, 1, DModel_});
	return out;
}
