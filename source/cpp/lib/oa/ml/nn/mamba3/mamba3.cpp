// oa::Mamba3Module — Mamba-3 selective state space model block
//
// Reference: Mamba-3 paper (https://arxiv.org/abs/2603.15569)
// Based on: https://github.com/state-spaces/mamba (mamba_ssm/modules/mamba3.py)
//
// forward/step drive the verified Mamba3Siso kernels (per-token selective A, rotary,
// trapezoidal, full outer-product state). End-to-end LM training: see
// tutorialNlpMamba3Ag (Mamba-3 reference + flat residual + gated out-norm).

#include <oa/ml/nn/mamba3/mamba3.h>
#include <oa/core/fnMatrix.h>
#include <oa/ml/nn.h>
#include <oa/runtime/executionSession.h>
#include <cmath>
#include <limits>
#include <stdexcept>

#if __has_include(<nvtx3/nvToolsExt.h>)
#include <nvtx3/nvToolsExt.h>
#define NVTX_RANGE_PUSH(name) nvtxRangePushA(name)
#define NVTX_RANGE_POP() nvtxRangePop()
#else
#define NVTX_RANGE_PUSH(name) ((void)0)
#define NVTX_RANGE_POP() ((void)0)
#endif

namespace {

oa::I32 mamba3InnerDim(oa::I32 inDModel, oa::I32 inExpand) noexcept
{
	if (inDModel <= 0 || inExpand <= 0) return 0;
	const oa::I64 product = static_cast<oa::I64>(inDModel) * inExpand;
	return product <= std::numeric_limits<oa::I32>::max()
		? static_cast<oa::I32>(product) : 0;
}

oa::I32 mamba3RopeDim(oa::I32 inDState, oa::F32 inFraction) noexcept
{
	if (inDState <= 0 || not std::isfinite(inFraction)
		|| inFraction < 0.0F || inFraction > 1.0F)
	{
		return 0;
	}
	return static_cast<oa::I32>(static_cast<oa::F64>(inDState) * inFraction) & ~1;
}

} // namespace

oa::Mamba3Module::Mamba3Module(
	oa::I32 inDModel,
	oa::I32 inDState,
	oa::I32 inExpand,
	oa::I32 inHeadDim,
	oa::I32 inNGroups,
	oa::F32 inRopeFraction,
	bool inIsMimo,
	oa::I32 inMimoRank,
	oa::F32 inDtMin,
	oa::F32 inDtMax,
	oa::F32 inDtInitFloor,
	oa::F32 inAFloor,
	bool inIsOutprojNorm)
	: dModel_(inDModel)
	, dState_(inDState)
	, expand_(inExpand)
	, headDim_(inHeadDim)
	, nGroups_(inNGroups)
	, dInner_(mamba3InnerDim(inDModel, inExpand))
	, nHeads_(headDim_ > 0 ? dInner_ / headDim_ : 0)
	, ropeFraction_(inRopeFraction)
	, ropeDim_(mamba3RopeDim(inDState, inRopeFraction))
	, numRopeAngles_(ropeDim_ / 2)
	, isMimo_(inIsMimo)
	, isOutprojNorm_(inIsOutprojNorm)
	, mimoRank_(inIsMimo ? inMimoRank : 1)
	, dtMin_(inDtMin)
	, dtMax_(inDtMax)
	, dtInitFloor_(inDtInitFloor)
	, aFloor_(inAFloor)
{
	if (dModel_ <= 0 || dState_ <= 0 || expand_ <= 0 || headDim_ <= 0
		|| dInner_ <= 0 || nHeads_ <= 0
		|| nGroups_ <= 0 || dInner_ % headDim_ != 0
		|| nHeads_ % nGroups_ != 0 || mimoRank_ <= 0
		|| mimoRank_ > 8
		|| not std::isfinite(ropeFraction_) || ropeFraction_ < 0.0F
		|| ropeFraction_ > 1.0F
		|| not std::isfinite(dtMin_) || not std::isfinite(dtMax_)
		|| not std::isfinite(dtInitFloor_) || not std::isfinite(aFloor_)
		|| dtMin_ <= 0.0F || dtMax_ < dtMin_ || dtInitFloor_ <= 0.0F
		|| aFloor_ <= 0.0F)
	{
		throw std::invalid_argument(
			"oa::Mamba3Module: dimensions and stability bounds are invalid; dimensions "
			"must be positive, d_inner/head_dim and n_heads/n_groups must divide "
			"exactly, rope_fraction must be in [0,1], dt bounds must be positive and "
			"ordered, A_floor must be positive, and MIMO rank must be in [1,8]");
	}
	auto wd = oa::FnMatrix::weightDtype();

	// in_proj weight: [dInProj, d_model]
	// output order: [z, x, B, C, dd_dt, dd_A, trap, angle]
	const oa::I64 groupedState = static_cast<oa::I64>(dState_) * nGroups_;
	if (groupedState > std::numeric_limits<oa::I64>::max() / mimoRank_) {
		throw std::invalid_argument("oa::Mamba3Module: projected dimension overflows");
	}
	const oa::I64 groupedMimoState = groupedState * mimoRank_;
	const oa::I64 dInProj64 = 2 * static_cast<oa::I64>(dInner_)
		+ 2 * groupedMimoState + numRopeAngles_
		+ 3 * static_cast<oa::I64>(nHeads_);
	if (dInProj64 > std::numeric_limits<oa::I32>::max()) {
		throw std::invalid_argument("oa::Mamba3Module: projected dimension exceeds oa::I32");
	}
	const oa::I32 dInProj = static_cast<oa::I32>(dInProj64);
	inProj_ = oa::FnMatrix::randGlorotUniform(oa::MatrixShape{dInProj, dModel_}, wd);

	// dt_bias: [n_heads]
	auto dtRand = oa::FnMatrix::rand(oa::MatrixShape{nHeads_}, oa::ScalarType::Float32);
	auto dtLogMin = std::log(dtMin_);
	auto dtLogMax = std::log(dtMax_);
	auto dtLog = dtRand * (dtLogMax - dtLogMin) + dtLogMin;
	auto dtExp = oa::FnMatrix::exp(dtLog);
	auto dtClamped = oa::FnMatrix::clampMin(dtExp, dtInitFloor_);
	auto dtExpNeg = oa::FnMatrix::exp(-dtClamped);
	auto dtBias = dtClamped + oa::FnMatrix::log(-dtExpNeg + 1.0f);
	// Pre-reshape to [1, n_heads] before register so the broadcast add in Preprocess
	// (and any bwd) uses a desc with the registered shape. Eliminates another
	// rank-changing view on a leaf param (dt_bias). Tape normalizes handle any
	// residual, but this is cleaner and matches the B/C bias fix.
	dtBias_ = dtBias.reshape(oa::MatrixShape{1, nHeads_});

	// B_bias, C_bias: use 2D [n_heads, d_state] for SISO (!mimo) to avoid rank-changing
	// reshape views of registered parameters (which break autograd leaf grad delivery
	// due to shape mismatch between captured oa::Parameter.grad and graph inputs).
	// MIMO keeps the rank dim; the per-rank slices still go through slice (grad supported).
	oa::MatrixShape biasShape = isMimo_
		? oa::MatrixShape{nHeads_, mimoRank_, dState_}
		: oa::MatrixShape{nHeads_, dState_};
	bBias_ = oa::FnMatrix::ones(biasShape, oa::ScalarType::Float32);
	cBias_ = oa::FnMatrix::ones(biasShape, oa::ScalarType::Float32);

	// MIMO projections: [nheads, mimo_rank, headdim]
	if (isMimo_) {
		mimoX_ = oa::FnMatrix::ones(oa::MatrixShape{nHeads_, mimoRank_, headDim_}, oa::ScalarType::Float32) / static_cast<oa::F32>(mimoRank_);
		mimoZ_ = oa::FnMatrix::ones(oa::MatrixShape{nHeads_, mimoRank_, headDim_}, oa::ScalarType::Float32);
		mimoO_ = oa::FnMatrix::ones(oa::MatrixShape{nHeads_, mimoRank_, headDim_}, oa::ScalarType::Float32) / static_cast<oa::F32>(mimoRank_);
	}

	// D skip: [n_heads]. Start at 0 (instead of 1) so the SSM selective path must carry
	// the sequence context from the beginning. With D=1 the direct x residual can dominate
	// the output (as noted in oaMamba3.md), starving the state/rotary/angle params of
	// gradient signal on easy overfit tasks. The param remains learnable and can grow if
	// the data benefits from a skip; for the byte LM tutorial this makes Mamba3 behave
	// more like the RNN/transformer (which have no such strong fixed skip).
	d_ = oa::FnMatrix::zeros(oa::MatrixShape{nHeads_}, oa::ScalarType::Float32);

	// out_proj: [d_model, d_inner]
	outProj_ = oa::FnMatrix::randGlorotUniform(oa::MatrixShape{dModel_, dInner_}, wd);

	// Gated output RMSNorm weight: [n_heads, headdim] = [d_inner], per-channel, group=headdim.
	if (isOutprojNorm_ || isMimo_) {
		normWeight_ = oa::FnMatrix::ones(oa::MatrixShape{nHeads_, headDim_}, oa::ScalarType::Float32);
	}

	registerParameter("in_proj", inProj_);
	registerParameter("dt_bias", dtBias_);
	registerParameter("B_bias", bBias_);
	registerParameter("C_bias", cBias_);
	if (isMimo_) {
		registerParameter("mimo_x", mimoX_);
		registerParameter("mimo_z", mimoZ_);
		registerParameter("mimo_o", mimoO_);
	}
	registerParameter("D", d_);
	registerParameter("out_proj", outProj_);
	if (isOutprojNorm_) registerParameter("norm_weight", normWeight_);

	// Re-alias all members to the *exact* oa::Matrix descriptors stored inside the
	// oa::Parameter list (after RegisterParameter has done setRequiresGrad + captured
	// the grad alias). This guarantees that every MatMul / Mamba3Siso / Add etc.
	// that receives e.g. inProj_ or bBias_ records the *canonical* desc as graph
	// input. Subsequent accumulateGrad (in tape) therefore writes into the grad
	// buffer visible via allParameterPtrs() / p->data.gradMatrix() / the optimizer.
	// Eliminates the last possible source of "member vs registered handle aliasing"
	// that was suspected in the original oaMamba3.md diagnosis.
	inProj_ = parameters()[0].data;
	dtBias_ = parameters()[1].data;
	bBias_ = parameters()[2].data;
	cBias_ = parameters()[3].data;
	oa::Usize idx = 4;
	if (isMimo_) {
		mimoX_ = parameters()[idx++].data;
		mimoZ_ = parameters()[idx++].data;
		mimoO_ = parameters()[idx++].data;
	}
	d_       = parameters()[idx++].data;
	outProj_ = parameters()[idx++].data;
	if (isOutprojNorm_) {
		normWeight_ = parameters()[idx++].data;
	}
}

oa::Matrix oa::Mamba3Module::forward(const oa::Matrix& inInput) {
	NVTX_RANGE_PUSH("Mamba3::forward");
	// The SSM scan needs an explicit [batch, seqLen, d_model] layout: batch and
	// seqLen are distinct scan axes (state resets per sequence). A flat [N, D]
	// embedding (oa::Embedding/Gather return [B*S, D]) silently misreads D as seqLen
	// and produces internally inconsistent shapes that only blow up later in
	// backward. Reject it loudly here instead — callers must reshape to 3D first
	// (e.g. emb.reshape([B, S, D]), as EmpyrealmCore::forwardEmbedded does).
	if (inInput.rank() != 3 || inInput.size(2) != static_cast<oa::I64>(dModel_)) {
		throw std::invalid_argument(
			"oa::Mamba3Module::forward expects a 3D [batch, seqLen, d_model] input; "
			"reshape a flat [B*S, D] embedding to [B, S, D] before calling.");
	}
	oa::I32 batch = static_cast<oa::I32>(inInput.size(0));
	oa::I32 seqLen = static_cast<oa::I32>(inInput.size(1));

	NVTX_RANGE_PUSH("Mamba3::preprocess");
	auto pp = preprocess(inInput, batch, seqLen);
	NVTX_RANGE_POP();

	// forward intermediates are local values. persistent state belongs only to
	// the explicit recurrent step session below.
	oa::Matrix X = pp.x;
	oa::Matrix Z = pp.z;
	oa::Matrix Y;

	// Fused Mamba-3 SISO scan (rotary + trapezoidal + selective A + D skip + silu(z) gate).
	oa::SsmConfig config{
		.batch = static_cast<oa::U32>(batch),
		.seqLen = static_cast<oa::U32>(seqLen),
		.nHeads = static_cast<oa::U32>(nHeads_),
		.nGroups = static_cast<oa::U32>(nGroups_),
		.headDim = static_cast<oa::U32>(headDim_),
		.stateSize = static_cast<oa::U32>(dState_),
		.numRopeAngles = static_cast<oa::U32>(numRopeAngles_),
		.mimoRank = static_cast<oa::U32>(mimoRank_),
		.hasZ = isMimo_ ? 1u : (isOutprojNorm_ ? 0u : 1u),
		.hasD = 1u,
		.hasOutNorm = isOutprojNorm_ ? 1u : 0u
	};

	// FP32-island: the D skip param feeds the scan alongside the (now fp32) preprocess
	// outputs — cast it up so the scan dispatch stays uniformly fp32.
	const oa::Matrix dF = oa::FnMatrix::cast(d_, oa::ScalarType::Float32);

	if (isMimo_) {
		NVTX_RANGE_PUSH("Mamba3Mimo");
		auto normWeight = oa::FnMatrix::cast(normWeight_, oa::ScalarType::Float32);
		Y = oa::FnMatrix::mamba3Mimo(
			pp.ch, pp.bh, X, Z, pp.adt3, pp.dt3, pp.trap3, pp.angle3,
			pp.cBias2, pp.bBias2, dF, mimoX_, mimoZ_, mimoO_, normWeight, config);
		NVTX_RANGE_POP();
	} else {
		NVTX_RANGE_PUSH("Mamba3Siso");
		// Verified full-sequence Mamba-3 recurrence.
		Y = oa::FnMatrix::mamba3Siso(pp.ch, pp.bh, X, Z, pp.adt3, pp.dt3, pp.trap3, pp.angle3,
			pp.cBias2, pp.bBias2, dF, config);  // [B, L, H, P]
		NVTX_RANGE_POP();
	}

	oa::Matrix yFlat;
	if (isOutprojNorm_ && !isMimo_) {
		// normalize each head over P while broadcasting the real [H,P] affine
		// weight across [B,L]. This is one semantic/physical normalization path;
		// no synthetic ones/zero parameters or trailing multiply are materialized.
		auto weight = oa::FnMatrix::cast(normWeight_, oa::ScalarType::Float32);
		auto noBias = oa::FnMatrix::empty(oa::MatrixShape{0}, oa::ScalarType::Float32);
		auto normed = oa::FnMatrix::rmsNormGated(
			Y, weight, noBias, Z, 1e-5f, true);
		yFlat = normed.reshape(oa::MatrixShape{batch, seqLen, dInner_});
	} else {
		yFlat = Y.reshape(oa::MatrixShape{batch, seqLen, dInner_});
	}
	// Close the FP32 island: cast back to the out-proj weight dtype (bf16) so the
	// out-projection runs as a plain bf16 GEMM and the block output matches the graph.
	auto yFlat2d = oa::FnMatrix::cast(yFlat.reshape(oa::MatrixShape{batch * seqLen, dInner_}), outProj_.getDtype());
	NVTX_RANGE_PUSH("Mamba3::outProjLinear");
	auto out2d = oa::FnMatrix::linear(yFlat2d, outProj_);
	NVTX_RANGE_POP();
	auto out = out2d.reshape(oa::MatrixShape{batch, seqLen, dModel_});

	NVTX_RANGE_POP(); // Mamba3::forward
	return out;
}

oa::Mamba3Module::PreprocOut oa::Mamba3Module::preprocess(
	const oa::Matrix& inInput, oa::I32 batch, oa::I32 seqLen) {
	NVTX_RANGE_PUSH("Mamba3::preprocess");
	auto in2d = inInput.reshape(oa::MatrixShape{batch * seqLen, dModel_});
	NVTX_RANGE_PUSH("Mamba3::inProjLinear");
	auto projected2d = oa::FnMatrix::linear(in2d, inProj_);
	NVTX_RANGE_POP();

	// Mixed precision: the SSM math (preprocess split/RMSNorm/discretization + the
	// selective scan) is precision-sensitive and its kernels compute in FP32. run
	// the whole SSM core as an FP32 island — cast the in-proj output and the SSM
	// bias params up here; the scan output is cast back to bf16 before out-proj in
	// forward. The in/out projections stay bf16 (plain GEMMs are fine). oa::GradCast
	// threads gradients; all casts are no-ops in fp32 mode. See [[oa-bf16-dtype-mess]].
	projected2d = oa::FnMatrix::cast(projected2d, oa::ScalarType::Float32);
	const oa::Matrix dtBiasF = oa::FnMatrix::cast(dtBias_, oa::ScalarType::Float32);

	// Fused preprocess: split + RMSNorm + dt + adt in one dispatch
	oa::Mamba3PreprocessConfig ppCfg{
		.dInner = dInner_,
		.dState = dState_,
		.nHeads = nHeads_,
		.numRopeAngles = numRopeAngles_,
		.nGroups = nGroups_,
		.mimoRank = mimoRank_,
		.eps = 1e-5f,
		.dtMin = dtMin_,
		.dtMax = dtMax_,
		.aFloor = aFloor_
	};
	auto pp = oa::FnMatrix::mamba3Preprocess(projected2d, dtBiasF, ppCfg);

	PreprocOut o;
	o.x = pp.x.reshape(oa::MatrixShape{batch, seqLen, nHeads_, headDim_});
	o.z = pp.z.reshape(oa::MatrixShape{batch, seqLen, nHeads_, headDim_});

	o.bh = pp.bh.reshape(
		oa::MatrixShape{batch, seqLen, nGroups_ * mimoRank_, dState_});
	o.ch = pp.ch.reshape(
		oa::MatrixShape{batch, seqLen, nGroups_ * mimoRank_, dState_});

	o.dt3   = pp.dt.reshape(oa::MatrixShape{batch, seqLen, nHeads_});
	o.adt3  = pp.adt.reshape(oa::MatrixShape{batch, seqLen, nHeads_});
	o.trap3 = pp.trap.reshape(oa::MatrixShape{batch, seqLen, nHeads_});
	o.angle3 = pp.angle.reshape(oa::MatrixShape{batch, seqLen, numRopeAngles_});
	o.cBias2 = oa::FnMatrix::cast(cBias_, oa::ScalarType::Float32);
	o.bBias2 = oa::FnMatrix::cast(bBias_, oa::ScalarType::Float32);
	NVTX_RANGE_POP(); // preprocess
	return o;
}

void oa::Mamba3Module::resetState(oa::I32 inBatch) {
	stepSsm_   = oa::FnMatrix::zeros(oa::MatrixShape{inBatch, nHeads_, headDim_, dState_}, oa::ScalarType::Float32);
	stepAngle_ = oa::FnMatrix::zeros(oa::MatrixShape{inBatch, nHeads_, numRopeAngles_}, oa::ScalarType::Float32);
	stepK_ = isMimo_
		? oa::FnMatrix::zeros(oa::MatrixShape{inBatch, nHeads_, mimoRank_, dState_}, oa::ScalarType::Float32)
		: oa::FnMatrix::zeros(oa::MatrixShape{inBatch, nHeads_, dState_}, oa::ScalarType::Float32);
	stepV_ = isMimo_
		? oa::FnMatrix::zeros(oa::MatrixShape{inBatch, nHeads_, mimoRank_, headDim_}, oa::ScalarType::Float32)
		: oa::FnMatrix::zeros(oa::MatrixShape{inBatch, nHeads_, headDim_}, oa::ScalarType::Float32);
}

oa::Matrix oa::Mamba3Module::step(const oa::Matrix& inInput) {
	// Single-token autoregressive step (inInput: [B, 1, d_model]).
	oa::I32 batch = static_cast<oa::I32>(inInput.size(0));
	if (stepSsm_.numElements() == 0 or static_cast<oa::I32>(stepSsm_.size(0)) != batch) {
		resetState(batch);
	}

	auto pp = preprocess(inInput, batch, 1);

	oa::SsmConfig config{
		.batch = static_cast<oa::U32>(batch),
		.seqLen = 1u,
		.nHeads = static_cast<oa::U32>(nHeads_),
		.nGroups = static_cast<oa::U32>(nGroups_),
		.headDim = static_cast<oa::U32>(headDim_),
		.stateSize = static_cast<oa::U32>(dState_),
		.numRopeAngles = static_cast<oa::U32>(numRopeAngles_),
		.mimoRank = static_cast<oa::U32>(mimoRank_),
		.hasZ = isMimo_ ? 1u : (isOutprojNorm_ ? 0u : 1u),
		.hasD = 1u,
		.hasOutNorm = isOutprojNorm_ ? 1u : 0u
	};

	oa::Matrix y;
	if (isMimo_) {
		auto normWeight = oa::FnMatrix::cast(normWeight_, oa::ScalarType::Float32);
		y = oa::FnMatrix::mamba3MimoStep(
			pp.ch, pp.bh, pp.x, pp.z, pp.adt3, pp.dt3, pp.trap3, pp.angle3,
			pp.cBias2, pp.bBias2, d_, mimoX_, mimoZ_, mimoO_, normWeight,
			stepSsm_, stepAngle_, stepK_, stepV_, config);
	} else {
		y = oa::FnMatrix::mamba3SisoStep(
			pp.ch, pp.bh, pp.x, pp.z, pp.adt3, pp.dt3, pp.trap3,
			pp.angle3, pp.cBias2, pp.bBias2, d_, stepSsm_, stepAngle_, stepK_, stepV_, config);
	}

	oa::Matrix yFlat;
	if (isOutprojNorm_ && !isMimo_) {
		auto weight = oa::FnMatrix::cast(normWeight_, oa::ScalarType::Float32);
		auto noBias = oa::FnMatrix::empty(oa::MatrixShape{0}, oa::ScalarType::Float32);
		auto normed = oa::FnMatrix::rmsNormGated(
			y, weight, noBias, pp.z, 1e-5f, true);
		yFlat = normed.reshape(oa::MatrixShape{batch, 1, dInner_});
	} else {
		yFlat = y.reshape(oa::MatrixShape{batch, 1, dInner_});
	}
	auto yFlat2d = yFlat.reshape(oa::MatrixShape{batch, dInner_});
	auto out2d = oa::FnMatrix::linear(yFlat2d, outProj_);
	auto out = out2d.reshape(oa::MatrixShape{batch, 1, dModel_});
	return out;
}
