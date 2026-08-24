// oa::EmpyrealmModule — empyrealm SSM mixer implementation.
//
// 1:1 copy of oa::Mamba3Module forward/Preprocess/step, dispatching empyrealm*
// kernels instead of Mamba3*. See EmpyrealmModule.h for design rationale.

#include "empyrealmModule.h"
#include <oa/core/fnMatrix.h>
#include <oa/ml/nn.h>
#include <oa/runtime/executionSession.h>
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

oa::Matrix oa::EmpyrealmModule::forward(const oa::Matrix& inInput) {
	NVTX_RANGE_PUSH("empyrealm::forward");
	if (inInput.rank() != 3 or inInput.size(2) != static_cast<oa::I64>(dModel_)) {
		throw std::invalid_argument(
			"oa::EmpyrealmModule::forward expects a 3D [batch, seqLen, d_model] input; "
			"reshape a flat [B*S, D] embedding to [B, S, D] before calling.");
	}
	oa::I32 batch = static_cast<oa::I32>(inInput.size(0));
	oa::I32 seqLen = static_cast<oa::I32>(inInput.size(1));

	NVTX_RANGE_PUSH("empyrealm::Preprocess");
	auto pp = preprocess(inInput, batch, seqLen);
	NVTX_RANGE_POP();

	oa::Matrix X = pp.x;
	oa::Matrix Z = pp.z;
	oa::Matrix Y;

	oa::SsmConfig config{
		.batch = static_cast<oa::U32>(batch),
		.seqLen = static_cast<oa::U32>(seqLen),
		.nHeads = static_cast<oa::U32>(nHeads_),
		.headDim = static_cast<oa::U32>(headDim_),
		.stateSize = static_cast<oa::U32>(dState_),
		.numRopeAngles = static_cast<oa::U32>(numRopeAngles_),
		.hasZ = isOutprojNorm_ ? 0u : 1u,
		.hasD = 1u
	};

	// FP32-island: cast the D skip param up so the scan dispatch stays uniformly fp32.
	const oa::Matrix dF = oa::FnMatrix::cast(d_, oa::ScalarType::Float32);

	if (isMimo_) {
		auto headOnes = oa::FnMatrix::ones(oa::MatrixShape{nHeads_, 1}, oa::ScalarType::Float32);
		oa::Matrix yTotal;
		for (oa::I32 r = 0; r < mimoRank_; r++) {
			auto Br = oa::FnMatrix::slice(pp.bh, 2, r, r + 1);
			auto Cr = oa::FnMatrix::slice(pp.ch, 2, r, r + 1);
			auto Brh = (Br * headOnes).reshape(oa::MatrixShape{batch, seqLen, nHeads_, dState_});
			auto Crh = (Cr * headOnes).reshape(oa::MatrixShape{batch, seqLen, nHeads_, dState_});
			auto cbr = oa::FnMatrix::slice(cBias_, 1, r, r + 1).reshape(oa::MatrixShape{nHeads_, dState_});
			auto bbr = oa::FnMatrix::slice(bBias_, 1, r, r + 1).reshape(oa::MatrixShape{nHeads_, dState_});
			auto mxr = oa::FnMatrix::slice(mimoX_, 1, r, r + 1).reshape(oa::MatrixShape{1, 1, nHeads_, headDim_});
			auto mzr = oa::FnMatrix::slice(mimoZ_, 1, r, r + 1).reshape(oa::MatrixShape{1, 1, nHeads_, headDim_});
			auto mor = oa::FnMatrix::slice(mimoO_, 1, r, r + 1).reshape(oa::MatrixShape{1, 1, nHeads_, headDim_});
			auto xr = X * mxr;
			auto zr = Z * mzr;
			auto yr = oa::FnMatrix::empyrealmSiso(Crh, Brh, xr, zr, pp.adt3, pp.dt3, pp.trap3, pp.angle3,
				cbr, bbr, dF, config);
			auto contrib = yr * mor;
			yTotal = (r == 0) ? contrib : (yTotal + contrib);
		}
		Y = yTotal;
	} else {
		NVTX_RANGE_PUSH("EmpyrealmSiso");
		// Verified full-sequence empyrealm recurrence.
		Y = oa::FnMatrix::empyrealmSiso(pp.ch, pp.bh, X, Z, pp.adt3, pp.dt3, pp.trap3, pp.angle3,
			pp.cBias2, pp.bBias2, dF, config);
		NVTX_RANGE_POP();
	}

	oa::Matrix yFlat;
	if (isOutprojNorm_) {
		auto weight = oa::FnMatrix::cast(normWeight_, oa::ScalarType::Float32);
		auto noBias = oa::FnMatrix::empty(oa::MatrixShape{0}, oa::ScalarType::Float32);
		auto normed = oa::FnMatrix::rmsNormGated(
			Y, weight, noBias, Z, 1e-5f, true);
		yFlat = normed.reshape(oa::MatrixShape{batch, seqLen, dInner_});
	} else {
		yFlat = Y.reshape(oa::MatrixShape{batch, seqLen, dInner_});
	}
	// Close the FP32 island: cast back to the out-proj weight dtype before the GEMM.
	auto yFlat2d = oa::FnMatrix::cast(yFlat.reshape(oa::MatrixShape{batch * seqLen, dInner_}), outProj_.getDtype());
	NVTX_RANGE_PUSH("empyrealm::outProjLinear");
	auto out2d = oa::FnMatrix::linear(yFlat2d, outProj_);
	NVTX_RANGE_POP();
	auto out = out2d.reshape(oa::MatrixShape{batch, seqLen, dModel_});

	NVTX_RANGE_POP();
	return out;
}

oa::Mamba3Module::PreprocOut oa::EmpyrealmModule::preprocess(
	const oa::Matrix& inInput, oa::I32 batch, oa::I32 seqLen) {
	NVTX_RANGE_PUSH("empyrealm::Preprocess");
	auto in2d = inInput.reshape(oa::MatrixShape{batch * seqLen, dModel_});
	NVTX_RANGE_PUSH("empyrealm::inProjLinear");
	auto projected2d = oa::FnMatrix::linear(in2d, inProj_);
	NVTX_RANGE_POP();

	// Mixed precision: run the SSM math (preprocess + selective scan) as an FP32
	// island — cast the in-proj output and SSM bias params up here, cast back to
	// bf16 before out-proj in forward. See oa::Mamba3Module::Preprocess / [[oa-bf16-dtype-mess]].
	projected2d = oa::FnMatrix::cast(projected2d, oa::ScalarType::Float32);
	const oa::Matrix dtBiasF = oa::FnMatrix::cast(dtBias_, oa::ScalarType::Float32);

	// Fused preprocess: split + RMSNorm + dt + adt in one dispatch
	// (EmpyrealmPreprocess is a renamed copy of Mamba3Preprocess; same math today.)
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
	auto pp = oa::FnMatrix::empyrealmPreprocess(projected2d, dtBiasF, ppCfg);

	PreprocOut o;
	o.x = pp.x.reshape(oa::MatrixShape{batch, seqLen, nHeads_, headDim_});
	o.z = pp.z.reshape(oa::MatrixShape{batch, seqLen, nHeads_, headDim_});

	if (isMimo_) {
		o.bh = pp.bh.reshape(oa::MatrixShape{batch, seqLen, nGroups_ * mimoRank_, dState_});
		o.ch = pp.ch.reshape(oa::MatrixShape{batch, seqLen, nGroups_ * mimoRank_, dState_});
	} else if (nGroups_ == nHeads_) {
		o.bh = pp.bh.reshape(oa::MatrixShape{batch, seqLen, nHeads_, dState_});
		o.ch = pp.ch.reshape(oa::MatrixShape{batch, seqLen, nHeads_, dState_});
	} else {
		auto b4 = pp.bh.reshape(oa::MatrixShape{batch, seqLen, nGroups_ * mimoRank_, dState_});
		auto c4 = pp.ch.reshape(oa::MatrixShape{batch, seqLen, nGroups_ * mimoRank_, dState_});
		auto headOnes = oa::FnMatrix::ones(oa::MatrixShape{nHeads_, 1}, oa::ScalarType::Float32);
		o.bh = (b4 * headOnes).reshape(oa::MatrixShape{batch, seqLen, nHeads_, dState_});
		o.ch = (c4 * headOnes).reshape(oa::MatrixShape{batch, seqLen, nHeads_, dState_});
	}

	o.dt3   = pp.dt.reshape(oa::MatrixShape{batch, seqLen, nHeads_});
	o.adt3  = pp.adt.reshape(oa::MatrixShape{batch, seqLen, nHeads_});
	o.trap3 = pp.trap.reshape(oa::MatrixShape{batch, seqLen, nHeads_});
	o.angle3 = pp.angle.reshape(oa::MatrixShape{batch, seqLen, numRopeAngles_});
	o.cBias2 = oa::FnMatrix::cast(cBias_, oa::ScalarType::Float32);
	o.bBias2 = oa::FnMatrix::cast(bBias_, oa::ScalarType::Float32);
	NVTX_RANGE_POP();
	return o;
}

oa::Matrix oa::EmpyrealmModule::step(const oa::Matrix& inInput) {
	oa::I32 batch = static_cast<oa::I32>(inInput.size(0));
	if (stepSsm_.numElements() == 0 or static_cast<oa::I32>(stepSsm_.size(0)) != batch) {
		resetState(batch);
	}

	auto pp = preprocess(inInput, batch, 1);

	oa::SsmConfig config{
		.batch = static_cast<oa::U32>(batch),
		.seqLen = 1u,
		.nHeads = static_cast<oa::U32>(nHeads_),
		.headDim = static_cast<oa::U32>(headDim_),
		.stateSize = static_cast<oa::U32>(dState_),
		.numRopeAngles = static_cast<oa::U32>(numRopeAngles_),
		.hasZ = isOutprojNorm_ ? 0u : 1u,
		.hasD = 1u
	};

	auto y = oa::FnMatrix::empyrealmSisoStep(pp.ch, pp.bh, pp.x, pp.z, pp.adt3, pp.dt3, pp.trap3,
		pp.angle3, pp.cBias2, pp.bBias2, d_, stepSsm_, stepAngle_, stepK_, stepV_, config);

	oa::Matrix yFlat;
	if (isOutprojNorm_) {
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
