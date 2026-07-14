// OaEmpyrealmModule — Empyrealm SSM mixer implementation.
//
// 1:1 copy of OaMamba3Module Forward/Preprocess/Step, dispatching Empyrealm*
// kernels instead of Mamba3*. See EmpyrealmModule.h for design rationale.

#include "EmpyrealmModule.h"
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

OaMatrix OaEmpyrealmModule::Forward(const OaMatrix& InInput) {
	NVTX_RANGE_PUSH("Empyrealm::Forward");
	if (InInput.Rank() != 3 or InInput.Size(2) != static_cast<OaI64>(DModel_)) {
		throw std::invalid_argument(
			"OaEmpyrealmModule::Forward expects a 3D [batch, seqLen, d_model] input; "
			"reshape a flat [B*S, D] embedding to [B, S, D] before calling.");
	}
	OaI32 batch = static_cast<OaI32>(InInput.Size(0));
	OaI32 seqLen = static_cast<OaI32>(InInput.Size(1));

	NVTX_RANGE_PUSH("Empyrealm::Preprocess");
	auto pp = Preprocess(InInput, batch, seqLen);
	NVTX_RANGE_POP();

	OaMatrix X = pp.X;
	OaMatrix Z = pp.Z;
	OaMatrix Y;

	OaFnMatrix::OaSsmConfig config{
		.Batch = static_cast<OaU32>(batch),
		.SeqLen = static_cast<OaU32>(seqLen),
		.NHeads = static_cast<OaU32>(NHeads_),
		.HeadDim = static_cast<OaU32>(HeadDim_),
		.StateSize = static_cast<OaU32>(DState_),
		.NumRopeAngles = static_cast<OaU32>(NumRopeAngles_),
		.HasZ = IsOutprojNorm_ ? 0u : 1u,
		.HasD = 1u
	};

	// FP32-island: cast the D skip param up so the scan dispatch stays uniformly fp32.
	const OaMatrix dF = OaFnMatrix::Cast(D_, OaScalarType::Float32);

	if (IsMimo_) {
		auto headOnes = OaFnMatrix::Ones(OaMatrixShape{NHeads_, 1}, OaScalarType::Float32);
		OaMatrix yTotal;
		for (OaI32 r = 0; r < MimoRank_; r++) {
			auto Br = OaFnMatrix::Slice(pp.Bh, 2, r, r + 1);
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
			auto yr = OaFnMatrix::EmpyrealmSiso(Crh, Brh, xr, zr, pp.ADT3, pp.DT3, pp.Trap3, pp.Angle3,
				cbr, bbr, dF, config);
			auto contrib = yr * mor;
			yTotal = (r == 0) ? contrib : (yTotal + contrib);
		}
		Y = yTotal;
	} else {
		NVTX_RANGE_PUSH("EmpyrealmSiso");
		// Verified full-sequence Empyrealm recurrence.
		Y = OaFnMatrix::EmpyrealmSiso(pp.Ch, pp.Bh, X, Z, pp.ADT3, pp.DT3, pp.Trap3, pp.Angle3,
			pp.CBias2, pp.BBias2, dF, config);
		NVTX_RANGE_POP();
	}

	OaMatrix yFlat;
	if (IsOutprojNorm_) {
		OaI64 rows = static_cast<OaI64>(batch) * seqLen * NHeads_;
		auto yr = Y.Reshape(OaMatrixShape{rows, HeadDim_});
		auto zr = Z.Reshape(OaMatrixShape{rows, HeadDim_});
		auto ones = OaFnMatrix::Ones(OaMatrixShape{HeadDim_}, OaScalarType::Float32);
		auto noBias = OaFnMatrix::Zeros(OaMatrixShape{HeadDim_}, OaScalarType::Float32);
		auto normed = OaFnMatrix::RmsNormGated(yr, ones, noBias, zr, 1e-5f, true);
		auto normed4 = normed.Reshape(OaMatrixShape{batch, seqLen, NHeads_, HeadDim_});
		auto wB = OaFnMatrix::Cast(NormWeight_, OaScalarType::Float32).Reshape(OaMatrixShape{1, 1, NHeads_, HeadDim_});
		yFlat = (normed4 * wB).Reshape(OaMatrixShape{batch, seqLen, DInner_});
	} else {
		yFlat = Y.Reshape(OaMatrixShape{batch, seqLen, DInner_});
	}
	// Close the FP32 island: cast back to the out-proj weight dtype before the GEMM.
	auto yFlat2d = OaFnMatrix::Cast(yFlat.Reshape(OaMatrixShape{batch * seqLen, DInner_}), OutProj_.GetDtype());
	NVTX_RANGE_PUSH("Empyrealm::OutProjMatMul");
	auto out2d = OaFnMatrix::MatMulNt(yFlat2d, OutProj_);
	NVTX_RANGE_POP();
	auto out = out2d.Reshape(OaMatrixShape{batch, seqLen, DModel_});

	NVTX_RANGE_POP();
	return out;
}

OaMamba3Module::PreprocOut OaEmpyrealmModule::Preprocess(
	const OaMatrix& InInput, OaI32 batch, OaI32 seqLen) {
	NVTX_RANGE_PUSH("Empyrealm::Preprocess");
	auto in2d = InInput.Reshape(OaMatrixShape{batch * seqLen, DModel_});
	NVTX_RANGE_PUSH("Empyrealm::InProjMatMul");
	auto projected2d = OaFnMatrix::MatMulNt(in2d, InProj_);
	NVTX_RANGE_POP();

	// Mixed precision: run the SSM math (preprocess + selective scan) as an FP32
	// island — cast the in-proj output and SSM bias params up here, cast back to
	// bf16 before out-proj in Forward. See OaMamba3Module::Preprocess / [[oa-bf16-dtype-mess]].
	projected2d = OaFnMatrix::Cast(projected2d, OaScalarType::Float32);
	const OaMatrix dtBiasF = OaFnMatrix::Cast(DtBias_, OaScalarType::Float32);

	// Fused preprocess: split + RMSNorm + dt + adt in one dispatch
	// (EmpyrealmPreprocess is a renamed copy of Mamba3Preprocess; same math today.)
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
	auto pp = OaFnMatrix::EmpyrealmPreprocess(projected2d, dtBiasF, ppCfg);

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
	NVTX_RANGE_POP();
	return o;
}

OaMatrix OaEmpyrealmModule::Step(const OaMatrix& InInput) {
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

	auto y = OaFnMatrix::EmpyrealmSisoStep(pp.Ch, pp.Bh, pp.X, pp.Z, pp.ADT3, pp.DT3, pp.Trap3,
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
