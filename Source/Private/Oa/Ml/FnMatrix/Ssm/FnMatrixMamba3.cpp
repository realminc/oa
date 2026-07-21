// OaFnMatrix — Mamba-3 SSM operations
//
// Selective state space model dispatch: verified full-sequence recurrence plus
// fused per-token dt and A·dt terms.

#include <Oa/Ml/FnMatrix.h>
#include <Oa/Ml/Autograd/Nodes.h>
#include <Oa/Core/Matrix.h>
#include <Oa/Core/FnMatrix.h>
#include <Oa/Core/Status.h>
#include <Oa/Core/Types.h>
#include <Oa/Runtime/Context.h>

#include <cassert>

// ============================================================================
// Mamba3Siso — selective state space scan (Ssm/Mamba3/Mamba3SisoFwd.slang)
// ============================================================================

OaMatrix OaFnMatrix::Mamba3Siso(
	const OaMatrix& InC,
	const OaMatrix& InB,
	const OaMatrix& InX,
	const OaMatrix& InZ,
	const OaMatrix& InAdt,
	const OaMatrix& InDt,
	const OaMatrix& InTrap,
	const OaMatrix& InAngle,
	const OaMatrix& InCBias,
	const OaMatrix& InBBias,
	const OaMatrix& InD,
	const OaSsmConfig& InConfig)
{
	assert(InConfig.HeadDim <= 128 && "Mamba3Siso: headdim (P) must be <= 128");
	assert(InConfig.StateSize <= 128 && "Mamba3Siso: d_state (N) must be <= 128");
	assert(InConfig.NumRopeAngles <= 64 && "Mamba3Siso: num_rope_angles (A) must be <= 64");

	// Mixed precision: the selective-scan kernels compute in FP32 through the
	// always-fp32 Storage helpers, and the cross-token recurrent state is far too
	// precision-sensitive for bf16 round-trips. In bf16 mode run the scan as an
	// FP32 island — cast the operands up, scan, cast the result back down.
	// OaGradCast threads gradients across both boundaries; Cast is a no-op on the
	// recursive fp32 call, so this reduces to the plain body below.
	if (InX.GetDtype() == OaScalarType::BFloat16 || InX.GetDtype() == OaScalarType::Float16) {
		const auto up = [](const OaMatrix& m) { return OaFnMatrix::Cast(m, OaScalarType::Float32); };
		return OaFnMatrix::Cast(
			Mamba3Siso(up(InC), up(InB), up(InX), up(InZ), up(InAdt), up(InDt),
				up(InTrap), up(InAngle), up(InCBias), up(InBBias), up(InD), InConfig),
			InX.GetDtype());
	}

	auto& ctx = OaContext::GetDefault();

	// Every active output lane is written exactly once by Mamba3SisoFwd.
	OaMatrix output = OaFnMatrix::Empty(
		OaMatrixShape{InConfig.Batch, InConfig.SeqLen, InConfig.NHeads, InConfig.HeadDim},
		InX.Dtype_);

	struct {
		OaU32 B, L, H, P, N, A, has_z, has_d;
	} push{
		InConfig.Batch, InConfig.SeqLen, InConfig.NHeads, InConfig.HeadDim,
		InConfig.StateSize, InConfig.NumRopeAngles, InConfig.HasZ, InConfig.HasD
	};

	OaBufferAccess access[] = {
		OaBufferAccess::Read,  // c
		OaBufferAccess::Read,  // b
		OaBufferAccess::Read,  // x
		OaBufferAccess::Read,  // z
		OaBufferAccess::Read,  // adt
		OaBufferAccess::Read,  // dt
		OaBufferAccess::Read,  // trap
		OaBufferAccess::Read,  // angle
		OaBufferAccess::Read,  // c_bias
		OaBufferAccess::Read,  // b_bias
		OaBufferAccess::Read,  // d
		OaBufferAccess::Write  // y
	};

	ctx.Add("Mamba3SisoFwd",
		{&InC, &InB, &InX, &InZ, &InAdt, &InDt, &InTrap, &InAngle, &InCBias, &InBBias, &InD, &output},
		access, &push, sizeof(push), InConfig.Batch * InConfig.NHeads, 1, 1);

	if (OaFnAutograd::IsEnabled() and
		(InC.RequiresGrad() or InB.RequiresGrad() or InX.RequiresGrad() or
		 InZ.RequiresGrad() or InAdt.RequiresGrad() or InDt.RequiresGrad() or
		 InTrap.RequiresGrad() or InAngle.RequiresGrad() or InCBias.RequiresGrad() or
		 InBBias.RequiresGrad() or InD.RequiresGrad())) {
		auto gradFn = OaMakeSharedPtr<OaGradMamba3Siso>();
		gradFn->Saved_ = OaVec<OaMatrix>{InC, InB, InX, InZ, InAdt, InDt, InTrap,
			InAngle, InCBias, InBBias, InD};
		gradFn->SetGraphInputs(OaVec<OaMatrix>{InC, InB, InX, InZ, InAdt, InDt, InTrap,
			InAngle, InCBias, InBBias, InD});
		gradFn->SequenceNr_ = OaFnAutograd::NextSeq();
		gradFn->Config_ = InConfig;
		gradFn->OutputShape_ = output.GetShape();
		output.MutAutograd().GradFn = gradFn;
	}

	return output;
}

// ============================================================================
// Mamba3SisoStep — single-token step for autoregressive generation
// ============================================================================

OaMatrix OaFnMatrix::Mamba3SisoStep(
	const OaMatrix& InC,
	const OaMatrix& InB,
	const OaMatrix& InX,
	const OaMatrix& InZ,
	const OaMatrix& InAdt,
	const OaMatrix& InDt,
	const OaMatrix& InTrap,
	const OaMatrix& InAngle,
	const OaMatrix& InCBias,
	const OaMatrix& InBBias,
	const OaMatrix& InD,
	const OaMatrix& InSsmState,
	const OaMatrix& InAngleState,
	const OaMatrix& InKState,
	const OaMatrix& InVState,
	const OaSsmConfig& InConfig)
{
	assert(InConfig.SeqLen == 1 && "Mamba3SisoStep: SeqLen must be 1");
	auto& ctx = OaContext::GetDefault();

	// Every active output lane is written exactly once by Mamba3SisoStep.
	OaMatrix output = OaFnMatrix::Empty(
		OaMatrixShape{InConfig.Batch, 1, InConfig.NHeads, InConfig.HeadDim}, InX.Dtype_);

	struct {
		OaU32 B, H, P, N, A, has_z, has_d;
	} push{ InConfig.Batch, InConfig.NHeads, InConfig.HeadDim, InConfig.StateSize,
		InConfig.NumRopeAngles, InConfig.HasZ, InConfig.HasD };

	OaBufferAccess access[] = {
		OaBufferAccess::Read,       // c
		OaBufferAccess::Read,       // b
		OaBufferAccess::Read,       // x
		OaBufferAccess::Read,       // z
		OaBufferAccess::Read,       // adt
		OaBufferAccess::Read,       // dt
		OaBufferAccess::Read,       // trap
		OaBufferAccess::Read,       // angle
		OaBufferAccess::Read,       // c_bias
		OaBufferAccess::Read,       // b_bias
		OaBufferAccess::Read,       // d
		OaBufferAccess::Write,      // y
		OaBufferAccess::ReadWrite,  // ssm_state
		OaBufferAccess::ReadWrite,  // angle_state
		OaBufferAccess::ReadWrite,  // k_state
		OaBufferAccess::ReadWrite   // v_state
	};

	ctx.Add("Mamba3SisoStep",
		{&InC, &InB, &InX, &InZ, &InAdt, &InDt, &InTrap, &InAngle, &InCBias, &InBBias, &InD,
		 &output, &InSsmState, &InAngleState, &InKState, &InVState},
		access, &push, sizeof(push), InConfig.Batch * InConfig.NHeads, 1, 1);

	return output;
}

// ============================================================================
// Mamba3SisoBwd — backward pass for the selective scan
// ============================================================================

OaFnMatrix::OaSsmBwdResult OaFnMatrix::Mamba3SisoBwd(
	const OaMatrix& InDOut,
	const OaMatrix& InC,
	const OaMatrix& InB,
	const OaMatrix& InX,
	const OaMatrix& InZ,
	const OaMatrix& InAdt,
	const OaMatrix& InDt,
	const OaMatrix& InTrap,
	const OaMatrix& InAngle,
	const OaMatrix& InCBias,
	const OaMatrix& InBBias,
	const OaMatrix& InD,
	const OaSsmConfig& InConfig)
{
	auto& ctx = OaContext::GetDefault();

	const OaU32 B = InConfig.Batch;
	const OaU32 L = InConfig.SeqLen;
	const OaU32 H = InConfig.NHeads;
	const OaU32 P = InConfig.HeadDim;
	const OaU32 N = InConfig.StateSize;
	const OaU32 A = InConfig.NumRopeAngles;

	// The Android NLP shape needs a bounded global-scratch specialization:
	// Turnip rejects both the 32 KiB shared-memory short kernel and the generic
	// MAXN=128 kernel. Desktop keeps its existing, faster shared-memory route.
#if defined(OA_PLATFORM_ANDROID)
	const bool mobileKernel = L <= 16u && P <= 16u && N <= 32u && A <= 8u;
#else
	const bool mobileKernel = false;
#endif
	const OaU32 chunk = mobileKernel ? 16u : 32u;
	const OaU32 nchunks = (L + chunk - 1u) / chunk;
	// Boundary/recompute scratch and the direct outputs are fully overwritten by
	// the kernel. Only dDt/dTrap use in-kernel read-modify-write and require zero.
	const bool shortKernel = !mobileKernel && L <= 16u && P <= 16u && N <= 32u && A <= 8u;
	const bool oneChunk = nchunks == 1u;
	OaMatrix entry = OaFnMatrix::Empty(oneChunk
		? OaMatrixShape{1}
		: OaMatrixShape{B * H, nchunks, P, N}, InX.Dtype_);
	OaMatrix thetaEnt = OaFnMatrix::Empty(oneChunk
		? OaMatrixShape{1}
		: OaMatrixShape{B * H, nchunks, A}, InX.Dtype_);
	OaMatrix chunkBuf = OaFnMatrix::Empty(shortKernel
		? OaMatrixShape{1}
		: OaMatrixShape{B * H, chunk, P, N}, InX.Dtype_);

	OaMatrix dC    = OaFnMatrix::Empty(InC.GetShape(),   InC.Dtype_);
	OaMatrix dB    = OaFnMatrix::Empty(InB.GetShape(),   InB.Dtype_);
	OaMatrix dX    = OaFnMatrix::Empty(InX.GetShape(),   InX.Dtype_);
	OaMatrix dZ    = OaFnMatrix::Empty(InZ.GetShape(),   InZ.Dtype_);
	OaMatrix dAdt  = OaFnMatrix::Empty(InAdt.GetShape(), InAdt.Dtype_);
	OaMatrix dDt   = OaFnMatrix::Zeros(InDt.GetShape(),  InDt.Dtype_);
	OaMatrix dTrapP = OaFnMatrix::Zeros(InTrap.GetShape(), InTrap.Dtype_);
	OaMatrix dAngH = OaFnMatrix::Empty(OaMatrixShape{B, H, L, A}, InAngle.Dtype_);
	OaMatrix dDTok = OaFnMatrix::Empty(InDt.GetShape(),  InDt.Dtype_);

	struct {
		OaU32 B, L, H, P, N, A, has_z, has_d;
	} push{ B, L, H, P, N, A, InConfig.HasZ, InConfig.HasD };

	OaBufferAccess access[] = {
		OaBufferAccess::Read,       // c
		OaBufferAccess::Read,       // b
		OaBufferAccess::Read,       // x
		OaBufferAccess::Read,       // z
		OaBufferAccess::Read,       // adt
		OaBufferAccess::Read,       // dt
		OaBufferAccess::Read,       // trap
		OaBufferAccess::Read,       // angle
		OaBufferAccess::Read,       // c_bias
		OaBufferAccess::Read,       // b_bias
		OaBufferAccess::Read,       // d
		OaBufferAccess::Read,       // d_out
		OaBufferAccess::ReadWrite,  // entry
		OaBufferAccess::ReadWrite,  // thetaEntry
		OaBufferAccess::ReadWrite,  // chunkBuf
		OaBufferAccess::Write,      // dc
		OaBufferAccess::Write,      // db
		OaBufferAccess::Write,      // dx
		OaBufferAccess::Write,      // dz
		OaBufferAccess::Write,      // dadt
		OaBufferAccess::ReadWrite,  // ddt
		OaBufferAccess::ReadWrite,  // dtrap
		OaBufferAccess::Write,      // dangle
		OaBufferAccess::Write       // ddtok
	};

	const char* kernel = mobileKernel
		? "Mamba3SisoBwdMobile"
		: shortKernel ? "Mamba3SisoBwdShort" : "Mamba3SisoBwd";
	ctx.Add(kernel,
		{&InC, &InB, &InX, &InZ, &InAdt, &InDt, &InTrap, &InAngle, &InCBias, &InBBias, &InD,
		 &InDOut, &entry, &thetaEnt, &chunkBuf,
		 &dC, &dB, &dX, &dZ, &dAdt, &dDt, &dTrapP, &dAngH, &dDTok},
		access, &push, sizeof(push), B * H, 1, 1);

	OaSsmBwdResult result;
	result.DC   = dC;
	result.DB   = dB;
	result.DX   = dX;
	result.DZ   = dZ;
	result.DAdt = dAdt;
	result.DDt  = dDt;

	OaMatrix trapS = OaFnMatrix::Sigmoid(InTrap);
	result.DTrap = OaFnMatrix::SigmoidBwd(trapS, dTrapP);

	result.DAngle = OaFnMatrix::Sum(dAngH, 1).Reshape(OaMatrixShape{B, L, A});

	result.DCBias = OaFnMatrix::Sum(dC.Reshape(OaMatrixShape{B * L, H * N}), 0).Reshape(OaMatrixShape{H, N});
	result.DBBias = OaFnMatrix::Sum(dB.Reshape(OaMatrixShape{B * L, H * N}), 0).Reshape(OaMatrixShape{H, N});
	result.DD     = OaFnMatrix::Sum(dDTok.Reshape(OaMatrixShape{B * L, H}), 0).Reshape(OaMatrixShape{H});

	return result;
}

// ============================================================================
// Mamba3Adt — fused per-token A·dt term (Ssm/Mamba3/Mamba3Adt.slang)
//
// Collapses HeavyTailActivation(6) + Neg + ClampMax + Mul (~9 elementwise
// dispatches) into one kernel.
// ============================================================================

OaMatrix OaFnMatrix::Mamba3Adt(
	const OaMatrix& InDdA, const OaMatrix& InDt, OaF32 InAFloor)
{
	auto& ctx = OaContext::GetDefault();
	OaMatrix out = OaFnMatrix::Empty(InDdA.Shape_, InDdA.Dtype_);
	OaU32 n = static_cast<OaU32>(InDdA.NumElements());

	struct { OaU32 count; OaF32 afloor; } push{ n, InAFloor };
	OaBufferAccess access[] = {
		OaBufferAccess::Read,   // dd_A
		OaBufferAccess::Read,   // dt
		OaBufferAccess::Write   // ADT
	};
	ctx.Add("Mamba3Adt", {&InDdA, &InDt, &out},
		access, &push, sizeof(push), OaDivCeil(n, 256));

	if (OaFnAutograd::IsEnabled() and (InDdA.RequiresGrad() or InDt.RequiresGrad())) {
		auto gradFn = OaMakeSharedPtr<OaGradMamba3Adt>(InAFloor);
		gradFn->Saved_ = OaVec<OaMatrix>{InDdA, InDt};
		gradFn->SetGraphInputs(OaVec<OaMatrix>{InDdA, InDt});
		gradFn->SequenceNr_  = OaFnAutograd::NextSeq();
		gradFn->OutputShape_ = out.GetShape();
		out.MutAutograd().GradFn = gradFn;
	}
	return out;
}

OaFnMatrix::OaMamba3AdtBwdResult OaFnMatrix::Mamba3AdtBwd(
	const OaMatrix& InDOut, const OaMatrix& InDdA, const OaMatrix& InDt, OaF32 InAFloor)
{
	auto& ctx = OaContext::GetDefault();
	OaMatrix dDdA = OaFnMatrix::Empty(InDdA.Shape_, InDdA.Dtype_);
	OaMatrix dDt  = OaFnMatrix::Empty(InDt.Shape_, InDt.Dtype_);
	OaU32 n = static_cast<OaU32>(InDdA.NumElements());

	struct { OaU32 count; OaF32 afloor; } push{ n, InAFloor };
	OaBufferAccess access[] = {
		OaBufferAccess::Read,   // dd_A
		OaBufferAccess::Read,   // dt
		OaBufferAccess::Read,   // d_out
		OaBufferAccess::Write,  // d_dd_A
		OaBufferAccess::Write   // d_dt
	};
	ctx.Add("Mamba3AdtBwd", {&InDdA, &InDt, &InDOut, &dDdA, &dDt},
		access, &push, sizeof(push), OaDivCeil(n, 256));

	return {.DDdA = dDdA, .DDt = dDt};
}

// ============================================================================
// Mamba3Dt — fused per-token dt term (Ssm/Mamba3/Mamba3Dt.slang)
//
// Collapses Softplus + ClampMin + ClampMax (3 elementwise dispatches) into
// one kernel.
// ============================================================================

OaMatrix OaFnMatrix::Mamba3Dt(
	const OaMatrix& InX, OaF32 InDtMin, OaF32 InDtMax)
{
	auto& ctx = OaContext::GetDefault();
	OaMatrix out = OaFnMatrix::Empty(InX.Shape_, InX.Dtype_);
	OaU32 n = static_cast<OaU32>(InX.NumElements());

	struct { OaU32 count; OaF32 dt_min; OaF32 dt_max; } push{ n, InDtMin, InDtMax };
	OaBufferAccess access[] = {
		OaBufferAccess::Read,   // x
		OaBufferAccess::Write   // DT
	};
	ctx.Add("Mamba3Dt", {&InX, &out},
		access, &push, sizeof(push), OaDivCeil(n, 256));

	if (OaFnAutograd::IsEnabled() and InX.RequiresGrad()) {
		auto gradFn = OaMakeSharedPtr<OaGradMamba3Dt>(InDtMin, InDtMax);
		gradFn->Saved_ = OaVec<OaMatrix>{InX};
		gradFn->SetGraphInputs(OaVec<OaMatrix>{InX});
		gradFn->SequenceNr_  = OaFnAutograd::NextSeq();
		gradFn->OutputShape_ = out.GetShape();
		out.MutAutograd().GradFn = gradFn;
	}
	return out;
}

OaMatrix OaFnMatrix::Mamba3DtBwd(
	const OaMatrix& InDOut, const OaMatrix& InX, OaF32 InDtMin, OaF32 InDtMax)
{
	auto& ctx = OaContext::GetDefault();
	OaMatrix dX = OaFnMatrix::Empty(InX.Shape_, InX.Dtype_);
	OaU32 n = static_cast<OaU32>(InX.NumElements());

	struct { OaU32 count; OaF32 dt_min; OaF32 dt_max; } push{ n, InDtMin, InDtMax };
	OaBufferAccess access[] = {
		OaBufferAccess::Read,   // x
		OaBufferAccess::Read,   // d_out
		OaBufferAccess::Write   // d_x
	};
	ctx.Add("Mamba3DtBwd", {&InX, &InDOut, &dX},
		access, &push, sizeof(push), OaDivCeil(n, 256));

	return dX;
}

// ============================================================================
// Mamba3Preprocess — fused in_proj split + RMSNorm + dt + A·dt (forward)
//
// Replaces 11-13 small dispatches with a single kernel. One workgroup per
// row of the projected [B*S, dInProj] tensor.
// ============================================================================

OaFnMatrix::OaMamba3PreprocessResult OaFnMatrix::Mamba3Preprocess(
	const OaMatrix& InProjected, const OaMatrix& InDtBias,
	const OaMamba3PreprocessConfig& InConfig)
{
	auto& ctx = OaContext::GetDefault();

	OaI32 batchTimesSeq = static_cast<OaI32>(InProjected.Size(0));
	OaI32 dInProj = static_cast<OaI32>(InProjected.Size(1));
	OaI32 dInner = InConfig.DInner;
	OaI32 dState = InConfig.DState;
	OaI32 nHeads = InConfig.NHeads;
	OaI32 numRopeAngles = InConfig.NumRopeAngles;
	OaI32 nGroups = InConfig.NGroups;
	OaI32 mimoRank = InConfig.MimoRank;

	// Offsets within each row (must match Preprocess() in Mamba3.cpp)
	OaI32 xOffset = dInner;
	OaI32 bOffset = 2 * dInner;
	OaI32 cOffset = bOffset + dState * nGroups * mimoRank;
	OaI32 ddDtOffset = cOffset + dState * nGroups * mimoRank;
	OaI32 ddAOffset = ddDtOffset + nHeads;
	OaI32 trapOffset = ddAOffset + nHeads;
	OaI32 angleOffset = trapOffset + nHeads;

	OaI32 bcWidth = dState * nGroups * mimoRank;

	// Allocate outputs
	OaMatrix xOut = OaFnMatrix::Empty(OaMatrixShape{batchTimesSeq, dInner}, InProjected.Dtype_);
	OaMatrix zOut = OaFnMatrix::Empty(OaMatrixShape{batchTimesSeq, dInner}, InProjected.Dtype_);
	OaMatrix bhOut = OaFnMatrix::Empty(OaMatrixShape{batchTimesSeq, bcWidth}, InProjected.Dtype_);
	OaMatrix chOut = OaFnMatrix::Empty(OaMatrixShape{batchTimesSeq, bcWidth}, InProjected.Dtype_);
	OaMatrix dtOut = OaFnMatrix::Empty(OaMatrixShape{batchTimesSeq, nHeads}, InProjected.Dtype_);
	OaMatrix adtOut = OaFnMatrix::Empty(OaMatrixShape{batchTimesSeq, nHeads}, InProjected.Dtype_);
	OaMatrix trapOut = OaFnMatrix::Empty(OaMatrixShape{batchTimesSeq, nHeads}, InProjected.Dtype_);
	OaMatrix angleOut = OaFnMatrix::Empty(OaMatrixShape{batchTimesSeq, numRopeAngles}, InProjected.Dtype_);

	struct Push {
		OaU32 rows, d_inner, d_state, n_heads, n_rope_angles;
		OaU32 n_bc_rows, bc_width;
		OaU32 z_offset, x_offset, b_offset, c_offset, dd_dt_offset, dd_a_offset, trap_offset, angle_offset;
		OaU32 d_in_proj;
		OaF32 eps, dt_min, dt_max, afloor;
	} push{
		static_cast<OaU32>(batchTimesSeq),
		static_cast<OaU32>(dInner),
		static_cast<OaU32>(dState),
		static_cast<OaU32>(nHeads),
		static_cast<OaU32>(numRopeAngles),
		static_cast<OaU32>(nGroups * mimoRank),
		static_cast<OaU32>(bcWidth),
		0,
		static_cast<OaU32>(xOffset),
		static_cast<OaU32>(bOffset),
		static_cast<OaU32>(cOffset),
		static_cast<OaU32>(ddDtOffset),
		static_cast<OaU32>(ddAOffset),
		static_cast<OaU32>(trapOffset),
		static_cast<OaU32>(angleOffset),
		static_cast<OaU32>(dInProj),
		InConfig.Eps, InConfig.DtMin, InConfig.DtMax, InConfig.AFloor
	};

	OaBufferAccess access[] = {
		OaBufferAccess::Read,   // projected
		OaBufferAccess::Read,   // dt_bias
		OaBufferAccess::Write,  // z
		OaBufferAccess::Write,  // x
		OaBufferAccess::Write,  // bh
		OaBufferAccess::Write,  // ch
		OaBufferAccess::Write,  // dt
		OaBufferAccess::Write,  // adt
		OaBufferAccess::Write,  // trap
		OaBufferAccess::Write   // angle
	};
	ctx.Add("Mamba3Preprocess",
		{&InProjected, &InDtBias, &zOut, &xOut, &bhOut, &chOut, &dtOut, &adtOut, &trapOut, &angleOut},
		access, &push, sizeof(push), static_cast<OaU32>(batchTimesSeq));

	// Autograd: create 8 thin grad nodes sharing one SharedState
	bool needGrad = OaFnAutograd::IsEnabled() and
		(InProjected.RequiresGrad() or InDtBias.RequiresGrad());
	if (needGrad) {
		auto state = OaMakeSharedPtr<OaGradMamba3Preprocess::SharedState>();
		state->Projected = InProjected;
		state->DtBias = InDtBias;
		state->Config = InConfig;

		auto attachGrad = [&](OaMatrix& out, OaI32 idx) {
			auto gradFn = OaMakeSharedPtr<OaGradMamba3Preprocess>();
			gradFn->State_ = state;
			gradFn->OutputIndex_ = idx;
			gradFn->SetGraphInputs(OaVec<OaMatrix>{InProjected, InDtBias});
			gradFn->SequenceNr_ = OaFnAutograd::NextSeq();
			gradFn->OutputShape_ = out.GetShape();
			out.MutAutograd().GradFn = gradFn;
			out.MutAutograd().RequiresGrad = true;
		};
		attachGrad(zOut, 0);
		attachGrad(xOut, 1);
		attachGrad(bhOut, 2);
		attachGrad(chOut, 3);
		attachGrad(dtOut, 4);
		attachGrad(adtOut, 5);
		attachGrad(trapOut, 6);
		attachGrad(angleOut, 7);
	}

	return {
		.X = xOut, .Z = zOut, .Bh = bhOut, .Ch = chOut,
		.DT = dtOut, .ADT = adtOut, .Trap = trapOut, .Angle = angleOut
	};
}

// ============================================================================
// Mamba3PreprocessBwd — fused backward (1 dispatch instead of 11+)
// ============================================================================

OaFnMatrix::OaMamba3PreprocessBwdResult OaFnMatrix::Mamba3PreprocessBwd(
	const OaMatrix& InProjected, const OaMatrix& InDtBias,
	const OaMatrix& InDZ, const OaMatrix& InDX,
	const OaMatrix& InDBh, const OaMatrix& InDCh,
	const OaMatrix& InDDT, const OaMatrix& InDADT,
	const OaMatrix& InDTrap, const OaMatrix& InDAngle,
	const OaMamba3PreprocessConfig& InConfig)
{
	auto& ctx = OaContext::GetDefault();

	OaI32 batchTimesSeq = static_cast<OaI32>(InProjected.Size(0));
	OaI32 dInProj = static_cast<OaI32>(InProjected.Size(1));
	OaI32 dInner = InConfig.DInner;
	OaI32 dState = InConfig.DState;
	OaI32 nHeads = InConfig.NHeads;
	OaI32 numRopeAngles = InConfig.NumRopeAngles;
	OaI32 nGroups = InConfig.NGroups;
	OaI32 mimoRank = InConfig.MimoRank;

	OaI32 xOffset = dInner;
	OaI32 bOffset = 2 * dInner;
	OaI32 cOffset = bOffset + dState * nGroups * mimoRank;
	OaI32 ddDtOffset = cOffset + dState * nGroups * mimoRank;
	OaI32 ddAOffset = ddDtOffset + nHeads;
	OaI32 trapOffset = ddAOffset + nHeads;
	OaI32 angleOffset = trapOffset + nHeads;
	OaI32 bcWidth = dState * nGroups * mimoRank;

	OaMatrix dProj = OaFnMatrix::Zeros(InProjected.Shape_, InProjected.Dtype_);
	OaMatrix dDtBias = OaFnMatrix::Zeros(OaMatrixShape{nHeads}, InProjected.Dtype_);

	struct Push {
		OaU32 rows, d_inner, d_state, n_heads, n_rope_angles;
		OaU32 n_bc_rows, bc_width;
		OaU32 z_offset, x_offset, b_offset, c_offset, dd_dt_offset, dd_a_offset, trap_offset, angle_offset;
		OaU32 d_in_proj;
		OaF32 eps, dt_min, dt_max, afloor;
	} push{
		static_cast<OaU32>(batchTimesSeq),
		static_cast<OaU32>(dInner),
		static_cast<OaU32>(dState),
		static_cast<OaU32>(nHeads),
		static_cast<OaU32>(numRopeAngles),
		static_cast<OaU32>(nGroups * mimoRank),
		static_cast<OaU32>(bcWidth),
		0,
		static_cast<OaU32>(xOffset),
		static_cast<OaU32>(bOffset),
		static_cast<OaU32>(cOffset),
		static_cast<OaU32>(ddDtOffset),
		static_cast<OaU32>(ddAOffset),
		static_cast<OaU32>(trapOffset),
		static_cast<OaU32>(angleOffset),
		static_cast<OaU32>(dInProj),
		InConfig.Eps, InConfig.DtMin, InConfig.DtMax, InConfig.AFloor
	};

	OaBufferAccess access[] = {
		OaBufferAccess::Read,   // projected
		OaBufferAccess::Read,   // dt_bias
		OaBufferAccess::Read,   // dz
		OaBufferAccess::Read,   // dx
		OaBufferAccess::Read,   // dbh
		OaBufferAccess::Read,   // dch
		OaBufferAccess::Read,   // ddt
		OaBufferAccess::Read,   // dadt
		OaBufferAccess::Read,   // dtrap
		OaBufferAccess::Read,   // dangle
		OaBufferAccess::Write,  // dproj
		OaBufferAccess::Write   // ddt_bias (atomic add)
	};
	ctx.Add("Mamba3PreprocessBwd",
		{&InProjected, &InDtBias, &InDZ, &InDX, &InDBh, &InDCh,
		 &InDDT, &InDADT, &InDTrap, &InDAngle, &dProj, &dDtBias},
		access, &push, sizeof(push), static_cast<OaU32>(batchTimesSeq));

	return {.DProjected = dProj, .DDtBias = dDtBias};
}
