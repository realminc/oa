// OaFnMatrix — Empyrealm SSM operations
//
// Empyrealm-specific SSM dispatch: fused dt + A·dt preprocess, selective scan,
// and backward passes. Kernels are renamed copies of the Mamba-3 versions,
// ready for future architecture-specific divergence.

#include <Oa/Ml/FnMatrix.h>
#include <Oa/Ml/Autograd/Nodes.h>
#include <Oa/Core/Matrix.h>
#include <Oa/Core/FnMatrix.h>
#include <Oa/Core/Status.h>
#include <Oa/Core/Types.h>
#include <Oa/Runtime/Context.h>

#include <cassert>

// ============================================================================
// EmpyrealmAdt — fused per-token A·dt term (Ssm/Empyrealm/EmpyrealmAdt.slang)
// ============================================================================

OaMatrix OaFnMatrix::EmpyrealmAdt(const OaMatrix& InDdA, const OaMatrix& InDt, OaF32 InAFloor) {
	auto& ctx = OaContext::GetDefault();
	OaMatrix out = OaFnMatrix::Empty(InDdA.Shape_, InDdA.Dtype_);
	OaU32 n = static_cast<OaU32>(InDdA.NumElements());

	struct { OaU32 count; OaF32 afloor; } push{ n, InAFloor };
	OaBufferAccess access[] = {
		OaBufferAccess::Read,   // dd_A
		OaBufferAccess::Read,   // dt
		OaBufferAccess::Write   // ADT
	};
	ctx.Add("EmpyrealmAdt", {&InDdA, &InDt, &out},
		access, &push, sizeof(push), OaDivCeil(n, 256));

	if (OaFnAutograd::IsEnabled() and (InDdA.RequiresGrad() or InDt.RequiresGrad())) {
		auto gradFn = OaMakeSharedPtr<OaGradEmpyrealmAdt>(InAFloor);
		gradFn->Saved_ = OaVec<OaMatrix>{InDdA, InDt};
		gradFn->SetGraphInputs(OaVec<OaMatrix>{InDdA, InDt});
		gradFn->SequenceNr_  = OaFnAutograd::NextSeq();
		gradFn->OutputShape_ = out.GetShape();
		out.MutAutograd().GradFn = gradFn;
	}
	return out;
}

OaFnMatrix::OaMamba3AdtBwdResult OaFnMatrix::EmpyrealmAdtBwd(
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
	ctx.Add("EmpyrealmAdtBwd", {&InDdA, &InDt, &InDOut, &dDdA, &dDt},
		access, &push, sizeof(push), OaDivCeil(n, 256));

	return {.DDdA = dDdA, .DDt = dDt};
}

// ============================================================================
// EmpyrealmDt — fused per-token dt term (Ssm/Empyrealm/EmpyrealmDt.slang)
//
// Collapses Softplus + ClampMin + ClampMax (3 elementwise dispatches) into
// one kernel.
// ============================================================================

OaMatrix OaFnMatrix::EmpyrealmDt(
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
	ctx.Add("EmpyrealmDt", {&InX, &out},
		access, &push, sizeof(push), OaDivCeil(n, 256));

	if (OaFnAutograd::IsEnabled() and InX.RequiresGrad()) {
		auto gradFn = OaMakeSharedPtr<OaGradEmpyrealmDt>(InDtMin, InDtMax);
		gradFn->Saved_ = OaVec<OaMatrix>{InX};
		gradFn->SetGraphInputs(OaVec<OaMatrix>{InX});
		gradFn->SequenceNr_  = OaFnAutograd::NextSeq();
		gradFn->OutputShape_ = out.GetShape();
		out.MutAutograd().GradFn = gradFn;
	}
	return out;
}

OaMatrix OaFnMatrix::EmpyrealmDtBwd(
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
	ctx.Add("EmpyrealmDtBwd", {&InX, &InDOut, &dX},
		access, &push, sizeof(push), OaDivCeil(n, 256));

	return dX;
}

// ============================================================================
// EmpyrealmSiso — selective state space scan (Ssm/Empyrealm/EmpyrealmSisoFwd.slang)
//
// Dispatches EmpyrealmSiso*.slang kernels, which are exact copies of the
// verified Mamba3Siso*.slang kernels. The dispatch signatures mirror Mamba3Siso
// exactly. EmpyrealmSiso returns raw [B,L,H,P] scan output (no in-kernel gating
// or norm), matching Mamba3Siso.
// ============================================================================

OaMatrix OaFnMatrix::EmpyrealmSiso(
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
	assert(InConfig.HeadDim <= 128 && "EmpyrealmSiso: headdim (P) must be <= 128");
	assert(InConfig.StateSize <= 128 && "EmpyrealmSiso: d_state (N) must be <= 128");
	assert(InConfig.NumRopeAngles <= 64 && "EmpyrealmSiso: num_rope_angles (A) must be <= 64");

	// Mixed precision: run the selective scan as an FP32 island (the kernels use the
	// always-fp32 Storage helpers and the cross-token state is precision-sensitive).
	// Cast operands up, scan, cast the result back — OaGradCast threads gradients.
	// Cast is a no-op on the recursive fp32 call. See [[oa-bf16-dtype-mess]].
	if (InX.GetDtype() == OaScalarType::BFloat16 || InX.GetDtype() == OaScalarType::Float16) {
		const auto up = [](const OaMatrix& m) { return OaFnMatrix::Cast(m, OaScalarType::Float32); };
		return OaFnMatrix::Cast(
			EmpyrealmSiso(up(InC), up(InB), up(InX), up(InZ), up(InAdt), up(InDt),
				up(InTrap), up(InAngle), up(InCBias), up(InBBias), up(InD), InConfig),
			InX.GetDtype());
	}

	auto& ctx = OaContext::GetDefault();

	OaMatrix output = OaFnMatrix::Zeros(
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

	ctx.Add("EmpyrealmSisoFwd",
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
// EmpyrealmSisoStep — single-token step for autoregressive generation
// ============================================================================

OaMatrix OaFnMatrix::EmpyrealmSisoStep(
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
	assert(InConfig.SeqLen == 1 && "EmpyrealmSisoStep: SeqLen must be 1");
	auto& ctx = OaContext::GetDefault();

	OaMatrix output = OaFnMatrix::Zeros(
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

	ctx.Add("EmpyrealmSisoStep",
		{&InC, &InB, &InX, &InZ, &InAdt, &InDt, &InTrap, &InAngle, &InCBias, &InBBias, &InD,
		 &output, &InSsmState, &InAngleState, &InKState, &InVState},
		access, &push, sizeof(push), InConfig.Batch * InConfig.NHeads, 1, 1);

	return output;
}

// ============================================================================
// EmpyrealmSisoBwd — backward pass for the selective scan
// ============================================================================

OaFnMatrix::OaSsmBwdResult OaFnMatrix::EmpyrealmSisoBwd(
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

	const OaU32 CHUNK = 32u;
	const OaU32 nchunks = (L + CHUNK - 1u) / CHUNK;
	OaMatrix entry    = OaFnMatrix::Zeros(OaMatrixShape{B * H, nchunks, P, N}, InX.Dtype_);
	OaMatrix thetaEnt = OaFnMatrix::Zeros(OaMatrixShape{B * H, nchunks, A}, InX.Dtype_);
	OaMatrix chunkBuf = OaFnMatrix::Zeros(OaMatrixShape{B * H, CHUNK, P, N}, InX.Dtype_);

	OaMatrix dC    = OaFnMatrix::Zeros(InC.GetShape(),   InC.Dtype_);
	OaMatrix dB    = OaFnMatrix::Zeros(InB.GetShape(),   InB.Dtype_);
	OaMatrix dX    = OaFnMatrix::Zeros(InX.GetShape(),   InX.Dtype_);
	OaMatrix dZ    = OaFnMatrix::Zeros(InZ.GetShape(),   InZ.Dtype_);
	OaMatrix dAdt  = OaFnMatrix::Zeros(InAdt.GetShape(), InAdt.Dtype_);
	OaMatrix dDt   = OaFnMatrix::Zeros(InDt.GetShape(),  InDt.Dtype_);
	OaMatrix dTrapP = OaFnMatrix::Zeros(InTrap.GetShape(), InTrap.Dtype_);
	OaMatrix dAngH = OaFnMatrix::Zeros(OaMatrixShape{B, H, L, A}, InAngle.Dtype_);
	OaMatrix dDTok = OaFnMatrix::Zeros(InDt.GetShape(),  InDt.Dtype_);

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

	ctx.Add("EmpyrealmSisoBwd",
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
// EmpyrealmDtAdt — fused dt + A·dt forward (Ssm/Empyrealm/EmpyrealmDtAdt.slang)
//
// Replaces two sequential dispatches (EmpyrealmDt + EmpyrealmAdt) with one.
// Backward uses the existing separate EmpyrealmDtBwd + EmpyrealmAdtBwd kernels
// via separate autograd nodes on each output.
// ============================================================================

OaFnMatrix::OaEmpyrealmDtAdtResult OaFnMatrix::EmpyrealmDtAdt(
	const OaMatrix& InDtRaw, const OaMatrix& InDdA,
	OaF32 InDtMin, OaF32 InDtMax, OaF32 InAFloor)
{
	auto& ctx = OaContext::GetDefault();
	OaU32 n = static_cast<OaU32>(InDtRaw.NumElements());

	OaMatrix dt  = OaFnMatrix::Empty(InDtRaw.Shape_, InDtRaw.Dtype_);
	OaMatrix adt = OaFnMatrix::Empty(InDtRaw.Shape_, InDtRaw.Dtype_);

	struct {
		OaU32 count;
		OaF32 dt_min, dt_max, afloor;
	} push{n, InDtMin, InDtMax, InAFloor};

	OaBufferAccess access[] = {
		OaBufferAccess::Read,   // dtRaw
		OaBufferAccess::Read,   // ddA
		OaBufferAccess::Write,  // DT
		OaBufferAccess::Write   // ADT
	};
	ctx.Add("EmpyrealmDtAdt", {&InDtRaw, &InDdA, &dt, &adt},
		access, &push, sizeof(push), OaDivCeil(n, 256));

	if (OaFnAutograd::IsEnabled() and (InDtRaw.RequiresGrad() or InDdA.RequiresGrad())) {
		auto dtGrad = OaMakeSharedPtr<OaGradEmpyrealmDt>(InDtMin, InDtMax);
		dtGrad->Saved_ = OaVec<OaMatrix>{InDtRaw};
		dtGrad->SetGraphInputs(OaVec<OaMatrix>{InDtRaw});
		dtGrad->SequenceNr_  = OaFnAutograd::NextSeq();
		dtGrad->OutputShape_ = dt.GetShape();
		dt.MutAutograd().GradFn = dtGrad;

		auto adtGrad = OaMakeSharedPtr<OaGradEmpyrealmAdt>(InAFloor);
		adtGrad->Saved_ = OaVec<OaMatrix>{InDdA, dt};
		adtGrad->SetGraphInputs(OaVec<OaMatrix>{InDdA, dt});
		adtGrad->SequenceNr_  = OaFnAutograd::NextSeq();
		adtGrad->OutputShape_ = adt.GetShape();
		adt.MutAutograd().GradFn = adtGrad;
	}

	return {.DT = dt, .ADT = adt};
}

// ============================================================================
// EmpyrealmPreprocess — fused in_proj split + RMSNorm + dt + A·dt (forward)
//
// Renamed copy of Mamba3Preprocess; dispatches Ssm/Empyrealm/EmpyrealmPreprocess.
// ============================================================================

OaFnMatrix::OaMamba3PreprocessResult OaFnMatrix::EmpyrealmPreprocess(
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

	OaI32 xOffset = dInner;
	OaI32 bOffset = 2 * dInner;
	OaI32 cOffset = bOffset + dState * nGroups * mimoRank;
	OaI32 ddDtOffset = cOffset + dState * nGroups * mimoRank;
	OaI32 ddAOffset = ddDtOffset + nHeads;
	OaI32 trapOffset = ddAOffset + nHeads;
	OaI32 angleOffset = trapOffset + nHeads;

	OaI32 bcWidth = dState * nGroups * mimoRank;

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
	ctx.Add("EmpyrealmPreprocess",
		{&InProjected, &InDtBias, &zOut, &xOut, &bhOut, &chOut, &dtOut, &adtOut, &trapOut, &angleOut},
		access, &push, sizeof(push), static_cast<OaU32>(batchTimesSeq));

	bool needGrad = OaFnAutograd::IsEnabled() and
		(InProjected.RequiresGrad() or InDtBias.RequiresGrad());
	if (needGrad) {
		auto state = OaMakeSharedPtr<OaGradMamba3Preprocess::SharedState>();
		state->Projected = InProjected;
		state->DtBias = InDtBias;
		state->Config = InConfig;

		auto attachGrad = [&](OaMatrix& out, OaI32 idx) {
			auto gradFn = OaMakeSharedPtr<OaGradEmpyrealmPreprocess>();
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
// EmpyrealmPreprocessBwd — fused backward (1 dispatch instead of 11+)
// ============================================================================

OaFnMatrix::OaMamba3PreprocessBwdResult OaFnMatrix::EmpyrealmPreprocessBwd(
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
	ctx.Add("EmpyrealmPreprocessBwd",
		{&InProjected, &InDtBias, &InDZ, &InDX, &InDBh, &InDCh,
		 &InDDT, &InDADT, &InDTrap, &InDAngle, &dProj, &dDtBias},
		access, &push, sizeof(push), static_cast<OaU32>(batchTimesSeq));

	return {.DProjected = dProj, .DDtBias = dDtBias};
}
