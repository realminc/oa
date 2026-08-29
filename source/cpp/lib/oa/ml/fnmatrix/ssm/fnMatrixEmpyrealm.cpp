// oa::FnMatrix — empyrealm SSM operations
//
// empyrealm-specific SSM dispatch: fused dt + A·dt preprocess, selective scan,
// and backward passes. Kernels are renamed copies of the Mamba-3 versions,
// ready for future architecture-specific divergence.

#include <oa/ml/fnMatrix.h>
#include <oa/core/autograd/matrix/autogradDtype.h>
#include <oa/ml/autograd/matrix/autogradSsm.h>
#include <oa/core/matrix.h>
#include <oa/core/fnMatrix.h>
#include <oa/core/op.h>
#include <oa/core/status.h>
#include <oa/core/types.h>
#include <oa/runtime/executionSession.h>

#include "fnMatrixSsmInternal.h"

#include <oa/core/std/assert.h>

// ============================================================================
// EmpyrealmAdt — fused per-token A·dt term (Ssm/empyrealm/EmpyrealmAdt.slang)
// ============================================================================

oa::Matrix oa::FnMatrix::empyrealmAdt(const oa::Matrix& inDdA, const oa::Matrix& inDt, oa::F32 inAFloor) {
	auto& ctx = oa::ExecutionSession::getActive();
	oa::OpLoweringScope lowering(ctx);
	oa::Matrix out = oa::FnMatrix::empty(inDdA.getShape(), inDdA.getDtype());
	oa::U32 n = static_cast<oa::U32>(inDdA.numElements());

	struct { oa::U32 count; oa::F32 afloor; } push{ n, inAFloor };
	oa::BufferAccess access[] = {
		oa::BufferAccess::Read,   // dd_A
		oa::BufferAccess::Read,   // dt
		oa::BufferAccess::Write   // ADT
	};
	ctx.add( "EmpyrealmAdt", {&inDdA, &inDt, &out},
		access, &push, sizeof(push), oa::divCeil(n, 256));
	const auto semantic = lowering.commitWithId(
		oa::detail::opRegistry::FnMatrix::empyrealmAdt,
		{&inDdA, &inDt},
		{&out},
		{oa::OpAttribute::fromFloat("aFloor", inAFloor)});
	if (not semantic.isOk()) return {};

	if (oa::FnAutograd::isEnabled() and (inDdA.requiresGrad() or inDt.requiresGrad())) {
		auto gradFn = oa::makeShared<oa::GradEmpyrealmAdt>(inAFloor);
		gradFn->saveForBackward(inDdA, inDt);
		gradFn->setGraphInputs(oa::Vector<oa::Matrix>{inDdA, inDt});
		gradFn->sequenceNr_  = oa::FnAutograd::nextSeq();
		gradFn->outputShape_ = out.getShape();
		if (not oa::FnAutograd::attachSemantic(
			gradFn, semantic.getValue()).isOk())
		{
			return {};
		}
		out.mutAutograd().gradFn = gradFn;
	}
	return out;
}

oa::EmpyrealmAdtBwdResult oa::FnMatrix::empyrealmAdtBwd(
	const oa::Matrix& inDOut, const oa::Matrix& inDdA, const oa::Matrix& inDt, oa::F32 inAFloor)
{
	auto& ctx = oa::ExecutionSession::getActive();
	oa::OpLoweringScope lowering(ctx);
	oa::Matrix dDdA = oa::FnMatrix::empty(inDdA.getShape(), inDdA.getDtype());
	oa::Matrix dDt  = oa::FnMatrix::empty(inDt.getShape(), inDt.getDtype());
	oa::U32 n = static_cast<oa::U32>(inDdA.numElements());

	struct { oa::U32 count; oa::F32 afloor; } push{ n, inAFloor };
	oa::BufferAccess access[] = {
		oa::BufferAccess::Read,   // dd_A
		oa::BufferAccess::Read,   // dt
		oa::BufferAccess::Read,   // d_out
		oa::BufferAccess::Write,  // d_dd_A
		oa::BufferAccess::Write   // d_dt
	};
	ctx.add( "EmpyrealmAdtBwd", {&inDdA, &inDt, &inDOut, &dDdA, &dDt},
		access, &push, sizeof(push), oa::divCeil(n, 256));

	if (not lowering.commit(
		oa::detail::opRegistry::FnMatrix::empyrealmAdtBwd,
		{&inDOut, &inDdA, &inDt},
		{&dDdA, &dDt},
		{oa::OpAttribute::fromFloat("aFloor", inAFloor)}).isOk())
	{
		return {};
	}
	return {.dDdA = dDdA, .dDt = dDt};
}

// ============================================================================
// EmpyrealmDt — fused per-token dt term (Ssm/empyrealm/EmpyrealmDt.slang)
//
// Collapses Softplus + ClampMin + clampMax (3 elementwise dispatches) into
// one kernel.
// ============================================================================

oa::Matrix oa::FnMatrix::empyrealmDt(
	const oa::Matrix& inX, oa::F32 inDtMin, oa::F32 inDtMax)
{
	auto& ctx = oa::ExecutionSession::getActive();
	oa::OpLoweringScope lowering(ctx);
	oa::Matrix out = oa::FnMatrix::empty(inX.getShape(), inX.getDtype());
	oa::U32 n = static_cast<oa::U32>(inX.numElements());

	struct { oa::U32 count; oa::F32 dt_min; oa::F32 dt_max; } push{ n, inDtMin, inDtMax };
	oa::BufferAccess access[] = {
		oa::BufferAccess::Read,   // x
		oa::BufferAccess::Write   // DT
	};
	ctx.add( "EmpyrealmDt", {&inX, &out},
		access, &push, sizeof(push), oa::divCeil(n, 256));
	const auto semantic = lowering.commitWithId(
		oa::detail::opRegistry::FnMatrix::empyrealmDt,
		{&inX},
		{&out},
		{oa::OpAttribute::fromFloat("dtMin", inDtMin),
			oa::OpAttribute::fromFloat("dtMax", inDtMax)});
	if (not semantic.isOk()) return {};

	if (oa::FnAutograd::isEnabled() and inX.requiresGrad()) {
		auto gradFn = oa::makeShared<oa::GradEmpyrealmDt>(inDtMin, inDtMax);
		gradFn->saveForBackward(inX);
		gradFn->setGraphInputs(oa::Vector<oa::Matrix>{inX});
		gradFn->sequenceNr_  = oa::FnAutograd::nextSeq();
		gradFn->outputShape_ = out.getShape();
		if (not oa::FnAutograd::attachSemantic(
			gradFn, semantic.getValue()).isOk())
		{
			return {};
		}
		out.mutAutograd().gradFn = gradFn;
	}
	return out;
}

oa::Matrix oa::FnMatrix::empyrealmDtBwd(
	const oa::Matrix& inDOut, const oa::Matrix& inX, oa::F32 inDtMin, oa::F32 inDtMax)
{
	auto& ctx = oa::ExecutionSession::getActive();
	oa::OpLoweringScope lowering(ctx);
	oa::Matrix dX = oa::FnMatrix::empty(inX.getShape(), inX.getDtype());
	oa::U32 n = static_cast<oa::U32>(inX.numElements());

	struct { oa::U32 count; oa::F32 dt_min; oa::F32 dt_max; } push{ n, inDtMin, inDtMax };
	oa::BufferAccess access[] = {
		oa::BufferAccess::Read,   // x
		oa::BufferAccess::Read,   // d_out
		oa::BufferAccess::Write   // d_x
	};
	ctx.add( "EmpyrealmDtBwd", {&inX, &inDOut, &dX},
		access, &push, sizeof(push), oa::divCeil(n, 256));

	if (not lowering.commit(
		oa::detail::opRegistry::FnMatrix::empyrealmDtBwd,
		{&inDOut, &inX},
		{&dX},
		{oa::OpAttribute::fromFloat("dtMin", inDtMin),
			oa::OpAttribute::fromFloat("dtMax", inDtMax)}).isOk())
	{
		return {};
	}
	return dX;
}

// ============================================================================
// EmpyrealmSiso — selective state space scan (Ssm/empyrealm/EmpyrealmSisoFwd.slang)
//
// Dispatches EmpyrealmSiso*.slang kernels, which are exact copies of the
// verified Mamba3Siso*.slang kernels. The dispatch signatures mirror Mamba3Siso
// exactly. EmpyrealmSiso returns raw [B,L,H,P] scan output (no in-kernel gating
// or norm), matching Mamba3Siso.
// ============================================================================

oa::Matrix oa::FnMatrix::empyrealmSiso(
	const oa::Matrix& inC,
	const oa::Matrix& inB,
	const oa::Matrix& inX,
	const oa::Matrix& inZ,
	const oa::Matrix& inAdt,
	const oa::Matrix& inDt,
	const oa::Matrix& inTrap,
	const oa::Matrix& inAngle,
	const oa::Matrix& inCBias,
	const oa::Matrix& inBBias,
	const oa::Matrix& inD,
	const oa::SsmConfig& inConfig)
{
	OA_REQUIRE_MSG(inConfig.headDim <= 128, "EmpyrealmSiso: headdim (P) must be <= 128");
	OA_REQUIRE_MSG(inConfig.stateSize <= 128, "EmpyrealmSiso: d_state (N) must be <= 128");
	OA_REQUIRE_MSG(inConfig.numRopeAngles <= 64, "EmpyrealmSiso: num_rope_angles (A) must be <= 64");
	auto& ctx = oa::ExecutionSession::getActive();
	oa::OpLoweringScope lowering(ctx);
	const oa::U64 configIdentity =
		oa::FnMatrixPrivate::ssmConfigIdentity(inConfig);

	// Mixed precision: run the selective scan as an FP32 island (the kernels use the
	// always-fp32 storage helpers and the cross-token state is precision-sensitive).
	// Cast operands up, scan, cast the result back — oa::GradCast threads gradients.
	// Cast is a no-op on the recursive fp32 call. See [[oa-bf16-dtype-mess]].
	if (inX.getDtype() == oa::ScalarType::BFloat16) {
		const auto up = [](const oa::Matrix& m) { return oa::FnMatrix::cast(m, oa::ScalarType::Float32); };
		auto output = oa::FnMatrix::cast(
			empyrealmSiso(up(inC), up(inB), up(inX), up(inZ), up(inAdt), up(inDt),
				up(inTrap), up(inAngle), up(inCBias), up(inBBias), up(inD), inConfig),
			inX.getDtype());
		const auto semantic = lowering.commitWithId(
			oa::detail::opRegistry::FnMatrix::empyrealmSiso,
			{&inC, &inB, &inX, &inZ, &inAdt, &inDt, &inTrap, &inAngle,
				&inCBias, &inBBias, &inD},
			{&output},
			{oa::OpAttribute::fromUnsignedInteger(
				"configIdentity", configIdentity)});
		if (not semantic.isOk()) return {};
		if (auto grad = output.getGradFn()) {
			if (not oa::FnAutograd::attachSemantic(
				grad, semantic.getValue()).isOk())
			{
				return {};
			}
		}
		return output;
	}

	// Every active output lane is written exactly once by EmpyrealmSisoFwd.
	oa::Matrix output = oa::FnMatrix::empty(
		oa::MatrixShape{inConfig.batch, inConfig.seqLen, inConfig.nHeads, inConfig.headDim},
		inX.getDtype());

	struct {
		oa::U32 B, L, H, P, N, A, has_z, has_d;
	} push{
		inConfig.batch, inConfig.seqLen, inConfig.nHeads, inConfig.headDim,
		inConfig.stateSize, inConfig.numRopeAngles, inConfig.hasZ, inConfig.hasD
	};

	oa::BufferAccess access[] = {
		oa::BufferAccess::Read,  // c
		oa::BufferAccess::Read,  // b
		oa::BufferAccess::Read,  // x
		oa::BufferAccess::Read,  // z
		oa::BufferAccess::Read,  // adt
		oa::BufferAccess::Read,  // dt
		oa::BufferAccess::Read,  // trap
		oa::BufferAccess::Read,  // angle
		oa::BufferAccess::Read,  // c_bias
		oa::BufferAccess::Read,  // b_bias
		oa::BufferAccess::Read,  // d
		oa::BufferAccess::Write  // y
	};

	ctx.add( "EmpyrealmSisoFwd",
		{&inC, &inB, &inX, &inZ, &inAdt, &inDt, &inTrap, &inAngle, &inCBias, &inBBias, &inD, &output},
		access, &push, sizeof(push), inConfig.batch * inConfig.nHeads, 1, 1);
	const auto semantic = lowering.commitWithId(
		oa::detail::opRegistry::FnMatrix::empyrealmSiso,
		{&inC, &inB, &inX, &inZ, &inAdt, &inDt, &inTrap, &inAngle,
			&inCBias, &inBBias, &inD},
		{&output},
		{oa::OpAttribute::fromUnsignedInteger(
			"configIdentity", configIdentity)});
	if (not semantic.isOk()) return {};

	if (oa::FnAutograd::isEnabled() and
		(inC.requiresGrad() or inB.requiresGrad() or inX.requiresGrad() or
		 inZ.requiresGrad() or inAdt.requiresGrad() or inDt.requiresGrad() or
		 inTrap.requiresGrad() or inAngle.requiresGrad() or inCBias.requiresGrad() or
		 inBBias.requiresGrad() or inD.requiresGrad())) {
		auto gradFn = oa::makeShared<oa::GradEmpyrealmSiso>();
		gradFn->saveForBackward(inC, inB, inX, inZ, inAdt, inDt, inTrap,
			inAngle, inCBias, inBBias, inD);
		gradFn->setGraphInputs(oa::Vector<oa::Matrix>{inC, inB, inX, inZ, inAdt, inDt, inTrap,
			inAngle, inCBias, inBBias, inD});
		gradFn->sequenceNr_ = oa::FnAutograd::nextSeq();
		gradFn->config_ = inConfig;
		gradFn->outputShape_ = output.getShape();
		if (not oa::FnAutograd::attachSemantic(
			gradFn, semantic.getValue()).isOk())
		{
			return {};
		}
		output.mutAutograd().gradFn = gradFn;
	}

	return output;
}

// ============================================================================
// EmpyrealmSisoStep — single-token step for autoregressive generation
// ============================================================================

oa::Matrix oa::FnMatrix::empyrealmSisoStep(
	const oa::Matrix& inC,
	const oa::Matrix& inB,
	const oa::Matrix& inX,
	const oa::Matrix& inZ,
	const oa::Matrix& inAdt,
	const oa::Matrix& inDt,
	const oa::Matrix& inTrap,
	const oa::Matrix& inAngle,
	const oa::Matrix& inCBias,
	const oa::Matrix& inBBias,
	const oa::Matrix& inD,
	const oa::Matrix& inSsmState,
	const oa::Matrix& inAngleState,
	const oa::Matrix& inKState,
	const oa::Matrix& inVState,
	const oa::SsmConfig& inConfig)
{
	OA_REQUIRE_MSG(inConfig.seqLen == 1, "EmpyrealmSisoStep: seqLen must be 1");
	auto& ctx = oa::ExecutionSession::getActive();
	oa::OpLoweringScope lowering(ctx);
	const oa::U64 configIdentity =
		oa::FnMatrixPrivate::ssmConfigIdentity(inConfig);

	// Every active output lane is written exactly once by EmpyrealmSisoStep.
	oa::Matrix output = oa::FnMatrix::empty(
		oa::MatrixShape{inConfig.batch, 1, inConfig.nHeads, inConfig.headDim}, inX.getDtype());

	struct {
		oa::U32 B, H, P, N, A, has_z, has_d;
	} push{ inConfig.batch, inConfig.nHeads, inConfig.headDim, inConfig.stateSize,
		inConfig.numRopeAngles, inConfig.hasZ, inConfig.hasD };

	oa::BufferAccess access[] = {
		oa::BufferAccess::Read,       // c
		oa::BufferAccess::Read,       // b
		oa::BufferAccess::Read,       // x
		oa::BufferAccess::Read,       // z
		oa::BufferAccess::Read,       // adt
		oa::BufferAccess::Read,       // dt
		oa::BufferAccess::Read,       // trap
		oa::BufferAccess::Read,       // angle
		oa::BufferAccess::Read,       // c_bias
		oa::BufferAccess::Read,       // b_bias
		oa::BufferAccess::Read,       // d
		oa::BufferAccess::Write,      // y
		oa::BufferAccess::ReadWrite,  // ssm_state
		oa::BufferAccess::ReadWrite,  // angle_state
		oa::BufferAccess::ReadWrite,  // k_state
		oa::BufferAccess::ReadWrite   // v_state
	};

	ctx.add( "EmpyrealmSisoStep",
		{&inC, &inB, &inX, &inZ, &inAdt, &inDt, &inTrap, &inAngle, &inCBias, &inBBias, &inD,
		 &output, &inSsmState, &inAngleState, &inKState, &inVState},
		access, &push, sizeof(push), inConfig.batch * inConfig.nHeads, 1, 1);

	if (not lowering.commit(
		oa::detail::opRegistry::FnMatrix::empyrealmSisoStep,
		{&inSsmState, &inAngleState, &inKState, &inVState,
			&inC, &inB, &inX, &inZ, &inAdt, &inDt, &inTrap, &inAngle,
			&inCBias, &inBBias, &inD},
		{&output},
		{oa::OpAttribute::fromUnsignedInteger(
			"configIdentity", configIdentity)}).isOk())
	{
		return {};
	}
	return output;
}

// ============================================================================
// EmpyrealmSisoBwd — backward pass for the selective scan
// ============================================================================

oa::SsmBwdResult oa::FnMatrix::empyrealmSisoBwd(
	const oa::Matrix& inDOut,
	const oa::Matrix& inC,
	const oa::Matrix& inB,
	const oa::Matrix& inX,
	const oa::Matrix& inZ,
	const oa::Matrix& inAdt,
	const oa::Matrix& inDt,
	const oa::Matrix& inTrap,
	const oa::Matrix& inAngle,
	const oa::Matrix& inCBias,
	const oa::Matrix& inBBias,
	const oa::Matrix& inD,
	const oa::SsmConfig& inConfig)
{
	auto& ctx = oa::ExecutionSession::getActive();
	oa::OpLoweringScope lowering(ctx);
	const oa::U64 configIdentity =
		oa::FnMatrixPrivate::ssmConfigIdentity(inConfig);

	const oa::U32 B = inConfig.batch;
	const oa::U32 L = inConfig.seqLen;
	const oa::U32 H = inConfig.nHeads;
	const oa::U32 P = inConfig.headDim;
	const oa::U32 N = inConfig.stateSize;
	const oa::U32 A = inConfig.numRopeAngles;

	const oa::U32 CHUNK = 32u;
	const oa::U32 nchunks = (L + CHUNK - 1u) / CHUNK;
	// The generic backward kernel fully overwrites scratch, direct adjoints,
	// angle adjoints, and the per-token D adjoint. Only dDt and dTrap use
	// read-modify-write accumulation and therefore require zero initialization.
	oa::Matrix entry    = oa::FnMatrix::empty(oa::MatrixShape{B * H, nchunks, P, N}, inX.getDtype());
	oa::Matrix thetaEnt = oa::FnMatrix::empty(oa::MatrixShape{B * H, nchunks, A}, inX.getDtype());
	oa::Matrix chunkBuf = oa::FnMatrix::empty(oa::MatrixShape{B * H, CHUNK, P, N}, inX.getDtype());

	oa::Matrix dC    = oa::FnMatrix::empty(inC.getShape(),   inC.getDtype());
	oa::Matrix dB    = oa::FnMatrix::empty(inB.getShape(),   inB.getDtype());
	oa::Matrix dX    = oa::FnMatrix::empty(inX.getShape(),   inX.getDtype());
	oa::Matrix dZ    = oa::FnMatrix::empty(inZ.getShape(),   inZ.getDtype());
	oa::Matrix dAdt  = oa::FnMatrix::empty(inAdt.getShape(), inAdt.getDtype());
	oa::Matrix dDt   = oa::FnMatrix::zeros(inDt.getShape(),  inDt.getDtype());
	oa::Matrix dTrapP = oa::FnMatrix::zeros(inTrap.getShape(), inTrap.getDtype());
	oa::Matrix dAngH = oa::FnMatrix::empty(oa::MatrixShape{B, H, L, A}, inAngle.getDtype());
	oa::Matrix dDTok = oa::FnMatrix::empty(inDt.getShape(),  inDt.getDtype());

	struct {
		oa::U32 B, L, H, P, N, A, has_z, has_d;
	} push{ B, L, H, P, N, A, inConfig.hasZ, inConfig.hasD };

	oa::BufferAccess access[] = {
		oa::BufferAccess::Read,       // c
		oa::BufferAccess::Read,       // b
		oa::BufferAccess::Read,       // x
		oa::BufferAccess::Read,       // z
		oa::BufferAccess::Read,       // adt
		oa::BufferAccess::Read,       // dt
		oa::BufferAccess::Read,       // trap
		oa::BufferAccess::Read,       // angle
		oa::BufferAccess::Read,       // c_bias
		oa::BufferAccess::Read,       // b_bias
		oa::BufferAccess::Read,       // d
		oa::BufferAccess::Read,       // d_out
		oa::BufferAccess::ReadWrite,  // entry
		oa::BufferAccess::ReadWrite,  // thetaEntry
		oa::BufferAccess::ReadWrite,  // chunkBuf
		oa::BufferAccess::Write,      // dc
		oa::BufferAccess::Write,      // db
		oa::BufferAccess::Write,      // dx
		oa::BufferAccess::Write,      // dz
		oa::BufferAccess::Write,      // dadt
		oa::BufferAccess::ReadWrite,  // ddt
		oa::BufferAccess::ReadWrite,  // dtrap
		oa::BufferAccess::Write,      // dangle
		oa::BufferAccess::Write       // ddtok
	};

	ctx.add( "EmpyrealmSisoBwd",
		{&inC, &inB, &inX, &inZ, &inAdt, &inDt, &inTrap, &inAngle, &inCBias, &inBBias, &inD,
		 &inDOut, &entry, &thetaEnt, &chunkBuf,
		 &dC, &dB, &dX, &dZ, &dAdt, &dDt, &dTrapP, &dAngH, &dDTok},
		access, &push, sizeof(push), B * H, 1, 1);

	oa::SsmBwdResult result;
	result.dC   = dC;
	result.dB   = dB;
	result.dX   = dX;
	result.dZ   = dZ;
	result.dAdt = dAdt;
	result.dDt  = dDt;

	oa::Matrix trapS = oa::FnMatrix::sigmoid(inTrap);
	result.dTrap = oa::FnMatrix::sigmoidBwd(trapS, dTrapP);

	result.dAngle = oa::FnMatrix::sum(dAngH, 1).reshape(oa::MatrixShape{B, L, A});

	result.dCBias = oa::FnMatrix::sum(dC.reshape(oa::MatrixShape{B * L, H * N}), 0).reshape(oa::MatrixShape{H, N});
	result.dBBias = oa::FnMatrix::sum(dB.reshape(oa::MatrixShape{B * L, H * N}), 0).reshape(oa::MatrixShape{H, N});
	result.dD     = oa::FnMatrix::sum(dDTok.reshape(oa::MatrixShape{B * L, H}), 0).reshape(oa::MatrixShape{H});

	if (not lowering.commit(
		oa::detail::opRegistry::FnMatrix::empyrealmSisoBwd,
		{&inDOut, &inC, &inB, &inX, &inZ, &inAdt, &inDt, &inTrap,
			&inAngle, &inCBias, &inBBias, &inD},
		{&result.dC, &result.dB, &result.dX, &result.dZ, &result.dAdt,
			&result.dDt, &result.dTrap, &result.dAngle, &result.dCBias,
			&result.dBBias, &result.dD},
		{oa::OpAttribute::fromUnsignedInteger(
			"configIdentity", configIdentity)}).isOk())
	{
		return {};
	}
	return result;
}

// ============================================================================
// EmpyrealmDtAdt — fused dt + A·dt forward (Ssm/empyrealm/EmpyrealmDtAdt.slang)
//
// Replaces two sequential dispatches (EmpyrealmDt + EmpyrealmAdt) with one.
// backward uses the existing separate EmpyrealmDtBwd + EmpyrealmAdtBwd kernels
// via separate autograd nodes on each output.
// ============================================================================

oa::EmpyrealmDtAdtResult oa::FnMatrix::empyrealmDtAdt(
	const oa::Matrix& inDtRaw, const oa::Matrix& inDdA,
	oa::F32 inDtMin, oa::F32 inDtMax, oa::F32 inAFloor)
{
	auto& ctx = oa::ExecutionSession::getActive();
	oa::OpLoweringScope lowering(ctx);
	oa::U32 n = static_cast<oa::U32>(inDtRaw.numElements());

	oa::Matrix dt  = oa::FnMatrix::empty(inDtRaw.getShape(), inDtRaw.getDtype());
	oa::Matrix adt = oa::FnMatrix::empty(inDtRaw.getShape(), inDtRaw.getDtype());

	struct {
		oa::U32 count;
		oa::F32 dt_min, dt_max, afloor;
	} push{n, inDtMin, inDtMax, inAFloor};

	oa::BufferAccess access[] = {
		oa::BufferAccess::Read,   // dtRaw
		oa::BufferAccess::Read,   // ddA
		oa::BufferAccess::Write,  // DT
		oa::BufferAccess::Write   // ADT
	};
	ctx.add( "EmpyrealmDtAdt", {&inDtRaw, &inDdA, &dt, &adt},
		access, &push, sizeof(push), oa::divCeil(n, 256));
	const auto semantic = lowering.commitWithId(
		oa::detail::opRegistry::FnMatrix::empyrealmDtAdt,
		{&inDtRaw, &inDdA},
		{&dt, &adt},
		{oa::OpAttribute::fromFloat("dtMin", inDtMin),
			oa::OpAttribute::fromFloat("dtMax", inDtMax),
			oa::OpAttribute::fromFloat("aFloor", inAFloor)});
	if (not semantic.isOk()) return {};

	if (oa::FnAutograd::isEnabled() and (inDtRaw.requiresGrad() or inDdA.requiresGrad())) {
		auto dtGrad = oa::makeShared<oa::GradEmpyrealmDt>(inDtMin, inDtMax);
		dtGrad->saveForBackward(inDtRaw);
		dtGrad->setGraphInputs(oa::Vector<oa::Matrix>{inDtRaw});
		dtGrad->sequenceNr_  = oa::FnAutograd::nextSeq();
		dtGrad->outputShape_ = dt.getShape();
		if (not oa::FnAutograd::attachSemantic(
			dtGrad, semantic.getValue(), 0U).isOk())
		{
			return {};
		}
		dt.mutAutograd().gradFn = dtGrad;

		auto adtGrad = oa::makeShared<oa::GradEmpyrealmAdt>(inAFloor);
		adtGrad->saveForBackward(inDdA, dt);
		adtGrad->setGraphInputs(oa::Vector<oa::Matrix>{inDdA, dt});
		adtGrad->sequenceNr_  = oa::FnAutograd::nextSeq();
		adtGrad->outputShape_ = adt.getShape();
		if (not oa::FnAutograd::attachSemantic(
			adtGrad, semantic.getValue(), 1U).isOk())
		{
			return {};
		}
		adt.mutAutograd().gradFn = adtGrad;
	}

	return {.dt = dt, .adt = adt};
}

// ============================================================================
// EmpyrealmPreprocess — fused in_proj split + RMSNorm + dt + A·dt (forward)
//
// Renamed copy of Mamba3Preprocess; dispatches Ssm/empyrealm/EmpyrealmPreprocess.
// ============================================================================

oa::Mamba3PreprocessResult oa::FnMatrix::empyrealmPreprocess(
	const oa::Matrix& inProjected, const oa::Matrix& inDtBias,
	const oa::Mamba3PreprocessConfig& inConfig)
{
	auto& ctx = oa::ExecutionSession::getActive();
	oa::OpLoweringScope lowering(ctx);
	const oa::U64 configIdentity =
		oa::FnMatrixPrivate::mamba3PreprocessConfigIdentity(inConfig);

	oa::I32 batchTimesSeq = static_cast<oa::I32>(inProjected.size(0));
	oa::I32 dInProj = static_cast<oa::I32>(inProjected.size(1));
	oa::I32 dInner = inConfig.dInner;
	oa::I32 dState = inConfig.dState;
	oa::I32 nHeads = inConfig.nHeads;
	oa::I32 numRopeAngles = inConfig.numRopeAngles;
	oa::I32 nGroups = inConfig.nGroups;
	oa::I32 mimoRank = inConfig.mimoRank;

	oa::I32 xOffset = dInner;
	oa::I32 bOffset = 2 * dInner;
	oa::I32 cOffset = bOffset + dState * nGroups * mimoRank;
	oa::I32 ddDtOffset = cOffset + dState * nGroups * mimoRank;
	oa::I32 ddAOffset = ddDtOffset + nHeads;
	oa::I32 trapOffset = ddAOffset + nHeads;
	oa::I32 angleOffset = trapOffset + nHeads;

	oa::I32 bcWidth = dState * nGroups * mimoRank;

	oa::Matrix xOut = oa::FnMatrix::empty(oa::MatrixShape{batchTimesSeq, dInner}, inProjected.getDtype());
	oa::Matrix zOut = oa::FnMatrix::empty(oa::MatrixShape{batchTimesSeq, dInner}, inProjected.getDtype());
	oa::Matrix bhOut = oa::FnMatrix::empty(oa::MatrixShape{batchTimesSeq, bcWidth}, inProjected.getDtype());
	oa::Matrix chOut = oa::FnMatrix::empty(oa::MatrixShape{batchTimesSeq, bcWidth}, inProjected.getDtype());
	oa::Matrix dtOut = oa::FnMatrix::empty(oa::MatrixShape{batchTimesSeq, nHeads}, inProjected.getDtype());
	oa::Matrix adtOut = oa::FnMatrix::empty(oa::MatrixShape{batchTimesSeq, nHeads}, inProjected.getDtype());
	oa::Matrix trapOut = oa::FnMatrix::empty(oa::MatrixShape{batchTimesSeq, nHeads}, inProjected.getDtype());
	oa::Matrix angleOut = oa::FnMatrix::empty(oa::MatrixShape{batchTimesSeq, numRopeAngles}, inProjected.getDtype());

	struct Push {
		oa::U32 rows, d_inner, d_state, n_heads, n_rope_angles;
		oa::U32 n_bc_rows, bc_width;
		oa::U32 z_offset, x_offset, b_offset, c_offset, dd_dt_offset, dd_a_offset, trap_offset, angle_offset;
		oa::U32 d_in_proj;
		oa::F32 eps, dt_min, dt_max, afloor;
	} push{
		static_cast<oa::U32>(batchTimesSeq),
		static_cast<oa::U32>(dInner),
		static_cast<oa::U32>(dState),
		static_cast<oa::U32>(nHeads),
		static_cast<oa::U32>(numRopeAngles),
		static_cast<oa::U32>(nGroups * mimoRank),
		static_cast<oa::U32>(bcWidth),
		0,
		static_cast<oa::U32>(xOffset),
		static_cast<oa::U32>(bOffset),
		static_cast<oa::U32>(cOffset),
		static_cast<oa::U32>(ddDtOffset),
		static_cast<oa::U32>(ddAOffset),
		static_cast<oa::U32>(trapOffset),
		static_cast<oa::U32>(angleOffset),
		static_cast<oa::U32>(dInProj),
		inConfig.eps, inConfig.dtMin, inConfig.dtMax, inConfig.aFloor
	};

	oa::BufferAccess access[] = {
		oa::BufferAccess::Read,   // projected
		oa::BufferAccess::Read,   // dt_bias
		oa::BufferAccess::Write,  // z
		oa::BufferAccess::Write,  // x
		oa::BufferAccess::Write,  // bh
		oa::BufferAccess::Write,  // ch
		oa::BufferAccess::Write,  // dt
		oa::BufferAccess::Write,  // adt
		oa::BufferAccess::Write,  // trap
		oa::BufferAccess::Write   // angle
	};
	ctx.add( "EmpyrealmPreprocess",
		{&inProjected, &inDtBias, &zOut, &xOut, &bhOut, &chOut, &dtOut, &adtOut, &trapOut, &angleOut},
		access, &push, sizeof(push), static_cast<oa::U32>(batchTimesSeq));
	const auto semantic = lowering.commitWithId(
		oa::detail::opRegistry::FnMatrix::empyrealmPreprocess,
		{&inProjected, &inDtBias},
		{&xOut, &zOut, &bhOut, &chOut, &dtOut, &adtOut, &trapOut, &angleOut},
		{oa::OpAttribute::fromUnsignedInteger(
			"configIdentity", configIdentity)});
	if (not semantic.isOk()) return {};

	bool needGrad = oa::FnAutograd::isEnabled() and
		(inProjected.requiresGrad() or inDtBias.requiresGrad());
	if (needGrad) {
		auto state = oa::makeShared<oa::GradMamba3Preprocess::SharedState>();
		state->projected = inProjected;
		state->dtBias = inDtBias;
		state->config = inConfig;

		auto attachGrad = [&](oa::Matrix& out, oa::I32 gradIndex, oa::U32 semanticIndex) {
			auto gradFn = oa::makeShared<oa::GradEmpyrealmPreprocess>();
			gradFn->state_ = state;
			gradFn->outputIndex_ = gradIndex;
			gradFn->setGraphInputs(oa::Vector<oa::Matrix>{inProjected, inDtBias});
			gradFn->sequenceNr_ = oa::FnAutograd::nextSeq();
			gradFn->outputShape_ = out.getShape();
			if (not oa::FnAutograd::attachSemantic(
				gradFn, semantic.getValue(), semanticIndex).isOk())
			{
				return false;
			}
			out.mutAutograd().gradFn = gradFn;
			out.mutAutograd().requiresGrad_ = true;
			return true;
		};
		if (not attachGrad(zOut, 0, 1) or
			not attachGrad(xOut, 1, 0) or
			not attachGrad(bhOut, 2, 2) or
			not attachGrad(chOut, 3, 3) or
			not attachGrad(dtOut, 4, 4) or
			not attachGrad(adtOut, 5, 5) or
			not attachGrad(trapOut, 6, 6) or
			not attachGrad(angleOut, 7, 7))
		{
			return {};
		}
	}

	return {
		.x = xOut, .z = zOut, .bh = bhOut, .ch = chOut,
		.dt = dtOut, .adt = adtOut, .trap = trapOut, .angle = angleOut
	};
}

// ============================================================================
// EmpyrealmPreprocessBwd — fused backward (1 dispatch instead of 11+)
// ============================================================================

oa::Mamba3PreprocessBwdResult oa::FnMatrix::empyrealmPreprocessBwd(
	const oa::Matrix& inProjected, const oa::Matrix& inDtBias,
	const oa::Matrix& inDZ, const oa::Matrix& inDX,
	const oa::Matrix& inDBh, const oa::Matrix& inDCh,
	const oa::Matrix& inDDT, const oa::Matrix& inDADT,
	const oa::Matrix& inDTrap, const oa::Matrix& inDAngle,
	const oa::Mamba3PreprocessConfig& inConfig)
{
	auto& ctx = oa::ExecutionSession::getActive();
	oa::OpLoweringScope lowering(ctx);
	const oa::U64 configIdentity =
		oa::FnMatrixPrivate::mamba3PreprocessConfigIdentity(inConfig);

	oa::I32 batchTimesSeq = static_cast<oa::I32>(inProjected.size(0));
	oa::I32 dInProj = static_cast<oa::I32>(inProjected.size(1));
	oa::I32 dInner = inConfig.dInner;
	oa::I32 dState = inConfig.dState;
	oa::I32 nHeads = inConfig.nHeads;
	oa::I32 numRopeAngles = inConfig.numRopeAngles;
	oa::I32 nGroups = inConfig.nGroups;
	oa::I32 mimoRank = inConfig.mimoRank;

	oa::I32 xOffset = dInner;
	oa::I32 bOffset = 2 * dInner;
	oa::I32 cOffset = bOffset + dState * nGroups * mimoRank;
	oa::I32 ddDtOffset = cOffset + dState * nGroups * mimoRank;
	oa::I32 ddAOffset = ddDtOffset + nHeads;
	oa::I32 trapOffset = ddAOffset + nHeads;
	oa::I32 angleOffset = trapOffset + nHeads;
	oa::I32 bcWidth = dState * nGroups * mimoRank;

	oa::Matrix dProj = oa::FnMatrix::zeros(inProjected.getShape(), inProjected.getDtype());
	oa::Matrix dDtBias = oa::FnMatrix::zeros(oa::MatrixShape{nHeads}, inProjected.getDtype());

	struct Push {
		oa::U32 rows, d_inner, d_state, n_heads, n_rope_angles;
		oa::U32 n_bc_rows, bc_width;
		oa::U32 z_offset, x_offset, b_offset, c_offset, dd_dt_offset, dd_a_offset, trap_offset, angle_offset;
		oa::U32 d_in_proj;
		oa::F32 eps, dt_min, dt_max, afloor;
	} push{
		static_cast<oa::U32>(batchTimesSeq),
		static_cast<oa::U32>(dInner),
		static_cast<oa::U32>(dState),
		static_cast<oa::U32>(nHeads),
		static_cast<oa::U32>(numRopeAngles),
		static_cast<oa::U32>(nGroups * mimoRank),
		static_cast<oa::U32>(bcWidth),
		0,
		static_cast<oa::U32>(xOffset),
		static_cast<oa::U32>(bOffset),
		static_cast<oa::U32>(cOffset),
		static_cast<oa::U32>(ddDtOffset),
		static_cast<oa::U32>(ddAOffset),
		static_cast<oa::U32>(trapOffset),
		static_cast<oa::U32>(angleOffset),
		static_cast<oa::U32>(dInProj),
		inConfig.eps, inConfig.dtMin, inConfig.dtMax, inConfig.aFloor
	};

	oa::BufferAccess access[] = {
		oa::BufferAccess::Read,   // projected
		oa::BufferAccess::Read,   // dt_bias
		oa::BufferAccess::Read,   // dz
		oa::BufferAccess::Read,   // dx
		oa::BufferAccess::Read,   // dbh
		oa::BufferAccess::Read,   // dch
		oa::BufferAccess::Read,   // ddt
		oa::BufferAccess::Read,   // dadt
		oa::BufferAccess::Read,   // dtrap
		oa::BufferAccess::Read,   // dangle
		oa::BufferAccess::Write,  // dproj
		oa::BufferAccess::Write   // ddt_bias (atomic add)
	};
	ctx.add( "EmpyrealmPreprocessBwd",
		{&inProjected, &inDtBias, &inDZ, &inDX, &inDBh, &inDCh,
		 &inDDT, &inDADT, &inDTrap, &inDAngle, &dProj, &dDtBias},
		access, &push, sizeof(push), static_cast<oa::U32>(batchTimesSeq));

	if (not lowering.commit(
		oa::detail::opRegistry::FnMatrix::empyrealmPreprocessBwd,
		{&inProjected, &inDtBias, &inDZ, &inDX, &inDBh, &inDCh,
			&inDDT, &inDADT, &inDTrap, &inDAngle},
		{&dProj, &dDtBias},
		{oa::OpAttribute::fromUnsignedInteger(
			"configIdentity", configIdentity)}).isOk())
	{
		return {};
	}
	return {.dProjected = dProj, .dDtBias = dDtBias};
}
