// oa::FnMatrix — Mamba-3 SSM operations.
//
// Selective state space model dispatch: verified full-sequence recurrence plus
// fused per-token dt and A·dt terms.

#include <oa/ml/fnMatrix.h>
#include <oa/core/autograd/matrix/autogradDtype.h>
#include <oa/core/envFlag.h>
#include <oa/ml/autograd/matrix/autogradSsm.h>
#include <oa/core/matrix.h>
#include <oa/core/fnMatrix.h>
#include <oa/core/log.h>
#include <oa/core/op.h>
#include <oa/core/status.h>
#include <oa/core/std/limits.h>
#include <oa/core/std/utility.h>
#include <oa/core/types.h>
#include <oa/runtime/executionSession.h>

#include "fnMatrixSsmInternal.h"

#include <assert.h>
namespace {

constexpr oa::U32 kMamba3Chunk = 16u;
constexpr oa::U64 kMamba3ChunkHistoryLimit = 256ull * 1024ull * 1024ull;
constexpr oa::U64 kMamba3MaxF32Elements = 1ull << 30u;

template<typename... Dimensions>
bool isMamba3ProductAddressable(Dimensions... inDimensions) {
	oa::U64 elements = 1u;
	const oa::U32 dimensions[] = {static_cast<oa::U32>(inDimensions)...};
	for (const oa::U32 dimension : dimensions) {
		if (dimension != 0u && elements > kMamba3MaxF32Elements / dimension)
			return false;
		elements *= dimension;
	}
	return true;
}

bool areMamba3MimoInputsFloat32(
	const oa::Matrix* const* inInputs,
	oa::Usize inCount)
{
	for (oa::Usize inputIndex = 0; inputIndex < inCount; ++inputIndex) {
		if (inInputs[inputIndex] == nullptr
			or inInputs[inputIndex]->getDtype() != oa::ScalarType::Float32)
		{
			return false;
		}
	}
	return true;
}

bool isMamba3MimoConfigValid(
	const oa::SsmConfig& inConfig,
	const oa::Matrix& inNormWeight)
{
	const oa::U32 groups = inConfig.nGroups == 0u
		? inConfig.nHeads : inConfig.nGroups;
	const bool basicConfigValid = inConfig.batch > 0u && inConfig.seqLen > 0u
		&& inConfig.nHeads > 0u && groups > 0u
		&& inConfig.nHeads % groups == 0u
		&& inConfig.headDim > 0u && inConfig.headDim <= 128u
		&& inConfig.stateSize > 0u && inConfig.stateSize <= 128u
		&& static_cast<oa::U64>(inConfig.headDim) * inConfig.stateSize <= 4096u
		&& inConfig.numRopeAngles <= 64u
		&& 2u * inConfig.numRopeAngles <= inConfig.stateSize
		&& inConfig.mimoRank > 0u && inConfig.mimoRank <= 8u
		&& inConfig.hasZ <= 1u && inConfig.hasD <= 1u
		&& inConfig.hasOutNorm <= 1u
		&& inNormWeight.getShape()
			== oa::MatrixShape{inConfig.nHeads, inConfig.headDim};
	if (not basicConfigValid) return false;

	const oa::U32 B = inConfig.batch, L = inConfig.seqLen;
	const oa::U32 H = inConfig.nHeads, G = groups;
	const oa::U32 P = inConfig.headDim, N = inConfig.stateSize;
	const oa::U32 A = inConfig.numRopeAngles, R = inConfig.mimoRank;
	if (not isMamba3ProductAddressable(B, L, R, G, N)
		or not isMamba3ProductAddressable(B, L, H, R, N)
		or not isMamba3ProductAddressable(B, H, L, P, N)
		or not isMamba3ProductAddressable(B, H, L, A)
		or not isMamba3ProductAddressable(B, H, R, P)
		or not isMamba3ProductAddressable(H, R, N)
		or not isMamba3ProductAddressable(H, R, P)
		or not isMamba3ProductAddressable(H, P))
	{
		return false;
	}

	const oa::U64 groupCount = static_cast<oa::U64>(B) * L * R * G * N;
	const oa::U64 angleCount = static_cast<oa::U64>(B) * L * A;
	const oa::U64 biasCount = static_cast<oa::U64>(H) * R * N;
	const oa::U64 mimoCount = static_cast<oa::U64>(H) * R * P;
	const oa::U64 normCount = static_cast<oa::U64>(H) * P;
	const oa::U64 reduceCount = groupCount + angleCount + biasCount + H
		+ mimoCount + normCount;
	return reduceCount
		<= static_cast<oa::U64>(oa::Limits<oa::U32>::max() - 255u);
}

void logInvalidMamba3Mimo(const char* inOperation) {
	OaLogError(oa::LogComponent::Ml,
		"%s requires exact MIMO tensor shapes, B/L/H/G/P/N > 0, H%%G=0, "
		"P/N<=128, P*N<=4096, A<=64 and 2*A<=N, R in [1,8], "
		"boolean flags, addressable tensor sizes, and FP32 operands",
		inOperation);
}

struct Mamba3ChunkPlan {
	oa::Matrix cRot;
	oa::Matrix bRot;
	oa::Matrix theta;
	oa::Matrix qk;
	oa::Matrix decay;
	oa::Matrix gamma;
	oa::Matrix scale;
	oa::Matrix summary;
	oa::Matrix product;
	oa::Matrix entry;
	oa::U32 chunks = 0u;
};

Mamba3ChunkPlan prepareMamba3Chunks(
	oa::ExecutionSession& inContext,
	const oa::Matrix& inC,
	const oa::Matrix& inB,
	const oa::Matrix& inX,
	const oa::Matrix& inAdt,
	const oa::Matrix& inDt,
	const oa::Matrix& inTrap,
	const oa::Matrix& inAngle,
	const oa::Matrix& inCBias,
	const oa::Matrix& inBBias,
	oa::U32 inBatch,
	oa::U32 inLength,
	oa::U32 inHeads,
	oa::U32 inGroups,
	oa::U32 inHeadDim,
	oa::U32 inState,
	oa::U32 inAngles)
{
	Mamba3ChunkPlan plan;
	plan.chunks = (inLength + kMamba3Chunk - 1u) / kMamba3Chunk;
	const oa::U32 bh = inBatch * inHeads;
	plan.cRot = oa::FnMatrix::empty(
		oa::MatrixShape{inBatch, inLength, inHeads, inState}, inC.getDtype());
	plan.bRot = oa::FnMatrix::empty(plan.cRot.getShape(), inB.getDtype());
	plan.theta = oa::FnMatrix::empty(
		oa::MatrixShape{bh, inLength, inAngles}, inAngle.getDtype());
	plan.qk = oa::FnMatrix::empty(inDt.getShape(), inDt.getDtype());
	plan.decay = oa::FnMatrix::empty(inDt.getShape(), inDt.getDtype());
	plan.gamma = oa::FnMatrix::empty(inDt.getShape(), inDt.getDtype());
	plan.scale = oa::FnMatrix::empty(inDt.getShape(), inDt.getDtype());
	plan.summary = oa::FnMatrix::empty(
		oa::MatrixShape{bh, plan.chunks, inHeadDim, inState}, inX.getDtype());
	plan.product = oa::FnMatrix::empty(
		oa::MatrixShape{bh, plan.chunks}, inAdt.getDtype());
	plan.entry = oa::FnMatrix::empty(plan.summary.getShape(), inX.getDtype());

	struct PreparePush { oa::U32 B, L, H, G, N, A; }
		preparePush{inBatch, inLength, inHeads, inGroups, inState, inAngles};
	oa::BufferAccess prepareAccess[] = {
		oa::BufferAccess::Read, oa::BufferAccess::Read, oa::BufferAccess::Read,
		oa::BufferAccess::Read, oa::BufferAccess::Read, oa::BufferAccess::Read,
		oa::BufferAccess::Read, oa::BufferAccess::Read,
		oa::BufferAccess::Write, oa::BufferAccess::Write, oa::BufferAccess::Write,
		oa::BufferAccess::Write, oa::BufferAccess::Write, oa::BufferAccess::Write,
		oa::BufferAccess::Write};
	inContext.add( "Mamba3SisoBwdShortPrepare",
		{&inC, &inB, &inDt, &inTrap, &inAdt, &inAngle, &inCBias, &inBBias,
		 &plan.cRot, &plan.bRot, &plan.theta, &plan.qk, &plan.decay,
		 &plan.gamma, &plan.scale},
		prepareAccess, &preparePush, sizeof(preparePush), (bh + 63u) / 64u, 1, 1);

	struct SummaryPush { oa::U32 B, L, H, P, N, chunks; }
		summaryPush{inBatch, inLength, inHeads, inHeadDim, inState, plan.chunks};
	oa::BufferAccess summaryAccess[] = {
		oa::BufferAccess::Read, oa::BufferAccess::Read, oa::BufferAccess::Read,
		oa::BufferAccess::Read, oa::BufferAccess::Write, oa::BufferAccess::Write};
	inContext.add( "Mamba3SisoChunkSummary",
		{&plan.bRot, &inX, &plan.decay, &plan.scale, &plan.summary, &plan.product},
		summaryAccess, &summaryPush, sizeof(summaryPush), bh * plan.chunks, 1, 1);

	struct PrefixPush { oa::U32 BH, P, N, chunks; }
		prefixPush{bh, inHeadDim, inState, plan.chunks};
	oa::BufferAccess prefixAccess[] = {
		oa::BufferAccess::Read, oa::BufferAccess::Read, oa::BufferAccess::Write};
	inContext.add( "Mamba3SisoChunkPrefix",
		{&plan.summary, &plan.product, &plan.entry}, prefixAccess,
		&prefixPush, sizeof(prefixPush), bh, 1, 1);
	return plan;
}

} // namespace

// ============================================================================
// Mamba3Siso — selective state space scan (Ssm/Mamba3/Mamba3SisoFwd.slang)
// ============================================================================

oa::Matrix oa::FnMatrix::mamba3Siso(
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
	assert(inConfig.headDim <= 128 && "Mamba3Siso: headdim (P) must be <= 128");
	assert(inConfig.stateSize <= 128 && "Mamba3Siso: d_state (N) must be <= 128");
	assert(inConfig.numRopeAngles <= 64 && "Mamba3Siso: num_rope_angles (A) must be <= 64");
	const oa::U32 groups = inConfig.nGroups == 0u
		? inConfig.nHeads : inConfig.nGroups;
	assert(groups > 0u && inConfig.nHeads % groups == 0u
		&& "Mamba3Siso: nHeads must be divisible by nGroups");
	auto& ctx = oa::ExecutionSession::getActive();
	oa::OpLoweringScope lowering(ctx);
	const oa::U64 configIdentity =
		oa::FnMatrixPrivate::ssmConfigIdentity(inConfig);

	// Mixed precision: the selective-scan kernels compute in FP32 through the
	// always-fp32 storage helpers, and the cross-token recurrent state is far too
	// precision-sensitive for bf16 round-trips. in bf16 mode run the scan as an
	// FP32 island — cast the operands up, scan, cast the result back down.
	// oa::GradCast threads gradients across both boundaries; Cast is a no-op on the
	// recursive fp32 call, so this reduces to the plain body below.
	if (inX.getDtype() == oa::ScalarType::BFloat16) {
		const auto up = [](const oa::Matrix& m) { return oa::FnMatrix::cast(m, oa::ScalarType::Float32); };
		auto output = oa::FnMatrix::cast(
			mamba3Siso(up(inC), up(inB), up(inX), up(inZ), up(inAdt), up(inDt),
				up(inTrap), up(inAngle), up(inCBias), up(inBBias), up(inD), inConfig),
			inX.getDtype());
		const auto semantic = lowering.commitWithId(
			oa::detail::opRegistry::FnMatrix::mamba3Siso,
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

	// Every active output lane is written exactly once by Mamba3SisoFwd.
	oa::Matrix output = oa::FnMatrix::empty(
		oa::MatrixShape{inConfig.batch, inConfig.seqLen, inConfig.nHeads, inConfig.headDim},
		inX.getDtype());

	struct {
		oa::U32 B, L, H, G, P, N, A, has_z, has_d;
	} push{
		inConfig.batch, inConfig.seqLen, inConfig.nHeads, groups, inConfig.headDim,
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

	ctx.add( "Mamba3SisoFwd",
		{&inC, &inB, &inX, &inZ, &inAdt, &inDt, &inTrap, &inAngle,
		 &inCBias, &inBBias, &inD, &output},
		access, &push, sizeof(push), inConfig.batch * inConfig.nHeads, 1, 1);
	const auto semantic = lowering.commitWithId(
		oa::detail::opRegistry::FnMatrix::mamba3Siso,
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
		auto gradFn = oa::makeShared<oa::GradMamba3Siso>();
		gradFn->saveForBackward(inC, inB, inX, inZ, inAdt, inDt, inTrap,
			inAngle, inCBias, inBBias, inD);
		gradFn->setGraphInputs(oa::Vec<oa::Matrix>{inC, inB, inX, inZ, inAdt, inDt, inTrap,
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
// Mamba3SisoStep — single-token step for autoregressive generation
// ============================================================================

oa::Matrix oa::FnMatrix::mamba3SisoStep(
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
	assert(inConfig.seqLen == 1 && "Mamba3SisoStep: seqLen must be 1");
	const oa::U32 groups = inConfig.nGroups == 0u
		? inConfig.nHeads : inConfig.nGroups;
	assert(groups > 0u && inConfig.nHeads % groups == 0u
		&& "Mamba3SisoStep: nHeads must be divisible by nGroups");
	auto& ctx = oa::ExecutionSession::getActive();
	oa::OpLoweringScope lowering(ctx);
	const oa::U64 configIdentity =
		oa::FnMatrixPrivate::ssmConfigIdentity(inConfig);

	// Every active output lane is written exactly once by Mamba3SisoStep.
	oa::Matrix output = oa::FnMatrix::empty(
		oa::MatrixShape{inConfig.batch, 1, inConfig.nHeads, inConfig.headDim}, inX.getDtype());

	struct {
		oa::U32 B, H, G, P, N, A, has_z, has_d;
	} push{ inConfig.batch, inConfig.nHeads, groups, inConfig.headDim, inConfig.stateSize,
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

	ctx.add( "Mamba3SisoStep",
		{&inC, &inB, &inX, &inZ, &inAdt, &inDt, &inTrap, &inAngle, &inCBias, &inBBias, &inD,
		 &output, &inSsmState, &inAngleState, &inKState, &inVState},
		access, &push, sizeof(push), inConfig.batch * inConfig.nHeads, 1, 1);

	if (not lowering.commit(
		oa::detail::opRegistry::FnMatrix::mamba3SisoStep,
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
// Mamba3SisoBwd — backward pass for the selective scan
// ============================================================================

oa::SsmBwdResult oa::FnMatrix::mamba3SisoBwd(
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
	const oa::U32 G = inConfig.nGroups == 0u ? H : inConfig.nGroups;
	const oa::U32 P = inConfig.headDim;
	const oa::U32 N = inConfig.stateSize;
	const oa::U32 A = inConfig.numRopeAngles;
	assert(G > 0u && H % G == 0u
		&& "Mamba3SisoBwd: nHeads must be divisible by nGroups");

	// The android NLP shape needs a bounded global-scratch specialization:
	// Turnip rejects both the 32 KiB shared-memory short kernel and the generic
	// MAXN=128 kernel. Desktop keeps its existing, faster shared-memory route.
#if defined(OA_PLATFORM_ANDROID)
	const bool mobileKernel = L <= 16u && P <= 16u && N <= 32u && A <= 8u;
#else
	const bool mobileKernel = false;
#endif
	// The P16 specialization performs one 16-lane reduction with WaveActiveSum.
	// A smaller hardware subgroup would produce independent partial sums, so it
	// must use the general multi-subgroup implementation instead. unknown (zero)
	// capabilities are conservative as well.
	const bool shortKernel = !mobileKernel
		&& ctx.subgroupSize() >= 16u
		&& L <= 16u && P <= 16u && N <= 32u && A <= 8u;
#if defined(OA_PLATFORM_ANDROID)
	const bool chunkedKernel = false;
#else
	const oa::U64 historyBytes = static_cast<oa::U64>(B) * H * L * P * N
		* sizeof(oa::F32);
	const bool chunkedKernel = !shortKernel
		&& L >= 64u && P <= 32u && N <= 32u && A <= 8u
		&& historyBytes <= kMamba3ChunkHistoryLimit;
#endif
	oa::Matrix dC    = oa::FnMatrix::empty(oa::MatrixShape{B, L, H, N}, inC.getDtype());
	oa::Matrix dB    = oa::FnMatrix::empty(dC.getShape(), inB.getDtype());
	oa::Matrix dX    = oa::FnMatrix::empty(inX.getShape(),   inX.getDtype());
	oa::Matrix dZ    = oa::FnMatrix::empty(inZ.getShape(),   inZ.getDtype());
	oa::Matrix dAdt  = oa::FnMatrix::empty(inAdt.getShape(), inAdt.getDtype());
	oa::SsmBwdResult result;
	if (shortKernel) {
		// The short training corner is decomposed into independent physical stages.
		// This deliberately trades bounded transient storage for occupancy: no stage
		// reserves the old 32 KiB workgroup history or serializes value rows behind
		// full-workgroup barriers.
		oa::Matrix cRot = oa::FnMatrix::empty(oa::MatrixShape{B, L, H, N}, inC.getDtype());
		oa::Matrix bRot = oa::FnMatrix::empty(cRot.getShape(), inB.getDtype());
		oa::Matrix theta = oa::FnMatrix::empty(oa::MatrixShape{B * H, L, A}, inAngle.getDtype());
		oa::Matrix qk = oa::FnMatrix::empty(inDt.getShape(), inDt.getDtype());
		oa::Matrix decay = oa::FnMatrix::empty(inDt.getShape(), inDt.getDtype());
		oa::Matrix gamma = oa::FnMatrix::empty(inDt.getShape(), inDt.getDtype());
		oa::Matrix scale = oa::FnMatrix::empty(inDt.getShape(), inDt.getDtype());
		oa::Matrix hPrev = oa::FnMatrix::empty(oa::MatrixShape{B * H, L, P, N}, inX.getDtype());
		oa::Matrix dPre = oa::FnMatrix::empty(inX.getShape(), inX.getDtype());
		oa::Matrix dGamma = oa::FnMatrix::empty(inDt.getShape(), inDt.getDtype());
		oa::Matrix dScale = oa::FnMatrix::empty(inDt.getShape(), inDt.getDtype());
		oa::Matrix dTheta = oa::FnMatrix::empty(theta.getShape(), inAngle.getDtype());
		oa::Matrix dDt = oa::FnMatrix::empty(inDt.getShape(), inDt.getDtype());
		oa::Matrix dTrap = oa::FnMatrix::empty(inTrap.getShape(), inTrap.getDtype());
		oa::Matrix dAngle = oa::FnMatrix::empty(inAngle.getShape(), inAngle.getDtype());
		oa::Matrix dCBias = oa::FnMatrix::empty(inCBias.getShape(), inCBias.getDtype());
		oa::Matrix dBBias = oa::FnMatrix::empty(inBBias.getShape(), inBBias.getDtype());
		oa::Matrix dD = oa::FnMatrix::empty(inD.getShape(), inD.getDtype());
		oa::Matrix dCBiasBatch = oa::FnMatrix::empty(
			oa::MatrixShape{B, H, N}, inCBias.getDtype());
		oa::Matrix dBBiasBatch = oa::FnMatrix::empty(
			oa::MatrixShape{B, H, N}, inBBias.getDtype());
		oa::Matrix dDBatch = oa::FnMatrix::empty(
			oa::MatrixShape{B, H}, inD.getDtype());
		oa::Matrix dAngleHead = oa::FnMatrix::empty(
			oa::MatrixShape{B, H, L, A}, inAngle.getDtype());

		struct PreparePush { oa::U32 B, L, H, G, N, A; } preparePush{B, L, H, G, N, A};
		oa::BufferAccess prepareAccess[] = {
			oa::BufferAccess::Read, oa::BufferAccess::Read, oa::BufferAccess::Read,
			oa::BufferAccess::Read, oa::BufferAccess::Read, oa::BufferAccess::Read,
			oa::BufferAccess::Read, oa::BufferAccess::Read,
			oa::BufferAccess::Write, oa::BufferAccess::Write, oa::BufferAccess::Write,
			oa::BufferAccess::Write, oa::BufferAccess::Write, oa::BufferAccess::Write,
			oa::BufferAccess::Write};
		ctx.add( "Mamba3SisoBwdShortPrepare",
			{&inC, &inB, &inDt, &inTrap, &inAdt, &inAngle, &inCBias, &inBBias,
			 &cRot, &bRot, &theta, &qk, &decay, &gamma, &scale},
			prepareAccess, &preparePush, sizeof(preparePush), (B * H + 63u) / 64u, 1, 1);

		struct StatePush { oa::U32 B, L, H, P, N, has_z, has_d; }
			statePush{B, L, H, P, N, inConfig.hasZ, inConfig.hasD};
		oa::BufferAccess stateAccess[] = {
			oa::BufferAccess::Read, oa::BufferAccess::Read, oa::BufferAccess::Read,
			oa::BufferAccess::Read, oa::BufferAccess::Read, oa::BufferAccess::Read,
			oa::BufferAccess::Read, oa::BufferAccess::Read, oa::BufferAccess::Read,
			oa::BufferAccess::Read, oa::BufferAccess::Write, oa::BufferAccess::Write,
			oa::BufferAccess::Write};
		ctx.add( "Mamba3SisoBwdShortState",
			{&bRot, &inX, &decay, &scale, &cRot, &inZ, &gamma, &inD, &qk,
			 &inDOut, &hPrev, &dPre, &dZ}, stateAccess,
			&statePush, sizeof(statePush), B * H, 1, 1);

		struct ReversePush { oa::U32 B, L, H, P, N, A, has_d; }
			reversePush{B, L, H, P, N, A, inConfig.hasD};
		oa::BufferAccess reverseAccess[] = {
			oa::BufferAccess::Read, oa::BufferAccess::Read, oa::BufferAccess::Read,
			oa::BufferAccess::Read, oa::BufferAccess::Read, oa::BufferAccess::Read,
			oa::BufferAccess::Read, oa::BufferAccess::Read, oa::BufferAccess::Read,
			oa::BufferAccess::Read, oa::BufferAccess::Read,
			oa::BufferAccess::Write, oa::BufferAccess::Write, oa::BufferAccess::Write,
			oa::BufferAccess::Write, oa::BufferAccess::Write, oa::BufferAccess::Write,
			oa::BufferAccess::Write, oa::BufferAccess::Write,
			oa::BufferAccess::Write, oa::BufferAccess::Write};
		ctx.add( "Mamba3SisoBwdShortReverseP16",
			{&dPre, &cRot, &bRot, &theta, &gamma, &scale, &decay, &hPrev,
			 &inX, &inD, &qk, &dC, &dB, &dX, &dAdt, &dGamma, &dScale,
			 &dTheta, &dCBiasBatch, &dBBiasBatch, &dDBatch},
			reverseAccess, &reversePush, sizeof(reversePush), B * H, 1, 1);

		struct FinalizePush { oa::U32 B, L, H, A; } finalizePush{B, L, H, A};
		oa::BufferAccess finalizeAccess[] = {
			oa::BufferAccess::Read, oa::BufferAccess::Read, oa::BufferAccess::Read,
			oa::BufferAccess::Read, oa::BufferAccess::Read, oa::BufferAccess::Read,
			oa::BufferAccess::Write, oa::BufferAccess::Write, oa::BufferAccess::Write};
		ctx.add( "Mamba3SisoBwdShortFinalize",
			{&inDt, &inTrap, &inAngle, &dGamma, &dScale, &dTheta, &dDt, &dTrap,
			 &dAngleHead}, finalizeAccess, &finalizePush, sizeof(finalizePush),
			(B * H + 63u) / 64u, 1, 1);

		const oa::U32 count = B * L * A + 2u * H * N + H;
		struct ReducePush { oa::U32 B, L, H, N, A, Count; }
			push{B, L, H, N, A, count};
		oa::BufferAccess reduceAccess[] = {
			oa::BufferAccess::Read, oa::BufferAccess::Read, oa::BufferAccess::Read,
			oa::BufferAccess::Read, oa::BufferAccess::Write, oa::BufferAccess::Write,
			oa::BufferAccess::Write, oa::BufferAccess::Write};
		ctx.add( "Mamba3SisoBwdShortReduce",
			{&dAngleHead, &dCBiasBatch, &dBBiasBatch, &dDBatch,
			 &dAngle, &dCBias, &dBBias, &dD},
			reduceAccess, &push, sizeof(push), oa::divCeil(count, 256u));

		result = {.dC = dC, .dB = dB, .dX = dX, .dZ = dZ, .dAdt = dAdt,
			.dDt = dDt, .dTrap = dTrap, .dAngle = dAngle, .dCBias = dCBias,
			.dBBias = dBBias, .dD = dD};
	} else if (chunkedKernel) {
		auto plan = prepareMamba3Chunks(
			ctx, inC, inB, inX, inAdt, inDt, inTrap, inAngle, inCBias, inBBias,
			B, L, H, G, P, N, A);
		oa::Matrix dPre = oa::FnMatrix::empty(inX.getShape(), inX.getDtype());
		oa::Matrix reverseSummary = oa::FnMatrix::empty(
			plan.summary.getShape(), inX.getDtype());
		oa::Matrix chunkExit = oa::FnMatrix::empty(
			plan.summary.getShape(), inX.getDtype());
		oa::Matrix hPrev = oa::FnMatrix::empty(
			oa::MatrixShape{B * H, L, P, N}, inX.getDtype());
		oa::Matrix dGamma = oa::FnMatrix::empty(inDt.getShape(), inDt.getDtype());
		oa::Matrix dScale = oa::FnMatrix::empty(inDt.getShape(), inDt.getDtype());
		oa::Matrix dTheta = oa::FnMatrix::empty(plan.theta.getShape(), inAngle.getDtype());
		oa::Matrix dDt = oa::FnMatrix::empty(inDt.getShape(), inDt.getDtype());
		oa::Matrix dTrap = oa::FnMatrix::empty(inTrap.getShape(), inTrap.getDtype());
		oa::Matrix dAngleHead = oa::FnMatrix::empty(
			oa::MatrixShape{B, H, L, A}, inAngle.getDtype());
		oa::Matrix dDToken = oa::FnMatrix::empty(inDt.getShape(), inD.getDtype());

		struct DprePush {
			oa::U32 B, L, H, P, N, chunks, hasZ, hasD;
		} dprePush{B, L, H, P, N, plan.chunks, inConfig.hasZ, inConfig.hasD};
		oa::BufferAccess dpreAccess[] = {
			oa::BufferAccess::Read, oa::BufferAccess::Read, oa::BufferAccess::Read,
			oa::BufferAccess::Read, oa::BufferAccess::Read, oa::BufferAccess::Read,
			oa::BufferAccess::Read, oa::BufferAccess::Read, oa::BufferAccess::Read,
			oa::BufferAccess::Read, oa::BufferAccess::Read, oa::BufferAccess::Write,
			oa::BufferAccess::Write, oa::BufferAccess::Write};
		ctx.add( "Mamba3SisoChunkDpre",
			{&plan.cRot, &plan.bRot, &inX, &inZ, &plan.decay, &plan.gamma,
			 &plan.scale, &plan.qk, &inD, &inDOut, &plan.entry, &hPrev, &dPre,
			 &dZ},
			dpreAccess, &dprePush, sizeof(dprePush), B * H * plan.chunks, 1, 1);

		struct SummaryPush { oa::U32 B, L, H, P, N, chunks; }
			summaryPush{B, L, H, P, N, plan.chunks};
		oa::BufferAccess summaryAccess[] = {
			oa::BufferAccess::Read, oa::BufferAccess::Read, oa::BufferAccess::Read,
			oa::BufferAccess::Write};
		ctx.add( "Mamba3SisoChunkBwdSummary",
			{&plan.cRot, &plan.decay, &dPre, &reverseSummary},
			summaryAccess, &summaryPush, sizeof(summaryPush),
			B * H * plan.chunks, 1, 1);

		struct PrefixPush { oa::U32 BH, P, N, chunks; }
			prefixPush{B * H, P, N, plan.chunks};
		oa::BufferAccess prefixAccess[] = {
			oa::BufferAccess::Read, oa::BufferAccess::Read, oa::BufferAccess::Write};
		ctx.add( "Mamba3SisoChunkBwdPrefix",
			{&reverseSummary, &plan.product, &chunkExit}, prefixAccess,
			&prefixPush, sizeof(prefixPush), B * H, 1, 1);

		struct ReversePush {
			oa::U32 B, L, H, P, N, A, chunks, hasD;
		} reversePush{B, L, H, P, N, A, plan.chunks, inConfig.hasD};
		oa::BufferAccess reverseAccess[] = {
			oa::BufferAccess::Read, oa::BufferAccess::Read, oa::BufferAccess::Read,
			oa::BufferAccess::Read, oa::BufferAccess::Read, oa::BufferAccess::Read,
			oa::BufferAccess::Read, oa::BufferAccess::Read, oa::BufferAccess::Read,
			oa::BufferAccess::Read, oa::BufferAccess::Read, oa::BufferAccess::Read,
			oa::BufferAccess::Write, oa::BufferAccess::Write, oa::BufferAccess::Write,
			oa::BufferAccess::Write, oa::BufferAccess::Write, oa::BufferAccess::Write,
			oa::BufferAccess::Write, oa::BufferAccess::Write};
		ctx.add( "Mamba3SisoChunkBwd",
			{&plan.cRot, &plan.bRot, &plan.theta, &inX, &plan.decay,
			 &plan.gamma, &plan.scale, &plan.qk, &inD, &dPre, &hPrev,
			 &chunkExit, &dC, &dB, &dX, &dAdt, &dGamma, &dScale, &dTheta,
			 &dDToken},
			reverseAccess, &reversePush, sizeof(reversePush),
			B * H * plan.chunks, 1, 1);

		struct FinalizePush { oa::U32 B, L, H, A; } finalizePush{B, L, H, A};
		oa::BufferAccess finalizeAccess[] = {
			oa::BufferAccess::Read, oa::BufferAccess::Read, oa::BufferAccess::Read,
			oa::BufferAccess::Read, oa::BufferAccess::Read, oa::BufferAccess::Read,
			oa::BufferAccess::Write, oa::BufferAccess::Write, oa::BufferAccess::Write};
		ctx.add( "Mamba3SisoChunkFinalize",
			{&inDt, &inTrap, &inAngle, &dGamma, &dScale, &dTheta, &dDt,
			 &dTrap, &dAngleHead},
			finalizeAccess, &finalizePush, sizeof(finalizePush),
			(B * H + 63u) / 64u, 1, 1);

		oa::Matrix dAngle = oa::FnMatrix::sum(dAngleHead, 1)
			.reshape(oa::MatrixShape{B, L, A});
		oa::Matrix dCBias = oa::FnMatrix::sum(
			dC.reshape(oa::MatrixShape{B * L, H * N}), 0)
			.reshape(oa::MatrixShape{H, N});
		oa::Matrix dBBias = oa::FnMatrix::sum(
			dB.reshape(oa::MatrixShape{B * L, H * N}), 0)
			.reshape(oa::MatrixShape{H, N});
		oa::Matrix dD = oa::FnMatrix::sum(
			dDToken.reshape(oa::MatrixShape{B * L, H}), 0)
			.reshape(oa::MatrixShape{H});
		result = {.dC = dC, .dB = dB, .dX = dX, .dZ = dZ, .dAdt = dAdt,
			.dDt = dDt, .dTrap = dTrap, .dAngle = dAngle, .dCBias = dCBias,
			.dBBias = dBBias, .dD = dD};
	} else {
		const oa::U32 chunk = mobileKernel ? 16u : 32u;
		const oa::U32 nchunks = (L + chunk - 1u) / chunk;
		const bool oneChunk = nchunks == 1u;
		oa::Matrix entry = oa::FnMatrix::empty(oneChunk
			? oa::MatrixShape{1}
			: oa::MatrixShape{B * H, nchunks, P, N}, inX.getDtype());
		oa::Matrix thetaEnt = oa::FnMatrix::empty(oneChunk
			? oa::MatrixShape{1}
			: oa::MatrixShape{B * H, nchunks, A}, inX.getDtype());
		oa::Matrix chunkBuf = oa::FnMatrix::empty(oa::MatrixShape{B * H, chunk, P, N}, inX.getDtype());
		oa::Matrix dDt = oa::FnMatrix::zeros(inDt.getShape(), inDt.getDtype());
		oa::Matrix dTrapP = oa::FnMatrix::zeros(inTrap.getShape(), inTrap.getDtype());
		oa::Matrix dAngH = oa::FnMatrix::empty(oa::MatrixShape{B, H, L, A}, inAngle.getDtype());
		oa::Matrix dDTok = oa::FnMatrix::empty(inDt.getShape(), inDt.getDtype());
		const oa::U32 flags = (inConfig.hasZ != 0u ? 1u : 0u)
			| (inConfig.hasD != 0u ? 2u : 0u);
		struct { oa::U32 B, L, H, G, P, N, A, Flags; }
			push{B, L, H, G, P, N, A, flags};
		oa::BufferAccess access[] = {
			oa::BufferAccess::Read, oa::BufferAccess::Read, oa::BufferAccess::Read,
			oa::BufferAccess::Read, oa::BufferAccess::Read, oa::BufferAccess::Read,
			oa::BufferAccess::Read, oa::BufferAccess::Read, oa::BufferAccess::Read,
			oa::BufferAccess::Read, oa::BufferAccess::Read, oa::BufferAccess::Read,
			oa::BufferAccess::ReadWrite, oa::BufferAccess::ReadWrite, oa::BufferAccess::ReadWrite,
			oa::BufferAccess::Write, oa::BufferAccess::Write, oa::BufferAccess::Write,
			oa::BufferAccess::Write, oa::BufferAccess::Write, oa::BufferAccess::ReadWrite,
			oa::BufferAccess::ReadWrite, oa::BufferAccess::Write, oa::BufferAccess::Write};
		ctx.add( mobileKernel ? "Mamba3SisoBwdMobile" : "Mamba3SisoBwd",
			{&inC, &inB, &inX, &inZ, &inAdt, &inDt, &inTrap, &inAngle, &inCBias, &inBBias,
			 &inD, &inDOut, &entry, &thetaEnt, &chunkBuf, &dC, &dB, &dX, &dZ, &dAdt,
			 &dDt, &dTrapP, &dAngH, &dDTok}, access, &push, sizeof(push), B * H, 1, 1);
		oa::Matrix trapS = oa::FnMatrix::sigmoid(inTrap);
		result = {
			.dC = dC, .dB = dB, .dX = dX, .dZ = dZ, .dAdt = dAdt, .dDt = dDt,
			.dTrap = oa::FnMatrix::sigmoidBwd(trapS, dTrapP),
			.dAngle = oa::FnMatrix::sum(dAngH, 1).reshape(oa::MatrixShape{B, L, A}),
			.dCBias = oa::FnMatrix::sum(dC.reshape(oa::MatrixShape{B * L, H * N}), 0)
				.reshape(oa::MatrixShape{H, N}),
			.dBBias = oa::FnMatrix::sum(dB.reshape(oa::MatrixShape{B * L, H * N}), 0)
				.reshape(oa::MatrixShape{H, N}),
			.dD = oa::FnMatrix::sum(dDTok.reshape(oa::MatrixShape{B * L, H}), 0)
				.reshape(oa::MatrixShape{H})};
	}

	if (G < H) {
		oa::Matrix dCGroup = oa::FnMatrix::empty(inC.getShape(), inC.getDtype());
		oa::Matrix dBGroup = oa::FnMatrix::empty(inB.getShape(), inB.getDtype());
		const oa::U32 count = B * L * G * N;
		struct GroupReducePush { oa::U32 B, L, H, G, N, Count; }
			push{B, L, H, G, N, count};
		oa::BufferAccess access[] = {
			oa::BufferAccess::Read, oa::BufferAccess::Read,
			oa::BufferAccess::Write, oa::BufferAccess::Write};
		ctx.add( "Mamba3SisoBwdGroupReduce",
			{&dC, &dB, &dCGroup, &dBGroup}, access,
			&push, sizeof(push), oa::divCeil(count, 256u));
		result.dC = dCGroup;
		result.dB = dBGroup;
	}

	if (not lowering.commit(
		oa::detail::opRegistry::FnMatrix::mamba3SisoBwd,
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
// Mamba3Mimo — shared-state multi-input/multi-output selective scan
// ============================================================================

oa::Matrix oa::FnMatrix::mamba3Mimo(
	const oa::Matrix& inC, const oa::Matrix& inB,
	const oa::Matrix& inX, const oa::Matrix& inZ,
	const oa::Matrix& inAdt, const oa::Matrix& inDt,
	const oa::Matrix& inTrap, const oa::Matrix& inAngle,
	const oa::Matrix& inCBias, const oa::Matrix& inBBias,
	const oa::Matrix& inD, const oa::Matrix& inMimoX,
	const oa::Matrix& inMimoZ, const oa::Matrix& inMimoO,
	const oa::Matrix& inNormWeight, const oa::SsmConfig& inConfig)
{
	const oa::U32 B = inConfig.batch, L = inConfig.seqLen;
	const oa::U32 H = inConfig.nHeads;
	const oa::U32 G = inConfig.nGroups == 0u ? H : inConfig.nGroups;
	const oa::U32 P = inConfig.headDim, N = inConfig.stateSize;
	const oa::U32 A = inConfig.numRopeAngles, R = inConfig.mimoRank;
	const oa::Matrix* inputs[] = {
		&inC, &inB, &inX, &inZ, &inAdt, &inDt, &inTrap, &inAngle,
		&inCBias, &inBBias, &inD, &inMimoX, &inMimoZ, &inMimoO,
		&inNormWeight};
	if (not isMamba3MimoConfigValid(inConfig, inNormWeight)
		or not areMamba3MimoInputsFloat32(inputs, oa::arraySize(inputs))
		or inC.getShape() != oa::MatrixShape{B, L, R * G, N}
		or inB.getShape() != oa::MatrixShape{B, L, R * G, N}
		or inX.getShape() != oa::MatrixShape{B, L, H, P}
		or inZ.getShape() != oa::MatrixShape{B, L, H, P}
		or inAdt.getShape() != oa::MatrixShape{B, L, H}
		or inDt.getShape() != oa::MatrixShape{B, L, H}
		or inTrap.getShape() != oa::MatrixShape{B, L, H}
		or inAngle.getShape() != oa::MatrixShape{B, L, A}
		or inCBias.getShape() != oa::MatrixShape{H, R, N}
		or inBBias.getShape() != oa::MatrixShape{H, R, N}
		or inD.getShape() != oa::MatrixShape{H}
		or inMimoX.getShape() != oa::MatrixShape{H, R, P}
		or inMimoZ.getShape() != oa::MatrixShape{H, R, P}
		or inMimoO.getShape() != oa::MatrixShape{H, R, P})
	{
		logInvalidMamba3Mimo("Mamba3Mimo");
		return {};
	}
	auto& ctx = oa::ExecutionSession::getActive();
	oa::OpLoweringScope lowering(ctx);
	const oa::U64 configIdentity = oa::FnMatrixPrivate::ssmConfigIdentity(inConfig);
	oa::Matrix output = oa::FnMatrix::empty(oa::MatrixShape{B, L, H, P}, inX.getDtype());
	struct Push {
		oa::U32 B, L, H, G, P, N, A, R, hasZ, hasD, HasNorm;
		oa::F32 eps;
	} push{B, L, H, G, P, N, A, R, inConfig.hasZ, inConfig.hasD,
		inConfig.hasOutNorm, 1e-5F};
	oa::BufferAccess access[] = {
		oa::BufferAccess::Read, oa::BufferAccess::Read, oa::BufferAccess::Read,
		oa::BufferAccess::Read, oa::BufferAccess::Read, oa::BufferAccess::Read,
		oa::BufferAccess::Read, oa::BufferAccess::Read, oa::BufferAccess::Read,
		oa::BufferAccess::Read, oa::BufferAccess::Read, oa::BufferAccess::Read,
		oa::BufferAccess::Read, oa::BufferAccess::Read, oa::BufferAccess::Read,
		oa::BufferAccess::Write};
	ctx.add( "Mamba3MimoFwd",
		{&inC, &inB, &inX, &inZ, &inAdt, &inDt, &inTrap, &inAngle,
		 &inCBias, &inBBias, &inD, &inMimoX, &inMimoZ, &inMimoO,
		 &inNormWeight, &output}, access, &push, sizeof(push), B * H, 1, 1);
	const auto semantic = lowering.commitWithId(
		oa::detail::opRegistry::FnMatrix::mamba3Mimo,
		{&inC, &inB, &inX, &inZ, &inAdt, &inDt, &inTrap, &inAngle,
			&inCBias, &inBBias, &inD, &inMimoX, &inMimoZ, &inMimoO,
			&inNormWeight}, {&output},
		{oa::OpAttribute::fromUnsignedInteger("configIdentity", configIdentity)});
	if (not semantic.isOk()) return {};
	if (oa::FnAutograd::isEnabled() and
		(inC.requiresGrad() or inB.requiresGrad() or inX.requiresGrad() or
		 inZ.requiresGrad() or inAdt.requiresGrad() or inDt.requiresGrad() or
		 inTrap.requiresGrad() or inAngle.requiresGrad() or inCBias.requiresGrad() or
		 inBBias.requiresGrad() or inD.requiresGrad() or inMimoX.requiresGrad() or
		 inMimoZ.requiresGrad() or inMimoO.requiresGrad() or
		 inNormWeight.requiresGrad()))
	{
		auto gradFn = oa::makeShared<oa::GradMamba3Mimo>();
		gradFn->saveForBackward(inC, inB, inX, inZ, inAdt, inDt,
			inTrap, inAngle, inCBias, inBBias, inD, inMimoX, inMimoZ, inMimoO,
			inNormWeight);
		gradFn->setGraphInputs(oa::Vec<oa::Matrix>{inC, inB, inX, inZ, inAdt,
			inDt, inTrap, inAngle, inCBias, inBBias, inD, inMimoX, inMimoZ,
			inMimoO, inNormWeight});
		gradFn->sequenceNr_ = oa::FnAutograd::nextSeq();
		gradFn->config_ = inConfig;
		gradFn->outputShape_ = output.getShape();
		if (not oa::FnAutograd::attachSemantic(gradFn, semantic.getValue()).isOk())
			return {};
		output.mutAutograd().gradFn = gradFn;
	}
	return output;
}

oa::Mamba3MimoBwdResult oa::FnMatrix::mamba3MimoBwd(
	const oa::Matrix& inDOut, const oa::Matrix& inC, const oa::Matrix& inB,
	const oa::Matrix& inX, const oa::Matrix& inZ,
	const oa::Matrix& inAdt, const oa::Matrix& inDt,
	const oa::Matrix& inTrap, const oa::Matrix& inAngle,
	const oa::Matrix& inCBias, const oa::Matrix& inBBias,
	const oa::Matrix& inD, const oa::Matrix& inMimoX,
	const oa::Matrix& inMimoZ, const oa::Matrix& inMimoO,
	const oa::Matrix& inNormWeight, const oa::SsmConfig& inConfig)
{
	const oa::U32 B = inConfig.batch, L = inConfig.seqLen, H = inConfig.nHeads;
	const oa::U32 G = inConfig.nGroups == 0u ? H : inConfig.nGroups;
	const oa::U32 P = inConfig.headDim, N = inConfig.stateSize;
	const oa::U32 A = inConfig.numRopeAngles, R = inConfig.mimoRank;
	const oa::Matrix* inputs[] = {
		&inDOut, &inC, &inB, &inX, &inZ, &inAdt, &inDt, &inTrap,
		&inAngle, &inCBias, &inBBias, &inD, &inMimoX, &inMimoZ,
		&inMimoO, &inNormWeight};
	if (not isMamba3MimoConfigValid(inConfig, inNormWeight)
		or not areMamba3MimoInputsFloat32(inputs, oa::arraySize(inputs))
		or inDOut.getShape() != oa::MatrixShape{B, L, H, P}
		or inC.getShape() != oa::MatrixShape{B, L, R * G, N}
		or inB.getShape() != oa::MatrixShape{B, L, R * G, N}
		or inX.getShape() != oa::MatrixShape{B, L, H, P}
		or inZ.getShape() != oa::MatrixShape{B, L, H, P}
		or inAdt.getShape() != oa::MatrixShape{B, L, H}
		or inDt.getShape() != oa::MatrixShape{B, L, H}
		or inTrap.getShape() != oa::MatrixShape{B, L, H}
		or inAngle.getShape() != oa::MatrixShape{B, L, A}
		or inCBias.getShape() != oa::MatrixShape{H, R, N}
		or inBBias.getShape() != oa::MatrixShape{H, R, N}
		or inD.getShape() != oa::MatrixShape{H}
		or inMimoX.getShape() != oa::MatrixShape{H, R, P}
		or inMimoZ.getShape() != oa::MatrixShape{H, R, P}
		or inMimoO.getShape() != oa::MatrixShape{H, R, P})
	{
		logInvalidMamba3Mimo("Mamba3MimoBwd");
		return {};
	}
	auto& ctx = oa::ExecutionSession::getActive();
	oa::OpLoweringScope lowering(ctx);
	const oa::U64 configIdentity = oa::FnMatrixPrivate::ssmConfigIdentity(inConfig);

	oa::Matrix q = oa::FnMatrix::empty(oa::MatrixShape{B, L, H, R, N}, inC.getDtype());
	oa::Matrix k = oa::FnMatrix::empty(q.getShape(), inB.getDtype());
	oa::Matrix theta = oa::FnMatrix::empty(oa::MatrixShape{B, H, L, A}, inAngle.getDtype());
	oa::Matrix hPrev = oa::FnMatrix::empty(oa::MatrixShape{B * H, L, P, N}, inX.getDtype());
	oa::Matrix decay = oa::FnMatrix::empty(inDt.getShape(), inDt.getDtype());
	oa::Matrix gamma = oa::FnMatrix::empty(inDt.getShape(), inDt.getDtype());
	oa::Matrix scale = oa::FnMatrix::empty(inDt.getShape(), inDt.getDtype());
	struct PreparePush { oa::U32 B, L, H, G, P, N, A, R; }
		preparePush{B, L, H, G, P, N, A, R};
	oa::BufferAccess prepareAccess[] = {
		oa::BufferAccess::Read, oa::BufferAccess::Read, oa::BufferAccess::Read,
		oa::BufferAccess::Read, oa::BufferAccess::Read, oa::BufferAccess::Read,
		oa::BufferAccess::Read, oa::BufferAccess::Read, oa::BufferAccess::Read,
		oa::BufferAccess::Read, oa::BufferAccess::Write, oa::BufferAccess::Write,
		oa::BufferAccess::Write, oa::BufferAccess::Write, oa::BufferAccess::Write,
		oa::BufferAccess::Write, oa::BufferAccess::Write};
	ctx.add( "Mamba3MimoBwdPrepare",
		{&inC, &inB, &inX, &inAdt, &inDt, &inTrap, &inAngle,
		 &inCBias, &inBBias, &inMimoX, &q, &k, &theta, &hPrev,
		 &decay, &gamma, &scale}, prepareAccess, &preparePush,
		sizeof(preparePush), B * H, 1, 1);

	oa::Matrix dPre = oa::FnMatrix::empty(oa::MatrixShape{B, L, H, R, P}, inX.getDtype());
	oa::Matrix dZ = oa::FnMatrix::empty(inZ.getShape(), inZ.getDtype());
	oa::Matrix dMimoZBatch = oa::FnMatrix::empty(oa::MatrixShape{B, H, R, P}, inMimoZ.getDtype());
	oa::Matrix dMimoOBatch = oa::FnMatrix::empty(oa::MatrixShape{B, H, R, P}, inMimoO.getDtype());
	oa::Matrix dNormBatch = oa::FnMatrix::empty(oa::MatrixShape{B, H, P}, inX.getDtype());
	struct PostPush {
		oa::U32 B, L, H, P, N, R, hasZ, hasD, HasNorm; oa::F32 eps;
	} postPush{B, L, H, P, N, R, inConfig.hasZ, inConfig.hasD,
		inConfig.hasOutNorm, 1e-5F};
	oa::BufferAccess postAccess[] = {
		oa::BufferAccess::Read, oa::BufferAccess::Read, oa::BufferAccess::Read,
		oa::BufferAccess::Read, oa::BufferAccess::Read, oa::BufferAccess::Read,
		oa::BufferAccess::Read, oa::BufferAccess::Read, oa::BufferAccess::Read,
		oa::BufferAccess::Read, oa::BufferAccess::Read, oa::BufferAccess::Read,
		oa::BufferAccess::Read, oa::BufferAccess::Write, oa::BufferAccess::Write,
		oa::BufferAccess::Write, oa::BufferAccess::Write, oa::BufferAccess::Write};
	ctx.add( "Mamba3MimoBwdPost",
		{&inDOut, &inX, &inZ, &inD, &inMimoX, &inMimoZ, &inMimoO,
		 &inNormWeight, &q, &k, &hPrev, &decay, &gamma, &dPre, &dZ,
		 &dMimoZBatch, &dMimoOBatch, &dNormBatch}, postAccess,
		&postPush, sizeof(postPush), B * H, 1, 1);

	oa::Matrix dQ = oa::FnMatrix::empty(q.getShape(), q.getDtype());
	oa::Matrix dK = oa::FnMatrix::empty(k.getShape(), k.getDtype());
	oa::Matrix dX = oa::FnMatrix::empty(inX.getShape(), inX.getDtype());
	oa::Matrix dAdt = oa::FnMatrix::empty(inAdt.getShape(), inAdt.getDtype());
	oa::Matrix dDt = oa::FnMatrix::empty(inDt.getShape(), inDt.getDtype());
	oa::Matrix dTrap = oa::FnMatrix::empty(inTrap.getShape(), inTrap.getDtype());
	oa::Matrix dDBatch = oa::FnMatrix::empty(oa::MatrixShape{B, H}, inD.getDtype());
	oa::Matrix dMimoXBatch = oa::FnMatrix::empty(oa::MatrixShape{B, H, R, P}, inMimoX.getDtype());
	oa::Matrix dScaleTmp = oa::FnMatrix::empty(inDt.getShape(), inDt.getDtype());
	struct CorePush { oa::U32 B, L, H, P, N, R, hasD; }
		corePush{B, L, H, P, N, R, inConfig.hasD};
	oa::BufferAccess coreAccess[] = {
		oa::BufferAccess::Read, oa::BufferAccess::Read, oa::BufferAccess::Read,
		oa::BufferAccess::Read, oa::BufferAccess::Read, oa::BufferAccess::Read,
		oa::BufferAccess::Read, oa::BufferAccess::Read, oa::BufferAccess::Read,
		oa::BufferAccess::Read, oa::BufferAccess::Read, oa::BufferAccess::Read,
		oa::BufferAccess::Write, oa::BufferAccess::Write, oa::BufferAccess::Write,
		oa::BufferAccess::Write, oa::BufferAccess::Write, oa::BufferAccess::Write,
		oa::BufferAccess::Write, oa::BufferAccess::Write, oa::BufferAccess::Write};
	ctx.add( "Mamba3MimoBwdCore",
		{&dPre, &inX, &inD, &inMimoX, &q, &k, &hPrev, &decay, &gamma,
		 &scale, &inDt, &inTrap, &dQ, &dK, &dX, &dAdt, &dDt, &dTrap,
		 &dDBatch, &dMimoXBatch, &dScaleTmp}, coreAccess,
		&corePush, sizeof(corePush), B * H, 1, 1);

	oa::Matrix dCHead = oa::FnMatrix::empty(q.getShape(), inC.getDtype());
	oa::Matrix dBHead = oa::FnMatrix::empty(k.getShape(), inB.getDtype());
	oa::Matrix dAngleHead = oa::FnMatrix::empty(oa::MatrixShape{B, H, L, A}, inAngle.getDtype());
	oa::Matrix dCBiasBatch = oa::FnMatrix::empty(oa::MatrixShape{B, H, R, N}, inCBias.getDtype());
	oa::Matrix dBBiasBatch = oa::FnMatrix::empty(oa::MatrixShape{B, H, R, N}, inBBias.getDtype());
	struct RotatePush { oa::U32 B, L, H, G, N, A, R; }
		rotatePush{B, L, H, G, N, A, R};
	oa::BufferAccess rotateAccess[] = {
		oa::BufferAccess::Read, oa::BufferAccess::Read, oa::BufferAccess::Read,
		oa::BufferAccess::Read, oa::BufferAccess::Read, oa::BufferAccess::Read,
		oa::BufferAccess::Read, oa::BufferAccess::Read, oa::BufferAccess::Read,
		oa::BufferAccess::Write, oa::BufferAccess::Write, oa::BufferAccess::Write,
		oa::BufferAccess::Write, oa::BufferAccess::Write, oa::BufferAccess::ReadWrite};
	ctx.add( "Mamba3MimoBwdRotate",
		{&dQ, &dK, &inC, &inB, &inDt, &inAngle, &inCBias, &inBBias,
		 &theta, &dCHead, &dBHead, &dAngleHead, &dCBiasBatch,
		 &dBBiasBatch, &dDt}, rotateAccess, &rotatePush,
		sizeof(rotatePush), B * H, 1, 1);

	oa::Mamba3MimoBwdResult result;
	result.dC = oa::FnMatrix::empty(inC.getShape(), inC.getDtype());
	result.dB = oa::FnMatrix::empty(inB.getShape(), inB.getDtype());
	result.dX = dX;
	result.dZ = dZ;
	result.dAdt = dAdt;
	result.dDt = dDt;
	result.dTrap = dTrap;
	result.dAngle = oa::FnMatrix::empty(inAngle.getShape(), inAngle.getDtype());
	result.dCBias = oa::FnMatrix::empty(inCBias.getShape(), inCBias.getDtype());
	result.dBBias = oa::FnMatrix::empty(inBBias.getShape(), inBBias.getDtype());
	result.dD = oa::FnMatrix::empty(inD.getShape(), inD.getDtype());
	result.dMimoX = oa::FnMatrix::empty(inMimoX.getShape(), inMimoX.getDtype());
	result.dMimoZ = oa::FnMatrix::empty(inMimoZ.getShape(), inMimoZ.getDtype());
	result.dMimoO = oa::FnMatrix::empty(inMimoO.getShape(), inMimoO.getDtype());
	result.dNormWeight = oa::FnMatrix::empty(oa::MatrixShape{H, P}, inX.getDtype());
	const oa::U32 groupCount = B * L * R * G * N;
	const oa::U32 angleCount = B * L * A;
	const oa::U32 biasCount = H * R * N;
	const oa::U32 mimoCount = H * R * P;
	const oa::U32 normCount = H * P;
	const oa::U32 count = groupCount + angleCount + biasCount + H + mimoCount + normCount;
	struct ReducePush { oa::U32 B, L, H, G, P, N, A, R, Count; }
		reducePush{B, L, H, G, P, N, A, R, count};
	oa::BufferAccess reduceAccess[] = {
		oa::BufferAccess::Read, oa::BufferAccess::Read, oa::BufferAccess::Read,
		oa::BufferAccess::Read, oa::BufferAccess::Read, oa::BufferAccess::Read,
		oa::BufferAccess::Read, oa::BufferAccess::Read, oa::BufferAccess::Read,
		oa::BufferAccess::Read, oa::BufferAccess::Write, oa::BufferAccess::Write,
		oa::BufferAccess::Write, oa::BufferAccess::Write, oa::BufferAccess::Write,
		oa::BufferAccess::Write, oa::BufferAccess::Write, oa::BufferAccess::Write,
		oa::BufferAccess::Write, oa::BufferAccess::Write};
	ctx.add( "Mamba3MimoBwdReduce",
		{&dCHead, &dBHead, &dAngleHead, &dCBiasBatch, &dBBiasBatch,
		 &dDBatch, &dMimoXBatch, &dMimoZBatch, &dMimoOBatch, &dNormBatch,
		 &result.dC, &result.dB, &result.dAngle, &result.dCBias,
		 &result.dBBias, &result.dD, &result.dMimoX, &result.dMimoZ,
		 &result.dMimoO, &result.dNormWeight}, reduceAccess, &reducePush,
		sizeof(reducePush), oa::divCeil(count, 256u));

	if (not lowering.commit(
		oa::detail::opRegistry::FnMatrix::mamba3MimoBwd,
		{&inDOut, &inC, &inB, &inX, &inZ, &inAdt, &inDt, &inTrap,
			&inAngle, &inCBias, &inBBias, &inD, &inMimoX, &inMimoZ,
			&inMimoO, &inNormWeight},
		{&result.dC, &result.dB, &result.dX, &result.dZ, &result.dAdt,
			&result.dDt, &result.dTrap, &result.dAngle, &result.dCBias,
			&result.dBBias, &result.dD, &result.dMimoX, &result.dMimoZ,
			&result.dMimoO, &result.dNormWeight},
		{oa::OpAttribute::fromUnsignedInteger("configIdentity", configIdentity)}).isOk())
	{
		return {};
	}
	return result;
}

oa::Matrix oa::FnMatrix::mamba3MimoStep(
	const oa::Matrix& inC, const oa::Matrix& inB,
	const oa::Matrix& inX, const oa::Matrix& inZ,
	const oa::Matrix& inAdt, const oa::Matrix& inDt,
	const oa::Matrix& inTrap, const oa::Matrix& inAngle,
	const oa::Matrix& inCBias, const oa::Matrix& inBBias,
	const oa::Matrix& inD, const oa::Matrix& inMimoX,
	const oa::Matrix& inMimoZ, const oa::Matrix& inMimoO,
	const oa::Matrix& inNormWeight, const oa::Matrix& inSsmState,
	const oa::Matrix& inAngleState, const oa::Matrix& inKState,
	const oa::Matrix& inVState, const oa::SsmConfig& inConfig)
{
	const oa::U32 B = inConfig.batch, H = inConfig.nHeads;
	const oa::U32 G = inConfig.nGroups == 0u ? H : inConfig.nGroups;
	const oa::U32 P = inConfig.headDim, N = inConfig.stateSize;
	const oa::U32 A = inConfig.numRopeAngles, R = inConfig.mimoRank;
	const oa::Matrix* inputs[] = {
		&inC, &inB, &inX, &inZ, &inAdt, &inDt, &inTrap, &inAngle,
		&inCBias, &inBBias, &inD, &inMimoX, &inMimoZ, &inMimoO,
		&inNormWeight, &inSsmState, &inAngleState, &inKState, &inVState};
	if (inConfig.seqLen != 1u
		or not isMamba3MimoConfigValid(inConfig, inNormWeight)
		or not areMamba3MimoInputsFloat32(inputs, oa::arraySize(inputs))
		or inC.getShape() != oa::MatrixShape{B, 1, R * G, N}
		or inB.getShape() != oa::MatrixShape{B, 1, R * G, N}
		or inX.getShape() != oa::MatrixShape{B, 1, H, P}
		or inZ.getShape() != oa::MatrixShape{B, 1, H, P}
		or inAdt.getShape() != oa::MatrixShape{B, 1, H}
		or inDt.getShape() != oa::MatrixShape{B, 1, H}
		or inTrap.getShape() != oa::MatrixShape{B, 1, H}
		or inAngle.getShape() != oa::MatrixShape{B, 1, A}
		or inCBias.getShape() != oa::MatrixShape{H, R, N}
		or inBBias.getShape() != oa::MatrixShape{H, R, N}
		or inD.getShape() != oa::MatrixShape{H}
		or inMimoX.getShape() != oa::MatrixShape{H, R, P}
		or inMimoZ.getShape() != oa::MatrixShape{H, R, P}
		or inMimoO.getShape() != oa::MatrixShape{H, R, P}
		or inSsmState.getShape() != oa::MatrixShape{B, H, P, N}
		or inAngleState.getShape() != oa::MatrixShape{B, H, A}
		or inKState.getShape() != oa::MatrixShape{B, H, R, N}
		or inVState.getShape() != oa::MatrixShape{B, H, R, P})
	{
		logInvalidMamba3Mimo("Mamba3MimoStep");
		return {};
	}
	auto& ctx = oa::ExecutionSession::getActive();
	oa::OpLoweringScope lowering(ctx);
	const oa::U64 configIdentity = oa::FnMatrixPrivate::ssmConfigIdentity(inConfig);
	oa::Matrix output = oa::FnMatrix::empty(oa::MatrixShape{B, 1, H, P}, inX.getDtype());
	struct Push {
		oa::U32 B, H, G, P, N, A, R, hasZ, hasD, HasNorm; oa::F32 eps;
	} push{B, H, G, P, N, A, R, inConfig.hasZ, inConfig.hasD,
		inConfig.hasOutNorm, 1e-5F};
	oa::BufferAccess access[] = {
		oa::BufferAccess::Read, oa::BufferAccess::Read, oa::BufferAccess::Read,
		oa::BufferAccess::Read, oa::BufferAccess::Read, oa::BufferAccess::Read,
		oa::BufferAccess::Read, oa::BufferAccess::Read, oa::BufferAccess::Read,
		oa::BufferAccess::Read, oa::BufferAccess::Read, oa::BufferAccess::Read,
		oa::BufferAccess::Read, oa::BufferAccess::Read, oa::BufferAccess::Read,
		oa::BufferAccess::Write, oa::BufferAccess::ReadWrite,
		oa::BufferAccess::ReadWrite, oa::BufferAccess::ReadWrite,
		oa::BufferAccess::ReadWrite};
	ctx.add( "Mamba3MimoStep",
		{&inC, &inB, &inX, &inZ, &inAdt, &inDt, &inTrap, &inAngle,
		 &inCBias, &inBBias, &inD, &inMimoX, &inMimoZ, &inMimoO,
		 &inNormWeight, &output, &inSsmState, &inAngleState, &inKState,
		 &inVState}, access, &push, sizeof(push), B * H, 1, 1);
	if (not lowering.commit(
		oa::detail::opRegistry::FnMatrix::mamba3MimoStep,
		{&inSsmState, &inAngleState, &inKState, &inVState,
			&inC, &inB, &inX, &inZ, &inAdt, &inDt, &inTrap, &inAngle,
			&inCBias, &inBBias, &inD, &inMimoX, &inMimoZ, &inMimoO,
			&inNormWeight}, {&output},
		{oa::OpAttribute::fromUnsignedInteger("configIdentity", configIdentity)}).isOk())
	{
		return {};
	}
	return output;
}

// ============================================================================
// Mamba3Preprocess — fused in_proj split + RMSNorm + dt + A·dt (forward)
//
// Replaces 11-13 small dispatches with a single kernel. One workgroup per
// row of the projected [B*S, dInProj] tensor.
// ============================================================================

oa::Mamba3PreprocessResult oa::FnMatrix::mamba3Preprocess(
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

	// offsets within each row (must match preprocess() in Mamba3.cpp)
	oa::I32 xOffset = dInner;
	oa::I32 bOffset = 2 * dInner;
	oa::I32 cOffset = bOffset + dState * nGroups * mimoRank;
	oa::I32 ddDtOffset = cOffset + dState * nGroups * mimoRank;
	oa::I32 ddAOffset = ddDtOffset + nHeads;
	oa::I32 trapOffset = ddAOffset + nHeads;
	oa::I32 angleOffset = trapOffset + nHeads;

	oa::I32 bcWidth = dState * nGroups * mimoRank;

	// allocate outputs
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
	ctx.add( "Mamba3Preprocess",
		{&inProjected, &inDtBias, &zOut, &xOut, &bhOut, &chOut, &dtOut, &adtOut, &trapOut, &angleOut},
		access, &push, sizeof(push), static_cast<oa::U32>(batchTimesSeq));
	const auto semantic = lowering.commitWithId(
		oa::detail::opRegistry::FnMatrix::mamba3Preprocess,
		{&inProjected, &inDtBias},
		{&xOut, &zOut, &bhOut, &chOut, &dtOut, &adtOut, &trapOut, &angleOut},
		{oa::OpAttribute::fromUnsignedInteger(
			"configIdentity", configIdentity)});
	if (not semantic.isOk()) return {};

	// autograd: create 8 thin grad nodes sharing one SharedState
	bool needGrad = oa::FnAutograd::isEnabled() and
		(inProjected.requiresGrad() or inDtBias.requiresGrad());
	if (needGrad) {
		auto state = oa::makeShared<oa::GradMamba3Preprocess::SharedState>();
		state->projected = inProjected;
		state->dtBias = inDtBias;
		state->config = inConfig;

		auto attachGrad = [&](oa::Matrix& out, oa::I32 gradIndex, oa::U32 semanticIndex) {
			auto gradFn = oa::makeShared<oa::GradMamba3Preprocess>();
			gradFn->state_ = state;
			gradFn->outputIndex_ = gradIndex;
			gradFn->setGraphInputs(oa::Vec<oa::Matrix>{inProjected, inDtBias});
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
// Mamba3PreprocessBwd — fused backward (1 dispatch instead of 11+)
// ============================================================================

oa::Mamba3PreprocessBwdResult oa::FnMatrix::mamba3PreprocessBwd(
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
	ctx.add( "Mamba3PreprocessBwd",
		{&inProjected, &inDtBias, &inDZ, &inDX, &inDBh, &inDCh,
		 &inDDT, &inDADT, &inDTrap, &inDAngle, &dProj, &dDtBias},
		access, &push, sizeof(push), static_cast<oa::U32>(batchTimesSeq));

	if (not lowering.commit(
		oa::detail::opRegistry::FnMatrix::mamba3PreprocessBwd,
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
