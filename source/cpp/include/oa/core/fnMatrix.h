#pragma once

#include <oa/core/matrix.h>
#include <oa/core/matrixShape.h>
#include <oa/core/status.h>

namespace oa {

class Engine;

struct TopKResult {
	Matrix values;   // top-k values, shape same as input with last dim = k
	Matrix indices;  // top-k indices (Int32), same shape as values
};

struct MoeExpertPlan {
	Matrix counts;       // UInt32 [E]
	Matrix offsets;      // UInt32 [E+1], exclusive; offsets[E] == T*K
	Matrix packedToken;  // UInt32 [T*K]
	Matrix packedExpert; // UInt32 [T*K]
	Matrix packedSlot;   // UInt32 [T*K]
	Matrix inverse;      // UInt32 [T*K]
};

struct GroupedGemmMBwdResult {
	Matrix dInput;
	Matrix dWeight;
};

struct GroupedLinearMBwdResult {
	Matrix dInput;
	Matrix dWeight;
	Matrix dBias;
};

struct MoeCombineBwdResult {
	Matrix dPacked;
	Matrix dRouteGate;
};

struct CompactRowsResult {
	Matrix values;       // [T,D], first count rows valid
	Matrix rowMap;       // UInt32 [T], first count entries map packed -> source row
	Matrix count;        // UInt32 [1]
	Matrix dispatchArgs; // UInt32 [3], GPU-authored ceil(count*D/256),1,1
};

struct SsmConfig {
	oa::U32 batch         = 0;
	oa::U32 seqLen        = 0;
	oa::U32 nHeads        = 0;
	oa::U32 nGroups       = 0;  // 0 means nHeads; nHeads must be divisible by nGroups
	oa::U32 headDim       = 0;  // <= 128
	oa::U32 stateSize     = 0;  // <= 128
	oa::U32 numRopeAngles = 0;  // <= 64
	oa::U32 mimoRank      = 1;  // <= 8
	oa::U32 hasZ          = 0;
	oa::U32 hasD          = 0;
	oa::U32 hasOutNorm    = 0;
};

struct SsmBwdResult {
	Matrix dC;     // [B,L,G,N]
	Matrix dB;     // [B,L,G,N]
	Matrix dX;     // [B,L,H,P]
	Matrix dZ;     // [B,L,H,P]
	Matrix dAdt;   // [B,L,H]
	Matrix dDt;    // [B,L,H]
	Matrix dTrap;  // [B,L,H]
	Matrix dAngle; // [B,L,A]
	Matrix dCBias; // [H,N]
	Matrix dBBias; // [H,N]
	Matrix dD;     // [H]
};

struct Mamba3MimoBwdResult {
	Matrix dC, dB, dX, dZ, dAdt, dDt, dTrap, dAngle, dCBias, dBBias, dD;
	Matrix dMimoX, dMimoZ, dMimoO, dNormWeight;
};

struct EmpyrealmAdtBwdResult {
	Matrix dDdA; // [B*S, H]
	Matrix dDt;  // [B*S, H]
};

struct EmpyrealmDtAdtResult {
	Matrix dt;  // [B*S, H]
	Matrix adt; // [B*S, H]
};

struct Mamba3PreprocessResult {
	Matrix x;     // [B*S, dInner]
	Matrix z;     // [B*S, dInner]
	Matrix bh;    // [B*S, H*dState] (rmsnorm'd, flat)
	Matrix ch;    // [B*S, H*dState] (rmsnorm'd, flat)
	Matrix dt;    // [B*S, H]
	Matrix adt;   // [B*S, H]
	Matrix trap;  // [B*S, H]
	Matrix angle; // [B*S, numRopeAngles]
};

struct Mamba3PreprocessConfig {
	oa::I32 dInner        = 0;
	oa::I32 dState        = 0;
	oa::I32 nHeads        = 0;
	oa::I32 numRopeAngles = 0;
	oa::I32 nGroups       = 0;
	oa::I32 mimoRank      = 0;
	oa::F32 eps           = 0.0F;
	oa::F32 dtMin         = 0.0F;
	oa::F32 dtMax         = 0.0F;
	oa::F32 aFloor        = 0.0F;
};

struct Mamba3PreprocessBwdResult {
	Matrix dProjected; // [B*S, dInProj]
	Matrix dDtBias;    // [H]
};

struct ResidualRmsNormResult {
	Matrix out;      // rmsNorm(a + b, weight, eps)
	Matrix residual; // a + b (pre-norm sum)
};

// oa::FnMatrix — stateless matrix/tensor operations.
// usage: oa::FnMatrix::zeros({10, 20}), oa::FnMatrix::silu(x), etc.
namespace FnMatrix {

	// --- Configuration ---
	// weight allocation follows the precision of the active engine context.
	[[nodiscard]] oa::ScalarType weightDtype();

	// --- Factory functions ---
	[[nodiscard]] Matrix empty(
		MatrixShape inShape,
		oa::ScalarType inDtype = weightDtype(),
		oa::MemoryPlacement inPlacement = oa::MemoryPlacement::Auto
	);
	[[nodiscard]] Matrix zeros(MatrixShape inShape, oa::ScalarType inDtype = weightDtype());
	[[nodiscard]] Matrix ones(MatrixShape inShape, oa::ScalarType inDtype = weightDtype());
	[[nodiscard]] Matrix full(MatrixShape inShape, oa::F64 inValue, oa::ScalarType inDtype = weightDtype());
	[[nodiscard]] Matrix rand(MatrixShape inShape, oa::ScalarType inDtype = weightDtype());
	[[nodiscard]] Matrix randN(MatrixShape inShape, oa::ScalarType inDtype = weightDtype());
	[[nodiscard]] Matrix randXavier(MatrixShape inShape, oa::ScalarType inDtype = weightDtype());
	[[nodiscard]] Matrix randGlorotUniform(MatrixShape inShape, oa::ScalarType inDtype = weightDtype());
	[[nodiscard]] Matrix randKaimingUniform(MatrixShape inShape, oa::ScalarType inDtype = weightDtype());
	[[nodiscard]] Matrix fromBytes(oa::Span<const oa::U8> inData, MatrixShape inShape, oa::ScalarType inDtype = weightDtype());
	[[nodiscard]] Matrix fromInt32(oa::Span<const oa::I32> inData, MatrixShape inShape, oa::ScalarType inDtype = oa::ScalarType::Int32);
	[[nodiscard]] Matrix causalMask(oa::I64 inSeqLen);

	// --- dtype cast ---
	// cast: allocate a new matrix of inDtype and convert inSrc into it.
	// castInto: convert inSrc into the pre-allocated outDst in place.
	[[nodiscard]] Matrix cast(const Matrix& inSrc, oa::ScalarType inDtype);
	void castInto(const Matrix& inSrc, Matrix& outDst);

	// --- RNG ---
	// seed the host-side seed generator. call once at startup for reproducibility.
	void setRngSeed(oa::U64 inSeed);
	// Inverted dropout. inP must be [0,1).
	[[nodiscard]] Matrix dropout(const Matrix& inA, oa::F32 inP, oa::U64 inSeed = 0);
	// samples one class per last-axis row. temperature <= 0 is greedy;
	// inTopK <= 0 keeps the full vocabulary; inTopP is clamped to (0,1].
	[[nodiscard]] Matrix sampleLogits(
		const Matrix& inLogits, oa::F32 inTemperature = 1.0F, oa::I32 inTopK = 0, oa::F32 inTopP = 1.0F, oa::U64 inSeed = 0
	);

	// --- Transfer ---
	/// Copy device matrix data to host memory. inBytes must be >= inSrc.byteSize().
	[[nodiscard]] oa::Status copyToHost(const Matrix& inSrc, void* outHost, oa::U64 inBytes);
	/// Extract first element as F32 scalar (requires single-element matrix).
	[[nodiscard]] oa::F32 scalar(const Matrix& inSrc);

	// --- Core operations ---
	// Generated overloads from operation schemas.
	// Regenerate via: python3 tools/gen/fn/generate.py --live
	#include <oa/core/fnmatrix/fnMatrix.gen.h>

	// in-place operations
	void addInPlace(Matrix& inSelf, const Matrix& inOther);
	void subInPlace(Matrix& inSelf, const Matrix& inOther);
	void mulInPlace(Matrix& inSelf, const Matrix& inOther);
	void divInPlace(Matrix& inSelf, const Matrix& inOther);
	void scaleInPlace(Matrix& inSelf, oa::F32 inScalar);
	void addScalarInPlace(Matrix& inSelf, oa::F32 inScalar);
	void subScalarInPlace(Matrix& inSelf, oa::F32 inScalar);
	void divScalarInPlace(Matrix& inSelf, oa::F32 inScalar);

	// Element-wise binary ops — broadcast-aware.
	[[nodiscard]] Matrix sub(const Matrix& inA, const Matrix& inB);
	[[nodiscard]] Matrix mul(const Matrix& inA, const Matrix& inB);
	[[nodiscard]] Matrix div(const Matrix& inA, const Matrix& inB);

	// backward ops
	[[nodiscard]] Matrix softmaxBwd(
		const Matrix& inForwardOutput, const Matrix& inGradOutput, oa::I32 inDim = -1
	);
	[[nodiscard]] Matrix logSoftmaxBwd(
		const Matrix& inForwardOutput, const Matrix& inGradOutput, oa::I32 inDim = -1
	);
	[[nodiscard]] Matrix maxBwd(
		const Matrix& inInput, const Matrix& inMaxValue, const Matrix& inGradOutput
	);

	// Indexing
	[[nodiscard]] Matrix gather(const Matrix& inSelf, const Matrix& inIndices);
	[[nodiscard]] Matrix gatherBwd(
		const Matrix& inIndices, const Matrix& inGradOutput, oa::I32 inVocabSize,	oa::I32 inEmbedDim
	);
	[[nodiscard]] Matrix gatherLastDim(const Matrix& inSelf, const Matrix& inIndices);
	[[nodiscard]] Matrix gatherLastDimBwd(const Matrix& inGradOut, const Matrix& inIndices, oa::I32 inInputWidth);
	[[nodiscard]] Matrix concat(oa::Span<Matrix> inInputs, oa::I32 inDim = 0);
	[[nodiscard]] oa::Vector<Matrix> split(const Matrix& inSelf, oa::Span<oa::I64> inSizes, oa::I32 inDim = 0);
	[[nodiscard]] Matrix sliceBwd(
		MatrixShape inInputShape, oa::I32 inDim, oa::I64 inStart,	oa::I64 inEnd, const Matrix& inDOut
	);

	// Shape helpers
	[[nodiscard]] Matrix repeatInterleave(const Matrix& inA, oa::I32 inRepeats, oa::I32 inDim);
	[[nodiscard]] Matrix repeatInterleaveBwd(const Matrix& inGradOut, MatrixShape inInputShape, oa::I32 inRepeats, oa::I32 inDim);
	[[nodiscard]] Matrix causalMask(const Matrix& inScores);
	[[nodiscard]] Matrix causalMaskBwd(const Matrix& inGradOut);
	[[nodiscard]] TopKResult topK(const Matrix& inA, oa::I32 inK, oa::I32 inDim = -1);
	[[nodiscard]] Matrix topKMask(const Matrix& inIndices, oa::I32 inNumExperts);
	[[nodiscard]] Matrix equal(const Matrix& inA, oa::F32 inValue);
	[[nodiscard]] Matrix greaterEqual(const Matrix& inA, oa::F32 inValue);
	[[nodiscard]] Matrix categoricalAccuracyCount(const Matrix& inLogits, const Matrix& inLabels);
	[[nodiscard]] Matrix maskedCategoricalAccuracyCount(
		const Matrix& inLogits, const Matrix& inLabels, const Matrix& inMask
	);
	void moeRoutingBiasUpdate(
		const Matrix& inSelectionMask, Matrix& inOutBias,	oa::I32 inExpertsPerToken, oa::F32 inGamma
	);

	[[nodiscard]] MoeExpertPlan moeExpertPlan(const Matrix& inExpertIndices, oa::I32 inNumExperts);

	[[nodiscard]] Matrix moeRouteWeights(const Matrix& inProbs, const Matrix& inExpertIndices);
	[[nodiscard]] Matrix moeRouteWeightsBwd(
		const Matrix& inDOut,	const Matrix& inProbs, const Matrix& inExpertIndices,	const Matrix& inRouteWeights
	);

	[[nodiscard]] Matrix groupedGemmM(const Matrix& inX, const Matrix& inWeight, const Matrix& inOffsets);
	[[nodiscard]] Matrix groupedLinearM(const Matrix& inX, const Matrix& inWeight, const Matrix& inBias, const Matrix& inOffsets);
	[[nodiscard]] GroupedGemmMBwdResult groupedGemmMBwd(const Matrix& inDOut,
		const Matrix& inX, const Matrix& inWeight, const Matrix& inOffsets);
	[[nodiscard]] Matrix groupedLinearMBiasBwd(const Matrix& inDOut, const Matrix& inOffsets, oa::I32 inNumExperts);
	[[nodiscard]] GroupedLinearMBwdResult groupedLinearMBwd(
		const Matrix& inDOut, const Matrix& inX, const Matrix& inWeight, const Matrix& inOffsets
	);
	[[nodiscard]] Matrix moeCombine(
		const Matrix& inPacked,	const Matrix& inRouteGate, const Matrix& inInverse,	const Matrix& inPackedSlot
	);
	[[nodiscard]] MoeCombineBwdResult moeCombineBwd(const Matrix& inDOut,
		const Matrix& inPacked, const Matrix& inRouteGate,
		const Matrix& inInverse, const Matrix& inPackedSlot
	);
	[[nodiscard]] Matrix moeGather(const Matrix& inSelf, const Matrix& inIndices, const Matrix& inInverse);
	[[nodiscard]] Matrix moeGatherBwd(const Matrix& inSource,	const Matrix& inInverse, oa::I32 inOutRows);
	[[nodiscard]] Matrix scatterAddRows(const Matrix& inSource,	const Matrix& inIndices, oa::I32 inOutRows);
	[[nodiscard]] CompactRowsResult compactRows(const Matrix& inSelf, const Matrix& inMask);
	[[nodiscard]] Matrix compactRowsBwd(
		const Matrix& inGradOut, const Matrix& inRowMap, const Matrix& inCount,	MatrixShape inInputShape
	);
	[[nodiscard]] Matrix compactRowsBwd(
		const Matrix& inGradOut, const Matrix& inRowMap, const Matrix& inCount,
		const Matrix& inDispatchArgs, MatrixShape inInputShape
	);
	[[nodiscard]] Matrix scatterRows(
		const Matrix& inSelf, const Matrix& inSource,	const Matrix& inRowMap, const Matrix& inCount
	);
	[[nodiscard]] Matrix scatterRows(const Matrix& inSelf, const Matrix& inSource, const CompactRowsResult& inPlan);
	[[nodiscard]] Matrix scatterRowsBwdSource(
		const Matrix& inGradOut, const Matrix& inRowMap, const Matrix& inCount
	);
	[[nodiscard]] Matrix scatterRowsBwdSource(
		const Matrix& inGradOut, const Matrix& inRowMap, const Matrix& inCount,	const Matrix& inDispatchArgs
	);

	// --- Mamba-3 SISO selective scan ---
	[[nodiscard]] Matrix mamba3Siso(
		const Matrix& inC, const Matrix& inB, const Matrix& inX, const Matrix& inZ,
		const Matrix& inAdt, const Matrix& inDt, const Matrix& inTrap, const Matrix& inAngle,
		const Matrix& inCBias, const Matrix& inBBias, const Matrix& inD,
		const SsmConfig& inConfig
	);

	[[nodiscard]] Matrix mamba3SisoStep(
		const Matrix& inC, const Matrix& inB, const Matrix& inX, const Matrix& inZ,
		const Matrix& inAdt, const Matrix& inDt, const Matrix& inTrap, const Matrix& inAngle,
		const Matrix& inCBias, const Matrix& inBBias, const Matrix& inD,
		const Matrix& inSsmState, const Matrix& inAngleState,
		const Matrix& inKState, const Matrix& inVState,
		const SsmConfig& inConfig
	);

	[[nodiscard]] SsmBwdResult mamba3SisoBwd(
		const Matrix& inDOut,
		const Matrix& inC, const Matrix& inB, const Matrix& inX, const Matrix& inZ,
		const Matrix& inAdt, const Matrix& inDt, const Matrix& inTrap, const Matrix& inAngle,
		const Matrix& inCBias, const Matrix& inBBias, const Matrix& inD,
		const SsmConfig& inConfig
	);

	// --- Mamba-3 MIMO selective scan ---
	[[nodiscard]] Matrix mamba3Mimo(
		const Matrix& inC, const Matrix& inB, const Matrix& inX, const Matrix& inZ,
		const Matrix& inAdt, const Matrix& inDt, const Matrix& inTrap, const Matrix& inAngle,
		const Matrix& inCBias, const Matrix& inBBias, const Matrix& inD,
		const Matrix& inMimoX, const Matrix& inMimoZ, const Matrix& inMimoO,
		const Matrix& inNormWeight, const SsmConfig& inConfig
	);

	[[nodiscard]] Mamba3MimoBwdResult mamba3MimoBwd(
		const Matrix& inDOut,
		const Matrix& inC, const Matrix& inB, const Matrix& inX, const Matrix& inZ,
		const Matrix& inAdt, const Matrix& inDt, const Matrix& inTrap, const Matrix& inAngle,
		const Matrix& inCBias, const Matrix& inBBias, const Matrix& inD,
		const Matrix& inMimoX, const Matrix& inMimoZ, const Matrix& inMimoO,
		const Matrix& inNormWeight, const SsmConfig& inConfig
	);

	[[nodiscard]] Matrix mamba3MimoStep(
		const Matrix& inC, const Matrix& inB, const Matrix& inX, const Matrix& inZ,
		const Matrix& inAdt, const Matrix& inDt, const Matrix& inTrap, const Matrix& inAngle,
		const Matrix& inCBias, const Matrix& inBBias, const Matrix& inD,
		const Matrix& inMimoX, const Matrix& inMimoZ, const Matrix& inMimoO,
		const Matrix& inNormWeight,
		const Matrix& inSsmState, const Matrix& inAngleState,
		const Matrix& inKState, const Matrix& inVState,
		const SsmConfig& inConfig
	);

	// --- empyrealm operations ---
	[[nodiscard]] Matrix empyrealmAdt(const Matrix& inDdA, const Matrix& inDt, oa::F32 inAFloor);
	[[nodiscard]] EmpyrealmAdtBwdResult empyrealmAdtBwd(const Matrix& inDOut, const Matrix& inDdA, const Matrix& inDt, oa::F32 inAFloor);
	[[nodiscard]] Matrix empyrealmDt(const Matrix& inX, oa::F32 inDtMin, oa::F32 inDtMax);
	[[nodiscard]] Matrix empyrealmDtBwd(const Matrix& inDOut, const Matrix& inX, oa::F32 inDtMin, oa::F32 inDtMax);

	[[nodiscard]] Matrix empyrealmSiso(
		const Matrix& inC, const Matrix& inB, const Matrix& inX, const Matrix& inZ,
		const Matrix& inAdt, const Matrix& inDt, const Matrix& inTrap, const Matrix& inAngle,
		const Matrix& inCBias, const Matrix& inBBias, const Matrix& inD,
		const SsmConfig& inConfig
	);

	[[nodiscard]] Matrix empyrealmSisoStep(
		const Matrix& inC, const Matrix& inB, const Matrix& inX, const Matrix& inZ,
		const Matrix& inAdt, const Matrix& inDt, const Matrix& inTrap, const Matrix& inAngle,
		const Matrix& inCBias, const Matrix& inBBias, const Matrix& inD,
		const Matrix& inSsmState, const Matrix& inAngleState,
		const Matrix& inKState, const Matrix& inVState,
		const SsmConfig& inConfig
	);

	[[nodiscard]] SsmBwdResult empyrealmSisoBwd(
		const Matrix& inDOut,
		const Matrix& inC, const Matrix& inB, const Matrix& inX, const Matrix& inZ,
		const Matrix& inAdt, const Matrix& inDt, const Matrix& inTrap, const Matrix& inAngle,
		const Matrix& inCBias, const Matrix& inBBias, const Matrix& inD,
		const SsmConfig& inConfig
	);

	[[nodiscard]] EmpyrealmDtAdtResult empyrealmDtAdt(
		const Matrix& inDtRaw, const Matrix& inDdA, oa::F32 inDtMin, oa::F32 inDtMax, oa::F32 inAFloor
	);

	// --- Mamba3Preprocess: fused in_proj split + RMSNorm + dt + A·dt ---
	[[nodiscard]] Mamba3PreprocessResult mamba3Preprocess(
		const Matrix& inProjected, const Matrix& inDtBias, const Mamba3PreprocessConfig& inConfig
	);

	[[nodiscard]] Mamba3PreprocessBwdResult mamba3PreprocessBwd(
		const Matrix& inProjected, const Matrix& inDtBias,
		const Matrix& inDZ, const Matrix& inDX,
		const Matrix& inDBh, const Matrix& inDCh,
		const Matrix& inDDT, const Matrix& inDADT,
		const Matrix& inDTrap, const Matrix& inDAngle,
		const Mamba3PreprocessConfig& inConfig
	);

	[[nodiscard]] Mamba3PreprocessResult empyrealmPreprocess(
		const Matrix& inProjected, const Matrix& inDtBias, const Mamba3PreprocessConfig& inConfig
	);

	[[nodiscard]] Mamba3PreprocessBwdResult empyrealmPreprocessBwd(
		const Matrix& inProjected, const Matrix& inDtBias,
		const Matrix& inDZ, const Matrix& inDX,
		const Matrix& inDBh, const Matrix& inDCh,
		const Matrix& inDDT, const Matrix& inDADT,
		const Matrix& inDTrap, const Matrix& inDAngle,
		const Mamba3PreprocessConfig& inConfig
	);

	// --- ResidualRmsNorm: fused residual + RmsNorm ---
	[[nodiscard]] ResidualRmsNormResult residualRmsNorm(
		const Matrix& inA, const Matrix& inB, const Matrix& inWeight, oa::F32 inEps
	);

} // namespace FnMatrix

} // namespace oa
