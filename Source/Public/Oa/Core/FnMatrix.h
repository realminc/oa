#pragma once

#include <Oa/Core/Matrix.h>
#include <Oa/Core/MatrixStorage.h>
#include <Oa/Core/MatrixShape.h>
#include <Oa/Core/Status.h>

class OaEngine;
class OaVkBatch;

struct OaTopKResult {
	OaMatrix Values;   // top-k values, shape same as input with last dim = k
	OaMatrix Indices;  // top-k indices (Int32), same shape as Values
};

// OaFnMatrix — namespace-based API for matrix/tensor operations across Core and ML.
// Domain APIs follow this function-set pattern with their own namespace when the
// vocabulary is not pure tensor math, e.g. OaFnImage (internal, rt-taking) for
// image/pixel transforms, OaFnAudio for audio DSP.
// Usage: OaFnMatrix::Zeros({10, 20}), OaFnMatrix::Silu(x), etc.
namespace OaFnMatrix {

// --- Configuration ---
// Global engine selection is internal runtime plumbing. OaFn bodies record through the
// current engine-owned OaContext and do not maintain another ownership layer here.
void SetWeightDtype(OaScalarType InDtype);
[[nodiscard]] OaScalarType GetWeightDtype();

// --- Factory functions ---
[[nodiscard]] OaMatrix Empty(OaMatrixShape InShape,
	OaScalarType InDtype = GetWeightDtype(),
	OaMemoryPlacement InPlacement = OaMemoryPlacement::Auto);
[[nodiscard]] OaMatrix Zeros(OaMatrixShape InShape, OaScalarType InDtype = GetWeightDtype());
[[nodiscard]] OaMatrix Ones(OaMatrixShape InShape, OaScalarType InDtype = GetWeightDtype());
[[nodiscard]] OaMatrix Full(OaMatrixShape InShape, OaF64 InValue, OaScalarType InDtype = GetWeightDtype());
[[nodiscard]] OaMatrix Rand(OaMatrixShape InShape, OaScalarType InDtype = GetWeightDtype());
[[nodiscard]] OaMatrix RandN(OaMatrixShape InShape, OaScalarType InDtype = GetWeightDtype());
[[nodiscard]] OaMatrix RandXavier(OaMatrixShape InShape, OaScalarType InDtype = GetWeightDtype());
[[nodiscard]] OaMatrix RandGlorotUniform(OaMatrixShape InShape, OaScalarType InDtype = GetWeightDtype());
[[nodiscard]] OaMatrix RandKaimingUniform(OaMatrixShape InShape, OaScalarType InDtype = GetWeightDtype());
[[nodiscard]] OaMatrix FromBytes(OaSpan<const OaU8> InData, OaMatrixShape InShape, OaScalarType InDtype = GetWeightDtype());
[[nodiscard]] OaMatrix FromInt32(OaSpan<const OaI32> InData, OaMatrixShape InShape, OaScalarType InDtype = OaScalarType::Int32);
[[nodiscard]] OaMatrix EmptyOn(OaMatrixShape InShape, OaScalarType InDtype, OaU32 InNodeIndex);
[[nodiscard]] OaMatrix CausalMask(OaI64 InSeqLen);

// --- Dtype cast (bf16 ⇆ fp32) ---
// A cast is a genuine mixed-precision op (src dtype ≠ dst dtype), so it uses
// dedicated hardcoded kernels rather than the DTYPE-branched Storage helpers.
// Cast:     allocate a new tensor of InDtype and convert InSrc into it.
// CastInto: convert InSrc into the pre-allocated OutDst *in place* (OutDst's
//           dtype decides the direction). Used for mixed-precision boundaries —
//           e.g. writing an fp32 optimizer master back into the bf16 weight the
//           forward pass reads, without changing the weight's buffer identity.
[[nodiscard]] OaMatrix Cast(const OaMatrix& InSrc, OaScalarType InDtype);
void CastInto(const OaMatrix& InSrc, OaMatrix& OutDst);

// RNG — Random Number Generation (GPU-native Philox PRNG)
// Seed the host-side seed generator used whenever a Philox call is made with
// InSeed == 0 (the default). Call once at startup for reproducible weight init /
// training runs; without it, the seed source is std::random_device (per-run
// non-deterministic). Thread-local: seed on the thread that builds the model.
void SetRngSeed(OaU64 InSeed);
// Low-level GPU RNG primitives (manual_context):
[[nodiscard]] OaMatrix PhiloxUniform(const OaMatrix& InA, OaF32 InLow = 0.0F, OaF32 InHigh = 1.0F, OaU64 InSeed = 0);
[[nodiscard]] OaMatrix PhiloxNormal(const OaMatrix& InA, OaF32 InMean = 0.0F, OaF32 InStddev = 1.0F, OaU64 InSeed = 0);
// Inverted dropout composed entirely from deferred GPU operations. InP must be [0,1).
[[nodiscard]] OaMatrix Dropout(const OaMatrix& InA, OaF32 InP, OaU64 InSeed = 0);
// Samples one class per last-axis row on GPU. Temperature <= 0 is greedy;
// InTopK <= 0 keeps the full vocabulary; InTopP is clamped to (0,1].
// Returns Int32 with shape [rows] (or [1] for a rank-1 input).
[[nodiscard]] OaMatrix SampleLogits(const OaMatrix& InLogits, OaF32 InTemperature = 1.0F,
	OaI32 InTopK = 0, OaF32 InTopP = 1.0F, OaU64 InSeed = 0);

// High-level initialization functions (Rand, RandN, RandXavier, etc.) use PhiloxUniform/PhiloxNormal.
// See FnMatrixAlloc.cpp for tensor initialization API.

// --- Transfer functions ---
/// Copy device matrix data to host memory
/// InBytes must be >= InSrc.ByteSize()
[[nodiscard]] OaStatus CopyToHost(const OaMatrix& InSrc, void* OutHost, OaU64 InBytes);

/// Extract first element as F32 scalar (requires single-element tensor)
[[nodiscard]] OaF32 Scalar(const OaMatrix& InSrc);

// --- Quantization functions ---
/// Quantize FP32 tensor to block-quantized format (Q4_K, Q8_0, etc.)
/// Input must be FP32. Output is quantized blocks stored contiguously.
/// Supported formats: Q4_0, Q4_1, Q5_0, Q5_1, Q8_0, Q8_1, Q2_K, Q3_K, Q4_K, Q5_K, Q6_K, Q8_K
[[nodiscard]] OaMatrix Quantize(const OaMatrix& InTensor, OaScalarType InTargetType);

/// Dequantize block-quantized tensor to FP32
/// Input must be a quantized type. Output is FP32.
[[nodiscard]] OaMatrix Dequantize(const OaMatrix& InQuantized);

// --- Core operations ---
// Generated overloads from OaFnAutogen schemas
// Regenerate via: python3 Tools/OaFnAutogen/oafnautogen.py
#include "../../../Private/Oa/Core/FnMatrix/FnMatrix.gen.h"

// In-place operations
void AddInPlace(OaMatrix& InSelf, const OaMatrix& InOther);
void ScaleInPlace(OaMatrix& InSelf, OaF32 InScalar);
void AddScalarInPlace(OaMatrix& InSelf, OaF32 InScalar);
void SubScalarInPlace(OaMatrix& InSelf, OaF32 InScalar);
void DivScalarInPlace(OaMatrix& InSelf, OaF32 InScalar);
void Fill(OaMatrix& InSelf, OaF32 InValue);

// Multi-tensor batch operations. Complete groups of four are fused; any
// remainder uses the equivalent direct operation without dropping inputs.
void MultiFill(OaSpan<OaMatrix> InTensors, OaF32 InValue);
void MultiAdd(OaSpan<OaMatrix> InDst, OaSpan<const OaMatrix> InSrc);

// Flush any deferred leaf-gradient accumulations batched by AccumulateGrad.
// Called automatically by OaGradientTape::Backward; manual use is optional.
void FlushDeferredAccum();

// Element-wise binary ops — broadcast-aware implementations. Add's public
// contract is schema-owned; the remaining signatures migrate incrementally.
[[nodiscard]] OaMatrix Sub(const OaMatrix& InA, const OaMatrix& InB);
[[nodiscard]] OaMatrix Mul(const OaMatrix& InA, const OaMatrix& InB);
[[nodiscard]] OaMatrix Div(const OaMatrix& InA, const OaMatrix& InB);

// Scale/Neg/Abs/Log/Sqrt/Pow/AddScalar/SubScalar/DivScalar: see FnMatrixElemwise.gen.h (autogen).

// MatMulNt's public declaration and semantic contract are generated from
// CoreFnMatrixBlas.toml. Its implementation remains hand-tuned and routes
// through the internal GEMM planner.

// Reduction operations
// Sum, Mean, and Max declarations and semantic contracts are generated from
// CoreFnMatrixReduce.toml. Their private reduction lowerings remain manual.
[[nodiscard]] OaI64 Argmax(const OaMatrix& InA, OaI32 InDim = -1);

// DescribeSum / DescribeMax buffer-level helpers retired. Use the
// OaMatrix-level Sum/Max APIs; implementation records through OaContext.

// Softmax and LogSoftmax declarations and semantic contracts are generated
// from CoreFnMatrixReduce.toml. Their private axis lowerings remain manual.

// Math operations
// Sqrt/Pow overloads: see FnMatrixElemwise.gen.h (autogen).

// Indexing operations
[[nodiscard]] OaMatrix Gather(const OaMatrix& InSelf, const OaMatrix& InIndices);
[[nodiscard]] OaMatrix GatherLastDim(const OaMatrix& InSelf, const OaMatrix& InIndices);
[[nodiscard]] OaMatrix GatherLastDimBwd(const OaMatrix& InGradOut,
	const OaMatrix& InIndices, OaI32 InInputWidth);
[[nodiscard]] OaMatrix Slice(
	const OaMatrix& InSelf, OaI32 InDim, OaI64 InStart, OaI64 InEnd
);
[[nodiscard]] OaMatrix Concat(OaSpan<OaMatrix> InInputs, OaI32 InDim = 0);
[[nodiscard]] OaVec<OaMatrix> Split(
	const OaMatrix& InSelf, OaSpan<OaI64> InSizes, OaI32 InDim = 0
);

// --- Shape & indexing helpers ---
[[nodiscard]] OaMatrix Reshape(const OaMatrix& InA, OaMatrixShape InShape);
[[nodiscard]] OaMatrix Transpose(const OaMatrix& InA, OaI32 InDim0, OaI32 InDim1);
[[nodiscard]] OaMatrix RepeatInterleave(const OaMatrix& InA, OaI32 InRepeats, OaI32 InDim);
[[nodiscard]] OaMatrix RepeatInterleaveBwd(const OaMatrix& InGradOut, OaMatrixShape InInputShape, OaI32 InRepeats, OaI32 InDim);
[[nodiscard]] OaMatrix CausalMask(const OaMatrix& InScores);
[[nodiscard]] OaMatrix CausalMaskBwd(const OaMatrix& InGradOut);
[[nodiscard]] OaTopKResult TopK(const OaMatrix& InA, OaI32 InK, OaI32 InDim = -1);
// Exact dense [T,E] membership mask from TopK's [T,K] Int32 indices. Ties are
// already resolved deterministically by TopK, so every row contains exactly K ones.
[[nodiscard]] OaMatrix TopKMask(const OaMatrix& InIndices, OaI32 InNumExperts);
[[nodiscard]] OaMatrix Equal(const OaMatrix& InA, OaF32 InValue);
[[nodiscard]] OaMatrix GreaterEqual(const OaMatrix& InA, OaF32 InValue);
// Returns a device UInt32 scalar containing the number of correct last-axis
// argmax predictions. Labels may be UInt8, UInt32, or Int32.
[[nodiscard]] OaMatrix CategoricalAccuracyCount(const OaMatrix& InLogits, const OaMatrix& InLabels);
// Last-axis argmax accuracy over rows whose same-dtype floating mask is non-zero.
// Returns one UInt32 correct-count scalar; callers divide by the valid-row count.
[[nodiscard]] OaMatrix MaskedCategoricalAccuracyCount(
	const OaMatrix& InLogits, const OaMatrix& InLabels, const OaMatrix& InMask);
// In-place auxiliary-loss-free MoE routing-bias update from a [T,E] 0/1
// selection mask. Entirely deferred; no route mask or bias readback.
void MoeRoutingBiasUpdate(const OaMatrix& InSelectionMask, OaMatrix& InOutBias,
	OaI32 InExpertsPerToken, OaF32 InGamma);
// Stable dropless expert-major MoE route plan. Input is TopK Int32 [T,K].
// PackedSlot stores source route slot t*K+k; Inverse maps it back to its row.
struct OaMoeExpertPlan {
	OaMatrix Counts;       // UInt32 [E]
	OaMatrix Offsets;      // UInt32 [E+1], exclusive; Offsets[E] == T*K
	OaMatrix PackedToken;  // UInt32 [T*K]
	OaMatrix PackedExpert; // UInt32 [T*K]
	OaMatrix PackedSlot;   // UInt32 [T*K]
	OaMatrix Inverse;      // UInt32 [T*K]
};
[[nodiscard]] OaMoeExpertPlan MoeExpertPlan(const OaMatrix& InExpertIndices,
	OaI32 InNumExperts);
// Normalize only exact selected routes: Probs[T,E], Int32 Indices[T,K] -> [T,K].
// Backward writes the exact dense [T,E] probability gradient without atomics.
[[nodiscard]] OaMatrix MoeRouteWeights(const OaMatrix& InProbs,
	const OaMatrix& InExpertIndices);
[[nodiscard]] OaMatrix MoeRouteWeightsBwd(const OaMatrix& InDOut,
	const OaMatrix& InProbs, const OaMatrix& InExpertIndices,
	const OaMatrix& InRouteWeights);
// Variable-M grouped GEMM over expert-major packed rows:
// X [R,K], W [E,N,K], Offsets [E+1] -> Y [R,N]. Empty experts are valid.
[[nodiscard]] OaMatrix GroupedGemmM(const OaMatrix& InX, const OaMatrix& InWeight,
	const OaMatrix& InOffsets);
[[nodiscard]] OaMatrix GroupedLinearM(const OaMatrix& InX, const OaMatrix& InWeight,
	const OaMatrix& InBias, const OaMatrix& InOffsets);
struct OaGroupedGemmMBwdResult {
	OaMatrix DInput;
	OaMatrix DWeight;
};
[[nodiscard]] OaGroupedGemmMBwdResult GroupedGemmMBwd(const OaMatrix& InDOut,
	const OaMatrix& InX, const OaMatrix& InWeight, const OaMatrix& InOffsets);
[[nodiscard]] OaMatrix GroupedLinearMBiasBwd(const OaMatrix& InDOut,
	const OaMatrix& InOffsets, OaI32 InNumExperts);
struct OaGroupedLinearMBwdResult {
	OaMatrix DInput;
	OaMatrix DWeight;
	OaMatrix DBias;
};
[[nodiscard]] OaGroupedLinearMBwdResult GroupedLinearMBwd(
	const OaMatrix& InDOut, const OaMatrix& InX, const OaMatrix& InWeight,
	const OaMatrix& InOffsets);
[[nodiscard]] OaMatrix MoeCombine(const OaMatrix& InPacked,
	const OaMatrix& InRouteGate, const OaMatrix& InInverse,
	const OaMatrix& InPackedSlot);
struct OaMoeCombineBwdResult {
	OaMatrix DPacked;
	OaMatrix DRouteGate;
};
[[nodiscard]] OaMoeCombineBwdResult MoeCombineBwd(const OaMatrix& InDOut,
	const OaMatrix& InPacked, const OaMatrix& InRouteGate,
	const OaMatrix& InInverse, const OaMatrix& InPackedSlot);
// MoE token packing. Forward is the same UInt32 row gather as Gather. The
// route-plan inverse makes backward a deterministic O(T*K*D) reduction with
// one writer per output scalar (no atomics and no output-clear pass).
[[nodiscard]] OaMatrix MoeGather(const OaMatrix& InSelf,
	const OaMatrix& InIndices, const OaMatrix& InInverse);
[[nodiscard]] OaMatrix MoeGatherBwd(const OaMatrix& InSource,
	const OaMatrix& InInverse, OaI32 InOutRows);
[[nodiscard]] OaMatrix ScatterAddRows(const OaMatrix& InSource,
	const OaMatrix& InIndices, OaI32 InOutRows);
// GPU-native row compaction keeps fixed capacity T so shape construction never
// reads a device mask on the host. Count and RowMap remain device tensors.
struct OaCompactRowsResult {
	OaMatrix Values;  // [T,D], first Count rows valid, remaining rows zero
	OaMatrix RowMap;  // UInt32 [T], first Count entries map packed -> source row
	OaMatrix Count;   // UInt32 [1]
	OaMatrix DispatchArgs; // UInt32 [3], GPU-authored ceil(Count*D/256),1,1
};
[[nodiscard]] OaCompactRowsResult CompactRows(const OaMatrix& InSelf, const OaMatrix& InMask);
[[nodiscard]] OaMatrix CompactRowsBwd(
	const OaMatrix& InGradOut, const OaMatrix& InRowMap, const OaMatrix& InCount,
	OaMatrixShape InInputShape);
[[nodiscard]] OaMatrix CompactRowsBwd(
	const OaMatrix& InGradOut, const OaMatrix& InRowMap, const OaMatrix& InCount,
	const OaMatrix& InDispatchArgs, OaMatrixShape InInputShape);
[[nodiscard]] OaMatrix ScatterRows(
	const OaMatrix& InSelf, const OaMatrix& InSource,
	const OaMatrix& InRowMap, const OaMatrix& InCount);
[[nodiscard]] OaMatrix ScatterRows(
	const OaMatrix& InSelf, const OaMatrix& InSource,
	const OaCompactRowsResult& InPlan);
[[nodiscard]] OaMatrix ScatterRowsBwdSource(
	const OaMatrix& InGradOut, const OaMatrix& InRowMap, const OaMatrix& InCount);
[[nodiscard]] OaMatrix ScatterRowsBwdSource(
	const OaMatrix& InGradOut, const OaMatrix& InRowMap, const OaMatrix& InCount,
	const OaMatrix& InDispatchArgs);

// --- Mamba-3 SISO selective scan (full outer-product state, rotary, trapezoidal) ---
// Implements the exact Mamba-3 SISO recurrence (arxiv 2603.15569). The kernel folds
// bias + interleaved rotary (cumulative angle) + trapezoidal discretization into a
// single fused sequential scan over the [headdim_v, d_state] state.
struct OaSsmConfig {
	OaU32 Batch;          // B
	OaU32 SeqLen;         // L
	OaU32 NHeads;         // H
	OaU32 HeadDim;        // P  (headdim_v)   <= 128
	OaU32 StateSize;      // N  (d_state / headdim_qk)   <= 128
	OaU32 NumRopeAngles;  // A  (<= N/2)      <= 64
	OaU32 HasZ;           // apply silu(z) gate
	OaU32 HasD;           // apply D skip
};

// Inputs (all FP32):
//   InC,InB     : [B, L, H, N]   (RMSNorm'd C=Q, B=K; broadcast G->H)
//   InX,InZ     : [B, L, H, P]   (x = V, z = gate)
//   InAdt,InDt  : [B, L, H]      (ADT = A*DT log-decay; DT = softplus dt)
//   InTrap      : [B, L, H]      (raw trap; sigmoid applied in kernel)
//   InAngle     : [B, L, A]      (raw rotary rates; cumulative angle = cumsum(angle*DT))
//   InCBias,InBBias : [H, N]     InD : [H]
// Returns y : [B, L, H, P]
[[nodiscard]] OaMatrix Mamba3Siso(
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
	const OaSsmConfig& InConfig
);

// Single-token recurrent step for autoregressive decode. State matrices (ssm/angle/k/v)
// are updated in place. SeqLen in the config must be 1. Returns y [B,1,H,P]. No autograd.
[[nodiscard]] OaMatrix Mamba3SisoStep(
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
	const OaSsmConfig& InConfig);

// Backward result for Mamba3Siso (grads w.r.t. the 11 forward inputs).
struct OaSsmBwdResult {
	OaMatrix DC;      // [B,L,H,N]
	OaMatrix DB;      // [B,L,H,N]
	OaMatrix DX;      // [B,L,H,P]
	OaMatrix DZ;      // [B,L,H,P]
	OaMatrix DAdt;    // [B,L,H]
	OaMatrix DDt;     // [B,L,H]
	OaMatrix DTrap;   // [B,L,H]  (w.r.t. raw trap)
	OaMatrix DAngle;  // [B,L,A]
	OaMatrix DCBias;  // [H,N]
	OaMatrix DBBias;  // [H,N]
	OaMatrix DD;      // [H]
};

[[nodiscard]] OaSsmBwdResult Mamba3SisoBwd(
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
	const OaSsmConfig& InConfig);

// --- Mamba3Adt: fused per-token A·dt term ---
// Collapses HeavyTailActivation + Neg + ClampMax + Mul (~9 elementwise dispatches)
// into one kernel:  ADT = min(-heavy_tail(dd_A), -AFloor) * dt.
//   InDdA, InDt : [B*S, H]   →   ADT : [B*S, H]
[[nodiscard]] OaMatrix Mamba3Adt(
	const OaMatrix& InDdA, const OaMatrix& InDt, OaF32 InAFloor);

// Backward result for Mamba3Adt (grads w.r.t. dd_A and dt).
struct OaMamba3AdtBwdResult {
	OaMatrix DDdA;   // [B*S, H]
	OaMatrix DDt;    // [B*S, H]
};

[[nodiscard]] OaMamba3AdtBwdResult Mamba3AdtBwd(
	const OaMatrix& InDOut, const OaMatrix& InDdA, const OaMatrix& InDt, OaF32 InAFloor);

// --- Mamba3Dt: fused per-token dt term ---
// Collapses Softplus + ClampMin + ClampMax (3 elementwise dispatches) into one kernel:
//   DT = clamp(softplus(x), dt_min, dt_max)
//   InX : [B*S, H]  (dd_dt + dt_bias)   →   DT : [B*S, H]
[[nodiscard]] OaMatrix Mamba3Dt(const OaMatrix& InX, OaF32 InDtMin, OaF32 InDtMax);

[[nodiscard]] OaMatrix Mamba3DtBwd(const OaMatrix& InDOut, const OaMatrix& InX, OaF32 InDtMin, OaF32 InDtMax);

// --- EmpyrealmAdt: 1:1 copy of Mamba3Adt, renamed for namespace separation ---
//   ADT = min(-heavy_tail(dd_A), -AFloor) * dt
//   InDdA, InDt : [B*S, H]   →   ADT : [B*S, H]
[[nodiscard]] OaMatrix EmpyrealmAdt(const OaMatrix& InDdA, const OaMatrix& InDt, OaF32 InAFloor);

[[nodiscard]] OaMamba3AdtBwdResult EmpyrealmAdtBwd(const OaMatrix& InDOut, const OaMatrix& InDdA, const OaMatrix& InDt, OaF32 InAFloor);

// --- EmpyrealmDt: 1:1 copy of Mamba3Dt, renamed for namespace separation ---
//   DT = clamp(softplus(x), dt_min, dt_max)
//   InX : [B*S, H]  (dd_dt + dt_bias)   →   DT : [B*S, H]
[[nodiscard]] OaMatrix EmpyrealmDt(const OaMatrix& InX, OaF32 InDtMin, OaF32 InDtMax);

[[nodiscard]] OaMatrix EmpyrealmDtBwd(const OaMatrix& InDOut, const OaMatrix& InX, OaF32 InDtMin, OaF32 InDtMax);

// --- Empyrealm SISO (verified Mamba-3 copy kernels) ---
// Dispatch EmpyrealmSiso*.slang kernels — exact copies of Mamba3Siso*.slang.
// Same signatures as Mamba3Siso*; separate namespace for future optimization work.
[[nodiscard]] OaMatrix EmpyrealmSiso(
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
	const OaSsmConfig& InConfig);

[[nodiscard]] OaMatrix EmpyrealmSisoStep(
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
	const OaSsmConfig& InConfig
);

[[nodiscard]] OaSsmBwdResult EmpyrealmSisoBwd(
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
	const OaSsmConfig& InConfig
);

// --- EmpyrealmDtAdt: fused dt + A·dt forward (2 dispatches → 1) ---
// Combines EmpyrealmDt + EmpyrealmAdt into a single kernel:
//   DT  = clamp(softplus(InDtRaw), InDtMin, InDtMax)
//   ADT = min(-heavy_tail(InDdA), -InAFloor) * DT
//   InDtRaw, InDdA : [B*S, H]   →   DT, ADT : [B*S, H]
struct OaEmpyrealmDtAdtResult {
	OaMatrix DT;    // [B*S, H]
	OaMatrix ADT;   // [B*S, H]
};

[[nodiscard]] OaEmpyrealmDtAdtResult EmpyrealmDtAdt(
	const OaMatrix& InDtRaw, const OaMatrix& InDdA,
	OaF32 InDtMin, OaF32 InDtMax, OaF32 InAFloor
);

// --- Mamba3Preprocess: fused in_proj split + RMSNorm + dt + A·dt (11+ dispatches → 1) ---
// Takes the projected [B*S, dInProj] tensor and produces all 8 preprocess outputs
// in a single kernel dispatch: z, x, Bh (rmsnorm'd), Ch (rmsnorm'd), DT, ADT, trap, angle.
// The caller provides the in_proj offsets and config. dt_bias is added in-kernel.
// When autograd is enabled, a single OaGradMamba3Preprocess node handles the backward.
struct OaMamba3PreprocessResult {
	OaMatrix X;         // [B*S, DInner]
	OaMatrix Z;         // [B*S, DInner]
	OaMatrix Bh;        // [B*S, H*DState] (rmsnorm'd, flat)
	OaMatrix Ch;        // [B*S, H*DState] (rmsnorm'd, flat)
	OaMatrix DT;        // [B*S, H]
	OaMatrix ADT;       // [B*S, H]
	OaMatrix Trap;      // [B*S, H]
	OaMatrix Angle;     // [B*S, NumRopeAngles]
};

struct OaMamba3PreprocessConfig {
	OaI32 DInner;
	OaI32 DState;
	OaI32 NHeads;
	OaI32 NumRopeAngles;
	OaI32 NGroups;
	OaI32 MimoRank;
	OaF32 Eps;
	OaF32 DtMin;
	OaF32 DtMax;
	OaF32 AFloor;
};

[[nodiscard]] OaMamba3PreprocessResult Mamba3Preprocess(
	const OaMatrix& InProjected, const OaMatrix& InDtBias,
	const OaMamba3PreprocessConfig& InConfig
);

struct OaMamba3PreprocessBwdResult {
	OaMatrix DProjected;  // [B*S, dInProj]
	OaMatrix DDtBias;     // [H]
};

[[nodiscard]] OaMamba3PreprocessBwdResult Mamba3PreprocessBwd(
	const OaMatrix& InProjected, const OaMatrix& InDtBias,
	const OaMatrix& InDZ, const OaMatrix& InDX,
	const OaMatrix& InDBh, const OaMatrix& InDCh,
	const OaMatrix& InDDT, const OaMatrix& InDADT,
	const OaMatrix& InDTrap, const OaMatrix& InDAngle,
	const OaMamba3PreprocessConfig& InConfig
);

// --- EmpyrealmPreprocess: 1:1 copy of Mamba3Preprocess, renamed for namespace separation ---
// Dispatch Ssm/Empyrealm/EmpyrealmPreprocess.slang; identical math today.
[[nodiscard]] OaMamba3PreprocessResult EmpyrealmPreprocess(
	const OaMatrix& InProjected, const OaMatrix& InDtBias,
	const OaMamba3PreprocessConfig& InConfig
);

[[nodiscard]] OaMamba3PreprocessBwdResult EmpyrealmPreprocessBwd(
	const OaMatrix& InProjected, const OaMatrix& InDtBias,
	const OaMatrix& InDZ, const OaMatrix& InDX,
	const OaMatrix& InDBh, const OaMatrix& InDCh,
	const OaMatrix& InDDT, const OaMatrix& InDADT,
	const OaMatrix& InDTrap, const OaMatrix& InDAngle,
	const OaMamba3PreprocessConfig& InConfig
);

// --- ResidualRmsNorm: fused residual + RmsNorm (2 dispatches → 1) ---
// Out = RmsNorm(A + B, Weight, Eps). Also returns the pre-norm residual (A + B).
struct OaResidualRmsNormResult {
	OaMatrix Out;        // RmsNorm(A + B, Weight, Eps)
	OaMatrix Residual;   // A + B (pre-norm sum)
};

[[nodiscard]] OaResidualRmsNormResult ResidualRmsNorm(
	const OaMatrix& InA, const OaMatrix& InB,
	const OaMatrix& InWeight, OaF32 InEps
);

} // namespace OaFnMatrix
