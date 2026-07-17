#pragma once

#include <cstring>

#include <Oa/Runtime/ComputeKernel.h>

// ============================================================================
// OA KERNEL REGISTRY — SINGLE SOURCE OF TRUTH
// ============================================================================
//
// This header defines the authoritative list of all compute kernels shipped
// with OA. It serves as the foundation for:
//
// 1. Runtime dispatch (name → shader lookup)
// 2. CMake build system (automatic shader compilation)
// 3. SPIR-V embedding (automatic inclusion in binaries)
// 4. Documentation generation (auto-generated registry docs)
// 5. Validation (collision detection, ID uniqueness)
//
// RULES:
// - Every kernel MUST have a unique (Prefix, Local) ID pair
// - Never reuse IDs once merged to main
// - Never change Prefix for shipped blocks
// - Keep Local ordinals sequential within each category
// - Update the registry schema and generated metadata together whenever the
//   contract, prefix allocation, or fixed-family status changes
// ============================================================================

namespace OaKernelRegistry {

// ============================================================================
// ML KERNELS (Prefix: 0x00081000)
// ============================================================================
// Core ML operations: element-wise, reductions, matrix ops, activations,
// normalization, loss functions, optimizers, embeddings, convolutions.
// Order matches CMake OA_ML_SHADER_REG for historical compatibility.

static constexpr OaComputeKernel MlKernels[] = {
	// Element-wise operations (1-8)
	{ "Add",                OA_COMPUTE_KERNEL_ID(OaComputeKernelPrefix::Ml, 1),  OaComputeKernelCategory::Ml, "oa" },
	{ "Sub",                OA_COMPUTE_KERNEL_ID(OaComputeKernelPrefix::Ml, 2),  OaComputeKernelCategory::Ml, "oa" },
	{ "Mul",                OA_COMPUTE_KERNEL_ID(OaComputeKernelPrefix::Ml, 3),  OaComputeKernelCategory::Ml, "oa" },
	{ "Div",                OA_COMPUTE_KERNEL_ID(OaComputeKernelPrefix::Ml, 4),  OaComputeKernelCategory::Ml, "oa" },
	{ "Neg",                OA_COMPUTE_KERNEL_ID(OaComputeKernelPrefix::Ml, 5),  OaComputeKernelCategory::Ml, "oa" },
	{ "Fill",               OA_COMPUTE_KERNEL_ID(OaComputeKernelPrefix::Ml, 6),  OaComputeKernelCategory::Ml, "oa" },
	{ "Scale",              OA_COMPUTE_KERNEL_ID(OaComputeKernelPrefix::Ml, 7),  OaComputeKernelCategory::Ml, "oa" },
	{ "Log",                OA_COMPUTE_KERNEL_ID(OaComputeKernelPrefix::Ml, 8),  OaComputeKernelCategory::Ml, "oa" },

	// Matrix operations (9) — 10 freed (naive Transpose retired for TransposeTiled)
	{ "Broadcast",          OA_COMPUTE_KERNEL_ID(OaComputeKernelPrefix::Ml, 9),  OaComputeKernelCategory::Ml, "oa" },
	{ "TransposeBatched",   OA_COMPUTE_KERNEL_ID(OaComputeKernelPrefix::Ml, 180), OaComputeKernelCategory::Ml, "oa" },

	// Reductions (11-15)
	{ "Sum",                OA_COMPUTE_KERNEL_ID(OaComputeKernelPrefix::Ml, 11), OaComputeKernelCategory::Ml, "oa" },
	{ "Max",                OA_COMPUTE_KERNEL_ID(OaComputeKernelPrefix::Ml, 12), OaComputeKernelCategory::Ml, "oa" },
	{ "Argmax",             OA_COMPUTE_KERNEL_ID(OaComputeKernelPrefix::Ml, 13), OaComputeKernelCategory::Ml, "oa" },
	{ "TopK",               OA_COMPUTE_KERNEL_ID(OaComputeKernelPrefix::Ml, 195), OaComputeKernelCategory::Ml, "oa" },
	{ "ReduceCols",         OA_COMPUTE_KERNEL_ID(OaComputeKernelPrefix::Ml, 14), OaComputeKernelCategory::Ml, "oa" },
	{ "ReduceMean",         OA_COMPUTE_KERNEL_ID(OaComputeKernelPrefix::Ml, 15), OaComputeKernelCategory::Ml, "oa" },

	// Element-wise unary (16)
	{ "Abs",                OA_COMPUTE_KERNEL_ID(OaComputeKernelPrefix::Ml, 16), OaComputeKernelCategory::Ml, "oa" },

	// Legacy ID range reserved (17-19): old Matmul* aliases removed

	// Linear layer operations (20-21)
	{ "BiasAdd",            OA_COMPUTE_KERNEL_ID(OaComputeKernelPrefix::Ml, 20), OaComputeKernelCategory::Ml, "oa" },
	{ "Embedding",          OA_COMPUTE_KERNEL_ID(OaComputeKernelPrefix::Ml, 21), OaComputeKernelCategory::Ml, "oa" },

	// Gather operations (22)
	{ "Gather",             OA_COMPUTE_KERNEL_ID(OaComputeKernelPrefix::Ml, 22), OaComputeKernelCategory::Ml, "oa" },

	// Activation functions (23-30)
	{ "Silu",               OA_COMPUTE_KERNEL_ID(OaComputeKernelPrefix::Ml, 23), OaComputeKernelCategory::Ml, "oa" },
	{ "SiluBwd",            OA_COMPUTE_KERNEL_ID(OaComputeKernelPrefix::Ml, 24), OaComputeKernelCategory::Ml, "oa" },
	{ "Gelu",               OA_COMPUTE_KERNEL_ID(OaComputeKernelPrefix::Ml, 25), OaComputeKernelCategory::Ml, "oa" },
	{ "GeluBwd",            OA_COMPUTE_KERNEL_ID(OaComputeKernelPrefix::Ml, 26), OaComputeKernelCategory::Ml, "oa" },
	{ "Relu",               OA_COMPUTE_KERNEL_ID(OaComputeKernelPrefix::Ml, 27), OaComputeKernelCategory::Ml, "oa" },
	{ "ReluBwd",            OA_COMPUTE_KERNEL_ID(OaComputeKernelPrefix::Ml, 28), OaComputeKernelCategory::Ml, "oa" },
	{ "Swiglu",             OA_COMPUTE_KERNEL_ID(OaComputeKernelPrefix::Ml, 29), OaComputeKernelCategory::Ml, "oa" },
	{ "SwigluBwd",          OA_COMPUTE_KERNEL_ID(OaComputeKernelPrefix::Ml, 30), OaComputeKernelCategory::Ml, "oa" },
	{ "SiluMul",            OA_COMPUTE_KERNEL_ID(OaComputeKernelPrefix::Ml, 136), OaComputeKernelCategory::Ml, "oa" },
	{ "Geglu",              OA_COMPUTE_KERNEL_ID(OaComputeKernelPrefix::Ml, 137), OaComputeKernelCategory::Ml, "oa" },
	{ "SiluMulBwd",         OA_COMPUTE_KERNEL_ID(OaComputeKernelPrefix::Ml, 196), OaComputeKernelCategory::Ml, "oa" },
	{ "GegluBwd",           OA_COMPUTE_KERNEL_ID(OaComputeKernelPrefix::Ml, 197), OaComputeKernelCategory::Ml, "oa" },
	// ID 183 retired: was GemmFusedGateUpCoopMatBf16 (dead code, Ffn uses separate MatMul+Swiglu).
	// Fused GRU recurrent cell pointwise gate combine (forward + backward).
	{ "GruCellPointwise",    OA_COMPUTE_KERNEL_ID(OaComputeKernelPrefix::Ml, 191), OaComputeKernelCategory::Ml, "oa" },
	{ "GruCellPointwiseBwd", OA_COMPUTE_KERNEL_ID(OaComputeKernelPrefix::Ml, 192), OaComputeKernelCategory::Ml, "oa" },
	// Fused vanilla-RNN recurrent cell pointwise combine (forward + backward).
	{ "RnnCellPointwise",    OA_COMPUTE_KERNEL_ID(OaComputeKernelPrefix::Ml, 193), OaComputeKernelCategory::Ml, "oa" },
	{ "RnnCellPointwiseBwd", OA_COMPUTE_KERNEL_ID(OaComputeKernelPrefix::Ml, 194), OaComputeKernelCategory::Ml, "oa" },

	// Normalization (31-42)
	{ "RmsNorm",            OA_COMPUTE_KERNEL_ID(OaComputeKernelPrefix::Ml, 31), OaComputeKernelCategory::Ml, "oa" },
	{ "RmsNormBwd",         OA_COMPUTE_KERNEL_ID(OaComputeKernelPrefix::Ml, 32), OaComputeKernelCategory::Ml, "oa" },
	{ "Softmax",            OA_COMPUTE_KERNEL_ID(OaComputeKernelPrefix::Ml, 33), OaComputeKernelCategory::Ml, "oa" },
	{ "SoftmaxBwd",         OA_COMPUTE_KERNEL_ID(OaComputeKernelPrefix::Ml, 34), OaComputeKernelCategory::Ml, "oa" },
	{ "SoftmaxScaledMasked",     OA_COMPUTE_KERNEL_ID(OaComputeKernelPrefix::Ml, 198), OaComputeKernelCategory::Ml, "oa" },
	{ "SoftmaxScaledMaskedBwd",  OA_COMPUTE_KERNEL_ID(OaComputeKernelPrefix::Ml, 199), OaComputeKernelCategory::Ml, "oa" },
	{ "SplitHeads",              OA_COMPUTE_KERNEL_ID(OaComputeKernelPrefix::Ml, 235), OaComputeKernelCategory::Ml, "oa" },
	{ "MergeHeads",              OA_COMPUTE_KERNEL_ID(OaComputeKernelPrefix::Ml, 236), OaComputeKernelCategory::Ml, "oa" },
	{ "FlashCausal",             OA_COMPUTE_KERNEL_ID(OaComputeKernelPrefix::Ml, 267), OaComputeKernelCategory::Ml, "oa" },
	{ "FlashCausalBwdQ",         OA_COMPUTE_KERNEL_ID(OaComputeKernelPrefix::Ml, 268), OaComputeKernelCategory::Ml, "oa" },
	{ "FlashCausalBwdKV",        OA_COMPUTE_KERNEL_ID(OaComputeKernelPrefix::Ml, 269), OaComputeKernelCategory::Ml, "oa" },
	{ "GruScan",                 OA_COMPUTE_KERNEL_ID(OaComputeKernelPrefix::Ml, 200), OaComputeKernelCategory::Ml, "oa" },
	{ "GruScanBwd",              OA_COMPUTE_KERNEL_ID(OaComputeKernelPrefix::Ml, 201), OaComputeKernelCategory::Ml, "oa" },
	{ "RnnScan",                 OA_COMPUTE_KERNEL_ID(OaComputeKernelPrefix::Ml, 202), OaComputeKernelCategory::Ml, "oa" },
	{ "RnnScanBwd",              OA_COMPUTE_KERNEL_ID(OaComputeKernelPrefix::Ml, 203), OaComputeKernelCategory::Ml, "oa" },
	// Fused Mamba-3 preprocess (split + RMSNorm + dt + adt in one kernel).
	{ "Mamba3Preprocess",        OA_COMPUTE_KERNEL_ID(OaComputeKernelPrefix::Ml, 204), OaComputeKernelCategory::Ml, "oa" },
	{ "Mamba3PreprocessBwd",     OA_COMPUTE_KERNEL_ID(OaComputeKernelPrefix::Ml, 205), OaComputeKernelCategory::Ml, "oa" },
	// Empyrealm-branded copy; identical SPIR-V today, reserved for future divergence.
	{ "EmpyrealmPreprocess",     OA_COMPUTE_KERNEL_ID(OaComputeKernelPrefix::Ml, 206), OaComputeKernelCategory::Ml, "oa" },
	{ "EmpyrealmPreprocessBwd",  OA_COMPUTE_KERNEL_ID(OaComputeKernelPrefix::Ml, 207), OaComputeKernelCategory::Ml, "oa" },
	{ "CrossEntropy",       OA_COMPUTE_KERNEL_ID(OaComputeKernelPrefix::Ml, 35), OaComputeKernelCategory::Ml, "oa" },
	{ "CrossEntropyBwd",    OA_COMPUTE_KERNEL_ID(OaComputeKernelPrefix::Ml, 36), OaComputeKernelCategory::Ml, "oa" },
	{ "CrossEntropyLossGradBwd", OA_COMPUTE_KERNEL_ID(OaComputeKernelPrefix::Ml, 121), OaComputeKernelCategory::Ml, "oa" },
	{ "LayerNorm",          OA_COMPUTE_KERNEL_ID(OaComputeKernelPrefix::Ml, 41), OaComputeKernelCategory::Ml, "oa" },
	{ "LayerNormBwd",       OA_COMPUTE_KERNEL_ID(OaComputeKernelPrefix::Ml, 42), OaComputeKernelCategory::Ml, "oa" },
	{ "RmsNormGated",       OA_COMPUTE_KERNEL_ID(OaComputeKernelPrefix::Ml, 122), OaComputeKernelCategory::Ml, "oa" },

	// Convolution operations (40) — 37/38/39 freed (37: scalar direct Conv1d retired for
	// the im2col+GEMM path Conv1dGemm; 38/39: dead Conv1d backward kernels purged)
	{ "Conv2d",             OA_COMPUTE_KERNEL_ID(OaComputeKernelPrefix::Ml, 40), OaComputeKernelCategory::Ml, "oa" },

	// Byte-level embeddings (43-44)
	{ "ByteEmbed",          OA_COMPUTE_KERNEL_ID(OaComputeKernelPrefix::Ml, 43), OaComputeKernelCategory::Ml, "oa" },
	{ "ByteEmbedBwd",       OA_COMPUTE_KERNEL_ID(OaComputeKernelPrefix::Ml, 44), OaComputeKernelCategory::Ml, "oa" },

	// Backward pass operations (45-47)
	{ "GatherBwd",          OA_COMPUTE_KERNEL_ID(OaComputeKernelPrefix::Ml, 45), OaComputeKernelCategory::Ml, "oa" },
	{ "LinearDataBwd",      OA_COMPUTE_KERNEL_ID(OaComputeKernelPrefix::Ml, 46), OaComputeKernelCategory::Ml, "oa" },
	{ "LinearWeightBwd",    OA_COMPUTE_KERNEL_ID(OaComputeKernelPrefix::Ml, 47), OaComputeKernelCategory::Ml, "oa" },
	{ "LinearWeightBiasBwd", OA_COMPUTE_KERNEL_ID(OaComputeKernelPrefix::Ml, 123), OaComputeKernelCategory::Ml, "oa" },
	{ "LinearDataReluBwd",  OA_COMPUTE_KERNEL_ID(OaComputeKernelPrefix::Ml, 125), OaComputeKernelCategory::Ml, "oa" },
	{ "LinearReluBwdData",  OA_COMPUTE_KERNEL_ID(OaComputeKernelPrefix::Ml, 187), OaComputeKernelCategory::Ml, "oa" },

	// Optimizers (48-51)
	{ "Sgd",                OA_COMPUTE_KERNEL_ID(OaComputeKernelPrefix::Ml, 48), OaComputeKernelCategory::Ml, "oa" },
	{ "Adam",               OA_COMPUTE_KERNEL_ID(OaComputeKernelPrefix::Ml, 49), OaComputeKernelCategory::Ml, "oa" },
	{ "Adamw",              OA_COMPUTE_KERNEL_ID(OaComputeKernelPrefix::Ml, 50), OaComputeKernelCategory::Ml, "oa" },
	{ "SgdMomentum",        OA_COMPUTE_KERNEL_ID(OaComputeKernelPrefix::Ml, 51), OaComputeKernelCategory::Ml, "oa" },
	{ "AdamwGraph",         OA_COMPUTE_KERNEL_ID(OaComputeKernelPrefix::Ml, 52), OaComputeKernelCategory::Ml, "oa" },
	{ "AdamwMany4",         OA_COMPUTE_KERNEL_ID(OaComputeKernelPrefix::Ml, 189), OaComputeKernelCategory::Ml, "oa" },
	{ "AdamwMany4Graph",    OA_COMPUTE_KERNEL_ID(OaComputeKernelPrefix::Ml, 250), OaComputeKernelCategory::Ml, "oa" },
	{ "ScatterAddRows",     OA_COMPUTE_KERNEL_ID(OaComputeKernelPrefix::Ml, 251), OaComputeKernelCategory::Ml, "oa" },
	{ "GroupedLinearMWeightBiasBwd", OA_COMPUTE_KERNEL_ID(OaComputeKernelPrefix::Ml, 252), OaComputeKernelCategory::Ml, "oa" },
	{ "LinearWeightBiasBwdTiled", OA_COMPUTE_KERNEL_ID(OaComputeKernelPrefix::Ml, 253), OaComputeKernelCategory::Ml, "oa" },
	{ "AdamwGraphAdvance",  OA_COMPUTE_KERNEL_ID(OaComputeKernelPrefix::Ml, 254), OaComputeKernelCategory::Ml, "oa" },
	{ "MoeGatherBwd",       OA_COMPUTE_KERNEL_ID(OaComputeKernelPrefix::Ml, 255), OaComputeKernelCategory::Ml, "oa" },
	{ "PhiloxUniformGraph", OA_COMPUTE_KERNEL_ID(OaComputeKernelPrefix::Ml, 256), OaComputeKernelCategory::Ml, "oa" },
	{ "PhiloxNormalGraph",  OA_COMPUTE_KERNEL_ID(OaComputeKernelPrefix::Ml, 257), OaComputeKernelCategory::Ml, "oa" },
	{ "PhiloxGraphAdvance", OA_COMPUTE_KERNEL_ID(OaComputeKernelPrefix::Ml, 258), OaComputeKernelCategory::Ml, "oa" },
	{ "CompactRowsBwdIndirect", OA_COMPUTE_KERNEL_ID(OaComputeKernelPrefix::Ml, 259), OaComputeKernelCategory::Ml, "oa" },
	{ "ScatterRowsIndirect", OA_COMPUTE_KERNEL_ID(OaComputeKernelPrefix::Ml, 260), OaComputeKernelCategory::Ml, "oa" },
	{ "ScatterRowsBwdIndirect", OA_COMPUTE_KERNEL_ID(OaComputeKernelPrefix::Ml, 261), OaComputeKernelCategory::Ml, "oa" },
	{ "PackedLinear",       OA_COMPUTE_KERNEL_ID(OaComputeKernelPrefix::Ml, 263), OaComputeKernelCategory::Ml, "oa" },
	{ "MuonNormalize",      OA_COMPUTE_KERNEL_ID(OaComputeKernelPrefix::Ml, 156), OaComputeKernelCategory::Ml, "oa" },
	{ "MuonApply",          OA_COMPUTE_KERNEL_ID(OaComputeKernelPrefix::Ml, 158), OaComputeKernelCategory::Ml, "oa" },
	{ "MuonVector",         OA_COMPUTE_KERNEL_ID(OaComputeKernelPrefix::Ml, 159), OaComputeKernelCategory::Ml, "oa" },
	{ "MuonNesterov",       OA_COMPUTE_KERNEL_ID(OaComputeKernelPrefix::Ml, 171), OaComputeKernelCategory::Ml, "oa" },
	{ "MatrixCopyRegion",    OA_COMPUTE_KERNEL_ID(OaComputeKernelPrefix::Ml, 190), OaComputeKernelCategory::Ml, "oa" },

	// CoopMat2 kernel IDs 126,127,132-135,144-154,184,212-231 retired.
	// NV-specific CoopMat2 kernels removed in Phase 4 cleanup.
	// KHR CoopMat1 (GemmCmSgBf16 / GemmCmWgBf16) replaces all CoopMat2 paths.

	// Distributed operations (53)
	{ "CollectiveReduce",   OA_COMPUTE_KERNEL_ID(OaComputeKernelPrefix::Ml, 53), OaComputeKernelCategory::Ml, "oa" },

	// Legacy ID range partially reused for canonical GEMM variants (54-59)
	// ID 56 retired: was GemmCoopMatBf16 (old CoopMat1), superseded by GemmCmSgBf16.
	// ID 188 retired: was GemmCoopMatSmallMBf16 (old CoopMat1 small-M).
	{ "TransposeTiled",           OA_COMPUTE_KERNEL_ID(OaComputeKernelPrefix::Ml, 57), OaComputeKernelCategory::Ml, "oa" },
	// ID 58 retired: was GemmSiluCoopMatBf16, superseded by GemmSiluCmSgBf16.
	// ID 59 retired: was GemmSwiGluCoopMatBf16 (dead code, Ffn uses separate MatMul+Swiglu).

	// RNG operations (60-61)
	{ "PhiloxNormal",       OA_COMPUTE_KERNEL_ID(OaComputeKernelPrefix::Ml, 60), OaComputeKernelCategory::Ml, "oa" },
	{ "PhiloxUniform",      OA_COMPUTE_KERNEL_ID(OaComputeKernelPrefix::Ml, 61), OaComputeKernelCategory::Ml, "oa" },

	// SSM scan operations (62-70)
	// Fused Mamba-3 per-token A·dt term (heavy_tail + clamp + mul) — forward + backward.
	{ "Mamba3Adt",          OA_COMPUTE_KERNEL_ID(OaComputeKernelPrefix::Ml, 62), OaComputeKernelCategory::Ml, "oa" },
	{ "Mamba3AdtBwd",       OA_COMPUTE_KERNEL_ID(OaComputeKernelPrefix::Ml, 63), OaComputeKernelCategory::Ml, "oa" },
	// Fused Mamba-3 per-token dt term (softplus + clamp) — forward + backward.
	{ "Mamba3Dt",           OA_COMPUTE_KERNEL_ID(OaComputeKernelPrefix::Ml, 64), OaComputeKernelCategory::Ml, "oa" },
	{ "Mamba3DtBwd",        OA_COMPUTE_KERNEL_ID(OaComputeKernelPrefix::Ml, 65), OaComputeKernelCategory::Ml, "oa" },
	// Empyrealm fused dt term (1:1 copy of Mamba3Dt, renamed for namespace separation).
	{ "EmpyrealmDt",        OA_COMPUTE_KERNEL_ID(OaComputeKernelPrefix::Ml, 66), OaComputeKernelCategory::Ml, "oa" },
	{ "EmpyrealmDtBwd",     OA_COMPUTE_KERNEL_ID(OaComputeKernelPrefix::Ml, 67), OaComputeKernelCategory::Ml, "oa" },
	// Empyrealm fused A·dt term (1:1 copy of Mamba3Adt, renamed for namespace separation).
	{ "EmpyrealmAdt",       OA_COMPUTE_KERNEL_ID(OaComputeKernelPrefix::Ml, 68), OaComputeKernelCategory::Ml, "oa" },
	{ "EmpyrealmAdtBwd",    OA_COMPUTE_KERNEL_ID(OaComputeKernelPrefix::Ml, 69), OaComputeKernelCategory::Ml, "oa" },
	// Empyrealm fused dt + A·dt (forward only; backward uses separate Dt/Adt kernels).
	{ "EmpyrealmDtAdt",     OA_COMPUTE_KERNEL_ID(OaComputeKernelPrefix::Ml, 70), OaComputeKernelCategory::Ml, "oa" },

	// Stochastic rounding (71)
	{ "Sr/SrRoundBf16",     OA_COMPUTE_KERNEL_ID(OaComputeKernelPrefix::Ml, 71), OaComputeKernelCategory::Ml, "oa" },

	// New GEMM kernels (Phase 4 consolidation) (72-79)
	{ "GemmNaive",                OA_COMPUTE_KERNEL_ID(OaComputeKernelPrefix::Ml, 72), OaComputeKernelCategory::Ml, "oa" },
	{ "GemmStrided",              OA_COMPUTE_KERNEL_ID(OaComputeKernelPrefix::Ml, 262), OaComputeKernelCategory::Ml, "oa" },
	#include <Oa/Core/OaTileKernelRegistry.gen.inc>
	// ID 76 retired: grouped-M=16 did not win a controlled product-shape sweep.
	// ID 77 retired: was GemmCoopMatStreamKBf16 (StreamK path removed).
	// ID 78 retired: was GemmReduceStreamK (StreamK path removed).
	{ "GemmCoopVec",              OA_COMPUTE_KERNEL_ID(OaComputeKernelPrefix::Ml, 79), OaComputeKernelCategory::Ml, "oa" },

	// Reserved for future gradient operations (80-95)
	{ "TanhBwd",                  OA_COMPUTE_KERNEL_ID(OaComputeKernelPrefix::Ml, 80), OaComputeKernelCategory::Ml, "oa" },
	{ "SigmoidBwd",               OA_COMPUTE_KERNEL_ID(OaComputeKernelPrefix::Ml, 81), OaComputeKernelCategory::Ml, "oa" },
	{ "LeakyReluBwd",             OA_COMPUTE_KERNEL_ID(OaComputeKernelPrefix::Ml, 82), OaComputeKernelCategory::Ml, "oa" },
	{ "Elu",                      OA_COMPUTE_KERNEL_ID(OaComputeKernelPrefix::Ml, 83), OaComputeKernelCategory::Ml, "oa" },
	{ "EluBwd",                   OA_COMPUTE_KERNEL_ID(OaComputeKernelPrefix::Ml, 84), OaComputeKernelCategory::Ml, "oa" },
	{ "Mish",                     OA_COMPUTE_KERNEL_ID(OaComputeKernelPrefix::Ml, 85), OaComputeKernelCategory::Ml, "oa" },
	{ "MishBwd",                  OA_COMPUTE_KERNEL_ID(OaComputeKernelPrefix::Ml, 86), OaComputeKernelCategory::Ml, "oa" },
	{ "Softplus",                OA_COMPUTE_KERNEL_ID(OaComputeKernelPrefix::Ml, 172), OaComputeKernelCategory::Ml, "oa" },
	{ "SoftplusBwd",             OA_COMPUTE_KERNEL_ID(OaComputeKernelPrefix::Ml, 179), OaComputeKernelCategory::Ml, "oa" },
	{ "MeanBwd",                  OA_COMPUTE_KERNEL_ID(OaComputeKernelPrefix::Ml, 87), OaComputeKernelCategory::Ml, "oa" },
	{ "MaxBwd",                   OA_COMPUTE_KERNEL_ID(OaComputeKernelPrefix::Ml, 88), OaComputeKernelCategory::Ml, "oa" },
	{ "AvgPool2d",                OA_COMPUTE_KERNEL_ID(OaComputeKernelPrefix::Ml, 89), OaComputeKernelCategory::Ml, "oa" },
	{ "AvgPool2dBwd",             OA_COMPUTE_KERNEL_ID(OaComputeKernelPrefix::Ml, 90), OaComputeKernelCategory::Ml, "oa" },
	{ "MaxPool2d",                OA_COMPUTE_KERNEL_ID(OaComputeKernelPrefix::Ml, 91), OaComputeKernelCategory::Ml, "oa" },
	{ "MaxPool2dBwd",             OA_COMPUTE_KERNEL_ID(OaComputeKernelPrefix::Ml, 92), OaComputeKernelCategory::Ml, "oa" },
	// IDs 93/94 freed (dead Conv1dInputBwd/Conv1dWeightBwd kernels purged)
	{ "Im2Col",                   OA_COMPUTE_KERNEL_ID(OaComputeKernelPrefix::Ml, 95), OaComputeKernelCategory::Ml, "oa" },
	{ "Col2Im",                   OA_COMPUTE_KERNEL_ID(OaComputeKernelPrefix::Ml, 96), OaComputeKernelCategory::Ml, "oa" },
	{ "Sqrt",                     OA_COMPUTE_KERNEL_ID(OaComputeKernelPrefix::Ml, 97), OaComputeKernelCategory::Ml, "oa" },
	{ "Pow",                      OA_COMPUTE_KERNEL_ID(OaComputeKernelPrefix::Ml, 98), OaComputeKernelCategory::Ml, "oa" },

	// Registry completion: every shipped compute-entrypoint .slang file (99-112)
	{ "Copy",                     OA_COMPUTE_KERNEL_ID(OaComputeKernelPrefix::Ml, 99),  OaComputeKernelCategory::Ml, "oa" },
	{ "CopyAtOffset",             OA_COMPUTE_KERNEL_ID(OaComputeKernelPrefix::Ml, 100), OaComputeKernelCategory::Ml, "oa" },
	{ "SumDim",                   OA_COMPUTE_KERNEL_ID(OaComputeKernelPrefix::Ml, 101), OaComputeKernelCategory::Ml, "oa" },
	{ "Tanh",                     OA_COMPUTE_KERNEL_ID(OaComputeKernelPrefix::Ml, 102), OaComputeKernelCategory::Ml, "oa" },
	{ "Sigmoid",                  OA_COMPUTE_KERNEL_ID(OaComputeKernelPrefix::Ml, 103), OaComputeKernelCategory::Ml, "oa" },
	{ "LeakyRelu",                OA_COMPUTE_KERNEL_ID(OaComputeKernelPrefix::Ml, 104), OaComputeKernelCategory::Ml, "oa" },
	{ "MulBwd",                   OA_COMPUTE_KERNEL_ID(OaComputeKernelPrefix::Ml, 105), OaComputeKernelCategory::Ml, "oa" },
	{ "AddBcast",                 OA_COMPUTE_KERNEL_ID(OaComputeKernelPrefix::Ml, 160), OaComputeKernelCategory::Ml, "oa" },
	{ "SubBcast",                 OA_COMPUTE_KERNEL_ID(OaComputeKernelPrefix::Ml, 161), OaComputeKernelCategory::Ml, "oa" },
	{ "MulBcast",                 OA_COMPUTE_KERNEL_ID(OaComputeKernelPrefix::Ml, 162), OaComputeKernelCategory::Ml, "oa" },
	{ "DivBcast",                 OA_COMPUTE_KERNEL_ID(OaComputeKernelPrefix::Ml, 163), OaComputeKernelCategory::Ml, "oa" },
	{ "AddScalar",                OA_COMPUTE_KERNEL_ID(OaComputeKernelPrefix::Ml, 164), OaComputeKernelCategory::Ml, "oa" },
	{ "SubScalar",                OA_COMPUTE_KERNEL_ID(OaComputeKernelPrefix::Ml, 165), OaComputeKernelCategory::Ml, "oa" },
	{ "DivScalar",                OA_COMPUTE_KERNEL_ID(OaComputeKernelPrefix::Ml, 166), OaComputeKernelCategory::Ml, "oa" },
	{ "Exp",                      OA_COMPUTE_KERNEL_ID(OaComputeKernelPrefix::Ml, 167), OaComputeKernelCategory::Ml, "oa" },
	{ "Reciprocal",               OA_COMPUTE_KERNEL_ID(OaComputeKernelPrefix::Ml, 168), OaComputeKernelCategory::Ml, "oa" },
	{ "ClampMax",                 OA_COMPUTE_KERNEL_ID(OaComputeKernelPrefix::Ml, 169), OaComputeKernelCategory::Ml, "oa" },
	{ "ClampMin",                 OA_COMPUTE_KERNEL_ID(OaComputeKernelPrefix::Ml, 170), OaComputeKernelCategory::Ml, "oa" },
	{ "Sin",                      OA_COMPUTE_KERNEL_ID(OaComputeKernelPrefix::Ml, 181), OaComputeKernelCategory::Ml, "oa" },
	{ "Cos",                      OA_COMPUTE_KERNEL_ID(OaComputeKernelPrefix::Ml, 182), OaComputeKernelCategory::Ml, "oa" },
	{ "LinearDataAccumBwd",       OA_COMPUTE_KERNEL_ID(OaComputeKernelPrefix::Ml, 106), OaComputeKernelCategory::Ml, "oa" },
	{ "ResidualRmsNorm",          OA_COMPUTE_KERNEL_ID(OaComputeKernelPrefix::Ml, 108), OaComputeKernelCategory::Ml, "oa" },
	{ "CastU8ToU32",              OA_COMPUTE_KERNEL_ID(OaComputeKernelPrefix::Ml, 112), OaComputeKernelCategory::Ml, "oa" },
	{ "CastBf16ToF32",            OA_COMPUTE_KERNEL_ID(OaComputeKernelPrefix::Ml, 208), OaComputeKernelCategory::Ml, "oa" },
	{ "CastF32ToBf16",            OA_COMPUTE_KERNEL_ID(OaComputeKernelPrefix::Ml, 209), OaComputeKernelCategory::Ml, "oa" },
	{ "Equal",                     OA_COMPUTE_KERNEL_ID(OaComputeKernelPrefix::Ml, 210), OaComputeKernelCategory::Ml, "oa" },
	{ "CausalMaskApply",           OA_COMPUTE_KERNEL_ID(OaComputeKernelPrefix::Ml, 211), OaComputeKernelCategory::Ml, "oa" },
	{ "CompactRows",               OA_COMPUTE_KERNEL_ID(OaComputeKernelPrefix::Ml, 212), OaComputeKernelCategory::Ml, "oa" },
	{ "CompactRowsBwd",            OA_COMPUTE_KERNEL_ID(OaComputeKernelPrefix::Ml, 213), OaComputeKernelCategory::Ml, "oa" },
	{ "ScatterRows",               OA_COMPUTE_KERNEL_ID(OaComputeKernelPrefix::Ml, 214), OaComputeKernelCategory::Ml, "oa" },
	{ "ScatterRowsBwd",            OA_COMPUTE_KERNEL_ID(OaComputeKernelPrefix::Ml, 215), OaComputeKernelCategory::Ml, "oa" },
	{ "GreaterEqual",              OA_COMPUTE_KERNEL_ID(OaComputeKernelPrefix::Ml, 216), OaComputeKernelCategory::Ml, "oa" },
	{ "CategoricalAccuracyCount",  OA_COMPUTE_KERNEL_ID(OaComputeKernelPrefix::Ml, 217), OaComputeKernelCategory::Ml, "oa" },
	{ "MoeRoutingBiasUpdate",      OA_COMPUTE_KERNEL_ID(OaComputeKernelPrefix::Ml, 218), OaComputeKernelCategory::Ml, "oa" },
	{ "CausalMaskBwd",             OA_COMPUTE_KERNEL_ID(OaComputeKernelPrefix::Ml, 219), OaComputeKernelCategory::Ml, "oa" },
	{ "SampleSortedLogits",        OA_COMPUTE_KERNEL_ID(OaComputeKernelPrefix::Ml, 220), OaComputeKernelCategory::Ml, "oa" },
	{ "SampleDenseLogits",         OA_COMPUTE_KERNEL_ID(OaComputeKernelPrefix::Ml, 221), OaComputeKernelCategory::Ml, "oa" },
	{ "MoeExpertPlan",             OA_COMPUTE_KERNEL_ID(OaComputeKernelPrefix::Ml, 222), OaComputeKernelCategory::Ml, "oa" },
	{ "MoeRouteWeights",           OA_COMPUTE_KERNEL_ID(OaComputeKernelPrefix::Ml, 237), OaComputeKernelCategory::Ml, "oa" },
	{ "MoeRouteWeightsBwd",        OA_COMPUTE_KERNEL_ID(OaComputeKernelPrefix::Ml, 238), OaComputeKernelCategory::Ml, "oa" },
	{ "Mamba3SisoBwdMobile",       OA_COMPUTE_KERNEL_ID(OaComputeKernelPrefix::Ml, 239), OaComputeKernelCategory::Ml, "oa" },
	{ "GroupedGemmM",              OA_COMPUTE_KERNEL_ID(OaComputeKernelPrefix::Ml, 223), OaComputeKernelCategory::Ml, "oa" },
	{ "GroupedGemmMDataBwd",       OA_COMPUTE_KERNEL_ID(OaComputeKernelPrefix::Ml, 224), OaComputeKernelCategory::Ml, "oa" },
	{ "GroupedGemmMWeightBwd",     OA_COMPUTE_KERNEL_ID(OaComputeKernelPrefix::Ml, 225), OaComputeKernelCategory::Ml, "oa" },
	{ "GatherLastDim",             OA_COMPUTE_KERNEL_ID(OaComputeKernelPrefix::Ml, 226), OaComputeKernelCategory::Ml, "oa" },
	{ "GatherLastDimBwd",          OA_COMPUTE_KERNEL_ID(OaComputeKernelPrefix::Ml, 227), OaComputeKernelCategory::Ml, "oa" },
	{ "GroupedLinearM",            OA_COMPUTE_KERNEL_ID(OaComputeKernelPrefix::Ml, 228), OaComputeKernelCategory::Ml, "oa" },
	{ "GroupedLinearMBiasBwd",     OA_COMPUTE_KERNEL_ID(OaComputeKernelPrefix::Ml, 229), OaComputeKernelCategory::Ml, "oa" },
	{ "MoeCombine",                OA_COMPUTE_KERNEL_ID(OaComputeKernelPrefix::Ml, 230), OaComputeKernelCategory::Ml, "oa" },
	{ "MoeCombineBwd",             OA_COMPUTE_KERNEL_ID(OaComputeKernelPrefix::Ml, 231), OaComputeKernelCategory::Ml, "oa" },
	{ "MaskedCrossEntropy",        OA_COMPUTE_KERNEL_ID(OaComputeKernelPrefix::Ml, 232), OaComputeKernelCategory::Ml, "oa" },
	{ "MaskedCrossEntropyBwd",     OA_COMPUTE_KERNEL_ID(OaComputeKernelPrefix::Ml, 233), OaComputeKernelCategory::Ml, "oa" },
	{ "MaskedCategoricalAccuracyCount", OA_COMPUTE_KERNEL_ID(OaComputeKernelPrefix::Ml, 234), OaComputeKernelCategory::Ml, "oa" },

	// INT8 quantization operations (113-116)
	{ "Quantize/QuantizeInt8",    OA_COMPUTE_KERNEL_ID(OaComputeKernelPrefix::Ml, 113), OaComputeKernelCategory::Ml, "oa" },
	{ "Quantize/DequantizeInt8",  OA_COMPUTE_KERNEL_ID(OaComputeKernelPrefix::Ml, 114), OaComputeKernelCategory::Ml, "oa" },
	{ "Quantize/ComputeScaleInt8", OA_COMPUTE_KERNEL_ID(OaComputeKernelPrefix::Ml, 115), OaComputeKernelCategory::Ml, "oa" },
	{ "Quantize/GemmInt8",        OA_COMPUTE_KERNEL_ID(OaComputeKernelPrefix::Ml, 116), OaComputeKernelCategory::Ml, "oa" },

	// Q4 quantization operations (llama.cpp compatibility) (138-140)
	{ "Quantize/QuantizeQ4_0",    OA_COMPUTE_KERNEL_ID(OaComputeKernelPrefix::Ml, 138), OaComputeKernelCategory::Ml, "oa" },
	{ "Quantize/DequantizeQ4_0",  OA_COMPUTE_KERNEL_ID(OaComputeKernelPrefix::Ml, 139), OaComputeKernelCategory::Ml, "oa" },
	{ "Quantize/ComputeScaleQ4_0", OA_COMPUTE_KERNEL_ID(OaComputeKernelPrefix::Ml, 140), OaComputeKernelCategory::Ml, "oa" },

	// Q8_0 quantization operations (llama.cpp compatibility) (175-176)
	{ "Quantize/QuantizeQ8_0",    OA_COMPUTE_KERNEL_ID(OaComputeKernelPrefix::Ml, 175), OaComputeKernelCategory::Ml, "oa" },
	{ "Quantize/DequantizeQ8_0",  OA_COMPUTE_KERNEL_ID(OaComputeKernelPrefix::Ml, 176), OaComputeKernelCategory::Ml, "oa" },

	// Q4_K quantization operations (llama.cpp compatibility) (177-178)
	{ "Quantize/DequantizeQ4_K",  OA_COMPUTE_KERNEL_ID(OaComputeKernelPrefix::Ml, 177), OaComputeKernelCategory::Ml, "oa" },
	{ "Quantize/QuantizeQ4_K",    OA_COMPUTE_KERNEL_ID(OaComputeKernelPrefix::Ml, 178), OaComputeKernelCategory::Ml, "oa" },

	// BF16 CoopMat fused GEMM operations (117-120, 124). IDs 117, 118, 124
	// retired (bias/relu/gelu forward variants superseded by GemmCmSgBf16 family).
	// IDs 119, 120 retired: were GemmBiasReluBwdCoopMatBf16, GemmBiasGeluBwdCoopMatBf16 (dead code).
	{ "MultiMatrixFill",          OA_COMPUTE_KERNEL_ID(OaComputeKernelPrefix::Ml, 173), OaComputeKernelCategory::Ml, "oa" },
	{ "MultiMatrixAdd",          OA_COMPUTE_KERNEL_ID(OaComputeKernelPrefix::Ml, 174), OaComputeKernelCategory::Ml, "oa" },
	{ "RlGae",                   OA_COMPUTE_KERNEL_ID(OaComputeKernelPrefix::Ml, 264), OaComputeKernelCategory::Ml, "oa" },
	{ "RlPpoClip",               OA_COMPUTE_KERNEL_ID(OaComputeKernelPrefix::Ml, 265), OaComputeKernelCategory::Ml, "oa" },
	{ "RlPpoClipBwd",            OA_COMPUTE_KERNEL_ID(OaComputeKernelPrefix::Ml, 266), OaComputeKernelCategory::Ml, "oa" },
	{ "RlRolloutAppend",         OA_COMPUTE_KERNEL_ID(OaComputeKernelPrefix::Ml, 272), OaComputeKernelCategory::Ml, "oa" },
	{ "RlRolloutReset",          OA_COMPUTE_KERNEL_ID(OaComputeKernelPrefix::Ml, 273), OaComputeKernelCategory::Ml, "oa" },
	{ "RlCartPoleReset",          OA_COMPUTE_KERNEL_ID(OaComputeKernelPrefix::Ml, 274), OaComputeKernelCategory::Ml, "oa" },
	{ "RlCartPoleStep",           OA_COMPUTE_KERNEL_ID(OaComputeKernelPrefix::Ml, 275), OaComputeKernelCategory::Ml, "oa" },
	{ "RlReplayAppend",           OA_COMPUTE_KERNEL_ID(OaComputeKernelPrefix::Ml, 276), OaComputeKernelCategory::Ml, "oa" },
	{ "RlReplaySample",           OA_COMPUTE_KERNEL_ID(OaComputeKernelPrefix::Ml, 277), OaComputeKernelCategory::Ml, "oa" },
	{ "RlDqnTarget",              OA_COMPUTE_KERNEL_ID(OaComputeKernelPrefix::Ml, 278), OaComputeKernelCategory::Ml, "oa" },
	{ "RlSacTarget",              OA_COMPUTE_KERNEL_ID(OaComputeKernelPrefix::Ml, 279), OaComputeKernelCategory::Ml, "oa" },
	{ "LogSoftmax",              OA_COMPUTE_KERNEL_ID(OaComputeKernelPrefix::Ml, 270), OaComputeKernelCategory::Ml, "oa" },
	{ "LogSoftmaxBwd",           OA_COMPUTE_KERNEL_ID(OaComputeKernelPrefix::Ml, 271), OaComputeKernelCategory::Ml, "oa" },
};

// ============================================================================
// VISION KERNELS (Prefix: 0x00082000)
// ============================================================================
// Image/video preprocessing and CV kernels. Runtime shader names are flattened
// to basename to match SPIR-V embedding and OaVkImageDispatch/OaVkDispatch.

static constexpr OaComputeKernel VisionKernels[] = {
	// Conversion (1-6)
	{ "CvtNv12ToBf16",        OA_COMPUTE_KERNEL_ID(OaComputeKernelPrefix::Vision, 1),  OaComputeKernelCategory::Vision, "oa" },
	{ "CvtNv12ToRgb",         OA_COMPUTE_KERNEL_ID(OaComputeKernelPrefix::Vision, 2),  OaComputeKernelCategory::Vision, "oa" },
	{ "CvtNv12YcbcrToBf16",   OA_COMPUTE_KERNEL_ID(OaComputeKernelPrefix::Vision, 3),  OaComputeKernelCategory::Vision, "oa" },
	{ "CvtNv12YcbcrToRgba",   OA_COMPUTE_KERNEL_ID(OaComputeKernelPrefix::Vision, 4),  OaComputeKernelCategory::Vision, "oa" },
	{ "CvtRgbToGray",         OA_COMPUTE_KERNEL_ID(OaComputeKernelPrefix::Vision, 5),  OaComputeKernelCategory::Vision, "oa" },
	{ "NormalizeImage",       OA_COMPUTE_KERNEL_ID(OaComputeKernelPrefix::Vision, 6),  OaComputeKernelCategory::Vision, "oa" },

	// Geometric (7-12)
	{ "ResizeBilinear",       OA_COMPUTE_KERNEL_ID(OaComputeKernelPrefix::Vision, 7),  OaComputeKernelCategory::Vision, "oa" },
	{ "ResizeNearest",        OA_COMPUTE_KERNEL_ID(OaComputeKernelPrefix::Vision, 8),  OaComputeKernelCategory::Vision, "oa" },
	{ "CropNchw",             OA_COMPUTE_KERNEL_ID(OaComputeKernelPrefix::Vision, 9),  OaComputeKernelCategory::Vision, "oa" },
	{ "FlipNchw",             OA_COMPUTE_KERNEL_ID(OaComputeKernelPrefix::Vision, 10), OaComputeKernelCategory::Vision, "oa" },
	{ "RotateNchw",           OA_COMPUTE_KERNEL_ID(OaComputeKernelPrefix::Vision, 11), OaComputeKernelCategory::Vision, "oa" },
	// ID 12 retired with the one-off constant-only padding shader.

	// Filter/feature (13-15). IDs 13-14 are retired: the original one-off
	// Gaussian/Sobel shaders were replaced by the generic filter primitives.
	{ "Canny",                OA_COMPUTE_KERNEL_ID(OaComputeKernelPrefix::Vision, 15), OaComputeKernelCategory::Vision, "oa" },

	// Vision NN (16-17)
	{ "BatchNorm2dStats",     OA_COMPUTE_KERNEL_ID(OaComputeKernelPrefix::Vision, 16), OaComputeKernelCategory::Vision, "oa" },
	{ "BatchNorm2dNorm",      OA_COMPUTE_KERNEL_ID(OaComputeKernelPrefix::Vision, 17), OaComputeKernelCategory::Vision, "oa" },

	// Conversion (encoder direction — 18+). RGBA → NV12 multi-plane storage
	// image, used by OaVideoEncoder to prepare its NV12 input picture.
	{ "CvtRgbaToNv12",        OA_COMPUTE_KERNEL_ID(OaComputeKernelPrefix::Vision, 18), OaComputeKernelCategory::Vision, "oa" },
	{ "CvtRgbaImageToNv12",   OA_COMPUTE_KERNEL_ID(OaComputeKernelPrefix::Vision, 19), OaComputeKernelCategory::Vision, "oa" },

	// Generic image-processing primitives (20-30). Public semantic operations
	// compose these reusable kernels rather than growing one shader per API name.
	{ "ImageConvolve2d",      OA_COMPUTE_KERNEL_ID(OaComputeKernelPrefix::Vision, 20), OaComputeKernelCategory::Vision, "oa" },
	{ "ImageSeparableConvolve", OA_COMPUTE_KERNEL_ID(OaComputeKernelPrefix::Vision, 21), OaComputeKernelCategory::Vision, "oa" },
	{ "ImageMorphology",      OA_COMPUTE_KERNEL_ID(OaComputeKernelPrefix::Vision, 22), OaComputeKernelCategory::Vision, "oa" },
	{ "ImageNeighborhood",    OA_COMPUTE_KERNEL_ID(OaComputeKernelPrefix::Vision, 23), OaComputeKernelCategory::Vision, "oa" },
	{ "ImagePointwise",       OA_COMPUTE_KERNEL_ID(OaComputeKernelPrefix::Vision, 24), OaComputeKernelCategory::Vision, "oa" },
	{ "ImageChannelTransform", OA_COMPUTE_KERNEL_ID(OaComputeKernelPrefix::Vision, 25), OaComputeKernelCategory::Vision, "oa" },
	{ "ImageComposite",       OA_COMPUTE_KERNEL_ID(OaComputeKernelPrefix::Vision, 26), OaComputeKernelCategory::Vision, "oa" },
	{ "ImageColorMatrix",     OA_COMPUTE_KERNEL_ID(OaComputeKernelPrefix::Vision, 27), OaComputeKernelCategory::Vision, "oa" },
	{ "ImageSaltPepper",      OA_COMPUTE_KERNEL_ID(OaComputeKernelPrefix::Vision, 28), OaComputeKernelCategory::Vision, "oa" },
	{ "ImagePad",             OA_COMPUTE_KERNEL_ID(OaComputeKernelPrefix::Vision, 29), OaComputeKernelCategory::Vision, "oa" },
	{ "ImageWarp",            OA_COMPUTE_KERNEL_ID(OaComputeKernelPrefix::Vision, 30), OaComputeKernelCategory::Vision, "oa" },

	// Detection postprocess and evaluation (31-34).
	{ "DetectionBoxIou",      OA_COMPUTE_KERNEL_ID(OaComputeKernelPrefix::Vision, 31), OaComputeKernelCategory::Vision, "oa" },
	{ "DetectionNms",         OA_COMPUTE_KERNEL_ID(OaComputeKernelPrefix::Vision, 32), OaComputeKernelCategory::Vision, "oa" },
	{ "DetectionConfusion",   OA_COMPUTE_KERNEL_ID(OaComputeKernelPrefix::Vision, 33), OaComputeKernelCategory::Vision, "oa" },
	{ "DetectionBinaryCounts", OA_COMPUTE_KERNEL_ID(OaComputeKernelPrefix::Vision, 34), OaComputeKernelCategory::Vision, "oa" },
	{ "DetectionMetricCurves", OA_COMPUTE_KERNEL_ID(OaComputeKernelPrefix::Vision, 35), OaComputeKernelCategory::Vision, "oa" },
	{ "DetectionAveragePrecision", OA_COMPUTE_KERNEL_ID(OaComputeKernelPrefix::Vision, 36), OaComputeKernelCategory::Vision, "oa" },
	{ "DetectionMeanAveragePrecision", OA_COMPUTE_KERNEL_ID(OaComputeKernelPrefix::Vision, 37), OaComputeKernelCategory::Vision, "oa" },
	{ "SegmentationMetrics", OA_COMPUTE_KERNEL_ID(OaComputeKernelPrefix::Vision, 38), OaComputeKernelCategory::Vision, "oa" },
	{ "ImageSegmentationOverlay", OA_COMPUTE_KERNEL_ID(OaComputeKernelPrefix::Vision, 39), OaComputeKernelCategory::Vision, "oa" },
};

// ============================================================================
// UI KERNELS (Prefix: 0x00083000)
// ============================================================================
// Presentation/compose kernels.

static constexpr OaComputeKernel UiKernels[] = {
	{ "BlitRgba",             OA_COMPUTE_KERNEL_ID(OaComputeKernelPrefix::Ui, 1), OaComputeKernelCategory::Ui, "oa" },
	{ "BlitPlanar",           OA_COMPUTE_KERNEL_ID(OaComputeKernelPrefix::Ui, 2), OaComputeKernelCategory::Ui, "oa" },
	{ "BlitImageRgba",        OA_COMPUTE_KERNEL_ID(OaComputeKernelPrefix::Ui, 3), OaComputeKernelCategory::Ui, "oa" },
	{ "DrawRectOutline",      OA_COMPUTE_KERNEL_ID(OaComputeKernelPrefix::Ui, 4), OaComputeKernelCategory::Ui, "oa" },
	{ "DrawRectOutlines",      OA_COMPUTE_KERNEL_ID(OaComputeKernelPrefix::Ui, 5), OaComputeKernelCategory::Ui, "oa" },
	{ "DrawGlyphs",            OA_COMPUTE_KERNEL_ID(OaComputeKernelPrefix::Ui, 6), OaComputeKernelCategory::Ui, "oa" },
	{ "DrawRect",              OA_COMPUTE_KERNEL_ID(OaComputeKernelPrefix::Ui, 7), OaComputeKernelCategory::Ui, "oa" },
	{ "DrawWaveform",          OA_COMPUTE_KERNEL_ID(OaComputeKernelPrefix::Ui, 8), OaComputeKernelCategory::Ui, "oa" },
};

// ============================================================================
// RENDER KERNELS (Prefix: 0x00085000)
// ============================================================================
// Graphics shaders: vertex, fragment, geometry, etc.
// Not compute — these feed into VkGraphicsPipelineCreateInfo, not dispatch.

static constexpr OaComputeKernel RenderKernels[] = {
	// Unlit textured (image/video viewer)
	{ "UnlitTextured.vert",   OA_COMPUTE_KERNEL_ID(OaComputeKernelPrefix::Render, 1), OaComputeKernelCategory::Render, "oa" },
	{ "UnlitTextured.frag",   OA_COMPUTE_KERNEL_ID(OaComputeKernelPrefix::Render, 2), OaComputeKernelCategory::Render, "oa" },
};

// ============================================================================
// AUDIO KERNELS (Prefix: 0x00084000)
// ============================================================================
// Audio DSP operations: STFT, mel filterbank, spectral features.
// FP32-only (no quantization needed for DSP precision requirements).

static constexpr OaComputeKernel AudioKernels[] = {
	// Spectral analysis (1-2)
	{ "Stft",               OA_COMPUTE_KERNEL_ID(OaComputeKernelPrefix::Audio, 1), OaComputeKernelCategory::Audio, "oa" },
	{ "MelFilterbank",      OA_COMPUTE_KERNEL_ID(OaComputeKernelPrefix::Audio, 2), OaComputeKernelCategory::Audio, "oa" },
	
	// Signal processing (3-6)
	{ "AudioFade",          OA_COMPUTE_KERNEL_ID(OaComputeKernelPrefix::Audio, 3), OaComputeKernelCategory::Audio, "oa" },
	{ "AudioResample",      OA_COMPUTE_KERNEL_ID(OaComputeKernelPrefix::Audio, 4), OaComputeKernelCategory::Audio, "oa" },
	{ "AudioMix",           OA_COMPUTE_KERNEL_ID(OaComputeKernelPrefix::Audio, 6), OaComputeKernelCategory::Audio, "oa" },
	{ "AudioWaveformEnvelope", OA_COMPUTE_KERNEL_ID(OaComputeKernelPrefix::Audio, 7), OaComputeKernelCategory::Audio, "oa" },
};

// ============================================================================
// CRYPTO KERNELS (Prefix: 0x00000200)
// ============================================================================
// Post-quantum cryptography: Keccak, SHAKE, NTT, polynomial operations,
// ML-DSA (Dilithium) signature scheme.

static constexpr OaComputeKernel CryptoKernels[] = {
	// Keccak permutation (1)
	{ "KeccakF1600",        OA_COMPUTE_KERNEL_ID(OaComputeKernelPrefix::Crypto, 1),  OaComputeKernelCategory::Crypto, "oa" },

	// SHAKE XOF (2-3)
	{ "Shake/Shake128",     OA_COMPUTE_KERNEL_ID(OaComputeKernelPrefix::Crypto, 2),  OaComputeKernelCategory::Crypto, "oa" },
	{ "Shake/Shake256",     OA_COMPUTE_KERNEL_ID(OaComputeKernelPrefix::Crypto, 3),  OaComputeKernelCategory::Crypto, "oa" },

	// Number Theoretic Transform (4-6)
	{ "Ntt/NttForward",     OA_COMPUTE_KERNEL_ID(OaComputeKernelPrefix::Crypto, 4),  OaComputeKernelCategory::Crypto, "oa" },
	{ "Ntt/NttInverse",     OA_COMPUTE_KERNEL_ID(OaComputeKernelPrefix::Crypto, 5),  OaComputeKernelCategory::Crypto, "oa" },
	{ "Ntt/NttPointwise",   OA_COMPUTE_KERNEL_ID(OaComputeKernelPrefix::Crypto, 6),  OaComputeKernelCategory::Crypto, "oa" },

	// Polynomial operations (7-10)
	{ "Poly/PolyArith",     OA_COMPUTE_KERNEL_ID(OaComputeKernelPrefix::Crypto, 7),  OaComputeKernelCategory::Crypto, "oa" },
	{ "Poly/PolyRound",     OA_COMPUTE_KERNEL_ID(OaComputeKernelPrefix::Crypto, 8),  OaComputeKernelCategory::Crypto, "oa" },
	{ "Poly/PolyHint",      OA_COMPUTE_KERNEL_ID(OaComputeKernelPrefix::Crypto, 9),  OaComputeKernelCategory::Crypto, "oa" },
	{ "Poly/PolySample",    OA_COMPUTE_KERNEL_ID(OaComputeKernelPrefix::Crypto, 10), OaComputeKernelCategory::Crypto, "oa" },

	// ML-DSA (Dilithium) operations (11-14)
	{ "Mldsa/MldsaVerify",      OA_COMPUTE_KERNEL_ID(OaComputeKernelPrefix::Crypto, 11), OaComputeKernelCategory::Crypto, "oa" },
	{ "Mldsa/MldsaSign",        OA_COMPUTE_KERNEL_ID(OaComputeKernelPrefix::Crypto, 12), OaComputeKernelCategory::Crypto, "oa" },
	{ "Mldsa/MldsaKeygen",      OA_COMPUTE_KERNEL_ID(OaComputeKernelPrefix::Crypto, 13), OaComputeKernelCategory::Crypto, "oa" },
	{ "Mldsa/MldsaVerifyBatch", OA_COMPUTE_KERNEL_ID(OaComputeKernelPrefix::Crypto, 14), OaComputeKernelCategory::Crypto, "oa" },

	// KMAC (15)
	{ "Kmac/Kmac256",           OA_COMPUTE_KERNEL_ID(OaComputeKernelPrefix::Crypto, 15), OaComputeKernelCategory::Crypto, "oa" },

	// Merkle tree operations (16-17)
	{ "Merkle/MerkleReduce",    OA_COMPUTE_KERNEL_ID(OaComputeKernelPrefix::Crypto, 16), OaComputeKernelCategory::Crypto, "oa" },
	{ "Merkle/MerkleVerify",    OA_COMPUTE_KERNEL_ID(OaComputeKernelPrefix::Crypto, 17), OaComputeKernelCategory::Crypto, "oa" },
};

// ============================================================================
// REGISTRY ACCESS
// ============================================================================

// Get all ML kernels
inline OaSpan<const OaComputeKernel> GetMlKernels() {
	return OaSpan<const OaComputeKernel>(MlKernels, sizeof(MlKernels) / sizeof(MlKernels[0]));
}

// Get all crypto kernels
inline OaSpan<const OaComputeKernel> GetCryptoKernels() {
	return OaSpan<const OaComputeKernel>(CryptoKernels, sizeof(CryptoKernels) / sizeof(CryptoKernels[0]));
}

// Get all Vision kernels
inline OaSpan<const OaComputeKernel> GetVisionKernels() {
	return OaSpan<const OaComputeKernel>(VisionKernels, sizeof(VisionKernels) / sizeof(VisionKernels[0]));
}

// Get all UI kernels
inline OaSpan<const OaComputeKernel> GetUiKernels() {
	return OaSpan<const OaComputeKernel>(UiKernels, sizeof(UiKernels) / sizeof(UiKernels[0]));
}

// Get all Audio kernels
inline OaSpan<const OaComputeKernel> GetAudioKernels() {
	return OaSpan<const OaComputeKernel>(AudioKernels, sizeof(AudioKernels) / sizeof(AudioKernels[0]));
}

// Get all Render kernels
inline OaSpan<const OaComputeKernel> GetRenderKernels() {
	return OaSpan<const OaComputeKernel>(RenderKernels, sizeof(RenderKernels) / sizeof(RenderKernels[0]));
}

// Get total kernel count
static constexpr OaUsize GetTotalKernelCount() {
	return (sizeof(MlKernels) / sizeof(MlKernels[0])) +
	       (sizeof(CryptoKernels) / sizeof(CryptoKernels[0])) +
	       (sizeof(VisionKernels) / sizeof(VisionKernels[0])) +
	       (sizeof(UiKernels) / sizeof(UiKernels[0])) +
	       (sizeof(AudioKernels) / sizeof(AudioKernels[0])) +
	       (sizeof(RenderKernels) / sizeof(RenderKernels[0]));
}

// Register extension kernels into the runtime dynamic table.
// Called via OaExtKernelRegistry::Add during OaComputeEngine::Create.
void RegisterDynamic(OaSpan<const OaComputeKernel> InKernels);

} // namespace OaKernelRegistry
