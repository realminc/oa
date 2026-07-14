// ML Kernel Registry — Single Source of Truth for ML Shaders
// Auto-included by ML CMake to drive shader compilation
// Add new kernels here → CMake auto-compiles → embedding auto-includes

#pragma once

#include <cstdint>

using OaU32 = uint32_t;

namespace MlKernelRegistry {

// ============================================================================
// OA Core Kernels (From OA library — element-wise, reductions, matrix ops)
// ============================================================================
// These kernels are provided by the OA library and must be available in ML.
// They're compiled in OA but ML needs to know about them for shader loading.

constexpr const char* OaCoreKernels[] = {
	// Element-wise operations
	"Add",
	"Sub",
	"Mul",
	"Div",
	"Neg",
	"Fill",
	"Scale",
	"Log",
	"Copy",
	
	// Reductions
	"Sum",
	"Max",
	"Argmax",
	"ReduceMean",
	
	// Matrix operations
	"TransposeSquare",
	"Broadcast",
	
	// RNG
	"PhiloxNormal",
	
	// BLAS/GEMM (these are loaded on-demand but need to be known)
	"Matmul",
	"MatmulTiled",
	"GemmWmma",
	"GemmNaive",
	
	// Loss functions
	"CrossEntropy",
	
	nullptr  // Sentinel
};

// ============================================================================
// Core Neural Network Kernels (Basic ops used by all models)
// ============================================================================

constexpr const char* NnCoreKernels[] = {
	// NOTE: Basic kernels (ByteEmbed, Gelu, Relu, Silu, LayerNorm, RmsNorm,
	// Softmax, BiasAdd, and their Bwd versions) are provided by OA library.
	// Only ML-specific extensions are listed here.
	
	// Embedding (OA has ByteEmbed, this is the standard embedding)
	"Embedding",
	
	// Linear layers (BiasAdd is in OA Core)
	// (none currently - BiasAdd moved to OA)
	
	nullptr  // Sentinel
};

// ============================================================================
// LLM Kernels (Advanced language model components)
// ============================================================================s

constexpr const char* LlmKernels[] = {
	// Gradient management
	"GradNormAccum",
	"GradClipApply",
	
	// Attention (causal)
	"CausalAttn",
	"CausalAttnBwd",
	
	// SOD (Second-Order Dynamics) — A/ENGEL v0.3
	"SodUpdate",
	"SodUpdateBwd",
	"SodUpdateComplex",
	"SodUpdateComplexBwd",
	"AttractorDiversity",
	"SodLanceReset",
	"OrbitRadiusLoss",
	"ComplexQr",
	"GobDrift",
	
	// PAR (Phase-Adaptive Routing) — Phase 2
	"ParSurpriseGate",
	"ParGatedAdd",
	"ParSurpriseGateBwd",
	"ParGatedAddBwd",
	
	// UNIT (Dual-Unit Cross-Mix) — Phase 3
	"UnitCrossMix",
	"UnitCrossMixBwd",
	
	// CAM (Content-Addressable Memory) — Phase 4
	"CamRead",
	"CamWriteSoft",
	
	nullptr  // Sentinel
};

// ============================================================================
// GQA kernels (generic RoPE primitives; not owned by a model architecture)
// ============================================================================

constexpr const char* GqaKernels[] = {
	"RopeApply",
	"RopeApplyBwd",

	nullptr  // Sentinel
};

// ============================================================================
// FFN Kernels (Feed-Forward Networks)
// ============================================================================

constexpr const char* FfnKernels[] = {
	nullptr  // Sentinel
};

// ============================================================================
// MOE Kernels (Mixture of Experts)
// ============================================================================

constexpr const char* MoeKernels[] = {
	nullptr  // Sentinel
};

constexpr const char* SsmKernels[] = {
	nullptr  // SSM kernels live in OA ML now.
};

// ============================================================================
// Optimizer Kernels
// ============================================================================

constexpr const char* OptimKernels[] = {
	// NOTE: Standard optimizers (Sgd, Adam, AdamW) are provided by OA library.
	// Only ML-specific optimizers are listed here.
	
	// Muon optimizer (momentum + orthogonalization)
	"MuonNesterov",
	"MuonNormalize",
	"MuonApply",
	"MuonVector",
	
	nullptr  // Sentinel
};

// ============================================================================
// Data Processing Kernels
// ============================================================================

constexpr const char* DataKernels[] = {
	"MnistNormalize",
	"ImageAugment",
	
	nullptr  // Sentinel
};

// ============================================================================
// Registry Metadata
// ============================================================================

// Helper: Count kernels in array
constexpr OaU32 CountKernels(const char* const* kernels) {
	OaU32 count = 0;
	while (kernels[count] != nullptr) {
		++count;
	}
	return count;
}

// Kernel counts per category
constexpr OaU32 OaCoreKernelCount  = CountKernels(OaCoreKernels);
constexpr OaU32 NnCoreKernelCount  = CountKernels(NnCoreKernels);
constexpr OaU32 LlmKernelCount     = CountKernels(LlmKernels);
constexpr OaU32 GqaKernelCount     = CountKernels(GqaKernels);
constexpr OaU32 FfnKernelCount     = CountKernels(FfnKernels);
constexpr OaU32 MoeKernelCount     = CountKernels(MoeKernels);
constexpr OaU32 SsmKernelCount     = CountKernels(SsmKernels);
constexpr OaU32 OptimKernelCount   = CountKernels(OptimKernels);
constexpr OaU32 DataKernelCount    = CountKernels(DataKernels);

// Total kernel count (ML kernels + OA Core kernels)
constexpr OaU32 TotalKernelCount =
	OaCoreKernelCount + NnCoreKernelCount + LlmKernelCount + GqaKernelCount +
	FfnKernelCount + MoeKernelCount + SsmKernelCount + OptimKernelCount + DataKernelCount;

}  // namespace MlKernelRegistry
