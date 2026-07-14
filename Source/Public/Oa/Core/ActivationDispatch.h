// OA Activation Dispatch — Centralized activation kernel selection
// Provides unified interface for selecting and dispatching activation kernels
// Pattern inspired by TileGym's @dispatch decorator for backend-agnostic activation ops

#pragma once

#include <Oa/Core/Types.h>
#include <Oa/Runtime/GemmTypes.h>

// Activation kernel descriptor
struct OaActivationKernel {
	OaKernelId      KernelId;
	OaStringView    Name;
	OaGemmPrecision Precision;
};

// Activation dispatch registry
class OaActivationDispatch {
public:
	// Select activation kernel based on type and precision
	[[nodiscard]] static OaActivationKernel Select(
		OaActivation    InActivation,
		OaGemmPrecision InPrecision = OaGemmPrecision::Auto);
	
	// Get kernel name for activation type
	[[nodiscard]] static OaStringView GetKernelName(
		OaActivation    InActivation,
		OaGemmPrecision InPrecision = OaGemmPrecision::Auto);
	
	// Check if activation is supported
	[[nodiscard]] static bool IsSupported(OaActivation InActivation);
	
	// List all supported activations
	[[nodiscard]] static const char* GetActivationName(OaActivation InActivation);
};
