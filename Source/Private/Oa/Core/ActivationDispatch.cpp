// OA Activation Dispatch — Centralized activation kernel selection

#include <Oa/Core/ActivationDispatch.h>
#include <Oa/Core/KernelRegistry.h>

OaActivationKernel OaActivationDispatch::Select(
	OaActivation    InActivation,
	OaGemmPrecision InPrecision)
{
	OaActivationKernel result{};
	result.Precision = InPrecision;
	
	switch (InActivation) {
		case OaActivation::None:
			result.KernelId = OA_COMPUTE_KERNEL_ID(OaComputeKernelPrefix::Ml, 0);  // Invalid
			result.Name = "None";
			break;
		case OaActivation::Relu:
			result.KernelId = OA_COMPUTE_KERNEL_ID(OaComputeKernelPrefix::Ml, 27);
			result.Name = "Relu";
			break;
		case OaActivation::Gelu:
			result.KernelId = OA_COMPUTE_KERNEL_ID(OaComputeKernelPrefix::Ml, 25);
			result.Name = "Gelu";
			break;
		case OaActivation::Silu:
			result.KernelId = OA_COMPUTE_KERNEL_ID(OaComputeKernelPrefix::Ml, 23);
			result.Name = "Silu";
			break;
		case OaActivation::Swiglu:
			result.KernelId = OA_COMPUTE_KERNEL_ID(OaComputeKernelPrefix::Ml, 29);
			result.Name = "Swiglu";
			break;
		case OaActivation::Geglu:
			result.KernelId = OA_COMPUTE_KERNEL_ID(OaComputeKernelPrefix::Ml, 137);
			result.Name = "Geglu";
			break;
		case OaActivation::SiluMul:
			result.KernelId = OA_COMPUTE_KERNEL_ID(OaComputeKernelPrefix::Ml, 136);
			result.Name = "SiluMul";
			break;
		default:
			result.KernelId = OA_COMPUTE_KERNEL_ID(OaComputeKernelPrefix::Ml, 0);
			result.Name = "Unknown";
			break;
	}
	
	return result;
}

OaStringView OaActivationDispatch::GetKernelName(
	OaActivation    InActivation,
	OaGemmPrecision InPrecision)
{
	(void)InPrecision;  // Precision not yet used in kernel selection
	return Select(InActivation, InPrecision).Name;
}

bool OaActivationDispatch::IsSupported(OaActivation InActivation) {
	switch (InActivation) {
		case OaActivation::None:
		case OaActivation::Relu:
		case OaActivation::Gelu:
		case OaActivation::Silu:
		case OaActivation::Swiglu:
		case OaActivation::Geglu:
		case OaActivation::SiluMul:
			return true;
		default:
			return false;
	}
}

const char* OaActivationDispatch::GetActivationName(OaActivation InActivation) {
	switch (InActivation) {
		case OaActivation::None: return "None";
		case OaActivation::Relu: return "Relu";
		case OaActivation::Gelu: return "Gelu";
		case OaActivation::Silu: return "Silu";
		case OaActivation::Swiglu: return "Swiglu";
		case OaActivation::Geglu: return "Geglu";
		case OaActivation::SiluMul: return "SiluMul";
		default: return "Unknown";
	}
}
