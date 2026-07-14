// OA Vulkan Compute Device Implementation.
//
// These are the methods that justify OaVkComputeDevice existing as a class
// instead of staying as bare fields on OaVkDevice. They centralize the
// post-trust-gate "is this capability actually usable?" decisions so callers
// can ask the device directly instead of decoding Info.Software.* fields.

#include <Oa/Runtime/Device.h>
#include <Oa/Core/EnvFlag.h>
#include <Oa/Core/Log.h>


bool OaVkComputeDevice::TrustCoopMatForVendor() const {
	return OaCoopMatTrust(
		Info.Hardware.VendorId,
		Info.Hardware.DeviceId,
		Info.Software.DriverId
	);
}


bool OaVkComputeDevice::TrustBf16ForVendor() const {
	// The vendor/driver bf16 blacklist (OaBf16Trust) is already applied at device
	// build time in DeviceBuilder (mirrors the CoopMat gate): it zeroes the
	// Info.Software.ShaderBfloat16* fields for untrusted vendors BEFORE
	// SyncFromSoftwareInfo populates HasBFloat16. So HasBFloat16 here already
	// reflects the trust gate — nothing further to decide.
	return HasBFloat16;
}


void OaVkComputeDevice::SyncFromSoftwareInfo() {
	// Single source of truth: class-level fields mirror Info.Software.* AFTER
	// the trust gate has run. Calling this before the gate (or skipping it) is
	// the bug PR-2 originally papered over by populating from FeatureBundle —
	// that path bypasses the gate.
	HasCooperativeMatrix       = Info.Software.HasCooperativeMatrix;
	HasCooperativeVector       = Info.Software.HasCooperativeVector;
	HasCooperativeMatrix2      = Info.Software.HasCooperativeMatrix2;
	HasCooperativeMatrixDecodeVector = Info.Software.HasCooperativeMatrixDecodeVector;
	HasBFloat16                = Info.Software.ShaderBfloat16ExtensionEnabled;
	HasIntegerDotProduct       = Info.Software.ShaderIntegerDotProductEnabled;
	HasDeviceGeneratedCommands = Info.Software.HasDeviceGeneratedCommands;
	CoopMatShapes              = Info.Software.CoopMatShapes;
	HasVideoDecodeQueue        = Info.Software.HasVideoDecodeQueue;
	HasVideoEncodeQueue        = Info.Software.HasVideoEncodeQueue;
	HasSamplerYcbcrConversion  = Info.Software.HasSamplerYcbcrConversion;
}


void OaVkComputeDevice::LogCoopMatShapes() const {
	if (!HasCooperativeMatrix) return;
	if (!OaEnvFlag::IsSet("OA_LOG_COOPMAT_SHAPES")) return;
	OaVkLogCoopMatShapes(CoopMatShapes, "      ");
}


OaU32 OaVkComputeDevice::GetShaderCoreCount() const {
	return Info.Hardware.NumSMs;
}
