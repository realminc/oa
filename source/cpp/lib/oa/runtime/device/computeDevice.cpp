// OA vulkan Compute device Implementation.
//
// These are the methods that justify oavk::ComputeDevice existing as a class
// instead of staying as bare fields on oavk::Device. They centralize the
// post-trust-gate "is this capability actually usable?" decisions so callers
// can ask the device directly instead of decoding info.software.* fields.

#include <oa/runtime/device.h>
#include <oa/core/envFlag.h>
#include <oa/core/log.h>


bool oavk::ComputeDevice::trustCoopMatForVendor() const {
	return oavk::coopMatTrust(
		info.hardware.vendorId,
		info.hardware.deviceId,
		info.software.driverId
	);
}


bool oavk::ComputeDevice::trustBf16ForVendor() const {
	// The vendor/driver bf16 blacklist (oavk::bf16Trust) is already applied at device
	// build time in deviceBuilder (mirrors the CoopMat gate): it zeroes the
	// info.software.shaderBfloat16* fields for untrusted vendors BEFORE
	// syncFromSoftwareInfo populates HasBFloat16. so HasBFloat16 here already
	// reflects the trust gate — nothing further to decide.
	return HasBFloat16;
}


void oavk::ComputeDevice::syncFromSoftwareInfo() {
	// Single source of truth: class-level fields mirror info.software.* AFTER
	// the trust gate has run. Calling this before the gate (or skipping it) is
	// the bug PR-2 originally papered over by populating from FeatureBundle —
	// that path bypasses the gate.
	hasCooperativeMatrix       = info.software.hasCooperativeMatrix;
	hasCooperativeVector       = info.software.hasCooperativeVector;
	hasCooperativeMatrix2      = info.software.hasCooperativeMatrix2;
	hasCooperativeMatrixDecodeVector = info.software.hasCooperativeMatrixDecodeVector;
	HasBFloat16                = info.software.shaderBfloat16ExtensionEnabled;
	HasIntegerDotProduct       = info.software.shaderIntegerDotProductEnabled;
	hasDeviceGeneratedCommands = info.software.hasDeviceGeneratedCommands;
	coopMatShapes              = info.software.coopMatShapes;
	hasVideoDecodeQueue        = info.software.hasVideoDecodeQueue;
	hasVideoEncodeQueue        = info.software.hasVideoEncodeQueue;
	hasSamplerYcbcrConversion  = info.software.hasSamplerYcbcrConversion;
}


void oavk::ComputeDevice::logCoopMatShapes() const {
	if (!hasCooperativeMatrix) return;
	if (!oa::EnvFlag::isSet("OA_LOG_COOPMAT_SHAPES")) return;
	oavk::logCoopMatShapes(coopMatShapes, "      ");
}


oa::U32 oavk::ComputeDevice::getShaderCoreCount() const {
	return info.hardware.numSMs;
}
