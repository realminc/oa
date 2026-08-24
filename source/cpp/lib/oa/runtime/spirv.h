#pragma once

#include <oa/core/types.h>

// Embedded SPIR-V registry (populated at build time via embed_spirv.cmake).
// Stable logical IDs for each entry point are defined by
// oa/runtime/computeKernel.h and the private runtime kernel registry.
namespace oavk {

class SpirvEntry {
public:
	const char* name;
	const oa::U8* data;
	oa::U32 size;
	// first 64 bits of the compiled module's build-time SHA-256 digest.
	oa::U64 contentHash;
};

// Build-generated immutable registry (generated into oa_spirv_embed.cpp).
// This is the sole runtime source of OA shader modules.
const SpirvEntry* findSpirv(const char* inName);
const SpirvEntry* findSpirvByIndex(oa::U32 inIndex);
oa::U32 spirvCount();

// ─── Push-constant reflection (debug buffer-binding validation) ─────────────
// Reflects the total byte size of a SPIR-V module's PushConstant block by
// parsing its type/decoration instructions. Returns 0 when the module has no
// push-constant block OR when any member type cannot be sized exactly (so the
// result is conservative: a non-zero answer is always exact). Used to assert
// the bindless contract `4*numBuffers + sizeof(hostPush) == declaredBlockSize`.
oa::U32 spirvPushConstantBlockSize(const oa::U8* inSpirv, oa::U32 inSizeBytes);

// Name-keyed wrapper over spirvPushConstantBlockSize: looks the kernel's SPIR-V
// up via findSpirv and caches the reflected size. Returns 0 if not found /
// not sizeable. Thread-safe.
oa::U32 spirvPushConstantBlockSizeByName(const char* inName);

} // namespace oavk
