#pragma once

#include <Oa/Core/Types.h>

// Embedded SPIR-V registry (populated at build time via embed_spirv.cmake).
// Stable logical ids for each entry point: Oa/Runtime/ComputeKernel.h + Docs/OaComputeKernelRegistry.md
class OaSpvEntry {
public:
	const char* Name;
	const OaU8* Data;
	OaU32 Size;
};

// liboa's built-in registry (generated into oa_spirv_embed.cpp)
const OaSpvEntry* OaSpvFind(const char* InName);
const OaSpvEntry* OaSpvFindByIndex(OaU32 InIndex);
OaU32 OaSpvCount();

// External SPIR-V provider registration (for consumer libraries like libml)
using OaSpvFindFn = const OaSpvEntry* (*)(const char*);
void OaSpvRegisterProvider(OaSpvFindFn InProvider);

// Searches liboa's built-in registry first, then all registered external providers
const OaSpvEntry* OaSpvFindAny(const char* InName);

// ─── Push-constant reflection (debug buffer-binding validation) ─────────────
// Reflects the total byte size of a SPIR-V module's PushConstant block by
// parsing its type/decoration instructions. Returns 0 when the module has no
// push-constant block OR when any member type cannot be sized exactly (so the
// result is conservative: a non-zero answer is always exact). Used to assert
// the bindless contract `4*numBuffers + sizeof(hostPush) == declaredBlockSize`.
OaU32 OaSpvPushConstantBlockSize(const OaU8* InSpirv, OaU32 InSizeBytes);

// Name-keyed wrapper over OaSpvPushConstantBlockSize: looks the kernel's SPIR-V
// up via OaSpvFindAny and caches the reflected size. Returns 0 if not found /
// not sizeable. Thread-safe.
OaU32 OaSpvPushConstantBlockSizeByName(const char* InName);
