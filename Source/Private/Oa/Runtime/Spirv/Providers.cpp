#include <Oa/Runtime/ShaderProvider.h>
#include <Oa/Runtime/Spirv.h>

// ============================================================================
// LEGACY COMPATIBILITY LAYER
// ============================================================================
//
// This file maintains backward compatibility by forwarding old API calls
// to the new unified ShaderProvider system.
// ============================================================================

// Forward OaSpvFindAny to new system
const OaSpvEntry* OaSpvFindAny(const char* InName) {
	return OaShaderProviderFind(InName);
}

// Force-link anchor: ensures oa_spirv_embed.cpp is pulled from static library.
// Without this, the linker may not include the embed object file because no
// direct symbol references exist at link time (OaSpvFind/Count are only called
// at runtime via OaShaderProviderFind). This dummy function creates a hard reference.
struct OaSpvEmbedAnchor {
	OaSpvEmbedAnchor() {
		// Touch the embed symbols to force linker inclusion
		volatile auto count = OaSpvCount();
		volatile auto entry = OaSpvFindByIndex(0);
		(void)count;
		(void)entry;
	}
};
static OaSpvEmbedAnchor g_Anchor;
