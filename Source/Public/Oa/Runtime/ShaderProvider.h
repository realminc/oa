#pragma once

#include <Oa/Core/Types.h>
#include <Oa/Runtime/Spirv.h>

// ============================================================================
// OA SHADER PROVIDER — UNIFIED LOADING SYSTEM
// ============================================================================
//
// This is the new unified shader loading system that replaces the fragmented
// approach of OaSpvFind, MlSpvFind, unified embedding, on-demand BLAS, etc.
//
// DESIGN PRINCIPLES:
// 1. Single entry point: OaShaderProviderFind(name) → shader data
// 2. Automatic strategy detection (embedded/file/hybrid)
// 3. Transparent fallback chain
// 4. Registry-driven (uses KernelRegistry.h as source of truth)
// 5. Submodule-aware (handles OA as submodule in ML)
//
// LOADING STRATEGIES:
// - Embedded: Shaders compiled into binary (release builds)
// - File: Load .spv from disk (development builds)
// - Hybrid: Embedded + on-demand file loading for BLAS variants
//
// See Docs/OaComputeKernel.md for complete architecture.
// ============================================================================

// Shader provider initialization modes
enum class OaShaderProviderMode : OaU8 {
	// Automatic detection based on build configuration
	Auto = 0,
	
	// Force embedded shaders only (fail if not found)
	EmbeddedOnly = 1,
	
	// Force file loading only (fail if not found)
	FileOnly = 2,
	
	// Hybrid: try embedded first, then files
	Hybrid = 3,
};

// Shader provider configuration
struct OaShaderProviderConfig {
	// Loading mode (default: Auto)
	OaShaderProviderMode Mode = OaShaderProviderMode::Auto;
	
	// Base directory for file loading (default: "./Shaders")
	const char* ShaderDir = nullptr;
	
	// Enable verbose logging for debugging
	bool Verbose = false;
	
	// Enable on-demand BLAS kernel loading
	bool EnableBlasOnDemand = true;
};

// ============================================================================
// INITIALIZATION
// ============================================================================

// Initialize the shader provider system
// Must be called before any shader lookups
// Thread-safe: can be called multiple times (idempotent)
void OaShaderProviderInit(const OaShaderProviderConfig* InConfig = nullptr);

// Shutdown the shader provider system
// Releases all resources
void OaShaderProviderShutdown();

// Check if the shader provider is initialized
[[nodiscard]] bool OaShaderProviderIsInitialized();

// ============================================================================
// SHADER LOOKUP
// ============================================================================

// Find shader by name (primary API)
// Returns nullptr if not found
// Thread-safe after initialization
[[nodiscard]] const OaSpvEntry* OaShaderProviderFind(const char* InName);

// Find shader by packed kernel ID
// Returns nullptr if not found
// Thread-safe after initialization
[[nodiscard]] const OaSpvEntry* OaShaderProviderFindById(OaU64 InPackedId);

// ============================================================================
// STATISTICS & DEBUGGING
// ============================================================================

// Get shader provider statistics
struct OaShaderProviderStats {
	OaU32 TotalShaders;        // Total shaders available
	OaU32 EmbeddedShaders;     // Shaders loaded from embedded data
	OaU32 FileShaders;         // Shaders loaded from files
	OaU32 CacheHits;           // Number of cache hits
	OaU32 CacheMisses;         // Number of cache misses
	OaU32 LoadFailures;        // Number of failed loads
};

[[nodiscard]] OaShaderProviderStats OaShaderProviderGetStats();

// Dump all available shaders to stdout (for debugging)
void OaShaderProviderDumpRegistry();

// ============================================================================
// LEGACY COMPATIBILITY
// ============================================================================

// Legacy API is implemented in Spirv/Providers.cpp
// OaSpvFindAny() forwards to OaShaderProviderFind()
// OaSpvRegisterProvider() is implemented in ShaderProvider.cpp
