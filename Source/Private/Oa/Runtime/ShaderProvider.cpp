#include <Oa/Runtime/ShaderProvider.h>

#include <Oa/Core/KernelRegistry.h>
#include <Oa/Core/Std/HashMap.h>
#include <Oa/Core/Std/String.h>
#include <Oa/Core/Std/Vec.h>
#include <Oa/Runtime/ComputeKernel.h>

#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <mutex>

// ============================================================================
// INTERNAL STATE
// ============================================================================

struct ShaderProviderState {
	// Configuration
	OaShaderProviderConfig Config;
	bool Initialized = false;
	
	// Shader cache: name → OaSpvEntry
	OaHashMap<OaString, OaSpvEntry> Cache;
	
	// File-loaded shader data (owned by provider)
	OaVec<OaVec<OaU8>> FileData;
	
	// Statistics
	OaShaderProviderStats Stats = {};
	
	// Thread safety
	std::mutex Mutex;
	
	// Legacy external providers (for compatibility)
	OaVec<OaSpvFindFn> ExternalProviders;
};

static ShaderProviderState g_State;

// ============================================================================
// EMBEDDED SHADER LOOKUP
// ============================================================================

// Try to find shader in embedded data (OaSpvFind from oa_spirv_embed.cpp)
static const OaSpvEntry* FindEmbedded(const char* InName) {
	const OaSpvEntry* entry = OaSpvFind(InName);
	if (entry) {
		g_State.Stats.EmbeddedShaders++;
		return entry;
	}
	return nullptr;
}

// ============================================================================
// FILE-BASED SHADER LOADING
// ============================================================================

// Load shader from .spv file
static const OaSpvEntry* LoadFromFile(const char* InName) {
	if (!g_State.Config.ShaderDir) {
		return nullptr;
	}
	
	// Construct file path: ShaderDir/InName.spv
	std::filesystem::path shaderPath = g_State.Config.ShaderDir;
	shaderPath /= InName;
	shaderPath += ".spv";
	
	if (g_State.Config.Verbose) {
		std::printf("[ShaderProvider] Attempting to load: %s\n", shaderPath.string().c_str());
	}
	
	// Check if file exists
	if (!std::filesystem::exists(shaderPath)) {
		if (g_State.Config.Verbose) {
			std::printf("[ShaderProvider] File not found: %s\n", shaderPath.string().c_str());
		}
		return nullptr;
	}
	
	// Read file
	std::ifstream file(shaderPath, std::ios::binary | std::ios::ate);
	if (!file.is_open()) {
		if (g_State.Config.Verbose) {
			std::printf("[ShaderProvider] Failed to open: %s\n", shaderPath.string().c_str());
		}
		g_State.Stats.LoadFailures++;
		return nullptr;
	}
	
	// Get file size
	std::streamsize size = file.tellg();
	file.seekg(0, std::ios::beg);
	
	// Allocate buffer
	OaVec<OaU8> buffer(static_cast<OaUsize>(size));
	
	// Read data
	if (!file.read(reinterpret_cast<char*>(buffer.data()), size)) {
		if (g_State.Config.Verbose) {
			std::printf("[ShaderProvider] Failed to read: %s\n", shaderPath.string().c_str());
		}
		g_State.Stats.LoadFailures++;
		return nullptr;
	}
	
	// Store data (we own it)
	g_State.FileData.push_back(std::move(buffer));
	const OaVec<OaU8>& storedData = g_State.FileData.back();
	
	// Create entry
	OaSpvEntry entry;
	entry.Name = InName;
	entry.Data = storedData.data();
	entry.Size = static_cast<OaU32>(storedData.size());
	
	// Cache it
	OaString nameStr(InName);
	auto result = g_State.Cache.Insert({nameStr, entry});
	g_State.Stats.FileShaders++;
	
	if (g_State.Config.Verbose) {
		std::printf("[ShaderProvider] Loaded %u bytes from: %s\n", entry.Size, shaderPath.string().c_str());
	}
	
	return &result.first->second;
}

// ============================================================================
// LEGACY EXTERNAL PROVIDERS
// ============================================================================

// Try legacy external providers (for compatibility)
static const OaSpvEntry* FindExternal(const char* InName) {
	for (OaUsize i = 0; i < g_State.ExternalProviders.Size(); ++i) {
		auto provider = g_State.ExternalProviders[i];
		if (const OaSpvEntry* entry = provider(InName)) {
			return entry;
		}
	}
	return nullptr;
}

// ============================================================================
// PUBLIC API IMPLEMENTATION
// ============================================================================

static void OaShaderProviderInitLocked(const OaShaderProviderConfig* InConfig) {
	if (g_State.Initialized) {
		return; // Already initialized
	}
	
	// Copy config
	if (InConfig) {
		g_State.Config = *InConfig;
	} else {
		// Default config
		g_State.Config.Mode = OaShaderProviderMode::Auto;
		// Use OA_SPIRV_DIR if defined (set by CMake), otherwise fall back to ./Shaders
		#ifdef OA_SPIRV_DIR
			g_State.Config.ShaderDir = OA_SPIRV_DIR;
		#else
			g_State.Config.ShaderDir = "./Shaders";
		#endif
		g_State.Config.Verbose = false; // Disable verbose by default
		g_State.Config.EnableBlasOnDemand = true;
	}
	
	// Auto-detect mode if needed
	if (g_State.Config.Mode == OaShaderProviderMode::Auto) {
		// Check if we have embedded shaders OR external providers
		OaU32 embeddedCount = OaSpvCount();
		OaUsize externalCount = g_State.ExternalProviders.Size();
		
		if (embeddedCount > 0 || externalCount > 0) {
			g_State.Config.Mode = OaShaderProviderMode::Hybrid;
			if (g_State.Config.Verbose) {
				std::printf("[ShaderProvider] Auto-detected mode: Hybrid (embedded=%u, external=%zu)\n",
					embeddedCount, externalCount);
			}
		} else {
			g_State.Config.Mode = OaShaderProviderMode::FileOnly;
			if (g_State.Config.Verbose) {
				std::printf("[ShaderProvider] Auto-detected mode: FileOnly (no shaders available)\n");
			}
		}
	}
	
	g_State.Initialized = true;
	
	if (g_State.Config.Verbose) {
		std::printf("[ShaderProvider] Initialized (mode=%d, shaderDir=%s)\n",
			static_cast<int>(g_State.Config.Mode),
			g_State.Config.ShaderDir ? g_State.Config.ShaderDir : "null");
	}
}

void OaShaderProviderInit(const OaShaderProviderConfig* InConfig) {
	std::lock_guard<std::mutex> lock(g_State.Mutex);
	OaShaderProviderInitLocked(InConfig);
}

void OaShaderProviderShutdown() {
	std::lock_guard<std::mutex> lock(g_State.Mutex);
	
	if (!g_State.Initialized) {
		return;
	}
	
	// Clear all state
	g_State.Cache.Clear();
	g_State.FileData.Clear();
	g_State.ExternalProviders.Clear();
	g_State.Stats = {};
	g_State.Initialized = false;
	
	if (g_State.Config.Verbose) {
		std::printf("[ShaderProvider] Shutdown complete\n");
	}
}

bool OaShaderProviderIsInitialized() {
	std::lock_guard<std::mutex> lock(g_State.Mutex);
	return g_State.Initialized;
}

const OaSpvEntry* OaShaderProviderFind(const char* InName) {
	if (!InName) {
		return nullptr;
	}

	std::lock_guard<std::mutex> lock(g_State.Mutex);
	// Initialize under the already-owned lock. Calling the public Init here used
	// to acquire this non-recursive mutex twice and deadlock on first lookup.
	OaShaderProviderInitLocked(nullptr);
	
	// Check cache first
	OaString nameStr(InName);
	auto it = g_State.Cache.Find(nameStr);
	if (it != g_State.Cache.End()) {
		g_State.Stats.CacheHits++;
		return &it->second;
	}
	
	g_State.Stats.CacheMisses++;
	
	// Try loading based on mode
	const OaSpvEntry* entry = nullptr;
	
	switch (g_State.Config.Mode) {
		case OaShaderProviderMode::EmbeddedOnly:
			entry = FindEmbedded(InName);
			break;
			
		case OaShaderProviderMode::FileOnly:
			entry = LoadFromFile(InName);
			break;
			
		case OaShaderProviderMode::Hybrid:
		case OaShaderProviderMode::Auto:
			// Try embedded first
			entry = FindEmbedded(InName);
			if (!entry) {
				// Try external providers (legacy)
				entry = FindExternal(InName);
			}
			if (!entry) {
				// Try file loading
				entry = LoadFromFile(InName);
			}
			break;
	}
	
	if (!entry) {
		g_State.Stats.LoadFailures++;
		if (g_State.Config.Verbose) {
			std::printf("[ShaderProvider] Failed to find shader: %s\n", InName);
		}
	}
	
	return entry;
}

const OaSpvEntry* OaShaderProviderFindById(OaU64 InPackedId) {
	// Look up kernel name from registry
	const OaComputeKernel* kernel = OaComputeKernelFindByPackedId(InPackedId);
	if (!kernel) {
		return nullptr;
	}
	
	// Find shader by name
	return OaShaderProviderFind(kernel->Name);
}

OaShaderProviderStats OaShaderProviderGetStats() {
	std::lock_guard<std::mutex> lock(g_State.Mutex);
	
	// Update total count
	g_State.Stats.TotalShaders = static_cast<OaU32>(g_State.Cache.Size()) + OaSpvCount();
	
	return g_State.Stats;
}

void OaShaderProviderDumpRegistry() {
	std::lock_guard<std::mutex> lock(g_State.Mutex);
	
	std::printf("=== Shader Provider Registry ===\n");
	std::printf("Mode: %d\n", static_cast<int>(g_State.Config.Mode));
	std::printf("Shader Dir: %s\n", g_State.Config.ShaderDir ? g_State.Config.ShaderDir : "null");
	std::printf("\n");
	
	// Dump embedded shaders
	OaU32 embeddedCount = OaSpvCount();
	std::printf("Embedded Shaders (%u):\n", embeddedCount);
	for (OaU32 i = 0; i < embeddedCount; ++i) {
		const OaSpvEntry* entry = OaSpvFindByIndex(i);
		if (entry) {
			std::printf("  [%u] %s (%u bytes)\n", i, entry->Name, entry->Size);
		}
	}
	std::printf("\n");
	
	// Dump cached shaders
	std::printf("Cached Shaders (%zu):\n", g_State.Cache.Size());
	for (const auto& [name, entry] : g_State.Cache) {
		std::printf("  %s (%u bytes)\n", name.c_str(), entry.Size);
	}
	std::printf("\n");
	
	// Dump stats
	auto stats = g_State.Stats;
	std::printf("Statistics:\n");
	std::printf("  Total Shaders: %u\n", stats.TotalShaders);
	std::printf("  Embedded: %u\n", stats.EmbeddedShaders);
	std::printf("  File: %u\n", stats.FileShaders);
	std::printf("  Cache Hits: %u\n", stats.CacheHits);
	std::printf("  Cache Misses: %u\n", stats.CacheMisses);
	std::printf("  Load Failures: %u\n", stats.LoadFailures);
	std::printf("================================\n");
}

// ============================================================================
// LEGACY COMPATIBILITY
// ============================================================================

void OaSpvRegisterProvider(OaSpvFindFn InProvider) {
	if (!InProvider) {
		return;
	}
	
	std::lock_guard<std::mutex> lock(g_State.Mutex);
	
	// Add to external providers list
	g_State.ExternalProviders.push_back(InProvider);
	
	if (g_State.Config.Verbose) {
		std::printf("[ShaderProvider] Registered external provider (total: %zu)\n",
			g_State.ExternalProviders.size());
	}
}
