#pragma once

#include <Oa/Core/Types.h>
#include <Oa/Runtime/ComputeKernel.h>

class OaAdapterRegistry;

// ─── OaExtKernelRegistry ────────────────────────────────────────────────────
// Mutable kernel table exposed to extensions during OaExtension::RegisterKernels.
// Only valid during engine Init — do not hold a pointer past that call.

class OaExtKernelRegistry {
public:
	// Append a batch of kernel metadata rows. All IDs must be non-zero and globally unique.
	// Asserts in debug if a (Prefix, Local) pair already exists (same ID registered twice).
	void Add(OaSpan<const OaComputeKernel> InKernels);
};

// ─── OaExtension ────────────────────────────────────────────────────────────
// One implementation per extension repo. The engine calls RegisterKernels and
// RegisterAdapters exactly once during OaComputeEngine::Create.
//
// Extension pointer must remain valid for the engine's lifetime.
// Static singletons are the standard pattern:
//
//   class OaMlExtension final : public OaExtension { ... };
//   OaMlExtension& OaMlExtension::Get() { static OaMlExtension I; return I; }
//
// Wire-up in binary main():
//   OaMlExtension::Get().RegisterAdapters(OaAdapterRegistry::Get());  // before engine
//   app.AddExtension(&OaMlExtension::Get());                          // before app.Main()

class OaExtension {
public:
	virtual ~OaExtension() = default;

	// Short tag for logs and collision messages ("ml", "chain", "anim").
	[[nodiscard]] virtual OaStringView Name() const = 0;

	// Register kernel metadata rows for this extension.
	// Called once during OaComputeEngine::Create, before any pipeline compilation.
	virtual void RegisterKernels(OaExtKernelRegistry& InRegistry) = 0;

	// Register adapter factories into the process-level OaAdapterRegistry.
	// May be called before engine creation — OaAdapterRegistry::Get() is always valid.
	virtual void RegisterAdapters(OaAdapterRegistry& InRegistry) = 0;
};
