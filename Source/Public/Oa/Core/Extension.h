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
// Experimental startup kernel-registration seam. The engine currently calls
// RegisterKernels during OaEngine::Create. RegisterAdapters is unresolved
// compatibility debt: OaAdapterRegistry has no implementation in this tree and
// the engine does not call it.
//
// Extension pointer must remain valid for the engine's lifetime.
// Static singletons are the standard pattern:
//
//   class OaMlExtension final : public OaExtension { ... };
//   OaMlExtension& OaMlExtension::Get() { static OaMlExtension I; return I; }
//
// Wire-up in binary main():
//   app.AddExtension(&MyExtension::Get());  // before app.Main()

class OaExtension {
public:
	virtual ~OaExtension() = default;

	// Short tag for logs and collision messages ("ml", "chain", "anim").
	[[nodiscard]] virtual OaStringView Name() const = 0;

	// Register kernel metadata rows for this extension.
	// Called once during OaEngine::Create, before any pipeline compilation.
	virtual void RegisterKernels(OaExtKernelRegistry& InRegistry) = 0;

	// Unresolved compatibility hook. Do not document adapter registration as
	// supported until OaAdapterRegistry has one implemented owner and consumer.
	virtual void RegisterAdapters(OaAdapterRegistry& InRegistry) = 0;
};
