#pragma once

#include <Oa/Core/Types.h>
#include <Oa/Core/Status.h>
#include <Oa/Runtime/GemmTypes.h>
#include <mutex>

// OaBlasKernelCache — private per-shape kernel selection cache.
// Thread-safe reads after OaGemmTunerRun(); writes are single-threaded.
// Default: empty (OaGemmRouter falls through to heuristic selection).
class OaBlasKernelCache {
public:
	// Look up cached kernel for shape. Returns Auto if not cached.
	OaGemmKernel Lookup(OaU32 InM, OaU32 InN, OaU32 InK) const noexcept;

	// Store a tuned kernel choice for a shape.
	void Store(OaU32 InM, OaU32 InN, OaU32 InK, OaGemmKernel InKernel);

	// Serialize/deserialize to a flat binary file (4-byte records).
	OaStatus SaveToFile(OaStringView InPath) const;
	OaStatus LoadFromFile(OaStringView InPath);

	void Clear() noexcept;
	OaU32 Size() const noexcept;

private:
	OaHashMap<OaGemmShapeKey, OaGemmKernel, OaGemmShapeKeyHash> Map_;
	mutable std::mutex Mutex_;
};

// Global process-wide cache — accessed by OaGemmRouter.
OaBlasKernelCache &OaGetGemmKernelCache() noexcept;
