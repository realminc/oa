// Runtime compatibility wrapper for a single fixed-size prefetched device buffer.
// New code should use OaUploadRing directly: it batches multiple destinations,
// returns a GPU-chainable completion token and owns staging lifetimes safely.

#pragma once

#include <Oa/Core/Types.h>
#include <Oa/Core/Status.h>
#include <Oa/Core/Std.h>

class OaComputeEngine;
class OaVkBuffer;

// ============================================================================
// OaPrefetchPipeline — fixed-destination compatibility upload pipeline
// ============================================================================

class OaPrefetchPipeline {
public:
	OaPrefetchPipeline() = default;
	OaPrefetchPipeline(OaPrefetchPipeline&&) noexcept;
	OaPrefetchPipeline& operator=(OaPrefetchPipeline&&) noexcept;
	OaPrefetchPipeline(const OaPrefetchPipeline&) = delete;
	OaPrefetchPipeline& operator=(const OaPrefetchPipeline&) = delete;
	~OaPrefetchPipeline();

	// Create one device-local destination with two persistent upload frames.
	// InBufferSize: destination capacity (typically batch size x element size).
	static OaResult<OaPrefetchPipeline> Create(
		OaComputeEngine &InRt,
		OaU64 InBufferSize);

	// Copy CPU data into the next persistent mapped upload frame.
	// The source need only remain valid for this call.
	// InSize: number of bytes to copy (must be ≤ buffer size)
	OaStatus BeginCopy(const void *InData, OaU64 InSize);

	// Submit the queued staging -> device transfer and wait at this legacy CPU
	// boundary. New code should use OaUploadRing::Submit and chain its completion
	// token into the GPU consumer instead of blocking here.
	OaResult<OaVkBuffer> Wait();

	// Retained for source compatibility. Frame rotation is automatic.
	void Swap();

	// Cleanup
	void Destroy(OaComputeEngine &InRt);

	// Query state
	bool IsBusy() const;
	OaU64 BufferSize() const;

private:
	struct Impl;
	OaUniquePtr<Impl> Impl_;
};
