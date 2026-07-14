// OaPrefetch.h — Async Staging Pipeline for CPU→GPU Data Streaming
//
// Double-buffer prefetch for large-scale GPU workloads.
// Replaces cudaMemPrefetchAsync with explicit Vulkan staging buffers.
//
// Use case: Training on datasets larger than GPU VRAM (e.g. 100GB corpus on 24GB GPU).
// Pattern: While GPU processes batch N, CPU prepares batch N+1 in staging buffer.
//
// Performance: Hides CPU→GPU transfer latency (~10-20ms per batch) behind compute.
// Measured: 15-25% throughput improvement on large-batch training (B=256, S=2048).

#pragma once

#include <Oa/Core/Types.h>
#include <Oa/Core/Status.h>
#include <Oa/Core/Std.h>
#include <Oa/Core/Thread.h>

class OaComputeEngine;
class OaVkBuffer;

// ============================================================================
// OaPrefetchPipeline — Double-Buffer Async Staging
// ============================================================================

class OaPrefetchPipeline {
public:
	// Create pipeline with two staging buffers
	// InBufferSize: size of each staging buffer (typically batch size × element size)
	static OaResult<OaPrefetchPipeline> Create(
		OaComputeEngine &InRt,
		OaU64 InBufferSize);
	
	// Start async copy from CPU to staging buffer
	// Returns immediately; copy happens in background thread
	// InData: CPU source data (must remain valid until Wait())
	// InSize: number of bytes to copy (must be ≤ buffer size)
	OaStatus BeginCopy(const void *InData, OaU64 InSize);
	
	// Wait for copy to complete, then transfer staging → device
	// Returns device buffer ready for compute
	// Blocks until background copy finishes, then submits GPU transfer
	OaResult<OaVkBuffer> Wait();
	
	// Swap buffers for next iteration
	// Call after Wait() to prepare for next BeginCopy()
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

// ============================================================================
// Convenience API — Single-Buffer Prefetch
// ============================================================================

// Simple prefetch: copy CPU data to GPU-local buffer
// Blocks until transfer completes
// Use for small transfers or when double-buffering is not needed
OaStatus OaPrefetchSync(
	OaComputeEngine &InRt,
	const void *InData,
	OaU64 InSize,
	OaVkBuffer OutDeviceBuffer);

// Async prefetch: start transfer, returns immediately
// Caller must call OaPrefetchWait() before using the buffer
OaStatus OaPrefetchAsync(
	OaComputeEngine &InRt,
	const void *InData,
	OaU64 InSize,
	OaVkBuffer OutDeviceBuffer);

// Wait for async prefetch to complete
OaStatus OaPrefetchWait(OaComputeEngine &InRt);
