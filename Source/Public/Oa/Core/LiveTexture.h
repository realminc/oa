// OaLiveTexture — Host-coherent mailbox for live texture updates
//
// Architecture/OaArchitecture.md §10: sync-free mailbox for producer/consumer threading.
// Same pattern as LiveLoss in OaItTraining.
//
// Producer thread (training step / SD denoise / video decode):
//   - Writes to OaTexture
//   - Calls Publish(texture) with atomic sequence number increment
//   - No fence, no sync
//
// Consumer thread (UI / viewer):
//   - Polls sequence number
//   - If increased since last read, blits the new texture to swapchain
//   - Double-buffering tolerates mid-read producer updates
//
// This enables:
//   - Live training loss visualization
//   - Stable Diffusion live preview
//   - Video streaming
//   - Camera capture display
// All without blocking the producer thread on presentation.

#pragma once

#include <Oa/Core/Types.h>
#include <Oa/Core/Status.h>
#include <Oa/Ui/Image.h>

#include <atomic>
#include <memory>

class OaComputeEngine;

// ─── OaLiveTexture ─────────────────────────────────────────────────────────────
//
// Mailbox handle for a texture that updates from a producer thread.
// The consumer (UI thread) polls for updates and presents the latest version.

class OaLiveTexture {
public:
	OaLiveTexture() = default;

	// Create a live texture mailbox with double-buffered storage.
	// Both buffers are allocated on the GPU. The producer writes to one,
	// the consumer reads from the other. Publish() swaps them atomically.
	static OaResult<OaLiveTexture> Create(
		OaComputeEngine& InEngine,
		OaI32              InWidth,
		OaI32              InHeight);

	// Non-copyable, movable
	OaLiveTexture(const OaLiveTexture&) = delete;
	OaLiveTexture& operator=(const OaLiveTexture&) = delete;
	OaLiveTexture(OaLiveTexture&&) noexcept;
	OaLiveTexture& operator=(OaLiveTexture&&) noexcept;
	~OaLiveTexture();

	// ─── Producer API ────────────────────────────────────────────────────────

	// Publish a new texture from the producer thread.
	// Increments the sequence number atomically. The consumer will detect
	// the change and blit the new texture on its next frame.
	//
	// The texture is copied into the producer buffer (async GPU copy).
	// The caller can continue immediately; the copy completes in the background.
	void Publish(const OaTexture& InTexture);

	// ─── Consumer API ────────────────────────────────────────────────────────

	// Get the current sequence number. Call this to check if a new frame
	// is available without blocking.
	[[nodiscard]] OaU64 Sequence() const noexcept { return Sequence_.load(std::memory_order_acquire); }

	// Get the consumer-side texture for blitting.
	// This is the texture the consumer should read from. It's guaranteed
	// to be stable until the next Publish() call.
	[[nodiscard]] const OaTexture& ConsumerTexture() const noexcept { return ConsumerTex_; }

	// Check if a new frame is available since the last read.
	// Returns true if Sequence() > InLastSequence.
	[[nodiscard]] bool HasNewFrame(OaU64 InLastSequence) const noexcept {
		return Sequence_.load(std::memory_order_acquire) > InLastSequence;
	}

	// ─── Accessors ────────────────────────────────────────────────────────────

	[[nodiscard]] OaI32 Width() const noexcept { return Width_; }
	[[nodiscard]] OaI32 Height() const noexcept { return Height_; }
	[[nodiscard]] bool IsValid() const noexcept { return Width_ > 0 && Height_ > 0; }

	void Destroy(OaComputeEngine& InEngine);

private:
	OaLiveTexture(OaI32 InWidth, OaI32 InHeight);

	void MoveFrom(OaLiveTexture&& InOther) noexcept;

	OaI32 Width_  = 0;
	OaI32 Height_ = 0;

	// Double-buffered textures
	OaTexture ProducerTex_;  // Written by producer
	OaTexture ConsumerTex_;  // Read by consumer

	// Atomic sequence number for change detection
	std::atomic<OaU64> Sequence_{0};

	OaComputeEngine* Engine_ = nullptr;
};
