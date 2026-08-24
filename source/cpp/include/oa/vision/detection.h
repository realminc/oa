// DetectionBuffer - GPU-consumable object detection results.
//
// The packed record layout is shared by host reference providers, GPU
// postprocess kernels, and render consumers. Geometry is normalized to the
// source image so camera/view transforms remain GPU-side.

#pragma once

#include <cstddef>

#include <oa/core/status.h>
#include <oa/core/std/uniquePtr.h>
#include <oa/core/types.h>

namespace oa {
class Engine;
class Event;

struct Detection {
	F32 centerX = 0.0F;
	F32 centerY = 0.0F;
	F32 width = 0.0F;
	F32 height = 0.0F;
	F32 confidence = 0.0F;
	U32 classId = 0;
	// Optional per-instance display color in 0xRRGGBBAA. Zero selects the
	// overlay style fallback and preserves compatibility with older producers.
	U32 colorRgba = 0;
	// Stable identity for trackers. Rendering ignores it; postprocess and
	// interaction code can retain the same record layout across frames.
	U32 trackId = 0;
};
static_assert(sizeof(Detection) == 32);
static_assert(offsetof(Detection, colorRgba) == 24);
static_assert(offsetof(Detection, trackId) == 28);

class DetectionBuffer {
public:
	DetectionBuffer() = default;
	DetectionBuffer(const DetectionBuffer&) = delete;
	DetectionBuffer& operator=(const DetectionBuffer&) = delete;
	DetectionBuffer(DetectionBuffer&& inOther) noexcept;
	DetectionBuffer& operator=(DetectionBuffer&& inOther) noexcept;
	~DetectionBuffer();

	// Host-visible creation path for sidecars, tests, and CPU integrations.
	// Native GPU postprocess will use the same record layout and consumer API.
	[[nodiscard]] static Result<DetectionBuffer> createHostUpload(
		Engine& inRuntime,
		U32 inCapacity);

	// Replaces the current records. Fails while a previous render submission
	// still consumes this slot.
	[[nodiscard]] Status upload(Span<const Detection> inDetections);

	[[nodiscard]] Status markConsumed(const Event& inCompletion);

	[[nodiscard]] bool isReady() const;
	[[nodiscard]] bool isValid() const noexcept;
	[[nodiscard]] U32 count() const noexcept;
	[[nodiscard]] U32 capacity() const noexcept;
	[[nodiscard]] U32 bindlessIndex() const noexcept;

private:
	struct Impl;
	void reset_() noexcept;

	UniquePtr<Impl> impl_;
};

} // namespace oa
