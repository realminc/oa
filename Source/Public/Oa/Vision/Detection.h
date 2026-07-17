// OaDetectionBuffer - GPU-consumable object detection results.
//
// The packed record layout is shared by host reference providers, GPU
// postprocess kernels, and render consumers. Geometry is normalized to the
// source image so camera/view transforms remain GPU-side.

#pragma once

#include <cstddef>

#include <Oa/Core/Status.h>
#include <Oa/Core/Types.h>
#include <Oa/Runtime/Allocator.h>
#include <Oa/Runtime/Sync.h>

class OaComputeEngine;

struct OaDetection {
	OaF32 CenterX = 0.0F;
	OaF32 CenterY = 0.0F;
	OaF32 Width = 0.0F;
	OaF32 Height = 0.0F;
	OaF32 Confidence = 0.0F;
	OaU32 ClassId = 0;
	// Optional per-instance display color in 0xRRGGBBAA. Zero selects the
	// overlay style fallback and preserves compatibility with older producers.
	OaU32 ColorRgba = 0;
	// Stable identity for trackers. Rendering ignores it; postprocess and
	// interaction code can retain the same record layout across frames.
	OaU32 TrackId = 0;
};
static_assert(sizeof(OaDetection) == 32);
static_assert(offsetof(OaDetection, ColorRgba) == 24);
static_assert(offsetof(OaDetection, TrackId) == 28);

class OaDetectionBuffer {
public:
	OaDetectionBuffer() = default;
	OaDetectionBuffer(const OaDetectionBuffer&) = delete;
	OaDetectionBuffer& operator=(const OaDetectionBuffer&) = delete;
	OaDetectionBuffer(OaDetectionBuffer&& InOther) noexcept;
	OaDetectionBuffer& operator=(OaDetectionBuffer&& InOther) noexcept;
	~OaDetectionBuffer();

	// Host-visible creation path for sidecars, tests, and CPU integrations.
	// Native GPU postprocess will use the same record layout and consumer API.
	[[nodiscard]] static OaResult<OaDetectionBuffer> CreateHostUpload(OaComputeEngine& InRuntime,	OaU32 InCapacity);

	void Destroy();

	// Replaces the current records. Fails while a previous render submission
	// still consumes this slot.
	[[nodiscard]] OaStatus Upload(OaSpan<const OaDetection> InDetections);

	void MarkConsumed(
		const OaVkTimelineSemaphore& InSemaphore,
		OaU64 InValue);

	[[nodiscard]] bool IsReady() const;
	[[nodiscard]] bool IsValid() const noexcept { return Buffer_.Buffer != nullptr; }
	[[nodiscard]] OaU32 Count() const noexcept { return Count_; }
	[[nodiscard]] OaU32 Capacity() const noexcept { return Capacity_; }
	[[nodiscard]] OaU32 HomeNode() const noexcept { return Buffer_.NodeIndex; }
	[[nodiscard]] OaU32 BindlessIndex() const noexcept { return Buffer_.BindlessIndex; }
	[[nodiscard]] const OaVkBuffer& DeviceBuffer() const noexcept { return Buffer_; }

private:
	OaComputeEngine* Runtime_ = nullptr;
	OaVkBuffer Buffer_;
	OaVkTimelineSemaphore ConsumerSemaphore_;
	OaU64 ConsumerValue_ = 0;
	OaU32 Count_ = 0;
	OaU32 Capacity_ = 0;
};
