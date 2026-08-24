// oa::Engine is the one concrete vulkan execution owner. Optional window-system
// presentation is a borrowing session declared in <oa/runtime/presenter.h>.

#pragma once

#include <oa/core/types.h>
#include <oa/core/status.h>
#include <oa/core/device.h>
#include <oa/core/envFlag.h>
#include <oa/core/log.h>
#include <oa/core/std/function.h>
#include <oa/runtime/executionPlan.h>
#include <oa/runtime/clockCalibration.h>

namespace oa {

class Timer;
class ExecutionStats;

// ─── Config ─────────────────────────────────────────────────────────────────

enum class DevicePreference : oa::U8 {
	Discrete,
	Integrated,
	Cpu,
	ByIndex,
};

// PresentationMode controls what graphics surface the engine is built for.
//   None       — compute-only engine. No graphics queue, no swapchain.
//                Headless training, batch ML inference, CI.
//   Headless   — graphics queue requested, but no VK_KHR_swapchain extension
//                and no surface ever attached. hasGraphics=true, hasPresent
//                =false permanently. Render-farm worker, server-side
//                rendering, batch image-sequence export, CI for graphics
//                code paths. Sinks: saveImage / encodeFrame.
//   Swapchain  — graphics queue + swapchain extension. surface attached via
//                two-phase init (initPresentation). GUI mode. Sinks:
//                present (+ optionally saveImage / encodeFrame).
enum class PresentationMode : oa::U8 {
	None,
	Headless,
	Swapchain,
};

class EngineConfig {
public:
	DevicePreference  devicePref       = DevicePreference::Discrete;
	PresentationMode  presentationMode = PresentationMode::None;
	oa::Precision       precision        = oa::Precision::FP32;
	oa::NumericMode     numericMode      = oa::NumericMode::Fast;

#ifdef OA_VULKAN_VALIDATION
	oa::Bool enableValidation = true;
#else
	oa::Bool enableValidation = false;
#endif
	oa::Bool   enablePipelineCache        = true;
	// Eagerly instantiate every supported exact-dtype embedded compute pipeline
	// during engine init, keeping pipeline creation outside training dispatches.
	// memory-constrained embedders may disable this.
	oa::Bool   preloadEmbeddedPipelines   = true;
	oa::String pipelineCacheDir           = defaultPipelineCacheDir_();
	oa::String appName                    = "OaApp";
	oa::U32    appVersion                 = 1;
	// SDL3: fill from SDL_Vulkan_GetInstanceExtensions (VK_KHR_surface + platform ext).
	oa::Vec<oa::String> instanceExtraExtensions;

	oa::U32 deviceIndex = 0;

	// Select this engine's private eager recorder for the creating thread.
	oa::Bool selectForThread = true;

	// Diagnostics are engine-owned. calls made before create or from a thread
	// without an engine selection use the synchronous stderr fallback.
	oa::LogOptions log;

private:
	[[nodiscard]] static oa::String defaultPipelineCacheDir_();
};


enum class EngineState : oa::U8 {
	Empty,
	Initializing,
	Ready,
	Failed,
	Destroying,
	Destroyed,
};


// The engine is deliberately concrete and pinned. Optional presentation,
// video and collective services compose with it; they do not subclass it.
class Engine {
public:
	// Pinned: the engine owns a VkInstance/VkDevice/VMA/queues/mutexes and self-
	// referential pools, so it must never move. create() returns an owning pointer;
	// hold the engine by reference. Empty engines and two-phase initialization
	// are private implementation details; public ownership begins at create().
	Engine(Engine&&)            = delete;
	Engine(const Engine&)       = delete;
	~Engine();

	[[nodiscard]] static oa::Result<oa::UniquePtr<Engine>> create(const EngineConfig& inConfig = {});

	// Explicit shutdown boundary. close() drains engine-owned submissions, releases
	// resources, and reports completion failures.
	[[nodiscard]] oa::Status close();

	[[nodiscard]] bool hasCompute()  const noexcept;
	[[nodiscard]] bool hasGraphics() const noexcept;
	[[nodiscard]] bool isReady()     const noexcept;
	[[nodiscard]] EngineState getState() const noexcept;

	// Record one isolated semantic region and compile it into immutable reusable
	// work. capture() restores the previously selected recorder and never submits
	// or waits. The returned plan borrows this engine, which must outlive it.
	[[nodiscard]] oa::Result<oa::ExecutionPlan> capture(oa::Fn<void()> inRecord);

	// submit one compiled plan without waiting. The exact event is the host and
	// cross-queue completion boundary; the plan may be resubmitted while alive.
	[[nodiscard]] oa::Result<oa::Event> submit(oa::ExecutionPlan& inPlan);

	// submit the current eager recording without waiting.
	[[nodiscard]] oa::Result<oa::Event> submit(oa::Timer* inTimer = nullptr);

	// wait for an event owned by this engine. Waiting the exact eager event also
	// reclaims its private one-shot recording state; plan and service events are
	// ordinary idempotent waits.
	[[nodiscard]] oa::Status wait(const oa::Event& inEvent);
	[[nodiscard]] const oa::ExecutionStats& lastExecutionStats() const noexcept;
	[[nodiscard]] bool supportsClockCalibration() const noexcept;
	// query the device and a POSIX monotonic clock quasi-simultaneously and
	// return the lowest-deviation sample. Unsupported devices return
	// oa::StatusCode::Unavailable; no guessed host/device offset is substituted.
	[[nodiscard]] oa::Result<oa::ClockCalibration> calibrateClock(oa::U32 inSampleCount = 8U) const;

	// Boolean provenance check for completion-consuming APIs.
	[[nodiscard]] bool ownsEvent(const oa::Event& inEvent) const noexcept;

	[[nodiscard]] oa::U64 deviceVramBytes() const noexcept;
	[[nodiscard]] oa::MemoryUsage getMemoryUsage() const;

	[[nodiscard]] oa::StringView deviceName()        const noexcept;
	[[nodiscard]] oa::StringView deviceVendorName()  const noexcept;
	[[nodiscard]] oa::DeviceType deviceType()        const noexcept;
	[[nodiscard]] oa::StringView driverName()        const noexcept;
	[[nodiscard]] oa::StringView driverVersion()     const noexcept;
	[[nodiscard]] oa::StringView vulkanApiVersion()  const noexcept;
	[[nodiscard]] oa::Precision  getPrecision()      const noexcept;

	Engine& operator=(Engine&&)      = delete;
	Engine& operator=(const Engine&) = delete;

private:
	Engine();

	friend class EngineAccess;

	class Impl;
	oa::UniquePtr<Impl> impl_;
};

} // namespace oa

// ---------------------------------------------------------------------------
// Legacy compatibility aliases — migration targets, remove when consumers updated.
// ---------------------------------------------------------------------------
