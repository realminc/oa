// OA RUNTIME - Unified CPU/GPU Timer
//
// Mode-agnostic timing: GPU (hardware timestamps) or CPU (steady_clock).
// Same API regardless of backend — switch mode at construction.
//
// OaTimerMode::Gpu  — vkCmdWriteTimestamp2 (low-variance, matches nsys)
// OaTimerMode::Cpu  — OaTimestamp::Now() wall clock (no Vulkan required)
// OaTimerMode::Auto — GPU if engine is available, else CPU
//
//   OaTimer timer(OaTimerMode::Gpu, "training_step");
//   timer.Init(rt);
//   // ... record forward / backward / optimizer into a context ...
//   auto done = context.Submit(timer.GpuRegion());
//   context.Wait(*done);
//   OaF64 ms = timer.Commit(rt.Device, kBatch);  // readback + throughput
//
// CPU usage (data loading, model init, etc.):
//   OaTimer timer(OaTimerMode::Cpu, "dataset_load");
//   timer.CpuBegin();
//   LoadData(...);
//   timer.CpuEnd();
//   printf("%.2f ms\n", timer.ElapsedMs());

#pragma once

#include <Oa/Core/Types.h>
#include <Oa/Core/Status.h>
#include <Oa/Core/Time.h>
#include <Oa/Runtime/GpuTimer.h>

class OaEngine;
class OaVkDevice;
class OaVkStream;

enum class OaTimerMode : OaU8 { Auto, Cpu, Gpu };

class OaTimer {
public:
    explicit OaTimer(OaTimerMode InMode = OaTimerMode::Auto, const char* InName = "");
    ~OaTimer();

    // Initialize GPU query pool. No-op in CPU mode. Required before Begin in GPU mode.
    [[nodiscard]] OaStatus Init(OaEngine& InRt);

    void Destroy(const OaVkDevice& InDevice);

    // Explicit GPU region passed to an event-returning submission owner.
    [[nodiscard]] OaGpuTimer* GpuRegion() noexcept {
        return Mode_ == OaTimerMode::Gpu && GpuInitialized_ ? &GpuTimer_ : nullptr;
    }

    // ── CPU-only shorthand ────────────────────────────────────────────────────
    void CpuBegin();
    void CpuEnd();

    // ── Commit (readback + throughput) ────────────────────────────────────────
    // Call after the GPU submission event completes or immediately in CPU mode.
    // InUnitsThisStep: samples / tokens / frames processed this step (for throughput).
    // Returns elapsed ms for this measurement.
    [[nodiscard]] OaF64 Commit(const OaVkDevice& InDevice, OaF64 InUnitsThisStep = 1.0);

    // ── Results ───────────────────────────────────────────────────────────────
    [[nodiscard]] OaF64 ElapsedMs()  const { return LastMs_; }
    [[nodiscard]] OaF64 Throughput() const;   // InUnitsThisStep / (ElapsedMs / 1000)

    // ── Mode accessors ────────────────────────────────────────────────────────
    [[nodiscard]] OaTimerMode  Mode()  const { return Mode_; }
    [[nodiscard]] bool         IsGpu() const { return Mode_ == OaTimerMode::Gpu; }
    [[nodiscard]] bool         IsCpu() const { return Mode_ == OaTimerMode::Cpu; }
    [[nodiscard]] const char*  GetName() const { return Name_; }

    OaTimer(const OaTimer&)            = delete;
    OaTimer& operator=(const OaTimer&) = delete;
    OaTimer(OaTimer&&) noexcept;
    OaTimer& operator=(OaTimer&&) noexcept;

private:
    OaTimerMode  Mode_;
    const char*  Name_;

    // GPU path
    OaGpuTimer   GpuTimer_;
    bool         GpuInitialized_ = false;

    // CPU path
    OaTimestamp  CpuStart_;
    OaTimestamp  CpuEnd_;
    bool         CpuRunning_ = false;

    // Last committed result
    OaF64        LastMs_    = 0.0;
    OaF64        LastUnits_ = 1.0;
};
