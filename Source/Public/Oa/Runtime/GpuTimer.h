// OA RUNTIME - GPU Region Timer
//
// Measures GPU execution time for a named region using hardware timestamp queries.
// Wraps OaVkTimestamp (2 slots: start + end). Works in batch mode and standalone.
//
// Batch mode (typical training loop usage):
//   OaGpuTimer timer;
//   timer.Init(rt, "training_step");
//       timer.Begin(rt);   // vkCmdWriteTimestamp2 into active batch CB
//       // ... forward / backward / optimizer ...
//       timer.End(rt);
//       return loss;
//   });
//   OaF64 gpuMs = timer.ReadbackMs(rt.Device);  // after FlushComputeBatch
//
// Standalone stream usage:
//   stream->Begin(rt.Device);
//   timer.Begin(stream);
//   // ... dispatches ...
//   timer.End(stream);
//   stream->SubmitAndWait(rt);
//   OaF64 gpuMs = timer.ReadbackMs(rt.Device);

#pragma once

#include <Oa/Core/Types.h>
#include <Oa/Core/Status.h>
#include <Oa/Runtime/Timestamp.h>

class OaComputeEngine;
class OaVkDevice;
class OaVkStream;

class OaGpuTimer {
public:
    OaGpuTimer() = default;
    ~OaGpuTimer();

    // Create the query pool (2 slots). Call once after engine init.
    [[nodiscard]] OaStatus Init(OaComputeEngine& InRt, const char* InName = "");

    void Destroy(const OaVkDevice& InDevice);

    // Record start timestamp into the current command buffer.
    // Auto-detects batch mode vs standalone (asserts if neither is available).
    void Begin(OaComputeEngine& InRt);
    void Begin(OaVkStream* InStream);

    // Record end timestamp.
    void End(OaComputeEngine& InRt);
    void End(OaVkStream* InStream);

    // Read back GPU results. Call after GPU work completes (after FlushComputeBatch).
    // Returns elapsed time in milliseconds or nanoseconds.
    [[nodiscard]] OaF64 ReadbackMs(const OaVkDevice& InDevice);
    [[nodiscard]] OaF64 ReadbackNs(const OaVkDevice& InDevice);

    [[nodiscard]] bool        IsPending() const { return Pending_; }
    [[nodiscard]] bool        IsInitialized() const { return Ts_.Pool != nullptr; }
    [[nodiscard]] const char* GetName()    const { return Name_; }

    OaGpuTimer(const OaGpuTimer&)            = delete;
    OaGpuTimer& operator=(const OaGpuTimer&) = delete;
    OaGpuTimer(OaGpuTimer&&) noexcept;
    OaGpuTimer& operator=(OaGpuTimer&&) noexcept;

private:
    OaVkTimestamp Ts_;
    const char*   Name_    = "";
    bool          Pending_ = false;
};
