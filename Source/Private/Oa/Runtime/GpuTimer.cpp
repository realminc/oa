// OA RUNTIME - GPU Region Timer

#include <Oa/Runtime/GpuTimer.h>
#include <Oa/Runtime/Engine.h>
#include <Oa/Runtime/Stream.h>
#include <Oa/Core/Log.h>
#include <cassert>

OaGpuTimer::~OaGpuTimer() = default;

OaStatus OaGpuTimer::Init(OaEngine& InRt, const char* InName) {
    Name_ = InName;
    auto result = OaVkTimestamp::Create(InRt, 2);
    if (not result.IsOk()) { return result.GetStatus(); }
    Ts_ = static_cast<OaVkTimestamp&&>(result.GetValue());
    return OaStatus::Ok();
}

void OaGpuTimer::Destroy(const OaVkDevice& InDevice) {
    Ts_.Destroy(InDevice);
    Pending_ = false;
}

void OaGpuTimer::Begin(OaVkStream* InStream) {
    if (InStream == nullptr or Ts_.Pool == nullptr) {
        OA_LOG_ERROR(OaLogComponent::Core,
            "GpuTimer::Begin: %s — timer skipped",
            InStream == nullptr ? "null stream" : "called before Init");
        assert(false and "GpuTimer::Begin: null stream or not initialized");
        return;
    }
    Ts_.Reset(InStream);
    Ts_.WriteTimestamp(InStream);
    Pending_ = true;
}

void OaGpuTimer::End(OaVkStream* InStream) {
    if (InStream == nullptr or not Pending_) {
        OA_LOG_ERROR(OaLogComponent::Core,
            "GpuTimer::End: %s — timer skipped",
            InStream == nullptr ? "null stream" : "no matching Begin");
        assert(false and "GpuTimer::End: null stream or no matching Begin");
        return;
    }
    Ts_.WriteTimestamp(InStream);
}

OaF64 OaGpuTimer::ReadbackMs(const OaVkDevice& InDevice) {
    if (not Pending_) { return 0.0; }
    auto status = Ts_.Readback(InDevice);
    if (not status.IsOk()) { return 0.0; }
    Pending_ = false;
    return Ts_.ElapsedMs(0, 1);
}

OaF64 OaGpuTimer::ReadbackNs(const OaVkDevice& InDevice) {
    if (not Pending_) { return 0.0; }
    auto status = Ts_.Readback(InDevice);
    if (not status.IsOk()) { return 0.0; }
    Pending_ = false;
    return Ts_.ElapsedNs(0, 1);
}

OaGpuTimer::OaGpuTimer(OaGpuTimer&& InOther) noexcept
    : Ts_(static_cast<OaVkTimestamp&&>(InOther.Ts_))
    , Name_(InOther.Name_)
    , Pending_(InOther.Pending_) {
    InOther.Name_    = "";
    InOther.Pending_ = false;
}

OaGpuTimer& OaGpuTimer::operator=(OaGpuTimer&& InOther) noexcept {
    if (this != &InOther) {
        Ts_      = static_cast<OaVkTimestamp&&>(InOther.Ts_);
        Name_    = InOther.Name_;
        Pending_ = InOther.Pending_;
        InOther.Name_    = "";
        InOther.Pending_ = false;
    }
    return *this;
}
