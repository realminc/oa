// OA RUNTIME - Unified CPU/GPU Timer

#include <Oa/Runtime/Timer.h>
#include <Oa/Runtime/Engine.h>

OaTimer::OaTimer(OaTimerMode InMode, const char* InName)
    : Mode_(InMode), Name_(InName) {}

OaTimer::~OaTimer() = default;

OaStatus OaTimer::Init(OaComputeEngine& InRt) {
    if (Mode_ == OaTimerMode::Auto) {
        Mode_ = OaTimerMode::Gpu;
    }
    if (Mode_ == OaTimerMode::Cpu) {
        return OaStatus::Ok();
    }
    auto status = GpuTimer_.Init(InRt, Name_);
    if (status.IsOk()) {
        GpuInitialized_ = true;
    }
    return status;
}

void OaTimer::Destroy(const OaVkDevice& InDevice) {
    if (GpuInitialized_) {
        GpuTimer_.Destroy(InDevice);
        GpuInitialized_ = false;
    }
}

void OaTimer::Begin(OaComputeEngine& InRt) {
    if (Mode_ == OaTimerMode::Gpu) {
        GpuTimer_.Begin(InRt);
    } else {
        CpuBegin();
    }
}

void OaTimer::End(OaComputeEngine& InRt) {
    if (Mode_ == OaTimerMode::Gpu) {
        GpuTimer_.End(InRt);
    } else {
        CpuEnd();
    }
}

void OaTimer::CpuBegin() {
    CpuStart_   = OaTimestamp::Now();
    CpuRunning_ = true;
}

void OaTimer::CpuEnd() {
    CpuEnd_     = OaTimestamp::Now();
    CpuRunning_ = false;
}

OaF64 OaTimer::Commit(const OaVkDevice& InDevice, OaF64 InUnitsThisStep) {
    LastUnits_ = InUnitsThisStep;
    if (Mode_ == OaTimerMode::Gpu) {
        LastMs_ = GpuTimer_.ReadbackMs(InDevice);
    } else {
        OaTimestamp end = CpuRunning_ ? OaTimestamp::Now() : CpuEnd_;
        LastMs_ = (end - CpuStart_).ToMs();
    }
    return LastMs_;
}

OaF64 OaTimer::Throughput() const {
    if (LastMs_ <= 0.0) { return 0.0; }
    return LastUnits_ / (LastMs_ / 1000.0);
}

OaTimer::OaTimer(OaTimer&& InOther) noexcept
    : Mode_(InOther.Mode_)
    , Name_(InOther.Name_)
    , GpuTimer_(static_cast<OaGpuTimer&&>(InOther.GpuTimer_))
    , GpuInitialized_(InOther.GpuInitialized_)
    , CpuStart_(InOther.CpuStart_)
    , CpuEnd_(InOther.CpuEnd_)
    , CpuRunning_(InOther.CpuRunning_)
    , LastMs_(InOther.LastMs_)
    , LastUnits_(InOther.LastUnits_) {
    InOther.GpuInitialized_ = false;
    InOther.CpuRunning_     = false;
    InOther.LastMs_         = 0.0;
}

OaTimer& OaTimer::operator=(OaTimer&& InOther) noexcept {
    if (this != &InOther) {
        Mode_            = InOther.Mode_;
        Name_            = InOther.Name_;
        GpuTimer_        = static_cast<OaGpuTimer&&>(InOther.GpuTimer_);
        GpuInitialized_  = InOther.GpuInitialized_;
        CpuStart_        = InOther.CpuStart_;
        CpuEnd_          = InOther.CpuEnd_;
        CpuRunning_      = InOther.CpuRunning_;
        LastMs_          = InOther.LastMs_;
        LastUnits_       = InOther.LastUnits_;
        InOther.GpuInitialized_ = false;
        InOther.CpuRunning_     = false;
        InOther.LastMs_         = 0.0;
    }
    return *this;
}
