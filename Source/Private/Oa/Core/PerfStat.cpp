// OA CORE - Performance Statistics Accumulator

#include <Oa/Core/PerfStat.h>
#include <algorithm>
#include <cmath>

OaPerfStat::OaPerfStat(const char* InName, OaU32 InWindow, OaU32 InWarmup)
    : Name_(InName), Window_(InWindow), Warmup_(InWarmup) {
    Ring_.Resize(InWindow, 0.0);
}

void OaPerfStat::Push(OaF64 InValue) {
    LastVal_ = InValue;
    ++TotalCount_;

    if (TotalCount_ <= static_cast<OaU64>(Warmup_)) {
        return;
    }

    if (Filled_ == Window_) {
        OaF64 evicted = Ring_[Head_];
        Sum_   -= evicted;
        SumSq_ -= evicted * evicted;
    } else {
        ++Filled_;
    }

    Ring_[Head_] = InValue;
    Head_ = (Head_ + 1) % Window_;
    Sum_   += InValue;
    SumSq_ += InValue * InValue;
    Dirty_ = true;
}

bool OaPerfStat::IsReady() const {
    return Filled_ > 0;
}

OaF64 OaPerfStat::Mean() const {
    if (Filled_ == 0) { return 0.0; }
    return Sum_ / static_cast<OaF64>(Filled_);
}

OaF64 OaPerfStat::Stddev() const {
    if (Filled_ < 2) { return 0.0; }
    OaF64 n   = static_cast<OaF64>(Filled_);
    OaF64 var = (SumSq_ - (Sum_ * Sum_) / n) / (n - 1.0);
    return var > 0.0 ? std::sqrt(var) : 0.0;
}

OaF64 OaPerfStat::Min() const {
    EnsureSorted();
    return SortedBuf_.Empty() ? 0.0 : SortedBuf_[0];
}

OaF64 OaPerfStat::Max() const {
    EnsureSorted();
    return SortedBuf_.Empty() ? 0.0 : SortedBuf_[SortedBuf_.Size() - 1];
}

OaF64 OaPerfStat::P50() const { return Percentile(0.50); }
OaF64 OaPerfStat::P95() const { return Percentile(0.95); }
OaF64 OaPerfStat::P99() const { return Percentile(0.99); }
OaF64 OaPerfStat::Last() const { return LastVal_; }

void OaPerfStat::Reset() {
    TotalCount_ = 0;
    Head_    = 0;
    Filled_  = 0;
    Sum_     = 0.0;
    SumSq_   = 0.0;
    LastVal_ = 0.0;
    Dirty_   = true;
    SortedBuf_.Clear();
    for (OaU32 i = 0; i < Window_; ++i) {
        Ring_[i] = 0.0;
    }
}

void OaPerfStat::EnsureSorted() const {
    if (not Dirty_) { return; }
    SortedBuf_.Clear();
    SortedBuf_.Reserve(Filled_);
    for (OaU32 i = 0; i < Filled_; ++i) {
        SortedBuf_.PushBack(Ring_[i]);
    }
    std::sort(SortedBuf_.Data(), SortedBuf_.Data() + SortedBuf_.Size());
    Dirty_ = false;
}

OaF64 OaPerfStat::Percentile(OaF64 InP) const {
    EnsureSorted();
    if (SortedBuf_.Empty()) { return 0.0; }
    OaF64  raw  = InP * static_cast<OaF64>(SortedBuf_.Size() - 1);
    OaUsize lo  = static_cast<OaUsize>(raw);
    OaF64  frac = raw - static_cast<OaF64>(lo);
    if (lo + 1 >= SortedBuf_.Size()) { return SortedBuf_[lo]; }
    return SortedBuf_[lo] * (1.0 - frac) + SortedBuf_[lo + 1] * frac;
}
