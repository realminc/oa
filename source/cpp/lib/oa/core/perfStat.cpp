// OA CORE - Performance statistics Accumulator

#include <oa/core/perfStat.h>
#include <algorithm>
#include <cmath>

oa::PerfStat::PerfStat(const char* inName, oa::U32 inWindow, oa::U32 inWarmup)
    : name_(inName), window_(inWindow), warmup_(inWarmup) {
    ring_.resize(inWindow, 0.0);
}

void oa::PerfStat::push(oa::F64 inValue) {
    lastVal_ = inValue;
    ++totalCount_;

    if (totalCount_ <= static_cast<oa::U64>(warmup_)) {
        return;
    }

    if (filled_ == window_) {
        oa::F64 evicted = ring_[head_];
        sum_   -= evicted;
        sumSq_ -= evicted * evicted;
    } else {
        ++filled_;
    }

    ring_[head_] = inValue;
    head_ = (head_ + 1) % window_;
    sum_   += inValue;
    sumSq_ += inValue * inValue;
    dirty_ = true;
}

bool oa::PerfStat::isReady() const {
    return filled_ > 0;
}

oa::F64 oa::PerfStat::mean() const {
    if (filled_ == 0) { return 0.0; }
    return sum_ / static_cast<oa::F64>(filled_);
}

oa::F64 oa::PerfStat::stddev() const {
    if (filled_ < 2) { return 0.0; }
    oa::F64 n   = static_cast<oa::F64>(filled_);
    oa::F64 var = (sumSq_ - (sum_ * sum_) / n) / (n - 1.0);
    return var > 0.0 ? std::sqrt(var) : 0.0;
}

oa::F64 oa::PerfStat::min() const {
    ensureSorted();
    return sortedBuf_.empty() ? 0.0 : sortedBuf_[0];
}

oa::F64 oa::PerfStat::max() const {
    ensureSorted();
    return sortedBuf_.empty() ? 0.0 : sortedBuf_[sortedBuf_.size() - 1];
}

oa::F64 oa::PerfStat::p50() const { return percentile(0.50); }
oa::F64 oa::PerfStat::p95() const { return percentile(0.95); }
oa::F64 oa::PerfStat::p99() const { return percentile(0.99); }
oa::F64 oa::PerfStat::last() const { return lastVal_; }

void oa::PerfStat::reset() {
    totalCount_ = 0;
    head_    = 0;
    filled_  = 0;
    sum_     = 0.0;
    sumSq_   = 0.0;
    lastVal_ = 0.0;
    dirty_   = true;
    sortedBuf_.clear();
    for (oa::U32 i = 0; i < window_; ++i) {
        ring_[i] = 0.0;
    }
}

void oa::PerfStat::ensureSorted() const {
    if (not dirty_) { return; }
    sortedBuf_.clear();
    sortedBuf_.reserve(filled_);
    for (oa::U32 i = 0; i < filled_; ++i) {
        sortedBuf_.pushBack(ring_[i]);
    }
    std::sort(sortedBuf_.data(), sortedBuf_.data() + sortedBuf_.size());
    dirty_ = false;
}

oa::F64 oa::PerfStat::percentile(oa::F64 inP) const {
    ensureSorted();
    if (sortedBuf_.empty()) { return 0.0; }
    oa::F64  raw  = inP * static_cast<oa::F64>(sortedBuf_.size() - 1);
    oa::Usize lo  = static_cast<oa::Usize>(raw);
    oa::F64  frac = raw - static_cast<oa::F64>(lo);
    if (lo + 1 >= sortedBuf_.size()) { return sortedBuf_[lo]; }
    return sortedBuf_[lo] * (1.0 - frac) + sortedBuf_[lo + 1] * frac;
}
